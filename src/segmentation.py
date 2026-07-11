"""
Generic, rule-based customer segmentation.

The goal here was to support a bunch of segmentation dimensions (industry,
tier, engagement, tenure, ticket frequency, status, NPS, churn risk)
without writing the same "loop over customers, bucket them, average the
numbers" logic eight separate times.

How it's structured:
1. `_bucket_functions` maps a dimension name to a small function that
   takes one customer's precomputed signals and returns which bucket
   they fall into for that dimension.
2. `segment_customers()` does all the DB work up front in a handful of
   batched queries (no N+1), builds one signals dict per customer, and
   then just calls the right bucket function per customer.

Adding a new dimension later is a ~5-line function, not a new copy of the
whole segmentation loop.

Note: crm.get_customer_segments() (the original industry-only version used
by GET /customers/segments/by-industry) is untouched by this module - it's
kept separate on purpose so that route's response shape never changes.
This module powers the newer GET /customers/segments/{dimension} route.
"""

from collections import defaultdict
from datetime import date, datetime
from typing import Callable, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.models import Customer, Ticket, Interaction
from src.cohort import compute_churn_score

VALID_DIMENSIONS = [
    "industry", "tier", "engagement", "tenure",
    "ticket_frequency", "status", "nps", "churn_risk",
]


# ── Build per-customer signals in one batched pass (no N+1) ──

def _build_customer_signals(db: Session) -> Dict[int, dict]:
    """
    Computes everything any bucket function might need, using a small
    fixed number of grouped queries (independent of how many customers
    there are), then returns {customer_id: signals_dict}.

    This is the only place that talks to the DB - every bucket function
    below just reads from the dict this returns.
    """
    customers = db.query(Customer).all()

    ticket_counts = dict(
        db.query(Ticket.customer_id, func.count(Ticket.id))
        .group_by(Ticket.customer_id)
        .all()
    )

    status_rows = (
        db.query(Ticket.customer_id, Ticket.status, func.count(Ticket.id))
        .group_by(Ticket.customer_id, Ticket.status)
        .all()
    )
    status_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for customer_id, status, count in status_rows:
        status_counts[customer_id][status] = count

    interaction_agg = (
        db.query(
            Interaction.customer_id,
            func.avg(Interaction.sentiment),
            func.avg(Interaction.csat_score),
            func.max(Interaction.timestamp),
        )
        .group_by(Interaction.customer_id)
        .all()
    )
    interaction_stats = {
        cid: {
            "avg_sentiment": float(s) if s is not None else None,
            "avg_csat": float(c) if c is not None else None,
            "last_interaction": ts,
        }
        for cid, s, c, ts in interaction_agg
    }

    now = datetime.now()
    today = date.today()
    signals: Dict[int, dict] = {}

    for customer in customers:
        t_count = ticket_counts.get(customer.id, 0)
        stats = interaction_stats.get(customer.id, {})
        last_interaction = stats.get("last_interaction") or customer.last_interaction_date
        days_since_last_interaction = (
            (now - last_interaction).days if last_interaction else None
        )
        open_tickets = (
            status_counts.get(customer.id, {}).get("Open", 0)
            + status_counts.get(customer.id, {}).get("In Progress", 0)
        )
        escalated_tickets = status_counts.get(customer.id, {}).get("Escalated", 0)

        churn_score, churn_reasons = compute_churn_score(
            customer,
            t_count,
            0,
            open_tickets=open_tickets,
            escalated_tickets=escalated_tickets,
            avg_sentiment=stats.get("avg_sentiment"),
            avg_csat=stats.get("avg_csat"),
            days_since_last_interaction=days_since_last_interaction,
        )

        tenure_days = (today - customer.signup_date).days if customer.signup_date else 0

        signals[customer.id] = {
            "customer": customer,
            "ticket_count": t_count,
            "tenure_days": tenure_days,
            "churn_score": churn_score,
            "churn_reasons": churn_reasons,
            "avg_sentiment": stats.get("avg_sentiment"),
            "avg_csat": stats.get("avg_csat"),
            "days_since_last_interaction": days_since_last_interaction,
        }

    return signals


# ── Bucket functions - one small function per dimension ──

def _bucket_industry(s: dict) -> str:
    return s["customer"].industry or "Unknown"


def _bucket_tier(s: dict) -> str:
    return s["customer"].tier or "Unknown"


def _bucket_status(s: dict) -> str:
    return s["customer"].status or "Unknown"


def _bucket_engagement(s: dict) -> str:
    score = s["customer"].engagement_score or 0
    if score >= 70:
        return "High Engagement (70+)"
    if score >= 40:
        return "Medium Engagement (40-69)"
    return "Low Engagement (<40)"


def _bucket_tenure(s: dict) -> str:
    days = s["tenure_days"]
    if days < 30:
        return "New (<30 days)"
    if days < 90:
        return "Growing (30-89 days)"
    if days < 365:
        return "Established (90-364 days)"
    return "Veteran (365+ days)"


def _bucket_ticket_frequency(s: dict) -> str:
    count = s["ticket_count"]
    if count == 0:
        return "No Tickets"
    if count <= 2:
        return "Low (1-2 tickets)"
    if count <= 5:
        return "Medium (3-5 tickets)"
    return "High (6+ tickets)"


def _bucket_nps(s: dict) -> str:
    nps = s["customer"].nps_score
    if nps is None:
        return "Not Surveyed"
    if nps <= 6:
        return "Detractor (0-6)"
    if nps <= 8:
        return "Passive (7-8)"
    return "Promoter (9-10)"


def _bucket_churn_risk(s: dict) -> str:
    score = s["churn_score"]
    if score >= 70:
        return "High Risk"
    if score >= 40:
        return "Medium Risk"
    return "Low Risk"


_BUCKET_FUNCTIONS: Dict[str, Callable[[dict], str]] = {
    "industry": _bucket_industry,
    "tier": _bucket_tier,
    "status": _bucket_status,
    "engagement": _bucket_engagement,
    "tenure": _bucket_tenure,
    "ticket_frequency": _bucket_ticket_frequency,
    "nps": _bucket_nps,
    "churn_risk": _bucket_churn_risk,
}


# ── Public entry point ──

def segment_customers(db: Session, dimension: str) -> dict:
    """
    Segments every customer along the given dimension and returns bucket
    counts plus a couple of aggregates per bucket (avg engagement, avg
    churn score, active/churned counts).
    """
    bucket_fn = _BUCKET_FUNCTIONS.get(dimension)
    if bucket_fn is None:
        raise ValueError(
            f"Unknown segmentation dimension '{dimension}'. "
            f"Valid options: {VALID_DIMENSIONS}"
        )

    signals = _build_customer_signals(db)

    buckets: Dict[str, List[dict]] = defaultdict(list)
    for s in signals.values():
        label = bucket_fn(s)
        buckets[label].append(s)

    segments = []
    for label, members in sorted(buckets.items()):
        engagement_scores = [m["customer"].engagement_score for m in members]
        churn_scores = [m["churn_score"] for m in members]
        active = sum(1 for m in members if m["customer"].status == "Active")
        churned = sum(1 for m in members if m["customer"].status == "Churned")

        segments.append({
            "segment": label,
            "total_customers": len(members),
            "avg_engagement": round(sum(engagement_scores) / len(engagement_scores), 2) if engagement_scores else 0,
            "avg_churn_score": round(sum(churn_scores) / len(churn_scores), 2) if churn_scores else 0,
            "active_count": active,
            "churned_count": churned,
        })

    return {"dimension": dimension, "segments": segments}