from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from collections import Counter
from typing import Optional

from src.models import Interaction, Customer, Ticket


# ─── Customer Health Scoring ──────────────────────────────────────────────────

def _determine_health_label(
    engagement_score: Optional[float],
    nps_score: Optional[float],
    avg_sentiment: float,
    escalated_tickets: int,
    recent_30d_interactions: int,
) -> str:
    """
    Rule-based customer health label derived from engagement score, NPS,
    interaction sentiment, escalation volume, and recent activity.

    - "High Risk"       : low engagement, a detractor NPS (<=3), strongly
                           negative sentiment, or 2+ escalated tickets.
    - "Needs Attention"  : moderate warning signs (a passive/borderline NPS,
                           mild negative sentiment, any escalation, or no
                           activity in 30 days).
    - "Healthy"         : none of the above risk signals present.
    """
    engagement_score = engagement_score if engagement_score is not None else 50.0

    if (
        engagement_score < 40
        or (nps_score is not None and nps_score <= 3)
        or avg_sentiment < -0.4
        or escalated_tickets >= 2
    ):
        return "High Risk"

    if (
        engagement_score < 60
        or (nps_score is not None and nps_score <= 6)
        or avg_sentiment < -0.1
        or escalated_tickets >= 1
        or recent_30d_interactions == 0
    ):
        return "Needs Attention"

    return "Healthy"


def get_customer_memory(customer_id: int, db: Session) -> dict:
    """
    Returns two-layer memory for a customer:
    - short_term: last 5 interactions (for immediate context)
    - long_term_summary: aggregated behavioural summary across all history

    Enriched (new, additive fields - existing fields are unchanged) with
    ticket-history signals so downstream consumers (e.g. the AI agent) get
    a fuller picture without needing a separate query of their own:
      - most_common_ticket_category / most_common_ticket_priority (historical pattern)
      - latest_ticket_category / latest_ticket_priority (the newest ticket only)
      - open_tickets / resolved_tickets / escalated_tickets / total_tickets
      - latest_ticket_title / latest_ticket_status / latest_resolution_date
      - customer_health ("Healthy" / "Needs Attention" / "High Risk"),
        derived from engagement score, NPS, sentiment, escalations, and
        recent activity

    Only two queries are made regardless of history size (one for
    interactions, one for tickets) - no N+1 behaviour.
    """

    all_interactions = (
        db.query(Interaction)
        .filter(Interaction.customer_id == customer_id)
        .order_by(Interaction.timestamp.desc())
        .all()
    )

    # ── Ticket history (single query, newest first) ────────────────────────
    all_tickets = (
        db.query(Ticket)
        .filter(Ticket.customer_id == customer_id)
        .order_by(Ticket.created_at.desc())
        .all()
    )

    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    engagement_score = customer.engagement_score if customer else None
    nps_score = customer.nps_score if customer else None

    total_tickets = len(all_tickets)
    open_tickets = sum(1 for t in all_tickets if t.status in ("Open", "In Progress"))
    resolved_tickets_count = sum(1 for t in all_tickets if t.status in ("Resolved", "Closed"))
    escalated_tickets_count = sum(1 for t in all_tickets if t.status == "Escalated")

    category_counts = Counter(t.category for t in all_tickets)
    most_common_category = category_counts.most_common(1)[0][0] if category_counts else None

    priority_counts = Counter(t.priority for t in all_tickets)
    most_common_priority = priority_counts.most_common(1)[0][0] if priority_counts else None

    latest_ticket = all_tickets[0] if all_tickets else None
    latest_ticket_title = latest_ticket.title if latest_ticket else None
    latest_ticket_status = latest_ticket.status if latest_ticket else None
    latest_ticket_category = latest_ticket.category if latest_ticket else None
    latest_ticket_priority = latest_ticket.priority if latest_ticket else None
    latest_resolution_date = (
        latest_ticket.resolved_at.isoformat()
        if latest_ticket and latest_ticket.resolved_at
        else None
    )

    ticket_fields = {
        "total_tickets": total_tickets,
        "open_tickets": open_tickets,
        "resolved_tickets": resolved_tickets_count,
        "escalated_tickets": escalated_tickets_count,
        "most_common_ticket_category": most_common_category,
        "most_common_ticket_priority": most_common_priority,
        "latest_ticket_title": latest_ticket_title,
        "latest_ticket_status": latest_ticket_status,
        # Distinct from most_common_ticket_category/priority above - these
        # describe the single newest ticket, not the customer's historical
        # pattern. Keeping both avoids the "latest" vs "most common" mixup.
        "latest_ticket_category": latest_ticket_category,
        "latest_ticket_priority": latest_ticket_priority,
        "latest_resolution_date": latest_resolution_date,
        # Additive: real DB id of the latest ticket, needed so downstream
        # consumers (agents.py source-citation logic) can cite a concrete,
        # existing record like "Ticket #381" instead of just its title.
        "latest_ticket_id": latest_ticket.id if latest_ticket else None,
    }

    if not all_interactions:
        avg_sentiment = 0.0
        recent_30d_interactions = 0
        health_label = _determine_health_label(
            engagement_score, nps_score, avg_sentiment, escalated_tickets_count, recent_30d_interactions
        )
        return {
            "short_term": [],
            "long_term_summary": "No interaction history found for this customer.",
            "total_interactions": 0,
            "avg_sentiment": avg_sentiment,
            "avg_csat": None,
            "customer_health": health_label,
            **ticket_fields,
        }

    # ── Short-term: last 5 interactions ───────────────────────────────────────
    short_term = [
        {
            # Additive: real DB id of this interaction, so downstream
            # consumers (agents.py source-citation logic) can cite it
            # unambiguously as an existing record.
            "interaction_id": i.id,
            "channel": i.channel,
            "message": i.message,
            "sentiment": round(i.sentiment, 2) if i.sentiment else None,
            "csat_score": i.csat_score,
            "timestamp": i.timestamp.isoformat() if i.timestamp else None,
        }
        for i in all_interactions[:5]
    ]

    # ── Long-term: aggregated behaviour summary ────────────────────────────────
    total = len(all_interactions)

    # Channel breakdown
    channel_counts = Counter(i.channel for i in all_interactions)
    preferred_channel = channel_counts.most_common(1)[0][0]

    # Sentiment trend
    sentiments = [i.sentiment for i in all_interactions if i.sentiment is not None]
    avg_sentiment = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0
    sentiment_label = (
        "positive" if avg_sentiment > 0.2
        else "negative" if avg_sentiment < -0.2
        else "neutral"
    )

    # CSAT trend
    csat_scores = [i.csat_score for i in all_interactions if i.csat_score is not None]
    avg_csat = round(sum(csat_scores) / len(csat_scores), 2) if csat_scores else None

    # Recency
    latest = all_interactions[0].timestamp
    oldest = all_interactions[-1].timestamp
    span_days = (latest - oldest).days if latest and oldest else 0

    # 30-day activity
    thirty_days_ago = datetime.now() - timedelta(days=30)
    recent_count = sum(
        1 for i in all_interactions
        if i.timestamp and i.timestamp >= thirty_days_ago
    )

    long_term_summary = (
        f"Customer has {total} total interactions over {span_days} days. "
        f"Preferred channel: {preferred_channel} "
        f"({dict(channel_counts)}). "
        f"Overall sentiment: {sentiment_label} (avg {avg_sentiment}). "
        f"Avg CSAT: {avg_csat if avg_csat else 'N/A'}. "
        f"Interactions in last 30 days: {recent_count}."
    )

    health_label = _determine_health_label(
        engagement_score, nps_score, avg_sentiment, escalated_tickets_count, recent_count
    )

    return {
        "short_term": short_term,
        "long_term_summary": long_term_summary,
        "total_interactions": total,
        "preferred_channel": preferred_channel,
        "avg_sentiment": avg_sentiment,
        "avg_csat": avg_csat,
        "recent_30d_interactions": recent_count,
        "customer_health": health_label,
        **ticket_fields,
    }
