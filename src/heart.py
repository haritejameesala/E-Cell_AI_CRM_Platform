from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, date, timedelta

from src.models import Customer, Ticket, Interaction


def heart_metrics(db: Session) -> dict:
    """
    Computes the HEART framework scores.

    API response shape and field names are UNCHANGED from the original
    implementation - only the internal calculations for Engagement and
    Task_Success were made richer (see the comments in each section).
    """
    customers = db.query(Customer).all()
    tickets = db.query(Ticket).all()
    interactions = db.query(Interaction).all()

    total_customers = len(customers)
    total_tickets = len(tickets)

    # ── H — HAPPINESS ─────────────────────────────────────────────────────────
    # Signal: average CSAT score (1-5 scale, normalised to 0-100)
    # + average NPS score (0-10 scale, normalised to 0-100)
    #
    # NOTE: `nps_score` here is each individual customer's own 0-10
    # recommendation rating (their per-customer survey response), not the
    # traditional aggregate Net Promoter Score (-100 to +100, computed as
    # %promoters - %detractors across a population). Averaging these
    # per-customer 0-10 ratings and normalising to 0-100 is a reasonable
    # happiness proxy, but it should not be presented as "NPS" in the
    # traditional sense in a report - call it "average customer rating"
    # to avoid confusion with the standard metric.
    csat_scores = [i.csat_score for i in interactions if i.csat_score is not None]
    avg_csat = (sum(csat_scores) / len(csat_scores)) if csat_scores else 0
    csat_normalised = (avg_csat / 5) * 100  # convert to 0-100

    nps_scores = [c.nps_score for c in customers if c.nps_score is not None]
    avg_nps = (sum(nps_scores) / len(nps_scores)) if nps_scores else 0
    nps_normalised = (avg_nps / 10) * 100   # convert to 0-100

    happiness = round((csat_normalised + nps_normalised) / 2, 2)

    # ── E — ENGAGEMENT ────────────────────────────────────────────────────────
    # Previously: a single signal (% customers with an interaction in the
    # last 30 days). That undercounts engagement for customers who are
    # active via tickets but haven't had a logged interaction, and ignores
    # the overall engagement_score baseline we already track per customer.
    #
    # Now: a weighted blend of three signals -
    #   50% recent interactions   - % customers with an interaction in last 30 days
    #   25% recent ticket activity - % customers with a ticket touched in last 30 days
    #   25% baseline activity     - average of each customer's engagement_score
    thirty_days_ago = datetime.now() - timedelta(days=30)

    active_interactors = (
        db.query(Interaction.customer_id)
        .filter(Interaction.timestamp >= thirty_days_ago)
        .distinct()
        .count()
    )
    interaction_signal = (
        (active_interactors / total_customers * 100) if total_customers else 0
    )

    recently_active_ticket_customers = (
        db.query(Ticket.customer_id)
        .filter(Ticket.updated_at >= thirty_days_ago)
        .distinct()
        .count()
    )
    ticket_signal = (
        (recently_active_ticket_customers / total_customers * 100) if total_customers else 0
    )

    engagement_scores = [c.engagement_score for c in customers if c.engagement_score is not None]
    baseline_activity_signal = (
        (sum(engagement_scores) / len(engagement_scores)) if engagement_scores else 0
    )

    engagement = round(
        0.50 * interaction_signal + 0.25 * ticket_signal + 0.25 * baseline_activity_signal,
        2,
    )

    # ── A — ADOPTION ──────────────────────────────────────────────────────────
    # Signal: % of customers who have raised at least 1 ticket.
    # Note: this measures SUPPORT adoption specifically (i.e. customers who
    # have engaged with the support/ticketing system at all), not product
    # feature adoption - we don't track per-feature usage events, so ticket
    # creation is the closest available proxy. Kept as-is; only the comment
    # is clarified per the "rename internal comments" request.
    customers_with_tickets = (
        db.query(Ticket.customer_id)
        .distinct()
        .count()
    )
    adoption = round(
        (customers_with_tickets / total_customers * 100) if total_customers else 0, 2
    )

    # ── R — RETENTION ─────────────────────────────────────────────────────────
    # Signal: % of customers who signed up 90+ days ago and are still Active
    # (distinct from Adoption — this measures long-term stickiness)
    cutoff_date = date.today() - timedelta(days=90)
    tenured_customers = [
        c for c in customers if c.signup_date <= cutoff_date
    ]
    retained = sum(
        1 for c in tenured_customers if c.status == "Active"
    )
    retention = round(
        (retained / len(tenured_customers) * 100) if tenured_customers else 0, 2
    )

    # ── T — TASK SUCCESS ──────────────────────────────────────────────────────
    # Signal: % of tickets that reached Resolved or Closed status.
    # Internally we also compute average resolution time (created_at ->
    # resolved_at) as a supporting metric. It's surfaced as an extra field
    # below (avg_resolution_hours) but Task_Success itself is unchanged, so
    # existing consumers reading only Task_Success are unaffected.
    resolved_tickets = [t for t in tickets if t.status in ["Resolved", "Closed"]]
    task_success = round(
        (len(resolved_tickets) / total_tickets * 100) if total_tickets else 0, 2
    )

    resolution_hours = [
        (t.resolved_at - t.created_at).total_seconds() / 3600
        for t in resolved_tickets
        if t.resolved_at and t.created_at
    ]
    avg_resolution_hours = (
        round(sum(resolution_hours) / len(resolution_hours), 2) if resolution_hours else None
    )

    return {
        "Happiness": happiness,
        "Engagement": engagement,
        "Adoption": adoption,
        "Retention": retention,
        "Task_Success": task_success,
        "computed_at": datetime.utcnow().isoformat(),
        "signal_sources": {
            "Happiness": (
                "Avg CSAT (interactions) + Avg per-customer 0-10 rating "
                "('nps_score'), normalised to 0-100. Note: this is an "
                "average of individual customer ratings, not the "
                "traditional aggregate NPS (-100 to +100)."
            ),
            "Engagement": (
                "Weighted blend: 50% recent interactions (30d) + 25% recent "
                "ticket activity (30d) + 25% avg customer engagement_score"
            ),
            "Adoption": "% customers who raised at least 1 support ticket",
            "Retention": "% customers signed up 90+ days ago still Active",
            "Task_Success": "% tickets in Resolved or Closed status",
        },
        # Additive internal metric - not part of the original schema, safe
        # for existing consumers that only read the fields above.
        "avg_resolution_hours": avg_resolution_hours,
        # ── Additive (Feature 5): trend history + breakdowns. Existing
        # fields above are untouched, so old consumers keep working exactly
        # as before; new consumers can opt into these richer views.
        "trend": heart_trend(db),
        "by_agent": heart_by_agent(db),
        "by_ticket_category": heart_by_category(db),
        "by_channel": heart_by_channel(db),
    }


def heart_metrics_by_cohort(db: Session) -> list:
    """HEART scores broken down per signup-month cohort."""
    customers = db.query(Customer).all()
    cohorts: dict = {}

    for c in customers:
        key = c.signup_date.strftime("%Y-%m")
        cohorts.setdefault(key, []).append(c)

    results = []

    for cohort_month, members in sorted(cohorts.items()):
        ids = [c.id for c in members]
        total = len(members)

        tickets = db.query(Ticket).filter(Ticket.customer_id.in_(ids)).all()
        interactions = db.query(Interaction).filter(
            Interaction.customer_id.in_(ids)
        ).all()

        # Happiness
        csat = [i.csat_score for i in interactions if i.csat_score]
        happiness = round((sum(csat) / len(csat) / 5 * 100) if csat else 0, 2)

        # Engagement (weighted blend, same signals as heart_metrics)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        eng_ids = set(
            i.customer_id for i in interactions
            if i.timestamp and i.timestamp >= thirty_days_ago
        )
        interaction_signal = round(len(eng_ids) / total * 100 if total else 0, 2)

        recent_ticket_ids = set(
            t.customer_id for t in tickets
            if t.updated_at and t.updated_at >= thirty_days_ago
        )
        ticket_signal = round(len(recent_ticket_ids) / total * 100 if total else 0, 2)

        cohort_engagement_scores = [c.engagement_score for c in members if c.engagement_score is not None]
        baseline_activity_signal = (
            (sum(cohort_engagement_scores) / len(cohort_engagement_scores))
            if cohort_engagement_scores else 0
        )

        engagement = round(
            0.50 * interaction_signal + 0.25 * ticket_signal + 0.25 * baseline_activity_signal,
            2,
        )

        # Adoption (% of cohort with at least 1 support ticket)
        ticket_ids = set(t.customer_id for t in tickets)
        adoption = round(len(ticket_ids) / total * 100 if total else 0, 2)

        # Retention
        cutoff = date.today() - timedelta(days=90)
        tenured = [c for c in members if c.signup_date <= cutoff]
        retained = sum(1 for c in tenured if c.status == "Active")
        retention = round(retained / len(tenured) * 100 if tenured else 0, 2)

        # Task Success
        resolved = [t for t in tickets if t.status in ["Resolved", "Closed"]]
        task_success = round(len(resolved) / len(tickets) * 100 if tickets else 0, 2)

        resolution_hours = [
            (t.resolved_at - t.created_at).total_seconds() / 3600
            for t in resolved
            if t.resolved_at and t.created_at
        ]
        avg_resolution_hours = (
            round(sum(resolution_hours) / len(resolution_hours), 2) if resolution_hours else None
        )

        results.append({
            "cohort": cohort_month,
            "total_customers": total,
            "Happiness": happiness,
            "Engagement": engagement,
            "Adoption": adoption,
            "Retention": retention,
            "Task_Success": task_success,
            "avg_resolution_hours": avg_resolution_hours,
        })

    return results


# ─── Trend history (Feature 5) ─────────────────────────────────────────────────

def _week_start(d: date) -> date:
    """Monday of the ISO week containing `d` - used as the weekly bucket key."""
    return d - timedelta(days=d.weekday())


def heart_trend(db: Session, weeks: int = 8, months: int = 6) -> dict:
    """
    Weekly and monthly trend arrays for Happiness, Engagement, Task_Success.

    DATA AVAILABILITY NOTE: Retention (90+ day tenure & still Active) reflects
    a customer's CURRENT status, not a historical snapshot per period, so we
    can't reconstruct a true historical Retention trend from current data
    alone (same limitation documented in cohort.py's retention_curve). It is
    therefore omitted from the trend arrays here rather than fabricated;
    Retention itself is still reported as a single current value in
    heart_metrics().

    Bucketing:
      - Happiness  <- avg CSAT (normalised 0-100) of interactions per period
      - Engagement <- % of customers with >=1 interaction in that period
      - Task_Success <- % of tickets CREATED in that period that are now
        Resolved/Closed

    Returns oldest -> newest arrays, one point per week/month, so charting
    libraries can plot them directly left-to-right.
    """
    total_customers = db.query(Customer).count() or 1

    interactions = db.query(
        Interaction.timestamp, Interaction.csat_score, Interaction.customer_id
    ).all()
    tickets = db.query(Ticket.created_at, Ticket.status).all()

    def _bucket_key(dt: datetime, granularity: str):
        if granularity == "weekly":
            return _week_start(dt.date())
        return dt.date().replace(day=1)

    def _build_series(granularity: str, num_periods: int):
        today = datetime.now()
        if granularity == "weekly":
            period_keys = [
                _week_start((today - timedelta(weeks=i)).date())
                for i in range(num_periods - 1, -1, -1)
            ]
        else:
            period_keys = []
            cursor = today.date().replace(day=1)
            for _ in range(num_periods):
                period_keys.append(cursor)
                # step back one month
                prev_month = (cursor.replace(day=1) - timedelta(days=1)).replace(day=1)
                cursor = prev_month
            period_keys = list(reversed(period_keys))

        csat_by_period = defaultdict(list)
        customers_by_period = defaultdict(set)
        for ts, csat, customer_id in interactions:
            if ts is None:
                continue
            key = _bucket_key(ts, granularity)
            if csat is not None:
                csat_by_period[key].append(csat)
            customers_by_period[key].add(customer_id)

        tickets_by_period = defaultdict(list)
        for created_at, status in tickets:
            if created_at is None:
                continue
            key = _bucket_key(created_at, granularity)
            tickets_by_period[key].append(status)

        happiness_series, engagement_series, task_success_series, labels = [], [], [], []

        for key in period_keys:
            labels.append(key.isoformat())

            csats = csat_by_period.get(key, [])
            happiness_series.append(
                round((sum(csats) / len(csats)) / 5 * 100, 2) if csats else 0.0
            )

            engaged = len(customers_by_period.get(key, set()))
            engagement_series.append(round(engaged / total_customers * 100, 2))

            statuses = tickets_by_period.get(key, [])
            resolved = sum(1 for s in statuses if s in ["Resolved", "Closed"])
            task_success_series.append(
                round(resolved / len(statuses) * 100, 2) if statuses else 0.0
            )

        return {
            "labels": labels,
            "Happiness": happiness_series,
            "Engagement": engagement_series,
            "Task_Success": task_success_series,
        }

    return {
        "weekly": _build_series("weekly", weeks),
        "monthly": _build_series("monthly", months),
        "note": (
            "Retention is omitted from trend arrays - only current status is "
            "stored, not historical per-period snapshots, so a true "
            "historical Retention trend can't be reconstructed."
        ),
    }


# ─── Per-agent / per-category / per-channel breakdowns (Feature 5) ────────────

def heart_by_agent(db: Session) -> list:
    """Task Success + avg resolution time per assigned support agent."""
    rows = db.query(
        Ticket.assigned_agent, Ticket.status, Ticket.created_at, Ticket.resolved_at
    ).all()

    by_agent = defaultdict(list)
    for agent, status, created_at, resolved_at in rows:
        by_agent[agent].append((status, created_at, resolved_at))

    results = []
    for agent, entries in sorted(by_agent.items()):
        total = len(entries)
        resolved = [e for e in entries if e[0] in ("Resolved", "Closed")]
        task_success = round(len(resolved) / total * 100, 2) if total else 0.0

        res_hours = [
            (resolved_at - created_at).total_seconds() / 3600
            for _, created_at, resolved_at in resolved
            if resolved_at and created_at
        ]
        avg_resolution_hours = round(sum(res_hours) / len(res_hours), 2) if res_hours else None

        results.append({
            "agent": agent,
            "total_tickets": total,
            "task_success_pct": task_success,
            "avg_resolution_hours": avg_resolution_hours,
        })

    return results


def heart_by_category(db: Session) -> list:
    """Task Success + volume per ticket category."""
    rows = db.query(Ticket.category, Ticket.status).all()
    by_category = defaultdict(list)
    for category, status in rows:
        by_category[category].append(status)

    results = []
    for category, statuses in sorted(by_category.items()):
        total = len(statuses)
        resolved = sum(1 for s in statuses if s in ("Resolved", "Closed"))
        escalated = sum(1 for s in statuses if s == "Escalated")
        results.append({
            "category": category,
            "total_tickets": total,
            "task_success_pct": round(resolved / total * 100, 2) if total else 0.0,
            "escalation_pct": round(escalated / total * 100, 2) if total else 0.0,
        })

    return results


def heart_by_channel(db: Session) -> list:
    """Happiness signal (avg CSAT/sentiment) per interaction channel."""
    rows = db.query(Interaction.channel, Interaction.sentiment, Interaction.csat_score).all()
    by_channel = defaultdict(lambda: {"sentiments": [], "csats": [], "count": 0})
    for channel, sentiment, csat in rows:
        by_channel[channel]["count"] += 1
        if sentiment is not None:
            by_channel[channel]["sentiments"].append(sentiment)
        if csat is not None:
            by_channel[channel]["csats"].append(csat)

    results = []
    for channel, data in sorted(by_channel.items()):
        sentiments = data["sentiments"]
        csats = data["csats"]
        results.append({
            "channel": channel,
            "total_interactions": data["count"],
            "avg_sentiment": round(sum(sentiments) / len(sentiments), 3) if sentiments else None,
            "avg_csat": round(sum(csats) / len(csats), 2) if csats else None,
        })

    return results


# ─── System report metadata (Feature 10) ──────────────────────────────────────

def get_heart_metric_metadata() -> dict:
    """
    Machine-readable metric definitions/formulas/signal sources/business
    justification for the HEART dashboard, for the final System Report.
    """
    return {
        "Happiness": {
            "formula": "avg(CSAT normalised to 0-100) averaged with avg(per-customer 0-10 rating normalised to 0-100)",
            "signal_source": "Interaction.csat_score, Customer.nps_score",
            "business_justification": "Direct voice-of-customer signal; captures satisfaction independent of usage volume.",
        },
        "Engagement": {
            "formula": "50% * (%customers with interaction in last 30d) + 25% * (%customers with ticket activity in last 30d) + 25% * avg(engagement_score)",
            "signal_source": "Interaction.timestamp, Ticket.updated_at, Customer.engagement_score",
            "business_justification": "Blends multiple recency/activity signals so a customer isn't under-counted just because one channel is quiet.",
        },
        "Adoption": {
            "formula": "customers_with_>=1_ticket / total_customers * 100",
            "signal_source": "Ticket.customer_id (distinct)",
            "business_justification": "Proxy for product/support-system adoption in the absence of feature-usage telemetry.",
        },
        "Retention": {
            "formula": "retained_90d+_tenured_customers / tenured_customers * 100",
            "signal_source": "Customer.signup_date, Customer.status",
            "business_justification": "Long-term stickiness, distinct from short-term engagement.",
        },
        "Task_Success": {
            "formula": "resolved_or_closed_tickets / total_tickets * 100",
            "signal_source": "Ticket.status",
            "business_justification": "Operational effectiveness of the support function.",
        },
    }
