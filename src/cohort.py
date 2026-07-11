import io
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, datetime, timedelta
from collections import defaultdict
from collections import Counter
from typing import Optional

from src.models import Customer, Ticket, Interaction


# ── Churn scoring ──

def compute_churn_score(
    customer,
    tickets_count: int,
    interactions_count: int,
    open_tickets: int = 0,
    escalated_tickets: int = 0,
    avg_sentiment: Optional[float] = None,
    avg_csat: Optional[float] = None,
    days_since_last_interaction: Optional[int] = None,
) -> tuple:
    """
    Rule-based churn risk score, 0-100 (capped), plus the list of reasons
    that fed into it. Higher = more likely to churn.

    These weights are hand-picked, not fitted on real churn outcomes -
    they're a reasonable starting point for a demo, but should be called
    out as heuristic (not learned) anywhere this gets written up, and
    revisited if real labelled churn data ever becomes available.

    Rough point breakdown (they intentionally sum to a bit over 100, so
    several overlapping risk factors can compound - the total is capped
    at 100 either way):
      - Low engagement score                    -> up to +30
      - Low/detractor NPS                       -> up to +25
      - Negative average sentiment               -> up to +20
      - Low average CSAT                         -> up to +15
      - No recent activity / no interactions     -> up to +20
      - Escalated ticket(s) on file               -> up to +20
      - Multiple open tickets                     -> up to +15
      - High ticket volume + negative sentiment   -> +10

    High ticket volume by itself is deliberately NOT a risk signal - a
    heavy support user with good NPS/CSAT/engagement (think: a big
    Enterprise account) is usually just an engaged customer, not someone
    about to leave. It only counts here when paired with negative
    sentiment, which is what actually separates "frustrated repeat
    contact" from "engaged power user".
    """
    score = 0
    reasons = []

    engagement = customer.engagement_score or 0
    if engagement < 40:
        score += 30
        reasons.append("Low engagement score")
    elif engagement < 55:
        score += 15
        reasons.append("Moderate-low engagement score")

    if customer.nps_score is not None:
        if customer.nps_score <= 3:
            score += 25
            reasons.append("Very low NPS")
        elif customer.nps_score <= 6:
            score += 12
            reasons.append("Low NPS (detractor/passive)")

    if avg_sentiment is not None:
        if avg_sentiment < -0.4:
            score += 20
            reasons.append("Strong negative average sentiment")
        elif avg_sentiment < -0.15:
            score += 10
            reasons.append("Negative average sentiment")

    if avg_csat is not None:
        if avg_csat < 2.5:
            score += 15
            reasons.append("Very low average CSAT")
        elif avg_csat < 3.5:
            score += 8
            reasons.append("Low average CSAT")

    if days_since_last_interaction is None:
        score += 20
        reasons.append("No interaction history")
    elif days_since_last_interaction > 90:
        score += 20
        reasons.append("No recent activity for 90+ days")
    elif days_since_last_interaction > 60:
        score += 12
        reasons.append("No recent activity for 60+ days")

    if escalated_tickets >= 2:
        score += 20
        reasons.append("Multiple escalated tickets")
    elif escalated_tickets >= 1:
        score += 12
        reasons.append("Escalated ticket on file")

    if open_tickets >= 3:
        score += 15
        reasons.append("Multiple open tickets")
    elif open_tickets >= 1:
        score += 5
        reasons.append("Open ticket on file")

    if interactions_count <= 1:
        score += 10
        reasons.append("Very low interaction history")

    if tickets_count > 5 and avg_sentiment is not None and avg_sentiment < -0.2:
        score += 10
        reasons.append("High ticket volume with negative sentiment")

    if not reasons:
        reasons.append("No significant risk signals")

    return min(score, 100), reasons


# ── Retention curve approximation ──

def _approximate_retention_curve(retention_rate: float, months_since: int) -> list:
    """
    Estimates a monthly retention curve for a cohort even though we don't
    have historical status snapshots to draw a real one from.

    Worth being upfront about this limitation whenever it shows up in a
    report: we only know each customer's CURRENT status, not their status
    at every past month, so this is an approximation, not a reconstruction
    of what actually happened.

    A straight line from 100% down to the current rate doesn't look like
    real retention curves (which tend to drop fast early and level off),
    so instead we fit a geometric (compound) monthly decay that lands
    exactly on the cohort's real, current retention_rate at its actual
    age in months. It's the best shape we can infer from current-status
    data alone - not a substitute for real monthly snapshots.
    """
    months_since = max(months_since, 0)

    if months_since == 0 or retention_rate >= 100:
        return [
            {"month": month, "retention_pct": round(retention_rate, 2)}
            for month in range(min(months_since + 1, 12))
        ]

    # Solve for the per-month survival factor f such that f ** months_since
    # equals the observed retention fraction.
    survival_fraction = max(retention_rate, 0.01) / 100
    per_month_factor = survival_fraction ** (1 / months_since)

    curve = []
    for month in range(min(months_since + 1, 12)):
        pct = 100 * (per_month_factor ** month)
        curve.append({"month": month, "retention_pct": round(pct, 2)})

    return curve

def get_behavioral_tag(
    customer,
    ticket_count: int,
    interaction_count: int,
    churn_score: float,
):
    """
    Labels a customer with a rough behavioural tag using signals we
    already have - no new fields needed, it's computed on the fly.
    """

    engagement = customer.engagement_score or 0
    nps = customer.nps_score or 0

    if churn_score >= 70:
        return "High Risk"

    if engagement >= 80 and ticket_count <= 2:
        return "Power User"

    if engagement >= 70 and nps >= 9:
        return "Promoter"

    if ticket_count >= 8:
        return "Support Heavy"

    if engagement <= 40:
        return "Low Engagement"

    if interaction_count <= 2:
        return "Inactive"

    return "Standard"

# ── Cohort analysis ──

def cohort_analysis(db: Session) -> list:
    """
    Groups customers by signup month and, for each cohort, computes
    retention rate, churn rate, per-customer churn scores (with reasons),
    and an approximated retention curve.

    All the per-customer inputs (ticket status counts, sentiment/CSAT
    averages, last-interaction recency) come from batched, grouped
    queries up front - nothing gets queried inside the per-customer loop
    below.
    """

    customers = db.query(Customer).all()

    # ── Batched lookups so the loop below doesn't hit the DB per customer ──
    ticket_counts = dict(
        db.query(Ticket.customer_id, func.count(Ticket.id))
        .group_by(Ticket.customer_id)
        .all()
    )

    interaction_counts = dict(
        db.query(Interaction.customer_id, func.count(Interaction.id))
        .group_by(Interaction.customer_id)
        .all()
    )

    # Ticket status breakdown per customer (open/in-progress/escalated/etc.)
    ticket_status_rows = (
        db.query(Ticket.customer_id, Ticket.status, func.count(Ticket.id))
        .group_by(Ticket.customer_id, Ticket.status)
        .all()
    )
    ticket_status_counts: dict = defaultdict(lambda: defaultdict(int))
    for customer_id, ticket_status, count in ticket_status_rows:
        ticket_status_counts[customer_id][ticket_status] = count

    # Sentiment / CSAT / last-seen, all in one grouped query per customer.
    interaction_agg_rows = (
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
        customer_id: {
            "avg_sentiment": float(avg_sentiment) if avg_sentiment is not None else None,
            "avg_csat": float(avg_csat) if avg_csat is not None else None,
            "last_interaction": last_ts,
        }
        for customer_id, avg_sentiment, avg_csat, last_ts in interaction_agg_rows
    }

    # ── Group customers into signup-month cohorts ──
    cohorts: dict = defaultdict(list)
    for customer in customers:
        cohort_month = customer.signup_date.strftime("%Y-%m")
        cohorts[cohort_month].append(customer)

    analysis = []
    now = datetime.now()

    for cohort_month, members in sorted(cohorts.items()):
        total_customers = len(members)

        active_customers = sum(1 for c in members if c.status == "Active")
        churned_customers = sum(1 for c in members if c.status == "Churned")
        inactive_customers = sum(1 for c in members if c.status == "Inactive")

        retention_rate = round(active_customers / total_customers * 100, 2)
        churn_rate = round(churned_customers / total_customers * 100, 2)

        # ── Score every customer in this cohort ──
        customer_scores = []
        churn_windows = {
            "0-30 Days": 0,
            "31-60 Days": 0,
            "61-90 Days": 0,
            "90+ Days": 0,
        }
        high_risk_count = 0

        for customer in members:
            t_count = ticket_counts.get(customer.id, 0)
            i_count = interaction_counts.get(customer.id, 0)

            status_counts = ticket_status_counts.get(customer.id, {})
            open_tickets = status_counts.get("Open", 0) + status_counts.get("In Progress", 0)
            escalated_tickets = status_counts.get("Escalated", 0)

            stats = interaction_stats.get(customer.id, {})
            avg_sentiment = stats.get("avg_sentiment")
            avg_csat = stats.get("avg_csat")

            last_interaction = stats.get("last_interaction") or customer.last_interaction_date
            days_since_last_interaction = (
                (now - last_interaction).days if last_interaction else None
            )
            customer_age = (date.today() - customer.signup_date).days

            churn_score, churn_reasons = compute_churn_score(
                customer,
                t_count,
                i_count,
                open_tickets=open_tickets,
                escalated_tickets=escalated_tickets,
                avg_sentiment=avg_sentiment,
                avg_csat=avg_csat,
                days_since_last_interaction=days_since_last_interaction,
            )
            if customer.status == "Churned":
                # Bucket by how old the account was when it churned - shows
                # whether churn is mostly happening early or late.
                if customer_age <= 30:
                    churn_windows["0-30 Days"] += 1

                elif customer_age <= 60:
                    churn_windows["31-60 Days"] += 1

                elif customer_age <= 90:
                    churn_windows["61-90 Days"] += 1

                else:
                    churn_windows["90+ Days"] += 1

            behavioral_tag = get_behavioral_tag(
                customer,
                t_count,
                i_count,
                churn_score,
            )

            if churn_score >= 70:
                high_risk_count += 1

            customer_scores.append({
                "customer_id": customer.id,
                "name": customer.name,
                "status": customer.status,
                "engagement_score": customer.engagement_score,
                "ticket_count": t_count,
                "interaction_count": i_count,
                "churn_score": churn_score,
                "churn_risk": (
                    "High" if churn_score >= 70
                    else "Medium" if churn_score >= 40
                    else "Low"
                ),
                "behavioral_tag": behavioral_tag,
                "churn_reasons": churn_reasons,
            })

        # ── Retention curve for this cohort ──
        cohort_signup = date.fromisoformat(cohort_month + "-01")
        months_since = (date.today() - cohort_signup).days // 30
        retention_curve = _approximate_retention_curve(retention_rate, months_since)

        behavior_counts = Counter(
            c["behavioral_tag"]
            for c in customer_scores
        )

        analysis.append({
            "cohort": cohort_month,
            "cohort_id": f"cohort-{cohort_month}",
            "churn_windows": churn_windows,
            "total_customers": total_customers,
            "active_customers": active_customers,
            "churned_customers": churned_customers,
            "inactive_customers": inactive_customers,
            "retention_rate": retention_rate,
            "churn_rate": churn_rate,
            "behavioral_cohorts": dict(behavior_counts),
            "high_risk_customers": high_risk_count,
            "retention_curve": retention_curve,
            "retention_curve_note": (
                "Estimated via geometric decay from current status only - "
                "monthly historical snapshots are not stored, so this is an "
                "approximation, not an exact reconstruction."
            ),
            "customer_churn_scores": customer_scores,
        })

    return analysis


# ── Re-engagement rate ──

def re_engagement_analysis(db: Session, inactivity_threshold_days: int = 30) -> dict:
    """
    Finds customers who went quiet for more than `inactivity_threshold_days`
    and then came back with at least one more interaction afterward.

    We don't store periodic activity snapshots, just a timestamped
    interaction log per customer - so "went inactive, then came back" gets
    detected by sorting each customer's interaction timestamps and
    checking whether any consecutive pair is further apart than the
    threshold. If so, the interaction that closes that gap counts as a
    "re-engagement" event. Customers with fewer than 2 interactions can't
    show this pattern at all, so they're excluded from the denominator
    rather than silently counted as "not re-engaged".

    Pulls every interaction in a single query and does the grouping in
    Python, rather than issuing one query per customer.
    """
    rows = (
        db.query(Interaction.customer_id, Interaction.timestamp)
        .order_by(Interaction.customer_id, Interaction.timestamp)
        .all()
    )

    timestamps_by_customer: dict = defaultdict(list)
    for customer_id, ts in rows:
        if ts is not None:
            timestamps_by_customer[customer_id].append(ts)

    threshold = timedelta(days=inactivity_threshold_days)
    re_engaged_ids = []
    eligible = 0

    for customer_id, timestamps in timestamps_by_customer.items():
        if len(timestamps) < 2:
            continue
        eligible += 1
        timestamps.sort()
        for earlier, later in zip(timestamps, timestamps[1:]):
            if (later - earlier) > threshold:
                re_engaged_ids.append(customer_id)
                break  # one qualifying gap is enough - no need to keep scanning

    rate = round((len(re_engaged_ids) / eligible * 100), 2) if eligible else 0.0

    return {
        "inactivity_threshold_days": inactivity_threshold_days,
        "eligible_customers": eligible,
        "re_engaged_count": len(re_engaged_ids),
        "re_engagement_rate_pct": rate,
        "customer_ids": re_engaged_ids,
    }


# ── Metric metadata for the system report ──

def get_cohort_metric_metadata() -> dict:
    """Formulas/sources/justification for each cohort metric, for the write-up."""
    return {
        "retention_rate": {
            "formula": "active_customers / total_customers * 100 (per signup-month cohort)",
            "signal_source": "Customer.status == 'Active'",
            "business_justification": "Measures how much of a signup cohort is still actively using the product.",
        },
        "churn_rate": {
            "formula": "churned_customers / total_customers * 100 (per signup-month cohort)",
            "signal_source": "Customer.status == 'Churned'",
            "business_justification": "Direct measure of cohort attrition; complements retention_rate.",
        },
        "churn_score": {
            "formula": "Weighted heuristic sum (see compute_churn_score) over engagement, NPS, sentiment, CSAT, recency, escalations, open tickets, and ticket volume.",
            "signal_source": "Customer + Ticket + Interaction aggregates",
            "business_justification": "Hand-tuned early-warning score (0-100) to flag at-risk accounts before they churn; not a statistically fitted model.",
        },
        "retention_curve": {
            "formula": "Geometric decay curve calibrated to land on the cohort's real current retention_rate at its current age in months.",
            "signal_source": "Derived from retention_rate (current-status only, no historical snapshots)",
            "business_justification": "Approximates the shape of retention loss over time for visualization; explicitly labelled as an estimate.",
        },
        "re_engagement_rate_pct": {
            "formula": "re_engaged_count / eligible_customers * 100, where re-engaged = any consecutive interaction gap > inactivity_threshold_days followed by a later interaction.",
            "signal_source": "Interaction.timestamp sequence per customer",
            "business_justification": "Identifies win-back success: customers who went quiet and came back on their own, useful for evaluating retention campaigns.",
        },
    }


# ── PDF export ──

def export_cohort_pdf(db: Session, heart_metrics_fn=None) -> bytes:
    """
    Renders the same cohort analysis from cohort_analysis() (also exposed
    as JSON via /cohorts/analysis and /export/cohort) as a PDF report -
    on top of the JSON export, not replacing it.

    `heart_metrics_fn` is passed in (pass heart.heart_metrics) rather than
    imported directly, so this module doesn't need a hard dependency on
    heart.py. If it's not provided, the HEART section is just skipped.

    Returns raw PDF bytes - it's on the caller (the FastAPI route) to wrap
    this in a StreamingResponse with the right content-type.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    cohorts = cohort_analysis(db)
    re_engagement = re_engagement_analysis(db)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#16213e"),
    )
    story = []

    story.append(Paragraph("E-Cell AI CRM &mdash; Cohort &amp; HEART Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.utcnow().isoformat()} UTC", styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    # ── Cohort summary table ──
    story.append(Paragraph("Cohort Summary", styles["Heading2"]))
    table_data = [["Cohort", "Total", "Active", "Churned", "Retention %", "Churn %", "High Risk"]]
    for c in cohorts:
        table_data.append([
            c["cohort"], c["total_customers"], c["active_customers"],
            c["churned_customers"], c["retention_rate"], c["churn_rate"],
            c["high_risk_customers"],
        ])
    cohort_table = Table(table_data, hAlign="LEFT")
    cohort_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(cohort_table)
    story.append(Spacer(1, 0.25 * inch))

    # ── Re-engagement ──
    story.append(Paragraph("Re-engagement", styles["Heading2"]))
    story.append(Paragraph(
        f"{re_engagement['re_engaged_count']} of {re_engagement['eligible_customers']} "
        f"eligible customers re-engaged after a "
        f"{re_engagement['inactivity_threshold_days']}+ day gap "
        f"({re_engagement['re_engagement_rate_pct']}%).",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.25 * inch))

    # ── HEART metrics (skipped if heart_metrics_fn wasn't passed in) ──
    if heart_metrics_fn is not None:
        try:
            heart = heart_metrics_fn(db)
            story.append(Paragraph("HEART Framework", styles["Heading2"]))
            heart_rows = [["Metric", "Score"]]
            for key in ["Happiness", "Engagement", "Adoption", "Retention", "Task_Success"]:
                heart_rows.append([key, heart.get(key, "N/A")])
            heart_table = Table(heart_rows, hAlign="LEFT")
            heart_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(heart_table)
            story.append(Spacer(1, 0.25 * inch))
        except Exception:
            # A broken HEART calculation shouldn't take down the whole PDF
            # export - the JSON endpoints are still the source of truth for
            # HEART numbers, so we just drop this section and move on.
            pass

    # ── Top industries ──
    industry_counts: dict = defaultdict(int)
    for customer in db.query(Customer).all():
        industry_counts[customer.industry or "Unknown"] += 1
    top_industries = sorted(industry_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    story.append(Paragraph("Top Industries", styles["Heading2"]))
    industry_rows = [["Industry", "Customers"]] + [[name, count] for name, count in top_industries]
    industry_table = Table(industry_rows, hAlign="LEFT")
    industry_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(industry_table)

    doc.build(story)
    return buffer.getvalue()