from collections import defaultdict
from datetime import datetime, timedelta
from statistics import median
from typing import Iterable, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.cohort import cohort_analysis
from src.models import Customer, Interaction, Ticket
from src.segmentation import segment_customers


RESOLVED_STATUSES = {"Resolved", "Closed"}
OPEN_STATUSES = {"Open", "In Progress"}


def _hours_between(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    if not start or not end or end < start:
        return None
    return round((end - start).total_seconds() / 3600, 2)


def _summary(values: Iterable[float]) -> dict:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"avg_hours": None, "median_hours": None, "min_hours": None, "max_hours": None}
    return {
        "avg_hours": round(sum(vals) / len(vals), 2),
        "median_hours": round(median(vals), 2),
        "min_hours": round(vals[0], 2),
        "max_hours": round(vals[-1], 2),
    }


def time_to_first_resolution_by_cohort(db: Session) -> list:
    # Only the FIRST resolved ticket per customer counts here - that's why
    # we sort by resolved_at and skip a customer_id once we've already
    # recorded one for them.
    rows = (
        db.query(Customer.signup_date, Ticket.customer_id, Ticket.created_at, Ticket.resolved_at)
        .join(Ticket, Ticket.customer_id == Customer.id)
        .filter(Ticket.resolved_at.isnot(None))
        .order_by(Ticket.customer_id, Ticket.resolved_at)
        .all()
    )
    first_by_customer = {}
    for signup_date, customer_id, created_at, resolved_at in rows:
        if customer_id in first_by_customer:
            continue
        hours = _hours_between(created_at, resolved_at)
        if hours is None:
            continue
        first_by_customer[customer_id] = (signup_date.strftime("%Y-%m"), hours)

    by_cohort = defaultdict(list)
    for cohort, hours in first_by_customer.values():
        by_cohort[cohort].append(hours)

    return [
        {"cohort": cohort, "resolved_customers": len(hours), **_summary(hours)}
        for cohort, hours in sorted(by_cohort.items())
    ]


def resolution_metrics_by_category(db: Session, sla_hours: int = 48) -> list:
    rows = db.query(Ticket.category, Ticket.status, Ticket.created_at, Ticket.resolved_at).all()
    grouped = defaultdict(lambda: {"resolution_hours": [], "resolved": 0, "open": 0, "escalated": 0, "total": 0, "sla_breaches": 0})
    for category, status, created_at, resolved_at in rows:
        key = category or "Unknown"
        bucket = grouped[key]
        bucket["total"] += 1
        if status in RESOLVED_STATUSES:
            bucket["resolved"] += 1
            hours = _hours_between(created_at, resolved_at)
            if hours is not None:
                bucket["resolution_hours"].append(hours)
                if hours > sla_hours:
                    bucket["sla_breaches"] += 1
        elif status in OPEN_STATUSES:
            bucket["open"] += 1
        elif status == "Escalated":
            bucket["escalated"] += 1

    results = []
    for category, data in sorted(grouped.items()):
        resolved = data["resolved"]
        results.append({
            "category": category,
            "total_tickets": data["total"],
            "resolved_tickets": resolved,
            "open_tickets": data["open"],
            "escalated_tickets": data["escalated"],
            "sla_breach_pct": round(data["sla_breaches"] / resolved * 100, 2) if resolved else 0.0,
            **_summary(data["resolution_hours"]),
        })
    return results


def resolution_metrics_by_agent(db: Session) -> list:
    tickets = db.query(Ticket).all()
    customer_ids_by_agent = defaultdict(set)
    grouped = defaultdict(lambda: {"resolution_hours": [], "resolved": 0, "escalations": 0, "total": 0})
    for ticket in tickets:
        agent = ticket.assigned_agent or "Unassigned"
        grouped[agent]["total"] += 1
        customer_ids_by_agent[agent].add(ticket.customer_id)
        if ticket.status in RESOLVED_STATUSES:
            grouped[agent]["resolved"] += 1
            hours = _hours_between(ticket.created_at, ticket.resolved_at)
            if hours is not None:
                grouped[agent]["resolution_hours"].append(hours)
        if ticket.status == "Escalated":
            grouped[agent]["escalations"] += 1

    # CSAT/sentiment/NPS aren't linked to agents directly (interactions
    # only reference customers, not agents), so we approximate "this
    # agent's customer sentiment" by pooling stats across every customer
    # they've handled a ticket for.
    interaction_rows = db.query(Interaction.customer_id, Interaction.csat_score, Interaction.sentiment).all()
    customer_rows = db.query(Customer.id, Customer.nps_score).all()
    csat_by_customer = defaultdict(list)
    sentiment_by_customer = defaultdict(list)
    nps_by_customer = defaultdict(list)
    for customer_id, csat, sentiment in interaction_rows:
        if csat is not None:
            csat_by_customer[customer_id].append(csat)
        if sentiment is not None:
            sentiment_by_customer[customer_id].append(sentiment)
    for customer_id, nps in customer_rows:
        if nps is not None:
            nps_by_customer[customer_id].append(nps)

    results = []
    for agent, data in sorted(grouped.items()):
        customer_ids = customer_ids_by_agent[agent]
        csats = [v for cid in customer_ids for v in csat_by_customer.get(cid, [])]
        sentiments = [v for cid in customer_ids for v in sentiment_by_customer.get(cid, [])]
        nps_scores = [v for cid in customer_ids for v in nps_by_customer.get(cid, [])]
        summary = _summary(data["resolution_hours"])
        results.append({
            "agent": agent,
            "total_tickets": data["total"],
            "resolved_tickets": data["resolved"],
            "escalations": data["escalations"],
            "avg_resolution_hours": summary["avg_hours"],
            "avg_csat": round(sum(csats) / len(csats), 2) if csats else None,
            "avg_nps": round(sum(nps_scores) / len(nps_scores), 2) if nps_scores else None,
            "avg_sentiment": round(sum(sentiments) / len(sentiments), 3) if sentiments else None,
        })
    return results


def configurable_cohorts(db: Session, group_by: str = "signup") -> dict:
    if group_by == "signup":
        cohorts = cohort_analysis(db)
        return {"group_by": "signup", "groups": cohorts}
    if group_by in {"industry", "tier"}:
        return {"group_by": group_by, "groups": segment_customers(db, group_by)["segments"]}
    if group_by == "behavior":
        rows = cohort_analysis(db)
        totals = defaultdict(int)
        for cohort in rows:
            for behavior, count in cohort.get("behavioral_cohorts", {}).items():
                totals[behavior] += count
        return {
            "group_by": "behavior",
            "groups": [
                {"segment": behavior, "total_customers": count}
                for behavior, count in sorted(totals.items())
            ],
        }
    raise ValueError("group_by must be one of: signup, industry, tier, behavior")


def expanded_heart_metrics(db: Session) -> dict:
    """Deeper breakdowns behind the top-level HEART numbers - per-cohort CSAT, weekly active users, survival trend, etc."""
    cohort_rows = cohort_analysis(db)
    interactions = db.query(Interaction).all()
    tickets = db.query(Ticket).all()
    customers = db.query(Customer).all()
    customer_by_id = {c.id: c for c in customers}

    csat_by_cohort = defaultdict(list)
    csat_by_channel = defaultdict(list)
    for interaction in interactions:
        customer = customer_by_id.get(interaction.customer_id)
        if interaction.csat_score is None:
            continue
        if customer and customer.signup_date:
            csat_by_cohort[customer.signup_date.strftime("%Y-%m")].append(interaction.csat_score)
        csat_by_channel[interaction.channel or "Unknown"].append(interaction.csat_score)

    weekly_active = defaultdict(set)
    for interaction in interactions:
        if interaction.timestamp:
            start = interaction.timestamp.date() - timedelta(days=interaction.timestamp.weekday())
            weekly_active[start.isoformat()].add(interaction.customer_id)

    ticket_rate = defaultdict(int)
    for ticket in tickets:
        if ticket.created_at:
            ticket_rate[ticket.created_at.date().replace(day=1).isoformat()] += 1

    total_customers = len(customers) or 1
    resolved = sum(1 for t in tickets if t.status in RESOLVED_STATUSES)
    escalated = sum(1 for t in tickets if t.status == "Escalated")
    # No column distinguishes an AI-resolved ticket from a human-resolved
    # one, so we use the "_Team" suffix on assigned_agent as a stand-in.
    ai_resolved = sum(1 for t in tickets if t.status in RESOLVED_STATUSES and (t.assigned_agent or "").endswith("_Team"))

    return {
        "happiness": {
            "csat_by_cohort": [{"cohort": k, "avg_csat": round(sum(v) / len(v), 2)} for k, v in sorted(csat_by_cohort.items())],
            "csat_by_agent": resolution_metrics_by_agent(db),
            "csat_by_channel": [{"channel": k, "avg_csat": round(sum(v) / len(v), 2)} for k, v in sorted(csat_by_channel.items())],
        },
        "engagement": {
            "weekly_active_users": [{"week": k, "active_users": len(v)} for k, v in sorted(weekly_active.items())],
            "average_interaction_depth": round(len(interactions) / total_customers, 2),
            "ticket_creation_rate": [{"month": k, "tickets_created": v} for k, v in sorted(ticket_rate.items())],
        },
        "adoption": {
            "crm_usage_pct": round(len({t.customer_id for t in tickets}) / total_customers * 100, 2),
            "ai_usage_proxy_tickets_routed_pct": round(len([t for t in tickets if (t.assigned_agent or "").endswith("_Team")]) / len(tickets) * 100, 2) if tickets else 0.0,
            "feature_adoption": "Ticket creation is used as the available support-feature adoption proxy.",
        },
        "retention": {
            "monthly_retention": [{"cohort": c["cohort"], "retention_rate": c["retention_rate"]} for c in cohort_rows],
            "average_customer_lifespan_days": round(sum((datetime.now().date() - c.signup_date).days for c in customers if c.signup_date) / len(customers), 2) if customers else 0.0,
            "survival_trend": [{"cohort": c["cohort"], "retention_curve": c["retention_curve"]} for c in cohort_rows],
        },
        "task_success": {
            "resolution_pct": round(resolved / len(tickets) * 100, 2) if tickets else 0.0,
            "first_contact_resolution_pct": None,
            "ai_vs_human_resolution": {"ai_proxy": ai_resolved, "human_proxy": max(resolved - ai_resolved, 0)},
            "escalation_pct": round(escalated / len(tickets) * 100, 2) if tickets else 0.0,
            "note": "First-contact resolution is not directly stored; AI vs human uses assigned_agent naming as a proxy.",
        },
    }


def agent_quality_metrics(db: Session, latency_events: Optional[list] = None) -> dict:
    """
    Rough, explainable estimate of how well-grounded the AI agent's
    answers are likely to be, based on how much usable CRM context exists
    per customer. This is NOT a supervised evaluation score - we don't
    have human-labelled "was this answer good" data to train on, so this
    is the honest substitute: how much data did the agent actually have
    to work with.
    """
    customers = db.query(Customer).all()
    tickets = db.query(Ticket).all()
    interactions = db.query(Interaction).all()

    total_customers = len(customers)
    if not total_customers:
        return {
            "available": False,
            "agent_quality_score": None,
            "grounding_score": None,
            "confidence": None,
            "latency": None,
            "estimated_hallucination_risk": None,
            "response_completeness": None,
            "note": "Insufficient Data",
        }

    now = datetime.now()
    tickets_by_customer = defaultdict(list)
    interactions_by_customer = defaultdict(list)
    for ticket in tickets:
        tickets_by_customer[ticket.customer_id].append(ticket)
    for interaction in interactions:
        interactions_by_customer[interaction.customer_id].append(interaction)

    customer_ids_with_tickets = set(tickets_by_customer)
    customer_ids_with_interactions = set(interactions_by_customer)
    profile_fields = ["name", "email", "industry", "tier", "signup_date", "engagement_score", "status"]
    profile_completeness_values = []
    ticket_quality_values = []
    memory_quality_values = []
    freshness_values = []
    for customer in customers:
        present = sum(1 for field in profile_fields if getattr(customer, field, None) is not None)
        profile_completeness_values.append(present / len(profile_fields) * 100)

        customer_tickets = tickets_by_customer.get(customer.id, [])
        resolved = sum(1 for ticket in customer_tickets if ticket.status in RESOLVED_STATUSES)
        escalated = sum(1 for ticket in customer_tickets if ticket.status == "Escalated")
        open_count = sum(1 for ticket in customer_tickets if ticket.status in OPEN_STATUSES)
        ticket_volume_score = min(len(customer_tickets), 3) / 3 * 45
        resolution_signal = (resolved / len(customer_tickets) * 35) if customer_tickets else 0
        friction_penalty = min(30, escalated * 10 + open_count * 5)
        ticket_quality_values.append(max(0, ticket_volume_score + resolution_signal + 20 - friction_penalty))

        customer_interactions = interactions_by_customer.get(customer.id, [])
        interaction_depth = min(len(customer_interactions), 5) / 5 * 35
        sentiment_available = sum(1 for item in customer_interactions if item.sentiment is not None)
        csat_available = sum(1 for item in customer_interactions if item.csat_score is not None)
        sentiment_score = (sentiment_available / len(customer_interactions) * 25) if customer_interactions else 0
        csat_score = (csat_available / len(customer_interactions) * 25) if customer_interactions else 0
        avg_sentiment = (
            sum(item.sentiment for item in customer_interactions if item.sentiment is not None) / sentiment_available
            if sentiment_available else 0
        )
        sentiment_quality = max(0, min(15, (avg_sentiment + 1) / 2 * 15))
        memory_quality_values.append(interaction_depth + sentiment_score + csat_score + sentiment_quality)

        latest_interaction = max(
            (item.timestamp for item in customer_interactions if item.timestamp),
            default=None,
        )
        if latest_interaction is None:
            freshness_values.append(0)
        else:
            age_days = max(0, (now - latest_interaction).days)
            freshness_values.append(max(0, 100 - min(100, age_days / 90 * 100)))

    profile_completeness = sum(profile_completeness_values) / len(profile_completeness_values)
    ticket_coverage = len(customer_ids_with_tickets) / total_customers * 100
    memory_coverage = len(customer_ids_with_interactions) / total_customers * 100
    ticket_context_quality = sum(ticket_quality_values) / len(ticket_quality_values)
    memory_context_quality = sum(memory_quality_values) / len(memory_quality_values)
    recency_score = sum(freshness_values) / len(freshness_values)
    source_coverage = (ticket_coverage + memory_coverage) / 2

    grounded_customers = customer_ids_with_tickets | customer_ids_with_interactions
    grounding_score = round(
        0.35 * profile_completeness
        + 0.25 * ticket_context_quality
        + 0.25 * memory_context_quality
        + 0.15 * recency_score,
        2,
    )
    response_completeness = round(len(grounded_customers) / total_customers * 100, 2)

    latency_values = [
        event.get("latency")
        for event in (latency_events or [])
        if isinstance(event.get("latency"), (int, float))
    ]
    avg_latency = round(sum(latency_values) / len(latency_values), 4) if latency_values else None

    confidence = round(
        0.45 * grounding_score
        + 0.25 * source_coverage
        + 0.20 * response_completeness
        + 0.10 * recency_score,
        2,
    )
    escalation_rate = (
        len([ticket for ticket in tickets if ticket.status == "Escalated"]) / len(tickets) * 100
        if tickets else 0
    )
    hallucination_risk = round(
        max(0.0, min(100.0, 100.0 - grounding_score + escalation_rate * 0.15)),
        2,
    )
    quality_score = round(
        0.40 * grounding_score
        + 0.30 * confidence
        + 0.20 * response_completeness
        + 0.10 * max(0.0, 100.0 - hallucination_risk),
        2,
    )

    return {
        "available": True,
        "agent_quality_score": quality_score,
        "grounding_score": grounding_score,
        "confidence": confidence,
        "latency": avg_latency,
        "estimated_hallucination_risk": hallucination_risk,
        "response_completeness": response_completeness,
        "crm_profile_completeness": round(profile_completeness, 2),
        "ticket_context_coverage": round(ticket_coverage, 2),
        "memory_context_coverage": round(memory_coverage, 2),
        "ticket_context_quality": round(ticket_context_quality, 2),
        "memory_context_quality": round(memory_context_quality, 2),
        "recency_score": round(recency_score, 2),
        "note": (
            "Runtime quality estimate computed from CRM profile completeness, "
            "ticket context quality, interaction-memory richness, recency, "
            "source coverage, and current-process latency telemetry. This is "
            "not a supervised evaluator-label score."
        ),
    }


def cohort_evaluation(db: Session, threshold: int = 70) -> dict:
    """
    Treats Customer.status == "Churned" as ground truth and churn_score >=
    threshold as the prediction, then computes standard classification
    metrics against that. Useful for sanity-checking whether the churn
    heuristic is actually pointing at the right accounts.
    """
    cohorts = cohort_analysis(db)
    tp = fp = tn = fn = coverage_count = total = 0
    for cohort in cohorts:
        for customer in cohort.get("customer_churn_scores", []):
            total += 1
            predicted = customer["churn_score"] >= threshold
            actual = customer["status"] == "Churned"
            if predicted:
                coverage_count += 1
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and not actual:
                tn += 1
            elif not predicted and actual:
                fn += 1
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": round(accuracy, 3) if accuracy is not None else None,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "coverage": round(coverage_count / total * 100, 2) if total else 0.0,
        "cohort_completeness": round(len([c for c in cohorts if c.get("total_customers", 0) > 0]) / len(cohorts) * 100, 2) if cohorts else 0.0,
    }


def resolution_dashboard(db: Session) -> dict:
    tickets = db.query(Ticket).all()
    total = len(tickets)
    closed = [t for t in tickets if t.status in RESOLVED_STATUSES]
    ai_closed = [t for t in closed if (t.assigned_agent or "").endswith("_Team")]
    trend = defaultdict(int)
    for ticket in tickets:
        if ticket.status == "Escalated" and ticket.updated_at:
            trend[ticket.updated_at.date().replace(day=1).isoformat()] += 1
    return {
        "ticket_closure_pct": round(len(closed) / total * 100, 2) if total else 0.0,
        "ai_resolution_pct": round(len(ai_closed) / len(closed) * 100, 2) if closed else 0.0,
        "human_resolution_pct": round((len(closed) - len(ai_closed)) / len(closed) * 100, 2) if closed else 0.0,
        "escalation_trend": [{"month": k, "escalations": v} for k, v in sorted(trend.items())],
    }