from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime
from typing import Optional

from src.models import Customer, Ticket, Interaction
from src.schemas import CustomerCreate, TicketCreate


# ── Customer CRUD ──

def create_customer(db: Session, data: CustomerCreate) -> Customer:
    customer = Customer(**data.model_dump())
    db.add(customer)
    db.commit()
    db.refresh(customer)
    return customer


def get_customer(db: Session, customer_id: int) -> Optional[Customer]:
    return db.query(Customer).filter(Customer.id == customer_id).first()


def get_all_customers(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Customer).offset(skip).limit(limit).all()


def update_customer(db: Session, customer_id: int, updates: dict) -> Optional[Customer]:
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        return None
    for key, value in updates.items():
        setattr(customer, key, value)
    db.commit()
    db.refresh(customer)
    return customer


def delete_customer(db: Session, customer_id: int) -> bool:
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        return False
    db.delete(customer)
    db.commit()
    return True


# ── Ticket CRUD + lifecycle ──

def create_ticket(db: Session, data: TicketCreate) -> Ticket:
    ticket = Ticket(**data.model_dump(), created_at=datetime.utcnow())
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


def get_ticket(db: Session, ticket_id: int) -> Optional[Ticket]:
    return db.query(Ticket).filter(Ticket.id == ticket_id).first()


def get_tickets_by_customer(db: Session, customer_id: int):
    return (
        db.query(Ticket)
        .filter(Ticket.customer_id == customer_id)
        .order_by(desc(Ticket.created_at))
        .all()
    )


def update_ticket_status(db: Session, ticket_id: int, status: str) -> Optional[Ticket]:
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        return None

    ticket.status = status
    ticket.updated_at = datetime.utcnow()

    # Only stamp resolved_at the first time we hit a resolved/closed state -
    # don't clobber it if the ticket gets touched again later.
    if status in ["Resolved", "Closed"] and not ticket.resolved_at:
        ticket.resolved_at = datetime.utcnow()

    db.commit()
    db.refresh(ticket)
    return ticket


def delete_ticket(db: Session, ticket_id: int) -> bool:
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        return False
    db.delete(ticket)
    db.commit()
    return True


# ── Segmentation by industry ──
# This is the original industry-only version, kept as-is for the
# /customers/segments/by-industry route. src/segmentation.py has a more
# general version that handles other dimensions (tier, tenure, etc.) -
# didn't want to touch this one and risk changing its response shape.

def get_customer_segments(db: Session):
    customers = db.query(Customer).all()

    # One grouped query for ticket counts instead of a COUNT per customer
    # in the loop below - this used to be an N+1 and got slow once the
    # customer table grew.
    ticket_counts = dict(
        db.query(Ticket.customer_id, func.count(Ticket.id))
        .group_by(Ticket.customer_id)
        .all()
    )

    segments = {}

    for customer in customers:
        industry = customer.industry
        if industry not in segments:
            segments[industry] = {
                "industry": industry,
                "customers": [],
                "avg_engagement": 0,
                "active_count": 0,
                "churned_count": 0,
            }

        ticket_count = ticket_counts.get(customer.id, 0)

        segments[industry]["customers"].append({
            "id": customer.id,
            "name": customer.name,
            "tier": customer.tier,
            "status": customer.status,
            "engagement_score": customer.engagement_score,
            "ticket_count": ticket_count,
        })

        if customer.status == "Active":
            segments[industry]["active_count"] += 1
        if customer.status == "Churned":
            segments[industry]["churned_count"] += 1

    for seg in segments.values():
        scores = [c["engagement_score"] for c in seg["customers"]]
        seg["avg_engagement"] = round(sum(scores) / len(scores), 2) if scores else 0
        seg["total_customers"] = len(seg["customers"])
        del seg["customers"]  # don't ship the full customer list in the response

    return list(segments.values())


# ── Customer timeline ──

def get_customer_timeline(db: Session, customer_id: int):
    tickets = (
        db.query(Ticket)
        .filter(Ticket.customer_id == customer_id)
        .all()
    )

    interactions = (
        db.query(Interaction)
        .filter(Interaction.customer_id == customer_id)
        .all()
    )

    timeline = []

    for t in tickets:
        timeline.append({
            "type": "ticket",
            "id": t.id,
            "title": t.title,
            "status": t.status,
            "priority": t.priority,
            "category": t.category,
            "timestamp": t.created_at.isoformat() if t.created_at else None,
        })

    for i in interactions:
        timeline.append({
            "type": "interaction",
            "ticket_id": None,
            "id": i.id,
            "channel": i.channel,
            "message": i.message,
            "sentiment": i.sentiment,
            "csat_score": i.csat_score,
            "timestamp": i.timestamp.isoformat() if i.timestamp else None,
        })

    # Newest first, tickets and interactions interleaved
    timeline.sort(
        key=lambda x: x["timestamp"] or "",
        reverse=True
    )

    return {
        "customer_id": customer_id,
        "total_events": len(timeline),
        "timeline": timeline,
    }


# ── CRM context for the AI agent ──

def get_customer_context(db: Session, customer_id: int) -> dict:
    """
    Pulls the core Customer profile fields the agent needs for grounding.

    Deliberately just the Customer row - ticket/interaction aggregates
    already live in memory.get_customer_memory(), so we don't duplicate
    those queries here. The agent combines this profile with that memory
    dict itself.

    If we ever need account-management fields that don't exist on Customer
    yet (account manager, ARR, renewal date, etc.), this is where they'd
    get added.
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        return {"profile_available": False}

    return {
        "profile_available": True,
        "name": customer.name,
        "company": (
            customer.email.split("@", 1)[1].split(".", 1)[0].title()
            if customer.email and "@" in customer.email else None
        ),
        "industry": customer.industry,
        "tier": customer.tier,
        "signup_date": customer.signup_date.isoformat() if customer.signup_date else None,
        "status": customer.status,
        "engagement_score": customer.engagement_score,
        "nps_score": customer.nps_score,
        "last_interaction_date": (
            customer.last_interaction_date.isoformat()
            if customer.last_interaction_date else None
        ),
    }