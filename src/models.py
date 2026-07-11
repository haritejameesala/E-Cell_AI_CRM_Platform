from datetime import datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)

from src.db import Base


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)

    # LLM-generated during seeding, not user-entered
    company = Column(String(255), nullable=False)

    industry = Column(String(100), nullable=False)
    tier = Column(String(50), nullable=False)

    signup_date = Column(Date, nullable=False)

    # Feeds churn scoring and cohort grouping - see cohort.py
    engagement_score = Column(Float, nullable=False)
    status = Column(String(50), nullable=False)

    # 0-10 per-customer rating, not the classic -100/+100 NPS formula
    nps_score = Column(Float, nullable=True)

    # Backfilled from Interaction timestamps after seeding runs
    last_interaction_date = Column(DateTime, nullable=True)


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)

    customer_id = Column(
        Integer,
        ForeignKey("customers.id"),
        index=True,
        nullable=False,
    )

    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)

    category = Column(String(100), nullable=False)
    priority = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)

    assigned_agent = Column(String(100), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Only set once the ticket actually gets resolved/closed
    resolved_at = Column(DateTime, nullable=True)


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)

    customer_id = Column(
        Integer,
        ForeignKey("customers.id"),
        index=True,
        nullable=False,
    )

    ticket_id = Column(
        Integer,
        ForeignKey("tickets.id"),
        index=True,
        nullable=False,
    )

    channel = Column(String(50), nullable=False)

    message = Column(Text, nullable=False)

    # -1 to 1
    sentiment = Column(Float, nullable=False)

    # 1-5, only present when the customer actually rated the interaction
    csat_score = Column(Float, nullable=True)

    timestamp = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )