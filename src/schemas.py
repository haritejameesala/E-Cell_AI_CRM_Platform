from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional, List, Dict, Any


# ─── Input Schemas ─────────────────────────────────────────────────────────────

class CustomerCreate(BaseModel):
    name: str
    email: str
    industry: str
    tier: str
    signup_date: date
    engagement_score: float
    status: str
    nps_score: Optional[float] = None


class TicketCreate(BaseModel):
    customer_id: int
    title: str
    description: str
    category: str
    priority: str
    status: str
    assigned_agent: str


class AgentQuery(BaseModel):
    customer_id: int
    query: str


class TicketUpdate(BaseModel):
    status: str


# ─── Response Schemas ──────────────────────────────────────────────────────────

class CustomerResponse(BaseModel):
    id: int
    name: str
    email: str
    industry: str
    tier: str
    signup_date: date
    engagement_score: float
    status: str
    nps_score: Optional[float]
    cohort_assignment: Optional[str] = None

    class Config:
        from_attributes = True


class TicketResponse(BaseModel):
    id: int
    customer_id: int
    title: str
    category: str
    priority: str
    status: str
    assigned_agent: str
    created_at: Optional[datetime]
    resolved_at: Optional[datetime]

    class Config:
        from_attributes = True


class TicketSummaryResponse(BaseModel):
    ticket_id: int
    summary: str
    key_issues: str
    urgency: str
    suggested_response: str
    confidence_score: float
    processing_latency: float        # seconds
    agent_id: str
    timestamp: datetime


class AgentQueryResponse(BaseModel):
    customer_id: int
    answer: str
    source: str
    confidence_score: float
    agent_id: str
    processing_latency: float
    timestamp: datetime
    # ── Additive fields (Features 1 & 7) - existing fields above are
    # unchanged, so old consumers reading only those keep working. ──────────
    sources: Optional[List[Dict[str, Any]]] = None
    confidence_method: Optional[str] = None
    confidence_breakdown: Optional[Dict[str, Any]] = None
    confidence_reason: Optional[List[str]] = None


class CohortResponse(BaseModel):
    cohort_id: str
    total_customers: int
    retention_rate: float
    churn_rate: float
    heart_scores: Optional[dict] = None
    retention_curve: Optional[list] = None


class HEARTResponse(BaseModel):
    Happiness: float
    Engagement: float
    Adoption: float
    Retention: float
    Task_Success: float
    computed_at: datetime