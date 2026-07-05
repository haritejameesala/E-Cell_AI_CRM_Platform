from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime
import io
import logging
import time
import uuid

from src.db import SessionLocal, engine
from src import models
from src.schemas import (
    CustomerCreate, CustomerResponse,
    TicketCreate, TicketResponse, TicketUpdate,
    AgentQuery, AgentQueryResponse,
    TicketSummaryResponse, HEARTResponse
)
from src.models import Customer, Ticket, Interaction
from src.agents import summarize_ticket, customer_agent, run_ticket_workflow
from src.memory import get_customer_memory
from src.cohort import (
    cohort_analysis, re_engagement_analysis, export_cohort_pdf,
    get_cohort_metric_metadata,
)
from src.heart import heart_metrics, heart_metrics_by_cohort, get_heart_metric_metadata
from src.evaluation import (
    agent_quality_metrics,
    cohort_evaluation,
    configurable_cohorts,
    expanded_heart_metrics,
    resolution_dashboard,
    resolution_metrics_by_agent,
    resolution_metrics_by_category,
    time_to_first_resolution_by_cohort,
)
from src.segmentation import segment_customers, VALID_DIMENSIONS
from src import crm

logger = logging.getLogger(__name__)

# ─── Create tables on startup ─────────────────────────────────────────────────
models.Base.metadata.create_all(bind=engine)

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="E-Cell AI CRM Platform",
    description="AI-integrated CRM with LangChain summarization, LangGraph agents, HEART framework, and cohort analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Startup: warm up Ollama so first real request is fast ────────────────────
@app.on_event("startup")
async def warmup_ollama():
    """
    Send a tiny prompt to Ollama on startup so the model is already
    loaded into GPU memory before the first real API call.
    This prevents the first summarize/agent call from timing out.
    """
    import threading
    def _warmup():
        try:
            import requests as _req
            import os
            url   = os.getenv("OLLAMA_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "llama3")
            _req.post(f"{url}/api/generate",
                json={"model": model, "prompt": "hi", "stream": False},
                timeout=120)
            logger.info("Ollama warmup complete (%s loaded into memory)", model)
        except Exception as e:
            logger.warning("Ollama warmup failed (is Ollama running?): %s", e)
    threading.Thread(target=_warmup, daemon=True).start()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LATENCY_EVENTS = []


@app.middleware("http")
async def collect_latency(request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = round(time.time() - start, 4)
    LATENCY_EVENTS.append({
        "path": request.url.path,
        "method": request.method,
        "latency": latency,
        "timestamp": datetime.utcnow().isoformat(),
    })
    del LATENCY_EVENTS[:-500]
    response.headers["X-Process-Time"] = str(latency)
    return response

security = HTTPBearer(auto_error=False)

# ─── Role-based access control (Feature 6) ─────────────────────────────────────
# Simple token-role mapping (replace with JWT in production). Each token also
# optionally carries an `agent_id` - this is what lets the lightweight
# ownership check below restrict an "agent" role to only their own tickets,
# without needing a full auth system. In production, `agent_id` would come
# from a verified JWT claim instead of a static table.
ROLE_TOKENS = {
    "agent-token-001":      {"role": "agent",              "agent_id": "Priya_Sharma"},
    "supervisor-token-001": {"role": "supervisor",          "agent_id": None},
    "admin-token-001":      {"role": "admin",                "agent_id": None},
    "analytics-token-001":  {"role": "analytics_readonly",  "agent_id": None},
}

# Roles, per task spec:
#   Admin       - everything
#   Supervisor  - Tickets, Analytics (cohort/HEART), Customers
#   Agent       - Own tickets, Customer (agent) queries
#   Analytics   - Read-only (customers, tickets, cohort, HEART)
ROLE_PERMISSIONS = {
    "agent": [
        "read_customers", "read_tickets", "create_tickets", "summarize", "agent_query",
    ],
    "supervisor": [
        "read_customers", "read_tickets", "create_tickets", "summarize",
        "route", "update_status", "cohort", "heart", "segments",
    ],
    "admin": [
        "read_customers", "read_tickets", "create_tickets", "summarize", "route",
        "update_status", "create_customers", "delete", "cohort", "heart",
        "agent_query", "segments", "system_metadata",
    ],
    "analytics_readonly": [
        "read_customers", "read_tickets", "cohort", "heart", "segments",
    ],
}


def get_identity(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Returns {"role": str, "agent_id": Optional[str]} for the caller."""
    if credentials is None:
        return {"role": "admin", "agent_id": None}  # open for demo; restrict in production
    entry = ROLE_TOKENS.get(credentials.credentials)
    if not entry:
        raise HTTPException(status_code=401, detail="Invalid token")
    return entry


def get_role(identity: dict = Depends(get_identity)) -> str:
    """Backward-compatible: existing code/dependencies that only need the
    role string (not the full identity) keep working unchanged."""
    return identity["role"]


def require_permission(permission: str):
    def checker(role: str = Depends(get_role)):
        if permission not in ROLE_PERMISSIONS.get(role, []):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' does not have permission: {permission}"
            )
        return role
    return checker


def enforce_ticket_ownership(ticket, identity: dict):
    """
    Lightweight "own tickets only" restriction for the 'agent' role (Feature
    6). Supervisors/admins/analytics are unaffected. Agents may only act on
    tickets currently assigned to their own agent_id.
    """
    if identity["role"] == "agent" and ticket.assigned_agent != identity.get("agent_id"):
        raise HTTPException(
            status_code=403,
            detail="Agents may only access tickets assigned to them.",
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def audit(agent_id: str = "system", latency: float = 0.0):
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "agent_id": agent_id,
        "processing_latency": latency,
        "confidence_score": None,
        "api_version": "v1",
        "request_id": str(uuid.uuid4())[:8],
    }


# ─── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "message": "E-Cell AI CRM Platform is running",
        "version": "1.0.0",
        "docs": "/docs",
        **audit()
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Customer & Ticket Management
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Customers ────────────────────────────────────────────────────────────────

@app.post("/api/v1/customers", tags=["Customers"], status_code=201)
def create_customer(
    customer: CustomerCreate,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("create_customers"))
):
    start = time.time()
    existing = db.query(Customer).filter(Customer.email == customer.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Customer with this email already exists")

    new_customer = crm.create_customer(db, customer)
    cohort = new_customer.signup_date.strftime("%Y-%m")

    return {
        "id": new_customer.id,
        "status": "created",
        "cohort_assignment": cohort,
        **audit(latency=round(time.time() - start, 3))
    }


@app.get("/api/v1/customers", tags=["Customers"])
def list_customers(
    skip: int = 0, limit: int = 50,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("read_customers"))
):
    start = time.time()
    customers = crm.get_all_customers(db, skip=skip, limit=limit)
    return {
        "total": len(customers),
        "customers": [
            {
                "id": c.id, "name": c.name, "email": c.email,
                "industry": c.industry, "tier": c.tier,
                "status": c.status, "engagement_score": c.engagement_score,
                "signup_date": str(c.signup_date),
                "cohort": c.signup_date.strftime("%Y-%m"),
            }
            for c in customers
        ],
        **audit(latency=round(time.time() - start, 3))
    }


@app.get("/api/v1/customers/{customer_id}", tags=["Customers"])
def get_customer(
    customer_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("read_customers"))
):
    start = time.time()
    customer = crm.get_customer(db, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return {
        "id": customer.id, "name": customer.name, "email": customer.email,
        "industry": customer.industry, "tier": customer.tier,
        "status": customer.status, "engagement_score": customer.engagement_score,
        "signup_date": str(customer.signup_date), "nps_score": customer.nps_score,
        **audit(latency=round(time.time() - start, 3))
    }


@app.put("/api/v1/customers/{customer_id}", tags=["Customers"])
def update_customer(
    customer_id: int,
    updates: dict,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("create_customers"))
):
    start = time.time()
    customer = crm.update_customer(db, customer_id, updates)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return {"id": customer.id, "status": "updated", **audit(latency=round(time.time() - start, 3))}


@app.delete("/api/v1/customers/{customer_id}", tags=["Customers"])
def delete_customer(
    customer_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("delete"))
):
    start = time.time()
    success = crm.delete_customer(db, customer_id)
    if not success:
        raise HTTPException(status_code=404, detail="Customer not found")
    return {"id": customer_id, "status": "deleted", **audit(latency=round(time.time() - start, 3))}


@app.get("/api/v1/customers/{customer_id}/timeline", tags=["Customers"])
def customer_timeline(
    customer_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("read_customers"))
):
    start = time.time()
    timeline = crm.get_customer_timeline(db, customer_id)
    return {**timeline, **audit(latency=round(time.time() - start, 3))}


@app.get("/api/v1/customers/segments/by-industry", tags=["Customers"])
def customer_segments(
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("read_customers"))
):
    """Unchanged - preserved exactly for backward compatibility."""
    start = time.time()
    segments = crm.get_customer_segments(db)
    return {"segments": segments, **audit(latency=round(time.time() - start, 3))}


@app.get("/api/v1/customers/segments/{dimension}", tags=["Customers"])
def customer_segments_by_dimension(
    dimension: str,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("segments"))
):
    """
    Feature 2 - generic, rule-based segmentation across any supported
    dimension: industry, tier, engagement, tenure, ticket_frequency,
    status, nps, churn_risk. See src/segmentation.py for the reusable
    bucket-function engine backing this (no per-dimension code duplication).
    """
    start = time.time()
    if dimension not in VALID_DIMENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dimension '{dimension}'. Valid options: {VALID_DIMENSIONS}",
        )
    result = segment_customers(db, dimension)
    return {**result, **audit(latency=round(time.time() - start, 3))}


# ─── Tickets ──────────────────────────────────────────────────────────────────

@app.post("/api/v1/tickets/create", tags=["Tickets"], status_code=201)
def create_ticket(
    ticket: TicketCreate,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("create_tickets"))
):
    start = time.time()

    # Run LangGraph routing automatically on creation
    routing = run_ticket_workflow(ticket.category, ticket.priority)
    ticket.assigned_agent = routing["assigned_agent"]
    ticket.status = routing["status"]

    new_ticket = crm.create_ticket(db, ticket)

    return {
        "ticket_id": new_ticket.id,
        "category": new_ticket.category,
        "assigned_agent": new_ticket.assigned_agent,
        "status": new_ticket.status,
        **audit(latency=round(time.time() - start, 3))
    }


@app.get("/api/v1/tickets/{ticket_id}", tags=["Tickets"])
def get_ticket(
    ticket_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("read_tickets")),
    identity: dict = Depends(get_identity),
):
    start = time.time()
    ticket = crm.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    enforce_ticket_ownership(ticket, identity)
    return {
        "ticket_id": ticket.id, "customer_id": ticket.customer_id,
        "title": ticket.title, "category": ticket.category,
        "priority": ticket.priority, "status": ticket.status,
        "assigned_agent": ticket.assigned_agent,
        "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
        "resolved_at": ticket.resolved_at.isoformat() if ticket.resolved_at else None,
        **audit(latency=round(time.time() - start, 3))
    }


@app.post("/api/v1/tickets/{ticket_id}/summarize", tags=["Tickets"])
def summarize(
    ticket_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("summarize")),
    identity: dict = Depends(get_identity),
):
    ticket = crm.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    enforce_ticket_ownership(ticket, identity)

    ticket_text = f"{ticket.title}. {ticket.description}"
    result = summarize_ticket(ticket_text)

    return {
        "ticket_id": ticket_id,
        **result,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/tickets/{ticket_id}/route", tags=["Tickets"])
def route_ticket(
    ticket_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("route")),
    identity: dict = Depends(get_identity),
):
    start = time.time()
    ticket = crm.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    enforce_ticket_ownership(ticket, identity)

    result = run_ticket_workflow(ticket.category, ticket.priority)
    crm.update_ticket_status(db, ticket_id, result["status"])
    crm.update_customer(db, ticket.customer_id, {"last_interaction_date": datetime.utcnow()})

    return {
        "ticket_id": ticket_id,
        "assigned_agent": result["assigned_agent"],
        "status": result["status"],
        **audit(latency=round(time.time() - start, 3))
    }


@app.put("/api/v1/tickets/{ticket_id}/status", tags=["Tickets"])
def update_ticket_status(
    ticket_id: int,
    data: TicketUpdate,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("update_status"))
):
    start = time.time()
    ticket = crm.update_ticket_status(db, ticket_id, data.status)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return {
        "ticket_id": ticket.id,
        "new_status": ticket.status,
        "resolved_at": ticket.resolved_at.isoformat() if ticket.resolved_at else None,
        **audit(latency=round(time.time() - start, 3))
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — LLM Agent
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/query/agent", tags=["AI Agent"])
def query_agent(
    data: AgentQuery,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("agent_query"))
):
    # BUGFIX: this endpoint previously only loaded interaction/ticket
    # `memory` and never fetched the customer's CRM profile at all, so
    # every query - factual or reasoning - saw "profile unavailable" even
    # for customers that exist and have a full profile. customer_agent()
    # has always accepted a `crm_context` argument (see src/agents.py); it
    # just was never being passed in from this route.
    memory = get_customer_memory(data.customer_id, db)
    crm_context = crm.get_customer_context(db, data.customer_id)
    result = customer_agent(query=data.query, memory=memory, crm_context=crm_context)

    return {
        "customer_id": data.customer_id,
        **result,
        "timestamp": datetime.utcnow().isoformat()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — Cohort Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/cohorts/analysis", tags=["Cohort Analysis"])
def get_cohort_analysis(
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("cohort"))
):
    start = time.time()
    result = cohort_analysis(db)
    return {
        "total_cohorts": len(result),
        "cohort_analysis": result,
        **audit(latency=round(time.time() - start, 3))
    }


@app.get("/api/v1/export/cohort", tags=["Cohort Analysis"])
def export_cohort(
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("cohort"))
):
    """Export full cohort analysis as structured JSON. Unchanged."""
    start = time.time()
    result = cohort_analysis(db)
    return {
        "export_format": "json",
        "generated_at": datetime.utcnow().isoformat(),
        "total_cohorts": len(result),
        "data": result,
        **audit(latency=round(time.time() - start, 3))
    }


@app.get("/api/v1/cohorts/re-engagement", tags=["Cohort Analysis"])
def get_re_engagement(
    inactivity_threshold_days: int = 30,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("cohort"))
):
    """Feature 3 - re-engagement rate: customers inactive 30+ days (configurable)
    who later became active again. See cohort.re_engagement_analysis for the
    exact definition/limitations."""
    start = time.time()
    result = re_engagement_analysis(db, inactivity_threshold_days=inactivity_threshold_days)
    return {**result, **audit(latency=round(time.time() - start, 3))}


@app.get("/api/v1/export/pdf", tags=["Cohort Analysis"])
def export_pdf(
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("cohort"))
):
    """
    Feature 4 - PDF export of the cohort + HEART report (ReportLab), in
    ADDITION to (not instead of) the existing JSON export at /export/cohort.
    """
    pdf_bytes = export_cohort_pdf(db, heart_metrics_fn=heart_metrics)
    filename = f"crm_cohort_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

@app.get("/api/v1/cohort/behavior")
def behavioral_cohorts(
    db: Session = Depends(get_db)
):
    data = cohort_analysis(db)

    result = []

    for cohort in data:
        result.append({
            "cohort": cohort["cohort"],
            "behavioral_cohorts": cohort["behavioral_cohorts"]
        })

    return result

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — HEART Framework Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/heart/dashboard", tags=["HEART Framework"])
def get_heart_dashboard(
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("heart"))
):
    start = time.time()
    metrics = heart_metrics(db)
    return {**metrics, **audit(latency=round(time.time() - start, 3))}


@app.get("/api/v1/heart/by-cohort", tags=["HEART Framework"])
def get_heart_by_cohort(
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("heart"))
):
    start = time.time()
    metrics = heart_metrics_by_cohort(db)
    return {
        "cohort_heart_scores": metrics,
        **audit(latency=round(time.time() - start, 3))
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — System Report Metadata (Feature 10)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/system/report-metadata", tags=["System Report"])
def get_system_report_metadata(
    _: str = Depends(require_permission("system_metadata"))
):
    """
    Machine-readable metric definitions, formulas, signal sources, and
    business justification for every cohort + HEART metric this system
    produces. Intended to feed the final written System Report directly,
    instead of re-deriving definitions by hand from the code.
    """
    return {
        "cohort_metrics": get_cohort_metric_metadata(),
        "heart_metrics": get_heart_metric_metadata(),
        **audit(),
    }


@app.get("/api/v1/cohorts/configurable", tags=["Evaluation"])
def get_configurable_cohorts(
    group_by: str = "signup",
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("cohort"))
):
    start = time.time()
    try:
        result = configurable_cohorts(db, group_by=group_by)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {**result, **audit(latency=round(time.time() - start, 3))}


@app.get("/api/v1/resolution/cohort", tags=["Evaluation"])
def get_resolution_by_cohort(
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("cohort"))
):
    start = time.time()
    return {
        "time_to_first_resolution_by_cohort": time_to_first_resolution_by_cohort(db),
        **audit(latency=round(time.time() - start, 3)),
    }


@app.get("/api/v1/resolution/category", tags=["Evaluation"])
def get_resolution_by_category(
    sla_hours: int = 48,
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("heart"))
):
    start = time.time()
    return {
        "sla_hours": sla_hours,
        "categories": resolution_metrics_by_category(db, sla_hours=sla_hours),
        **audit(latency=round(time.time() - start, 3)),
    }


@app.get("/api/v1/resolution/agents", tags=["Evaluation"])
def get_resolution_by_agent(
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("heart"))
):
    start = time.time()
    return {
        "agents": resolution_metrics_by_agent(db),
        **audit(latency=round(time.time() - start, 3)),
    }


@app.get("/api/v1/heart/expanded", tags=["Evaluation"])
def get_expanded_heart(
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("heart"))
):
    start = time.time()
    return {**expanded_heart_metrics(db), **audit(latency=round(time.time() - start, 3))}


@app.get("/api/v1/evaluation/agent-quality", tags=["Evaluation"])
def get_agent_quality(
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("heart"))
):
    return {**agent_quality_metrics(db, LATENCY_EVENTS), **audit()}


@app.get("/api/v1/evaluation/latency", tags=["Evaluation"])
def get_latency_metrics(
    _: str = Depends(require_permission("heart"))
):
    latencies = sorted(event["latency"] for event in LATENCY_EVENTS)

    def percentile(pct: float):
        if not latencies:
            return 0.0
        index = min(len(latencies) - 1, int(round((pct / 100) * (len(latencies) - 1))))
        return round(latencies[index], 4)

    endpoint_totals = {}
    for event in LATENCY_EVENTS:
        key = f"{event['method']} {event['path']}"
        endpoint_totals.setdefault(key, []).append(event["latency"])

    endpoints = []
    for endpoint, values in sorted(endpoint_totals.items()):
        sorted_values = sorted(values)
        p95_index = min(len(sorted_values) - 1, int(round(0.95 * (len(sorted_values) - 1))))
        endpoints.append({
            "endpoint": endpoint,
            "count": len(values),
            "avg_latency": round(sum(values) / len(values), 4),
            "p95": round(sorted_values[p95_index], 4),
        })

    return {
        "average_latency": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        "p95": percentile(95),
        "p99": percentile(99),
        "endpoint_latency": endpoints,
        "events": LATENCY_EVENTS[-100:],
        **audit(),
    }


@app.get("/api/v1/evaluation/cohorts", tags=["Evaluation"])
def get_cohort_evaluation(
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("cohort"))
):
    start = time.time()
    return {**cohort_evaluation(db), **audit(latency=round(time.time() - start, 3))}


@app.get("/api/v1/evaluation/resolution", tags=["Evaluation"])
def get_resolution_evaluation(
    db: Session = Depends(get_db),
    _: str = Depends(require_permission("heart"))
):
    start = time.time()
    return {**resolution_dashboard(db), **audit(latency=round(time.time() - start, 3))}
