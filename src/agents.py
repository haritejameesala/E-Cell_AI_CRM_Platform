import requests
import math
import time
import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# ─── Ollama Config ────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL        = os.getenv("OLLAMA_MODEL", "llama3")
AGENT_ID     = f"ollama-{MODEL}"

REFUSAL_PHRASES = [
    "i don't know", "i cannot", "i'm not sure",
    "no information", "unclear", "i have no", "not available"
]


# ─── Core Ollama Call ─────────────────────────────────────────────────────────

def ollama_generate(prompt: str) -> dict:
    """
    Send a prompt to Ollama and return the full response dict
    including logprobs for confidence scoring.
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":    MODEL,
                "prompt":   prompt,
                "logprobs": True,      # enables token-level log probabilities
                "stream":   False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


# ─── True Confidence via Logprobs ─────────────────────────────────────────────

def compute_confidence_logprobs(token_logprobs: list) -> float:
    """
    True model confidence — directly from token-level log probabilities.

    Method: Geometric mean of per-token probabilities.
    - Each logprob converted to prob via e^logprob
    - Geometric mean chosen over arithmetic mean because:
      a single uncertain token (prob=0.02) should lower overall
      confidence — arithmetic mean hides this, geometric mean
      correctly penalises it.

    Returns: float 0.0 to 100.0
    """
    if not token_logprobs:
        return 0.0

    # Ollama returns logprobs as list of dicts:
    # [{"token": "Hi", "logprob": -0.12, "top_logprobs": [...]}, ...]
    # Extract float value regardless of format.
    raw_values = []
    for lp in token_logprobs:
        if isinstance(lp, dict):
            val = lp.get("logprob", None)      # Ollama dict format
        elif isinstance(lp, (int, float)):
            val = float(lp)                    # plain float format
        else:
            val = None
        if val is not None and isinstance(val, (int, float)) and val > -100:
            raw_values.append(val)

    if not raw_values:
        return 0.0

    probs = [math.exp(lp) for lp in raw_values]

    # Geometric mean via log-sum trick (numerically stable)
    log_sum = sum(math.log(p) for p in probs if p > 0)
    geometric_mean = math.exp(log_sum / len(probs))

    return round(geometric_mean * 100, 2)


def apply_confidence_penalties(confidence: float, answer: str) -> float:
    """
    Post-process confidence with two additional guards:
    1. Refusal phrase detection  — model signalling its own uncertainty
    2. Answer length sanity check — very short answers are suspicious
    """
    if any(phrase in answer.lower() for phrase in REFUSAL_PHRASES):
        confidence = min(confidence, 30.0)

    if len(answer.split()) < 5:
        confidence = min(confidence, 20.0)

    return round(confidence, 2)


# ─── Ticket Summarization ─────────────────────────────────────────────────────

def summarize_ticket(ticket_text: str) -> dict:
    """
    Summarize a support ticket using Llama3 via Ollama.
    Confidence is computed from true token-level logprobs.
    """
    start = time.time()

    prompt = f"""You are a CRM support analyst. Summarize the support ticket below.

Return your response in EXACTLY this format (no extra text):
SUMMARY: <one sentence summary>
KEY ISSUES: <comma-separated list of issues>
URGENCY: <Low / Medium / High / Critical>
SUGGESTED RESOLUTION: <one actionable next step>

Ticket:
{ticket_text}"""

    data     = ollama_generate(prompt)
    latency  = round(time.time() - start, 3)

    raw            = data.get("response", "")
    token_logprobs = data.get("logprobs", [])

    confidence = compute_confidence_logprobs(token_logprobs)
    confidence = apply_confidence_penalties(confidence, raw)

    # Parse structured output
    lines = {}
    for line in raw.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            lines[key.strip().upper()] = val.strip()

    return {
        "summary":            lines.get("SUMMARY", raw),
        "key_issues":         lines.get("KEY ISSUES", "N/A"),
        "urgency":            lines.get("URGENCY", "N/A"),
        "suggested_response": lines.get("SUGGESTED RESOLUTION", "N/A"),
        "confidence_score":   confidence,
        "confidence_method":  "logprobs_geometric_mean",
        "processing_latency": latency,
        "agent_id":           AGENT_ID,
    }


# ─── LangGraph Ticket Routing Workflow ────────────────────────────────────────

class TicketState(TypedDict):
    category:       str
    priority:       str
    status:         str
    assigned_agent: str


def route_ticket(state: TicketState) -> TicketState:
    """Node 1 — Route ticket to correct team based on category."""
    routing_map = {
        "Billing":   "Billing_Team",
        "Technical": "Tech_Team",
        "Account":   "Account_Team",
    }
    state["assigned_agent"] = routing_map.get(
        state.get("category", ""), "General_Team"
    )
    return state


def escalate_ticket(state: TicketState) -> TicketState:
    """Node 2 — Set ticket status based on priority."""
    state["status"] = (
        "Escalated"    if state.get("priority", "") in ["High", "Critical"]
        else "In Progress"
    )
    return state


# Build and compile LangGraph state machine
workflow = StateGraph(TicketState)
workflow.add_node("route_ticket",    route_ticket)
workflow.add_node("escalate_ticket", escalate_ticket)
workflow.set_entry_point("route_ticket")
workflow.add_edge("route_ticket",    "escalate_ticket")
workflow.add_edge("escalate_ticket", END)
ticket_graph = workflow.compile()


def run_ticket_workflow(category: str, priority: str) -> dict:
    """Run the full LangGraph routing + escalation pipeline."""
    return ticket_graph.invoke({
        "category":       category,
        "priority":       priority,
        "status":         "Open",
        "assigned_agent": "",
    })


# ─── Deterministic Factual Query Routing ──────────────────────────────────────
# Certain customer questions have a single, unambiguous answer that already
# lives in the CRM database (via crm_context / memory) - e.g. "what's my
# tier?" or "how many open tickets do I have?". Routing these through the
# LLM adds latency, cost, and a small but nonzero hallucination risk for no
# benefit: the "reasoning" the model would do is just restating a value it
# was handed verbatim. So before touching Ollama at all, customer_agent()
# tries to answer the query deterministically from data already loaded.
# Only questions that require actual reasoning/synthesis (ticket
# summarization, churn/health explanations, recommendations, sentiment
# analysis, general "tell me about my account" style questions, etc.) fall
# through to the LLM-based reasoning path further below.

# field -> keyword phrases that indicate the user is asking for exactly that
# fact. Order matters: more specific phrases are listed before broader ones
# that could otherwise shadow them (e.g. "open ticket" before a bare
# "ticket" match).
_FACTUAL_FIELD_KEYWORDS = [
    ("open_tickets",     ["open ticket", "how many open", "pending ticket"]),
    ("latest_ticket",    ["latest ticket", "last ticket", "most recent ticket", "newest ticket"]),
    ("total_tickets",    ["how many ticket", "total ticket", "number of tickets"]),
    ("company",          ["company"]),
    ("industry",         ["industry"]),
    ("tier",             ["tier", "subscription plan", "which plan"]),
    ("status",           ["account status", "customer status", "am i active", "my status"]),
    ("engagement_score", ["engagement score", "how engaged"]),
    ("nps_score",        ["nps score", "nps", "net promoter"]),
    ("signup_date",      ["signup date", "sign-up date", "sign up date", "when did i sign up", "member since", "joined on"]),
    ("name",             ["what is my name", "what's my name", "my name on file"]),
]


def _detect_factual_field(query: str) -> Optional[str]:
    """
    Returns the field name if `query` is asking for a single concrete fact
    this system already has on file, else None. Deliberately conservative
    (keyword-based) - anything ambiguous or that reads like it wants
    explanation/reasoning falls through to the LLM path rather than risk a
    wrong deterministic short-circuit.
    """
    q = (query or "").lower()
    for field, keywords in _FACTUAL_FIELD_KEYWORDS:
        if any(kw in q for kw in keywords):
            return field
    return None


def _format_factual_answer(field: str, crm_context: dict, memory: dict) -> tuple:
    """
    Looks up `field` directly from crm_context/memory (never the LLM) and
    returns (label, value_or_None, source_label, source_entry_or_None).
    """
    profile_ok = bool(crm_context and crm_context.get("profile_available"))

    profile_fields = {
        "company":          ("Company", "Customer Profile"),
        "industry":         ("Industry", "Customer Profile"),
        "tier":             ("Tier", "Customer Profile"),
        "status":           ("Status", "Customer Profile"),
        "engagement_score": ("Engagement Score", "Customer Profile"),
        "nps_score":        ("NPS Score", "Customer Profile"),
        "signup_date":      ("Signup Date", "Customer Profile"),
        "name":             ("Name", "Customer Profile"),
    }

    if field in profile_fields:
        label, source = profile_fields[field]
        value = crm_context.get(field) if profile_ok else None
        source_entry = {"type": "Customer Profile", "field": field} if value is not None else None
        return label, value, source, source_entry

    if field == "open_tickets":
        value = memory.get("open_tickets")
        source_entry = {"type": "Ticket History", "field": "open_tickets"} if value is not None else None
        return "Open Tickets", value, "Ticket History", source_entry

    if field == "total_tickets":
        value = memory.get("total_tickets")
        source_entry = {"type": "Ticket History", "field": "total_tickets"} if value is not None else None
        return "Total Tickets", value, "Ticket History", source_entry

    if field == "latest_ticket":
        title = memory.get("latest_ticket_title")
        status = memory.get("latest_ticket_status")
        ticket_id = memory.get("latest_ticket_id")
        if title is None:
            return "Latest Ticket", None, "Ticket History", None
        value = f"{title} (Status: {status})" if status else title
        source_entry = {"type": "Ticket", "ticket_id": ticket_id, "title": title}
        return "Latest Ticket", value, "Ticket History", source_entry

    return field.replace("_", " ").title(), None, "CRM Records", None


def _try_factual_response(query: str, crm_context: dict, memory: dict, start: float) -> Optional[dict]:
    """
    Attempts to answer `query` deterministically (no LLM call, no latency
    beyond a dict lookup) when it maps to one of the well-known factual
    fields above. Returns a fully-formed response dict (same shape as the
    LLM path, so callers/schemas don't need to branch) on success, or None
    if this query doesn't look factual - the caller should then fall
    through to the LLM-based reasoning path.

    Response style follows the concise CRM format requested for factual
    answers, e.g.:

        Company:
        TechNova Solutions

        Source:
        Customer Profile
    """
    field = _detect_factual_field(query)
    if field is None:
        return None

    label, value, source_label, source_entry = _format_factual_answer(field, crm_context, memory)

    if value is None:
        answer = f"{label}:\nUnavailable\n\nSource:\n{source_label} (no data on file)"
        confidence = 0.0
        sources: list = []
        reason = ["Requested field is not present in the CRM record"]
    else:
        answer = f"{label}:\n{value}\n\nSource:\n{source_label}"
        confidence = 100.0
        sources = [source_entry] if source_entry else []
        reason = None

    result = {
        "answer": answer,
        "source": f"deterministic_lookup:{source_label}",
        "sources": sources,
        "agent_id": AGENT_ID,
        "processing_latency": round(time.time() - start, 3),
        "confidence_score": confidence,
        "confidence_method": "deterministic_field_lookup",
        "confidence_breakdown": {"field": field, "data_available": value is not None},
    }
    if reason is not None:
        result["confidence_reason"] = reason
    return result


# ─── Customer Agent (Memory-Aware, CRM-Grounded) ──────────────────────────────

# Caps applied when rendering prompt sections, so a customer with a long
# history doesn't blow up the prompt size as interaction volume scales.
MAX_MESSAGE_CHARS = 200
MAX_SUMMARY_CHARS = 600


def _truncate(text, max_chars: int) -> str:
    text = str(text) if text is not None else ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _format_crm_profile(crm_context: dict) -> str:
    """Renders the Customer Profile section of the prompt, or a clear
    'unavailable' note if we don't have a profile to ground on."""
    if not crm_context or not crm_context.get("profile_available"):
        return "Customer profile is unavailable."

    def _val(key):
        value = crm_context.get(key)
        return value if value is not None else "Unavailable"

    return (
        f"Name: {_val('name')}\n"
        f"Company: {_val('company')}\n"
        f"Industry: {_val('industry')}\n"
        f"Tier: {_val('tier')}\n"
        f"Signup date: {_val('signup_date')}\n"
        f"Status: {_val('status')}\n"
        f"Engagement score: {_val('engagement_score')}\n"
        f"NPS score: {_val('nps_score')}\n"
        f"Last interaction date: {_val('last_interaction_date')}"
    )


def _format_ticket_history(memory: dict) -> str:
    """Renders the Ticket History section from the enriched memory dict
    (see memory.get_customer_memory).

    Deliberately keeps "latest ticket" facts (the newest ticket's own
    category/priority) separate from "most common" facts (the customer's
    historical pattern across all tickets) - conflating the two previously
    mislabeled the most-common values as if they described the latest
    ticket.
    """
    def _val(key, default="Unavailable"):
        value = memory.get(key)
        return value if value is not None else default

    return (
        f"Total tickets: {_val('total_tickets', 0)}\n"
        f"Open tickets: {_val('open_tickets', 0)}\n"
        f"Resolved tickets: {_val('resolved_tickets', 0)}\n"
        f"Escalated tickets: {_val('escalated_tickets', 0)}\n"
        f"Latest ticket: {_val('latest_ticket_title')}\n"
        f"Latest ticket status: {_val('latest_ticket_status')}\n"
        f"Latest ticket category: {_val('latest_ticket_category')}\n"
        f"Latest ticket priority: {_val('latest_ticket_priority')}\n"
        f"Most common ticket category: {_val('most_common_ticket_category')}\n"
        f"Most common ticket priority: {_val('most_common_ticket_priority')}\n"
        f"Latest resolution date: {_val('latest_resolution_date')}"
    )


def _format_interaction_summary(memory: dict) -> str:
    """Renders the Interaction Summary section (recent messages, preferred
    channel, sentiment/CSAT averages, recency, health label)."""
    def _val(key, default="Unavailable"):
        value = memory.get(key)
        return value if value is not None else default

    short_term = memory.get("short_term", [])
    if short_term:
        recent_lines = "\n".join(
            f"  [{m['channel']}] {_truncate(m['message'], MAX_MESSAGE_CHARS)}"
            for m in short_term
        )
    else:
        recent_lines = "  No recent interactions."

    return (
        f"Recent interactions:\n{recent_lines}\n"
        f"Preferred channel: {_val('preferred_channel')}\n"
        f"Average sentiment: {_val('avg_sentiment')}\n"
        f"Average CSAT: {_val('avg_csat')}\n"
        f"Recent activity (last 30 days): {_val('recent_30d_interactions', 0)}\n"
        f"Customer health: {_val('customer_health')}"
    )


def _text_contains(answer: str, value) -> bool:
    """Case-insensitive substring check used to detect whether a specific
    known fact actually shows up in the generated answer. Necessarily
    imprecise (a paraphrase like "Acme Corp" -> "Acme Corporation" won't
    match), so this is used as a soft evidence signal, not a hard gate."""
    if value is None or not str(value).strip():
        return False
    return str(value).strip().lower() in (answer or "").lower()


def _score_profile_usage(crm_context: dict, answer: str) -> float:
    """
    0-100: fraction of the customer's concrete profile values (name,
    company, industry, tier, status) that actually appear in the generated
    answer. Distinguishes "the profile was available" from "the profile
    was used" - a complete profile doesn't make an answer trustworthy if
    the model never drew on it (e.g. a question the CRM data can't answer
    at all, like "what's my refund amount?").
    """
    if not crm_context or not crm_context.get("profile_available"):
        return 0.0

    fields = ["name", "company", "industry", "tier", "status"]
    present = [f for f in fields if crm_context.get(f) is not None]
    if not present:
        return 0.0

    used = sum(1 for f in present if _text_contains(answer, crm_context.get(f)))
    return round((used / len(present)) * 100, 2)


def _score_ticket_usage(memory: dict, answer: str) -> float:
    """0-100: fraction of the customer's concrete ticket facts (latest
    ticket title/status, most common category) that show up in the
    answer."""
    if not memory.get("total_tickets"):
        return 0.0

    candidates = [
        memory.get("latest_ticket_title"),
        memory.get("latest_ticket_status"),
        memory.get("most_common_ticket_category"),
    ]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return 0.0

    used = sum(1 for c in candidates if _text_contains(answer, c))
    return round((used / len(candidates)) * 100, 2)


def _score_memory_usage(memory: dict, answer: str) -> float:
    """0-100: fraction of the customer's concrete interaction-memory facts
    (preferred channel, health label) that show up in the answer."""
    if not memory.get("total_interactions"):
        return 0.0

    candidates = [memory.get("preferred_channel"), memory.get("customer_health")]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return 0.0

    used = sum(1 for c in candidates if _text_contains(answer, c))
    return round((used / len(candidates)) * 100, 2)


def _score_profile_completeness(crm_context: dict) -> float:
    """
    0-100 score for how much of the CRM profile is actually present. This
    is the heaviest-weighted confidence component: an answer grounded in a
    sparse profile is inherently less trustworthy than one grounded in a
    complete one.
    """
    if not crm_context or not crm_context.get("profile_available"):
        return 0.0

    fields = [
        "name", "company", "industry", "tier", "signup_date",
        "status", "engagement_score", "nps_score", "last_interaction_date",
    ]
    present = sum(1 for f in fields if crm_context.get(f) is not None)
    return round((present / len(fields)) * 100, 2)


def _score_ticket_grounding(memory: dict) -> float:
    """0-100 score for how much ticket-history signal backs the answer."""
    total_tickets = memory.get("total_tickets") or 0
    if total_tickets == 0:
        return 0.0

    signals = [
        total_tickets > 0,
        memory.get("most_common_ticket_category") is not None,
        memory.get("latest_ticket_title") is not None,
        (
            (memory.get("resolved_tickets") or 0)
            + (memory.get("open_tickets") or 0)
            + (memory.get("escalated_tickets") or 0)
        ) > 0,
    ]
    return round((sum(1 for s in signals if s) / len(signals)) * 100, 2)


def _score_memory_grounding(memory: dict) -> float:
    """0-100 score for how much interaction-history signal backs the answer."""
    if not memory.get("total_interactions"):
        return 0.0

    signals = [
        (memory.get("total_interactions") or 0) > 0,
        memory.get("avg_sentiment") is not None,
        memory.get("avg_csat") is not None,
        len(memory.get("short_term", [])) > 0,
    ]
    return round((sum(1 for s in signals if s) / len(signals)) * 100, 2)


# ─── Source Citations (Feature 1) ─────────────────────────────────────────────

# Profile fields worth citing individually, in priority order.
_CITABLE_PROFILE_FIELDS = ["name", "company", "industry", "tier", "status",
                           "engagement_score", "nps_score", "signup_date"]


def _build_sources(crm_context: dict, memory: dict, answer: str) -> list:
    """
    Builds a list of concrete, existing CRM records the generated answer
    appears to have actually drawn on - never invented ones.

    This walks the SAME grounding data the model was given (profile fields,
    the latest ticket, and the short-term interaction window) and includes
    an entry only when there is textual evidence in `answer` that it was
    used (reusing the substring-match approach from the confidence scorers
    above). Because every candidate comes from real CRM_context/memory
    records with real ids, this list can never contain a hallucinated
    citation - at worst it under-cites (misses a paraphrased reference).
    """
    sources = []

    if crm_context and crm_context.get("profile_available"):
        for field in _CITABLE_PROFILE_FIELDS:
            value = crm_context.get(field)
            if value is not None and _text_contains(answer, value):
                sources.append({"type": "Customer Profile", "field": field})

    ticket_id = memory.get("latest_ticket_id")
    if ticket_id is not None:
        ticket_signals = [
            memory.get("latest_ticket_title"),
            memory.get("latest_ticket_status"),
            memory.get("latest_ticket_category"),
            str(ticket_id),
        ]
        if any(_text_contains(answer, s) for s in ticket_signals if s is not None):
            sources.append({
                "type": "Ticket",
                "ticket_id": ticket_id,
                "title": memory.get("latest_ticket_title"),
            })

    for interaction in memory.get("short_term", []):
        interaction_id = interaction.get("interaction_id")
        if interaction_id is None:
            continue
        channel = interaction.get("channel")
        if channel and _text_contains(answer, channel):
            sources.append({"type": "Interaction", "interaction_id": interaction_id, "channel": channel})
            break  # channel mentions are a weak/general signal - cite at most one match

    return sources


# ─── Confidence Explainability (Feature 7) ────────────────────────────────────

CONFIDENCE_EXPLAIN_THRESHOLD = 60.0


def _build_confidence_reasons(crm_context: dict, memory: dict, confidence: dict) -> list:
    """
    When overall confidence is low, explains WHY in plain language, using
    the same breakdown already computed by compute_agent_confidence. Only
    called when confidence is below CONFIDENCE_EXPLAIN_THRESHOLD - a
    high-confidence answer doesn't need justification.
    """
    reasons = []
    breakdown = confidence.get("confidence_breakdown", {})

    if breakdown.get("crm_profile_availability", 0) < 100:
        reasons.append("Customer profile incomplete")
    if not memory.get("total_tickets"):
        reasons.append("No recent ticket found")
    if not memory.get("total_interactions"):
        reasons.append("No interaction history")
    if breakdown.get("crm_profile_usage", 0) == 0 and crm_context.get("profile_available"):
        reasons.append("Answer did not draw on the available customer profile")
    if breakdown.get("token_logprob_confidence", 100) < 40:
        reasons.append("Low model token-level confidence")

    if not reasons:
        reasons.append("Multiple grounding signals were weak or partially available")

    return reasons


def compute_agent_confidence(
    crm_context: dict,
    memory: dict,
    token_logprobs: list,
    answer: str,
) -> dict:
    """
    Weighted confidence blend for the customer agent:
      40% CRM profile grounding
      30% Ticket/history grounding
      20% Memory (interaction-history) grounding
      10% Raw token-logprob confidence

    Each of the three CRM-derived components (profile/tickets/memory) is
    itself the average of two sub-scores:
      - availability: how much relevant data existed to draw on
      - usage:        how much of that data actually shows up in the
                       generated answer (see _score_*_usage)

    This closes the gap where a fully-complete profile could produce a
    high confidence score even when the model's answer didn't (or
    couldn't) actually use any of it - e.g. a question like "what's my
    refund amount?" that the CRM data simply can't answer. If the answer
    shows no evidence of using an available signal, that component's score
    drops by half instead of staying at 100% just because the data existed.

    Returns the blended score plus a breakdown of every sub-score so the
    caller (or a future UI) can see exactly what drove the number, instead
    of a single opaque figure.
    """
    profile_availability = _score_profile_completeness(crm_context)
    profile_usage = _score_profile_usage(crm_context, answer)
    profile_component = round((profile_availability + profile_usage) / 2, 2)

    ticket_availability = _score_ticket_grounding(memory)
    ticket_usage = _score_ticket_usage(memory, answer)
    ticket_component = round((ticket_availability + ticket_usage) / 2, 2)

    memory_availability = _score_memory_grounding(memory)
    memory_usage = _score_memory_usage(memory, answer)
    memory_component = round((memory_availability + memory_usage) / 2, 2)

    logprob_score = compute_confidence_logprobs(token_logprobs)
    logprob_score = apply_confidence_penalties(logprob_score, answer)

    blended = (
        profile_component * 0.40
        + ticket_component * 0.30
        + memory_component * 0.20
        + logprob_score * 0.10
    )

    return {
        "confidence_score": round(blended, 2),
        "confidence_method": (
            "weighted_blend(profile=40%,tickets=30%,memory=20%,logprobs=10%); "
            "each CRM component averages data-availability with evidence-of-usage in the answer"
        ),
        "confidence_breakdown": {
            "crm_profile_grounding": profile_component,
            "crm_profile_availability": profile_availability,
            "crm_profile_usage": profile_usage,
            "ticket_history_grounding": ticket_component,
            "ticket_history_availability": ticket_availability,
            "ticket_history_usage": ticket_usage,
            "memory_grounding": memory_component,
            "memory_availability": memory_availability,
            "memory_usage": memory_usage,
            "token_logprob_confidence": logprob_score,
        },
    }


def customer_agent(query: str, memory: dict, crm_context: Optional[dict] = None) -> dict:
    """
    Memory-aware, CRM-grounded multi-turn support agent.

    Combines three layers of context before answering:
      - CRM profile        (name, company, tier, status, engagement/NPS...)
      - Ticket history     (counts by status, latest ticket, common category/priority)
      - Interaction memory (short-term window + long-term behavioural summary)

    `crm_context` is expected to come from crm.get_customer_context(); if it
    isn't supplied (e.g. an older caller), the agent still runs using memory
    alone and is instructed to say profile info is unavailable rather than
    guess it - so this stays backward compatible with the original
    `customer_agent(query, memory)` call signature.

    Confidence is a weighted blend of profile/ticket/memory grounding and
    raw token-logprob confidence (see compute_agent_confidence).

    Fact vs. reasoning routing: before any LLM call, the query is checked
    against _detect_factual_field(). Simple factual look-ups (company,
    industry, tier, status, engagement score, NPS, signup date, open ticket
    count, latest ticket, etc.) are answered directly from crm_context/
    memory with 100% confidence and zero LLM latency. Only questions that
    require actual reasoning or synthesis (summaries, churn/health
    explanations, recommendations, sentiment analysis, ...) reach the
    Ollama call below.
    """
    start = time.time()
    crm_context = crm_context or {"profile_available": False}

    # ── Deterministic fact routing (skips the LLM entirely when possible) ──
    factual_response = _try_factual_response(query, crm_context, memory, start)
    if factual_response is not None:
        return factual_response

    # ── Hallucination guard (Feature 8) ────────────────────────────────────
    # If there is literally no grounding data for this customer at all (no
    # profile, no tickets, no interactions), there is nothing the model
    # could truthfully answer with - skip the LLM call entirely rather than
    # risk it generating plausible-sounding but fabricated details.
    has_profile = bool(crm_context.get("profile_available"))
    has_tickets = bool(memory.get("total_tickets"))
    has_interactions = bool(memory.get("total_interactions"))

    if not has_profile and not has_tickets and not has_interactions:
        no_data_answer = "I don't have enough information in the CRM database."
        return {
            "answer": no_data_answer,
            "source": "no_grounding_data_available",
            "sources": [],
            "agent_id": AGENT_ID,
            "processing_latency": round(time.time() - start, 3),
            "confidence_score": 0.0,
            "confidence_method": "no_grounding_data",
            "confidence_breakdown": {},
            "confidence_reason": [
                "Customer profile incomplete",
                "No recent ticket found",
                "No interaction history",
            ],
        }

    system_prompt = (
        "You are a CRM support assistant. Answer using ONLY the CRM Context "
        "and Memory provided below. Never invent facts, dates, ticket "
        "numbers, or figures that are not present in the context. If the "
        "information needed to answer is not present, say clearly that it "
        "is unavailable rather than guessing. Write in a concise, "
        "professional CRM-style tone with no conversational filler or "
        "sign-offs - this is a reasoning/explanation request (e.g. a "
        "summary, churn/health explanation, or recommendation), not a "
        "casual chat message."
    )

    crm_context_block = (
        f"Customer Profile\n-----------------\n{_format_crm_profile(crm_context)}\n\n"
        f"Ticket History\n--------------\n{_format_ticket_history(memory)}\n\n"
        f"Interaction Summary\n-------------------\n{_format_interaction_summary(memory)}"
    )

    long_term_text = _truncate(
        memory.get("long_term_summary", "No interaction history available."),
        MAX_SUMMARY_CHARS,
    )

    response_instructions = (
        "- Respond in a helpful, professional, and personalised way.\n"
        "- If you reference past interactions, cite the channel (e.g. \"In your last Chat...\").\n"
        "- When you use a specific ticket or interaction from the context above, cite it "
        "naturally by its category/type and ID if one is given, e.g. "
        "\"According to your latest Billing ticket (#381)...\" or "
        "\"In your last Chat interaction...\".\n"
        "- Do NOT invent facts, ticket numbers, dates, or figures not present in the context above.\n"
        "- If you are unsure or the context lacks the answer, say so clearly.\n"
        "- Do NOT use conversational filler such as \"I'm happy to help\", "
        "\"best regards\", \"I hope this helps\", or similar sign-offs. "
        "Respond in concise, professional CRM-style prose - lead with the "
        "answer, keep it tight."
    )

    prompt = f"""{system_prompt}

=== CRM CONTEXT ===
{crm_context_block}

=== LONG-TERM SUMMARY ===
{long_term_text}

=== CURRENT CUSTOMER QUESTION ===
{query}

=== RESPONSE INSTRUCTIONS ===
{response_instructions}"""

    data    = ollama_generate(prompt)
    latency = round(time.time() - start, 3)

    answer         = data.get("response", "")
    token_logprobs = data.get("logprobs", [])

    confidence = compute_agent_confidence(crm_context, memory, token_logprobs, answer)
    sources = _build_sources(crm_context, memory, answer)

    result = {
        "answer":             answer,
        "source":             f"crm_profile + ticket_history + customer_memory + ollama-{MODEL}",
        # Additive: structured list of the concrete CRM records (profile
        # fields / ticket / interaction) this answer appears grounded in.
        "sources":            sources,
        "agent_id":           AGENT_ID,
        "processing_latency": latency,
        **confidence,
    }

    # Additive (Feature 7): only attach an explanation when confidence is
    # actually low - a high-confidence answer doesn't need justification,
    # and this keeps the response lean for the common case.
    if result["confidence_score"] < CONFIDENCE_EXPLAIN_THRESHOLD:
        result["confidence_reason"] = _build_confidence_reasons(crm_context, memory, confidence)

    return result