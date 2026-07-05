"""
Relational synthetic CRM dataset generator (OPTIMIZED).

Creates realistic, connected records:
- 500 customer profiles
- at least 1,000 support tickets linked to customers
- at least 2,000 interactions linked to customers and their tickets

The generator still uses Ollama for every generated record. It falls back to
deterministic templates ONLY when the model returns invalid/missing JSON for
a given item (same "graceful degradation" behaviour as the original script).

KEY OPTIMIZATIONS vs the original version:
  1. Batch generation: customers/tickets/interactions are requested 10-20 at a
     time per Ollama call instead of one (or ten) at a time.
  2. Parallelism: independent batches are generated concurrently with a
     ThreadPoolExecutor, sized off the CPU count.
  3. Leaner Ollama requests: low temperature + bounded num_predict/num_ctx to
     cut generation cost, scaled to the batch size so output isn't truncated.
  4. Robust JSON parsing: strips markdown fences, tolerates leading/trailing
     text, and repairs truncated JSON arrays instead of discarding them.
  5. No retry storms: a single request per batch; on failure/parse failure we
     immediately fall back to deterministic generators for the missing items
     instead of re-calling the LLM.
  6. Fewer DB round trips: rows are added in chunks with one flush/commit per
     chunk (not per row), while still preserving auto-increment IDs and FK
     relationships.
  7. In-memory email de-duplication (no per-customer SELECT against MySQL).
  8. Timeline integrity: ticket age is clamped to the customer's
     signup_days_ago (never older than the account itself), and interactions
     always land after their ticket's creation time.

See the bottom of this file / the accompanying explanation for the expected
reduction in Ollama calls, runtime, and RAM.
"""

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import requests
from sqlalchemy import func

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import models
from src.db import SessionLocal, engine


models.Base.metadata.create_all(bind=engine)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

TARGET_CUSTOMERS = 500
TARGET_TICKETS = 1000
TARGET_INTERACTIONS = 2000

# ─── Batch / concurrency tuning ────────────────────────────────────────────────
CUSTOMER_BATCH_MIN = 10
CUSTOMER_BATCH_MAX = 20

TICKET_BATCH_MIN = 10
TICKET_BATCH_MAX = 20
TICKETS_PER_CUSTOMER_MAX = 6      # cap so one customer can't dominate a batch

INTERACTION_BATCH_MIN = 10
INTERACTION_BATCH_MAX = 20
INTERACTIONS_PER_TICKET_MAX = 6

DB_CHUNK_SIZE = 300                # rows per flush/commit (tune up on higher-RAM boxes)

# Reasonable default: a single local Ollama instance typically processes
# requests sequentially per model (one GPU/CPU worker under the hood), so
# more than 2 concurrent threads mostly just queues requests and increases
# RAM usage without actually speeding anything up. Raise OLLAMA_WORKERS only
# if you've measured that your server benefits from more concurrency (e.g. a
# beefier server, multiple GPUs, or a hosted API).
MAX_WORKERS = int(os.getenv("OLLAMA_WORKERS", str(max(1, min(2, (os.cpu_count() or 2))))))

INDUSTRIES = ["AI/ML SaaS", "FinTech", "Healthcare", "EdTech", "Retail/E-commerce"]
TIERS = ["Free", "Basic", "Premium", "Enterprise"]
STATUSES = ["Active", "Inactive", "Churned"]
CATEGORIES = ["Billing", "Technical", "Account", "Onboarding", "Integration"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
TICKET_STATUSES = ["Open", "In Progress", "Escalated", "Resolved", "Closed"]
CHANNELS = ["Chat", "Email", "Call"]
AGENTS = [
    "Priya_Sharma", "Ravi_Kumar", "Ananya_Iyer", "Karthik_Nair", "Divya_Pillai",
    "Arjun_Mehta", "Sneha_Reddy", "Vikram_Rao", "Meera_Joshi", "Suresh_Babu",
]

FIRST_NAMES = [
    "Aarav", "Aditya", "Akash", "Ananya", "Anjali", "Arjun", "Aryan", "Bhavna",
    "Chirag", "Deepak", "Divya", "Farhan", "Gaurav", "Harini", "Ishaan",
    "Karan", "Kavya", "Lakshmi", "Manav", "Meera", "Nandini", "Nikhil",
    "Nisha", "Pooja", "Pranav", "Priya", "Rahul", "Rajesh", "Riya",
    "Rohit", "Sachin", "Sanjay", "Sneha", "Tanvi", "Varun", "Vikram",
    "Vishal", "Yash", "Zara", "James", "Sarah", "Michael", "Emily",
    "David", "Jessica", "Robert", "Ashley", "John", "Amanda",
]
LAST_NAMES = [
    "Agarwal", "Bhatia", "Chakraborty", "Chopra", "Das", "Deshpande",
    "Ghosh", "Gupta", "Iyer", "Joshi", "Kapoor", "Kaur", "Khan",
    "Kumar", "Malhotra", "Mehta", "Mishra", "Mukherjee", "Nair",
    "Patel", "Pillai", "Rao", "Reddy", "Saxena", "Sharma", "Singh",
    "Srivastava", "Verma", "Yadav", "Smith", "Johnson", "Williams",
    "Brown", "Garcia", "Miller", "Davis",
]
COMPANIES = [
    "TechNova Solutions", "DataBridge Analytics", "CloudMind Systems",
    "NeuralPath AI", "FinEdge Capital", "MediTrack Health",
    "LearnSphere EdTech", "ShopVault Commerce", "PaySwift FinTech",
    "CuraHealth Systems", "EduPeak Learning", "RetailGenius",
    "AlgoTrade Finance", "BioSync Healthcare", "SkillForge Academy",
    "MarketPulse AI", "SecureVault FinTech", "HealConnect Medical",
    "BrightMinds Education", "CartFlow Retail",
]


# ─── Ollama call (single attempt, tuned options) ───────────────────────────────

def ollama(prompt, num_predict=800, num_ctx=2048, timeout=120):
    """
    Single-attempt Ollama call with generation options tuned to cut cost:
      - low temperature -> more deterministic, less rambling => fewer tokens
      - num_predict capped and scaled to the batch size (avoids either wasting
        budget on tiny requests or truncating big batched requests)
      - num_ctx scaled up only for the bigger batched ticket/interaction
        prompts, which need more headroom for prompt + JSON output

    NOTE ON REDUCED RETRIES: unlike the original implementation (which retried
    failed requests up to `max_retries` times with a sleep in between), this
    version makes exactly one attempt. If it fails or returns unusable JSON,
    the caller falls back to deterministic generation for the missing items
    instead of re-hitting the LLM. This is what "reduce retries" means in
    practice here - we don't pay for repeated slow calls when a fast local
    fallback exists.
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": num_predict,
                    "num_ctx": num_ctx,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return safe_str(payload.get("response"), "")
    except Exception as exc:
        print(f"    LLM call failed (falling back where needed): {exc}")
    return ""


# ─── Robust JSON parsing (handles markdown, extra text, truncation) ───────────

def parse_json(text):
    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    decoder = json.JSONDecoder()

    starts = [index for index, char in enumerate(cleaned) if char in "[{"]
    for start in starts:
        try:
            parsed, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(parsed, (list, dict)):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def repair_json_array(text):
    """
    Best-effort repair for a truncated JSON array (common when num_predict
    cuts the model off mid-object). Strategy: trim back to the last fully
    closed '}' before the truncation point, then close the array.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    start = cleaned.find("[")
    if start == -1:
        return None

    last_close = cleaned.rfind("}")
    if last_close == -1 or last_close < start:
        return None

    candidate = cleaned[start:last_close + 1]
    if not candidate.endswith("]"):
        candidate += "]"

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None


def parse_json_list(text):
    """Parse a JSON array robustly; try repair once if the first parse fails."""
    data = parse_json(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]

    repaired = repair_json_array(text)
    if isinstance(repaired, list):
        return repaired

    return []


# ─── Small safe-coercion helpers (unchanged behaviour) ─────────────────────────

def safe_str(value, default, max_length=None):
    if value is None:
        result = default
    elif isinstance(value, str):
        result = value.strip()
    else:
        result = str(value).strip()

    if not result:
        result = default
    if max_length is not None:
        result = result[:max_length]
    return result


def safe_choice(value, choices, default):
    if isinstance(value, str):
        normalized = value.strip()
        for choice in choices:
            if normalized.lower() == choice.lower():
                return choice
    return default


def safe_int(value, default, minimum, maximum):
    try:
        if isinstance(value, bool):
            raise ValueError
        number = int(float(value))
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def safe_float(value, default, minimum, maximum, digits=1):
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(minimum, min(maximum, number)), digits)


def clean_domain(company):
    return re.sub(r"[^a-z0-9]+", "", safe_str(company, "example").lower()) or "example"


def make_email(name, company, batch_num, index):
    local = re.sub(r"[^a-z0-9.]+", "", safe_str(name, "customer").lower().replace(" ", "."))
    local = local.strip(".") or "customer"
    return f"{local}.{batch_num}.{index}@{clean_domain(company)}.com"


def normalize_customer_scores(status, engagement_score, nps_score):
    if status == "Active":
        engagement_score = max(56.0, engagement_score)
        nps_score = max(6.0, nps_score)
    elif status == "Churned":
        engagement_score = min(44.0, engagement_score)
        nps_score = min(3.9, nps_score)
    else:
        engagement_score = max(20.0, min(75.0, engagement_score))
        nps_score = max(2.0, min(8.0, nps_score))
    return round(engagement_score, 1), round(nps_score, 1)


def fallback_customer(batch_num, index, industry, tier):
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    company = random.choice(COMPANIES)
    status = random.choices(STATUSES, weights=[75, 15, 10], k=1)[0]

    if status == "Active":
        engagement = random.uniform(58, 98)
        nps = random.uniform(6.0, 10.0)
    elif status == "Churned":
        engagement = random.uniform(20, 44)
        nps = random.uniform(0.0, 3.8)
    else:
        engagement = random.uniform(35, 65)
        nps = random.uniform(3.0, 7.0)

    return {
        "name": name,
        "email": make_email(name, company, batch_num, index),
        "company": company,
        "industry": industry,
        "tier": tier,
        "job_title": random.choice(
            ["CTO", "VP Engineering", "Head of Operations", "Product Manager", "DevOps Lead"]
        ),
        "status": status,
        "engagement_score": round(engagement, 1),
        "nps_score": round(nps, 1),
        "signup_days_ago": random.randint(30, 365),
        "pain_point": f"Needs better {random.choice(['analytics', 'automation', 'integration', 'reporting'])} for {industry} workflows.",
    }


# ─── Customer batch generation (LLM, 10-20 per call) ───────────────────────────

def generate_customer_batch(batch_num, size):
    size = safe_int(size, 15, 1, 100)
    industry = random.choice(INDUSTRIES)
    tier = random.choice(TIERS)
    names = [f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}" for _ in range(size)]
    companies = [random.choice(COMPANIES) for _ in range(size)]

    prompt = f"""Generate exactly {size} realistic CRM customer profiles as JSON.

Use these names: {names}
Use these companies: {companies}
Industry: {industry}
Tier: {tier}

Each object must have:
name, email, company, industry, tier, job_title, status, engagement_score,
nps_score, signup_days_ago, pain_point.

Rules:
- status is one of Active, Inactive, Churned
- Active engagement_score > 55
- Churned engagement_score < 45 and nps_score < 4
- email must be unique, lowercase, and realistic
- Return only a JSON array of {size} objects, nothing else."""

    num_predict = min(2600, 110 * size + 150)
    data = parse_json_list(ollama(prompt, num_predict=num_predict, num_ctx=2048))

    normalized = []
    for index in range(size):
        fallback = fallback_customer(batch_num, index, industry, tier)
        item = data[index] if index < len(data) and isinstance(data[index], dict) else {}

        name = safe_str(item.get("name"), fallback["name"])
        company = safe_str(item.get("company"), fallback["company"])
        email = safe_str(item.get("email"), "")
        if "@" not in email:
            email = make_email(name, company, batch_num, index)

        status = safe_choice(item.get("status"), STATUSES, fallback["status"])
        engagement = safe_float(item.get("engagement_score"), fallback["engagement_score"], 20.0, 98.0)
        nps = safe_float(item.get("nps_score"), fallback["nps_score"], 0.0, 10.0)
        engagement, nps = normalize_customer_scores(status, engagement, nps)

        normalized.append({
            "name": name,
            "email": email.lower().strip(),
            "company": company,
            "industry": safe_choice(item.get("industry"), INDUSTRIES, industry),
            "tier": safe_choice(item.get("tier"), TIERS, tier),
            "job_title": safe_str(item.get("job_title"), fallback["job_title"]),
            "status": status,
            "engagement_score": engagement,
            "nps_score": nps,
            "signup_days_ago": safe_int(item.get("signup_days_ago"), fallback["signup_days_ago"], 30, 365),
            "pain_point": safe_str(item.get("pain_point"), fallback["pain_point"]),
        })

    return normalized


# ─── Ticket generation (LLM, 10-20 tickets across several customers per call) ──

def fallback_ticket_text(customer, category):
    company = safe_str(customer.get("company"), "the customer")
    industry = safe_str(customer.get("industry"), "their")
    tier = safe_str(customer.get("tier"), "current")
    name = safe_str(customer.get("name"), "The customer")
    templates = {
        "Billing": (
            "Invoice mismatch after renewal",
            f"{company} reports that the latest {tier} renewal invoice does not match their active seat count. The finance team needs a corrected invoice before month-end reconciliation.",
        ),
        "Technical": (
            "Dashboard timeout on analytics export",
            f"{company} is seeing timeout errors when exporting analytics for their {industry} workflow. The issue blocks their weekly operations review.",
        ),
        "Account": (
            "SSO login redirects to error page",
            f"{name} cannot complete SSO login for {company}. Users are redirected to an error page after identity-provider authentication.",
        ),
        "Onboarding": (
            "Legacy CRM import failing",
            f"{company} is migrating historical customer data, but the import fails during validation. Their onboarding timeline is at risk.",
        ),
        "Integration": (
            "Slack alerts not firing",
            f"{company} configured Slack alerts for support updates, but no messages are being delivered to their operations channel.",
        ),
    }
    return templates.get(category, templates["Technical"])


def finalize_ticket(data, customer_context):
    """Turn a (possibly empty/partial) parsed dict into a full ticket dict,
    filling any missing/invalid fields from deterministic fallbacks - same
    behaviour as the original single-call generate_ticket().

    TIMELINE CONSTRAINT: a ticket can never be older than the customer's
    account. days_ago_created is clamped to the customer's signup_days_ago
    (and to the original 180-day cap, whichever is smaller) - this applies
    both to the LLM-provided value and to the random fallback, so neither
    path can produce a ticket that predates signup.
    """
    data = data if isinstance(data, dict) else {}
    category_default = random.choice(CATEGORIES)
    priority_default = random.choices(PRIORITIES, weights=[18, 42, 30, 10], k=1)[0]

    ticket_category = safe_choice(data.get("category"), CATEGORIES, category_default)
    title, description = fallback_ticket_text(customer_context, ticket_category)
    status = safe_choice(data.get("status"), TICKET_STATUSES, "Open")

    if not data:
        is_resolved = random.random() > 0.4
        status = random.choice(["Resolved", "Closed"]) if is_resolved else random.choice(
            ["Open", "In Progress", "Escalated"]
        )

    # Cap ticket age at the customer's account age (signup_days_ago), and at
    # most 180 days, so no ticket can predate the customer signing up.
    signup_days_ago = safe_int(customer_context.get("signup_days_ago"), 180, 1, 3650)
    max_ticket_age = max(1, min(180, signup_days_ago))
    days_ago_created = safe_int(
        data.get("days_ago_created"),
        random.randint(1, max_ticket_age),
        1,
        max_ticket_age,
    )

    return {
        "title": safe_str(data.get("title"), title, max_length=255),
        "description": safe_str(data.get("description"), description),
        "category": ticket_category,
        "priority": safe_choice(data.get("priority"), PRIORITIES, priority_default),
        "status": status,
        "days_ago_created": days_ago_created,
        "resolution_note": (
            safe_str(
                data.get("resolution_note"),
                "Support identified the root cause and shared the corrective steps.",
            )
            if status in ["Resolved", "Closed"]
            else None
        ),
    }


def generate_ticket_batch(batch_num, group):
    """
    group: list of (customer, context_dict, tickets_needed) tuples.
    Makes ONE Ollama call requesting tickets for every customer in the group.
    Returns the raw parsed list (caller maps entries back via customer_index).
    """
    lines = []
    total = 0
    for i, (_customer, ctx, count) in enumerate(group):
        total += count
        signup_days_ago = safe_int(ctx.get("signup_days_ago"), 180, 1, 3650)
        max_ticket_age = max(1, min(180, signup_days_ago))
        lines.append(
            f"{i}) name={safe_str(ctx.get('name'), 'Customer')}; "
            f"company={safe_str(ctx.get('company'), 'Unknown')}; "
            f"industry={safe_str(ctx.get('industry'), 'Unknown')}; "
            f"tier={safe_str(ctx.get('tier'), 'Basic')}; "
            f"role={safe_str(ctx.get('job_title'), 'Manager')}; "
            f"pain_point={safe_str(ctx.get('pain_point'), 'platform issues')}; "
            f"customer_signed_up_days_ago={signup_days_ago}; "
            f"max_days_ago_created={max_ticket_age}; "
            f"tickets_needed={count}"
        )
    customers_block = "\n".join(lines)

    prompt = f"""Generate realistic CRM support tickets as a single JSON array.

Customers (customer_index) and how many tickets each needs:
{customers_block}

Generate exactly tickets_needed ticket objects for EACH customer_index above
(total {total} objects in the array).

Each object must have these fields:
customer_index, title, description, category, priority, status,
days_ago_created, resolution_note.

Rules:
- customer_index must be an integer matching one of the indices above
- category is one of {CATEGORIES}
- priority is one of {PRIORITIES} (vary them: mostly Low/Medium, some High, few Critical)
- status is one of {TICKET_STATUSES}
- description must reference that specific customer's company or pain point
- days_ago_created must be an integer between 1 and that customer's
  max_days_ago_created (never older than when the customer signed up)
- resolution_note is null unless status is Resolved or Closed
- Return ONLY the JSON array, no markdown, no extra text."""

    # NOTE: num_predict scales with batch size (~190 tokens/ticket). If you
    # raise TICKET_BATCH_MAX well beyond 20, watch for truncated responses -
    # the repair logic in parse_json_list() helps, but very large batches may
    # need a higher formula constant or should be split into smaller batches.
    num_predict = min(3800, 190 * total + 200)
    num_ctx = 2048 if total <= 10 else 4096
    raw = ollama(prompt, num_predict=num_predict, num_ctx=num_ctx)
    return parse_json_list(raw), total


def process_ticket_batch(batch_num, group):
    """Runs one LLM batch call and reconciles results against tickets_needed
    per customer, back-filling with the deterministic fallback for any
    customer_index the model skipped, duplicated wrongly, or got wrong."""
    data, total = generate_ticket_batch(batch_num, group)

    used = [0] * len(group)
    results = []

    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            ci = int(entry.get("customer_index"))
        except (TypeError, ValueError):
            continue
        if ci < 0 or ci >= len(group):
            continue
        if used[ci] >= group[ci][2]:
            continue
        used[ci] += 1
        _customer, ctx, _count = group[ci]
        results.append((ci, finalize_ticket(entry, ctx)))

    for ci, (_customer, ctx, need) in enumerate(group):
        missing = need - used[ci]
        for _ in range(missing):
            results.append((ci, finalize_ticket({}, ctx)))

    return group, results


def ticket_datetimes(days_ago_created, status):
    now = datetime.now()
    days_ago = safe_int(days_ago_created, 7, 1, 180)
    created_at = now - timedelta(
        days=days_ago,
        hours=random.randint(0, 12),
        minutes=random.randint(0, 59),
    )

    if status in ["Resolved", "Closed"]:
        latest_resolution = min(now, created_at + timedelta(days=7))
        available_seconds = max(60, int((latest_resolution - created_at).total_seconds()))
        resolved_at = created_at + timedelta(seconds=random.randint(60, available_seconds))
        updated_at = resolved_at
    else:
        latest_update = max(created_at + timedelta(minutes=1), now - timedelta(minutes=1))
        available_seconds = max(60, int((latest_update - created_at).total_seconds()))
        updated_at = created_at + timedelta(seconds=random.randint(60, available_seconds))
        resolved_at = None

    return created_at, updated_at, resolved_at


# ─── Interaction generation (LLM, 10-20 across several tickets per call) ──────

# ─── Diverse interaction message templates ─────────────────────────────────
# The interactions table has no explicit "sender role" column, so realism
# comes purely from message content/voice. Rather than one fixed sentence
# per sentiment bucket (the previous behaviour - highly repetitive across
# 2,000 rows), we keep separate CUSTOMER- and AGENT-voiced template pools,
# grouped by conversational stage, and alternate between them per ticket so
# a ticket's interaction thread reads like an actual back-and-forth:
#   customer initial issue -> agent acknowledgement -> agent investigation/
#   troubleshooting -> (customer follow-up/escalation if unresolved) ->
#   agent resolution -> customer appreciation/follow-up.
#
# DATA CONSISTENCY (see also finalize_interaction / _choose_interaction_type
# below): "complaint"/"escalation" customer templates are only selectable
# while the ticket is still open/in-progress/escalated - a Resolved/Closed
# ticket won't spontaneously receive a new complaint, matching the "no new
# complaints on resolved tickets unless reopened" requirement.

CUSTOMER_TEMPLATES = {
    "initial_issue": [
        "We're running into an issue with '{title}' - {pain_point} Can someone take a look?",
        "Opening this because '{title}' is blocking our team at {company}. Please advise on next steps.",
        "Hi, {name} here from {company}. We just hit a problem related to '{title}' and need help resolving it.",
        "Flagging '{title}' - it started affecting our {industry} workflow this week.",
    ],
    "follow_up": [
        "Any update on '{title}'? It's been a few days and our team at {company} is waiting on this.",
        "Following up on '{title}' - just checking whether there's progress on our end.",
        "Wanted to check in on '{title}'. Let us know if you need anything further from us.",
        "Circling back on '{title}' - still open on our side, appreciate a status update.",
    ],
    "escalation": [
        "This is becoming urgent - '{title}' still isn't resolved and it's now impacting {company}'s operations.",
        "We need to escalate '{title}'. This has gone on longer than expected and leadership is asking questions.",
        "Please prioritize '{title}' - the delay is starting to affect our {industry} rollout timeline.",
        "Escalating '{title}' again - we were told this would be handled sooner.",
    ],
    "appreciation": [
        "Thanks for resolving '{title}' so quickly - really appreciate the support.",
        "Great turnaround on '{title}'. The {company} team is happy with the outcome.",
        "Wanted to say thanks for the help with '{title}' - smooth experience overall.",
        "Appreciate the quick fix on '{title}'. That was exactly what we needed.",
    ],
    "complaint": [
        "Not satisfied with how '{title}' was handled - this took far too long to sort out.",
        "We're frustrated with the delay on '{title}'. This isn't the experience we expected from {tier} support.",
        "'{title}' has been a repeated pain point for {company} and it doesn't feel like it's improving.",
        "Disappointed with the resolution on '{title}' - we may need to reconsider our plan here.",
    ],
}

AGENT_TEMPLATES = {
    "acknowledgement": [
        "Thanks for reaching out about '{title}' - we've logged this and are looking into it now.",
        "Got it - opening an investigation into '{title}' for {company} right away.",
        "Appreciate the report on '{title}'. Assigning this to our team now.",
        "Thanks for flagging '{title}'. We'll keep you posted as we dig in.",
    ],
    "investigation": [
        "We're currently investigating '{title}' and checking recent logs/config on your account.",
        "Looking into the root cause of '{title}' now - will share findings shortly.",
        "Our team is reproducing the '{title}' issue on our end to narrow down the cause.",
        "Digging into '{title}' - initial checks are underway.",
    ],
    "troubleshooting": [
        "As a next step for '{title}', could you confirm the exact steps that trigger the issue?",
        "We've made an initial change related to '{title}' - can you verify if the behavior has changed?",
        "Trying a fix for '{title}' on our side now - we'll confirm once it's deployed.",
        "For '{title}', we've identified a likely cause and are testing a resolution.",
    ],
    "escalation": [
        "We understand the urgency on '{title}' and have escalated this internally for faster resolution.",
        "Escalating '{title}' to our senior team given the impact on {company}.",
        "This has been marked high priority - '{title}' is now with our escalation team.",
        "Apologies for the delay on '{title}' - we've escalated and prioritized it.",
    ],
    "resolution": [
        "'{title}' has been resolved - the fix is live, please confirm on your end when you can.",
        "We've closed out '{title}'. Let us know if anything still looks off.",
        "Resolution deployed for '{title}' - thanks for your patience while we worked through it.",
        "'{title}' is now fixed. Reach out again if you see any recurrence.",
    ],
    "follow_up": [
        "Just checking back on '{title}' - is everything still working as expected?",
        "Following up after resolving '{title}' - happy to help if anything else comes up.",
        "Wanted to confirm '{title}' is still resolved on your end - let us know either way.",
        "Quick check-in on '{title}' post-resolution - all good so far?",
    ],
}


def _choose_interaction_type(customer_status, ticket_status, interaction_index):
    """
    Picks (role, category) for the interaction at `interaction_index` within
    a ticket's thread, alternating customer/agent voice and keeping the
    category consistent with ticket status (see module docstring above).

    - index 0 is always the customer's initial issue.
    - index 1 is always the agent's acknowledgement.
    - subsequent indices alternate, using status to bias category choice:
        - Escalated tickets bias toward escalation-flavoured messages.
        - Resolved/Closed tickets bias the final agent turn toward
          "resolution" and the final customer turn toward "appreciation"
          (or a neutral "follow_up" for already-Churned customers) rather
          than a fresh complaint - resolved tickets don't get new
          complaints unless genuinely reopened (a distinct ticket status).
        - Otherwise, open/in-progress tickets rotate through
          investigation/troubleshooting/follow_up.
    """
    is_customer_turn = (interaction_index % 2 == 0)

    if interaction_index == 0:
        return ("customer", "initial_issue")
    if interaction_index == 1:
        return ("agent", "acknowledgement")

    if ticket_status == "Escalated":
        return ("customer", "escalation") if is_customer_turn else ("agent", "escalation")

    if ticket_status in ("Resolved", "Closed"):
        if is_customer_turn:
            category = "follow_up" if customer_status == "Churned" else "appreciation"
            return ("customer", category)
        return ("agent", "resolution")

    # Open / In Progress: no resolution/appreciation yet, no unearned
    # complaints either - keep it to the natural mid-conversation beats.
    if is_customer_turn:
        category = "escalation" if customer_status == "Churned" else "follow_up"
        return ("customer", category)
    return ("agent", random.choice(["investigation", "troubleshooting", "follow_up"]))


def _render_template(pool, category, customer, ticket):
    template = random.choice(pool[category])
    return template.format(
        title=safe_str(ticket.get("title"), "the support request"),
        company=safe_str(customer.get("company"), "your company"),
        name=safe_str(customer.get("name"), "there"),
        industry=safe_str(customer.get("industry"), "your"),
        tier=safe_str(customer.get("tier"), "current"),
        pain_point=safe_str(customer.get("pain_point"), "a recurring issue"),
    )


def fallback_interaction(customer, ticket, max_days_ago, interaction_index=0):
    status = safe_choice(customer.get("status"), STATUSES, "Active")
    ticket_status = safe_choice(ticket.get("status"), TICKET_STATUSES, "Open")
    max_days_ago = safe_int(max_days_ago, 0, 0, 180)

    role, category = _choose_interaction_type(status, ticket_status, interaction_index)
    pool = CUSTOMER_TEMPLATES if role == "customer" else AGENT_TEMPLATES
    message = _render_template(pool, category, customer, ticket)

    label_by_category = {
        "initial_issue": "neutral", "follow_up": "neutral",
        "escalation": "frustrated", "appreciation": "satisfied",
        "complaint": "frustrated", "acknowledgement": "neutral",
        "investigation": "neutral", "troubleshooting": "neutral",
        "resolution": "positive",
    }
    label = label_by_category.get(category, "neutral")

    if role == "customer" and status == "Churned" and category in ("initial_issue", "follow_up"):
        label = "negative"

    if label in ("positive", "satisfied"):
        csat = round(random.uniform(3.6, 5.0), 1)
    elif label == "neutral":
        csat = round(random.uniform(2.8, 4.0), 1)
    else:
        csat = round(random.uniform(1.0, 2.8), 1)

    return {
        "channel": random.choice(CHANNELS),
        "message": message,
        "sentiment_label": label,
        "csat_score": csat,
        "days_ago": random.randint(0, max_days_ago) if max_days_ago > 0 else 0,
    }


def sentiment_from_label(label, customer_status):
    label = safe_str(label, "neutral").lower()
    customer_status = safe_choice(customer_status, STATUSES, "Active")

    if customer_status == "Churned" and label in ["positive", "satisfied"]:
        label = "frustrated"

    ranges = {
        "positive": (0.4, 0.9),
        "satisfied": (0.6, 1.0),
        "neutral": (-0.15, 0.25),
        "negative": (-0.75, -0.25),
        "frustrated": (-1.0, -0.55),
    }
    low, high = ranges.get(label, ranges["neutral"])
    return round(random.uniform(low, high), 3)


def interaction_timestamp_after(ticket_created_at):
    now = datetime.now()
    if not isinstance(ticket_created_at, datetime):
        ticket_created_at = now - timedelta(days=1)

    earliest = ticket_created_at + timedelta(minutes=1)
    if earliest >= now:
        return now

    seconds_available = max(1, int((now - earliest).total_seconds()))
    return earliest + timedelta(seconds=random.randint(1, seconds_available))


def finalize_interaction(data, customer_context, ticket_context, max_days_ago, interaction_index=0):
    data = data if isinstance(data, dict) else {}
    status = safe_choice(customer_context.get("status"), STATUSES, "Active")
    if not data:
        data = fallback_interaction(customer_context, ticket_context, max_days_ago, interaction_index)

    label = safe_str(data.get("sentiment_label"), "neutral").lower()
    sentiment = sentiment_from_label(label, status)
    csat = data.get("csat_score")
    if csat is not None:
        csat = safe_float(csat, 3.0, 1.0, 5.0)

    return {
        "channel": safe_choice(data.get("channel"), CHANNELS, random.choice(CHANNELS)),
        "message": safe_str(
            data.get("message"),
            f"Following up on {safe_str(ticket_context.get('title'), 'the support request')}.",
        ),
        "sentiment": sentiment,
        "csat_score": csat,
        "days_ago": safe_int(data.get("days_ago"), 0, 0, max_days_ago),
    }


def generate_interaction_batch(batch_num, group):
    """
    group: list of (ticket_ctx, customer_ctx, interactions_needed) tuples.
    Makes ONE Ollama call requesting interactions for every ticket in the group.
    """
    lines = []
    max_days_list = []
    total = 0
    for i, (ticket_ctx, cust_ctx, count) in enumerate(group):
        ticket_days_ago = safe_int(ticket_ctx.get("days_ago_created"), 7, 1, 180)
        max_days = max(0, ticket_days_ago - 1)
        max_days_list.append(max_days)
        total += count
        lines.append(
            f"{i}) customer={safe_str(cust_ctx.get('name'), 'Customer')}; "
            f"company={safe_str(cust_ctx.get('company'), 'Unknown')}; "
            f"status={safe_str(cust_ctx.get('status'), 'Active')}; "
            f"ticket_title={safe_str(ticket_ctx.get('title'), 'Support request')}; "
            f"ticket_created_days_ago={ticket_days_ago}; "
            f"max_days_ago={max_days}; "
            f"interactions_needed={count}"
        )
    tickets_block = "\n".join(lines)

    prompt = f"""Generate realistic CRM customer interactions as a single JSON array.

Tickets (ticket_index) and how many interactions each needs:
{tickets_block}

Generate exactly interactions_needed interaction objects for EACH ticket_index
above (total {total} objects in the array).

Each object must have these fields:
ticket_index, channel, message, sentiment_label, csat_score, days_ago.

Rules:
- ticket_index must be an integer matching one of the indices above
- channel is one of {CHANNELS}
- sentiment_label is one of positive, neutral, negative, frustrated, satisfied
- days_ago must be between 0 and that ticket's max_days_ago (so the
  interaction happens after the ticket was created)
- Churned customers should sound frustrated or negative
- Active customers can be neutral, positive, or satisfied
- Vary sentence structure and vocabulary across interactions - avoid
  reusing the same stock phrases for every ticket
- Where a ticket has multiple interactions, model a natural back-and-forth
  (customer describes the issue -> agent acknowledges/investigates ->
  agent resolution or customer follow-up/appreciation), not several
  near-duplicate messages
- Return ONLY the JSON array, no markdown, no extra text."""

    num_predict = min(3800, 140 * total + 200)
    num_ctx = 2048 if total <= 10 else 4096
    raw = ollama(prompt, num_predict=num_predict, num_ctx=num_ctx)
    return parse_json_list(raw), max_days_list


def process_interaction_batch(batch_num, group):
    data, max_days_list = generate_interaction_batch(batch_num, group)

    used = [0] * len(group)
    results = []

    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            ti = int(entry.get("ticket_index"))
        except (TypeError, ValueError):
            continue
        if ti < 0 or ti >= len(group):
            continue
        if used[ti] >= group[ti][2]:
            continue
        idx = used[ti]  # position of this interaction within its ticket's thread
        used[ti] += 1
        ticket_ctx, cust_ctx, _count = group[ti]
        results.append((ti, finalize_interaction(entry, cust_ctx, ticket_ctx, max_days_list[ti], idx)))

    for ti, (ticket_ctx, cust_ctx, need) in enumerate(group):
        missing = need - used[ti]
        for offset in range(missing):
            idx = used[ti] + offset
            results.append((ti, finalize_interaction({}, cust_ctx, ticket_ctx, max_days_list[ti], idx)))

    return group, results


def context_from_customer(customer):
    email = safe_str(customer.email, "customer@example.com")
    company = safe_str(customer.company, "Unknown Company")
    signup_days_ago = 30
    if customer.signup_date:
        signup_days_ago = max(30, (datetime.now().date() - customer.signup_date).days)

    return {
        "name": safe_str(customer.name, "Customer"),
        "email": email,
        "company": company,
        "industry": safe_choice(customer.industry, INDUSTRIES, "AI/ML SaaS"),
        "tier": safe_choice(customer.tier, TIERS, "Basic"),
        "job_title": "Manager",
        "status": safe_choice(customer.status, STATUSES, "Active"),
        "engagement_score": safe_float(customer.engagement_score, 60.0, 20.0, 98.0),
        "nps_score": safe_float(customer.nps_score, 5.0, 0.0, 10.0),
        "signup_days_ago": signup_days_ago,
        "pain_point": f"Needs better support for {safe_str(customer.industry, 'their')} workflows.",
    }


# ─── Allocation / batching helpers ─────────────────────────────────────────────

def allocate_counts(n_items, total, min_each=0, max_each=None):
    """
    Distribute `total` units across n_items (e.g. tickets per customer),
    respecting min/max per item, and GUARANTEED to sum to exactly `total`
    whenever that's mathematically possible (i.e. total <= n_items * max_each).

    Previous implementation used a bounded random.choice() retry loop, which
    could exhaust its guard counter and under-allocate whenever repeated
    random draws kept landing on items already at max_each - silently
    producing fewer tickets/interactions than requested. This version uses a
    randomized round-robin pass for natural variance, followed by a
    deterministic fill pass that mops up any remainder, so it can never fall
    short of `total` when capacity allows it.
    """
    if n_items <= 0:
        return []

    counts = [min_each] * n_items
    remaining = total - min_each * n_items
    if remaining < 0:
        raise ValueError(f"total ({total}) is smaller than n_items*min_each ({n_items * min_each})")

    indices = list(range(n_items))

    # Randomized round-robin pass: gives natural variance in the distribution.
    pass_count = 0
    while remaining > 0 and pass_count < 20:
        random.shuffle(indices)
        progressed = False
        for i in indices:
            if remaining <= 0:
                break
            if max_each is None or counts[i] < max_each:
                counts[i] += 1
                remaining -= 1
                progressed = True
        pass_count += 1
        if not progressed:
            break

    # Deterministic guarantee pass: mops up any leftover so we ALWAYS hit
    # exactly `total`, regardless of how the randomized pass landed.
    if remaining > 0:
        for i in indices:
            while remaining > 0 and (max_each is None or counts[i] < max_each):
                counts[i] += 1
                remaining -= 1
            if remaining <= 0:
                break

    if remaining > 0:
        raise ValueError(
            f"Cannot allocate {total} across {n_items} items with max_each={max_each} "
            f"(max possible is {n_items * max_each if max_each is not None else 'unbounded'})."
        )

    return counts


def build_batches(items_with_counts, target_min, target_max):
    """
    Groups (item, count) pairs sequentially into batches whose count sums land
    in roughly [target_min, target_max]. Only the final leftover batch may be
    smaller than target_min.
    """
    batches = []
    current = []
    current_sum = 0
    for item, count in items_with_counts:
        if count <= 0:
            continue
        # If adding this item would blow past target_max, close out the
        # current batch first (as long as it already has something in it).
        if current and current_sum + count > target_max:
            batches.append(current)
            current = []
            current_sum = 0
        current.append((item, count))
        current_sum += count
        if current_sum >= target_min:
            batches.append(current)
            current = []
            current_sum = 0
    if current:
        batches.append(current)
    return batches


def run_parallel(fn, jobs, label="batch"):
    """
    jobs: list of positional-arg tuples. Returns results in submission order.

    Fault tolerant: if a single batch raises an exception, it is printed and
    that batch's slot is set to None instead of the whole run being aborted.
    Callers must treat a None entry as "this batch produced nothing" and
    continue (the deterministic fallback generators still cover the affected
    records at the per-item level within batches that DID succeed; a batch
    that fails outright simply contributes fewer records, which is preferable
    to crashing the entire generation run).

    Progress reporting: as each batch completes (in completion order), prints
    the current batch count, remaining batches, and percent complete.
    """
    results = [None] * len(jobs)
    total = len(jobs)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fn, *job): idx for idx, job in enumerate(jobs)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                print(f"    [{label}] batch {idx} failed, skipping it and continuing: {exc}")
                results[idx] = None

            completed += 1
            remaining = total - completed
            pct = (completed / total * 100) if total else 100.0
            print(f"    [{label}] batch {completed}/{total} complete "
                  f"(remaining: {remaining}, {pct:.1f}% done)")

    return results


def make_email_dedup(db):
    """In-memory email de-duplication so we don't issue a SELECT per customer."""
    existing = {row[0] for row in db.query(models.Customer.email).all()}

    def dedup(email, batch_num, index):
        email = safe_str(email, "").lower()
        if "@" not in email:
            email = make_email("customer", "example", batch_num, index)
        original = email
        suffix = 0
        while email in existing:
            suffix += 1
            local, domain = original.split("@", 1)
            email = f"{local}.{batch_num}.{index}.{suffix}@{domain}"
        existing.add(email)
        return email

    return dedup


def validate_dataset(db):
    """
    Post-generation validation. Checks record counts, referential integrity,
    and timeline consistency (signup_date <= ticket.created_at <=
    interaction.timestamp <= today). Prints a report and returns True/False.
    Does not raise - a failed validation is reported, not fatal, so you can
    inspect the data afterwards.
    """
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    failures = []
    today = datetime.now().date()

    # ── Record counts ───────────────────────────────────────────────────────
    customer_count = db.query(models.Customer).count()
    ticket_count = db.query(models.Ticket).count()
    interaction_count = db.query(models.Interaction).count()

    print(f"Customers:    {customer_count} (expected {TARGET_CUSTOMERS})")
    print(f"Tickets:      {ticket_count} (expected {TARGET_TICKETS})")
    print(f"Interactions: {interaction_count} (expected {TARGET_INTERACTIONS})")

    if customer_count != TARGET_CUSTOMERS:
        failures.append(f"Customer count mismatch: got {customer_count}, expected {TARGET_CUSTOMERS}")
    if ticket_count != TARGET_TICKETS:
        failures.append(f"Ticket count mismatch: got {ticket_count}, expected {TARGET_TICKETS}")
    if interaction_count != TARGET_INTERACTIONS:
        failures.append(f"Interaction count mismatch: got {interaction_count}, expected {TARGET_INTERACTIONS}")

    # ── Referential integrity ───────────────────────────────────────────────
    customer_ids = {row.id for row in db.query(models.Customer.id).all()}
    ticket_rows = db.query(
        models.Ticket.id, models.Ticket.customer_id, models.Ticket.created_at
    ).all()
    ticket_ids = {row.id for row in ticket_rows}
    ticket_created_by_id = {row.id: row.created_at for row in ticket_rows}

    orphan_tickets = [row.id for row in ticket_rows if row.customer_id not in customer_ids]
    if orphan_tickets:
        shown = orphan_tickets[:10]
        suffix = "..." if len(orphan_tickets) > 10 else ""
        failures.append(f"{len(orphan_tickets)} ticket(s) reference a non-existent customer: {shown}{suffix}")
    else:
        print("All tickets reference an existing customer: OK")

    interaction_rows = db.query(
        models.Interaction.id,
        models.Interaction.customer_id,
        models.Interaction.ticket_id,
        models.Interaction.timestamp,
    ).all()

    orphan_int_customer = [row.id for row in interaction_rows if row.customer_id not in customer_ids]
    orphan_int_ticket = [row.id for row in interaction_rows if row.ticket_id not in ticket_ids]

    if orphan_int_customer:
        shown = orphan_int_customer[:10]
        suffix = "..." if len(orphan_int_customer) > 10 else ""
        failures.append(f"{len(orphan_int_customer)} interaction(s) reference a non-existent customer: {shown}{suffix}")
    else:
        print("All interactions reference an existing customer: OK")

    if orphan_int_ticket:
        shown = orphan_int_ticket[:10]
        suffix = "..." if len(orphan_int_ticket) > 10 else ""
        failures.append(f"{len(orphan_int_ticket)} interaction(s) reference a non-existent ticket: {shown}{suffix}")
    else:
        print("All interactions reference an existing ticket: OK")

    # ── Timeline consistency ────────────────────────────────────────────────
    customer_signup = {row.id: row.signup_date for row in db.query(models.Customer.id, models.Customer.signup_date).all()}

    signup_violations = []
    for row in ticket_rows:
        signup = customer_signup.get(row.customer_id)
        if signup is not None and row.created_at is not None and row.created_at.date() < signup:
            signup_violations.append(row.id)

    if signup_violations:
        shown = signup_violations[:10]
        suffix = "..." if len(signup_violations) > 10 else ""
        failures.append(f"{len(signup_violations)} ticket(s) created before customer signup: {shown}{suffix}")
    else:
        print("All tickets created on/after customer signup: OK")

    interaction_before_ticket = []
    interaction_after_today = []
    for row in interaction_rows:
        ticket_created = ticket_created_by_id.get(row.ticket_id)
        if ticket_created is not None and row.timestamp is not None and row.timestamp < ticket_created:
            interaction_before_ticket.append(row.id)
        if row.timestamp is not None and row.timestamp.date() > today:
            interaction_after_today.append(row.id)

    if interaction_before_ticket:
        shown = interaction_before_ticket[:10]
        suffix = "..." if len(interaction_before_ticket) > 10 else ""
        failures.append(f"{len(interaction_before_ticket)} interaction(s) occurred before their ticket was created: {shown}{suffix}")
    else:
        print("All interactions occur on/after their ticket's creation: OK")

    if interaction_after_today:
        shown = interaction_after_today[:10]
        suffix = "..." if len(interaction_after_today) > 10 else ""
        failures.append(f"{len(interaction_after_today)} interaction(s) timestamped in the future: {shown}{suffix}")
    else:
        print("No interactions timestamped in the future: OK")

    print("=" * 60)
    if failures:
        print(f"VALIDATION FAILED ({len(failures)} issue(s)):")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("VALIDATION PASSED: dataset is fully consistent.")
    print("=" * 60 + "\n")

    return len(failures) == 0


# ─── Main orchestration ─────────────────────────────────────────────────────────

def main():
    db = SessionLocal()
    customer_context_by_id = {}
    ticket_context_by_customer_id = {}
    run_start = time.time()

    try:
        print(f"\nE-Cell CRM relational dataset generator ({MODEL})")
        print(f"Workers: {MAX_WORKERS}\n")

        # ── Phase 1: customers ──────────────────────────────────────────────
        phase1_start = time.time()
        print("Phase 1: generating customers (batched + parallel)")

        remaining = TARGET_CUSTOMERS
        batch_jobs = []
        bn = 0
        while remaining > 0:
            size = min(random.randint(CUSTOMER_BATCH_MIN, CUSTOMER_BATCH_MAX), remaining)
            batch_jobs.append((bn, size))
            remaining -= size
            bn += 1

        print(f"  {len(batch_jobs)} Ollama calls for {TARGET_CUSTOMERS} customers "
              f"(was {TARGET_CUSTOMERS // 10} calls of 10 before)")

        batch_results = run_parallel(generate_customer_batch, batch_jobs, label="customers")
        all_customer_items = [item for batch in batch_results for item in (batch or [])]

        dedup_email = make_email_dedup(db)

        for chunk_start in range(0, len(all_customer_items), DB_CHUNK_SIZE):
            chunk = all_customer_items[chunk_start:chunk_start + DB_CHUNK_SIZE]
            rows_with_items = []
            for offset, item in enumerate(chunk):
                global_index = chunk_start + offset
                email = dedup_email(item.get("email"), 0, global_index)
                item["email"] = email

                signup_date = datetime.now().date() - timedelta(
                    days=safe_int(item.get("signup_days_ago"), 30, 30, 365)
                )
                row = models.Customer(
                    name=safe_str(item.get("name"), "Customer", max_length=255),
                    email=email,
                    company=item["company"],
                    industry=safe_choice(item.get("industry"), INDUSTRIES, "AI/ML SaaS"),
                    tier=safe_choice(item.get("tier"), TIERS, "Basic"),
                    signup_date=signup_date,
                    engagement_score=safe_float(item.get("engagement_score"), 60.0, 20.0, 98.0),
                    status=safe_choice(item.get("status"), STATUSES, "Active"),
                    nps_score=safe_float(item.get("nps_score"), 5.0, 0.0, 10.0),
                    last_interaction_date=datetime.now() - timedelta(days=random.randint(1, 180)),
                )
                row.engagement_score, row.nps_score = normalize_customer_scores(
                    row.status, row.engagement_score, row.nps_score
                )
                rows_with_items.append((row, item))

            db.add_all([row for row, _ in rows_with_items])
            db.flush()  # single flush per chunk assigns IDs to every row in it
            for row, item in rows_with_items:
                customer_context_by_id[row.id] = item
            print(f"  customers: {min(chunk_start + DB_CHUNK_SIZE, len(all_customer_items))}/{TARGET_CUSTOMERS}")
        db.commit()  # one commit for the whole phase - flush() already gave us IDs as we went

        customers = (
            db.query(models.Customer)
            .order_by(models.Customer.id.desc())
            .limit(TARGET_CUSTOMERS)
            .all()
        )
        customers = list(reversed(customers))
        if not customers:
            raise RuntimeError("No customers available for ticket generation.")

        for customer in customers:
            customer_context_by_id.setdefault(customer.id, context_from_customer(customer))

        print(f"Phase 1 completed in {time.time() - phase1_start:.2f} seconds")

        # ── Phase 2: tickets ────────────────────────────────────────────────
        phase2_start = time.time()
        print("\nPhase 2: generating tickets (batched + parallel)")

        ticket_counts = allocate_counts(
            len(customers), TARGET_TICKETS, min_each=0, max_each=TICKETS_PER_CUSTOMER_MAX
        )
        customer_count_pairs = list(zip(customers, ticket_counts))
        random.shuffle(customer_count_pairs)

        ticket_batches = build_batches(customer_count_pairs, TICKET_BATCH_MIN, TICKET_BATCH_MAX)
        print(f"  {len(ticket_batches)} Ollama calls for {TARGET_TICKETS} tickets "
              f"(was {TARGET_TICKETS} calls of 1 before)")

        ticket_jobs = []
        for bn, batch in enumerate(ticket_batches):
            group = [
                (customer, customer_context_by_id[customer.id], count)
                for customer, count in batch
            ]
            ticket_jobs.append((bn, group))

        batch_outputs = run_parallel(process_ticket_batch, ticket_jobs, label="tickets")

        ticket_rows_flat = []  # (customer, ticket_dict)

        for batch_index, output in enumerate(batch_outputs):

            if output is None:
                print(f"    Ticket batch {batch_index} failed. Using deterministic fallback.")

                group = ticket_jobs[batch_index][1]

                for customer, ctx, count in group:
                    for _ in range(count):
                        ticket = finalize_ticket({}, ctx)
                        ticket_rows_flat.append((customer, ticket))

                continue

            group, results = output

            for ci, ticket_dict in results:
                customer = group[ci][0]
                ticket_rows_flat.append((customer, ticket_dict))

        total_tickets_created = 0
        for chunk_start in range(0, len(ticket_rows_flat), DB_CHUNK_SIZE):
            chunk = ticket_rows_flat[chunk_start:chunk_start + DB_CHUNK_SIZE]
            rows_with_meta = []
            for customer, ticket in chunk:
                created_at, updated_at, resolved_at = ticket_datetimes(
                    ticket["days_ago_created"], ticket["status"]
                )
                row = models.Ticket(
                    customer_id=customer.id,
                    title=ticket["title"],
                    description=ticket["description"],
                    category=ticket["category"],
                    priority=ticket["priority"],
                    status=ticket["status"],
                    assigned_agent=random.choice(AGENTS),
                    created_at=created_at,
                    updated_at=updated_at,
                    resolved_at=resolved_at,
                )
                rows_with_meta.append((row, customer, ticket, created_at))

            db.add_all([row for row, _, _, _ in rows_with_meta])
            db.flush()  # assigns IDs for the whole chunk in one round trip
            for row, customer, ticket, created_at in rows_with_meta:
                ticket_context = {
                    **ticket,
                    "ticket_id": row.id,
                    "customer_id": customer.id,
                    "created_at": created_at,
                }
                ticket_context_by_customer_id.setdefault(customer.id, []).append(ticket_context)
            total_tickets_created += len(chunk)
            print(f"  tickets: {total_tickets_created}/{len(ticket_rows_flat)}")
        db.commit()  # one commit for the whole phase

        print(f"Phase 2 completed in {time.time() - phase2_start:.2f} seconds")

        # ── Phase 3: interactions ───────────────────────────────────────────
        phase3_start = time.time()
        print("\nPhase 3: generating interactions (batched + parallel)")

        all_tickets_flat = [
            (customer_id, ticket_ctx)
            for customer_id, tickets in ticket_context_by_customer_id.items()
            for ticket_ctx in tickets
        ]
        if not all_tickets_flat:
            raise RuntimeError("No tickets available for interaction generation.")

        interaction_counts = allocate_counts(
            len(all_tickets_flat), TARGET_INTERACTIONS, min_each=0, max_each=INTERACTIONS_PER_TICKET_MAX
        )
        ticket_count_pairs = list(zip(all_tickets_flat, interaction_counts))
        random.shuffle(ticket_count_pairs)

        interaction_batches = build_batches(ticket_count_pairs, INTERACTION_BATCH_MIN, INTERACTION_BATCH_MAX)
        print(f"  {len(interaction_batches)} Ollama calls for {TARGET_INTERACTIONS} interactions "
              f"(was {TARGET_INTERACTIONS} calls of 1 before)")

        interaction_jobs = []
        for bn, batch in enumerate(interaction_batches):
            group = [
                (ticket_ctx, customer_context_by_id[customer_id], count)
                for (customer_id, ticket_ctx), count in batch
            ]
            interaction_jobs.append((bn, group))

        batch_outputs = run_parallel(process_interaction_batch, interaction_jobs, label="interactions")

        interaction_rows_flat = []  # (customer_id, ticket_ctx, interaction_dict)
        for batch_index, output in enumerate(batch_outputs):

            if output is None:
                print(f"    Interaction batch {batch_index} failed. Using deterministic fallback.")

                group = interaction_jobs[batch_index][1]

                for ticket_ctx, cust_ctx, count in group:
                    max_days = max(
                        0,
                        safe_int(ticket_ctx.get("days_ago_created"), 7, 1, 180) - 1,
                    )

                    for idx in range(count):
                        interaction = finalize_interaction(
                            {},
                            cust_ctx,
                            ticket_ctx,
                            max_days,
                            idx,
                        )

                        interaction_rows_flat.append(
                            (
                                ticket_ctx["customer_id"],
                                ticket_ctx,
                                interaction,
                            )
                        )

                continue

            group, results = output

            for ti, interaction_dict in results:
                ticket_ctx, cust_ctx, _count = group[ti]

                interaction_rows_flat.append(
                    (
                        ticket_ctx["customer_id"],
                        ticket_ctx,
                        interaction_dict,
                    )
                )
        total_interactions_created = 0
        for chunk_start in range(0, len(interaction_rows_flat), DB_CHUNK_SIZE):
            chunk = interaction_rows_flat[chunk_start:chunk_start + DB_CHUNK_SIZE]
            rows = []
            for customer_id, ticket_ctx, interaction in chunk:
                row = models.Interaction(
                    customer_id=customer_id,
                    ticket_id=ticket_ctx["ticket_id"],
                    channel=interaction["channel"],
                    message=interaction["message"],
                    sentiment=interaction["sentiment"],
                    csat_score=interaction["csat_score"],
                    timestamp=interaction_timestamp_after(ticket_ctx.get("created_at")),
                )
                rows.append(row)

            db.add_all(rows)
            db.flush()
            total_interactions_created += len(chunk)
            print(f"  interactions: {total_interactions_created}/{len(interaction_rows_flat)}")
        db.commit()  # one commit for the whole phase

        print(f"Phase 3 completed in {time.time() - phase3_start:.2f} seconds")

        # ── Final pass: last_interaction_date per customer ──────────────────
        # Single aggregate query instead of one SELECT per customer (500 -> 1).
        latest_by_customer = dict(
            db.query(models.Interaction.customer_id, func.max(models.Interaction.timestamp))
            .group_by(models.Interaction.customer_id)
            .all()
        )
        for customer in customers:
            latest = latest_by_customer.get(customer.id)
            if latest:
                customer.last_interaction_date = latest
        db.commit()

        print("\nDataset generation complete")
        print(f"  Customers:    {db.query(models.Customer).count()}")
        print(f"  Tickets:      {db.query(models.Ticket).count()}")
        print(f"  Interactions: {db.query(models.Interaction).count()}")
        print(f"  Time span:    {(datetime.now() - timedelta(days=180)).date()} to {datetime.now().date()}")

        validate_dataset(db)

        print(f"Total runtime: {time.time() - run_start:.2f} seconds")

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()