# E-Cell AI CRM Platform

This project is an AI-powered CRM platform built for the E-Cell Task 3 submission. It manages customers, support tickets, customer interactions, AI-assisted support responses, cohort analysis, HEART metrics, resolution analytics, churn scoring, and Streamlit dashboards.

The project uses:

- FastAPI for the backend API
- SQLAlchemy with MySQL for persistence
- Ollama/Llama3 for local LLM responses
- LangGraph for deterministic ticket routing workflow
- Streamlit and Plotly for the dashboard
- Rule-based analytics for churn, retention, resolution, and evaluation metrics

The system is designed as a practical demo CRM: the database stores customers, tickets, and interactions; the backend computes metrics from those records; the dashboard visualizes those metrics.

## Project Structure

```text
task3/
│
├── api/
│   └── app.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── customers.csv
│   ├── tickets.csv
│   ├── interactions.csv
│   └── generate_data.py
│
├── models/
│   ├── configs/
│   │   └── ollama_config.json
│   │
│   ├── prompts/
│   │   ├── system.md
│   │   ├── customer_agent.md
│   │   ├── ticket_summary.md
│   │   ├── factual_router.md
│   │   └── README.md
│   │
│   └── README.md
│
├── src/
│   ├── agents.py
│   ├── crm.py
│   ├── memory.py
│   ├── cohort.py
│   ├── heart.py
│   ├── segmentation.py
│   ├── evaluation.py
│   ├── models.py
│   ├── schemas.py
│   └── db.py
│
├── README.md
├── System_Report.md
└── requirements.txt
```

## Data Model

The current database schema has three main tables.

### Customer

Stores the customer profile:

- `id`
- `name`
- `email`
- `industry`
- `tier`
- `signup_date`
- `engagement_score`
- `status`
- `nps_score`
- `last_interaction_date`

This table is used for customer CRUD, cohort grouping, retention, churn labels, NPS, and profile grounding for the AI agent.

### Ticket

Stores support tickets:

- `id`
- `customer_id`
- `title`
- `description`
- `category`
- `priority`
- `status`
- `assigned_agent`
- `created_at`
- `updated_at`
- `resolved_at`

This table powers ticket CRUD, routing, resolution metrics, agent leaderboard, HEART Task Success, and support activity signals.

### Interaction

Stores customer interactions:

- `id`
- `customer_id`
- `channel`
- `message`
- `sentiment`
- `csat_score`
- `timestamp`

This table powers memory, CSAT, channel analysis, sentiment, re-engagement, and engagement metrics.

Important limitation: `Interaction` does not have a `ticket_id` column, so interactions are linked to customers directly, not to individual tickets.

## Running the Project

Start Ollama:

```powershell
ollama serve
```

Make sure the model is available:

```powershell
ollama pull llama3
```

Configure environment variables in `.env`:

```env
DATABASE_URL=mysql+pymysql://root:yourpassword@localhost/crm_db
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

Run the backend:

```powershell
uvicorn api.app:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Run the dashboard:

```powershell
streamlit run dashboard/app.py
```

## Authentication and Roles

The API uses simple bearer tokens for the demo.

| Role | Token | Main Access |
|---|---|---|
| Admin | `admin-token-001` | Full access |
| Supervisor | `supervisor-token-001` | Tickets, customers, analytics |
| Agent | `agent-token-001` | Own tickets, summaries, agent query |
| Analytics | `analytics-token-001` | Read-only analytics |

Example:

```http
Authorization: Bearer admin-token-001
```

For local demo convenience, requests without credentials are treated as admin. This should be changed before production use.

## Main Features

### Customer CRUD

Customers can be created, listed, viewed, updated, and deleted through FastAPI endpoints. Customer records also feed analytics and AI grounding.

### Ticket CRUD and Lifecycle

Tickets can be created, viewed, routed, summarized, and updated. When a ticket is moved to `Resolved` or `Closed`, the backend sets `resolved_at` if it was not already set.

Resolution time is calculated only when both `created_at` and `resolved_at` exist and `resolved_at >= created_at`.

### Customer Timeline

The timeline combines tickets and interactions for a customer and sorts them by timestamp, newest first.

Because the schema does not store `ticket_id` on interactions, timeline interactions are customer-level events rather than ticket-specific events.

### Customer Memory

Customer memory has two layers:

1. Short-term memory: recent interactions.
2. Long-term memory: aggregate behavior such as preferred channel, sentiment, CSAT, recent activity, ticket counts, and health label.

This memory is injected into the AI agent prompt so the model has customer context before answering.

## AI Agent

The AI agent has two paths.

### 1. Deterministic factual path

If the user asks a direct factual question, such as:

- "What is my tier?"
- "How many open tickets do I have?"
- "What is my latest ticket?"

the system answers directly from CRM data without calling the LLM.

This reduces latency and hallucination risk.

### 2. LLM reasoning path

If the query requires explanation, summarization, or reasoning, the system builds a prompt containing:

- CRM profile
- ticket history
- recent interactions
- long-term memory summary
- strict instructions not to invent facts

Then it sends the prompt to Ollama/Llama3.

### LangGraph Workflow

The CRM agent uses a deterministic workflow before invoking the LLM.

```text
Customer Query
        │
        ▼
Intent Classification
        │
 ┌──────┴────────┐
 │               │
 ▼               ▼
Factual      Reasoning
 │               │
 ▼               ▼
SQL Lookup   CRM Context
 │               │
 └──────┬────────┘
        ▼
  CRM Response
```

This architecture minimizes hallucination risk by answering factual questions directly from CRM records while reserving the LLM for analytical and reasoning tasks.

## Prompt Templates

The project includes reusable prompt templates under:

models/prompts/

These files document the prompts used by the CRM assistant, ticket summarizer and deterministic factual router.

The runtime implementation currently embeds the same prompts inside `src/agents.py` to preserve backward compatibility.

## Model Configuration

The directory

models/configs/

contains model configuration files used for local LLM deployment.

Example

- ollama_config.json

This documents the default Ollama model, host and inference parameters used by the project.

### Current Ollama Configuration

The current local LLM configuration uses:

| Parameter | Value |
|------------|-------|
| Provider | Ollama |
| Model | Llama 3 |
| Host | http://localhost:11434 |
| Temperature | 0.2 |
| Context Window | 4096 |
| Max Prediction Tokens | 512 |

The configuration is documented under:

```text
models/configs/ollama_config.json
```

These values can be modified independently of the application source code.

## Confidence Score

The project uses two confidence approaches.

### Ticket summarization confidence

For ticket summaries, confidence is based on Ollama response signals and answer sanity checks.

The model response is checked for:

- available token confidence data
- refusal/uncertainty phrases
- extremely short answers

Very short or uncertain answers are penalized.

### Customer agent confidence

For customer-agent responses, confidence is a weighted blend:

```text
40% CRM profile grounding
30% ticket history grounding
20% interaction memory grounding
10% token confidence
```

Each grounding component checks both:

- whether useful data exists
- whether the answer appears to use that data

This avoids giving high confidence just because data exists. The answer must actually show evidence of using it.

## Churn Score

Churn score is a rule-based risk score from `0` to `100`.

It uses:

- engagement score
- NPS score
- average sentiment
- average CSAT
- days since last interaction
- open tickets
- escalated tickets
- interaction count
- high ticket volume with negative sentiment

The score is capped at `100`.

Risk labels:

```text
0-39   Low Risk
40-69  Medium Risk
70-100 High Risk
```

This is not a trained ML model. It is a transparent heuristic designed for explainability.

## Cohort Analysis

Default cohorts are grouped by signup month:

```text
cohort = customer.signup_date formatted as YYYY-MM
```

For each cohort, the backend computes:

- total customers
- active customers
- inactive customers
- churned customers
- retention rate
- churn rate
- high-risk customers
- churn windows
- behavioral cohorts
- retention curve
- customer-level churn scores

### Retention Rate

```text
retention_rate = active_customers / total_customers * 100
```

### Churn Rate

```text
churn_rate = churned_customers / total_customers * 100
```

### Churn Windows

Churned customers are grouped by customer age:

- `0-30 Days`
- `31-60 Days`
- `61-90 Days`
- `90+ Days`

### Retention Curve

The system does not store historical monthly customer status snapshots. Because of that, exact historical retention curves cannot be reconstructed.

Instead, the backend estimates a curve using geometric decay. The curve starts near `100%` and lands on the cohort's current retention rate.

This makes the curve useful for visualization, but it must be interpreted as an estimate.

## Configurable Segmentation

The API supports cohort/segment grouping by:

- signup
- industry
- tier
- behavior

Industry and tier segmentation reuse the generic segmentation engine. Behavior segmentation is derived from behavioral tags computed during cohort analysis.

## HEART Metrics

HEART stands for:

- Happiness
- Engagement
- Adoption
- Retention
- Task Success

### Happiness

Happiness combines CSAT and customer rating:

```text
csat_normalized = average_csat / 5 * 100
nps_normalized = average_nps_score / 10 * 100
happiness = (csat_normalized + nps_normalized) / 2
```

Note: `nps_score` is stored as a per-customer `0-10` rating. It is not the traditional aggregate NPS formula of `% promoters - % detractors`.

### Engagement

Engagement blends three signals:

```text
50% recent interaction signal
25% recent ticket activity signal
25% average engagement_score
```

Where:

```text
recent interaction signal = customers with interaction in last 30 days / total customers * 100
recent ticket activity signal = customers with updated ticket in last 30 days / total customers * 100
```

### Adoption

Adoption is based on support-system usage:

```text
adoption = customers with at least one ticket / total customers * 100
```

### Retention

Retention uses customers with at least 90 days of tenure:

```text
retention = active tenured customers / total tenured customers * 100
```

### Task Success

Task Success measures support completion:

```text
task_success = resolved_or_closed_tickets / total_tickets * 100
```

## Resolution Metrics

Resolution analytics are computed only from ticket records.

### Average Resolution Time

For resolved tickets:

```text
resolution_hours = (resolved_at - created_at) in hours
average_resolution_time = average(resolution_hours)
```

Tickets without `resolved_at` are excluded.

### Time to First Resolution by Cohort

For each customer:

1. Find the customer's first resolved ticket.
2. Calculate hours between `created_at` and `resolved_at`.
3. Group by signup cohort.

Then compute:

- average
- median
- minimum
- maximum

### Resolution by Category

Grouped by `Ticket.category`, the backend computes:

- average resolution time
- median resolution time
- resolved tickets
- open tickets
- escalated tickets
- SLA breach percentage

SLA breach percentage:

```text
sla_breach_pct = resolved tickets taking more than SLA hours / resolved tickets * 100
```

Default SLA is `48` hours.

### Assigned Agent Leaderboard

Grouped by `assigned_agent`, the backend computes:

- total tickets
- resolved tickets
- escalations
- average resolution hours
- average CSAT for the agent's customers
- average NPS for the agent's customers
- average sentiment for the agent's customers

CSAT, NPS, and sentiment are customer-context aggregates because interactions are linked to customers, not directly to agents.

## Agent Quality Estimate

Agent quality is not a supervised evaluator score because the database does not store historical agent answers and human labels.

Instead, the dashboard shows a runtime quality estimate from real CRM data.

The estimate uses:

- CRM profile completeness
- ticket context quality
- memory context quality
- interaction recency
- source coverage
- response completeness
- current-process latency telemetry

### Grounding Score

```text
grounding_score =
  35% profile completeness
  25% ticket context quality
  25% memory context quality
  15% recency score
```

### Confidence Estimate

```text
confidence =
  45% grounding_score
  25% source_coverage
  20% response_completeness
  10% recency_score
```

### Hallucination Risk Estimate

```text
estimated_hallucination_risk =
  100 - grounding_score + escalation_rate_penalty
```

This is clamped between `0` and `100`.

### Final Agent Quality Score

```text
agent_quality_score =
  40% grounding_score
  30% confidence
  20% response_completeness
  10% low_hallucination_risk
```

This score should not be interpreted as human-graded answer quality. It is a practical estimate of whether the agent has enough reliable CRM context to answer well.

## Cohort Evaluation Metrics

The project uses existing CRM data as labels:

```text
actual churn label = Customer.status == "Churned"
predicted churn label = churn_score >= 70
```

Then it computes:

```text
TP = predicted high risk and actually churned
FP = predicted high risk but not churned
TN = not high risk and not churned
FN = not high risk but churned
```

Metrics:

```text
accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
coverage = predicted high risk / total * 100
```

If a denominator is missing, the metric returns `None` instead of pretending the value is zero.

## Latency Metrics

FastAPI middleware records request latency in memory for the current backend process.

The latency dashboard computes:

- average latency
- P95 latency
- P99 latency
- endpoint-level average latency
- endpoint-level P95 latency

Because this telemetry is in memory, it resets when the server restarts.

## Dashboard Pages

The Streamlit dashboard includes:

- Overview
- HEART Dashboard
- Cohort Analysis
- Tickets
- Customers
- AI Agent
- System Evaluation
- System Metrics

All charts use backend API data. No dashboard metric should be manually hardcoded.

## Exports

The backend supports:

- JSON cohort export
- PDF cohort/HEART report export

## Known Limitations

- Interactions are not directly linked to tickets.
- Retention curves are estimated because historical monthly status snapshots are not stored.
- Agent quality is a runtime estimate, not supervised evaluation.
- AI-vs-human resolution is proxy-based because there is no resolver-type column.
- In-memory latency metrics reset on backend restart.
- Demo auth uses static bearer tokens.

## Production Improvements

Recommended next steps:

- Add persistent telemetry for agent answers and evaluator labels.
- Add `ticket_id` to interactions if ticket-specific interaction analysis is required.
- Store historical customer status snapshots for exact retention curves.
- Add persistent latency/event logging.
- Replace demo token auth with JWT/OAuth.
- Train a churn model if labelled churn data becomes available.
