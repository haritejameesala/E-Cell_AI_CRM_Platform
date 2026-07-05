# System Report: E-Cell AI CRM Platform

## 1. Executive Summary

This system is an AI-assisted CRM platform for managing customers, support tickets, interaction history, customer memory, AI responses, and analytics dashboards.

The project is not only a CRUD application. It also computes business metrics from CRM records:

- customer retention
- churn risk
- behavioral cohorts
- HEART metrics
- ticket resolution performance
- assigned-agent performance
- latency metrics
- runtime AI quality estimates

The important design principle is that metrics are calculated from existing CRM data. Where the database does not contain enough information for an exact metric, the system labels the result as a heuristic, estimate, or proxy.

## 2. System Architecture

The platform has four main layers.

### 2.1 Database Layer

The database is MySQL accessed through SQLAlchemy ORM models.

The main entities are:

- `Customer`
- `Ticket`
- `Interaction`

These records are the source of truth for analytics.

### 2.2 Backend Layer

FastAPI exposes all API routes. The backend handles:

- CRUD operations
- RBAC
- audit metadata
- ticket routing
- AI calls
- cohort analytics
- HEART metrics
- evaluation metrics
- PDF and JSON exports

### 2.3 AI Layer

The AI layer uses:

- Ollama/Llama3 for local LLM responses
- LangGraph for deterministic ticket routing
- rule-based factual routing for simple CRM questions
- confidence and grounding logic for hallucination risk control

### 2.4 Dashboard Layer

Streamlit displays:

- customer overview
- HEART dashboard
- cohort analysis
- ticket tools
- customer timeline
- AI agent interface
- system evaluation dashboard
- latency and performance views

Plotly is used for charts.

## 3. Database Design

### 3.1 Customer Table

The `Customer` table stores account-level data:

- identity fields: name, email
- segmentation fields: industry, tier
- lifecycle fields: signup_date, status
- behavioral fields: engagement_score, nps_score, last_interaction_date

This table is used for retention, cohort grouping, churn labels, NPS, and CRM grounding.

### 3.2 Ticket Table

The `Ticket` table stores support cases:

- customer_id
- title and description
- category
- priority
- status
- assigned_agent
- created_at
- updated_at
- resolved_at

Ticket timestamps drive resolution metrics.

### 3.3 Interaction Table

The `Interaction` table stores customer communication:

- customer_id
- channel
- message
- sentiment
- csat_score
- timestamp

Interactions drive memory, sentiment, CSAT, re-engagement, and engagement calculations.

### 3.4 Relationship Limitation

Tickets and interactions are both linked to customers. Interactions are not directly linked to tickets because the schema does not include `Interaction.ticket_id`.

This means:

- customer-level timeline is supported
- customer-level memory is supported
- ticket-specific interaction attribution is not exact

The system does not pretend otherwise.

## 4. LangGraph Ticket Routing

Ticket routing is deterministic.

The workflow has two steps:

1. Route by category.
2. Escalate by priority.

Category routing:

```text
Billing   -> Billing_Team
Technical -> Tech_Team
Account   -> Account_Team
Other     -> General_Team
```

Priority routing:

```text
High or Critical -> Escalated
Low or Medium    -> In Progress
```

This workflow is simple, explainable, and repeatable.

## 5. AI Agent Design

The AI agent is designed to avoid unnecessary hallucination.

### 5.1 Factual Routing

If the user asks for a fact that exists in the CRM, the system answers directly from the database.

Examples:

- customer tier
- account status
- latest ticket
- open ticket count
- signup date

This path does not call the LLM.

Why this matters:

```text
Direct database answer > LLM-generated answer
```

For factual questions, using the LLM would add latency and hallucination risk without adding value.

### 5.2 LLM Reasoning Path

If the query requires explanation or synthesis, the system calls Ollama/Llama3.

The prompt includes:

- CRM profile
- ticket history
- recent interactions
- long-term summary
- response rules

The model is explicitly told not to invent facts.

### 5.3 Memory Injection

The memory module supplies:

- recent interactions
- total interaction count
- preferred channel
- average sentiment
- average CSAT
- recent 30-day activity
- ticket counts
- latest ticket details
- customer health label

This makes the AI response grounded in actual CRM history.

## 6. Confidence and Hallucination Guard

The system uses confidence scoring to estimate answer reliability.

### 6.1 Ticket Summary Confidence

Ticket summaries use Ollama output and answer sanity checks.

The system checks:

- token confidence information where available
- refusal phrases
- very short answers

Short or uncertain answers are penalized.

### 6.2 Customer Agent Confidence

Customer-agent confidence is a weighted blend:

```text
confidence =
  40% CRM profile grounding
  30% ticket history grounding
  20% memory grounding
  10% token confidence
```

Grounding means two things:

1. The relevant data exists.
2. The answer appears to use that data.

This is important. A complete customer profile should not automatically produce a high-confidence answer if the response never uses the profile.

## 7. Customer Health Score

Customer health is a rule-based label:

- Healthy
- Needs Attention
- High Risk

Signals used:

- engagement score
- NPS score
- average sentiment
- escalated ticket count
- recent interaction activity

High Risk is assigned when strong negative signals exist, such as:

- low engagement
- very low NPS
- strongly negative sentiment
- multiple escalations

Needs Attention is assigned for moderate warning signs.

## 8. Churn Scoring

Churn score is a transparent rule-based score from `0` to `100`.

It is computed from these real CRM signals:

- engagement score
- NPS score
- sentiment
- CSAT
- recency
- open tickets
- escalations
- interaction count
- ticket frequency

### 8.1 Scoring Logic

The model gives higher risk points for stronger warning signs.

Examples:

```text
Very low engagement       -> higher risk
Moderate-low engagement   -> medium risk
Very low NPS              -> higher risk
Low NPS                   -> medium risk
Strong negative sentiment -> higher risk
Negative sentiment        -> medium risk
Very low CSAT             -> higher risk
No recent activity        -> higher risk
Multiple escalations      -> higher risk
Multiple open tickets     -> higher risk
```

The final score is capped at `100`.

### 8.2 Risk Labels

```text
0-39   Low Risk
40-69  Medium Risk
70-100 High Risk
```

The threshold for High Risk remains `70`. The scoring model was adjusted so customers can naturally fall into all three buckets based on real behavior.

### 8.3 Why This Is Not a Machine Learning Model

This is not trained on labelled churn history. It is a heuristic model.

The benefit is explainability:

- every point comes from a visible customer signal
- reasons can be shown to the user
- the score is easy to audit

The limitation is that it is not statistically fitted.

## 9. Cohort Analysis

The default cohort is signup month.

```text
cohort = YYYY-MM from Customer.signup_date
```

For each cohort, the system computes:

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

### 9.1 Retention Rate

```text
retention_rate = active_customers / total_customers * 100
```

This measures how much of a cohort is currently active.

### 9.2 Churn Rate

```text
churn_rate = churned_customers / total_customers * 100
```

This measures how much of a cohort has churned.

### 9.3 Churn Windows

For customers whose status is `Churned`, the system calculates their age since signup.

Then it groups them into:

```text
0-30 Days
31-60 Days
61-90 Days
90+ Days
```

This shows whether churn is happening early or after longer usage.

## 10. Retention Curves

The database stores current customer status, not historical month-by-month status.

Because of that, an exact historical retention curve cannot be reconstructed.

The system uses an estimated geometric decay curve.

The curve:

- starts near 100%
- decays month by month
- lands on the current retention rate

This is useful for visualization, but it is an approximation.

Correct interpretation:

```text
This shows a plausible retention survival shape based on current status.
```

Incorrect interpretation:

```text
This is exact historical retention.
```

## 11. Behavioral Cohorts

Behavioral tags describe customer behavior beyond signup month.

Examples:

- High Risk
- Power User
- Promoter
- Support Heavy
- Low Engagement
- Inactive
- Standard

These are computed from:

- churn score
- engagement score
- NPS
- ticket count
- interaction count

This helps separate customers who joined at the same time but behave differently.

## 12. Re-engagement Metrics

Re-engagement looks at interaction gaps.

A customer is eligible if they have at least two interactions.

The system sorts interaction timestamps and checks whether the customer had:

```text
a gap longer than inactivity_threshold_days
followed by a later interaction
```

Default threshold:

```text
30 days
```

Formula:

```text
re_engagement_rate =
  re_engaged_customers / eligible_customers * 100
```

Customers with fewer than two interactions are excluded because there is no gap to measure.

## 13. HEART Framework

The HEART framework measures product/customer experience.

### 13.1 Happiness

Happiness combines CSAT and per-customer rating.

```text
average_csat = average Interaction.csat_score
csat_normalized = average_csat / 5 * 100

average_rating = average Customer.nps_score
rating_normalized = average_rating / 10 * 100

happiness = (csat_normalized + rating_normalized) / 2
```

Important note:

`nps_score` is stored as a `0-10` rating. It is not the traditional aggregate NPS formula.

### 13.2 Engagement

Engagement uses three signals:

```text
recent_interaction_signal =
  customers with interaction in last 30 days / total customers * 100

recent_ticket_signal =
  customers with ticket updated in last 30 days / total customers * 100

baseline_activity =
  average Customer.engagement_score
```

Final formula:

```text
engagement =
  50% recent_interaction_signal
  25% recent_ticket_signal
  25% baseline_activity
```

### 13.3 Adoption

Adoption uses ticket creation as a support-system usage proxy.

```text
adoption =
  customers with at least one ticket / total customers * 100
```

### 13.4 Retention

Retention uses customers with at least 90 days of tenure.

```text
retention =
  active customers signed up 90+ days ago / customers signed up 90+ days ago * 100
```

### 13.5 Task Success

Task Success measures ticket completion.

```text
task_success =
  tickets with status Resolved or Closed / total tickets * 100
```

## 14. Resolution Analytics

Resolution metrics are calculated from ticket timestamps.

### 14.1 Resolution Time

For each resolved ticket:

```text
resolution_hours =
  (Ticket.resolved_at - Ticket.created_at) in hours
```

Only tickets with valid `resolved_at` are included.

### 14.2 Category Metrics

For each ticket category:

- average resolution time
- median resolution time
- resolved tickets
- open tickets
- escalated tickets
- SLA breach percentage

SLA breach:

```text
sla_breach_pct =
  resolved tickets where resolution_hours > SLA hours
  / resolved tickets * 100
```

Default SLA:

```text
48 hours
```

### 14.3 Assigned Agent Metrics

For each assigned agent:

- total tickets
- resolved tickets
- escalations
- average resolution time
- average CSAT
- average NPS
- average sentiment

CSAT, NPS, and sentiment are calculated from customers handled by that agent.

Because interactions are not linked to agents directly, this is customer-context performance, not exact per-agent interaction scoring.

### 14.4 Time to First Resolution

For each customer:

1. Find the first ticket that has a `resolved_at`.
2. Calculate resolution hours.
3. Group by the customer's signup cohort.

Per cohort, calculate:

- average
- median
- minimum
- maximum

## 15. Cohort Evaluation Metrics

The system evaluates churn scoring using existing CRM labels.

Observed label:

```text
actual_churn = Customer.status == "Churned"
```

Prediction:

```text
predicted_churn = churn_score >= 70
```

Confusion matrix:

```text
TP = predicted churn and actually churned
FP = predicted churn but not churned
TN = not predicted churn and not churned
FN = not predicted churn but actually churned
```

Metrics:

```text
accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
coverage = predicted_churn / total * 100
```

If a denominator is zero, the metric returns `None` instead of `0`.

This avoids misleading results.

## 16. Agent Quality Estimate

The project does not store human evaluator labels for agent answers.

Therefore, it cannot honestly compute supervised metrics like true faithfulness or relevance.

Instead, it computes runtime quality estimates from available CRM data.

### 16.1 Signals Used

The estimate uses:

- CRM profile completeness
- ticket context quality
- memory context quality
- recency of interactions
- source coverage
- response completeness
- current-process latency telemetry

### 16.2 Grounding Score

```text
grounding_score =
  35% profile_completeness
  25% ticket_context_quality
  25% memory_context_quality
  15% recency_score
```

### 16.3 Confidence Estimate

```text
confidence =
  45% grounding_score
  25% source_coverage
  20% response_completeness
  10% recency_score
```

### 16.4 Hallucination Risk Estimate

```text
estimated_hallucination_risk =
  100 - grounding_score + escalation_penalty
```

The value is clamped between `0` and `100`.

### 16.5 Agent Quality Score

```text
agent_quality_score =
  40% grounding_score
  30% confidence
  20% response_completeness
  10% low_hallucination_risk
```

Where:

```text
low_hallucination_risk = 100 - estimated_hallucination_risk
```

This score means:

```text
"How well-supported is the agent likely to be, given available CRM context?"
```

It does not mean:

```text
"A human evaluator graded this answer as correct."
```

## 17. Latency Metrics

FastAPI middleware records request timings in memory.

The backend computes:

- average latency
- P95 latency
- P99 latency
- endpoint-level average latency
- endpoint-level P95 latency

P95 means:

```text
95% of requests were faster than or equal to this value
```

P99 means:

```text
99% of requests were faster than or equal to this value
```

Limitation:

Latency telemetry is in memory and resets on server restart.

## 18. Dashboard Validation

The dashboard is designed to use backend API data only.

Examples:

- HEART charts use `/api/v1/heart/dashboard`
- cohort charts use `/api/v1/cohorts/analysis`
- resolution charts use `/api/v1/resolution/category`, `/resolution/agents`, and `/resolution/cohort`
- latency charts use `/api/v1/evaluation/latency`
- agent quality charts use `/api/v1/evaluation/agent-quality`

No dashboard metric should be manually hardcoded.

## 19. API Metadata

Most API responses include audit metadata:

- timestamp
- processing_latency
- confidence_score where applicable
- api_version
- request_id

This improves traceability during demonstrations and debugging.

## 20. Performance Notes

The code avoids major N+1 patterns by using grouped queries for:

- ticket counts
- interaction counts
- ticket status counts
- interaction aggregates
- segmentation signals

Some analytics still read bounded demo-size tables and aggregate in Python. For the current dataset size, this is acceptable. For production scale, these should be converted to more SQL-side aggregation or materialized analytics tables.

## 21. Limitations

The main limitations are:

1. No direct `ticket_id` on interactions.
2. No historical monthly status snapshots.
3. No persisted supervised agent-evaluation labels.
4. AI-vs-human resolution is proxy-based.
5. Latency metrics are in memory.
6. Authentication uses demo static tokens.
7. Churn scoring is heuristic, not trained.

## 22. Future Work

Recommended improvements:

- Add persistent LLM evaluation telemetry.
- Add evaluator labels for true faithfulness/relevance scoring.
- Add `ticket_id` to interactions if schema changes are allowed.
- Store monthly customer status snapshots.
- Store persistent API latency logs.
- Train a churn model using real churn labels.
- Replace static tokens with JWT/OAuth.
- Add background jobs for metric precomputation.

## 23. Conclusion

The system is a complete AI CRM demo with meaningful backend logic and dashboard analytics.

The strongest parts are:

- transparent churn scoring
- CRM-grounded AI responses
- deterministic factual routing
- HEART and cohort analytics
- resolution dashboards
- clear metric explainability

The most important thing to communicate during evaluation is that some metrics are exact and some are estimates.

Exact metrics include:

- ticket counts
- resolution time
- task success
- CSAT averages
- NPS averages
- current retention rate
- churn rate
- latency during current server process

Estimated or proxy metrics include:

- retention curves
- agent quality
- AI-vs-human resolution
- churn risk score

This distinction makes the system more credible because it does not overclaim what the data cannot prove.
