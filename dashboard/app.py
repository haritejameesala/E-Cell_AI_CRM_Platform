import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Cell AI CRM",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card .label {
        color: #8892b0; font-size: 13px; font-weight: 600;
        letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px;
    }
    .metric-card .value { color: #64ffda; font-size: 32px; font-weight: 700; }
    .metric-card .unit  { color: #8892b0; font-size: 13px; }
    .section-title {
        color: #ccd6f6; font-size: 22px; font-weight: 700;
        margin: 24px 0 12px 0; padding-bottom: 8px;
        border-bottom: 2px solid #64ffda; display: inline-block;
    }
    .response-box {
        background: #1a1f2e; border: 1px solid #64ffda; border-radius: 10px;
        padding: 16px; color: #ccd6f6; font-size: 14px;
        line-height: 1.7; margin-top: 12px;
    }
    .info-box {
        background: #16213e; border-left: 4px solid #64ffda; border-radius: 6px;
        padding: 12px 16px; color: #8892b0; font-size: 13px; margin: 8px 0;
    }
    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="metric-container"] {
        background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px; padding: 12px;
    }
    /* ── Timeline badges (Feature 11) ──────────────────────────────────────── */
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
        text-transform: uppercase; margin-left: 6px;
    }
    .badge-status-open, .badge-status-in-progress { background: #2d3748; color: #ffd93d; }
    .badge-status-escalated { background: #4a1f27; color: #ff6b6b; }
    .badge-status-resolved, .badge-status-closed { background: #123b33; color: #64ffda; }
    .badge-priority-low { background: #16213e; color: #8892b0; }
    .badge-priority-medium { background: #2d3748; color: #ffd93d; }
    .badge-priority-high { background: #4a2f1f; color: #f7971e; }
    .badge-priority-critical { background: #4a1f27; color: #ff6b6b; }
    .timeline-header { color: #ccd6f6; font-size: 14px; font-weight: 600; }
    .timeline-meta { color: #8892b0; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#ccd6f6",
    font_size=12,
)

# ── Timeouts: long for LLM calls, short for data calls ────────────────────────
DATA_TIMEOUT = 30       # fast DB queries
LLM_TIMEOUT  = 300      # Ollama LLM can take up to 5 minutes on first call

def api_get(path, timeout=DATA_TIMEOUT):
    try:
        r = requests.get(f"{BASE_URL}{path}", timeout=timeout)
        return r.json()
    except requests.exceptions.Timeout:
        st.error(f"⏱ Request timed out. The server took too long to respond.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure FastAPI is running: `uvicorn api.app:app --reload`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def api_post(path, payload=None, timeout=DATA_TIMEOUT):
    try:
        r = requests.post(f"{BASE_URL}{path}", json=payload, timeout=timeout)
        return r.json()
    except requests.exceptions.Timeout:
        st.error("⏱ LLM request timed out. Ollama may still be loading the model — try again in 30 seconds.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure FastAPI is running.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def metric_card(label, value, unit=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}<span class="unit">{unit}</span></div>
    </div>
    """, unsafe_allow_html=True)


def _status_badge(status):
    """Small colored pill for a ticket status - used in the timeline (Feature 11)."""
    slug = (status or "unknown").lower().replace(" ", "-")
    return f'<span class="badge badge-status-{slug}">{status or "Unknown"}</span>'


def _priority_badge(priority):
    """Small colored pill for a ticket priority - used in the timeline (Feature 11)."""
    slug = (priority or "unknown").lower()
    return f'<span class="badge badge-priority-{slug}">{priority or "Unknown"}</span>'

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 E-Cell AI CRM")
    st.markdown("**NIT Trichy · Task 3**")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "❤️ HEART Dashboard", "📊 Cohort Analysis",
         "🎫 Tickets", "👥 Customers", "🤖 AI Agent", "🧪 System Evaluation",
         "📈 System Metrics"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "<div class='info-box'>Backend: FastAPI<br>LLM: Ollama Llama3<br>"
        "Agents: LangGraph<br>DB: MySQL<br>Confidence: Logprobs</div>",
        unsafe_allow_html=True
    )
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🚀 E-Cell AI CRM Platform")
    st.markdown("**AI-Integrated Customer Relationship Management · NIT Trichy**")
    st.markdown("---")

    customers_data = api_get("/api/v1/customers?limit=500")
    heart_data     = api_get("/api/v1/heart/dashboard")

    if customers_data:
        customers = customers_data.get("customers", [])
        total     = customers_data.get("total", 0)
        active    = sum(1 for c in customers if c["status"] == "Active")
        churned   = sum(1 for c in customers if c["status"] == "Churned")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: metric_card("Total Customers", total)
        with col2: metric_card("Active", active)
        with col3: metric_card("Churned", churned)
        with col4:
            metric_card("Retention",
                f"{heart_data.get('Retention', 0):.1f}" if heart_data else "—", "%")
        with col5:
            metric_card("Task Success",
                f"{heart_data.get('Task_Success', 0):.1f}" if heart_data else "—", "%")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='section-title'>Customers by Industry</div>", unsafe_allow_html=True)
        if customers_data:
            df = pd.DataFrame(customers_data["customers"])
            ind_df = df["industry"].value_counts().reset_index()
            ind_df.columns = ["Industry", "Count"]
            fig = px.pie(ind_df, values="Count", names="Industry",
                color_discrete_sequence=["#64ffda","#7c83fd","#f7971e","#ff6b6b","#ffd93d"])
            fig.update_layout(**CHART_THEME, showlegend=True, height=300, margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-title'>Customer Status</div>", unsafe_allow_html=True)
        if customers_data:
            df = pd.DataFrame(customers_data["customers"])
            st_df = df["status"].value_counts().reset_index()
            st_df.columns = ["Status", "Count"]
            fig = px.bar(st_df, x="Status", y="Count", color="Status",
                color_discrete_map={"Active":"#64ffda","Churned":"#ff6b6b","Inactive":"#ffd93d"})
            fig.update_layout(**CHART_THEME, height=300, showlegend=False, margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HEART DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "❤️ HEART Dashboard":
    st.markdown("# ❤️ HEART Framework Dashboard")
    st.caption("Happiness · Engagement · Adoption · Retention · Task Success")
    st.markdown("---")

    data = api_get("/api/v1/heart/dashboard")
    if data:
        h = round(data.get("Happiness", 0), 1)
        e = round(data.get("Engagement", 0), 1)
        a = round(data.get("Adoption", 0), 1)
        r = round(data.get("Retention", 0), 1)
        t = round(data.get("Task_Success", 0), 1)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: metric_card("😊 Happiness",   h, "%")
        with col2: metric_card("⚡ Engagement",  e, "%")
        with col3: metric_card("🎯 Adoption",    a, "%")
        with col4: metric_card("🔄 Retention",   r, "%")
        with col5: metric_card("✅ Task Success", t, "%")

        st.markdown("<div class='section-title'>HEART Score Radar</div>", unsafe_allow_html=True)
        cats = ["Happiness","Engagement","Adoption","Retention","Task Success"]
        vals = [h, e, a, r, t]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
            fillcolor="rgba(100,255,218,0.15)",
            line=dict(color="#64ffda", width=2), name="HEART"
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100], color="#8892b0"),
                angularaxis=dict(color="#ccd6f6"),
                bgcolor="rgba(0,0,0,0)"
            ),
            **CHART_THEME, height=420, showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>Signal Sources</div>", unsafe_allow_html=True)
        for metric, source in data.get("signal_sources", {}).items():
            st.markdown(f"<div class='info-box'><b>{metric}</b>: {source}</div>",
                        unsafe_allow_html=True)

        st.markdown("<div class='section-title'>HEART by Cohort</div>", unsafe_allow_html=True)
        cohort_heart = api_get("/api/v1/heart/by-cohort")
        if cohort_heart:
            ch_df = pd.DataFrame(cohort_heart.get("cohort_heart_scores", []))
            if not ch_df.empty:
                fig2 = px.line(ch_df, x="cohort",
                    y=["Happiness","Engagement","Adoption","Retention","Task_Success"],
                    color_discrete_sequence=["#64ffda","#7c83fd","#f7971e","#ff6b6b","#ffd93d"])
                fig2.update_layout(**CHART_THEME, height=380, margin=dict(t=20,b=40))
                st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: COHORT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Cohort Analysis":
    st.markdown("# 📊 Cohort Analysis Engine")
    st.caption("Retention curves · Churn prediction · Customer segments")
    st.markdown("---")

    data = api_get("/api/v1/cohorts/analysis")
    if data:
        cohorts = data.get("cohort_analysis", [])
        df = pd.DataFrame([{
            "Cohort":      c["cohort"],
            "Total":       c["total_customers"],
            "Active":      c["active_customers"],
            "Churned":     c["churned_customers"],
            "Retention %": c["retention_rate"],
            "Churn %":     c["churn_rate"],
            "High Risk":   c["high_risk_customers"],
        } for c in cohorts])

        col1, col2, col3 = st.columns(3)
        with col1: metric_card("Total Cohorts", len(cohorts))
        with col2: metric_card("Avg Retention", f"{df['Retention %'].mean():.1f}", "%")
        with col3: metric_card("Avg Churn",     f"{df['Churn %'].mean():.1f}", "%")

        st.markdown("---")
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("<div class='section-title'>Retention Curve</div>", unsafe_allow_html=True)
            curve_rows = []
            for cohort in cohorts:
                for point in cohort.get("retention_curve", []):
                    curve_rows.append({
                        "Cohort": cohort["cohort"],
                        "Month": point.get("month", 0),
                        "Retention %": point.get("retention_pct", 0),
                    })
            curve_df = pd.DataFrame(curve_rows)
            if not curve_df.empty:
                fig = px.line(curve_df, x="Month", y="Retention %", color="Cohort", markers=True)
                fig.update_layout(**CHART_THEME, height=320, margin=dict(t=10,b=40))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No retention curve data available.")

        with col_r:
            st.markdown("<div class='section-title'>Churn Rate by Cohort</div>", unsafe_allow_html=True)
            fig2 = px.bar(df, x="Cohort", y="Churn %", color="Churn %",
                color_continuous_scale=["#64ffda","#ffd93d","#ff6b6b"])
            fig2.update_layout(**CHART_THEME, height=320, margin=dict(t=10,b=40),
                coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='section-title'>High Risk Customers by Cohort</div>",
            unsafe_allow_html=True)
        fig3 = px.bar(df, x="Cohort", y="High Risk",
            color_discrete_sequence=["#ff6b6b"])
        fig3.update_layout(**CHART_THEME, height=260, margin=dict(t=10,b=40))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("<div class='section-title'>Configurable Cohort Segmentation</div>",
            unsafe_allow_html=True)
        group_by = st.selectbox("Group cohorts by", ["signup", "industry", "tier", "behavior"])
        grouped = api_get(f"/api/v1/cohorts/configurable?group_by={group_by}")
        if grouped:
            groups = grouped.get("groups", [])
            group_df = pd.DataFrame(groups)
            if not group_df.empty:
                label_col = "cohort" if "cohort" in group_df.columns else "segment"
                value_col = "total_customers" if "total_customers" in group_df.columns else group_df.columns[-1]
                fig_group = px.bar(group_df, x=label_col, y=value_col,
                    color_discrete_sequence=["#7c83fd"])
                fig_group.update_layout(**CHART_THEME, height=300, margin=dict(t=10,b=40))
                st.plotly_chart(fig_group, use_container_width=True)

        # ─────────────────────────────────────────────────────────────
        # Behavioral Cohort Distribution
        # ─────────────────────────────────────────────────────────────

        st.markdown(
            "<div class='section-title'>Behavioral Cohorts</div>",
            unsafe_allow_html=True
        )

        rows = []

        for cohort in cohorts:
            for behavior, count in cohort.get("behavioral_cohorts", {}).items():
                rows.append({
                    "Cohort": cohort["cohort"],
                    "Behavior": behavior,
                    "Customers": count
                })

        if rows:
            behavior_df = pd.DataFrame(rows)

            fig = px.bar(
                behavior_df,
                x="Cohort",
               y="Customers",
                color="Behavior",
                barmode="stack",
                title="Behavioral Cohort Distribution"
            )

            fig.update_layout(
                **CHART_THEME,
                height=350,
                margin=dict(t=30, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

            # ─────────────────────────────────────────────────────────────
            # Churn Windows
            # ─────────────────────────────────────────────────────────────

            st.markdown(
                "<div class='section-title'>Churn Windows</div>",
                unsafe_allow_html=True
            )

            rows = []

            for cohort in cohorts:

                for window, count in cohort.get("churn_windows", {}).items():

                    rows.append({
                        "Cohort": cohort["cohort"],
                        "Window": window,
                        "Customers": count
                    })

            if rows:

                churn_df = pd.DataFrame(rows)

                fig = px.bar(
                    churn_df,
                    x="Window",
                    y="Customers",
                    color="Cohort",
                    barmode="group",
                    title="Customer Churn Windows"
                )

                fig.update_layout(
                    **CHART_THEME,
                    height=350,
                    margin=dict(t=30, b=40)
                )

                st.plotly_chart(fig, use_container_width=True)
		
        st.markdown("<div class='section-title'>Cohort Summary Table</div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=280)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TICKETS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎫 Tickets":
    st.markdown("# 🎫 Ticket Management")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔍 Summarize Ticket", "🔄 Route Ticket", "✏️ Update Status"])

    with tab1:
        st.markdown("### AI Ticket Summarization")
        st.caption("Uses LangChain + Llama3 (Ollama). Confidence via token logprobs.")
        st.markdown(
            "<div class='info-box'>⚠️ First call may take 60–120 seconds while Ollama loads the model into GPU memory. Subsequent calls are much faster.</div>",
            unsafe_allow_html=True
        )
        ticket_id = st.number_input("Ticket ID", min_value=1, value=1, key="sum_id")
        if st.button("🤖 Summarize", key="sum_btn", use_container_width=True):
            with st.spinner("Summarizing with Llama3... (may take up to 2 mins on first call)"):
                # ← LLM_TIMEOUT used here, not DATA_TIMEOUT
                res = api_post(f"/api/v1/tickets/{ticket_id}/summarize",
                               timeout=LLM_TIMEOUT)
            if res and "summary" in res:
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✅ Confidence: **{res.get('confidence_score', 0):.1f}%**")
                    st.info(f"⏱ Latency: **{res.get('processing_latency', 0):.3f}s**")
                with col2:
                    st.warning(f"🚨 Urgency: **{res.get('urgency', 'N/A')}**")
                    st.info(f"🤖 Agent: **{res.get('agent_id', 'N/A')}**")
                method = res.get("confidence_method", "N/A")
                st.caption(f"Confidence method: `{method}`")
                st.markdown(
                    f"<div class='response-box'>"
                    f"<b>Summary:</b><br>{res.get('summary','')}<br><br>"
                    f"<b>Key Issues:</b><br>{res.get('key_issues','')}<br><br>"
                    f"<b>Suggested Resolution:</b><br>{res.get('suggested_response','')}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            elif res:
                st.error(f"Error: {res}")
            # else: timeout message already shown by api_post

    with tab2:
        st.markdown("### LangGraph Ticket Routing")
        st.caption("State machine: route → escalate → assign. No LLM call — instant.")
        ticket_id2 = st.number_input("Ticket ID", min_value=1, value=1, key="route_id")
        if st.button("⚡ Route Ticket", use_container_width=True):
            with st.spinner("Running LangGraph workflow..."):
                res = api_post(f"/api/v1/tickets/{ticket_id2}/route")
            if res and "assigned_agent" in res:
                col1, col2 = st.columns(2)
                with col1: st.success(f"✅ Assigned to: **{res['assigned_agent']}**")
                with col2: st.info(f"📋 Status: **{res['status']}**")
            elif res:
                st.error(f"Error: {res}")

    with tab3:
        st.markdown("### Update Ticket Status")
        ticket_id3 = st.number_input("Ticket ID", min_value=1, value=1, key="upd_id")
        new_status = st.selectbox("New Status",
            ["Open","In Progress","Escalated","Resolved","Closed"])
        if st.button("💾 Update Status", use_container_width=True):
            res = api_post(f"/api/v1/tickets/{ticket_id3}/status",
                           {"status": new_status})
            if res and "new_status" in res:
                st.success(f"✅ Ticket #{ticket_id3} → **{res['new_status']}**")
            elif res:
                st.error(f"Error: {res}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customers":
    st.markdown("# 👥 Customer Management")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 All Customers", "🔍 Customer Timeline", "➕ Add Customer"])

    with tab1:
        data = api_get("/api/v1/customers?limit=500")
        if data:
            df = pd.DataFrame(data["customers"])
            st.markdown(f"**{data['total']} customers loaded**")
            col1, col2 = st.columns(2)
            with col1:
                status_f = st.multiselect("Filter by Status",
                    ["Active","Inactive","Churned"],
                    default=["Active","Inactive","Churned"])
            with col2:
                ind_f = st.multiselect("Filter by Industry",
                    df["industry"].unique().tolist(),
                    default=df["industry"].unique().tolist())
            filtered = df[df["status"].isin(status_f) & df["industry"].isin(ind_f)]
            st.dataframe(filtered, use_container_width=True, height=400)

    with tab2:
        cid = st.number_input("Customer ID", min_value=1, value=1)
        show_count = st.slider("Events to display", 5, 100, 20, key="timeline_count")
        if st.button("📅 Load Timeline", use_container_width=True):
            res = api_get(f"/api/v1/customers/{cid}/timeline")
            if res and "timeline" in res:
                st.info(f"**{res['total_events']} events** for Customer #{cid} "
                        f"(most recent first)")
                # Ticket Created → Customer Interaction → Agent Response →
                # Status Updates → Resolution: the backend already returns
                # every ticket + interaction sorted chronologically
                # (newest first), so rendering in that order naturally
                # reconstructs this flow per ticket.
                for event in res["timeline"][:show_count]:
                    ts = (event.get("timestamp") or "")[:16].replace("T", " ")

                    if event["type"] == "ticket":
                        # st.expander() labels only support a small markdown
                        # subset (bold/italic/code/emoji) - raw HTML like our
                        # badge <span> tags is NOT rendered there and would
                        # show up as literal tag text. So the expander label
                        # stays plain markdown, and the actual colored
                        # badges are rendered with st.markdown(unsafe_allow_
                        # html=True) inside the expander body instead.
                        label = (
                            f"🎫 `{ts}`  **{event['title']}**  "
                            f"[{event.get('priority','?')} · {event.get('status','?')}]"
                        )
                        with st.expander(label, expanded=False):
                            st.markdown(
                                f"<span class='timeline-header'>{event['title']}</span>"
                                f"{_priority_badge(event.get('priority'))}"
                                f"{_status_badge(event.get('status'))}",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<span class='timeline-meta'>Ticket #{event.get('id')} · "
                                f"Category: {event.get('category', 'Unknown')}</span>",
                                unsafe_allow_html=True,
                            )
                    else:
                        # The DB already stores the full message - only the
                        # UI used to truncate it. Show a one-line preview in
                        # the label and the FULL message inside the
                        # expander, so nothing is lost.
                        preview = event["message"][:80] + ("…" if len(event["message"]) > 80 else "")
                        label = f"💬 `{ts}`  **[{event.get('channel','Unknown')}]**  {preview}"
                        with st.expander(label, expanded=False):
                            st.markdown(
                                f"<div class='response-box'>{event['message']}</div>",
                                unsafe_allow_html=True,
                            )
                            meta_bits = []
                            if event.get("sentiment") is not None:
                                meta_bits.append(f"Sentiment: {event['sentiment']:.2f}")
                            if event.get("csat_score") is not None:
                                meta_bits.append(f"CSAT: {event['csat_score']}")
                            if event.get("ticket_id") is not None:
                                meta_bits.append(f"Ticket #{event['ticket_id']}")
                            if meta_bits:
                                st.markdown(
                                    f"<span class='timeline-meta'>{' · '.join(meta_bits)}</span>",
                                    unsafe_allow_html=True,
                                )
            elif res:
                st.error("Customer not found.")

    with tab3:
        st.markdown("### Add New Customer")
        with st.form("add_customer"):
            col1, col2 = st.columns(2)
            with col1:
                name     = st.text_input("Name")
                email    = st.text_input("Email")
                industry = st.selectbox("Industry",
                    ["AI","FinTech","Healthcare","EdTech","Retail"])
            with col2:
                tier   = st.selectbox("Tier", ["Free","Basic","Premium","Enterprise"])
                status = st.selectbox("Status", ["Active","Inactive"])
                eng    = st.slider("Engagement Score", 0.0, 100.0, 50.0)
            submitted = st.form_submit_button("➕ Create Customer", use_container_width=True)
            if submitted:
                payload = {
                    "name": name, "email": email, "industry": industry,
                    "tier": tier, "status": status, "engagement_score": eng,
                    "signup_date": datetime.now().date().isoformat(), "nps_score": None
                }
                res = api_post("/api/v1/customers", payload)
                if res and "id" in res:
                    st.success(f"✅ Created! ID: {res['id']} · Cohort: {res['cohort_assignment']}")
                elif res:
                    st.error(f"Failed: {res}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI AGENT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Agent":
    st.markdown("# 🤖 AI Agent — Memory-Aware Support")
    st.caption("Llama3 via Ollama · LangGraph state machine · True logprob confidence")
    st.markdown("---")

    st.markdown(
        "<div class='info-box'>⚠️ LLM calls can take 30–120 seconds depending on your GPU. "
        "The spinner will stay active until the response is ready — do not refresh.</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        customer_id = st.number_input("Customer ID", min_value=1, value=1)
        query = st.text_area("Customer Query", height=120,
            placeholder="e.g. Why was my last ticket escalated?")
        submit = st.button("🤖 Ask Agent", use_container_width=True)

    with col2:
        if submit and query:
            with st.spinner("Agent thinking... (up to 2 mins on first call)"):
                res = api_post("/api/v1/query/agent",
                               {"customer_id": customer_id, "query": query},
                               timeout=LLM_TIMEOUT)   # ← 300s timeout
            if res and "answer" in res:
                st.markdown(
                    "<div class='response-box'>" +
                    res["answer"].replace("\n", "<br>") +
                    "</div>",
                    unsafe_allow_html=True
                )
                st.markdown("---")
                col_a, col_b, col_c = st.columns(3)
                with col_a: st.metric("Confidence",  f"{res.get('confidence_score', 0):.1f}%")
                with col_b: st.metric("Latency",     f"{res.get('processing_latency', 0):.3f}s")
                with col_c: st.metric("Agent ID",    res.get("agent_id", "N/A"))
                st.caption(f"Method: `{res.get('confidence_method','N/A')}` · Source: {res.get('source','N/A')}")
            elif res:
                st.error(f"Agent error: {res}")
        else:
            st.markdown(
                "<div class='info-box'>Enter a customer ID and query, "
                "then click <b>Ask Agent</b>.</div>",
                unsafe_allow_html=True
            )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SYSTEM EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 System Evaluation":
    st.markdown("# 🧪 System Evaluation")
    st.caption("HEART · Cohorts · Resolution · Latency · Agent Quality · Churn · Retention")
    st.markdown("---")

    heart_data = api_get("/api/v1/heart/dashboard")
    expanded_heart = api_get("/api/v1/heart/expanded")
    cohort_eval = api_get("/api/v1/evaluation/cohorts")
    resolution_eval = api_get("/api/v1/evaluation/resolution")
    latency_data = api_get("/api/v1/evaluation/latency")
    agent_quality = api_get("/api/v1/evaluation/agent-quality")
    by_category = api_get("/api/v1/resolution/category")
    by_agent = api_get("/api/v1/resolution/agents")
    by_cohort = api_get("/api/v1/resolution/cohort")

    col1, col2, col3, col4 = st.columns(4)
    f1_value = cohort_eval.get("f1") if cohort_eval else None
    with col1: metric_card("HEART Happiness", f"{heart_data.get('Happiness', 0):.1f}" if heart_data else "—", "%")
    with col2: metric_card("Ticket Closure", f"{resolution_eval.get('ticket_closure_pct', 0):.1f}" if resolution_eval else "—", "%")
    with col3: metric_card("Cohort F1", f"{f1_value:.2f}" if f1_value is not None else "Insufficient")
    with col4: metric_card("P95 Latency", f"{latency_data.get('p95', 0):.3f}" if latency_data else "—", "s")

    st.markdown("<div class='section-title'>Resolution by Category</div>", unsafe_allow_html=True)
    if by_category:
        cat_df = pd.DataFrame(by_category.get("categories", []))
        if not cat_df.empty:
            fig = px.bar(cat_df, x="category", y=["resolved_tickets", "open_tickets", "escalated_tickets"], barmode="group")
            fig.update_layout(**CHART_THEME, height=320, margin=dict(t=10,b=40))
            st.plotly_chart(fig, use_container_width=True)
            fig_sla = px.bar(cat_df, x="category", y="sla_breach_pct", color="sla_breach_pct",
                color_continuous_scale=["#64ffda", "#ffd93d", "#ff6b6b"])
            fig_sla.update_layout(**CHART_THEME, height=280, margin=dict(t=10,b=40), coloraxis_showscale=False)
            st.plotly_chart(fig_sla, use_container_width=True)

    st.markdown("<div class='section-title'>Assigned Agent Leaderboard</div>", unsafe_allow_html=True)
    if by_agent:
        agent_df = pd.DataFrame(by_agent.get("agents", []))
        if not agent_df.empty:
            display_cols = ["agent", "resolved_tickets", "escalations", "avg_resolution_hours", "avg_csat", "avg_nps", "avg_sentiment"]
            st.dataframe(agent_df[display_cols], use_container_width=True, height=260)
            fig = px.bar(agent_df.sort_values("resolved_tickets", ascending=False), x="agent", y="resolved_tickets",
                color_discrete_sequence=["#64ffda"])
            fig.update_layout(**CHART_THEME, height=300, margin=dict(t=10,b=40))
            st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='section-title'>Time to First Resolution by Cohort</div>", unsafe_allow_html=True)
        if by_cohort:
            tfr_df = pd.DataFrame(by_cohort.get("time_to_first_resolution_by_cohort", []))
            if not tfr_df.empty:
                fig = px.line(tfr_df, x="cohort", y=["avg_hours", "median_hours", "min_hours", "max_hours"], markers=True)
                fig.update_layout(**CHART_THEME, height=320, margin=dict(t=10,b=40))
                st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-title'>Agent Quality</div>", unsafe_allow_html=True)
        if agent_quality and agent_quality.get("available"):
            quality_df = pd.DataFrame({
                "Metric": ["Quality", "Grounding", "Confidence", "Response Completeness", "Low Hallucination Risk"],
                "Score": [
                    agent_quality.get("agent_quality_score", 0),
                    agent_quality.get("grounding_score", 0),
                    agent_quality.get("confidence", 0),
                    agent_quality.get("response_completeness", 0),
                    100 - agent_quality.get("estimated_hallucination_risk", 0),
                ],
            })
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=quality_df["Score"].mean(),
                title={"text": "Overall Quality"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#64ffda"}},
            ))
            fig.update_layout(**CHART_THEME, height=320)
            st.plotly_chart(fig, use_container_width=True)
        elif agent_quality:
            st.markdown(
                f"<div class='info-box'>{agent_quality.get('note', 'Agent quality telemetry unavailable.')}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>Latency by Endpoint</div>", unsafe_allow_html=True)
    if latency_data:
        lat_df = pd.DataFrame(latency_data.get("endpoint_latency", []))
        if not lat_df.empty:
            fig = px.bar(lat_df, x="endpoint", y="avg_latency", color="p95",
                color_continuous_scale=["#64ffda", "#ffd93d", "#ff6b6b"])
            fig.update_layout(**CHART_THEME, height=340, margin=dict(t=10,b=80), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Expanded HEART Signals</div>", unsafe_allow_html=True)
    if expanded_heart:
        col_a, col_b = st.columns(2)
        with col_a:
            wau_df = pd.DataFrame(expanded_heart.get("engagement", {}).get("weekly_active_users", []))
            if not wau_df.empty:
                fig = px.line(wau_df, x="week", y="active_users", markers=True, title="Weekly Active Users")
                fig.update_layout(**CHART_THEME, height=280, margin=dict(t=35,b=40))
                st.plotly_chart(fig, use_container_width=True)
        with col_b:
            survival_rows = []
            for item in expanded_heart.get("retention", {}).get("survival_trend", []):
                for point in item.get("retention_curve", []):
                    survival_rows.append({"cohort": item["cohort"], "month": point["month"], "retention_pct": point["retention_pct"]})
            survival_df = pd.DataFrame(survival_rows)
            if not survival_df.empty:
                fig = px.line(survival_df, x="month", y="retention_pct", color="cohort", title="Survival Trend")
                fig.update_layout(**CHART_THEME, height=280, margin=dict(t=35,b=40))
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SYSTEM METRICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 System Metrics":
    st.markdown("# 📈 System Evaluation Metrics")
    st.caption("Agent quality · Latency · Cohort accuracy · Resolution rate")
    st.markdown("---")

    heart_data  = api_get("/api/v1/heart/dashboard")
    cohort_data = api_get("/api/v1/cohorts/analysis")

    if cohort_data:
        cohorts = cohort_data.get("cohort_analysis", [])
        df = pd.DataFrame([{
            "Cohort":    c["cohort"],
            "Retention": c["retention_rate"],
            "Churn":     c["churn_rate"],
            "HighRisk":  c["high_risk_customers"],
            "Total":     c["total_customers"],
        } for c in cohorts])

        col1, col2, col3, col4 = st.columns(4)
        with col1: metric_card("Avg Retention",  f"{df['Retention'].mean():.1f}", "%")
        with col2: metric_card("Avg Churn",      f"{df['Churn'].mean():.1f}", "%")
        with col3:
            metric_card("Task Success",
                f"{heart_data.get('Task_Success', 0):.1f}" if heart_data else "—", "%")
        with col4:
            metric_card("Happiness",
                f"{heart_data.get('Happiness', 0):.1f}" if heart_data else "—", "%")

        st.markdown("---")
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("<div class='section-title'>Retention vs Churn</div>",
                unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["Cohort"], y=df["Retention"],
                name="Retention", marker_color="#64ffda"))
            fig.add_trace(go.Bar(x=df["Cohort"], y=df["Churn"],
                name="Churn", marker_color="#ff6b6b"))
            fig.update_layout(**CHART_THEME, barmode="group", height=320,
                margin=dict(t=10,b=40), legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("<div class='section-title'>High Risk per Cohort</div>",
                unsafe_allow_html=True)
            fig2 = px.area(df, x="Cohort", y="HighRisk",
                color_discrete_sequence=["#f7971e"])
            fig2.update_layout(**CHART_THEME, height=320, margin=dict(t=10,b=40))
            st.plotly_chart(fig2, use_container_width=True)

        if heart_data:
            st.markdown("<div class='section-title'>Full HEART Scores</div>",
                unsafe_allow_html=True)
            hdf = pd.DataFrame({
                "Metric": ["Happiness","Engagement","Adoption","Retention","Task Success"],
                "Score (%)": [
                    heart_data.get("Happiness",0), heart_data.get("Engagement",0),
                    heart_data.get("Adoption",0),  heart_data.get("Retention",0),
                    heart_data.get("Task_Success",0),
                ],
                "Signal": [
                    "Avg CSAT + NPS normalised to 0–100",
                    "% customers with interaction in last 30 days",
                    "% customers who raised ≥1 ticket",
                    "% customers signed up 90+ days ago still Active",
                    "% tickets Resolved or Closed",
                ]
            })
            st.dataframe(hdf, use_container_width=True)
