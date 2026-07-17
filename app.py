import uuid

import streamlit as st
from dotenv import load_dotenv

from agent.graph import agent_graph
from agent.memory import init_db, save_session
from models.mandate import InvestorMandate

load_dotenv()
init_db()

st.set_page_config(
    page_title="PolarityIQ Intelligence Agent",
    page_icon="\U0001F3AF",
    layout="wide"
)

st.title("PolarityIQ Private Markets Intelligence Agent")
st.caption("Autonomous investor mandate research, powered by Claude and LangGraph")

with st.sidebar:
    st.header("Configuration")
    max_iterations = st.slider("Max Search Iterations", 1, 10, 5)
    max_results = st.slider("Max Results per Query", 3, 15, 5)
    st.divider()
    st.caption("Every agent decision is logged to /logs as structured JSON.")

mandate_input = st.text_area(
    "Enter Investor Mandate",
    placeholder="e.g. Family office focused on Series B SaaS companies in North America, "
                "typical check size $5M-$20M",
    height=100
)

if st.button("Run Intelligence Agent", type="primary"):
    if not mandate_input.strip():
        st.error("Please enter a mandate.")
    else:
        session_id = str(uuid.uuid4())[:8]
        mandate = InvestorMandate(
            raw_input=mandate_input,
            max_search_iterations=max_iterations,
            max_results_per_query=max_results
        )
        save_session(session_id, mandate_input)

        initial_state = {
            "mandate": mandate,
            "session_id": session_id,
            "search_queries": [],
            "decomposition_reasoning": "",
            "raw_results": [],
            "classified_results": [],
            "iterations_used": 0,
            "cost_ceiling_hit": False,
            "ambiguous_items_pending": [],
            "human_validation_required": False,
            "report": None,
            "decision_trace": [],
            "errors": [],
            "messages": []
        }

        with st.spinner("Agent running..."):
            final_state = agent_graph.invoke(initial_state)

        report = final_state.get("report")

        if report:
            col1, col2, col3 = st.columns(3)
            col1.metric("Strong Matches", len(report.strong_matches))
            col2.metric("Weak Matches", len(report.weak_matches))
            col3.metric("Searches Run", report.total_searches_performed)

            if report.cost_ceiling_hit:
                st.warning("Cost ceiling was hit - results may be partial.")

            st.divider()
            st.subheader("Intelligence Synthesis")
            st.write(report.synthesis)

            if report.recommended_actions:
                st.subheader("Recommended Actions")
                for action in report.recommended_actions:
                    st.write(f"• {action}")

            if report.strong_matches:
                st.divider()
                st.subheader("Strong Matches")
                for match in report.strong_matches:
                    with st.expander(match.entity_name or match.title):
                        st.write(f"**URL:** {match.url}")
                        st.write(f"**Confidence:** {match.confidence_score:.0%}")
                        st.write(f"**Reasoning:** {match.reasoning}")
                        if match.mandate_signals:
                            st.write(f"**Signals:** {', '.join(match.mandate_signals)}")

            if report.flagged_for_review:
                st.divider()
                st.subheader("Flagged for Human Review")
                for item in report.flagged_for_review:
                    with st.expander(item.entity_name or item.title):
                        st.write(f"**URL:** {item.url}")
                        st.write(f"**Agent reasoning:** {item.reasoning}")

            with st.expander("Agent Decision Trace"):
                st.json(report.agent_decisions_log)

        if final_state.get("errors"):
            st.error(f"Errors encountered during this run: {final_state['errors']}")
