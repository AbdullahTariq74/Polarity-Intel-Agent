from langgraph.graph import END, StateGraph

from agent.nodes import (
    classify_results,
    decompose_mandate,
    execute_searches,
    human_validation_checkpoint,
    synthesize_report,
)
from agent.state import AgentState


def _route_after_classification(state: AgentState) -> str:
    """
    Explicit routing decision after classification.
    If ambiguous items exist and human validation is required, route to
    the checkpoint node. Otherwise proceed directly to synthesis.
    """
    if state.get("human_validation_required") and state.get("ambiguous_items_pending"):
        return "validate"
    return "synthesize"


def build_graph() -> StateGraph:
    """
    Construct the agent graph with explicit node registration,
    deterministic edges, and conditional routing for ambiguous cases.
    """
    graph = StateGraph(AgentState)

    graph.add_node("decompose", decompose_mandate)
    graph.add_node("search", execute_searches)
    graph.add_node("classify", classify_results)
    graph.add_node("validate", human_validation_checkpoint)
    graph.add_node("synthesize", synthesize_report)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "search")
    graph.add_edge("search", "classify")

    graph.add_conditional_edges(
        "classify",
        _route_after_classification,
        {"validate": "validate", "synthesize": "synthesize"}
    )

    graph.add_edge("validate", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


agent_graph = build_graph()
