from typing import Annotated, List, Optional, TypedDict

from langgraph.graph.message import add_messages

from models.mandate import InvestorMandate
from models.report import IntelligenceReport
from models.result import ClassifiedResult


class AgentState(TypedDict):
    """
    Central state object passed through every node in the agent graph.
    Maintains clean separation between reasoning, retrieval, memory,
    and presentation layers. All mutations are explicit - no hidden state.
    """
    # Input
    mandate: InvestorMandate
    session_id: str

    # Decomposition
    search_queries: List[str]
    decomposition_reasoning: str

    # Search + Classification
    raw_results: List[dict]
    classified_results: List[ClassifiedResult]

    # Control flow
    iterations_used: int
    cost_ceiling_hit: bool
    ambiguous_items_pending: List[ClassifiedResult]
    human_validation_required: bool

    # Output
    report: Optional[IntelligenceReport]

    # Observability
    decision_trace: List[dict]
    errors: List[str]

    # LLM message history
    messages: Annotated[list, add_messages]
