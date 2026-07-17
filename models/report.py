from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

from .result import ClassifiedResult


class IntelligenceReport(BaseModel):
    """Final structured intelligence report produced by the agent."""

    mandate_summary: str
    session_id: str
    generated_at: datetime = Field(default_factory=datetime.now)
    total_searches_performed: int
    total_results_evaluated: int
    strong_matches: List[ClassifiedResult] = Field(default_factory=list)
    weak_matches: List[ClassifiedResult] = Field(default_factory=list)
    flagged_for_review: List[ClassifiedResult] = Field(default_factory=list)
    synthesis: str = Field(..., description="Synthesized intelligence narrative")
    recommended_actions: List[str] = Field(default_factory=list)
    cost_ceiling_hit: bool = Field(default=False)
    agent_decisions_log: List[dict] = Field(default_factory=list)
    errors_encountered: List[str] = Field(default_factory=list)
