from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ClassificationLabel(str, Enum):
    STRONG_MATCH = "strong_match"
    WEAK_MATCH = "weak_match"
    IRRELEVANT = "irrelevant"
    AMBIGUOUS = "ambiguous"


class ClassifiedResult(BaseModel):
    """A search result that has been classified against an investor mandate."""

    url: str
    title: str
    raw_content: str
    classification: ClassificationLabel
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explicit reasoning for this classification decision")
    entity_name: Optional[str] = Field(None, description="Extracted investor or entity name")
    mandate_signals: Optional[List[str]] = Field(default_factory=list, description="Specific mandate-matching signals found")
    contact_hints: Optional[List[str]] = Field(default_factory=list, description="Contact information signals")
    requires_human_review: bool = Field(default=False)
    source_query: Optional[str] = Field(None, description="The query that produced this result")
