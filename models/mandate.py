from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AssetClass(str, Enum):
    VENTURE = "venture_capital"
    PRIVATE_EQUITY = "private_equity"
    REAL_ESTATE = "real_estate"
    HEDGE_FUND = "hedge_fund"
    CREDIT = "private_credit"
    INFRASTRUCTURE = "infrastructure"
    MIXED = "mixed"


class InvestorMandate(BaseModel):
    """Structured representation of an investor mandate for intelligence research."""

    raw_input: str = Field(..., description="Natural language mandate from the user")
    investor_type: Optional[str] = Field(None, description="Investor type (family office, LP, endowment)")
    asset_classes: Optional[List[AssetClass]] = Field(None, description="Target asset classes")
    geographies: Optional[List[str]] = Field(None, description="Target geographies")
    stage_focus: Optional[str] = Field(None, description="Investment stage (Series B, buyout, etc.)")
    sector_focus: Optional[List[str]] = Field(None, description="Target sectors")
    check_size: Optional[str] = Field(None, description="Typical check size range")
    max_search_iterations: int = Field(default=5, ge=1, le=20, description="Cost ceiling: max search rounds")
    max_results_per_query: int = Field(default=8, ge=1, le=20, description="Action ceiling: results per query")
