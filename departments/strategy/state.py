"""
Business Strategy Department State.
"""
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from shared.state_utils import merge_events


class StrategyDeptState(TypedDict):
    """State for the Business Strategy Department subgraph."""
    # ── Input ──────────────────────────────────────────────────────
    task: str
    required_agents: List[str]          # from StrategyRouterAgent
    user_id: Optional[str]
    api_keys: Optional[Dict[str, str]]
    selected_model: Optional[str]

    # ── Intermediate outputs ────────────────────────────────────────
    market_research: str                # from MarketAnalystAgent (via inter_dept)
    competitor_data: str                # from CompetitorAnalystAgent
    financial_model: str                # from FinancialModelerAgent
    swot_analysis: Dict[str, Any]       # from SWOTAgent
    business_plan: str                  # from BusinessPlanAgent
    pitch_deck: str                     # from PitchDeckAgent

    # ── Output ─────────────────────────────────────────────────────
    final_strategy: str                 # from StrategySynthesizerAgent

    # ── Streaming ──────────────────────────────────────────────────
    events: Annotated[List[Dict[str, Any]], merge_events]
