"""
Financial Department State
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class FinancialDeptState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    task: str
    original_request: str
    api_keys: Optional[Dict[str, str]]   # User-provided API keys
    selected_model: Optional[str]        # User's chosen model id

    # ── Internal pipeline ──────────────────────────────────────────
    required_agents: List[str]    # Agents to run, decided by Router
    tickers: List[str]            # Extracted stock tickers
    market_data: Dict[str, Any]   # Output from Market Data Agent
    fundamental_data: Dict[str, Any] # Output from Fundamental Analysis Agent
    technical_data: Dict[str, Any] # Output from Technical Analysis Agent
    news_sentiment: Dict[str, Any] # Output from News & Sentiment Agent
    portfolio_analysis: Dict[str, Any] # Output from Portfolio Analyst Agent
    comparison_data: Dict[str, Any] # Output from Comparative Analysis Agent

    # ── Output ─────────────────────────────────────────────────────
    final_report: str

    # ── Events ─────────────────────────────────────────────────────
    events: List[Dict[str, Any]]
