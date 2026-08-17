"""
Inter-Department Direct Call Module
====================================
Allows specialist agents to call other departments directly — without routing
through the CEO. This is what separates a hive mind from a simple router:

  - Strategy's MarketAnalystAgent calls call_research_dept() directly
  - Strategy's FinancialModelerAgent calls call_analytics_dept() directly
  - Legal's ComplianceChecklistAgent calls call_research_dept() directly
  - Sales's LeadResearchAgent calls call_research_dept() directly

None of these round-trip through the CEO. The CEO only handles the initial
routing decision; after that, departments collaborate peer-to-peer.

Usage::

    from shared.inter_dept import call_research_dept, call_analytics_dept

    result = await call_research_dept(
        task="What is the current market size of B2B SaaS in APAC?",
        api_keys=api_keys,
        selected_model=selected_model,
    )
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


async def call_research_dept(
    task: str,
    api_keys: Optional[Dict[str, str]] = None,
    selected_model: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Call the Research Department directly for a sub-task.

    Runs the full Research subgraph (Router → Sources → FactChecker → Synthesizer)
    and returns the synthesized research text.

    Parameters:
        task: The research question or topic to investigate.
        api_keys: Optional per-user API key overrides.
        selected_model: Optional model override.
        user_id: Optional user ID for scoped RAG retrieval.

    Returns:
        Synthesized research text, or an error message string on failure.
    """
    try:
        from departments.research.graph import research_subgraph

        initial_state = {
            "task": task,
            "original_request": task,
            "user_id": user_id,
            "api_keys": api_keys,
            "selected_model": selected_model,
            "events": [],
        }

        logger.info("inter_dept: calling Research dept for task=%r", task[:80])
        final_state = await research_subgraph.ainvoke(initial_state)
        result = final_state.get("final_research", "")
        logger.info("inter_dept: Research dept returned %d chars", len(result))
        return result or f"Research completed for: {task}"

    except Exception as exc:
        logger.error("inter_dept: Research dept call failed: %s", exc)
        return f"Research lookup failed: {exc}"


async def call_analytics_dept(
    task: str,
    data_source: str = "",
    api_keys: Optional[Dict[str, str]] = None,
    selected_model: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Call the Analytics Department directly for data analysis.

    Runs the Analytics subgraph and returns structured insights text.

    Parameters:
        task: The analysis question or objective.
        data_source: Optional CSV/JSON string of data to analyze.
        api_keys: Optional per-user API key overrides.
        selected_model: Optional model override.
        user_id: Optional user ID for scoped RAG retrieval of uploaded CSVs.

    Returns:
        Analysis insights text, or an error message string on failure.
    """
    try:
        from departments.analytics.graph import analytics_subgraph

        initial_state = {
            "task": task,
            "data_source": data_source,
            "user_id": user_id,
            "api_keys": api_keys,
            "selected_model": selected_model,
            "events": [],
        }

        logger.info("inter_dept: calling Analytics dept for task=%r", task[:80])
        final_state = await analytics_subgraph.ainvoke(initial_state)
        result = final_state.get("analysis_result", "")
        logger.info("inter_dept: Analytics dept returned %d chars", len(result))
        return result or f"Analytics completed for: {task}"

    except Exception as exc:
        logger.error("inter_dept: Analytics dept call failed: %s", exc)
        return f"Analytics lookup failed: {exc}"


async def call_financial_dept(
    task: str,
    api_keys: Optional[Dict[str, str]] = None,
    selected_model: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Call the Financial Department directly for market/fundamental data.

    Parameters:
        task: The financial analysis question.
        api_keys: Optional per-user API key overrides.
        selected_model: Optional model override.
        user_id: Optional user ID for scoped RAG retrieval of uploaded filings.

    Returns:
        Financial analysis text, or an error message string on failure.
    """
    try:
        from departments.financial.graph import financial_subgraph

        initial_state = {
            "task": task,
            "original_request": task,
            "api_keys": api_keys,
            "selected_model": selected_model,
            "events": [],
        }

        logger.info("inter_dept: calling Financial dept for task=%r", task[:80])
        final_state = await financial_subgraph.ainvoke(initial_state)
        result = final_state.get("final_report", "")
        logger.info("inter_dept: Financial dept returned %d chars", len(result))
        return result or f"Financial analysis completed for: {task}"

    except Exception as exc:
        logger.error("inter_dept: Financial dept call failed: %s", exc)
        return f"Financial lookup failed: {exc}"
