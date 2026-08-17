"""
Tool Registry Bootstrap — registers all tools into the global registry in
a deterministic, import-order-independent sequence.

**Call this exactly once at application startup** (from ``api/main.py``).
Never rely on module import order to trigger registrations.

To add a new tool:
1. Create your tool function in a new file under ``shared/tools/``.
2. Add a single ``registry.register(ToolSpec(...))`` call here.
That is all — no other file needs to change.
"""
import logging

from shared.tool_registry import ToolSpec, registry

logger = logging.getLogger(__name__)


def bootstrap() -> None:
    """
    Register every tool into the global ``registry`` singleton.

    This function is idempotent with respect to errors — if a tool is already
    registered (e.g. bootstrap called twice), ``ToolAlreadyRegisteredError``
    will surface immediately so the problem is caught at startup.

    Raises:
        ToolAlreadyRegisteredError: If any tool name is duplicated.
        ValueError: If any ToolSpec has invalid fields.
    """
    logger.info("Tool registry bootstrap starting...")

    # ── Search tools ──────────────────────────────────────────────────────────
    from shared.tools.search import arxiv_search, wikipedia_search, web_search, fetch_web_content

    registry.register(ToolSpec(
        name="arxiv_search",
        description=(
            "Search arXiv for scientific papers and preprints. "
            "Returns paper titles, authors, publication dates, abstracts, and PDF links. "
            "Best for: academic research, scientific literature review, finding papers on ML/AI/physics/math."
        ),
        tags=["research"],
        fn=arxiv_search,
    ))

    registry.register(ToolSpec(
        name="wikipedia_search",
        description=(
            "Search Wikipedia for background knowledge, definitions, and domain context. "
            "Returns article summaries and URLs. "
            "Best for: foundational concepts, historical context, well-established facts."
        ),
        tags=["research"],
        fn=wikipedia_search,
    ))

    registry.register(ToolSpec(
        name="web_search",
        description=(
            "Search the live web using Tavily (primary) or DuckDuckGo (fallback). "
            "Returns current news, blog posts, documentation, and technical resources. "
            "Best for: real-time information, market data, recent events, product docs."
        ),
        tags=["research", "content"],
        fn=web_search,
    ))

    registry.register(ToolSpec(
        name="fetch_web_content",
        description=(
            "Fetch and extract plain text from a specific web page URL. "
            "Strips HTML, scripts, and styling. "
            "Best for: reading the full content of a known URL found in search results."
        ),
        tags=["research"],
        fn=fetch_web_content,
    ))

    # ── Code execution tools ──────────────────────────────────────────────────
    # Shared Code sandbox: tagged for ALL compute-heavy departments.
    # Code, Financial (TechnicalAnalysis, PortfolioAnalyst), Analytics, Strategy
    # all reuse this single tool — this shared reuse is a core hive-mind property.
    from shared.tools.code import execute_code, execute_code_local

    registry.register(ToolSpec(
        name="execute_code",
        description=(
            "Execute Python code in a sandboxed environment (E2B cloud or local subprocess). "
            "Returns stdout, stderr, and a success flag. "
            "Best for: running generated code, testing scripts, data analysis, financial modeling, "
            "statistical calculations, chart data generation, algorithmic output verification."
        ),
        tags=["code", "financial", "analytics", "strategy"],
        fn=execute_code,
    ))

    registry.register(ToolSpec(
        name="execute_code_local",
        description=(
            "Execute Python code in a local subprocess sandbox (no cloud dependency). "
            "Returns stdout, stderr, and a success flag. "
            "Use when E2B is unavailable or for low-risk script execution."
        ),
        tags=["code", "financial", "analytics", "strategy"],
        fn=execute_code_local,
    ))

    # ── RAG retrieval tool ────────────────────────────────────────────────────
    # HIVE MIND PROPERTY: Every specialist agent in any department can call
    # retrieve_from_knowledge_base — knowledge is NOT siloed per department.
    # Financial agents can search uploaded filings; Legal agents can search
    # uploaded contracts; Strategy agents can search market reports, etc.
    from shared.tools.rag_retrieval import rag_document_search

    registry.register(ToolSpec(
        name="rag_document_search",
        description=(
            "Search the authenticated user's uploaded documents using semantic vector similarity. "
            "Returns relevant excerpts with source attribution. "
            "Strictly scoped to the requesting user's documents — never returns other users' data. "
            "Best for: answering questions about documents the user has uploaded (PDFs, Excel files, "
            "CSVs, contracts, financial filings, research reports, or any private knowledge base)."
        ),
        tags=["research", "content", "code", "financial", "analytics", "strategy", "legal", "sales", "document"],
        fn=rag_document_search,
    ))

    # ── Image generation tool ─────────────────────────────────────────────────
    # Exclusively for the Design Department. The only genuinely new tool category
    # in the Tier 2 additions — every other Tier 2 dept reuses existing tools.
    try:
        from shared.tools.image_generation import generate_image

        registry.register(ToolSpec(
            name="generate_image",
            description=(
                "Generate an image using OpenAI DALL-E 3. "
                "Returns the URL of the generated image. "
                "Best for: logo concepts, brand mockups, pitch deck visuals, icon generation, "
                "product renders, and any visual design asset creation."
            ),
            tags=["design"],
            fn=generate_image,
        ))
    except Exception as _img_exc:
        logger.warning("generate_image tool not registered (missing openai or key): %s", _img_exc)

    # ── Demo / utility tools ──────────────────────────────────────────────────
    from shared.tools.ping import ping_tool

    registry.register(ToolSpec(
        name="ping",
        description=(
            "No-op health-check tool. Returns 'pong: <message>'. "
            "Used to verify the tool registry is functioning correctly."
        ),
        tags=["research", "content", "code"],
        fn=ping_tool,
    ))

    total = len(registry)
    logger.info(
        "Tool registry bootstrap complete: %d tools registered: %s",
        total,
        [spec.name for spec in registry.list_all()],
    )
