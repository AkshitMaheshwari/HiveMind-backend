"""
shared.tools — compatibility shim.

This file is preserved so that any existing code doing:
    ``from shared.tools import web_search``
or:
    ``import shared.tools``
continues to work without modification.

The canonical implementations now live in the ``shared/tools/`` package:
    - shared.tools.search   → web_search, arxiv_search, wikipedia_search, fetch_web_content
    - shared.tools.code     → execute_code, execute_code_local
    - shared.tools.rag_retrieval → rag_document_search
    - shared.tools.ping     → ping_tool

Do NOT add new tool implementations here. Add them to the appropriate
submodule and register them in shared/tools/registry_bootstrap.py.
"""

# Re-export everything from the package so ``from shared.tools import X`` works.
from shared.tools import (  # noqa: F401  (re-export)
    arxiv_search,
    fetch_web_content,
    web_search,
    wikipedia_search,
    execute_code,
    execute_code_local,
    rag_document_search,
    ping_tool,
)
