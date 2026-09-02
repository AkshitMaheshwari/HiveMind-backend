"""
shared.tools package — public API.

Re-exports all tool functions so that existing imports continue to work
unchanged after the refactor from a flat module to a package.

Agents importing from ``shared.tools`` directly (e.g.
``from shared.tools import web_search``) will continue to work without
any modification.

The canonical implementations now live in submodules:
- ``shared.tools.search``   → web_search, arxiv_search, wikipedia_search, fetch_web_content
- ``shared.tools.code``     → execute_code, execute_code_local
- ``shared.tools.rag_retrieval`` → rag_document_search
- ``shared.tools.ping``     → ping_tool
"""

from shared.tools.search import (
    arxiv_search,
    fetch_web_content,
    web_search,
    wikipedia_search,
)
from shared.tools.code import (
    execute_code,
    execute_code_local,
)
from shared.tools.rag_retrieval import rag_document_search
from shared.tools.ping import ping_tool
from shared.tools.github_tools import (
    github_read_file,
    github_list_files,
    github_create_branch,
    github_create_or_update_file,
    github_create_pull_request,
    github_list_user_repos,
)
from shared.tools.gmail_tools import (
    gmail_list_messages,
    gmail_search_emails,
    gmail_read_message,
    gmail_read_thread,
    gmail_create_draft,
    gmail_send_email,
)

__all__ = [
    "arxiv_search",
    "fetch_web_content",
    "web_search",
    "wikipedia_search",
    "execute_code",
    "execute_code_local",
    "rag_document_search",
    "ping_tool",
    "github_read_file",
    "github_list_files",
    "github_create_branch",
    "github_create_or_update_file",
    "github_create_pull_request",
    "github_list_user_repos",
    "gmail_list_messages",
    "gmail_search_emails",
    "gmail_read_message",
    "gmail_read_thread",
    "gmail_create_draft",
    "gmail_send_email",
]
