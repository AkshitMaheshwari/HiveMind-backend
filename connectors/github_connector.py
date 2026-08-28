"""
GitHubConnector — Ingests repositories from GitHub into normalised Document objects.
Fetches repository tree recursively, filters non-text / build assets, and converts
source files into Document objects with repository metadata for RAG indexing.
"""
import logging
import re
from typing import Any, List, Optional
import httpx

from connectors.base import BaseConnector
from connectors.document import Document
from connectors.exceptions import ConnectorError

logger = logging.getLogger(__name__)

# Files / extensions to ignore during repository ingestion
IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".mp4", ".mov", ".avi", ".mp3", ".wav",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".pdf", ".exe", ".dll", ".so", ".dylib", ".bin",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".pyo", ".pyd", ".class", ".o", ".obj",
    ".lock", "-lock.json", ".min.js", ".min.css", ".map",
}

IGNORED_DIRECTORIES = {
    ".git", "node_modules", "vendor", "dist", "build", "out",
    ".next", ".nuxt", "__pycache__", ".venv", "venv", "env",
    ".idea", ".vscode", ".turbo", "coverage", ".pytest_cache",
}

MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB per source file max


class GitHubConnector(BaseConnector):
    """
    Ingests public and private GitHub repositories by fetching source code files
    and wrapping them into Document objects for the RAG ingestion pipeline.
    """

    def __init__(
        self,
        repo: Optional[str] = None,
        token: Optional[str] = None,
        branch: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        if repo is None:
            raise ValueError("GitHubConnector requires a 'repo' parameter (e.g. 'owner/repo').")
        
        self.repo = self._clean_repo_name(repo)
        self.token = token
        self.branch = branch or "main"
        self.user_id = user_id or "default_user"
        self.source_type = "github"

    @staticmethod
    def _clean_repo_name(raw: str) -> str:
        """Strip URLs and trailing slashes/git extensions into 'owner/repo' format."""
        cleaned = raw.strip()
        cleaned = re.sub(r"^https?://github\.com/", "", cleaned)
        cleaned = re.sub(r"\.git$", "", cleaned)
        cleaned = cleaned.strip("/")
        parts = cleaned.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return cleaned

    def _get_headers(self) -> dict:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Universal-MultiAgent-Connector/1.0",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _should_include_file(self, path: str, size: int) -> bool:
        """Check if file should be ingested based on path, size, and extension."""
        parts = path.split("/")
        for part in parts:
            if part in IGNORED_DIRECTORIES or part.startswith("."):
                if part not in [".env.example", ".gitignore"]:
                    return False

        ext = "." + path.split(".")[-1].lower() if "." in path else ""
        if ext in IGNORED_EXTENSIONS:
            return False

        if size > MAX_FILE_SIZE_BYTES:
            return False

        return True

    async def ingest_async(
        self,
        max_files: int = 100,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Fetch repo tree and files asynchronously, returning a list of Document objects.
        """
        uid = user_id or self.user_id
        self.validate_user_id(uid)

        base_api = f"https://api.github.com/repos/{self.repo}"
        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=30.0) as client:
            tree_url = f"{base_api}/git/trees/{self.branch}?recursive=1"
            res = await client.get(tree_url, headers=headers)

            # If default branch wasn't 'main', try 'master'
            if res.status_code == 404 and self.branch == "main":
                self.branch = "master"
                tree_url = f"{base_api}/git/trees/master?recursive=1"
                res = await client.get(tree_url, headers=headers)

            if res.status_code == 401:
                raise ConnectorError("GitHub authentication failed. Please provide a valid Personal Access Token.")
            if res.status_code == 404:
                raise ConnectorError(f"GitHub repository '{self.repo}' or branch '{self.branch}' not found.")
            if res.status_code != 200:
                raise ConnectorError(f"Failed to fetch repository tree: {res.status_code} {res.text}")

            tree_data = res.json().get("tree", [])
            candidate_files = [
                item for item in tree_data
                if item.get("type") == "blob" and self._should_include_file(item.get("path", ""), item.get("size", 0))
            ][:max_files]

            if not candidate_files:
                raise ConnectorError(f"No indexable code files found in repository '{self.repo}'.")

            documents: List[Document] = []

            for item in candidate_files:
                path = item.get("path", "")
                sha = item.get("sha", "")
                raw_url = f"https://raw.githubusercontent.com/{self.repo}/{self.branch}/{path}"

                file_res = await client.get(raw_url, headers=headers)
                if file_res.status_code == 200 and file_res.text.strip():
                    doc = Document(
                        text=file_res.text,
                        source_type="github",
                        source_identifier=f"{self.repo}/{path}",
                        user_id=uid,
                        metadata={
                            "repo": self.repo,
                            "branch": self.branch,
                            "path": path,
                            "sha": sha,
                            "size": item.get("size", len(file_res.text)),
                        },
                    )
                    documents.append(doc)

            if not documents:
                raise ConnectorError(f"Could not extract text content from repository '{self.repo}'.")

            logger.info("Successfully fetched %d files from GitHub repo %s", len(documents), self.repo)
            return documents

    def ingest(self, **kwargs: Any) -> List[Document]:
        """
        Synchronous wrapper for BaseConnector contract.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.ingest_async(**kwargs))
            else:
                return loop.run_until_complete(self.ingest_async(**kwargs))
        except RuntimeError:
            return asyncio.run(self.ingest_async(**kwargs))

