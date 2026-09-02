"""
GitHub Tools for Universal Multi-Agent System.
Provides live repository operations: tree listing, reading source files,
branch management, file commits, and Pull Request creation.
"""
import base64
import logging
import os
import re
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)

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

MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB max


def _clean_repo_name(raw: str) -> str:
    """Normalize user input to 'owner/repo' format."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^https?://github\.com/", "", cleaned)
    cleaned = re.sub(r"\.git$", "", cleaned)
    cleaned = cleaned.strip("/")
    parts = cleaned.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return cleaned


def _get_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Build standard GitHub API headers with optional authentication."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Universal-MultiAgent-System/1.0",
    }
    tok = token or os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
    if tok:
        tok = tok.strip()
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def _should_include_file(path: str, size: int = 0) -> bool:
    """Filter out build artifacts, package managers, and binaries."""
    parts = path.split("/")
    for part in parts:
        if part in IGNORED_DIRECTORIES:
            return False
        if part.startswith(".") and part not in [".env.example", ".gitignore", ".env"]:
            return False

    ext = "." + path.split(".")[-1].lower() if "." in path else ""
    if ext in IGNORED_EXTENSIONS:
        return False

    if size > MAX_FILE_SIZE_BYTES:
        return False

    return True


async def github_list_user_repos(token: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch all repositories accessible to the user with the given token.
    Returns list of dicts with repository metadata.
    """
    headers = _get_headers(token)
    url = "https://api.github.com/user/repos?sort=updated&per_page=100&type=all"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.get(url, headers=headers)
            if res.status_code == 200:
                repos = res.json()
                return [
                    {
                        "name": r.get("name"),
                        "full_name": r.get("full_name"),
                        "private": r.get("private", False),
                        "description": r.get("description") or "",
                        "html_url": r.get("html_url"),
                        "default_branch": r.get("default_branch", "main"),
                    }
                    for r in repos
                ]
            else:
                return [{"error": f"GitHub API error ({res.status_code}): {res.text}"}]
    except Exception as e:
        logger.error("github_list_user_repos error: %s", e)
        return [{"error": str(e)}]


async def github_list_files(
    repo: str,
    token: Optional[str] = None,
    branch: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch the recursive file tree for a repository from GitHub API.
    Returns a list of items with 'path', 'type' ('blob'/'tree'), 'size', and 'sha'.
    """
    repo_clean = _clean_repo_name(repo)
    headers = _get_headers(token)
    branches_to_try = [branch] if branch else ["main", "master"]

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            # 1. Try finding working branch
            tree_data = None
            for b in branches_to_try:
                tree_url = f"https://api.github.com/repos/{repo_clean}/git/trees/{b}?recursive=1"
                res = await client.get(tree_url, headers=headers)
                if res.status_code == 200:
                    tree_data = res.json().get("tree", [])
                    break
                elif res.status_code == 401:
                    return [{"error": "GitHub authentication failed. Please check your Personal Access Token in Settings."}]

            # 2. If neither main nor master worked, query repo metadata for default_branch
            if tree_data is None:
                repo_url = f"https://api.github.com/repos/{repo_clean}"
                repo_res = await client.get(repo_url, headers=headers)
                if repo_res.status_code == 200:
                    default_b = repo_res.json().get("default_branch", "main")
                    tree_url = f"https://api.github.com/repos/{repo_clean}/git/trees/{default_b}?recursive=1"
                    res = await client.get(tree_url, headers=headers)
                    if res.status_code == 200:
                        tree_data = res.json().get("tree", [])
                elif repo_res.status_code == 404:
                    return [{"error": f"Repository '{repo_clean}' not found. Make sure the name is correct and token has access."}]
                elif repo_res.status_code == 401:
                    return [{"error": "GitHub authentication failed. Please verify your token."}]

            if tree_data is None:
                return [{"error": f"Could not fetch tree for repository '{repo_clean}'."}]

            filtered = [
                {
                    "path": item.get("path"),
                    "type": item.get("type"),
                    "size": item.get("size", 0),
                    "sha": item.get("sha"),
                }
                for item in tree_data
                if item.get("type") == "blob" and _should_include_file(item.get("path", ""), item.get("size", 0))
            ]
            return filtered

    except Exception as e:
        logger.error("github_list_files error for repo %s: %s", repo, e)
        return [{"error": str(e)}]


async def github_read_file(
    repo: str,
    path: str,
    token: Optional[str] = None,
    branch: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Read the contents of a specific file in a GitHub repository.
    Returns dict with 'content', 'path', 'sha', 'size'.
    """
    repo_clean = _clean_repo_name(repo)
    headers = _get_headers(token)

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            ref_param = f"?ref={branch}" if branch else ""
            url = f"https://api.github.com/repos/{repo_clean}/contents/{path}{ref_param}"
            res = await client.get(url, headers=headers)

            if res.status_code == 200:
                data = res.json()
                if isinstance(data, dict) and "content" in data:
                    encoding = data.get("encoding", "")
                    raw_content = data.get("content", "")
                    if encoding == "base64":
                        decoded = base64.b64decode(raw_content).decode("utf-8", errors="replace")
                    else:
                        decoded = raw_content
                    return {
                        "content": decoded,
                        "path": path,
                        "sha": data.get("sha", ""),
                        "size": data.get("size", len(decoded)),
                    }

            # Fallback to raw githubusercontent
            branches = [branch] if branch else ["main", "master"]
            for b in branches:
                raw_url = f"https://raw.githubusercontent.com/{repo_clean}/{b}/{path}"
                raw_res = await client.get(raw_url, headers=headers)
                if raw_res.status_code == 200:
                    return {
                        "content": raw_res.text,
                        "path": path,
                        "sha": "",
                        "size": len(raw_res.text),
                    }

            if res.status_code == 404:
                return {"error": f"File '{path}' not found in repository '{repo_clean}'."}
            elif res.status_code == 401:
                return {"error": "GitHub authentication failed. Please check token permissions."}
            else:
                return {"error": f"Failed to read file ({res.status_code}): {res.text}"}

    except Exception as e:
        logger.error("github_read_file error for %s/%s: %s", repo, path, e)
        return {"error": str(e)}


async def github_create_branch(
    repo: str,
    branch_name: str,
    from_branch: str = "main",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new git branch from an existing base branch.
    """
    if not token and not os.getenv("GITHUB_TOKEN"):
        return {"error": "GitHub Personal Access Token is required to create a branch."}

    repo_clean = _clean_repo_name(repo)
    headers = _get_headers(token)

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            # Check if branch already exists
            check_url = f"https://api.github.com/repos/{repo_clean}/git/ref/heads/{branch_name}"
            check_res = await client.get(check_url, headers=headers)
            if check_res.status_code == 200:
                return {"status": "already_exists", "branch": branch_name}

            # Get base SHA
            base_sha = None
            for b in [from_branch, "main", "master"]:
                ref_url = f"https://api.github.com/repos/{repo_clean}/git/ref/heads/{b}"
                ref_res = await client.get(ref_url, headers=headers)
                if ref_res.status_code == 200:
                    base_sha = ref_res.json().get("object", {}).get("sha")
                    break

            if not base_sha:
                # Query repo default branch
                repo_res = await client.get(f"https://api.github.com/repos/{repo_clean}", headers=headers)
                if repo_res.status_code == 200:
                    def_branch = repo_res.json().get("default_branch", "main")
                    ref_res = await client.get(f"https://api.github.com/repos/{repo_clean}/git/ref/heads/{def_branch}", headers=headers)
                    if ref_res.status_code == 200:
                        base_sha = ref_res.json().get("object", {}).get("sha")

            if not base_sha:
                return {"error": f"Could not find base branch '{from_branch}' to branch from."}

            create_url = f"https://api.github.com/repos/{repo_clean}/git/refs"
            payload = {
                "ref": f"refs/heads/{branch_name}",
                "sha": base_sha,
            }
            res = await client.post(create_url, json=payload, headers=headers)
            if res.status_code in [200, 201]:
                return {"status": "success", "branch": branch_name, "sha": base_sha}
            else:
                return {"error": f"Failed to create branch ({res.status_code}): {res.text}"}

    except Exception as e:
        logger.error("github_create_branch error for %s: %s", repo, e)
        return {"error": str(e)}


async def github_create_or_update_file(
    repo: str,
    path: str,
    content: str,
    message: str,
    branch: str,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create or update a file in a repository on a specific branch.
    """
    if not token and not os.getenv("GITHUB_TOKEN"):
        return {"error": "GitHub Personal Access Token is required to commit code."}

    repo_clean = _clean_repo_name(repo)
    headers = _get_headers(token)

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            # Check if file exists on this branch to get current SHA
            sha = None
            check_url = f"https://api.github.com/repos/{repo_clean}/contents/{path}?ref={branch}"
            check_res = await client.get(check_url, headers=headers)
            if check_res.status_code == 200:
                sha = check_res.json().get("sha")

            # Base64 encode file content
            b64_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")

            put_url = f"https://api.github.com/repos/{repo_clean}/contents/{path}"
            payload = {
                "message": message,
                "content": b64_content,
                "branch": branch,
            }
            if sha:
                payload["sha"] = sha

            res = await client.put(put_url, json=payload, headers=headers)
            if res.status_code in [200, 201]:
                return {"status": "success", "path": path, "commit": res.json().get("commit", {})}
            else:
                return {"status": "failed", "error": f"Commit failed ({res.status_code}): {res.text}"}

    except Exception as e:
        logger.error("github_create_or_update_file error for %s/%s: %s", repo, path, e)
        return {"error": str(e)}


async def github_create_pull_request(
    repo: str,
    title: str,
    head: str,
    base: str = "main",
    body: str = "",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a Pull Request in the target repository.
    """
    if not token and not os.getenv("GITHUB_TOKEN"):
        return {"error": "GitHub Personal Access Token is required to open a pull request."}

    repo_clean = _clean_repo_name(repo)
    headers = _get_headers(token)

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            # Verify base branch or fallback
            base_branch = base
            repo_res = await client.get(f"https://api.github.com/repos/{repo_clean}", headers=headers)
            if repo_res.status_code == 200:
                def_branch = repo_res.json().get("default_branch", "main")
                if base == "main" and def_branch != "main":
                    base_branch = def_branch

            url = f"https://api.github.com/repos/{repo_clean}/pulls"
            payload = {
                "title": title,
                "head": head,
                "base": base_branch,
                "body": body,
            }

            res = await client.post(url, json=payload, headers=headers)
            if res.status_code in [200, 201]:
                return res.json()
            elif res.status_code == 422:
                # Check if PR already exists
                existing_res = await client.get(f"{url}?head={repo_clean.split('/')[0]}:{head}", headers=headers)
                if existing_res.status_code == 200 and existing_res.json():
                    return existing_res.json()[0]
                return {"error": f"PR creation returned 422: {res.text}"}
            else:
                return {"error": f"Failed to create PR ({res.status_code}): {res.text}"}

    except Exception as e:
        logger.error("github_create_pull_request error for %s: %s", repo, e)
        return {"error": str(e)}
