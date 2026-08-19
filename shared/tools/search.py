"""
Search tools: web search, arXiv academic search, Wikipedia, and raw web fetch.

These are the concrete implementations. They are registered into the
ToolRegistry by ``shared/tools/registry_bootstrap.py`` at startup.
"""
import logging
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

logger = logging.getLogger(__name__)


# ─── arXiv ───────────────────────────────────────────────────────────────────

def arxiv_search(query: str, max_results: int = 5) -> str:
    """
    Query the arXiv API for scientific preprints and research papers.

    Parameters:
        query: Natural-language research query.
        max_results: Maximum number of papers to return (default 5).

    Returns:
        A formatted string containing paper title, authors, publication date,
        abstract snippet, and PDF link for each result. Returns an error
        description string on network or parsing failure — never raises.
    """
    if not query or not query.strip():
        logger.warning("arxiv_search called with empty query")
        return "arXiv search error: query must not be empty."

    try:
        encoded_query = urllib.parse.quote(query)
        url = (
            f"http://export.arxiv.org/api/query"
            f"?search_query=all:{encoded_query}"
            f"&start=0&max_results={max_results}"
        )

        req = urllib.request.Request(url, headers={"User-Agent": "MultiAgentOrchestrator/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        if not entries:
            return f"No arXiv papers found for '{query}'."

        formatted_papers = []
        for i, entry in enumerate(entries, 1):
            title_el = entry.find("atom:title", ns)
            title = title_el.text.strip().replace("\n", " ") if title_el is not None else "Untitled"

            summary_el = entry.find("atom:summary", ns)
            summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None else "No abstract"

            published_el = entry.find("atom:published", ns)
            published = published_el.text[:10] if published_el is not None else "Unknown date"

            authors = [
                author.find("atom:name", ns).text
                for author in entry.findall("atom:author", ns)
                if author.find("atom:name", ns) is not None
            ]
            authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

            pdf_link = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_link = link.attrib.get("href", "")
                    break
            if not pdf_link:
                id_el = entry.find("atom:id", ns)
                paper_id = id_el.text if id_el is not None else ""
                pdf_link = paper_id.replace("/abs/", "/pdf/") if paper_id else ""

            formatted_papers.append(
                f"[{i}] {title}\n"
                f"Authors: {authors_str} ({published})\n"
                f"PDF URL: {pdf_link}\n"
                f"Abstract: {summary[:500]}...\n"
            )

        return "\n".join(formatted_papers)

    except urllib.error.URLError as exc:
        logger.error("arxiv_search network error: query=%r error=%s", query, exc)
        return f"arXiv search network error: {exc}"
    except ET.ParseError as exc:
        logger.error("arxiv_search XML parse error: query=%r error=%s", query, exc)
        return f"arXiv search parse error: {exc}"
    except Exception as exc:
        logger.error("arxiv_search unexpected error: query=%r error=%s", query, exc, exc_info=True)
        return f"arXiv search error: {exc}"


# ─── Wikipedia ────────────────────────────────────────────────────────────────

def wikipedia_search(query: str, max_results: int = 3) -> str:
    """
    Query the Wikipedia REST API for background knowledge and domain overviews.

    Parameters:
        query: Topic to search for on Wikipedia.
        max_results: Maximum number of articles to return (default 3).

    Returns:
        A formatted string of article title, URL, and introductory summary
        for each result. Returns an error description string on failure.
    """
    import json

    if not query or not query.strip():
        logger.warning("wikipedia_search called with empty query")
        return "Wikipedia search error: query must not be empty."

    try:
        search_url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search"
            f"&srsearch={urllib.parse.quote(query)}"
            f"&format=json&srlimit={max_results}"
        )
        req = urllib.request.Request(
            search_url, headers={"User-Agent": "MultiAgentOrchestrator/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            search_data = json.loads(response.read().decode("utf-8"))

        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return f"No Wikipedia articles found for '{query}'."

        formatted_articles = []
        for i, res in enumerate(search_results, 1):
            page_title = res.get("title", "")
            page_url = (
                f"https://en.wikipedia.org/wiki/"
                f"{urllib.parse.quote(page_title.replace(' ', '_'))}"
            )

            # Fetch introductory extract
            extract_url = (
                f"https://en.wikipedia.org/w/api.php"
                f"?action=query&prop=extracts&exintro&explaintext"
                f"&titles={urllib.parse.quote(page_title)}&format=json"
            )
            ext_req = urllib.request.Request(
                extract_url, headers={"User-Agent": "MultiAgentOrchestrator/1.0"}
            )
            with urllib.request.urlopen(ext_req, timeout=10) as ext_resp:
                ext_data = json.loads(ext_resp.read().decode("utf-8"))
                pages = ext_data.get("query", {}).get("pages", {})
                extract = ""
                for _pid, pinfo in pages.items():
                    extract = pinfo.get("extract", "")
                    break

            snippet = (
                extract[:600]
                if extract
                else res.get("snippet", "")
                .replace('<span class="searchmatch">', "")
                .replace("</span>", "")
            )

            formatted_articles.append(
                f"[{i}] Wikipedia: {page_title}\n"
                f"URL: {page_url}\n"
                f"Summary: {snippet}...\n"
            )

        return "\n".join(formatted_articles)

    except urllib.error.URLError as exc:
        logger.error("wikipedia_search network error: query=%r error=%s", query, exc)
        return f"Wikipedia search network error: {exc}"
    except Exception as exc:
        logger.error("wikipedia_search unexpected error: query=%r error=%s", query, exc, exc_info=True)
        return f"Wikipedia search error: {exc}"


# ─── Web Search ───────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily (primary) with DuckDuckGo as fallback.

    Parameters:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A formatted string of numbered results with title, URL, and snippet.
        Returns an error description string if both providers fail.
    """
    if not query or not query.strip():
        logger.warning("web_search called with empty query")
        return "Web search error: query must not be empty."

    tavily_key = os.getenv("TAVILY_API_KEY")

    if tavily_key:
        try:
            from tavily import TavilyClient  # type: ignore[import]
            client = TavilyClient(api_key=tavily_key)
            response = client.search(query=query, max_results=max_results)
            results = response.get("results", [])
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"[{i}] {r.get('title', 'No title')}\n"
                    f"URL: {r.get('url', '')}\n"
                    f"Snippet: {r.get('content', '')[:400]}\n"
                )
            if formatted:
                return "\n".join(formatted)
            logger.warning("web_search Tavily returned no results: query=%r", query)
        except Exception as exc:
            logger.warning(
                "web_search Tavily failed, falling back to DuckDuckGo: query=%r error=%s",
                query,
                exc,
            )

    # Fallback: DuckDuckGo with strict timeout
    try:
        import concurrent.futures
        from langchain_community.tools import DuckDuckGoSearchRun  # type: ignore[import]

        def _do_ddg():
            tool = DuckDuckGoSearchRun()
            return tool.invoke(query)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_ddg)
            return future.result(timeout=6)
    except concurrent.futures.TimeoutError:
        logger.warning(f"web_search DuckDuckGo timed out after 6s for query: {query}")
        return f"Market web search timed out. Proceeding with LLM domain knowledge for '{query}'."
    except Exception as exc:
        logger.error("web_search DuckDuckGo fallback failed: query=%r error=%s", query, exc)
        return f"Web search unavailable: {exc}"


# ─── Raw web content ──────────────────────────────────────────────────────────

def fetch_web_content(url: str, max_chars: int = 3000) -> str:
    """
    Fetch and extract plain text from a specific web page URL.

    Strips HTML tags, scripts, and style blocks, returning clean text.

    Parameters:
        url: The web page URL to fetch.
        max_chars: Maximum characters to return from the extracted text (default 3000).

    Returns:
        The extracted text content, truncated to ``max_chars``.
        Returns an error description string on network or parse failure.
    """
    import re

    if not url or not url.strip():
        return "fetch_web_content error: URL must not be empty."

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")

        # Strip scripts, styles, tags
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text[:max_chars] if text else "No text extracted from page."

    except urllib.error.URLError as exc:
        logger.error("fetch_web_content network error: url=%r error=%s", url, exc)
        return f"Failed to fetch web content from {url}: {exc}"
    except Exception as exc:
        logger.error("fetch_web_content error: url=%r error=%s", url, exc, exc_info=True)
        return f"Failed to fetch web content from {url}: {exc}"
