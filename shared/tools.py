"""
Shared tools for the Universal Multi-Agent Orchestrator.
Department-agnostic — all agents import from here.
"""
import os
import subprocess
import tempfile
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ─── Academic & Knowledge Search Tools ────────────────────────────────────────

def arxiv_search(query: str, max_results: int = 5) -> str:
    """
    Queries the arXiv API for scientific preprints and research papers.
    Returns paper title, authors, publication date, abstract, and PDF link.
    """
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    try:
        encoded_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"
        
        req = urllib.request.Request(url, headers={"User-Agent": "MultiAgentOrchestrator/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)

        if not entries:
            return f"No arXiv papers found for '{query}'."

        formatted_papers = []
        for i, entry in enumerate(entries, 1):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ') if entry.find('atom:title', ns) is not None else "Untitled"
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ') if entry.find('atom:summary', ns) is not None else "No abstract"
            published = entry.find('atom:published', ns).text[:10] if entry.find('atom:published', ns) is not None else "Unknown date"
            
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns) if author.find('atom:name', ns) is not None]
            authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

            # PDF Link
            pdf_link = ""
            for link in entry.findall('atom:link', ns):
                if link.attrib.get('title') == 'pdf' or link.attrib.get('type') == 'application/pdf':
                    pdf_link = link.attrib.get('href', '')
                    break
            if not pdf_link:
                paper_id = entry.find('atom:id', ns).text if entry.find('atom:id', ns) is not None else ""
                pdf_link = paper_id.replace('/abs/', '/pdf/') if paper_id else ""

            formatted_papers.append(
                f"[{i}] {title}\n"
                f"Authors: {authors_str} ({published})\n"
                f"PDF URL: {pdf_link}\n"
                f"Abstract: {summary[:500]}...\n"
            )

        return "\n".join(formatted_papers)
    except Exception as e:
        return f"arXiv search error: {str(e)}"


def wikipedia_search(query: str, max_results: int = 3) -> str:
    """
    Queries Wikipedia REST API for background knowledge and domain overviews.
    Returns article titles, clean introductory summaries, and article URLs.
    """
    import urllib.request
    import urllib.parse
    import json

    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit={max_results}"
        req = urllib.request.Request(search_url, headers={"User-Agent": "MultiAgentOrchestrator/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            search_data = json.loads(response.read().decode('utf-8'))

        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return f"No Wikipedia articles found for '{query}'."

        formatted_articles = []
        for i, res in enumerate(search_results, 1):
            page_title = res.get("title", "")
            page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title.replace(' ', '_'))}"

            # Fetch intro extract
            extract_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={urllib.parse.quote(page_title)}&format=json"
            ext_req = urllib.request.Request(extract_url, headers={"User-Agent": "MultiAgentOrchestrator/1.0"})
            with urllib.request.urlopen(ext_req, timeout=10) as ext_resp:
                ext_data = json.loads(ext_resp.read().decode('utf-8'))
                pages = ext_data.get("query", {}).get("pages", {})
                extract = ""
                for pid, pinfo in pages.items():
                    extract = pinfo.get("extract", "")
                    break

            snippet = extract[:600] if extract else res.get("snippet", "").replace('<span class="searchmatch">', '').replace('</span>', '')

            formatted_articles.append(
                f"[{i}] Wikipedia: {page_title}\n"
                f"URL: {page_url}\n"
                f"Summary: {snippet}...\n"
            )

        return "\n".join(formatted_articles)
    except Exception as e:
        return f"Wikipedia search error: {str(e)}"


def fetch_web_content(url: str, max_chars: int = 3000) -> str:
    """
    Fetches raw HTML/text content from a specific web page URL.
    """
    import urllib.request
    import re

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')

        # Clean HTML tags
        text = re.sub(r'<script font.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text[:max_chars] if text else "No text extracted."
    except Exception as e:
        return f"Failed to fetch web content from {url}: {str(e)}"


# ─── Web Search ───────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> str:
    """
    Primary: Tavily Search API (structured, reliable).
    Fallback: DuckDuckGo (no API key required).
    Returns a formatted string of search results.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")

    if tavily_key:
        try:
            from tavily import TavilyClient
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
            return "\n".join(formatted) if formatted else "No results found."
        except Exception as e:
            pass  # Fall through to DuckDuckGo

    # Fallback: DuckDuckGo
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        tool = DuckDuckGoSearchRun()
        return tool.invoke(query)
    except Exception as e:
        return f"Search failed: {str(e)}"


# ─── Code Execution Sandbox ───────────────────────────────────────────────────

def execute_code_local(code: str, timeout: int = 30) -> dict:
    """
    Executes Python code in a local subprocess sandbox.
    Returns dict with stdout, stderr, and success flag.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds.",
            "success": False,
            "returncode": -1,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "success": False,
            "returncode": -1,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def execute_code(code: str, timeout: int = 30) -> dict:
    """
    Try E2B first (secure cloud sandbox), fall back to local.
    """
    e2b_key = os.getenv("E2B_API_KEY")
    if e2b_key:
        try:
            from e2b_code_interpreter import Sandbox
            os.environ["E2B_API_KEY"] = e2b_key
            with Sandbox(timeout=timeout) as sandbox:
                execution = sandbox.run_code(code)
                error_msg = str(execution.error) if execution.error else ""
                return {
                    "stdout": "\n".join([str(log) for log in execution.logs]) if execution.logs else "",
                    "stderr": error_msg,
                    "success": not bool(execution.error),
                    "returncode": 1 if execution.error else 0,
                }
        except Exception:
            pass

    return execute_code_local(code, timeout)
