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
