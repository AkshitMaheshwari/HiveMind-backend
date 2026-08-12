"""
Code execution tools — sandboxed Python execution via E2B (cloud) or subprocess (local).
"""
import logging
import os
import subprocess
import tempfile
from typing import Dict

logger = logging.getLogger(__name__)

# Backend root for resolving relative paths in generated code
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def execute_code_local(code: str, timeout: int = 90) -> Dict[str, object]:
    """
    Execute Python code in a local subprocess sandbox.

    Parameters:
        code: Python source code to execute.
        timeout: Maximum seconds to wait for execution (default 30).

    Returns:
        A dict with keys:
        - ``stdout`` (str): Standard output from the process.
        - ``stderr`` (str): Standard error / traceback.
        - ``success`` (bool): ``True`` if the process exited with code 0.
        - ``returncode`` (int): The process exit code.
    """
    if not code or not code.strip():
        return {"stdout": "", "stderr": "No code provided.", "success": False, "returncode": -1}

    tmp_path: str = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_BACKEND_ROOT,  # Ensure relative paths like data/uploads/{user_id}/ resolve
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        logger.warning("execute_code_local timed out after %ds", timeout)
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds.",
            "success": False,
            "returncode": -1,
        }
    except OSError as exc:
        logger.error("execute_code_local OS error: %s", exc)
        return {"stdout": "", "stderr": str(exc), "success": False, "returncode": -1}
    except Exception as exc:
        logger.error("execute_code_local unexpected error: %s", exc, exc_info=True)
        return {"stdout": "", "stderr": str(exc), "success": False, "returncode": -1}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def execute_code(code: str, timeout: int = 30) -> Dict[str, object]:
    """
    Execute Python code — tries E2B cloud sandbox first, falls back to local subprocess.

    E2B is used when the ``E2B_API_KEY`` environment variable is set.
    Falls back to :func:`execute_code_local` if E2B is unavailable or fails.

    Parameters:
        code: Python source code to execute.
        timeout: Maximum seconds to allow for execution (default 30).

    Returns:
        A dict with keys ``stdout``, ``stderr``, ``success``, ``returncode``.
        See :func:`execute_code_local` for details.
    """
    e2b_key = os.getenv("E2B_API_KEY")

    if e2b_key:
        try:
            from e2b_code_interpreter import Sandbox  # type: ignore[import]
            os.environ["E2B_API_KEY"] = e2b_key
            with Sandbox(timeout=timeout) as sandbox:
                execution = sandbox.run_code(code)
                error_msg = str(execution.error) if execution.error else ""
                return {
                    "stdout": (
                        "\n".join(str(log) for log in execution.logs)
                        if execution.logs
                        else ""
                    ),
                    "stderr": error_msg,
                    "success": not bool(execution.error),
                    "returncode": 1 if execution.error else 0,
                }
        except ImportError:
            logger.warning(
                "execute_code: e2b_code_interpreter not installed, falling back to local"
            )
        except Exception as exc:
            logger.warning(
                "execute_code: E2B sandbox failed (%s), falling back to local execution", exc
            )

    return execute_code_local(code, timeout)
