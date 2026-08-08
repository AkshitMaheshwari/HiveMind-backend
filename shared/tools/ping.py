"""
Ping tool — a no-op demonstration tool used to verify that adding a new tool
to the system requires exactly one new file (this one) plus one registration
call in ``registry_bootstrap.py``.

This is intentionally trivial. Its purpose is to prove the extensibility
contract: no existing code changes are needed when adding a new tool.
"""
import logging

logger = logging.getLogger(__name__)


def ping_tool(message: str = "ping") -> str:
    """
    Echo a message back as a health-check / demonstration tool.

    This is a no-op tool that always succeeds. It is registered in all
    department tags to demonstrate cross-department tool sharing.

    Parameters:
        message: Optional message to echo back (default ``"ping"``).

    Returns:
        The string ``"pong: <message>"``.
    """
    logger.debug("ping_tool called with message=%r", message)
    return f"pong: {message}"
