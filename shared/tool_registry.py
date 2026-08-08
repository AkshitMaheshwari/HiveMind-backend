"""
Tool Registry — central registry for all agent tools in the Multi-Agent Orchestrator.

Every tool registers itself here with a unique name, description, tags, and callable.
Agents retrieve tool lists by tag rather than importing tools directly.

Usage::

    from shared.tool_registry import registry, ToolSpec

    # Registration (done once at startup via registry_bootstrap)
    registry.register(ToolSpec(
        name="my_tool",
        description="Does something useful",
        tags=["research"],
        fn=my_tool_function,
    ))

    # Lookup
    spec = registry.get("my_tool")
    result = spec.fn(query="hello")
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Exceptions ───────────────────────────────────────────────────────────────

class ToolRegistryError(Exception):
    """Base class for all tool registry errors."""


class ToolAlreadyRegisteredError(ToolRegistryError):
    """
    Raised when a tool with the same name is registered twice.

    This is a hard failure at startup — duplicate names indicate either a
    copy-paste error or an unintended import ordering issue.
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            f"Tool '{name}' is already registered. Each tool name must be unique. "
            "Check registry_bootstrap.py for duplicate registration calls."
        )
        self.tool_name = name


class ToolNotFoundError(ToolRegistryError):
    """
    Raised when a requested tool name does not exist in the registry.

    Never returns None silently — callers must handle this exception if the
    tool name might be absent.
    """

    def __init__(self, name: str, available: Optional[List[str]] = None) -> None:
        hint = ""
        if available:
            hint = f" Available tools: {', '.join(sorted(available))}"
        super().__init__(
            f"Tool '{name}' is not registered.{hint}"
        )
        self.tool_name = name


# ─── ToolSpec ─────────────────────────────────────────────────────────────────

@dataclass
class ToolSpec:
    """
    The required interface that every registered tool must conform to.

    Attributes:
        name: Unique identifier for this tool across the entire registry.
              Must be a non-empty string containing only [a-z0-9_].
        description: Human-readable (and LLM-readable) description of what
                     this tool does, its inputs, and what it returns. Used
                     by agents for tool selection.
        tags: One or more department tags (e.g. ``["research"]``,
              ``["research", "content"]``). Agents filter by tag to get
              their allowed tool list.
        fn: The callable that implements the tool logic. Must be importable
            at registration time.
        metadata: Optional extra key-value pairs (e.g. version, author).

    Raises:
        ValueError: If ``name`` is empty, ``description`` is empty,
                    ``tags`` is empty, or ``fn`` is not callable.
    """

    name: str
    description: str
    tags: List[str]
    fn: Callable[..., Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the spec at construction time, before it reaches the registry."""
        errors: List[str] = []

        if not self.name or not isinstance(self.name, str):
            errors.append("'name' must be a non-empty string.")

        if not self.description or not isinstance(self.description, str):
            errors.append("'description' must be a non-empty string.")

        if not self.tags or not isinstance(self.tags, list) or not all(
            isinstance(t, str) and t for t in self.tags
        ):
            errors.append("'tags' must be a non-empty list of non-empty strings.")

        if not callable(self.fn):
            errors.append(f"'fn' must be callable, got {type(self.fn).__name__}.")

        if errors:
            raise ValueError(
                f"Invalid ToolSpec for '{self.name}': " + "; ".join(errors)
            )


# ─── ToolRegistry ─────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry that stores and provides lookup for all agent tools.

    This is intended to be used as a module-level singleton (``registry``).
    All registrations are performed once at application startup through
    ``registry_bootstrap.bootstrap()``, ensuring a deterministic,
    import-order-independent initialization.

    Thread safety: registrations happen at startup before any concurrent
    requests arrive, so no locking is required for the read-heavy runtime path.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """
        Register a tool with the registry.

        Parameters:
            spec: A fully validated :class:`ToolSpec` instance.

        Raises:
            ToolAlreadyRegisteredError: If a tool with ``spec.name`` is
                already registered. This is a hard startup failure, not a
                warning — it prevents silently overwriting existing tools.
        """
        if spec.name in self._tools:
            raise ToolAlreadyRegisteredError(spec.name)

        self._tools[spec.name] = spec
        logger.info(
            "Tool registered: name=%s tags=%s fn=%s",
            spec.name,
            spec.tags,
            spec.fn.__name__ if hasattr(spec.fn, "__name__") else repr(spec.fn),
        )

    def get(self, name: str) -> ToolSpec:
        """
        Look up a tool by its exact registered name.

        Parameters:
            name: The unique tool name to look up.

        Returns:
            The :class:`ToolSpec` for the named tool.

        Raises:
            ToolNotFoundError: If no tool with ``name`` is registered.
                Never returns ``None`` — callers must handle this exception.
        """
        if name not in self._tools:
            raise ToolNotFoundError(name, available=list(self._tools.keys()))
        return self._tools[name]

    def list_by_tag(self, tag: str) -> List[ToolSpec]:
        """
        Return all registered tools that carry the given tag.

        Parameters:
            tag: The department tag to filter by (e.g. ``"research"``).

        Returns:
            A list of :class:`ToolSpec` objects whose ``tags`` include
            ``tag``. Returns an empty list if no tools match — does not
            raise an exception.
        """
        return [spec for spec in self._tools.values() if tag in spec.tags]

    def list_all(self) -> List[ToolSpec]:
        """
        Return all registered tools in registration order.

        Returns:
            A list of all :class:`ToolSpec` objects currently in the registry.
        """
        return list(self._tools.values())

    def is_registered(self, name: str) -> bool:
        """
        Check whether a tool name exists in the registry without raising.

        Parameters:
            name: The tool name to check.

        Returns:
            ``True`` if registered, ``False`` otherwise.
        """
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={list(self._tools.keys())}>"


# ─── Module-level singleton ───────────────────────────────────────────────────

#: Global registry instance. Import this in agents and bootstrap modules.
registry: ToolRegistry = ToolRegistry()
