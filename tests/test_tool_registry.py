"""
Unit tests for shared/tool_registry.py

Covers:
- Happy path: register, get, list_by_tag, list_all
- Duplicate name → ToolAlreadyRegisteredError
- Missing name → ToolNotFoundError
- Invalid ToolSpec → ValueError at construction
- Edge cases: empty tag list, non-callable fn, is_registered
"""
import sys
from pathlib import Path

# Ensure backend root is on path for imports
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import pytest

from shared.tool_registry import (
    ToolRegistry,
    ToolSpec,
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _noop(*args, **kwargs) -> str:
    """A simple callable for use in tests."""
    return "noop"


def _make_spec(name: str = "test_tool", tags: list = None) -> ToolSpec:
    """Create a minimal valid ToolSpec."""
    return ToolSpec(
        name=name,
        description="A test tool that does nothing.",
        tags=tags or ["research"],
        fn=_noop,
    )


# ─── Happy path ───────────────────────────────────────────────────────────────

class TestToolRegistryHappyPath:
    def test_register_and_get(self):
        registry = ToolRegistry()
        spec = _make_spec("my_tool")
        registry.register(spec)
        retrieved = registry.get("my_tool")
        assert retrieved is spec
        assert retrieved.name == "my_tool"

    def test_list_by_tag_returns_matching_tools(self):
        registry = ToolRegistry()
        spec_research = _make_spec("research_tool", tags=["research"])
        spec_code = _make_spec("code_tool", tags=["code"])
        spec_both = _make_spec("multi_tool", tags=["research", "code"])
        registry.register(spec_research)
        registry.register(spec_code)
        registry.register(spec_both)

        research_tools = registry.list_by_tag("research")
        assert spec_research in research_tools
        assert spec_both in research_tools
        assert spec_code not in research_tools

    def test_list_all_returns_all_registered(self):
        registry = ToolRegistry()
        specs = [_make_spec(f"tool_{i}") for i in range(3)]
        for s in specs:
            registry.register(s)
        all_tools = registry.list_all()
        assert len(all_tools) == 3
        for s in specs:
            assert s in all_tools

    def test_list_by_tag_empty_when_no_match(self):
        registry = ToolRegistry()
        registry.register(_make_spec("my_tool", tags=["research"]))
        result = registry.list_by_tag("nonexistent_tag")
        assert result == []

    def test_is_registered_true_and_false(self):
        registry = ToolRegistry()
        registry.register(_make_spec("present_tool"))
        assert registry.is_registered("present_tool") is True
        assert registry.is_registered("absent_tool") is False

    def test_len(self):
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(_make_spec("tool_a"))
        assert len(registry) == 1
        registry.register(_make_spec("tool_b"))
        assert len(registry) == 2

    def test_tool_fn_is_callable_after_registration(self):
        registry = ToolRegistry()
        registry.register(_make_spec("callable_tool"))
        spec = registry.get("callable_tool")
        result = spec.fn()
        assert result == "noop"


# ─── Failure cases ────────────────────────────────────────────────────────────

class TestToolRegistryFailureCases:
    def test_duplicate_registration_raises(self):
        registry = ToolRegistry()
        spec = _make_spec("duplicate_tool")
        registry.register(spec)
        with pytest.raises(ToolAlreadyRegisteredError) as exc_info:
            registry.register(_make_spec("duplicate_tool"))
        assert "duplicate_tool" in str(exc_info.value)
        assert exc_info.value.tool_name == "duplicate_tool"

    def test_get_nonexistent_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError) as exc_info:
            registry.get("ghost_tool")
        assert "ghost_tool" in str(exc_info.value)
        assert exc_info.value.tool_name == "ghost_tool"

    def test_get_nonexistent_never_returns_none(self):
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            result = registry.get("missing")

    def test_toolnotfounderror_includes_available_names(self):
        registry = ToolRegistry()
        registry.register(_make_spec("existing_tool"))
        with pytest.raises(ToolNotFoundError) as exc_info:
            registry.get("missing_tool")
        # The error message should hint at available tools
        assert "existing_tool" in str(exc_info.value)


# ─── ToolSpec validation ──────────────────────────────────────────────────────

class TestToolSpecValidation:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            ToolSpec(name="", description="desc", tags=["research"], fn=_noop)

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            ToolSpec(name="tool", description="", tags=["research"], fn=_noop)

    def test_empty_tags_raises(self):
        with pytest.raises(ValueError, match="tags"):
            ToolSpec(name="tool", description="desc", tags=[], fn=_noop)

    def test_non_callable_fn_raises(self):
        with pytest.raises(ValueError, match="callable"):
            ToolSpec(name="tool", description="desc", tags=["research"], fn="not_callable")

    def test_none_fn_raises(self):
        with pytest.raises(ValueError):
            ToolSpec(name="tool", description="desc", tags=["research"], fn=None)

    def test_tags_with_empty_string_raises(self):
        with pytest.raises(ValueError, match="tags"):
            ToolSpec(name="tool", description="desc", tags=["", "research"], fn=_noop)

    def test_valid_spec_does_not_raise(self):
        spec = ToolSpec(name="valid", description="A valid tool.", tags=["code"], fn=_noop)
        assert spec.name == "valid"
        assert spec.tags == ["code"]
        assert callable(spec.fn)

    def test_metadata_defaults_to_empty_dict(self):
        spec = _make_spec()
        assert spec.metadata == {}
