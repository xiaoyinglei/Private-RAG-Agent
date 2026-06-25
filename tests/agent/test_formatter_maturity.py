"""PR6: Formatter maturity — ensure mature tools have custom formatters."""

from __future__ import annotations

from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.formatter import format_tool_result_fallback


# ── Tools that have custom formatters (should grow over time) ──

_MATURE_TOOLS = frozenset({
    # RAG retrieval (PR2)
    "vector_search",
    "keyword_search",
    "grounding",
    "rerank",
    "graph_expand",
    # File/workspace (PR2)
    "list_files",
    "read_file",
    "write_file",
    "run_python",
    "structured_probe",
    # Asset tools (PR6)
    "asset_list",
    "asset_inspect",
    "asset_read_slice",
    "asset_analyze",
    # LLM tools (PR6)
    "llm_generate",
    "llm_summarize",
    "llm_compare",
    # RAG answer (PR6)
    "rag_search_answer",
})

# ── Tools that intentionally use fallback (output is simple) ──
_FALLBACK_TOOLS = frozenset({
    "tool_search",
    "activate_tools",
    "task",
    "run_python_inline",
})


def _build_registry():
    return create_builtin_tool_registry()


class TestFormatterMaturity:
    """PR6: Mature tools MUST have a custom formatter."""

    def test_mature_tools_have_custom_formatter(self) -> None:
        """Every tool in _MATURE_TOOLS has a formatter registered."""
        registry = _build_registry()
        missing: list[str] = []
        for tool_name in sorted(_MATURE_TOOLS):
            formatter = registry.get_formatter(tool_name)
            if formatter is None:
                missing.append(tool_name)
        assert missing == [], (
            f"Mature tools missing custom formatters: {missing}"
        )

    def test_formatter_registry_consistency(self) -> None:
        """Registered formatter tool_names match a known ToolSpec name."""
        registry = _build_registry()
        spec_names = {spec.name for spec in registry.list_all()}
        mismatches: list[str] = []
        for formatter_tool_name in sorted(registry._formatters):
            if formatter_tool_name not in spec_names:
                mismatches.append(formatter_tool_name)
        assert mismatches == [], (
            f"Formatters with no matching ToolSpec: {mismatches}"
        )

    def test_formatter_gap_report(self) -> None:
        """List all tools using fallback (informational, not failure).

        This test always passes; it surfaces gaps for review.
        New tools should be assessed for custom formatter needs.
        """
        registry = _build_registry()
        fallback_tools: list[str] = []
        custom_tools: list[str] = []
        for spec in registry.list_all():
            formatter = registry.get_formatter(spec.name)
            if formatter is not None:
                custom_tools.append(spec.name)
            else:
                fallback_tools.append(spec.name)

        # Known fallback tools are expected
        unexpected = sorted(set(fallback_tools) - _FALLBACK_TOOLS)
        # Log the gap report (always visible in test output)
        print(f"\n  Custom formatters: {len(custom_tools)} tools")
        print(f"  Fallback formatters: {len(fallback_tools)} tools")
        if fallback_tools:
            print(f"  Fallback tool names: {sorted(fallback_tools)}")
        if unexpected:
            print(f"  UNEXPECTED (need assessment): {unexpected}")

        # Not a failure — informational only
        assert True

    def test_custom_formatter_not_just_fallback(self) -> None:
        """Each custom formatter's format_result is NOT the generic fallback."""
        registry = _build_registry()
        # The fallback function is format_tool_result_fallback — verify
        # that registered formatters for mature tools are different objects.
        for tool_name in sorted(_MATURE_TOOLS):
            formatter = registry.get_formatter(tool_name)
            assert formatter is not None, f"{tool_name} missing formatter"
            # Its format_result should not be the same function as fallback
            assert formatter.format_result is not format_tool_result_fallback, (
                f"{tool_name} uses generic fallback, needs custom formatter"
            )
