"""Formatters for LLM sub-agent tools (generate, summarize, compare)."""

from __future__ import annotations

from rag.agent.memory.models import ContextSection, ExternalizedToolOutput
from rag.agent.tools.spec import ToolResult


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _render_llm_text_output(
    result: ToolResult,
    tool_label: str,
) -> ContextSection | None:
    """Shared renderer for LLMTextOutput-based tools."""
    if result.status != "ok" or result.output is None:
        return None
    text = getattr(result.output, "text", "") or ""
    evidence_ids = getattr(result.output, "evidence_ids", []) or []
    citation_ids = getattr(result.output, "citation_ids", []) or []
    insufficient = getattr(result.output, "insufficient_evidence", False)

    lines: list[str] = []
    preview = _one_line(text)[:500]
    lines.append(f"  text: {preview}")
    if insufficient:
        lines.append("  insufficient_evidence=True")
    if evidence_ids:
        lines.append(f"  evidence_ids: [{', '.join(evidence_ids[:10])}]")
    if citation_ids:
        lines.append(f"  citation_ids: [{', '.join(citation_ids[:10])}]")
    return ContextSection(
        name="tool_results",
        content=f"{tool_label}:\n" + "\n".join(lines),
        token_count=0,
        required=False,
    )


class LLMGenerateFormatter:
    """Formatter for llm_generate tool results."""

    tool_name = "llm_generate"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _render_llm_text_output(result, "llm_generate")

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class LLMSummarizeFormatter:
    """Formatter for llm_summarize tool results."""

    tool_name = "llm_summarize"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _render_llm_text_output(result, "llm_summarize")

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class LLMCompareFormatter:
    """Formatter for llm_compare tool results."""

    tool_name = "llm_compare"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _render_llm_text_output(result, "llm_compare")

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None
