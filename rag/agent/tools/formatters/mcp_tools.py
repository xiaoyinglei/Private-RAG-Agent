"""MCP tool output formatter — renders MCP content blocks for the LLM."""

from __future__ import annotations

from rag.agent.memory.models import ContextSection, ExternalizedToolOutput
from rag.agent.tools.spec import ToolResult


def _one_line(text: str) -> str:
    return " ".join(text.split())


class MCPToolFormatter:
    """Generic formatter for all MCP tools.

    Renders text content, counts images/resources, and flags errors.
    Uses the canonical tool name to identify the server.
    """

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name

    def format_result(self, result: ToolResult) -> ContextSection | None:
        if result.status != "ok" or result.output is None:
            return None
        text = getattr(result.output, "text", "") or ""
        images = getattr(result.output, "images", []) or []
        resources = getattr(result.output, "resources", []) or []
        is_error = getattr(result.output, "is_error", False)
        raw = getattr(result.output, "raw", {}) or {}

        lines: list[str] = [f"[{self.tool_name}]"]
        if is_error:
            lines.append("  status: ERROR")
        if text:
            lines.append(f"  text: {_one_line(text)[:800]}")
        if images:
            lines.append(f"  images: {len(images)} image(s)")
        if resources:
            for r in resources[:5]:
                lines.append(f"  resource: {r}")
        if raw and "error" in raw:
            lines.append(f"  error: {raw['error']}")
        return ContextSection(
            name="tool_results",
            content="\n".join(lines),
            token_count=0,
            required=False,
        )

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        if ref.summary:
            return ContextSection(
                name="tool_results",
                content=f"[{self.tool_name}] {ref.summary}",
                token_count=0,
                required=False,
            )
        return None
