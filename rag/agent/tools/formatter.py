"""Tool output formatter contract — decouples ContextBuilder from tool semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from rag.agent.memory.models import ContextSection, ExternalizedToolOutput
from rag.agent.tools.spec import ToolResult


@runtime_checkable
class ToolOutputFormatter(Protocol):
    """Per-tool context renderer registered in ToolRegistry.

    ContextBuilder calls format_result() for each ToolResult.
    Returns a ContextSection or None (no context injection needed).
    """

    tool_name: str

    def format_result(self, result: ToolResult) -> ContextSection | None:
        """Render one tool result as a context section for the next turn."""
        ...

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        """Render an externalized (large) output as a summary section."""
        ...


def format_tool_result_fallback(result: ToolResult) -> ContextSection | None:
    """Generic fallback when no formatter is registered for a tool."""

    # Reuse existing _format_tool_results logic
    output_text = ""
    if result.status == "ok":
        output = result.output
        if isinstance(output, ExternalizedToolOutput):
            output_text = f"externalized_ref={output.ref.ref_id} status={output.status} summary={output.summary}"
        elif output is not None:
            output_text = str(output.model_dump(mode="json", exclude_none=True))
    else:
        error = result.error
        if error is not None:
            output_text = f"error_code={error.code} retryable={error.retryable} message={error.message}"

    if not output_text.strip():
        return None
    content = (
        f"- tool_call_id={result.tool_call_id} tool_name={result.tool_name} "
        f"status={result.status} latency_ms={result.latency_ms:.3g} "
        f"output={_one_line(output_text)}"
    )
    return _make_section(name="tool_results", content=content)


def _make_section(*, name: str, content: str) -> ContextSection:
    return ContextSection(
        name=name,  # type: ignore[arg-type]
        content=content,
        token_count=0,  # caller should set this
        required=False,
    )


def _one_line(text: str) -> str:
    return " ".join(text.split())


ToolOutputFormatterResolver = Callable[[str], "ToolOutputFormatter | None"]

__all__ = [
    "ToolOutputFormatter",
    "ToolOutputFormatterResolver",
    "format_tool_result_fallback",
]
