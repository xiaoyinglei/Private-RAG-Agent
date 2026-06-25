"""Formatters for generic coding-agent tools (search_text, apply_patch, run_command, update_plan)."""

from __future__ import annotations

from rag.agent.memory.models import ContextSection, ExternalizedToolOutput
from rag.agent.tools.spec import ToolResult


def _one_line(text: str) -> str:
    return " ".join(text.split())


class SearchTextFormatter:
    """Formatter for search_text results — render matches compactly."""

    tool_name = "search_text"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        if result.status != "ok" or result.output is None:
            return None
        matches = getattr(result.output, "matches", []) or []
        total = getattr(result.output, "total_matches", len(matches))
        truncated = getattr(result.output, "truncated", False)

        if not matches:
            return ContextSection(
                name="tool_results",
                content="search_text: 0 matches found.",
                token_count=0,
                required=False,
            )

        lines: list[str] = [f"search_text: {total} matches{' (truncated)' if truncated else ''}:"]
        for m in matches[:30]:
            path = getattr(m, "file_path", "")
            line_no = getattr(m, "line_number", 0)
            content = _one_line(getattr(m, "line_content", ""))[:120]
            lines.append(f"  {path}:{line_no}: {content}")
        if len(matches) > 30:
            lines.append(f"  ...(+{len(matches) - 30} more matches)")
        return ContextSection(
            name="tool_results",
            content="\n".join(lines),
            token_count=0,
            required=False,
        )

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class ApplyPatchFormatter:
    """Formatter for apply_patch results."""

    tool_name = "apply_patch"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        if result.status != "ok" or result.output is None:
            return None
        file_path = getattr(result.output, "file_path", "")
        replaced = getattr(result.output, "replaced", False)
        occurrences = getattr(result.output, "occurrences", 0)
        message = getattr(result.output, "message", "")
        content = f"apply_patch: {file_path} replaced={replaced} occurrences={occurrences}"
        if message:
            content += f" message={message}"
        return ContextSection(
            name="tool_results",
            content=content,
            token_count=0,
            required=False,
        )

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class RunCommandFormatter:
    """Formatter for run_command results — render exit code and output preview."""

    tool_name = "run_command"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        if result.status != "ok" or result.output is None:
            return None
        exit_code = getattr(result.output, "exit_code", -1)
        stdout = getattr(result.output, "stdout", "") or ""
        stderr = getattr(result.output, "stderr", "") or ""
        timed_out = getattr(result.output, "timed_out", False)
        truncated = getattr(result.output, "truncated", False)
        duration_ms = getattr(result.output, "duration_ms", 0.0)

        lines: list[str] = [
            f"run_command: exit_code={exit_code} duration={duration_ms:.0f}ms"
            + (" TIMED_OUT" if timed_out else "")
            + (" (truncated)" if truncated else ""),
        ]
        if stdout:
            lines.append(f"  stdout: {_one_line(stdout)[:500]}")
        if stderr:
            lines.append(f"  stderr: {_one_line(stderr)[:500]}")
        return ContextSection(
            name="tool_results",
            content="\n".join(lines),
            token_count=0,
            required=False,
        )

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class UpdatePlanFormatter:
    """Formatter for update_plan results — render current plan state."""

    tool_name = "update_plan"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        if result.status != "ok" or result.output is None:
            return None
        steps = getattr(result.output, "steps", []) or []
        summary = getattr(result.output, "summary", "") or ""
        message = getattr(result.output, "message", "") or ""

        lines: list[str] = []
        if summary:
            lines.append(f"  summary: {summary}")
        if steps:
            lines.append("  plan:")
            for s in steps:
                sid = getattr(s, "id", "?")
                desc = _one_line(getattr(s, "description", ""))[:100]
                status = getattr(s, "status", "pending")
                icon = {"pending": " ", "in_progress": ">", "completed": "✓", "blocked": "!"}.get(status, "?")
                lines.append(f"    [{icon}] {sid}: {desc} ({status})")
        if message:
            lines.append(f"  {message}")
        return ContextSection(
            name="tool_results",
            content="update_plan:\n" + "\n".join(lines) if lines else "update_plan: no steps",
            token_count=0,
            required=False,
        )

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None
