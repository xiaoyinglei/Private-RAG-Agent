"""Formatters for file/workspace tools.

Each formatter relocates rendering logic from observations.py context unit
functions and the ContextBuilder._format_locator method to produce
semantically identical output for file workspace tool results.
"""

from __future__ import annotations

from typing import cast

from rag.agent.memory.models import ContextSection, ExternalizedToolOutput
from rag.agent.tools.formatter import _make_section, _one_line
from rag.agent.tools.formatters.rag_retrieval import (
    _format_list,
    _render_locator,
)
from rag.agent.tools.spec import ToolResult

# ---------------------------------------------------------------------------
# Locator-building helpers (relocated from observations.py)
# ---------------------------------------------------------------------------


def _workspace_file_locator(file_info: object, *, source_tool: str) -> dict[str, object]:
    """Build locator dict for a workspace file entry.

    Relocated from ``observations._workspace_file_locator``.
    """
    values: dict[str, object] = {"source_tool": source_tool}
    for field, output_field in (
        ("path", "path"),
        ("name", "name"),
        ("size", "size_bytes"),
        ("is_dir", "is_dir"),
        ("mime_type", "mime_type"),
    ):
        value = getattr(file_info, field, None)
        if value not in (None, "", []):
            values[output_field] = value
    file_kind = getattr(file_info, "file_kind", None)
    has_file_kind = isinstance(file_kind, str) and file_kind not in {"", "unknown"}
    if has_file_kind:
        values["file_kind"] = file_kind
        for field in ("is_binary", "readable_as_text"):
            value = getattr(file_info, field, None)
            if isinstance(value, bool):
                values[field] = value
    return values


def _read_file_locator(output: object) -> dict[str, object]:
    """Build locator dict for a read_file output.

    Relocated from ``observations._read_file_locator``.
    """
    values: dict[str, object] = {"source_tool": "read_file"}
    for output_field, locator_field in (
        ("path", "path"),
        ("size_bytes", "size_bytes"),
        ("truncated", "truncated"),
        ("is_binary", "is_binary"),
        ("encoding", "encoding"),
    ):
        value = getattr(output, output_field, None)
        if value not in (None, "", []):
            values[locator_field] = value
    return values


def _write_file_locator(output: object) -> dict[str, object]:
    """Build locator dict for a write_file output.

    Relocated from ``observations._write_file_locator``.
    """
    values: dict[str, object] = {"source_tool": "write_file"}
    for field in ("path", "size_bytes"):
        value = getattr(output, field, None)
        if value not in (None, "", []):
            values[field] = value
    return values


def _run_python_locator(output: object) -> dict[str, object]:
    """Build locator dict for a run_python output.

    Relocated from ``observations._run_python_locator``.
    """
    values: dict[str, object] = {"source_tool": "run_python"}
    for field in (
        "ok",
        "exit_code",
        "duration_ms",
        "stdout_truncated",
        "stderr_truncated",
        "generated_files",
    ):
        value = getattr(output, field, None)
        if value not in (None, "", []):
            values[field] = value
    return values


def _structured_table_locator(path: str, table: object) -> dict[str, object]:
    """Build locator dict for a structured table.

    Relocated from ``observations._structured_table_locator``.
    """
    values: dict[str, object] = {
        "path": path,
        "source_tool": "structured_probe",
    }
    for output_field, locator_field in (
        ("table_index", "table_index"),
        ("name", "table_name"),
        ("used_range", "used_range"),
        ("row_count", "row_count"),
        ("column_count", "column_count"),
        ("data_start_row", "data_start_row"),
    ):
        value = getattr(table, output_field, None)
        if value not in (None, "", []):
            values[locator_field] = value
    candidates = getattr(table, "candidate_header_rows", None)
    if isinstance(candidates, list) and candidates:
        best = candidates[0]
        row_index = getattr(best, "row_index", None)
        confidence = getattr(best, "confidence", None)
        if isinstance(row_index, int):
            values["header_row_index"] = row_index
        if isinstance(confidence, int | float):
            values["header_confidence"] = float(confidence)
    return values


# ---------------------------------------------------------------------------
# Preview helpers (relocated from observations.py)
# ---------------------------------------------------------------------------


def _run_python_preview(output: object) -> str | None:
    """Build preview string for a run_python output.

    Relocated from ``observations._run_python_preview``.
    """
    lines: list[str] = []
    stdout = getattr(output, "stdout", None)
    if isinstance(stdout, str) and stdout.strip():
        lines.append("stdout: " + stdout.strip()[:500])
    stderr = getattr(output, "stderr", None)
    if isinstance(stderr, str) and stderr.strip():
        lines.append("stderr: " + stderr.strip()[:500])
    generated = getattr(output, "generated_files", None)
    if isinstance(generated, list) and generated:
        lines.append("generated_files: " + ", ".join(str(path) for path in generated[:20]))
    return "\n".join(lines) if lines else None


def _structured_table_preview(table: object) -> str | None:
    """Build preview string for a structured table.

    Relocated from ``observations._structured_table_preview``.
    """
    rows = getattr(table, "sample_rows", None)
    row_count = getattr(table, "row_count", None)
    column_count = getattr(table, "column_count", None)
    used_range = getattr(table, "used_range", None)
    parts: list[str] = [
        f"rows={row_count}" if isinstance(row_count, int) else "",
        f"columns={column_count}" if isinstance(column_count, int) else "",
        (f"used_range={used_range}" if isinstance(used_range, str) and used_range else ""),
    ]
    header_row = _header_sample_row(table, rows)
    if header_row is not None:
        parts.append(f"header_row={_bounded_row_preview(header_row)}")
    elif isinstance(rows, list) and rows:
        parts.append(f"first_row={_bounded_row_preview(rows[0])}")
    preview = " ".join(part for part in parts if part)
    return preview or None


def _header_sample_row(table: object, rows: object) -> object | None:
    """Extract the header candidate row from sample rows.

    Relocated from ``observations._header_sample_row``.
    """
    if not isinstance(rows, list) or not rows:
        return None
    candidates = getattr(table, "candidate_header_rows", None)
    if not isinstance(candidates, list) or not candidates:
        return None
    row_index = getattr(candidates[0], "row_index", None)
    if not isinstance(row_index, int):
        return None
    sample_index = row_index - 1
    if sample_index < 0 or sample_index >= len(rows):
        return None
    return cast(object, rows[sample_index])


def _bounded_row_preview(row: object) -> str:
    """Format a row as a bounded preview list.

    Relocated from ``observations._bounded_row_preview``.
    """
    if not isinstance(row, list):
        return _bounded_cell_preview(row)
    cells = [_bounded_cell_preview(cell) for cell in row[:8]]
    suffix = f", ...(+{len(row) - 8})" if len(row) > 8 else ""
    return "[" + ", ".join(cells) + suffix + "]"


def _bounded_cell_preview(cell: object) -> str:
    """Format a single cell with length bound.

    Relocated from ``observations._bounded_cell_preview``.
    """
    text = str(cell)
    if len(text) > 40:
        text = text[:40].rstrip() + "..."
    return repr(text)


# ---------------------------------------------------------------------------
# Main formatting functions
# ---------------------------------------------------------------------------


def _format_list_files(result: ToolResult) -> ContextSection | None:
    """Render list_files tool result as a context section."""
    if result.status != "ok" or result.output is None:
        return None
    output = result.output
    files = getattr(output, "files", []) or []
    if not files:
        return None

    lines: list[str] = []
    for file_info in files:
        locator = _workspace_file_locator(file_info, source_tool="list_files")
        path = locator.get("path")
        if not isinstance(path, str) or not path.strip():
            continue
        is_dir = bool(locator.get("is_dir", False))
        unit_id = f"workspace_dir:{path}" if is_dir else f"workspace_file:{path}"
        unit_type = "workspace_dir" if is_dir else "workspace_file"
        locator_text = _render_locator(locator)
        lines.append(f"- unit_id={unit_id} unit_type={unit_type} {locator_text}")
        preview = f"{path} ({locator.get('size_bytes', 0)} bytes)"
        lines.append(f"  preview: {preview}")
        raw = getattr(file_info, "capabilities", None)
        if isinstance(raw, list) and raw:
            capabilities = [str(item) for item in raw if str(item)]
        else:
            capabilities = ["list_files"] if is_dir else ["read_file"]
        lines.append("  capabilities: " + _format_list(capabilities))

    if not lines:
        return None
    return _make_section(
        name="tool_results",
        content="list_files results:\n" + "\n".join(lines),
    )


def _format_read_file(result: ToolResult) -> ContextSection | None:
    """Render read_file tool result as a context section."""
    if result.status != "ok" or result.output is None:
        return None
    output = result.output
    path = getattr(output, "path", None)
    if not isinstance(path, str) or not path.strip():
        return None

    locator = _read_file_locator(output)
    content = getattr(output, "content", None)
    preview = content[:1000] if isinstance(content, str) and content else None

    lines: list[str] = [
        f"- unit_id=workspace_file:{path} unit_type=workspace_file_content {_render_locator(locator)}",
    ]
    if preview:
        lines.append(f"  preview: {_one_line(preview)}")
    lines.append("  capabilities: [read_file]")
    return _make_section(
        name="tool_results",
        content="read_file results:\n" + "\n".join(lines),
    )


def _format_write_file(result: ToolResult) -> ContextSection | None:
    """Render write_file tool result as a context section."""
    if result.status != "ok" or result.output is None:
        return None
    output = result.output
    path = getattr(output, "path", None)
    if not isinstance(path, str) or not path.strip():
        return None

    locator = _write_file_locator(output)
    size = locator.get("size_bytes", 0)
    lines: list[str] = [
        f"- unit_id=workspace_file:{path} unit_type=workspace_file {_render_locator(locator)}",
        f"  preview: wrote {path} ({size} bytes)",
        "  capabilities: [read_file]",
    ]
    return _make_section(
        name="tool_results",
        content="write_file results:\n" + "\n".join(lines),
    )


def _format_run_python(result: ToolResult) -> ContextSection | None:
    """Render run_python tool result as a context section."""
    if result.status != "ok" or result.output is None:
        return None
    output = result.output

    locator = _run_python_locator(output)
    preview = _run_python_preview(output)

    lines: list[str] = [
        f"- unit_id=python_run:{result.tool_call_id} unit_type=python_execution {_render_locator(locator)}",
    ]
    if preview:
        lines.append(f"  preview: {_one_line(preview)}")
    lines.append("  capabilities: [run_python]")

    for path in getattr(output, "generated_files", []) or []:
        if not isinstance(path, str) or not path.strip():
            continue
        gen_locator: dict[str, object] = {
            "path": path,
            "source_tool": "run_python",
            "generated_by": result.tool_call_id,
        }
        lines.append(f"- unit_id=workspace_file:{path} unit_type=workspace_file {_render_locator(gen_locator)}")
        lines.append(f"  preview: generated {path}")
        lines.append("  capabilities: [read_file]")

    return _make_section(
        name="tool_results",
        content="run_python results:\n" + "\n".join(lines),
    )


def _format_structured_probe(result: ToolResult) -> ContextSection | None:
    """Render structured_probe tool result as a context section."""
    if result.status != "ok" or result.output is None:
        return None
    output = result.output
    path = getattr(output, "path", None)
    if not isinstance(path, str) or not path.strip():
        return None

    tables = getattr(output, "tables", []) or []
    if not tables:
        return None

    lines: list[str] = []
    for table in tables:
        table_index = getattr(table, "table_index", len(lines))
        locator = _structured_table_locator(path, table)
        preview = _structured_table_preview(table)
        lines.append(
            f"- unit_id=structured_table:{path}:{table_index} unit_type=structured_table {_render_locator(locator)}"
        )
        if preview:
            lines.append(f"  preview: {_one_line(preview)}")
        lines.append("  capabilities: [structured_probe, run_python]")

    return _make_section(
        name="tool_results",
        content="structured_probe results:\n" + "\n".join(lines),
    )


# ---------------------------------------------------------------------------
# Formatter classes (ToolOutputFormatter protocol)
# ---------------------------------------------------------------------------


class ListFilesFormatter:
    """Formatter for list_files tool results."""

    tool_name = "list_files"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_list_files(result)

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class ReadFileFormatter:
    """Formatter for read_file tool results."""

    tool_name = "read_file"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_read_file(result)

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class WriteFileFormatter:
    """Formatter for write_file tool results."""

    tool_name = "write_file"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_write_file(result)

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class RunPythonFormatter:
    """Formatter for run_python tool results."""

    tool_name = "run_python"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_run_python(result)

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class StructuredProbeFormatter:
    """Formatter for structured_probe tool results."""

    tool_name = "structured_probe"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_structured_probe(result)

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


__all__ = [
    "ListFilesFormatter",
    "ReadFileFormatter",
    "RunPythonFormatter",
    "StructuredProbeFormatter",
    "WriteFileFormatter",
]
