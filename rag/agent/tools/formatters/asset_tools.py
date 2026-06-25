"""Formatters for asset inspection and analysis tools."""

from __future__ import annotations

from rag.agent.memory.models import ContextSection, ExternalizedToolOutput
from rag.agent.tools.spec import ToolResult


def _one_line(text: str) -> str:
    return " ".join(text.split())


class AssetListFormatter:
    """Formatter for asset_list tool results."""

    tool_name = "asset_list"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        if result.status != "ok" or result.output is None:
            return None
        assets = getattr(result.output, "assets", []) or []
        truncated = getattr(result.output, "truncated", False)
        if not assets:
            return None
        lines: list[str] = []
        for a in assets[:20]:
            parts = [
                f"asset_id={getattr(a, 'asset_id', '')}",
                f"doc_id={getattr(a, 'doc_id', '')}",
            ]
            if source_id := getattr(a, "source_id", None):
                parts.append(f"source_id={source_id}")
            if sheet_name := getattr(a, "sheet_name", None):
                parts.append(f"sheet_name={_one_line(str(sheet_name))}")
            if asset_type := getattr(a, "asset_type", None):
                parts.append(f"asset_type={asset_type}")
            if caption := getattr(a, "caption", None):
                parts.append(f"caption={_one_line(str(caption))[:120]}")
            if col_count := getattr(a, "column_count", None):
                parts.append(f"columns={col_count}")
            lines.append("- " + " ".join(parts))
        if truncated or len(assets) > 20:
            lines.append(f"  ...(+{len(assets) - 20} more)" if len(assets) > 20 else "  (truncated)")
        return ContextSection(
            name="tool_results",
            content="asset_list results:\n" + "\n".join(lines),
            token_count=0,
            required=False,
        )

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class AssetInspectFormatter:
    """Formatter for asset_inspect tool results."""

    tool_name = "asset_inspect"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        if result.status != "ok" or result.output is None:
            return None
        output = result.output
        parts = [
            f"asset_id={getattr(output, 'asset_id', '')}",
            f"doc_id={getattr(output, 'doc_id', '')}",
        ]
        if source_id := getattr(output, "source_id", None):
            parts.append(f"source_id={source_id}")
        if asset_type := getattr(output, "asset_type", None):
            parts.append(f"asset_type={asset_type}")
        if sheet_name := getattr(output, "sheet_name", None):
            parts.append(f"sheet_name={_one_line(str(sheet_name))}")
        if caption := getattr(output, "caption", None):
            parts.append(f"caption={_one_line(str(caption))[:200]}")
        row_count = getattr(output, "row_count", None)
        col_count = getattr(output, "column_count", None)
        if row_count is not None:
            parts.append(f"rows={row_count}")
        if col_count is not None:
            parts.append(f"columns={col_count}")
        columns = getattr(output, "columns", []) or []
        if columns:
            parts.append("columns=" + str(columns[:30]))
        caps = getattr(output, "analysis_capabilities", []) or []
        if caps:
            parts.append("capabilities=" + str(caps))
        # Preview head rows
        head_rows = getattr(output, "head_rows", []) or []
        if head_rows:
            preview = head_rows[:5]
            parts.append(f"head_rows={len(head_rows)}")
            for row in preview:
                if isinstance(row, dict):
                    parts.append("  " + _one_line(str({k: v for k, v in list(row.items())[:8]})))
        return ContextSection(
            name="tool_results",
            content="asset_inspect: " + " ".join(parts),
            token_count=0,
            required=False,
        )

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class AssetReadSliceFormatter:
    """Formatter for asset_read_slice tool results."""

    tool_name = "asset_read_slice"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        if result.status != "ok" or result.output is None:
            return None
        output = result.output
        parts = [
            f"asset_id={getattr(output, 'asset_id', '')}",
            f"doc_id={getattr(output, 'doc_id', '')}",
        ]
        columns = getattr(output, "columns", []) or []
        if columns:
            parts.append(f"columns={columns[:20]}")
        rows = getattr(output, "rows", []) or []
        parts.append(f"rows={len(rows)}")
        total = getattr(output, "raw_row_count", None)
        if total is not None:
            parts.append(f"total_rows={total}")
        truncated = getattr(output, "truncated", False)
        if truncated:
            parts.append("truncated")
        # Preview first few rows
        if rows:
            parts.append("preview=" + _one_line(str(rows[0])[:200]))
        return ContextSection(
            name="tool_results",
            content="asset_read_slice: " + " ".join(parts),
            token_count=0,
            required=False,
        )

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class AssetAnalyzeFormatter:
    """Formatter for asset_analyze tool results."""

    tool_name = "asset_analyze"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        if result.status != "ok" or result.output is None:
            return None
        output = result.output
        parts = [
            f"asset_id={getattr(output, 'asset_id', '')}",
            f"operation={getattr(output, 'operation', '')}",
        ]
        if doc_id := getattr(output, "doc_id", None):
            parts.append(f"doc_id={doc_id}")
        if asset_type := getattr(output, "asset_type", None):
            parts.append(f"asset_type={asset_type}")
        observation_only = getattr(output, "observation_only", False)
        if observation_only:
            parts.append("observation_only")
        columns = getattr(output, "columns", []) or []
        if columns:
            parts.append(f"columns={columns[:20]}")
        rows = getattr(output, "rows", []) or []
        parts.append(f"result_rows={len(rows)}")
        # Preview first result row
        if rows:
            first_row = rows[0]
            if isinstance(first_row, list):
                parts.append("preview=" + _one_line(str(first_row)[:200]))
            elif isinstance(first_row, dict):
                parts.append("preview=" + _one_line(str(first_row)[:200]))
        return ContextSection(
            name="tool_results",
            content="asset_analyze: " + " ".join(parts),
            token_count=0,
            required=False,
        )

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None
