"""Formatters for RAG retrieval tools.

Each formatter relocates rendering logic from ContextBuilder's private methods
(_format_evidence, _format_structured_observations, _format_locator) to
produce semantically identical output.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.agent.memory.models import ContextSection, ExternalizedToolOutput
from rag.agent.tools.spec import ToolResult

# ---------------------------------------------------------------------------
# Low-level rendering helpers (relocated from ContextBuilder)
# ---------------------------------------------------------------------------


def _one_line(text: str) -> str:
    """Collapse whitespace into single spaces (relocated from ContextBuilder._one_line)."""
    return " ".join(text.split())


def _preserve_spaces_one_line(text: str) -> str:
    """Replace only newlines/tabs with spaces, preserve internal spaces (relocated)."""
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()


def _format_locator_value(field: str, value: object) -> str:
    """Match ContextBuilder._format_locator_value — preserve spaces for path-like fields."""
    if field in {"path", "name", "sheet_name", "element_ref", "generated_by"}:
        return _preserve_spaces_one_line(str(value))
    return _one_line(str(value))


def _format_list(values: Sequence[object], *, limit: int | None = None) -> str:
    """Match ContextBuilder._format_list format: [a, b, ...(+N)]."""
    effective_limit = limit if limit is not None else len(values)
    shown = [_one_line(str(value)) for value in values[:effective_limit]]
    remaining = len(values) - effective_limit
    suffix = f", ...(+{remaining})" if remaining > 0 else ""
    return "[" + ", ".join(shown) + suffix + "]"


def _format_row_preview(row: object) -> str:
    """Match ContextBuilder._format_row_preview — {key=value, ...}."""
    if not isinstance(row, dict):
        return _one_line(str(row))
    cells: list[str] = []
    for index, (key, value) in enumerate(row.items()):
        if index >= 12:
            cells.append("...")
            break
        cells.append(f"{key}={value}")
    return "{" + _one_line(", ".join(cells)) + "}"


# ---------------------------------------------------------------------------
# Evidence metadata rendering (relocated from ContextBuilder._format_evidence)
# ---------------------------------------------------------------------------


def _evidence_meta(ev: object) -> str:
    """Relocated from ContextBuilder._metadata_line() — produce same format.

    Extracts the same fields that _format_evidence renders for each EvidenceItem:
      evidence_id, doc_id, citation_anchor, record_type, file_name,
      source_id, source_type, score.
    """
    parts: list[str] = []
    for field in (
        "evidence_id",
        "doc_id",
        "citation_anchor",
        "record_type",
        "file_name",
        "source_id",
        "source_type",
    ):
        val = getattr(ev, field, None)
        if val not in (None, "", []):
            parts.append(f"{field}={val}")
    if (score := getattr(ev, "score", None)) is not None:
        parts.append(f"score={score}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Locator rendering (relocated from ContextBuilder._format_locator)
# ---------------------------------------------------------------------------


def _render_locator(locator: dict[str, object]) -> str:
    """Relocated from ContextBuilder._format_locator() — same 36-field whitelist.

    Includes the field-value special-casing (path-like fields preserve spaces),
    analysis_capabilities, columns/column_names, and head_rows rendering.
    """
    fields = (
        "asset_id",
        "doc_id",
        "source_id",
        "section_id",
        "asset_type",
        "table_index",
        "table_name",
        "used_range",
        "sheet_name",
        "page_no",
        "element_ref",
        "citation_anchor",
        "evidence_id",
        "path",
        "name",
        "size_bytes",
        "is_dir",
        "mime_type",
        "file_kind",
        "truncated",
        "is_binary",
        "readable_as_text",
        "encoding",
        "source_tool",
        "generated",
        "generated_by",
        "ok",
        "exit_code",
        "duration_ms",
        "stdout_truncated",
        "stderr_truncated",
        "header_row_index",
        "header_confidence",
        "data_start_row",
        "row_count",
        "column_count",
    )
    parts: list[str] = [
        f"{field}={_format_locator_value(field, locator[field])}"
        for field in fields
        if locator.get(field) not in (None, "", [])
    ]

    capabilities = locator.get("analysis_capabilities")
    if isinstance(capabilities, list) and capabilities:
        parts.append("analysis_capabilities=" + _format_list(capabilities))

    columns = locator.get("columns") or locator.get("column_names")
    if isinstance(columns, list) and columns:
        parts.append("columns=" + _format_list(columns, limit=40))

    head_rows = locator.get("head_rows")
    if isinstance(head_rows, list) and head_rows:
        rows = [_format_row_preview(row) for row in head_rows[:8]]
        parts.append("head_rows=" + _format_list(rows, limit=len(rows)))

    return " ".join(parts) if parts else _one_line(str(locator))


# ---------------------------------------------------------------------------
# Main retrieval result formatting (relocated from ContextBuilder)
# ---------------------------------------------------------------------------


def _format_retrieval_result(
    result: ToolResult,
    tool_name: str,
) -> ContextSection | None:
    """Relocated from ContextBuilder._format_evidence() + _format_locator().

    Renders EvidenceItem items from output.evidence/output.items, with
    citation anchors, doc_ids, scores, and text previews bounded to 500 chars.
    Matches the output format of the deleted ContextBuilder methods exactly.
    """
    if result.status != "ok" or result.output is None:
        return None

    items = getattr(result.output, "items", None)
    evidence = getattr(result.output, "evidence", []) or []
    citations = getattr(result.output, "citations", []) or []

    lines: list[str] = []

    # Render evidence items (relocated from _format_evidence)
    for ev in evidence:
        meta_parts = _evidence_meta(ev)
        lines.append(f"- {meta_parts}")
        text = getattr(ev, "text", "")
        if text:
            lines.append(f"  text: {_one_line(str(text)[:500])}")

    # Render search items with locators (relocated from _format_locator)
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            locator_text = _render_locator(item)
            if locator_text:
                lines.append(f"- {locator_text}")
            text = str(item.get("text", ""))
            if text:
                lines.append(f"  text: {_one_line(text[:500])}")

    # Render citations
    for c in citations:
        lines.append(
            f"- citation: evidence_id={getattr(c, 'evidence_id', '')} anchor={getattr(c, 'citation_anchor', '')}"
        )

    if not lines:
        return None

    return ContextSection(
        name="tool_results",
        content=f"{tool_name} results:\n" + "\n".join(lines),
        token_count=0,
        required=False,
    )


# ---------------------------------------------------------------------------
# Formatter classes (ToolOutputFormatter protocol)
# ---------------------------------------------------------------------------


class VectorSearchFormatter:
    """Formatter for vector_search tool results."""

    tool_name = "vector_search"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_retrieval_result(result, "vector_search")

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class KeywordSearchFormatter:
    """Formatter for keyword_search tool results."""

    tool_name = "keyword_search"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_retrieval_result(result, "keyword_search")

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class GroundingFormatter:
    """Formatter for grounding tool results."""

    tool_name = "grounding"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_retrieval_result(result, "grounding")

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class RerankFormatter:
    """Formatter for rerank tool results."""

    tool_name = "rerank"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_retrieval_result(result, "rerank")

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class GraphExpandFormatter:
    """Formatter for graph_expand tool results."""

    tool_name = "graph_expand"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_retrieval_result(result, "graph_expand")

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class RAGSearchAnswerFormatter:
    """Formatter for rag_search_answer tool results."""

    tool_name = "rag_search_answer"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        if result.status != "ok" or result.output is None:
            return None
        text = getattr(result.output, "text", "") or ""
        citations = getattr(result.output, "citations", []) or []
        evidence = getattr(result.output, "evidence", []) or []
        groundedness = getattr(result.output, "groundedness_flag", False)
        insufficient = getattr(result.output, "insufficient_evidence", False)

        lines: list[str] = []
        lines.append(f"  answer: {_one_line(str(text)[:800])}")
        if groundedness:
            lines.append("  groundedness=True")
        if insufficient:
            lines.append("  insufficient_evidence=True")
        if evidence:
            lines.append(f"  evidence_count={len(evidence)}")
            for ev in evidence[:5]:
                meta = _evidence_meta(ev)
                if meta:
                    lines.append(f"    - {meta}")
        if citations:
            lines.append(f"  citation_count={len(citations)}")
        return ContextSection(
            name="tool_results",
            content="rag_search_answer results:\n" + "\n".join(lines),
            token_count=0,
            required=False,
        )

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class SearchKnowledgeFormatter:
    """Formatter for search_knowledge — unified semantic retrieval results."""

    tool_name = "search_knowledge"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        if result.status != "ok" or result.output is None:
            return None
        results = getattr(result.output, "results", []) or []
        answer_text = getattr(result.output, "answer_text", "") or ""
        citations = getattr(result.output, "citations", []) or []
        kg = getattr(result.output, "kg_neighbors", []) or []
        groundedness = getattr(result.output, "groundedness_flag", False)
        insufficient = getattr(result.output, "insufficient_evidence", False)
        total = getattr(result.output, "total_found", len(results))

        lines: list[str] = [f"search_knowledge: {total} results found"]
        if answer_text:
            lines.append(f"  answer: {_one_line(str(answer_text)[:800])}")
        if groundedness:
            lines.append("  groundedness=True")
        if insufficient:
            lines.append("  insufficient_evidence=True")
        for r in results[:10]:
            eid = getattr(r, "evidence_id", "")
            anchor = getattr(r, "citation_anchor", "")
            score = getattr(r, "score", 0)
            text = _one_line(str(getattr(r, "text", "")))[:300]
            lines.append(f"  - {eid} {anchor} score={score:.2f}")
            if text:
                lines.append(f"    {text}")
        if citations:
            lines.append(f"  citations: {len(citations)} total")
        if kg:
            neighbors = [f"{getattr(n, 'entity_label', '')}({getattr(n, 'relation', '')})" for n in kg[:10]]
            lines.append(f"  kg_neighbors: [{', '.join(neighbors)}]")
        return ContextSection(
            name="tool_results",
            content="\n".join(lines),
            token_count=0,
            required=False,
        )

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class SearchAssetsFormatter:
    """Formatter for search_assets — unified asset results."""

    tool_name = "search_assets"

    def format_result(self, result: ToolResult) -> ContextSection | None:
        if result.status != "ok" or result.output is None:
            return None
        assets = getattr(result.output, "assets", []) or []
        total = getattr(result.output, "total_found", len(assets))
        truncated = getattr(result.output, "truncated", False)

        lines: list[str] = [
            f"search_assets: {total} assets found{' (truncated)' if truncated else ''}"
        ]
        for a in assets[:15]:
            aid = getattr(a, "asset_id", "?")
            atype = getattr(a, "asset_type", "")
            caption = _one_line(str(getattr(a, "caption", "") or ""))[:100]
            cols = getattr(a, "columns", []) or []
            rows = getattr(a, "row_count", None)
            preview_rows = getattr(a, "preview_rows", []) or []
            caps = getattr(a, "analysis_capabilities", []) or []
            line = f"  - asset_id={aid} type={atype}"
            if cols:
                line += f" columns={cols[:10]}"
            if rows is not None:
                line += f" rows={rows}"
            if caption:
                line += f" caption={caption}"
            if caps:
                line += f" caps={caps}"
            lines.append(line)
            if preview_rows:
                first = preview_rows[0]
                if isinstance(first, dict):
                    lines.append(f"    preview: {_one_line(str(first))[:200]}")
        return ContextSection(
            name="tool_results",
            content="\n".join(lines),
            token_count=0,
            required=False,
        )

    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


__all__ = [
    "GraphExpandFormatter",
    "GroundingFormatter",
    "KeywordSearchFormatter",
    "RAGSearchAnswerFormatter",
    "RerankFormatter",
    "SearchAssetsFormatter",
    "SearchKnowledgeFormatter",
    "VectorSearchFormatter",
]
