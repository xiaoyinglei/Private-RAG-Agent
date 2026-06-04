from __future__ import annotations

from collections.abc import Awaitable
from inspect import isawaitable
from typing import Any, Protocol

from pydantic import BaseModel

from rag.agent.state import AgentState
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem


class SynthesisRunResult(Protocol):
    status: str
    final_answer: str | None
    stop_reason: str | None
    tool_results: list[ToolResult]
    evidence: list[EvidenceItem]
    citations: list[AnswerCitation]
    groundedness_flag: bool
    insufficient_evidence_flag: bool


class SynthesisRunner(Protocol):
    def run_synthesis(
        self,
        *,
        parent_state: AgentState,
    ) -> SynthesisRunResult | Awaitable[SynthesisRunResult]: ...


async def synthesize_node(
    state: AgentState,
    *,
    synthesis_runner: SynthesisRunner | None = None,
) -> dict[str, Any]:
    if state.get("stop_reason") == "goal_satisfied":
        if final_answer := _structured_goal_final_answer(state):
            return {
                "status": "done",
                "final_answer": final_answer,
                "groundedness_flag": _has_traceable_support(state),
                "insufficient_evidence_flag": False,
            }

    if synthesis_runner is not None and _should_delegate_to_synthesis_agent(state):
        try:
            raw_result = synthesis_runner.run_synthesis(parent_state=state)
            result = await raw_result if isawaitable(raw_result) else raw_result
        except Exception as exc:
            fallback = _legacy_synthesize_node(state)
            return {
                **fallback,
                "stop_reason": f"synthesis_agent_failed: {exc}",
                "insufficient_evidence_flag": True,
            }
        return _synthesis_agent_update(state, result)

    return _legacy_synthesize_node(state)


def _structured_goal_final_answer(state: AgentState) -> str | None:
    for candidate in reversed(state.get("answer_candidates", [])):
        text = getattr(candidate, "text", None)
        if not isinstance(text, str) or not text.strip():
            continue
        lines = [text.strip()]
        source_tool_call_id = getattr(candidate, "source_tool_call_id", None)
        for unit in _supporting_asset_units(state, candidate):
            lines.append(f"出处：{_asset_source_line(unit)}")
        for computation in reversed(state.get("computation_results", [])):
            if getattr(computation, "source_tool_call_id", None) != source_tool_call_id:
                continue
            expression = getattr(computation, "expression", None)
            if isinstance(expression, str) and expression.strip():
                lines.append(f"计算：{expression.strip()}")
            break
        return "\n".join(lines)
    return None


def _supporting_asset_units(state: AgentState, candidate: object) -> list[object]:
    evidence_asset_ids = {
        int(evidence_id.split(":", maxsplit=1)[1])
        for ref in getattr(candidate, "evidence_refs", []) or []
        if isinstance((evidence_id := getattr(ref, "evidence_id", None)), str)
        and evidence_id.startswith("asset:")
        and evidence_id.split(":", maxsplit=1)[1].isdigit()
    }
    units = [
        unit
        for unit in state.get("context_units", [])
        if getattr(unit, "unit_type", None) in {"table_asset", "image_asset", "document_asset"}
    ]
    if not evidence_asset_ids:
        return []
    return [
        unit
        for unit in units
        if getattr(unit, "locator", {}).get("asset_id") in evidence_asset_ids
    ]


def _has_traceable_support(state: AgentState) -> bool:
    if state.get("evidence") or state.get("citations"):
        return True
    for ref in state.get("evidence_refs", []):
        citation_id = getattr(ref, "citation_id", None)
        citation_anchor = getattr(ref, "citation_anchor", None)
        if isinstance(citation_id, str) and citation_id.strip():
            return True
        if isinstance(citation_anchor, str) and citation_anchor.strip():
            return True
        evidence_id = getattr(ref, "evidence_id", None)
        if (
            getattr(ref, "source", None) == "asset"
            and isinstance(evidence_id, str)
            and evidence_id.startswith("asset:")
            and evidence_id.removeprefix("asset:").isdigit()
        ):
            return True
    return False


def _asset_source_line(unit: object) -> str:
    locator = getattr(unit, "locator", {})
    if not isinstance(locator, dict):
        return ""
    labels = {
        "asset_id": "asset_id",
        "sheet_name": "sheet",
        "element_ref": "element_ref",
        "page_no": "page",
        "doc_id": "doc_id",
        "source_id": "source_id",
        "section_id": "section_id",
    }
    return "；".join(
        f"{labels[field]}={value}"
        for field, value in locator.items()
        if field in labels
    )


def _legacy_synthesize_node(state: AgentState) -> dict[str, Any]:
    tool_results = state.get("tool_results", [])
    ok_results = [result for result in tool_results if result.status == "ok"]
    error_results = [result for result in tool_results if result.status == "error"]
    status = state.get("status")
    final_status = "failed" if status == "failed" else "done"
    return {
        "status": final_status,
        "final_answer": _final_answer(
            ok_results,
            error_results,
            status=status,
            stop_reason=state.get("stop_reason"),
        ),
        "groundedness_flag": bool(ok_results),
        "insufficient_evidence_flag": (
            state.get("insufficient_evidence_flag", False)
            or bool(error_results)
            or _has_insufficient_output(ok_results)
            or status == "failed"
        ),
    }


def _should_delegate_to_synthesis_agent(state: AgentState) -> bool:
    if state.get("status") == "failed":
        return False
    tool_results = state.get("tool_results", [])
    if _has_grounded_answer_tool_result(tool_results):
        return False
    return (
        bool(state.get("evidence"))
        or any(result.status == "ok" for result in tool_results)
    )


def _has_grounded_answer_tool_result(tool_results: list[ToolResult]) -> bool:
    for result in tool_results:
        if result.status != "ok" or result.tool_name != "rag_search_answer" or result.output is None:
            continue
        text = getattr(result.output, "text", None)
        if not isinstance(text, str) or not text.strip():
            continue
        if bool(getattr(result.output, "insufficient_evidence", False)):
            continue
        return True
    return False


def _synthesis_agent_update(state: AgentState, result: SynthesisRunResult) -> dict[str, Any]:
    fallback = _legacy_synthesize_node(state)
    if result.status != "done" or not result.final_answer:
        return {
            **fallback,
            "stop_reason": result.stop_reason or result.status,
            "insufficient_evidence_flag": True,
        }

    return {
        "status": "done",
        "final_answer": result.final_answer,
        "tool_results": result.tool_results,
        "evidence": result.evidence,
        "citations": result.citations,
        "groundedness_flag": (
            fallback["groundedness_flag"]
            or result.groundedness_flag
            or bool(result.evidence)
            or bool(result.citations)
        ),
        "insufficient_evidence_flag": (
            fallback["insufficient_evidence_flag"]
            or result.insufficient_evidence_flag
        ),
    }


def _final_answer(
    ok_results: list[ToolResult],
    error_results: list[ToolResult],
    *,
    status: str | None,
    stop_reason: str | None,
) -> str:
    answer_parts = [
        text
        for result in ok_results
        if result.output is not None and (text := _tool_result_text(result))
    ]
    if answer_parts:
        return "\n\n".join(answer_parts)
    if error_results:
        error_codes = ", ".join(
            result.error.code for result in error_results if result.error is not None
        )
        if error_codes:
            return f"No answer was generated because tool execution failed: {error_codes}."
        return "No answer was generated because tool execution failed."
    if status == "failed" and stop_reason:
        return f"Agent failed: {stop_reason}."
    return "No answer was generated because no tool results were available."


def _tool_result_text(result: ToolResult) -> str | None:
    if result.output is None:
        return None
    if result.tool_name == "structured_probe":
        if summary := _structured_probe_output_text(result.output):
            return summary
    return _output_text(result.output)


def _output_text(output: BaseModel) -> str:
    text = getattr(output, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return output.model_dump_json(exclude_none=True)


def _structured_probe_output_text(output: BaseModel) -> str | None:
    path = getattr(output, "path", None)
    tables = getattr(output, "tables", None)
    if not isinstance(path, str) or not isinstance(tables, list):
        return None

    lines = [f"表结构摘要：{path}"]
    file_kind = getattr(output, "file_kind", None)
    mime_type = getattr(output, "mime_type", None)
    file_parts = [
        str(part)
        for part in (file_kind, mime_type)
        if isinstance(part, str) and part.strip()
    ]
    if file_parts:
        lines.append("文件类型：" + "；".join(file_parts))

    if not tables:
        lines.append("未发现可解析的结构化表格。")
        return "\n".join(lines)

    for table in tables[:3]:
        lines.extend(_structured_table_summary_lines(table))

    remaining = len(tables) - 3
    if remaining > 0:
        lines.append(f"还有 {remaining} 个表未展开。")

    errors = getattr(output, "errors", None)
    if isinstance(errors, list) and errors:
        lines.append("探查警告：" + ", ".join(str(error) for error in errors[:3]))

    return "\n".join(lines)


def _structured_table_summary_lines(table: object) -> list[str]:
    table_index = getattr(table, "table_index", None)
    table_number = table_index + 1 if isinstance(table_index, int) else "?"
    name = getattr(table, "name", None)
    used_range = getattr(table, "used_range", None)
    row_count = getattr(table, "row_count", None)
    column_count = getattr(table, "column_count", None)

    labels = [f"表 {table_number}"]
    if isinstance(name, str) and name.strip():
        labels.append(name.strip())
    detail_parts = []
    if isinstance(used_range, str) and used_range.strip():
        detail_parts.append(f"范围 {used_range.strip()}")
    if isinstance(row_count, int) and isinstance(column_count, int):
        detail_parts.append(f"{row_count} 行 x {column_count} 列")

    title = "：".join(labels)
    if detail_parts:
        title += "（" + "，".join(detail_parts) + "）"
    lines = [title]

    header = _best_header_candidate(table)
    if header is not None:
        row_index = getattr(header, "row_index", None)
        confidence = getattr(header, "confidence", None)
        if isinstance(row_index, int):
            suffix = (
                f"（置信度 {confidence:.2f}）"
                if isinstance(confidence, int | float)
                else ""
            )
            lines.append(f"候选表头行：第 {row_index} 行{suffix}")
    else:
        lines.append("候选表头行：未识别")

    data_start_row = getattr(table, "data_start_row", None)
    if isinstance(data_start_row, int):
        lines.append(f"数据起始行：第 {data_start_row} 行")

    columns = _structured_table_columns(table, header)
    if columns:
        lines.append("关键字段：" + ", ".join(columns[:12]))
        if len(columns) > 12:
            lines[-1] += f", ...(+{len(columns) - 12})"

    if sample_judgement := _structured_table_sample_judgement(table, header):
        lines.append(sample_judgement)

    return lines


def _best_header_candidate(table: object) -> object | None:
    candidates = getattr(table, "candidate_header_rows", None)
    if not isinstance(candidates, list) or not candidates:
        return None
    candidate: object = candidates[0]
    return candidate


def _structured_table_columns(table: object, header: object | None) -> list[str]:
    rows = getattr(table, "sample_rows", None)
    if not isinstance(rows, list) or not rows:
        return []

    row_index = getattr(header, "row_index", None) if header is not None else None
    if isinstance(row_index, int) and 1 <= row_index <= len(rows):
        return _row_values(rows[row_index - 1])

    for row in rows:
        values = _row_values(row)
        if len(values) >= 2:
            return values
    return []


def _row_values(row: object) -> list[str]:
    if not isinstance(row, list):
        return []
    values: list[str] = []
    for value in row:
        text = _cell_text(value)
        if text:
            values.append(text)
    return values


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if not text:
        return ""
    return text if len(text) <= 40 else text[:37] + "..."


def _structured_table_sample_judgement(table: object, header: object | None) -> str | None:
    rows = getattr(table, "sample_rows", None)
    if not isinstance(rows, list) or not rows:
        return None
    row_count = len(rows)
    row_index = getattr(header, "row_index", None) if header is not None else None
    data_start_row = getattr(table, "data_start_row", None)

    parts: list[str] = []
    if isinstance(row_index, int):
        if row_index > 1:
            parts.append(f"第 1-{row_index - 1} 行更像标题或备注")
        parts.append(f"第 {row_index} 行更像表头")
    if isinstance(data_start_row, int):
        parts.append(f"第 {data_start_row} 行开始出现数据样本")
    if not parts:
        parts.append(f"已返回前 {row_count} 行有界样本")
    return "样本判断：" + "；".join(parts) + "。"


def _has_insufficient_output(ok_results: list[ToolResult]) -> bool:
    return any(
        bool(getattr(result.output, "insufficient_evidence", False))
        for result in ok_results
        if result.output is not None
    )
