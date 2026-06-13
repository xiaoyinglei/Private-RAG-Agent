from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from rag.agent.memory.models import MemoryRef
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem


class EvidenceRef(BaseModel):
    evidence_id: str | None = None
    citation_id: str | None = None
    citation_anchor: str | None = None
    doc_id: int | None = None
    source: str | None = None

    @property
    def key(self) -> str:
        return "|".join(
            value
            for value in (
                self.evidence_id,
                self.citation_id,
                self.citation_anchor,
                None if self.doc_id is None else str(self.doc_id),
                self.source,
            )
            if value
        )


class ContextUnit(BaseModel):
    unit_id: str
    unit_type: str
    locator: dict[str, object] = Field(default_factory=dict)
    preview: str | dict[str, object] | None = None
    content_ref: str | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.unit_id


class AnswerCandidate(BaseModel):
    text: str
    source_tool_call_id: str | None = None
    source_tool_name: str | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


class ComputationResult(BaseModel):
    source_tool_call_id: str
    source_tool_name: str
    operation: str | None = None
    value_preview: str | None = None
    expression: str | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return self.source_tool_call_id


class ContextBinding(BaseModel):
    binding_id: str
    constraint_id: str
    unit_id: str | None = None
    status: Literal["satisfied", "ambiguous", "violated"]
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    rationale: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.binding_id


class StructuredObservation(BaseModel):
    tool_call_id: str
    tool_name: str
    status: Literal["ok", "error"]
    answer_candidate: AnswerCandidate | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    context_units: list[ContextUnit] = Field(default_factory=list)
    locators: list[dict[str, object]] = Field(default_factory=list)
    asset_refs: list[int] = Field(default_factory=list)
    operation: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    raw_result_ref: str
    raw_memory_ref: MemoryRef | None = None
    related_step_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class ObservationError(BaseModel):
    tool_call_id: str
    tool_name: str
    code: str
    message: str
    retryable: bool
    detail: dict[str, object] = Field(default_factory=dict)


class ObservationBatch(BaseModel):
    structured_observations: list[StructuredObservation] = Field(default_factory=list)
    answer_candidates: list[AnswerCandidate] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    computation_results: list[ComputationResult] = Field(default_factory=list)
    context_units: list[ContextUnit] = Field(default_factory=list)
    locators: list[dict[str, object]] = Field(default_factory=list)
    asset_refs: list[int] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    errors: list[ObservationError] = Field(default_factory=list)

    def as_state_update(self) -> dict[str, object]:
        return {
            "structured_observations": list(self.structured_observations),
            "answer_candidates": list(self.answer_candidates),
            "evidence_refs": list(self.evidence_refs),
            "computation_results": list(self.computation_results),
            "context_units": list(self.context_units),
            "locators": list(self.locators),
            "asset_refs": list(self.asset_refs),
            "evidence": list(self.evidence),
            "citations": list(self.citations),
            "errors": [
                error.model_dump(mode="json")
                for error in self.errors
            ],
        }


class ObservationBuilder:
    def from_tool_result(self, result: ToolResult) -> StructuredObservation:
        if result.status == "error":
            error_message = (
                result.error.message
                if result.error is not None
                else "unknown tool error"
            )
            return StructuredObservation(
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                status="error",
                error=error_message,
                raw_result_ref=result.tool_call_id,
            )

        output = result.output
        observation_only = bool(getattr(output, "observation_only", False))
        if observation_only:
            evidence_refs: list[EvidenceRef] = []
        elif result.tool_name.startswith("agent_"):
            evidence_refs = _delegated_evidence_refs_from_output(output)
        else:
            evidence_refs = _dedupe_evidence_refs(
                [
                    *_evidence_refs_from_output(output),
                    *_search_evidence_refs_from_output(output),
                ]
            )
        answer_text = None if observation_only else _answer_text(result.tool_name, output)
        answer = (
            AnswerCandidate(
                text=answer_text,
                source_tool_call_id=result.tool_call_id,
                source_tool_name=result.tool_name,
                evidence_refs=evidence_refs,
            )
            if answer_text
            else None
        )
        locators = _locators_from_output(output, tool_name=result.tool_name)
        return StructuredObservation(
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            status="ok",
            answer_candidate=answer,
            evidence_refs=evidence_refs,
            context_units=_context_units_from_output(
                result,
                evidence_refs=evidence_refs,
                locators=locators,
            ),
            locators=locators,
            asset_refs=_asset_refs_from_output(output),
            operation=_operation_from_output(output),
            raw_result_ref=result.tool_call_id,
        )


class ObservationExtractor:
    """Reduce tool outcomes into provenance channels without loop control decisions."""

    def __init__(self, observation_builder: ObservationBuilder | None = None) -> None:
        self._observation_builder = observation_builder or ObservationBuilder()

    def extract(
        self,
        tool_results: Sequence[ToolResult],
        *,
        seen_tool_call_ids: Sequence[str] = (),
    ) -> ObservationBatch:
        seen = set(seen_tool_call_ids)
        selected_results = [
            result
            for result in tool_results
            if result.tool_call_id not in seen
        ]
        observations = [
            self._observation_builder.from_tool_result(result)
            for result in selected_results
        ]
        if not observations:
            return ObservationBatch()

        results_by_id = {
            result.tool_call_id: result
            for result in selected_results
        }
        computation_results = [
            ComputationResult(
                source_tool_call_id=observation.tool_call_id,
                source_tool_name=observation.tool_name,
                operation=observation.operation,
                value_preview=(
                    None
                    if observation.answer_candidate is None
                    else observation.answer_candidate.text[:300]
                ),
                expression=_computation_expression(
                    results_by_id.get(observation.tool_call_id)
                ),
                evidence_refs=observation.evidence_refs,
            )
            for observation in observations
            if observation.operation is not None
        ]
        return ObservationBatch(
            structured_observations=observations,
            answer_candidates=[
                observation.answer_candidate
                for observation in observations
                if observation.answer_candidate is not None
            ],
            evidence_refs=[
                ref
                for observation in observations
                for ref in observation.evidence_refs
            ],
            computation_results=computation_results,
            context_units=_dedupe_context_units(
                [
                    unit
                    for observation in observations
                    for unit in observation.context_units
                ]
            ),
            locators=[
                locator
                for observation in observations
                for locator in observation.locators
            ],
            asset_refs=[
                asset_ref
                for observation in observations
                for asset_ref in observation.asset_refs
            ],
            evidence=_evidence_from_outputs(observations, selected_results),
            citations=_citations_from_outputs(observations, selected_results),
            errors=_errors_from_results(selected_results),
        )

    def reduce_tool_results(self, state: dict[str, Any]) -> dict[str, Any]:
        seen = [
            observation.tool_call_id
            for observation in state.get("structured_observations", [])
            if isinstance(observation, StructuredObservation)
        ]
        batch = self.extract(
            list(state.get("tool_results", [])),
            seen_tool_call_ids=seen,
        )
        if not batch.structured_observations:
            return {}
        return cast(dict[str, Any], batch.as_state_update())


def _answer_text(tool_name: str, output: BaseModel | None) -> str | None:
    if output is None:
        return None
    if tool_name.startswith("agent_"):
        conclusion = getattr(output, "conclusion", None)
        if isinstance(conclusion, str) and conclusion.strip():
            return conclusion.strip()
        return None
    if tool_name in {
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "graph_expand",
        "asset_list",
        "asset_inspect",
        "asset_read_slice",
        "list_files",
        "read_file",
        "write_file",
        "run_python",
        "structured_probe",
    }:
        return None
    text = getattr(output, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    markdown = getattr(output, "markdown", None)
    if isinstance(markdown, str) and markdown.strip():
        return markdown.strip()
    return output.model_dump_json(exclude_none=True)


def _evidence_refs_from_output(output: BaseModel | None) -> list[EvidenceRef]:
    if output is None:
        return []
    refs: list[EvidenceRef] = []
    for evidence_id in getattr(output, "evidence_ids", []) or []:
        refs.append(EvidenceRef(evidence_id=str(evidence_id), source="tool_output"))
    for citation_id in getattr(output, "citation_ids", []) or []:
        refs.append(EvidenceRef(citation_id=str(citation_id), source="tool_output"))
    for evidence in getattr(output, "evidence", []) or []:
        evidence_item = EvidenceItem.model_validate(evidence)
        refs.append(
            EvidenceRef(
                evidence_id=evidence_item.evidence_id,
                citation_anchor=evidence_item.citation_anchor,
                doc_id=evidence_item.doc_id,
                source="evidence",
            )
        )
    for citation in getattr(output, "citations", []) or []:
        citation_item = AnswerCitation.model_validate(citation)
        refs.append(
            EvidenceRef(
                evidence_id=citation_item.evidence_id,
                citation_id=citation_item.citation_id,
                citation_anchor=citation_item.citation_anchor,
                doc_id=citation_item.doc_id,
                source="citation",
            )
        )
    locator = getattr(output, "locator", None)
    if locator is not None:
        locator_anchor = getattr(locator, "citation_anchor", None)
        refs.append(
            EvidenceRef(
                citation_anchor=str(locator_anchor) if locator_anchor else None,
                source="locator",
            )
        )
    asset_id = getattr(output, "asset_id", None)
    if isinstance(asset_id, int) and asset_id > 0:
        refs.append(EvidenceRef(evidence_id=f"asset:{asset_id}", source="asset"))
    return _dedupe_evidence_refs(refs)


def _delegated_evidence_refs_from_output(
    output: BaseModel | None,
) -> list[EvidenceRef]:
    if output is None:
        return []
    refs: list[EvidenceRef] = []
    for item in getattr(output, "evidence_refs", []) or []:
        evidence_id = getattr(item, "evidence_id", None)
        citation_id = getattr(item, "citation_id", None)
        citation_anchor = getattr(item, "citation_anchor", None)
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            continue
        if not (
            isinstance(citation_id, str) and citation_id.strip()
        ) and not (
            isinstance(citation_anchor, str) and citation_anchor.strip()
        ):
            continue
        doc_id = getattr(item, "doc_id", None)
        refs.append(
            EvidenceRef(
                evidence_id=evidence_id.strip(),
                citation_id=(
                    citation_id.strip()
                    if isinstance(citation_id, str)
                    else None
                ),
                citation_anchor=(
                    citation_anchor.strip()
                    if isinstance(citation_anchor, str)
                    else None
                ),
                doc_id=doc_id if isinstance(doc_id, int) else None,
                source="delegated_agent",
            )
        )
    for citation in getattr(output, "citations", []) or []:
        citation_item = AnswerCitation.model_validate(citation)
        if any(ref.evidence_id == citation_item.evidence_id for ref in refs):
            continue
        refs.append(
            EvidenceRef(
                evidence_id=citation_item.evidence_id,
                citation_id=citation_item.citation_id,
                citation_anchor=citation_item.citation_anchor,
                doc_id=citation_item.doc_id,
                source="delegated_agent",
            )
        )
    return _dedupe_evidence_refs(refs)


def _search_evidence_refs_from_output(
    output: BaseModel | None,
) -> list[EvidenceRef]:
    if output is None:
        return []
    items = getattr(output, "items", None)
    if not isinstance(items, list):
        return []
    refs: list[EvidenceRef] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ref = EvidenceRef(
            evidence_id=(
                str(item["evidence_id"])
                if item.get("evidence_id")
                else None
            ),
            citation_anchor=(
                str(item["citation_anchor"])
                if item.get("citation_anchor")
                else None
            ),
            doc_id=item["doc_id"] if isinstance(item.get("doc_id"), int) else None,
            source="retrieval",
        )
        if ref.key:
            refs.append(ref)
    return _dedupe_evidence_refs(refs)


def _context_units_from_output(
    result: ToolResult,
    *,
    evidence_refs: Sequence[EvidenceRef],
    locators: Sequence[dict[str, object]],
) -> list[ContextUnit]:
    output = result.output
    if output is None:
        return []
    if result.tool_name in {
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "graph_expand",
    }:
        return _retrieval_context_units(result, evidence_refs=evidence_refs)
    if result.tool_name in {"asset_list", "asset_inspect"}:
        return [
            _asset_context_unit(result.tool_name, locator)
            for locator in locators
            if isinstance(locator.get("asset_id"), int)
        ]
    if result.tool_name == "asset_analyze":
        preview = _answer_text(result.tool_name, output)
        units = [
            ContextUnit(
                unit_id=f"computed:{result.tool_call_id}",
                unit_type="computed_result",
                preview=preview[:1000] if preview else None,
                content_ref=result.tool_call_id,
                evidence_refs=list(evidence_refs),
                metadata={"source_tool": result.tool_name},
            )
        ]
        if not bool(getattr(output, "observation_only", False)):
            units.extend(
                _asset_context_unit(result.tool_name, locator)
                for locator in locators
                if isinstance(locator.get("asset_id"), int)
                and isinstance(locator.get("asset_type"), str)
            )
        return units
    if result.tool_name == "list_files":
        return _list_files_context_units(output)
    if result.tool_name == "read_file":
        unit = _read_file_context_unit(output, tool_call_id=result.tool_call_id)
        return [] if unit is None else [unit]
    if result.tool_name == "write_file":
        unit = _write_file_context_unit(output, tool_call_id=result.tool_call_id)
        return [] if unit is None else [unit]
    if result.tool_name == "run_python":
        return _run_python_context_units(output, tool_call_id=result.tool_call_id)
    if result.tool_name == "structured_probe":
        return _structured_probe_context_units(output)
    return []


def _retrieval_context_units(
    result: ToolResult,
    *,
    evidence_refs: Sequence[EvidenceRef],
) -> list[ContextUnit]:
    items = getattr(result.output, "items", None)
    if not isinstance(items, list):
        return []
    units: list[ContextUnit] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        locator = _retrieval_locator(item)
        item_refs = _search_evidence_refs_from_output(
            _SingleSearchItemOutput(items=[item])
        )
        record_type = str(item.get("record_type", "") or "")
        unit_type = (
            "document_section"
            if record_type == "section"
            else "retrieved_chunk"
        )
        identifier = (
            str(item["evidence_id"])
            if item.get("evidence_id")
            else f"{result.tool_call_id}:{index}"
        )
        text = str(item.get("text", "") or "")
        capabilities = ["text_extract", "text_synthesize", "quote"]
        if "ASSET_ANCHOR:" in text or "asset" in record_type:
            capabilities.append("asset_list")
        units.append(
            ContextUnit(
                unit_id=f"retrieval:{identifier}",
                unit_type=unit_type,
                locator=locator,
                preview=text[:1000] if text else None,
                content_ref=(
                    str(item["evidence_id"])
                    if item.get("evidence_id")
                    else result.tool_call_id
                ),
                evidence_refs=item_refs or list(evidence_refs),
                capabilities=capabilities,
                metadata={"source_tool": result.tool_name},
            )
        )
    return units


class _SingleSearchItemOutput(BaseModel):
    items: list[dict[str, object]]


def _retrieval_locator(item: dict[str, object]) -> dict[str, object]:
    return {
        field: item[field]
        for field in (
            "doc_id",
            "source_id",
            "section_id",
            "page_start",
            "page_end",
            "record_type",
            "citation_anchor",
            "evidence_id",
            "score",
            "rerank_score",
            "retrieval_channels",
            "retrieval_family",
        )
        if item.get(field) not in (None, "", [])
    }


def _list_files_context_units(output: BaseModel) -> list[ContextUnit]:
    units: list[ContextUnit] = []
    for file_info in getattr(output, "files", []) or []:
        locator = _workspace_file_locator(file_info, source_tool="list_files")
        path = locator.get("path")
        if not isinstance(path, str) or not path.strip():
            continue
        is_dir = bool(locator.get("is_dir", False))
        units.append(
            ContextUnit(
                unit_id=(
                    f"workspace_dir:{path}"
                    if is_dir
                    else f"workspace_file:{path}"
                ),
                unit_type="workspace_dir" if is_dir else "workspace_file",
                locator=locator,
                preview=f"{path} ({locator.get('size_bytes', 0)} bytes)",
                content_ref=path,
                capabilities=_workspace_file_capabilities(
                    file_info,
                    is_dir=is_dir,
                ),
                metadata={"source_tool": "list_files"},
            )
        )
    return units


def _read_file_context_unit(
    output: BaseModel,
    *,
    tool_call_id: str,
) -> ContextUnit | None:
    path = getattr(output, "path", None)
    if not isinstance(path, str) or not path.strip():
        return None
    content = getattr(output, "content", None)
    return ContextUnit(
        unit_id=f"workspace_file:{path}",
        unit_type="workspace_file_content",
        locator=_read_file_locator(output),
        preview=(
            content[:1000]
            if isinstance(content, str) and content
            else None
        ),
        content_ref=tool_call_id,
        capabilities=["read_file"],
        metadata={"source_tool": "read_file"},
    )


def _write_file_context_unit(
    output: BaseModel,
    *,
    tool_call_id: str,
) -> ContextUnit | None:
    path = getattr(output, "path", None)
    if not isinstance(path, str) or not path.strip():
        return None
    locator = _write_file_locator(output)
    return ContextUnit(
        unit_id=f"workspace_file:{path}",
        unit_type="workspace_file",
        locator=locator,
        preview=f"wrote {path} ({locator.get('size_bytes', 0)} bytes)",
        content_ref=tool_call_id,
        capabilities=["read_file"],
        metadata={"source_tool": "write_file"},
    )


def _run_python_context_units(
    output: BaseModel,
    *,
    tool_call_id: str,
) -> list[ContextUnit]:
    units = [
        ContextUnit(
            unit_id=f"python_run:{tool_call_id}",
            unit_type="python_execution",
            locator=_run_python_locator(output),
            preview=_run_python_preview(output),
            content_ref=tool_call_id,
            capabilities=["run_python"],
            metadata={"source_tool": "run_python"},
        )
    ]
    for path in getattr(output, "generated_files", []) or []:
        if not isinstance(path, str) or not path.strip():
            continue
        units.append(
            ContextUnit(
                unit_id=f"workspace_file:{path}",
                unit_type="workspace_file",
                locator={
                    "path": path,
                    "source_tool": "run_python",
                    "generated_by": tool_call_id,
                },
                preview=f"generated {path}",
                content_ref=path,
                capabilities=["read_file"],
                metadata={"source_tool": "run_python"},
            )
        )
    return units


def _structured_probe_context_units(output: BaseModel) -> list[ContextUnit]:
    path = getattr(output, "path", None)
    if not isinstance(path, str) or not path.strip():
        return []
    units: list[ContextUnit] = []
    for table in getattr(output, "tables", []) or []:
        table_index = getattr(table, "table_index", len(units))
        units.append(
            ContextUnit(
                unit_id=f"structured_table:{path}:{table_index}",
                unit_type="structured_table",
                locator=_structured_table_locator(path, table),
                preview=_structured_table_preview(table),
                content_ref=path,
                capabilities=["structured_probe", "run_python"],
                metadata={"source_tool": "structured_probe"},
            )
        )
    return units


def _asset_context_unit(
    tool_name: str,
    locator: dict[str, object],
) -> ContextUnit:
    asset_id = locator.get("asset_id")
    if not isinstance(asset_id, int):
        raise ValueError("asset context unit requires integer asset_id")
    asset_type = str(locator.get("asset_type", "") or "")
    unit_type = {
        "table": "table_asset",
        "image": "image_asset",
    }.get(asset_type, "document_asset")
    advertised = locator.get("analysis_capabilities", [])
    analysis_capabilities = (
        [str(capability) for capability in advertised]
        if isinstance(advertised, list)
        else []
    )
    capabilities = list(
        dict.fromkeys(
            [
                *(["asset_inspect"] if tool_name == "asset_list" else []),
                *analysis_capabilities,
            ]
        )
    )
    preview_fields = {
        field: locator[field]
        for field in ("columns", "row_count", "column_count", "head_rows")
        if locator.get(field) not in (None, "", [])
    }
    return ContextUnit(
        unit_id=f"asset:{asset_id}",
        unit_type=unit_type,
        locator=dict(locator),
        preview=preview_fields or None,
        content_ref=f"asset:{asset_id}",
        capabilities=capabilities,
        metadata={
            "source_tool": tool_name,
            "inspection_status": {
                "asset_list": "listed",
                "asset_inspect": "inspected",
                "asset_analyze": "analyzed",
            }.get(tool_name, "observed"),
        },
    )


def _dedupe_context_units(
    units: Sequence[ContextUnit],
) -> list[ContextUnit]:
    return list({unit.unit_id: unit for unit in units}.values())


def _locators_from_output(
    output: BaseModel | None,
    *,
    tool_name: str | None = None,
) -> list[dict[str, object]]:
    if output is None:
        return []
    primitive_locators = _primitive_locators_from_output(tool_name, output)
    if primitive_locators:
        return primitive_locators
    items = getattr(output, "items", None)
    if isinstance(items, list):
        return _search_asset_locators(items)
    locator = getattr(output, "locator", None)
    if locator is not None and hasattr(locator, "model_dump"):
        return [locator.model_dump(mode="json", exclude_none=True)]
    asset_id = getattr(output, "asset_id", None)
    if isinstance(asset_id, int) and asset_id > 0:
        values: dict[str, object] = {"asset_id": asset_id}
        for field in (
            "doc_id",
            "source_id",
            "section_id",
            "asset_type",
            "sheet_name",
            "page_no",
            "element_ref",
            "caption",
            "analysis_capabilities",
            "columns",
            "row_count",
            "column_count",
        ):
            value = getattr(output, field, None)
            if value not in (None, "", []):
                values[field] = value
        head_rows = getattr(output, "head_rows", None)
        if head_rows:
            values["head_rows"] = head_rows
        return [values]
    assets = getattr(output, "assets", None)
    if not isinstance(assets, list):
        return []
    return [
        _asset_locator_from_descriptor(asset)
        for asset in assets
        if hasattr(asset, "model_dump")
    ]


def _primitive_locators_from_output(
    tool_name: str | None,
    output: BaseModel,
) -> list[dict[str, object]]:
    if tool_name == "list_files":
        return [
            _workspace_file_locator(file_info, source_tool="list_files")
            for file_info in getattr(output, "files", []) or []
        ]
    if tool_name == "read_file":
        return [_read_file_locator(output)]
    if tool_name == "write_file":
        return [_write_file_locator(output)]
    if tool_name == "run_python":
        locators = [_run_python_locator(output)]
        locators.extend(
            {
                "path": path,
                "source_tool": "run_python",
                "generated": True,
            }
            for path in getattr(output, "generated_files", []) or []
            if isinstance(path, str) and path.strip()
        )
        return locators
    if tool_name == "structured_probe":
        path = getattr(output, "path", None)
        if not isinstance(path, str) or not path.strip():
            return []
        return [
            _structured_table_locator(path, table)
            for table in getattr(output, "tables", []) or []
        ]
    return []


def _workspace_file_locator(
    file_info: object,
    *,
    source_tool: str,
) -> dict[str, object]:
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
    has_file_kind = (
        isinstance(file_kind, str)
        and file_kind not in {"", "unknown"}
    )
    if has_file_kind:
        values["file_kind"] = file_kind
        for field in ("is_binary", "readable_as_text"):
            value = getattr(file_info, field, None)
            if isinstance(value, bool):
                values[field] = value
    return values


def _workspace_file_capabilities(
    file_info: object,
    *,
    is_dir: bool,
) -> list[str]:
    raw = getattr(file_info, "capabilities", None)
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw if str(item)]
    return ["list_files"] if is_dir else ["read_file"]


def _read_file_locator(output: BaseModel) -> dict[str, object]:
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


def _write_file_locator(output: BaseModel) -> dict[str, object]:
    values: dict[str, object] = {"source_tool": "write_file"}
    for field in ("path", "size_bytes"):
        value = getattr(output, field, None)
        if value not in (None, "", []):
            values[field] = value
    return values


def _run_python_locator(output: BaseModel) -> dict[str, object]:
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


def _run_python_preview(output: BaseModel) -> str | None:
    lines: list[str] = []
    stdout = getattr(output, "stdout", None)
    if isinstance(stdout, str) and stdout.strip():
        lines.append("stdout: " + stdout.strip()[:500])
    stderr = getattr(output, "stderr", None)
    if isinstance(stderr, str) and stderr.strip():
        lines.append("stderr: " + stderr.strip()[:500])
    generated = getattr(output, "generated_files", None)
    if isinstance(generated, list) and generated:
        lines.append(
            "generated_files: "
            + ", ".join(str(path) for path in generated[:20])
        )
    return "\n".join(lines) if lines else None


def _structured_table_locator(
    path: str,
    table: object,
) -> dict[str, object]:
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


def _structured_table_preview(table: object) -> str | None:
    rows = getattr(table, "sample_rows", None)
    row_count = getattr(table, "row_count", None)
    column_count = getattr(table, "column_count", None)
    used_range = getattr(table, "used_range", None)
    parts = [
        f"rows={row_count}" if isinstance(row_count, int) else "",
        f"columns={column_count}" if isinstance(column_count, int) else "",
        (
            f"used_range={used_range}"
            if isinstance(used_range, str) and used_range
            else ""
        ),
    ]
    header_row = _header_sample_row(table, rows)
    if header_row is not None:
        parts.append(f"header_row={_bounded_row_preview(header_row)}")
    elif isinstance(rows, list) and rows:
        parts.append(f"first_row={_bounded_row_preview(rows[0])}")
    preview = " ".join(part for part in parts if part)
    return preview or None


def _header_sample_row(table: object, rows: object) -> object | None:
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
    if not isinstance(row, list):
        return _bounded_cell_preview(row)
    cells = [_bounded_cell_preview(cell) for cell in row[:8]]
    suffix = f", ...(+{len(row) - 8})" if len(row) > 8 else ""
    return "[" + ", ".join(cells) + suffix + "]"


def _bounded_cell_preview(cell: object) -> str:
    text = str(cell)
    if len(text) > 40:
        text = text[:40].rstrip() + "..."
    return repr(text)


def _asset_locator_from_descriptor(asset: object) -> dict[str, object]:
    values: dict[str, object] = {}
    for field in (
        "asset_id",
        "doc_id",
        "source_id",
        "section_id",
        "asset_type",
        "page_no",
        "element_ref",
        "sheet_name",
        "caption",
        "row_count",
        "column_count",
        "columns",
        "analysis_capabilities",
    ):
        value = getattr(asset, field, None)
        if value not in (None, "", []):
            values[field] = value
    return values


def _asset_refs_from_output(output: BaseModel | None) -> list[int]:
    if output is None:
        return []
    assets = getattr(output, "assets", None)
    if isinstance(assets, list):
        return [
            asset_id
            for asset in assets
            if isinstance((asset_id := getattr(asset, "asset_id", None)), int)
            and asset_id > 0
        ]
    asset_id = getattr(output, "asset_id", None)
    return [asset_id] if isinstance(asset_id, int) and asset_id > 0 else []


def _search_asset_locators(
    items: Sequence[object],
) -> list[dict[str, object]]:
    locators: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "")
        record_type = str(item.get("record_type", "") or "")
        if (
            "ASSET_ANCHOR:" not in text
            and "asset" not in record_type
            and "section" not in record_type
        ):
            continue
        locator = _retrieval_locator(item)
        if locator.get("section_id") is not None:
            locators.append(locator)
    return locators


def _operation_from_output(output: BaseModel | None) -> str | None:
    if output is None:
        return None
    operation = getattr(output, "operation", None)
    return str(operation) if operation else None


def _computation_expression(result: ToolResult | None) -> str | None:
    if result is None or result.output is None:
        return None
    query = getattr(result.output, "query", None)
    if not isinstance(query, str) or not query.strip():
        return None
    return query.strip()[:1000]


def _dedupe_evidence_refs(
    refs: Sequence[EvidenceRef],
) -> list[EvidenceRef]:
    deduped: dict[str, EvidenceRef] = {}
    for ref in refs:
        if ref.key:
            deduped.setdefault(ref.key, ref)
    return list(deduped.values())


def _evidence_from_outputs(
    observations: Sequence[StructuredObservation],
    tool_results: Sequence[ToolResult],
) -> list[EvidenceItem]:
    observed_ids = {observation.tool_call_id for observation in observations}
    return [
        EvidenceItem.model_validate(item)
        for result in tool_results
        if result.tool_call_id in observed_ids and result.output is not None
        for item in getattr(result.output, "evidence", []) or []
    ]


def _citations_from_outputs(
    observations: Sequence[StructuredObservation],
    tool_results: Sequence[ToolResult],
) -> list[AnswerCitation]:
    observed_ids = {observation.tool_call_id for observation in observations}
    return [
        AnswerCitation.model_validate(item)
        for result in tool_results
        if result.tool_call_id in observed_ids and result.output is not None
        for item in getattr(result.output, "citations", []) or []
    ]


def _errors_from_results(
    tool_results: Sequence[ToolResult],
) -> list[ObservationError]:
    return [
        ObservationError(
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            code=result.error.code,
            message=result.error.message,
            retryable=result.error.retryable,
            detail=dict(result.error.detail),
        )
        for result in tool_results
        if result.status == "error" and result.error is not None
    ]


__all__ = [
    "AnswerCandidate",
    "ComputationResult",
    "ContextBinding",
    "ContextUnit",
    "EvidenceRef",
    "ObservationBatch",
    "ObservationBuilder",
    "ObservationError",
    "ObservationExtractor",
    "StructuredObservation",
]
