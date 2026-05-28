from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal, Protocol, Self, TypedDict

from pydantic import BaseModel, Field, model_validator

from rag.agent.state import ToolCallPlan
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem

RequirementId = Literal["answer", "evidence"]
DeliverableKind = Literal["answer", "evidence", "computation"]
AcceptanceRule = Literal[
    "non_empty_answer",
    "traceable_evidence",
    "reproducible_computation",
]


class GoalConstraint(BaseModel):
    constraint_id: str
    constraint_type: str
    expected_value: object
    required: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)


class GoalDeliverable(BaseModel):
    """One required output and the rule used to accept it."""

    deliverable_id: str
    kind: DeliverableKind
    acceptance_rule: AcceptanceRule
    required: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.deliverable_id

    @model_validator(mode="after")
    def validate_acceptance_rule(self) -> Self:
        expected_rules: dict[DeliverableKind, AcceptanceRule] = {
            "answer": "non_empty_answer",
            "evidence": "traceable_evidence",
            "computation": "reproducible_computation",
        }
        expected = expected_rules[self.kind]
        if self.acceptance_rule != expected:
            raise ValueError(
                f"acceptance_rule {self.acceptance_rule!r} is invalid for {self.kind!r}; "
                f"expected {expected!r}"
            )
        return self


class GoalSpec(BaseModel):
    """Completion contract for a user goal.

    This is not a task-type route. It only describes what must be true before
    the agent can stop.
    """

    original_query: str
    deliverables: list[GoalDeliverable] = Field(default_factory=list)
    # Backward-compatible inputs are normalized once into deliverables.
    required_outputs: list[str] = Field(default_factory=lambda: ["answer"])
    required_evidence: list[str] = Field(default_factory=list)
    required_operations: list[str] = Field(default_factory=list)
    constraints: list[GoalConstraint] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_legacy_deliverables(self) -> Self:
        if not self.deliverables:
            deliverables: list[GoalDeliverable] = []
            if self.required_outputs:
                deliverables.append(_answer_deliverable())
            if self.required_evidence:
                deliverables.append(_evidence_deliverable())
            if self.required_operations:
                deliverables.append(_computation_deliverable())
            self.deliverables = deliverables
        deliverable_ids = [deliverable.deliverable_id for deliverable in self.deliverables]
        if len(deliverable_ids) != len(set(deliverable_ids)):
            raise ValueError("duplicate deliverable_id values are not allowed")
        return self

    @property
    def requirement_ids(self) -> list[str]:
        requirements = [
            f"constraint:{constraint.constraint_id}"
            for constraint in self.constraints
            if constraint.required
        ]
        requirements.extend(
            deliverable.deliverable_id
            for deliverable in self.deliverables
            if deliverable.required
        )
        return requirements

    def open_gaps(self, satisfied_requirements: Sequence[str] = ()) -> list[GoalGap]:
        descriptions = {
            "answer": "Produce an answer.",
            "evidence": "Provide traceable evidence.",
            "computation": "Provide a reproducible computation with traceable evidence.",
        }
        gaps: list[GoalGap] = []
        constraints_by_requirement = {
            f"constraint:{constraint.constraint_id}": constraint
            for constraint in self.constraints
            if constraint.required
        }
        for requirement_id in self.requirement_ids:
            if requirement_id in satisfied_requirements:
                continue
            constraint = constraints_by_requirement.get(requirement_id)
            if constraint is not None:
                gaps.append(
                    GoalGap(
                        gap_id=requirement_id,
                        gap_type="context_binding",
                        description="Bind a context unit satisfying the required constraint.",
                        metadata={"constraint_id": constraint.constraint_id},
                    )
                )
                continue
            deliverable = next(
                (
                    item
                    for item in self.deliverables
                    if item.required and item.deliverable_id == requirement_id
                ),
                None,
            )
            gaps.append(
                GoalGap(
                    gap_id=requirement_id,
                    gap_type=requirement_id if deliverable is None else deliverable.kind,
                    description=descriptions.get(requirement_id, f"Satisfy {requirement_id}."),
                )
            )
        return gaps


class GoalGap(BaseModel):
    gap_id: str
    gap_type: str
    description: str
    required: bool = True
    related_unit_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.gap_id


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


class GoalConflict(BaseModel):
    conflict_id: str
    description: str
    related_unit_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


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
    resolved_gaps: list[str] = Field(default_factory=list)
    produced_gaps: list[str] = Field(default_factory=list)


class RuntimeState(TypedDict, total=False):
    task: str
    goal_spec: GoalSpec
    tool_results: list[ToolResult]
    structured_observations: list[StructuredObservation]
    context_units: list[ContextUnit]
    answer_candidates: list[AnswerCandidate]
    evidence_refs: list[EvidenceRef]
    computation_results: list[ComputationResult]
    context_bindings: list[ContextBinding]
    satisfied_requirements: list[str]
    open_gaps: list[GoalGap]
    conflicts: list[GoalConflict]
    pending_tool_calls: list[ToolCallPlan]
    no_progress_count: int
    iteration: int


class SatisfactionReport(BaseModel):
    is_done: bool = False
    open_gaps: list[GoalGap] = Field(default_factory=list)
    conflicts: list[GoalConflict] = Field(default_factory=list)
    is_stuck: bool = False
    reason: str


class GoalInitializer:
    def initialize(self, query: str) -> GoalSpec:
        stripped = query.strip()
        required_evidence: list[str] = []
        deliverables = [_answer_deliverable()]
        if _explicitly_requests_evidence(stripped):
            required_evidence.append("citation")
            deliverables.append(_evidence_deliverable())
        return GoalSpec(
            original_query=stripped,
            deliverables=deliverables,
            required_evidence=required_evidence,
            constraints=_explicit_context_title_constraints(stripped),
            success_criteria=_success_criteria(required_evidence=required_evidence),
        )


class ObservationBuilder:
    def from_tool_result(self, result: ToolResult) -> StructuredObservation:
        if result.status == "error":
            error_message = result.error.message if result.error is not None else "unknown tool error"
            return StructuredObservation(
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                status="error",
                error=error_message,
                raw_result_ref=result.tool_call_id,
                produced_gaps=["tool_error"],
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
        resolved_gaps: list[str] = []
        if answer is not None:
            resolved_gaps.append("answer")
        if evidence_refs:
            resolved_gaps.append("evidence")

        locators = _locators_from_output(output)
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
            resolved_gaps=resolved_gaps,
        )


class StateReducer:
    def __init__(self, observation_builder: ObservationBuilder | None = None) -> None:
        self._observation_builder = observation_builder or ObservationBuilder()

    def reduce_tool_results(self, state: dict[str, Any]) -> dict[str, Any]:
        tool_results = list(state.get("tool_results", []))
        seen_observations = {
            observation.tool_call_id
            for observation in state.get("structured_observations", [])
            if isinstance(observation, StructuredObservation)
        }
        new_observations = [
            self._observation_builder.from_tool_result(result)
            for result in tool_results
            if result.tool_call_id not in seen_observations
        ]
        if not new_observations:
            return {}

        goal = _goal_from_state(state)
        answer_candidates = [
            observation.answer_candidate
            for observation in new_observations
            if observation.answer_candidate is not None
        ]
        evidence_refs = [
            ref
            for observation in new_observations
            for ref in observation.evidence_refs
        ]
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
                    next(
                        (
                            result
                            for result in tool_results
                            if result.tool_call_id == observation.tool_call_id
                        ),
                        None,
                    )
                ),
                evidence_refs=observation.evidence_refs,
            )
            for observation in new_observations
            if observation.operation is not None
        ]
        context_units = _dedupe_context_units(
            [
                unit
                for observation in new_observations
                for unit in observation.context_units
            ]
        )
        locators = [
            locator
            for observation in new_observations
            for locator in observation.locators
        ]
        asset_refs = [
            asset_ref
            for observation in new_observations
            for asset_ref in observation.asset_refs
        ]

        all_answers = [*state.get("answer_candidates", []), *answer_candidates]
        all_evidence_refs = [*state.get("evidence_refs", []), *evidence_refs]
        all_computation_results = [
            *state.get("computation_results", []),
            *computation_results,
        ]
        satisfied = _satisfied_requirements(
            goal,
            all_answers,
            all_evidence_refs,
            state.get("context_bindings", []),
            all_computation_results,
        )
        previous_satisfied = set(state.get("satisfied_requirements", []))
        previous_accepted_refs = {
            ref.key
            for ref in state.get("evidence_refs", [])
            if isinstance(ref, EvidenceRef) and _is_traceable_evidence_ref(ref)
        }
        accepted_refs = {
            ref.key
            for ref in all_evidence_refs
            if isinstance(ref, EvidenceRef) and _is_traceable_evidence_ref(ref)
        }
        progress_made = bool(set(satisfied) - previous_satisfied) or bool(
            accepted_refs - previous_accepted_refs
        )
        open_gaps = goal.open_gaps(satisfied)
        error_seen = any(observation.status == "error" for observation in new_observations)

        return {
            "structured_observations": new_observations,
            "answer_candidates": answer_candidates,
            "evidence_refs": evidence_refs,
            "computation_results": computation_results,
            "context_units": context_units,
            "locators": locators,
            "asset_refs": asset_refs,
            "satisfied_requirements": satisfied,
            "open_gaps": open_gaps,
            "insufficient_evidence_flag": bool(state.get("insufficient_evidence_flag", False)) or error_seen,
            "no_progress_count": 0 if progress_made else int(state.get("no_progress_count", 0)) + 1,
            "iteration": int(state.get("iteration", 0)) + 1,
            "evidence": _evidence_from_outputs(new_observations, tool_results),
            "citations": _citations_from_outputs(new_observations, tool_results),
        }


class SatisfactionChecker:
    def check(self, state: dict[str, Any]) -> SatisfactionReport:
        goal = _goal_from_state(state)
        pending = list(state.get("pending_tool_calls", []))
        bindings = list(state.get("context_bindings", []))
        conflicts = _merge_conflicts(
            [
                conflict
                for conflict in _conflicts_from_state(state)
                if conflict.metadata.get("generated_by") != "context_binding"
            ],
            _conflicts_from_bindings(bindings),
        )
        if pending:
            return SatisfactionReport(
                open_gaps=_gaps_from_state(state, goal=goal),
                conflicts=conflicts,
                reason="pending_tool_calls",
            )

        answers = list(state.get("answer_candidates", []))
        evidence_refs = list(state.get("evidence_refs", []))
        satisfied = _satisfied_requirements(
            goal,
            answers,
            evidence_refs,
            bindings,
            state.get("computation_results", []),
        )
        open_gaps = goal.open_gaps(satisfied)
        if not open_gaps and not conflicts:
            return SatisfactionReport(
                is_done=True,
                open_gaps=[],
                conflicts=[],
                reason="goal_satisfied",
            )

        tool_results = list(state.get("tool_results", []))
        if tool_results and all(result.status == "error" for result in tool_results):
            return SatisfactionReport(
                is_done=True,
                open_gaps=open_gaps,
                conflicts=conflicts,
                reason="tool_error",
            )

        if int(state.get("no_progress_count", 0)) >= 3:
            return SatisfactionReport(
                open_gaps=open_gaps,
                conflicts=conflicts,
                is_stuck=True,
                reason="no_progress",
            )

        return SatisfactionReport(
            open_gaps=open_gaps,
            conflicts=conflicts,
            reason="open_gaps",
        )


def _goal_from_state(state: dict[str, Any]) -> GoalSpec:
    goal = state.get("goal_spec")
    if isinstance(goal, GoalSpec):
        return goal
    return GoalInitializer().initialize(str(state.get("task", "")))


def _gaps_from_state(state: dict[str, Any], *, goal: GoalSpec) -> list[GoalGap]:
    values = state.get("open_gaps")
    if not isinstance(values, list):
        return goal.open_gaps()
    gaps: list[GoalGap] = []
    for value in values:
        if isinstance(value, GoalGap):
            gaps.append(value)
        elif isinstance(value, str):
            matching = goal.open_gaps()
            gaps.extend(gap for gap in matching if gap.gap_id == value)
    return gaps or goal.open_gaps()


def _conflicts_from_state(state: dict[str, Any]) -> list[GoalConflict]:
    values = state.get("conflicts", [])
    if not isinstance(values, list):
        return []
    conflicts: list[GoalConflict] = []
    for index, value in enumerate(values):
        if isinstance(value, GoalConflict):
            conflicts.append(value)
        elif isinstance(value, str):
            conflicts.append(
                GoalConflict(
                    conflict_id=f"conflict:{index}",
                    description=value,
                )
            )
    return conflicts


def _conflicts_from_bindings(bindings: Sequence[object]) -> list[GoalConflict]:
    return [
        GoalConflict(
            conflict_id=f"constraint:{binding.constraint_id}:{binding.unit_id or 'unknown'}",
            description=binding.rationale or "A selected context unit violates a required constraint.",
            related_unit_ids=[binding.unit_id] if binding.unit_id else [],
            metadata={
                "constraint_id": binding.constraint_id,
                "generated_by": "context_binding",
            },
        )
        for binding in bindings
        if isinstance(binding, ContextBinding) and binding.status == "violated"
    ]


def _merge_conflicts(
    existing: Sequence[GoalConflict],
    generated: Sequence[GoalConflict],
) -> list[GoalConflict]:
    return list(
        {
            conflict.conflict_id: conflict
            for conflict in [*existing, *generated]
        }.values()
    )


def _explicitly_requests_evidence(query: str) -> bool:
    lowered = query.lower()
    markers = ("出处", "来源", "引用", "证据", "citation", "source", "reference")
    return any(marker in lowered for marker in markers)


def _explicit_context_title_constraints(query: str) -> list[GoalConstraint]:
    match = re.search(
        r"(?:^|[，,。；;])在(?P<title>[^，,。；;?？\n]{1,100}?)(?:表|sheet)中",
        query,
        flags=re.IGNORECASE,
    )
    if match is None:
        return []
    title = match.group("title").strip(" \t\"'“”《》")
    if not title:
        return []
    return [
        GoalConstraint(
            constraint_id="context-title-1",
            constraint_type="context_title",
            expected_value=title,
            metadata={"origin": "explicit_query_scope"},
        )
    ]


def _success_criteria(*, required_evidence: Sequence[str]) -> list[str]:
    criteria = ["answer"]
    if required_evidence:
        criteria.append("traceable_evidence")
    return criteria


def _satisfied_requirements(
    goal: GoalSpec,
    answer_candidates: Sequence[AnswerCandidate],
    evidence_refs: Sequence[EvidenceRef],
    context_bindings: Sequence[object] = (),
    computation_results: Sequence[ComputationResult] = (),
) -> list[str]:
    satisfied: list[str] = []
    for constraint in goal.constraints:
        if constraint.required and any(
            isinstance(binding, ContextBinding)
            and binding.constraint_id == constraint.constraint_id
            and binding.status == "satisfied"
            for binding in context_bindings
        ):
            satisfied.append(f"constraint:{constraint.constraint_id}")
    for deliverable in goal.deliverables:
        if not deliverable.required:
            continue
        if _deliverable_is_satisfied(
            deliverable,
            answer_candidates=answer_candidates,
            evidence_refs=evidence_refs,
            computation_results=computation_results,
        ):
            satisfied.append(deliverable.deliverable_id)
    return satisfied


def _deliverable_is_satisfied(
    deliverable: GoalDeliverable,
    *,
    answer_candidates: Sequence[AnswerCandidate],
    evidence_refs: Sequence[EvidenceRef],
    computation_results: Sequence[ComputationResult],
) -> bool:
    if deliverable.acceptance_rule == "non_empty_answer":
        return any(candidate.text.strip() for candidate in answer_candidates)
    if deliverable.acceptance_rule == "traceable_evidence":
        return any(_is_traceable_evidence_ref(ref) for ref in evidence_refs)
    if deliverable.acceptance_rule == "reproducible_computation":
        return any(_is_reproducible_computation(result) for result in computation_results)
    return False


def _is_traceable_evidence_ref(ref: EvidenceRef) -> bool:
    if isinstance(ref.citation_id, str) and ref.citation_id.strip():
        return True
    if isinstance(ref.citation_anchor, str) and ref.citation_anchor.strip():
        return True
    return (
        ref.source == "asset"
        and isinstance(ref.evidence_id, str)
        and ref.evidence_id.startswith("asset:")
        and ref.evidence_id.removeprefix("asset:").isdigit()
    )


def _is_reproducible_computation(result: ComputationResult) -> bool:
    return bool(
        isinstance(result.expression, str)
        and result.expression.strip()
        and any(_is_traceable_evidence_ref(ref) for ref in result.evidence_refs)
    )


def _answer_deliverable() -> GoalDeliverable:
    return GoalDeliverable(
        deliverable_id="answer",
        kind="answer",
        acceptance_rule="non_empty_answer",
    )


def _evidence_deliverable() -> GoalDeliverable:
    return GoalDeliverable(
        deliverable_id="evidence",
        kind="evidence",
        acceptance_rule="traceable_evidence",
    )


def _computation_deliverable() -> GoalDeliverable:
    return GoalDeliverable(
        deliverable_id="computation",
        kind="computation",
        acceptance_rule="reproducible_computation",
    )


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
        "asset_list",
        "asset_inspect",
        "asset_read_slice",
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


def _delegated_evidence_refs_from_output(output: BaseModel | None) -> list[EvidenceRef]:
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
                citation_id=citation_id.strip() if isinstance(citation_id, str) else None,
                citation_anchor=(
                    citation_anchor.strip() if isinstance(citation_anchor, str) else None
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


def _search_evidence_refs_from_output(output: BaseModel | None) -> list[EvidenceRef]:
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
            evidence_id=str(item["evidence_id"]) if item.get("evidence_id") else None,
            citation_anchor=(
                str(item["citation_anchor"]) if item.get("citation_anchor") else None
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
    if result.tool_name in {"vector_search", "keyword_search", "grounding", "rerank"}:
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
        locator = {
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
            )
            if item.get(field) not in (None, "", [])
        }
        item_refs = _search_evidence_refs_from_output(
            _SingleSearchItemOutput(items=[item])
        )
        record_type = str(item.get("record_type", "") or "")
        unit_type = "document_section" if record_type == "section" else "retrieved_chunk"
        identifier = (
            str(item.get("evidence_id"))
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
                    str(item["evidence_id"]) if item.get("evidence_id") else result.tool_call_id
                ),
                evidence_refs=item_refs or list(evidence_refs),
                capabilities=capabilities,
                metadata={"source_tool": result.tool_name},
            )
        )
    return units


class _SingleSearchItemOutput(BaseModel):
    items: list[dict[str, object]]


def _asset_context_unit(tool_name: str, locator: dict[str, object]) -> ContextUnit:
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


def _dedupe_context_units(units: Sequence[ContextUnit]) -> list[ContextUnit]:
    merged: dict[str, ContextUnit] = {}
    for unit in units:
        merged[unit.unit_id] = unit
    return list(merged.values())


def _locators_from_output(output: BaseModel | None) -> list[dict[str, object]]:
    if output is None:
        return []
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
    locators: list[dict[str, object]] = []
    for asset in assets:
        if not hasattr(asset, "model_dump"):
            continue
        locators.append(_asset_locator_from_descriptor(asset))
    return locators


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
        refs: list[int] = []
        for asset in assets:
            asset_id = getattr(asset, "asset_id", None)
            if isinstance(asset_id, int) and asset_id > 0:
                refs.append(asset_id)
        return refs
    asset_id = getattr(output, "asset_id", None)
    if isinstance(asset_id, int) and asset_id > 0:
        return [asset_id]
    return []


def _search_asset_locators(items: Sequence[object]) -> list[dict[str, object]]:
    locators: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "")
        record_type = str(item.get("record_type", "") or "")
        if "ASSET_ANCHOR:" not in text and "asset" not in record_type and "section" not in record_type:
            continue
        locator: dict[str, object] = {}
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
        ):
            value = item.get(field)
            if value not in (None, "", []):
                locator[field] = value
        if locator.get("section_id") is not None:
            locators.append(locator)
    return locators


class ContextBindingProvider(Protocol):
    def assess_bindings(
        self,
        *,
        state: dict[str, Any],
        constraints: Sequence[GoalConstraint],
        context_units: Sequence[ContextUnit],
    ) -> list[ContextBinding]: ...


class ContextBindingAssessor:
    def __init__(self, providers: Sequence[ContextBindingProvider] | None = None) -> None:
        if providers is None:
            from rag.agent.binding_providers import AssetContextBindingProvider

            providers = [AssetContextBindingProvider()]
        self._providers: list[ContextBindingProvider] = list(providers)

    def assess_bindings(
        self,
        state: dict[str, Any],
        *,
        context_units: Sequence[ContextUnit] | None = None,
    ) -> list[ContextBinding]:
        goal = _goal_from_state(state)
        effective_units = list(context_units or state.get("context_units", []))
        assessed = [
            binding
            for provider in self._providers
            for binding in provider.assess_bindings(
                state=state,
                constraints=goal.constraints,
                context_units=effective_units,
            )
        ]
        return list({binding.key: binding for binding in assessed}.values())


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


def _dedupe_evidence_refs(refs: Sequence[EvidenceRef]) -> list[EvidenceRef]:
    deduped: dict[str, EvidenceRef] = {}
    for ref in refs:
        key = ref.key
        if not key:
            continue
        deduped.setdefault(key, ref)
    return list(deduped.values())


def _evidence_from_outputs(
    observations: Sequence[StructuredObservation],
    tool_results: Sequence[ToolResult],
) -> list[EvidenceItem]:
    observed_ids = {observation.tool_call_id for observation in observations}
    evidence: list[EvidenceItem] = []
    for result in tool_results:
        if result.tool_call_id not in observed_ids or result.output is None:
            continue
        for item in getattr(result.output, "evidence", []) or []:
            evidence.append(EvidenceItem.model_validate(item))
    return evidence


def _citations_from_outputs(
    observations: Sequence[StructuredObservation],
    tool_results: Sequence[ToolResult],
) -> list[AnswerCitation]:
    observed_ids = {observation.tool_call_id for observation in observations}
    citations: list[AnswerCitation] = []
    for result in tool_results:
        if result.tool_call_id not in observed_ids or result.output is None:
            continue
        for item in getattr(result.output, "citations", []) or []:
            citations.append(AnswerCitation.model_validate(item))
    return citations


__all__ = [
    "AnswerCandidate",
    "ComputationResult",
    "ContextBinding",
    "ContextBindingAssessor",
    "ContextBindingProvider",
    "ContextUnit",
    "EvidenceRef",
    "GoalConflict",
    "GoalConstraint",
    "GoalDeliverable",
    "GoalGap",
    "GoalInitializer",
    "GoalSpec",
    "ObservationBuilder",
    "RuntimeState",
    "SatisfactionChecker",
    "SatisfactionReport",
    "StateReducer",
    "StructuredObservation",
]
