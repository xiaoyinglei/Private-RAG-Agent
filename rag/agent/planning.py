from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

PlanStepStatus = Literal["pending", "in_progress", "completed", "blocked", "skipped"]
PlanStatus = Literal["active", "complete", "blocked", "needs_replan"]
PlanUpdateMode = Literal["replace", "patch"]
PlanEventType = Literal[
    "initialized",
    "llm_update",
    "decision_progress",
    "observation_progress",
    "needs_replan",
    "completed",
    "blocked",
]

MAX_PLAN_STEPS = 12
MAX_PLAN_EVENTS = 30
MAX_STEP_TITLE_CHARS = 180
MAX_STEP_NOTES_CHARS = 240
MAX_PLAN_SUMMARY_CHARS = 800
MAX_OBJECTIVE_CHARS = 500
MAX_STEP_REFS = 16


class PlanStep(BaseModel):
    """A bounded todo item for the current agent run.

    Plan steps are planning state only. They do not authorize tools or bypass
    executor policy.
    """

    step_id: str
    title: str
    status: PlanStepStatus = "pending"
    related_gap_ids: list[str] = Field(default_factory=list)
    expected_tool_names: list[str] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    notes: str | None = None

    @property
    def key(self) -> str:
        return self.step_id

    @model_validator(mode="after")
    def bound_fields(self) -> Self:
        self.step_id = _safe_identifier(self.step_id, prefix="step")
        self.title = _bounded_text(self.title, MAX_STEP_TITLE_CHARS) or "Untitled step"
        self.related_gap_ids = _dedupe_texts(self.related_gap_ids, limit=MAX_STEP_REFS)
        self.expected_tool_names = _dedupe_texts(self.expected_tool_names, limit=MAX_STEP_REFS)
        self.tool_call_ids = _dedupe_texts(self.tool_call_ids, limit=MAX_STEP_REFS)
        self.evidence_refs = _dedupe_texts(self.evidence_refs, limit=MAX_STEP_REFS)
        if self.notes is not None:
            self.notes = _bounded_text(self.notes, MAX_STEP_NOTES_CHARS) or None
        return self


class PlanStepPatch(BaseModel):
    step_id: str
    title: str | None = None
    status: PlanStepStatus | None = None
    related_gap_ids: list[str] | None = None
    expected_tool_names: list[str] | None = None
    tool_call_ids: list[str] | None = None
    evidence_refs: list[str] | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def bound_fields(self) -> Self:
        self.step_id = _safe_identifier(self.step_id, prefix="step")
        if self.title is not None:
            self.title = _bounded_text(self.title, MAX_STEP_TITLE_CHARS) or None
        if self.related_gap_ids is not None:
            self.related_gap_ids = _dedupe_texts(self.related_gap_ids, limit=MAX_STEP_REFS)
        if self.expected_tool_names is not None:
            self.expected_tool_names = _dedupe_texts(self.expected_tool_names, limit=MAX_STEP_REFS)
        if self.tool_call_ids is not None:
            self.tool_call_ids = _dedupe_texts(self.tool_call_ids, limit=MAX_STEP_REFS)
        if self.evidence_refs is not None:
            self.evidence_refs = _dedupe_texts(self.evidence_refs, limit=MAX_STEP_REFS)
        if self.notes is not None:
            self.notes = _bounded_text(self.notes, MAX_STEP_NOTES_CHARS) or None
        return self


class AgentPlan(BaseModel):
    objective: str
    status: PlanStatus = "active"
    revision: int = Field(default=0, ge=0)
    active_step_id: str | None = None
    steps: list[PlanStep] = Field(default_factory=list)
    summary: str | None = None

    @model_validator(mode="after")
    def bound_fields(self) -> Self:
        self.objective = _bounded_text(self.objective, MAX_OBJECTIVE_CHARS) or "Current task"
        self.steps = self.steps[:MAX_PLAN_STEPS]
        if self.summary is not None:
            self.summary = _bounded_text(self.summary, MAX_PLAN_SUMMARY_CHARS) or None
        self.active_step_id = _valid_active_step_id(self.active_step_id, self.steps)
        return self


class PlanUpdate(BaseModel):
    """Optional plan delta returned by the LLM decision provider."""

    mode: PlanUpdateMode = "patch"
    objective: str | None = None
    status: PlanStatus | None = None
    steps: list[PlanStep] = Field(default_factory=list)
    step_updates: list[PlanStepPatch] = Field(default_factory=list)
    active_step_id: str | None = None
    summary: str | None = None

    @model_validator(mode="after")
    def bound_fields(self) -> Self:
        if self.objective is not None:
            self.objective = _bounded_text(self.objective, MAX_OBJECTIVE_CHARS) or None
        self.steps = self.steps[:MAX_PLAN_STEPS]
        self.step_updates = self.step_updates[:MAX_PLAN_STEPS]
        if self.active_step_id is not None:
            self.active_step_id = _safe_identifier(self.active_step_id, prefix="step")
        if self.summary is not None:
            self.summary = _bounded_text(self.summary, MAX_PLAN_SUMMARY_CHARS) or None
        return self


class PlanEvent(BaseModel):
    event_id: str
    event_type: PlanEventType
    plan_revision: int = Field(ge=0)
    message: str
    related_step_id: str | None = None
    tool_call_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return self.event_id

    @model_validator(mode="after")
    def bound_fields(self) -> Self:
        self.event_id = _safe_identifier(self.event_id, prefix="plan_event")
        self.message = _bounded_text(self.message, MAX_PLAN_SUMMARY_CHARS) or self.event_type
        if self.related_step_id is not None:
            self.related_step_id = _safe_identifier(self.related_step_id, prefix="step")
        self.tool_call_ids = _dedupe_texts(self.tool_call_ids, limit=MAX_STEP_REFS)
        self.warnings = _dedupe_texts(self.warnings, limit=MAX_STEP_REFS)
        return self


@dataclass(slots=True)
class PlanTracker:
    max_steps: int = MAX_PLAN_STEPS

    def initialize(self, *, task: str, open_gaps: Sequence[object]) -> tuple[AgentPlan, list[PlanEvent]]:
        steps = [
            PlanStep(
                step_id=_step_id_for_gap(gap, index=index),
                title=_gap_description(gap),
                related_gap_ids=[_gap_id(gap)] if _gap_id(gap) else [],
            )
            for index, gap in enumerate(open_gaps[: self.max_steps], start=1)
        ]
        if not steps:
            steps = [
                PlanStep(
                    step_id="step_answer",
                    title="Decide whether the goal is already satisfied.",
                )
            ]
        plan = AgentPlan(
            objective=task,
            status="active",
            active_step_id=steps[0].step_id,
            steps=steps,
        )
        return plan, [self._event("initialized", plan, message="Initialized autonomous plan.")]

    def apply_llm_update(
        self,
        plan: AgentPlan,
        update: PlanUpdate,
        *,
        allowed_tool_names: frozenset[str],
        open_gap_ids: frozenset[str],
    ) -> tuple[AgentPlan, list[PlanEvent]]:
        warnings: list[str] = []
        if update.mode == "replace" and update.steps:
            if len(update.steps) > self.max_steps:
                warnings.append("steps_truncated")
            steps = [
                self._normalize_step(
                    step,
                    allowed_tool_names=allowed_tool_names,
                    open_gap_ids=open_gap_ids,
                    warnings=warnings,
                )
                for step in update.steps[: self.max_steps]
            ]
        else:
            steps = [
                self._normalize_step(
                    step,
                    allowed_tool_names=allowed_tool_names,
                    open_gap_ids=open_gap_ids,
                    warnings=warnings,
                )
                for step in plan.steps
            ]
            steps = self._apply_patches(
                steps,
                update.step_updates,
                allowed_tool_names=allowed_tool_names,
                open_gap_ids=open_gap_ids,
                warnings=warnings,
            )

        active_step_id = update.active_step_id or plan.active_step_id
        updated = AgentPlan(
            objective=update.objective or plan.objective,
            status=update.status or plan.status,
            revision=plan.revision + 1,
            active_step_id=active_step_id,
            steps=steps,
            summary=update.summary if update.summary is not None else plan.summary,
        )
        if active_step_id is not None and updated.active_step_id != active_step_id:
            warnings.append("active_step_not_found")
        return updated, [
            self._event(
                "llm_update",
                updated,
                message="Applied LLM plan update.",
                related_step_id=updated.active_step_id,
                warnings=warnings,
            )
        ]

    def record_decision_progress(
        self,
        plan: AgentPlan,
        *,
        tool_call_ids: Sequence[str],
        tool_names: Sequence[str],
    ) -> tuple[AgentPlan, list[PlanEvent]]:
        if not tool_call_ids:
            return plan, []
        step = _active_or_next_step(plan)
        if step is None:
            return plan, []
        steps: list[PlanStep] = []
        for current in plan.steps:
            if current.step_id != step.step_id:
                steps.append(current)
                continue
            steps.append(
                current.model_copy(
                    update={
                        "status": "in_progress" if current.status == "pending" else current.status,
                        "tool_call_ids": _dedupe_texts(
                            [*current.tool_call_ids, *tool_call_ids],
                            limit=MAX_STEP_REFS,
                        ),
                        "expected_tool_names": _dedupe_texts(
                            [*current.expected_tool_names, *tool_names],
                            limit=MAX_STEP_REFS,
                        ),
                    }
                )
            )
        updated = plan.model_copy(
            update={
                "revision": plan.revision + 1,
                "active_step_id": step.step_id,
                "steps": steps,
            }
        )
        return updated, [
            self._event(
                "decision_progress",
                updated,
                message="Recorded tool decision against active plan step.",
                related_step_id=step.step_id,
                tool_call_ids=list(tool_call_ids),
            )
        ]

    def record_observation_progress(
        self,
        plan: AgentPlan | None,
        *,
        observations: Sequence[object],
        satisfied_requirement_ids: Sequence[str],
    ) -> tuple[AgentPlan | None, list[PlanEvent]]:
        if plan is None or not observations:
            return plan, []

        changed = False
        completed_step_id: str | None = None
        steps: list[PlanStep] = []
        for step in plan.steps:
            if step.status not in {"pending", "in_progress"}:
                steps.append(step)
                continue
            if not _step_observation_bound(step, observations):
                steps.append(step)
                continue
            changed = True
            completed_step_id = step.step_id
            steps.append(step.model_copy(update={"status": "completed"}))

        if not changed and any(getattr(observation, "status", None) != "error" for observation in observations):
            updated = plan.model_copy(
                update={
                    "status": "needs_replan",
                    "revision": plan.revision + 1,
                }
            )
            return updated, [
                self._event(
                    "needs_replan",
                    updated,
                    message="Structured observations were not bound to any active plan step.",
                    related_step_id=updated.active_step_id,
                    warnings=["observation_unbound"],
                )
            ]
        if not changed:
            return plan, []
        updated = plan.model_copy(
            update={
                "status": "active",
                "revision": plan.revision + 1,
                "steps": steps,
                "active_step_id": _valid_active_step_id(plan.active_step_id, steps),
            }
        )
        return updated, [
            self._event(
                "observation_progress",
                updated,
                message="Marked plan step complete from structured observations.",
                related_step_id=completed_step_id,
            )
        ]

    def record_completion(
        self,
        plan: AgentPlan | None,
        *,
        blocked: bool = False,
    ) -> tuple[AgentPlan | None, list[PlanEvent]]:
        if plan is None:
            return None, []
        target_status: PlanStatus = "blocked" if blocked else "complete"
        if plan.status == target_status:
            return plan, []
        step_status: PlanStepStatus = "blocked" if blocked else "completed"
        steps = [
            step.model_copy(update={"status": step_status})
            if step.status in {"pending", "in_progress"}
            else step
            for step in plan.steps
        ]
        updated = plan.model_copy(
            update={
                "status": target_status,
                "revision": plan.revision + 1,
                "steps": steps,
            }
        )
        event_type: PlanEventType = "blocked" if blocked else "completed"
        return updated, [
            self._event(
                event_type,
                updated,
                message="Plan blocked." if blocked else "Plan completed.",
                related_step_id=updated.active_step_id,
            )
        ]

    def _apply_patches(
        self,
        steps: list[PlanStep],
        patches: Sequence[PlanStepPatch],
        *,
        allowed_tool_names: frozenset[str],
        open_gap_ids: frozenset[str],
        warnings: list[str],
    ) -> list[PlanStep]:
        by_id = {step.step_id: step for step in steps}
        ordered_ids = [step.step_id for step in steps]
        for patch in patches:
            existing = by_id.get(patch.step_id)
            if existing is None:
                if patch.title is None or len(ordered_ids) >= self.max_steps:
                    if len(ordered_ids) >= self.max_steps:
                        warnings.append("steps_truncated")
                    continue
                existing = PlanStep(step_id=patch.step_id, title=patch.title)
                by_id[patch.step_id] = existing
                ordered_ids.append(patch.step_id)
            updated = _patch_step(existing, patch)
            by_id[patch.step_id] = self._normalize_step(
                updated,
                allowed_tool_names=allowed_tool_names,
                open_gap_ids=open_gap_ids,
                warnings=warnings,
            )
        return [by_id[step_id] for step_id in ordered_ids[: self.max_steps]]

    @staticmethod
    def _normalize_step(
        step: PlanStep,
        *,
        allowed_tool_names: frozenset[str],
        open_gap_ids: frozenset[str],
        warnings: list[str],
    ) -> PlanStep:
        expected_tools = list(step.expected_tool_names)
        if allowed_tool_names:
            filtered_tools = [name for name in expected_tools if name in allowed_tool_names]
            if len(filtered_tools) != len(expected_tools):
                warnings.append("unsupported_tool_names")
            expected_tools = filtered_tools

        related_gap_ids = list(step.related_gap_ids)
        if open_gap_ids:
            filtered_gap_ids = [gap_id for gap_id in related_gap_ids if gap_id in open_gap_ids]
            if len(filtered_gap_ids) != len(related_gap_ids):
                warnings.append("unknown_gap_ids")
            related_gap_ids = filtered_gap_ids

        return step.model_copy(
            update={
                "related_gap_ids": related_gap_ids,
                "expected_tool_names": expected_tools,
            }
        )

    @staticmethod
    def _event(
        event_type: PlanEventType,
        plan: AgentPlan,
        *,
        message: str,
        related_step_id: str | None = None,
        tool_call_ids: Sequence[str] = (),
        warnings: Sequence[str] = (),
    ) -> PlanEvent:
        return PlanEvent(
            event_id=f"plan_event_{uuid4().hex[:12]}",
            event_type=event_type,
            plan_revision=plan.revision,
            message=message,
            related_step_id=related_step_id,
            tool_call_ids=list(tool_call_ids),
            warnings=list(warnings),
        )


def _patch_step(step: PlanStep, patch: PlanStepPatch) -> PlanStep:
    update: dict[str, object] = {}
    if patch.title is not None:
        update["title"] = patch.title
    if patch.status is not None:
        update["status"] = patch.status
    if patch.related_gap_ids is not None:
        update["related_gap_ids"] = patch.related_gap_ids
    if patch.expected_tool_names is not None:
        update["expected_tool_names"] = patch.expected_tool_names
    if patch.tool_call_ids is not None:
        update["tool_call_ids"] = patch.tool_call_ids
    if patch.evidence_refs is not None:
        update["evidence_refs"] = patch.evidence_refs
    if patch.notes is not None:
        update["notes"] = patch.notes
    return step.model_copy(update=update)


def _active_or_next_step(plan: AgentPlan) -> PlanStep | None:
    if plan.active_step_id is not None:
        for step in plan.steps:
            if step.step_id == plan.active_step_id and step.status in {"pending", "in_progress"}:
                return step
    for step in plan.steps:
        if step.status in {"pending", "in_progress"}:
            return step
    return None


def _step_observation_bound(step: PlanStep, observations: Sequence[object]) -> bool:
    for observation in observations:
        if getattr(observation, "status", None) == "error":
            continue
        tool_call_id = getattr(observation, "tool_call_id", None)
        if isinstance(tool_call_id, str) and tool_call_id in step.tool_call_ids:
            return True

        resolved_gaps = {
            str(gap)
            for gap in (
                [
                    *list(getattr(observation, "resolved_gaps", []) or []),
                    *list(getattr(observation, "related_gap_ids", []) or []),
                ]
            )
            if str(gap)
        }
        if step.related_gap_ids and resolved_gaps.intersection(step.related_gap_ids):
            return True

        related_step_ids = {
            str(step_id)
            for step_id in getattr(observation, "related_step_ids", []) or []
            if str(step_id)
        }
        metadata = getattr(observation, "metadata", {}) or {}
        if isinstance(metadata, dict):
            related_step_id = metadata.get("related_step_id")
            if related_step_id:
                related_step_ids.add(str(related_step_id))
            raw_related_step_ids = metadata.get("related_step_ids")
            if isinstance(raw_related_step_ids, list):
                related_step_ids.update(str(item) for item in raw_related_step_ids if str(item))
        if step.step_id in related_step_ids:
            return True
    return False


def _valid_active_step_id(active_step_id: str | None, steps: Sequence[PlanStep]) -> str | None:
    step_ids = {step.step_id for step in steps}
    if active_step_id is not None:
        safe = _safe_identifier(active_step_id, prefix="step")
        if safe in step_ids:
            return safe
    for status in ("in_progress", "pending"):
        for step in steps:
            if step.status == status:
                return step.step_id
    return steps[0].step_id if steps else None


def _step_id_for_gap(gap: object, *, index: int) -> str:
    gap_id = _gap_id(gap)
    if gap_id:
        return _safe_identifier(f"step_{gap_id}", prefix="step")
    return f"step_{index:03d}"


def _gap_id(gap: object) -> str:
    if isinstance(gap, str):
        return gap
    value = getattr(gap, "gap_id", None)
    return str(value) if value else ""


def _gap_description(gap: object) -> str:
    if isinstance(gap, str):
        return f"Satisfy {gap}."
    value = getattr(gap, "description", None)
    return str(value) if value else "Satisfy an open goal gap."


def _safe_identifier(value: str, *, prefix: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        return f"{prefix}_unknown"
    if not cleaned.startswith(f"{prefix}_"):
        cleaned = f"{prefix}_{cleaned}"
    return cleaned[:80]


def _bounded_text(value: str, limit: int) -> str:
    normalized = " ".join(str(value).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip()


def _dedupe_texts(values: Sequence[str], *, limit: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _bounded_text(str(value), MAX_STEP_NOTES_CHARS)
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
        if len(deduped) >= limit:
            break
    return deduped


__all__ = [
    "AgentPlan",
    "MAX_PLAN_EVENTS",
    "MAX_PLAN_STEPS",
    "PlanTracker",
    "PlanEvent",
    "PlanStep",
    "PlanStepPatch",
    "PlanStatus",
    "PlanStepStatus",
    "PlanUpdate",
]
