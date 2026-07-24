from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolValidationError,
    json_schema_output,
    pydantic_input,
)

type PlanUpdater = Callable[
    [Mapping[str, JsonValue]],
    object | Awaitable[object],
]
_DISCOVERY_TOOL_NAMES = frozenset({"list_files", "search_text", "read_file"})


class PlanStepInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        description=(
            "Stable step identifier from the visible plan. Omit it for a new "
            "step; the runtime will assign one."
        ),
    )
    step: str = Field(
        min_length=1,
        max_length=180,
        description="A concise, verifiable implementation step.",
    )
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="Current state of this exact step."
    )
    goal_commitment_ids: list[str] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "Runtime-owned requirement identifiers from "
            "working_state.goal_contract.commitments that this step serves. "
            "Every step must bind at least one identifier, and the complete plan "
            "must cover every listed commitment."
        ),
    )
    expected_tool_names: list[str] = Field(
        min_length=1,
        max_length=4,
        description=(
            "Exact tool names whose successful result can complete this step. "
            "Use inspection tools only for a concrete unresolved question, "
            "apply_patch for edits, and run_command for verification."
        ),
    )

    @field_validator("expected_tool_names")
    @classmethod
    def validate_expected_tool_names(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value.strip() for value in values))
        if any(not value for value in normalized):
            raise ValueError("expected tool names must be non-empty")
        return normalized

    @field_validator("goal_commitment_ids")
    @classmethod
    def validate_goal_commitment_ids(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value.strip() for value in values))
        if any(not value or len(value) > 80 for value in normalized):
            raise ValueError("goal commitment identifiers must be concise")
        return normalized


class UpdatePlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_id: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "Opaque immutable goal identifier from "
            "working_state.goal_contract.goal_id. Copy it exactly; a plan cannot "
            "replace the goal it is meant to serve."
        ),
    )
    objective: str | None = Field(
        default=None,
        min_length=1,
        max_length=8_000,
        description=(
            "The immutable objective from working_state.goal_contract.objective. "
            "Copy it exactly apart from whitespace; change only the strategy fields."
        ),
    )
    target_files: list[str] = Field(
        max_length=12,
        description=(
            "Workspace-relative files currently implicated by observed "
            "evidence. Name concrete paths, not directories or guesses. Use "
            "an empty list only for a bounded discovery checkpoint with at "
            "least one remaining unknown and an active discovery step."
        ),
    )
    hypothesis: str = Field(
        min_length=12,
        max_length=800,
        description=(
            "Current causal hypothesis: what is wrong and what change is "
            "expected to fix it."
        ),
    )
    remaining_unknowns: list[str] = Field(
        max_length=8,
        description=(
            "Concrete unanswered questions. Use an empty list when the "
            "evidence is sufficient to edit or verify."
        ),
    )
    plan: list[PlanStepInput] = Field(
        min_length=1,
        max_length=20,
        description=(
            'The complete ordered plan; every item requires "step", "status", '
            'and "expected_tool_names" fields.'
        ),
    )
    explanation: str | None = Field(
        default=None,
        max_length=800,
        description="A short reason for the plan transition.",
    )

    @field_validator("target_files")
    @classmethod
    def validate_target_files(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw_value in values:
            value = raw_value.strip().replace("\\", "/")
            while value.startswith("./"):
                value = value[2:]
            parts = value.split("/")
            if (
                not value
                or value == "."
                or value.startswith("/")
                or (len(value) >= 2 and value[1] == ":")
                or ".." in parts
                or len(value) > 4096
            ):
                raise ValueError(
                    "target_files must contain workspace-relative file paths"
                )
            if value not in normalized:
                normalized.append(value)
        return normalized

    @field_validator("hypothesis")
    @classmethod
    def validate_hypothesis(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) < 12:
            raise ValueError("hypothesis must describe a concrete causal theory")
        return normalized

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("objective must preserve the immutable user goal")
        return normalized

    @field_validator("remaining_unknowns")
    @classmethod
    def validate_remaining_unknowns(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized = list(
            dict.fromkeys(" ".join(value.split()) for value in values)
        )
        if any(not value or len(value) > 300 for value in normalized):
            raise ValueError(
                "remaining_unknowns must contain concise non-empty questions"
            )
        return normalized

class UpdatePlanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    revision: int = Field(ge=0)
    message: str = Field(min_length=1, max_length=800)
    authority: Literal["advisory"] = "advisory"
    grounded_target_files: list[str] = Field(default_factory=list)
    unverified_target_files: list[str] = Field(default_factory=list)


_UPDATE_INPUT_SCHEMA, _validate_update_input = pydantic_input(UpdatePlanInput)
_UPDATE_OUTPUT_SCHEMA, _unused_update_output_validator = pydantic_input(
    UpdatePlanOutput
)


def _validate_update_plan_input(
    arguments: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    canonical = _validate_update_input(arguments)
    if canonical["target_files"]:
        return canonical
    if not canonical["remaining_unknowns"]:
        raise ToolValidationError(
            path="$.target_files",
            message=(
                "may be empty only while a concrete remaining_unknown "
                "is recorded"
            ),
        )
    plan = canonical["plan"]
    if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)):
        raise ToolValidationError(
            path="$.plan",
            message="must contain a bounded discovery step",
        )
    has_active_discovery = any(
        isinstance(item, Mapping)
        and item.get("status") in {"pending", "in_progress"}
        and isinstance(
            expected := item.get("expected_tool_names"),
            Sequence,
        )
        and not isinstance(expected, (str, bytes))
        and bool(set(expected) & _DISCOVERY_TOOL_NAMES)
        for item in plan
    )
    if not has_active_discovery:
        raise ToolValidationError(
            path="$.plan",
            message=(
                "an empty target_files checkpoint requires an active "
                "list_files, search_text, or read_file step"
            ),
        )
    return canonical


def create_update_plan_tool(plan_updater: PlanUpdater) -> Tool:
    if not callable(plan_updater):
        raise TypeError("plan_updater must be callable")
    return Tool(
        definition=ToolDefinition(
            name="update_plan",
            description=(
                "Update the strategy for the immutable runtime-owned goal. Copy "
                '"goal_id" and "objective" exactly from '
                "working_state.goal_contract; changing or omitting either is "
                "rejected as goal_drift. Bind every plan step to one or more "
                '"goal_commitment_ids" from '
                "working_state.goal_contract.commitments, and cover every "
                "commitment in the complete plan. Replace the visible plan "
                "with an ordered set of "
                "pending, in-progress, and completed steps. Use it when the work "
                "crosses meaningful checkpoints. Every plan item must be shaped like "
                '{"step": "verify tests", "status": "in_progress", '
                '"goal_commitment_ids": ["goal_002"], '
                '"expected_tool_names": ["run_command"]}. Also submit '
                '"target_files", a causal "hypothesis", and '
                '"remaining_unknowns". Inspection steps must name the concrete '
                "unresolved evidence in the step text. The tool reports state "
                "but grants no tool permission."
            ),
            input_schema=_UPDATE_INPUT_SCHEMA,
        ),
        validate_input=_validate_update_plan_input,
        run=plan_updater,
        normalize_output=_normalize_update_output,
        output_schema=_UPDATE_OUTPUT_SCHEMA,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(),
        ),
        execution_revision="builtin-update-plan-v2-goal-contract",
        idempotent=True,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        timeout_seconds=5.0,
        max_model_output_bytes=50_000,
    )


def _normalize_update_output(raw: object) -> NormalizedToolOutput:
    validated = UpdatePlanOutput.model_validate(raw)
    structured = json_schema_output(
        _UPDATE_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    return NormalizedToolOutput(structured_content=structured)


__all__ = [
    "PlanStepInput",
    "PlanUpdater",
    "UpdatePlanInput",
    "UpdatePlanOutput",
    "create_update_plan_tool",
]
