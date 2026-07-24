from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.goal_contract import GoalContractEvaluator, GoalSpec
from rag.agent.core.observations import (
    ComputationResult,
    ContextBinding,
    EvidenceRef,
    runtime_workspace_change,
)
from rag.agent.core.output_finalizer import (
    OutputValidationExhaustedError,
    StructuredOutputFinalizer,
    validated_final_output,
)
from rag.agent.core.output_models import ValidatedFinalOutput
from rag.agent.loop.state import (
    LoopState,
    StopHookFeedback,
    append_stop_hook_feedback,
    append_stop_hook_warning,
)
from rag.agent.tools.tool import ToolResult


class StopVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["accept", "warn", "block", "halt"]
    code: str = Field(min_length=1, max_length=120)
    message: str | None = Field(default=None, max_length=1000)
    detail: dict[str, object] = Field(default_factory=dict)
    final_output: ValidatedFinalOutput | None = None


class StopHookOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["accept", "warn", "block", "halt"]
    code: str
    message: str | None = None
    detail: dict[str, object] = Field(default_factory=dict)
    verdicts: tuple[StopVerdict, ...] = ()
    final_output: ValidatedFinalOutput | None = None

    @property
    def accepted(self) -> bool:
        return self.action in {"accept", "warn"}

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    @property
    def halted(self) -> bool:
        return self.action == "halt"


class StopHook(Protocol):
    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopVerdict: ...


@dataclass(frozen=True)
class StopHookBinding:
    name: str
    hook: StopHook
    critical: bool


class StopHookRunner:
    def __init__(
        self,
        *,
        hooks: list[StopHookBinding] | tuple[StopHookBinding, ...],
        max_blocks: int,
    ) -> None:
        if max_blocks < 1:
            raise ValueError("max_blocks must be positive")
        self._hooks = tuple(hooks)
        self._max_blocks = max_blocks

    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopHookOutcome:
        verdicts: list[StopVerdict] = []
        final_output: ValidatedFinalOutput | None = None
        warned = False
        for binding in self._hooks:
            try:
                verdict = await binding.hook.evaluate(
                    state=state,
                    candidate=candidate,
                )
            except Exception as exc:
                verdict = StopVerdict(
                    action="halt" if binding.critical else "warn",
                    code=f"{binding.name}_failed",
                    message=str(exc) or type(exc).__name__,
                    detail={"error_type": type(exc).__name__},
                )
            verdicts.append(verdict)
            if verdict.final_output is not None:
                final_output = verdict.final_output

            if verdict.action == "warn":
                warned = True
                append_stop_hook_warning(
                    state,
                    StopHookFeedback(
                        code=verdict.code,
                        message=verdict.message or verdict.code,
                    ),
                )
                continue
            if verdict.action == "block":
                feedback = append_stop_hook_feedback(
                    state,
                    StopHookFeedback(
                        code=verdict.code,
                        message=verdict.message or verdict.code,
                    ),
                )
                if feedback.occurrences >= self._max_blocks:
                    return StopHookOutcome(
                        action="halt",
                        code="stop_hook_block_limit",
                        message=("Equivalent stop-hook feedback reached the configured block limit."),
                        detail={
                            "blocked_code": verdict.code,
                            "occurrences": feedback.occurrences,
                        },
                        verdicts=tuple(verdicts),
                        final_output=final_output,
                    )
                return StopHookOutcome(
                    action="block",
                    code=verdict.code,
                    message=verdict.message,
                    detail=verdict.detail,
                    verdicts=tuple(verdicts),
                    final_output=final_output,
                )
            if verdict.action == "halt":
                return StopHookOutcome(
                    action="halt",
                    code=verdict.code,
                    message=verdict.message,
                    detail=verdict.detail,
                    verdicts=tuple(verdicts),
                    final_output=final_output,
                )

        return StopHookOutcome(
            action="warn" if warned else "accept",
            code="accepted_with_warnings" if warned else "accepted",
            verdicts=tuple(verdicts),
            final_output=final_output,
        )


class StructuredOutputStopHook:
    def __init__(
        self,
        *,
        definition: AgentRuntimePolicy,
        finalizer: StructuredOutputFinalizer | None,
    ) -> None:
        self._definition = definition
        self._finalizer = finalizer

    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopVerdict:
        if self._finalizer is None:
            raise RuntimeError("structured output is configured without a finalizer")
        try:
            output = await _await_output(
                self._finalizer.finalize(
                    definition=self._definition,
                    state=state,
                    candidate_text=candidate,
                )
            )
        except OutputValidationExhaustedError as exc:
            return StopVerdict(
                action="halt",
                code="structured_output_invalid",
                message=str(exc),
                detail={
                    "attempts": exc.attempts,
                    "validation_errors": exc.validation_errors,
                },
            )
        return StopVerdict(
            action="accept",
            code="structured_output_valid",
            final_output=validated_final_output(output),
        )


class GoalContractStopHook:
    def __init__(
        self,
        *,
        goal_spec: GoalSpec,
        workspace_root: Path | None = None,
    ) -> None:
        self._goal_spec = goal_spec
        self._workspace_root = (
            None if workspace_root is None else workspace_root.resolve()
        )

    @staticmethod
    def _collect_evidence_refs(
        tool_results: list[ToolResult],
    ) -> list[EvidenceRef]:
        """Derive evidence_refs from tool_results instead of deprecated state field."""
        refs: list[EvidenceRef] = []
        for result in tool_results:
            values = _structured_items(result, "evidence_refs")
            refs.extend(EvidenceRef.model_validate(item) for item in values)
        return refs

    @staticmethod
    def _collect_computation_results(
        tool_results: list[ToolResult],
    ) -> list[ComputationResult]:
        """Derive computation_results from tool_results instead of deprecated state field."""
        results: list[ComputationResult] = []
        for result in tool_results:
            values = _structured_items(result, "computation_results")
            results.extend(
                ComputationResult.model_validate(item) for item in values
            )
        return results

    @staticmethod
    def _collect_context_bindings(
        tool_results: list[ToolResult],
    ) -> list[ContextBinding]:
        """Derive context_bindings from tool_results instead of deprecated state field."""
        bindings: list[ContextBinding] = []
        for result in tool_results:
            values = _structured_items(result, "context_bindings")
            bindings.extend(ContextBinding.model_validate(item) for item in values)
        return bindings

    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopVerdict:
        tool_results = list(state.get("tool_results", []))
        runtime_owned_constraint_ids = {
            constraint.constraint_id
            for constraint in self._goal_spec.constraints
            if constraint.constraint_type
            in {"workspace_change", "verification_after_change"}
        }
        context_bindings = [
            binding
            for binding in self._collect_context_bindings(tool_results)
            if binding.constraint_id not in runtime_owned_constraint_ids
        ]
        workspace_change_constraints = tuple(
            constraint
            for constraint in self._goal_spec.constraints
            if (
                constraint.required
                and constraint.constraint_type == "workspace_change"
                and constraint.expected_value is True
            )
        )
        workspace_changed = _has_net_workspace_change(
            tool_results,
            workspace_root=self._workspace_root,
        )
        if workspace_changed:
            context_bindings.extend(
                ContextBinding(
                    binding_id=f"runtime:workspace_change:{constraint.constraint_id}",
                    constraint_id=constraint.constraint_id,
                    status="satisfied",
                    rationale="A runtime write tool reported a real workspace change.",
                )
                for constraint in workspace_change_constraints
            )
        verification_constraints = tuple(
            constraint
            for constraint in self._goal_spec.constraints
            if (
                constraint.required
                and constraint.constraint_type == "verification_after_change"
                and constraint.expected_value is True
            )
        )
        if _verification_succeeded_after_latest_change(state):
            context_bindings.extend(
                ContextBinding(
                    binding_id=(
                        "runtime:verification_after_change:"
                        f"{constraint.constraint_id}"
                    ),
                    constraint_id=constraint.constraint_id,
                    status="satisfied",
                    rationale=(
                        "Every recognized verification command after the latest "
                        "workspace change completed successfully."
                    ),
                )
                for constraint in verification_constraints
            )
        evaluation = GoalContractEvaluator().evaluate(
            goal_spec=self._goal_spec,
            candidate=candidate,
            evidence_refs=self._collect_evidence_refs(tool_results),
            computation_results=self._collect_computation_results(tool_results),
            context_bindings=context_bindings,
        )
        if evaluation.satisfied:
            return StopVerdict(
                action="accept",
                code="goal_contract_satisfied",
            )
        return StopVerdict(
            action="block",
            code="goal_contract_unsatisfied",
            message="; ".join(issue.description for issue in evaluation.issues)
            or "Explicit goal contract is not satisfied.",
            detail={
                "unsatisfied_issue_ids": evaluation.issue_ids,
            },
        )


_DIRECT_VERIFICATION_EXECUTABLES = frozenset(
    {
        "biome",
        "eslint",
        "jest",
        "mypy",
        "nox",
        "pyright",
        "pytest",
        "tsc",
        "tox",
        "vitest",
    }
)
_VERIFICATION_SUBCOMMANDS = frozenset(
    {"build", "check", "clippy", "lint", "test", "typecheck", "verify", "vet"}
)
_PYTHON_VERIFICATION_MODULES = frozenset(
    {"compileall", "mypy", "pytest", "unittest"}
)
_MUTATING_VERIFICATION_FLAGS = frozenset(
    {"--apply", "--fix", "--update-snapshots", "--write"}
)
_NON_EXECUTING_VERIFICATION_ARGUMENTS = frozenset(
    {
        "--collect-only",
        "--co",
        "--fixtures",
        "--fixtures-per-test",
        "--help",
        "--list",
        "--list-tests",
        "--listtests",
        "--markers",
        "--print-config",
        "--setup-plan",
        "--show-config",
        "--show-files",
        "--show-settings",
        "--showconfig",
        "--trace-config",
        "--version",
        "list",
    }
)
_SHELL_FAILURE_MASK = re.compile(r"\|\||(?<!\|)\|(?!\|)|;|\n")
_UNSAFE_VERIFICATION_SHELL_SYNTAX = re.compile(
    r"\$\(|`|[<>]|(?<!&)&(?!&)"
)


def _verification_succeeded_after_latest_change(state: LoopState) -> bool:
    tool_results = list(state.get("tool_results", ()))
    latest_change_index = max(
        (
            index
            for index, result in enumerate(tool_results)
            if _is_runtime_workspace_change(result)
        ),
        default=-1,
    )
    if latest_change_index < 0:
        return False

    attempts: list[bool] = []
    calls = state.get("canonical_tool_calls", {})
    for result in tool_results[latest_change_index + 1 :]:
        if result.tool_name != "run_command":
            continue
        call = calls.get(result.tool_call_id)
        if call is None:
            continue
        command = call.arguments.get("command")
        if not isinstance(command, str) or not _is_verification_command(command):
            continue
        attempts.append(_command_result_succeeded(result))
    return bool(attempts) and all(attempts)


def _is_runtime_workspace_change(result: ToolResult) -> bool:
    return runtime_workspace_change(result) is not None


def _has_net_workspace_change(
    tool_results: Sequence[ToolResult],
    *,
    workspace_root: Path | None = None,
) -> bool:
    hashes_by_path: dict[str, tuple[str, str]] = {}
    for result in tool_results:
        change = runtime_workspace_change(result)
        if change is None:
            continue
        path, before_sha256, after_sha256 = change
        previous = hashes_by_path.get(path)
        if previous is not None and before_sha256 != previous[1]:
            return False
        original_sha256 = (
            before_sha256 if previous is None else previous[0]
        )
        hashes_by_path[path] = (original_sha256, after_sha256)
    if not any(
        before_sha256 != after_sha256
        for before_sha256, after_sha256 in hashes_by_path.values()
    ):
        return False
    if workspace_root is None:
        return True
    return all(
        _workspace_file_sha256(workspace_root, path) == after_sha256
        for path, (_before_sha256, after_sha256) in hashes_by_path.items()
    )


def _workspace_file_sha256(workspace_root: Path, path: str) -> str | None:
    try:
        target = (workspace_root / path).resolve()
        target.relative_to(workspace_root)
        if not target.is_file():
            return None
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except (OSError, ValueError):
        return None


def _command_result_succeeded(result: ToolResult) -> bool:
    output = result.structured_content
    return bool(
        not result.is_error
        and isinstance(output, Mapping)
        and output.get("exit_code") == 0
        and output.get("timed_out") is False
        and output.get("sandbox_error") in (None, "")
    )


def _is_verification_command(command: str) -> bool:
    """Recognize check commands without trusting a model-supplied purpose label."""

    if (
        _SHELL_FAILURE_MASK.search(command)
        or _UNSAFE_VERIFICATION_SHELL_SYNTAX.search(command)
    ):
        return False
    segments = [segment.strip() for segment in command.split("&&")]
    if any(_segment_uses_mutating_verification_flag(segment) for segment in segments):
        return False
    verified_segments = [
        _segment_runs_verification(segment)
        for segment in segments
    ]
    return bool(verified_segments) and all(verified_segments)


def _segment_uses_mutating_verification_flag(segment: str) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return True
    return any(_is_mutating_verification_flag(token.lower()) for token in tokens)


def _segment_runs_verification(segment: str) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False
    while tokens and _is_environment_assignment(tokens[0]):
        tokens.pop(0)
    if tokens and PurePath(tokens[0]).name == "env":
        tokens.pop(0)
        while tokens and (
            tokens[0].startswith("-") or _is_environment_assignment(tokens[0])
        ):
            tokens.pop(0)
    if len(tokens) >= 2 and PurePath(tokens[0]).name in {
        "pipenv",
        "poetry",
        "uv",
    }:
        if tokens[1] != "run":
            return False
        tokens = tokens[2:]
    if not tokens:
        return False

    executable = PurePath(tokens[0]).name.lower()
    arguments = [value.lower() for value in tokens[1:]]
    if any(_is_mutating_verification_flag(argument) for argument in arguments):
        return False
    if any(
        _is_non_executing_verification_argument(argument)
        for argument in arguments
    ):
        return False
    if executable in _DIRECT_VERIFICATION_EXECUTABLES:
        return True
    if executable == "ruff":
        return bool(arguments and arguments[0] == "check")
    if executable.startswith("python"):
        return any(
            arguments[index] == "-m"
            and arguments[index + 1] in _PYTHON_VERIFICATION_MODULES
            for index in range(len(arguments) - 1)
        )
    if executable in {"npm", "pnpm", "yarn", "bun"}:
        package_args = (
            arguments[1:]
            if arguments and arguments[0] == "run"
            else arguments
        )
        return bool(
            package_args
            and package_args[0] in _VERIFICATION_SUBCOMMANDS
        )
    if executable in {"cargo", "dotnet", "go", "make"}:
        return bool(
            arguments and arguments[0] in _VERIFICATION_SUBCOMMANDS
        )
    if executable in {"gradle", "gradlew", "mvn", "mvnw"}:
        return any(
            value.lstrip("-") in _VERIFICATION_SUBCOMMANDS
            for value in arguments
        )
    return False


def _is_mutating_verification_flag(value: str) -> bool:
    return bool(
        value in _MUTATING_VERIFICATION_FLAGS
        or any(
            value.startswith(f"{flag}=")
            for flag in _MUTATING_VERIFICATION_FLAGS
        )
    )


def _is_non_executing_verification_argument(value: str) -> bool:
    return bool(
        value in _NON_EXECUTING_VERIFICATION_ARGUMENTS
        or any(
            value.startswith(f"{argument}=")
            for argument in _NON_EXECUTING_VERIFICATION_ARGUMENTS
            if argument.startswith("--")
        )
    )


def _is_environment_assignment(value: str) -> bool:
    name, separator, _assigned = value.partition("=")
    return bool(
        separator
        and name
        and (name[0].isalpha() or name[0] == "_")
        and all(character.isalnum() or character == "_" for character in name)
    )


def build_stop_hooks(
    *,
    definition: AgentRuntimePolicy,
    output_finalizer: StructuredOutputFinalizer | None = None,
    goal_spec: GoalSpec | None = None,
    workspace_root: Path | None = None,
) -> tuple[StopHookBinding, ...]:
    hooks: list[StopHookBinding] = []
    if goal_spec is not None:
        hooks.append(
            StopHookBinding(
                name="goal_contract",
                hook=GoalContractStopHook(
                    goal_spec=goal_spec,
                    workspace_root=workspace_root,
                ),
                critical=True,
            )
        )
    if definition.output_model is not None:
        hooks.append(
            StopHookBinding(
                name="structured_output",
                hook=StructuredOutputStopHook(
                    definition=definition,
                    finalizer=output_finalizer,
                ),
                critical=True,
            )
        )
    return tuple(hooks)


async def _await_output(value: object) -> BaseModel:
    from inspect import isawaitable

    if isawaitable(value):
        value = await value
    if not isinstance(value, BaseModel):
        raise TypeError("structured output finalizer returned a non-model value")
    return value


def _structured_items(
    result: ToolResult,
    key: str,
) -> tuple[Mapping[str, object], ...]:
    if result.is_error or not isinstance(result.structured_content, Mapping):
        return ()
    raw = result.structured_content.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


__all__ = [
    "GoalContractStopHook",
    "StopHook",
    "StopHookBinding",
    "StopHookOutcome",
    "StopHookRunner",
    "StopVerdict",
    "StructuredOutputStopHook",
    "build_stop_hooks",
]
