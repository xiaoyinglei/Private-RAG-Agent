from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from rag.agent.tools.tool import JsonValue, ResolvedToolUse, Tool, ToolEffect


class UseToolDecision(StrEnum):
    """Pure permission outcome; human interaction is deliberately external."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class CanUseToolResult:
    decision: UseToolDecision
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.decision, UseToolDecision):
            raise TypeError("decision must be a UseToolDecision")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Runtime facts consumed by guards, permission, and external approval."""

    workspace_root: Path | str | None = None
    cwd: Path | str | None = None
    allow_write_tools: bool = False
    allow_execute_tools: bool = False
    approved_tool_call_ids: frozenset[str] = frozenset()
    denied_tool_call_ids: frozenset[str] = frozenset()
    deny_effects: frozenset[ToolEffect] = frozenset()

    def __post_init__(self) -> None:
        for name in ("allow_write_tools", "allow_execute_tools"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        for name in ("approved_tool_call_ids", "denied_tool_call_ids"):
            values = frozenset(getattr(self, name))
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)
        if self.approved_tool_call_ids & self.denied_tool_call_ids:
            raise ValueError("a tool call cannot be both approved and denied")

        effects = frozenset(self.deny_effects)
        if any(not isinstance(effect, ToolEffect) for effect in effects):
            raise TypeError("deny_effects must contain ToolEffect values")
        object.__setattr__(self, "deny_effects", effects)

        for name in ("workspace_root", "cwd"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())


class ToolGuardError(ValueError):
    """Bounded hard-guard failure that approval must never bypass."""

    def __init__(self, code: str, reason: str) -> None:
        if not isinstance(code, str) or not code:
            raise ValueError("guard code must be a non-empty string")
        if not isinstance(reason, str) or not reason:
            raise ValueError("guard reason must be a non-empty string")
        self.code = code
        self.reason = " ".join(reason.split())[:512]
        super().__init__(self.reason)


def can_use_tool(
    tool: Tool,
    args: Mapping[str, JsonValue],
    resolved: ResolvedToolUse,
    context: ToolExecutionContext,
) -> CanUseToolResult:
    """Return allow/ask/deny from resolved facts without performing approval."""

    del tool, args
    denied = resolved.effects & context.deny_effects
    if denied:
        names = ", ".join(sorted(effect.value for effect in denied))
        return CanUseToolResult(
            UseToolDecision.DENY,
            f"effects denied by runtime policy: {names}",
        )

    approval_reasons: list[str] = []
    if (
        ToolEffect.WRITE_WORKSPACE in resolved.effects
        and not context.allow_write_tools
    ):
        approval_reasons.append("workspace write")
    if (
        ToolEffect.EXECUTE_PROCESS in resolved.effects
        and not context.allow_execute_tools
    ):
        approval_reasons.append("process execution")
    if ToolEffect.NETWORK in resolved.effects:
        approval_reasons.append("network access")
    if ToolEffect.DESTRUCTIVE in resolved.effects:
        approval_reasons.append("destructive operation")
    if approval_reasons:
        return CanUseToolResult(
            UseToolDecision.ASK,
            "approval required for " + ", ".join(approval_reasons),
        )
    return CanUseToolResult(UseToolDecision.ALLOW, "resolved effects are allowed")


def enforce_hard_guards(
    tool: Tool,
    args: Mapping[str, JsonValue],
    resolved: ResolvedToolUse,
    context: ToolExecutionContext,
) -> None:
    """Enforce non-bypassable effect floors and workspace/cwd containment."""

    del args
    if not tool.static_effects.issubset(resolved.effects):
        raise ToolGuardError(
            "effect_floor_violation",
            "resolved effects cannot remove the tool static effect floor",
        )

    workspace_targets = [
        target for target in resolved.targets if target.kind == "workspace_path"
    ]
    if ToolEffect.WRITE_WORKSPACE in resolved.effects and not workspace_targets:
        raise ToolGuardError(
            "workspace_target_required",
            "workspace writes require a resolved workspace_path target",
        )
    for target in workspace_targets:
        if context.workspace_root is None:
            raise ToolGuardError(
                "workspace_root_unavailable",
                "workspace target cannot be checked without a workspace root",
            )
        _enforce_containment(
            target.value,
            root=Path(context.workspace_root),
            code="workspace_escape",
            label="workspace",
        )

    cwd_targets = [target for target in resolved.targets if target.kind == "cwd_path"]
    if ToolEffect.EXECUTE_PROCESS in resolved.effects and context.cwd is None:
        raise ToolGuardError(
            "cwd_required",
            "process execution requires a resolved runtime cwd",
        )
    for target in cwd_targets:
        if context.cwd is None:
            raise ToolGuardError(
                "cwd_required",
                "cwd target cannot be checked without a runtime cwd",
            )
        _enforce_containment(
            target.value,
            root=Path(context.cwd),
            code="cwd_escape",
            label="cwd",
        )
    if context.workspace_root is not None and context.cwd is not None:
        _enforce_containment(
            str(context.cwd),
            root=Path(context.workspace_root),
            code="cwd_escape",
            label="workspace",
        )


def _enforce_containment(
    value: str,
    *,
    root: Path,
    code: str,
    label: str,
) -> None:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        common = Path(os.path.commonpath((str(root), str(candidate))))
    except ValueError:
        common = Path()
    if common != root:
        raise ToolGuardError(code, f"target escapes {label}")

__all__ = [
    "CanUseToolResult",
    "ToolExecutionContext",
    "ToolGuardError",
    "UseToolDecision",
    "can_use_tool",
    "enforce_hard_guards",
]
