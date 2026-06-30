from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_RUNTIME_DIAGNOSTICS = 20
MAX_RUNTIME_DIAGNOSTIC_MESSAGE_LENGTH = 500


class RuntimeDiagnostic(BaseModel):
    """Bounded, checkpoint-safe description of degraded runtime behavior."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(min_length=1, max_length=120)
    component: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=MAX_RUNTIME_DIAGNOSTIC_MESSAGE_LENGTH)
    severity: Literal["warning", "error"] = "warning"
    degraded: bool = True
    error_type: str | None = Field(default=None, max_length=120)

    @property
    def identity(self) -> tuple[str, str]:
        return self.code, self.component

    @classmethod
    def from_exception(
        cls,
        *,
        code: str,
        component: str,
        error: Exception,
        severity: Literal["warning", "error"] = "warning",
        degraded: bool = True,
    ) -> RuntimeDiagnostic:
        message = str(error).strip() or type(error).__name__
        return cls(
            code=code,
            component=component,
            message=message[:MAX_RUNTIME_DIAGNOSTIC_MESSAGE_LENGTH],
            severity=severity,
            degraded=degraded,
            error_type=type(error).__name__[:120],
        )


# Tool call metrics.


class ToolCallMetrics(BaseModel):
    """Lightweight counters for the three calling modes.

    Populated by AgentLoop during tool execution and attached to the loop
    state. A compact RuntimeDiagnostic summary is emitted for inline display.
    """

    model_config = ConfigDict(frozen=True)

    # Mode 1: Native direct calls
    native_calls: int = 0
    native_errors: int = 0
    native_latency_ms_total: float = 0.0

    # Mode 2: Deferred tool calls
    tool_search_calls: int = 0
    tool_search_hits: int = 0      # searches that returned ≥1 candidate
    activate_tools_calls: int = 0
    deferred_activations: int = 0
    deferred_calls: int = 0         # calls to activated deferred tools

    # Mode 3: Programmatic / batch
    tool_repl_calls: int = 0
    batch_declarations: int = 0    # total tools.declare() calls

    # MCP
    mcp_calls: int = 0
    mcp_errors: int = 0
    mcp_latency_ms_total: float = 0.0

    # Approval
    approval_allow: int = 0
    approval_deny: int = 0
    approval_ask: int = 0

    # Context budget
    context_tokens_start: int = 0
    context_tokens_end: int = 0

    @property
    def token_savings_pct(self) -> float:
        """Estimated token savings from deferred + programmatic modes."""
        if self.context_tokens_start == 0:
            return 0.0
        return (1.0 - self.context_tokens_end / self.context_tokens_start) * 100

    @property
    def tool_search_hit_rate(self) -> float:
        if self.tool_search_calls == 0:
            return 0.0
        return self.tool_search_hits / self.tool_search_calls * 100

    @property
    def mcp_avg_latency_ms(self) -> float:
        if self.mcp_calls == 0:
            return 0.0
        return self.mcp_latency_ms_total / self.mcp_calls

    @property
    def approval_ask_rate(self) -> float:
        total = self.approval_allow + self.approval_deny + self.approval_ask
        if total == 0:
            return 0.0
        return self.approval_ask / total * 100


def merge_runtime_diagnostics(
    left: Iterable[RuntimeDiagnostic],
    right: Iterable[RuntimeDiagnostic],
) -> list[RuntimeDiagnostic]:
    merged: dict[tuple[str, str], RuntimeDiagnostic] = {}
    for diagnostic in [*left, *right]:
        key = diagnostic.identity
        merged.pop(key, None)
        merged[key] = diagnostic
    return list(merged.values())[-MAX_RUNTIME_DIAGNOSTICS:]


__all__ = [
    "MAX_RUNTIME_DIAGNOSTICS",
    "RuntimeDiagnostic",
    "ToolCallMetrics",
    "merge_runtime_diagnostics",
]
