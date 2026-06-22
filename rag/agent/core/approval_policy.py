from __future__ import annotations

from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel

from rag.agent.core.human_input import HumanInputRequest, ToolCallSummary
from rag.agent.tools.spec import ExecutionCategory, RiskLevel, ToolSpec


class ApprovalAction(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ApprovalDecision(BaseModel):
    """ApprovalPolicy.decide() 的返回值。"""

    action: ApprovalAction
    reason: str
    risk_level: str = "low"  # "low" | "medium" | "high"
    request: HumanInputRequest | None = None  # action=ASK 时非空


_DENY_TOOLS = frozenset({
    "shell", "delete_file", "delete_directory", "drop_table",
})


class ApprovalPolicy:
    """基于 ToolSpec.execution_category 的工具审批策略。

    规则（优先级从高到低）：
    1. spec=None → DENY（未注册）
    2. tool_name 在 DENY_TOOLS 中 → DENY
    3. 工具契约或运行时策略要求确认 → ASK
    4. EXECUTE + auto_approve_sandboxed + 无网络/持久变更权限 → ALLOW
    5. EXECUTE / WRITE / MUTATE / NETWORK / SYSTEM → ASK
    6. READ / TRANSFORM → ALLOW
    """

    def decide(
        self,
        *,
        tool_name: str,
        arguments: dict[str, object],
        spec: ToolSpec | None,
        requires_confirmation: bool = False,
        auto_approve_sandboxed: bool = False,
    ) -> ApprovalDecision:
        if spec is None:
            return ApprovalDecision(
                action=ApprovalAction.DENY,
                reason=f"未注册工具: {tool_name}",
                risk_level="high",
            )

        # Deny tools by exact name match
        if tool_name in _DENY_TOOLS:
            return ApprovalDecision(
                action=ApprovalAction.DENY,
                reason=f"高风险工具被拒绝: {tool_name}",
                risk_level="high",
            )

        if requires_confirmation or spec.requires_confirmation:
            return self._ask_decision(
                tool_name=tool_name,
                arguments=arguments,
                spec=spec,
                risk_level=_risk_value(spec),
                reason="工具契约要求执行前确认",
            )

        category = spec.execution_category
        if not isinstance(category, ExecutionCategory):
            return self._ask_decision(
                tool_name=tool_name,
                arguments=arguments,
                spec=spec,
                risk_level="high",
                reason="工具执行类别无效，需要人工确认",
            )

        # Sandbox auto-approve: code execution inside a sandbox is safe
        # by boundary (restricted fs, no network, timeout), not by
        # user clicking "confirm" each time. Mirrors Claude/GPT behavior.
        if (
            category == ExecutionCategory.EXECUTE
            and auto_approve_sandboxed
            and _sandbox_auto_approvable(spec)
        ):
            return ApprovalDecision(
                action=ApprovalAction.ALLOW,
                reason="沙箱内代码执行，自动放行",
                risk_level="low",
            )

        # Ask for write / mutate / execute / network / system
        if category in (
            ExecutionCategory.WRITE,
            ExecutionCategory.MUTATE,
            ExecutionCategory.EXECUTE,
            ExecutionCategory.NETWORK,
            ExecutionCategory.SYSTEM,
        ):
            reason_map = {
                ExecutionCategory.WRITE: "写入操作需要确认",
                ExecutionCategory.MUTATE: "不可逆变更操作需要确认",
                ExecutionCategory.EXECUTE: "代码执行需要确认",
                ExecutionCategory.NETWORK: "外部网络访问需要确认",
                ExecutionCategory.SYSTEM: "系统操作需要确认",
            }
            risk_map = {
                ExecutionCategory.WRITE: "medium",
                ExecutionCategory.MUTATE: "high",
                ExecutionCategory.EXECUTE: "medium",
                ExecutionCategory.NETWORK: "medium",
                ExecutionCategory.SYSTEM: "high",
            }
            return self._ask_decision(
                tool_name=tool_name,
                arguments=arguments,
                spec=spec,
                risk_level=_risk_value(spec, minimum=risk_map[category]),
                reason=reason_map[category],
            )

        if spec.permissions_require_approval:
            return self._ask_decision(
                tool_name=tool_name,
                arguments=arguments,
                spec=spec,
                risk_level=_risk_value(spec),
                reason="工具权限包含写入、变更、代码执行或外部访问，需要确认",
            )

        # READ / TRANSFORM → allow
        return ApprovalDecision(
            action=ApprovalAction.ALLOW,
            reason="只读工具，自动允许",
            risk_level="low",
        )

    def _ask_decision(
        self,
        *,
        tool_name: str,
        arguments: dict[str, object],
        spec: ToolSpec,
        risk_level: str,
        reason: str,
    ) -> ApprovalDecision:
        args_preview = ", ".join(
            f"{key}={_truncate_arg(repr(value))}"
            for key, value in list(arguments.items())[:5]
        )
        summary = ToolCallSummary(
            tool_call_id=str(arguments.get("tool_call_id", "?")),
            tool_name=tool_name,
            args_preview=args_preview or "(无参数)",
            risk_level=risk_level,
            reason=reason,
        )
        request = HumanInputRequest(
            request_id=f"hir_{uuid4().hex[:12]}",
            kind="tool_approval",
            question=f"确认执行工具: {tool_name}",
            tool_calls=[summary],
            context={"tool_name": tool_name, "risk_level": risk_level},
            options=["allow_once", "deny", "abort"],
        )
        return ApprovalDecision(
            action=ApprovalAction.ASK,
            reason=reason,
            risk_level=risk_level,
            request=request,
        )


_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def merge_approval_requests(decisions: list[ApprovalDecision]) -> HumanInputRequest:
    """将多个 ASK 决策合并为一个 HumanInputRequest。"""
    if not decisions:
        raise ValueError("decisions must not be empty")

    all_summaries: list[ToolCallSummary] = []
    tool_names: list[str] = []
    for d in decisions:
        if d.request and d.request.tool_calls:
            all_summaries.extend(d.request.tool_calls)
            tool_names.append(d.request.tool_calls[0].tool_name)

    max_risk = max(
        (d.risk_level for d in decisions),
        key=lambda r: _RISK_ORDER.get(r, 0),
    )
    return HumanInputRequest(
        request_id=f"hir_{uuid4().hex[:12]}",
        kind="tool_approval",
        question=f"确认执行以下工具: {', '.join(tool_names)}",
        tool_calls=all_summaries,
        context={"tool_count": len(all_summaries), "risk_level": max_risk},
        options=["allow_once", "deny", "abort"],
    )


def _truncate_arg(value: str, max_len: int = 60) -> str:
    if len(value) <= max_len:
        return value
    return value[:max_len] + "..."


def _risk_value(spec: ToolSpec, *, minimum: str | None = None) -> str:
    risk = spec.risk_level
    if isinstance(risk, RiskLevel):
        risk_level = risk
    elif isinstance(risk, str):
        try:
            risk_level = RiskLevel(risk)
        except ValueError:
            risk_level = RiskLevel.HIGH
    else:
        risk_level = RiskLevel.HIGH

    minimum_level = spec.minimum_risk_level
    if minimum is not None:
        minimum_level = max(
            minimum_level,
            RiskLevel(minimum),
            key=lambda level: _RISK_ORDER[level],
        )
    return max(risk_level, minimum_level, key=lambda level: _RISK_ORDER[level]).value


def _sandbox_auto_approvable(spec: ToolSpec) -> bool:
    permissions = spec.permissions
    return permissions.execute_code and not (
        permissions.external_network
        or permissions.write_db
        or permissions.kg_mutation
        or permissions.user_data
    )


_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
}
