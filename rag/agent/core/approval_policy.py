from __future__ import annotations

from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel

from rag.agent.core.human_input import HumanInputRequest, ToolCallSummary
from rag.agent.tools.spec import ToolSpec


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
    """基于 ToolSpec.permissions 的工具审批策略。

    规则（优先级从高到低）：
    1. spec=None → DENY（未注册）
    2. tool_name 在 DENY_TOOLS 中 → DENY
    3. permissions.write_db / kg_mutation / user_data → ASK
    4. permissions.external_network → ASK
    5. 其余 → ALLOW
    """

    def decide(
        self,
        *,
        tool_name: str,
        arguments: dict[str, object],
        spec: ToolSpec | None,
    ) -> ApprovalDecision:
        if spec is None:
            return ApprovalDecision(
                action=ApprovalAction.DENY,
                reason=f"未注册工具: {tool_name}",
                risk_level="high",
            )

        permissions = spec.permissions

        # Deny tools by exact name match
        if tool_name in _DENY_TOOLS:
            return ApprovalDecision(
                action=ApprovalAction.DENY,
                reason=f"高风险工具被拒绝: {tool_name}",
                risk_level="high",
            )

        # Ask for write / mutation / user data
        if permissions.write_db or permissions.kg_mutation or permissions.user_data:
            return self._ask_decision(
                tool_name=tool_name,
                arguments=arguments,
                spec=spec,
                risk_level="medium",
                reason="写入或数据变更操作需要确认",
            )

        # Ask for external network
        if permissions.external_network:
            return self._ask_decision(
                tool_name=tool_name,
                arguments=arguments,
                spec=spec,
                risk_level="medium",
                reason="外部网络访问需要确认",
            )

        # Allow read-only tools
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
            tool_call_id=arguments.get("tool_call_id", "?"),
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

    first = decisions[0]
    return HumanInputRequest(
        request_id=f"hir_{uuid4().hex[:12]}",
        kind="tool_approval",
        question=f"确认执行以下工具: {', '.join(tool_names)}",
        tool_calls=all_summaries,
        context={"tool_count": len(all_summaries), "risk_level": first.risk_level},
        options=["allow_once", "deny", "abort"],
    )


def _truncate_arg(value: str, max_len: int = 60) -> str:
    if len(value) <= max_len:
        return value
    return value[:max_len] + "..."
