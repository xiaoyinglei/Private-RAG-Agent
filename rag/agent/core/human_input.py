from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ToolCallSummary(BaseModel):
    """面向用户的工具摘要，不暴露完整 ToolCallPlan。"""

    tool_call_id: str
    tool_name: str
    args_preview: str  # 如 "query='公积金政策', top_k=8"
    risk_level: str = "low"  # "low" | "medium" | "high"
    reason: str = ""  # 为什么需要审批


class HumanInputRequest(BaseModel):
    """Agent 暂停时向用户发送的输入请求。

    由 execute_node（经 ApprovalPolicy）或 evaluate_node（经 LLM）生成，
    存入 state.human_input_request。pause_node 读取后调用 interrupt()。
    """

    request_id: str  # 唯一 ID，如 "hir_a1b2c3d4"
    kind: Literal["tool_approval", "choice", "clarification"]
    question: str  # 面向用户的自然语言问题
    tool_calls: list[ToolCallSummary] = Field(default_factory=list)
    context: dict[str, object] = Field(default_factory=dict)
    options: list[str] = Field(default_factory=list)


class HumanInputResponse(BaseModel):
    """用户对 HumanInputRequest 的响应。

    由 pause_node 通过 interrupt() 接收，校验后写入 state。
    """

    request_id: str
    decision: Literal["allow_once", "deny", "continue", "abort"]
    approved_tool_call_ids: list[str] = Field(default_factory=list)
    denied_tool_call_ids: list[str] = Field(default_factory=list)
    user_message: str | None = None
