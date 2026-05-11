from __future__ import annotations

from langgraph.types import interrupt
from pydantic import ValidationError

from rag.agent.core.human_input import HumanInputResponse
from rag.agent.state import AgentState


class HumanInputRequestIdMismatchError(RuntimeError):
    """恢复时的 request_id 与当前 human_input_request.request_id 不匹配。"""


def pause_node(state: AgentState) -> dict:
    """暂停执行，等待用户输入。

    只做三件事：
    1. 读取 state.human_input_request → 调用 interrupt()
    2. 校验恢复值为 HumanInputResponse
    3. 写入 human_input_response、approved_tool_call_ids 等
    """
    request = state.get("human_input_request")
    if request is None:
        # 没有正式的 HumanInputRequest — 兼容旧 needs_user_input 路径
        fallback_question = state.get("needs_user_input", "需要你的决策")
        raw = interrupt({"question": fallback_question, "kind": "legacy_pause"})
        return {
            "user_decision": str(raw) if not isinstance(raw, dict) else raw.get("decision", str(raw)),
            "status": "running",
        }

    # 正式路径：HumanInputRequest → interrupt → HumanInputResponse
    raw = interrupt(request.model_dump(mode="json"))

    try:
        response = HumanInputResponse.model_validate(raw)
    except ValidationError as exc:
        # 恢复 payload 不合法 → 返回错误信息让调用方重试
        return {
            "status": "paused",
            "needs_user_input": f"Invalid response format: {exc}. Please provide a valid HumanInputResponse.",
        }

    # 校验 request_id 匹配
    if response.request_id != request.request_id:
        raise HumanInputRequestIdMismatchError(
            f"Response request_id={response.request_id!r} does not match "
            f"current request_id={request.request_id!r}"
        )

    existing_approved = list(state.get("approved_tool_call_ids", []))
    existing_denied = list(state.get("denied_tool_call_ids", []))

    return {
        "user_decision": response.decision,
        "user_message": response.user_message,
        "human_input_response": response,
        "approved_tool_call_ids": existing_approved + response.approved_tool_call_ids,
        "denied_tool_call_ids": existing_denied + response.denied_tool_call_ids,
        "status": "running",
    }
