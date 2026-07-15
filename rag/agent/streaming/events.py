"""
StreamEvent 定义 — 流式输出的基础类型。

设计原则：
- 不可变（frozen=True），可安全在协程间传递
- 高频字段提到顶层，不全塞 metadata
- EventType 用 Enum，方便 UI 层 pattern match
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    """流式事件类型。"""

    # ── LLM 流式输出 ──────────────────────────────────────
    TEXT_DELTA = "text_delta"  # 文本增量
    THINKING_DELTA = "thinking_delta"  # 思考增量（extended thinking，可选）

    # ── 工具生命周期 ──────────────────────────────────────
    TOOL_USE_START = "tool_use_start"  # 工具开始执行
    TOOL_USE_PROGRESS = "tool_use_progress"  # 工具执行进度
    TOOL_USE_RESULT = "tool_use_result"  # 工具执行完成
    TOOL_USE_ERROR = "tool_use_error"  # 工具执行失败

    # ── 上下文管理 ────────────────────────────────────────
    COMPACT_LAYER = "compact_layer"  # 单层压缩完成

    # ── 计划状态 ──────────────────────────────────────────
    PLAN_UPDATED = "plan_updated"  # update_plan 已写入 canonical PlanState

    # ── 会话控制 ──────────────────────────────────────────
    TURN_START = "turn_start"  # 一轮开始
    TURN_END = "turn_end"  # 一轮结束
    LOOP_END = "loop_end"  # 循环结束
    HUMAN_INPUT_REQUIRED = "human_input_required"  # 已持久化的人机交互请求
    RECOVERY = "recovery"  # 恢复尝试
    ABORT = "abort"  # 用户取消

    # ── Token 预算 ────────────────────────────────────────
    BUDGET_UPDATE = "budget_update"  # 预算消耗更新


@dataclass(frozen=True)
class StreamEvent:
    """流式事件 — 不可变，可安全在协程间传递。"""

    type: EventType
    run_id: str = ""
    session_id: str = ""
    turn: int = 0
    seq: int = 0
    timestamp_ms: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    span_id: str | None = None  # 关联同一个 tool 调用的 start/progress/result
    parent_id: str | None = None  # 子 agent / 嵌套事件

    def __post_init__(self) -> None:
        if self.timestamp_ms == 0:
            object.__setattr__(self, "timestamp_ms", _now_ms())

    @property
    def turn_id(self) -> str:
        """Public Session/Turn name for the legacy event run identifier."""

        return self.run_id


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


_seq_counter = 0


def next_seq() -> int:
    global _seq_counter
    _seq_counter += 1
    return _seq_counter


# ── 工厂函数 ──────────────────────────────────────────────


def text_delta(
    text: str,
    *,
    run_id: str = "",
    turn: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.TEXT_DELTA,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        data={"text": text},
    )


def thinking_delta(
    text: str,
    *,
    run_id: str = "",
    turn: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.THINKING_DELTA,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        data={"text": text},
    )


def tool_use_start(
    tool_name: str,
    tool_id: str,
    *,
    input_preview: str = "",
    run_id: str = "",
    turn: int = 0,
) -> StreamEvent:
    span = f"tool:{tool_id}"
    return StreamEvent(
        type=EventType.TOOL_USE_START,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        span_id=span,
        data={
            "tool_name": tool_name,
            "tool_id": tool_id,
            "input_preview": input_preview,
        },
    )


def tool_use_progress(
    tool_id: str,
    progress: str,
    *,
    percent: float | None = None,
    run_id: str = "",
    turn: int = 0,
) -> StreamEvent:
    span = f"tool:{tool_id}"
    d: dict[str, Any] = {"tool_id": tool_id, "progress": progress}
    if percent is not None:
        d["percent"] = percent
    return StreamEvent(
        type=EventType.TOOL_USE_PROGRESS,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        span_id=span,
        data=d,
    )


def tool_use_result(
    tool_name: str,
    tool_id: str,
    result: Any,
    *,
    elapsed_ms: float = 0,
    run_id: str = "",
    turn: int = 0,
) -> StreamEvent:
    span = f"tool:{tool_id}"
    return StreamEvent(
        type=EventType.TOOL_USE_RESULT,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        span_id=span,
        data={
            "tool_name": tool_name,
            "tool_id": tool_id,
            "result": result,
            "elapsed_ms": elapsed_ms,
        },
    )


def tool_use_error(
    tool_id: str,
    error: str,
    *,
    recoverable: bool = True,
    run_id: str = "",
    turn: int = 0,
) -> StreamEvent:
    span = f"tool:{tool_id}"
    return StreamEvent(
        type=EventType.TOOL_USE_ERROR,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        span_id=span,
        data={
            "tool_id": tool_id,
            "error": error,
            "recoverable": recoverable,
        },
    )


def compact_layer(
    layer_name: str,
    before_tokens: int,
    after_tokens: int,
    *,
    run_id: str = "",
    turn: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.COMPACT_LAYER,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        data={
            "layer": layer_name,
            "before": before_tokens,
            "after": after_tokens,
            "reduction": before_tokens - after_tokens,
        },
    )


def turn_start(*, run_id: str = "", turn: int = 0) -> StreamEvent:
    return StreamEvent(
        type=EventType.TURN_START,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
    )


def turn_end(
    *,
    run_id: str = "",
    turn: int = 0,
    stop_reason: str = "",
) -> StreamEvent:
    return StreamEvent(
        type=EventType.TURN_END,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        data={"stop_reason": stop_reason},
    )


def loop_end(
    *,
    reason: str,
    run_id: str = "",
    total_turns: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.LOOP_END,
        run_id=run_id,
        seq=next_seq(),
        data={"reason": reason, "total_turns": total_turns},
    )


def recovery_event(
    strategy: str,
    detail: str = "",
    *,
    run_id: str = "",
    turn: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.RECOVERY,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        data={"strategy": strategy, "detail": detail},
    )


def budget_update(
    used: int,
    remaining: int,
    *,
    run_id: str = "",
    turn: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.BUDGET_UPDATE,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        data={"used": used, "remaining": remaining},
    )
