from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0


@dataclass(frozen=True)
class AgentResult:
    answer: str | None
    status: str
    files: tuple[str, ...]
    tool_calls: tuple[str, ...]
    citations: tuple[Any, ...]
    usage: AgentUsage
    diagnostics: tuple[Any, ...]
    run_id: str
    thread_id: str
    raw: object | None

    @classmethod
    def from_internal(
        cls,
        result: Any,
        *,
        files: tuple[str, ...] = (),
    ) -> AgentResult:
        tool_results = tuple(getattr(result, "tool_results", ()) or ())
        latency_ms = sum(float(getattr(tool, "latency_ms", 0.0)) for tool in tool_results)
        return cls(
            answer=getattr(result, "final_answer", None),
            status=str(getattr(result, "status", "unknown")),
            files=files,
            tool_calls=tuple(str(getattr(tool, "tool_name", "")) for tool in tool_results),
            citations=tuple(getattr(result, "citations", ()) or ()),
            usage=AgentUsage(
                tool_calls=len(tool_results),
                latency_ms=latency_ms,
            ),
            diagnostics=tuple(getattr(result, "runtime_diagnostics", ()) or ()),
            run_id=str(getattr(result, "run_id", "")),
            thread_id=str(getattr(result, "thread_id", "")),
            raw=result,
        )
