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
    logical_input_tokens: int | None = None
    uncached_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    usage_source: str | None = None


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
        latency_profile = getattr(result, "latency_profile", None)
        latency_ms = (
            float(getattr(latency_profile, "total_ms", 0.0))
            if latency_profile is not None
            else sum(float(getattr(tool, "latency_ms", 0.0)) for tool in tool_results)
        )
        records = tuple(
            getattr(result, "model_call_records", ()) or ()
        )
        usages = tuple(record.usage for record in records)
        input_tokens = sum(usage.input_tokens for usage in usages)
        output_tokens = sum(usage.output_tokens for usage in usages)
        return cls(
            answer=getattr(result, "final_answer", None),
            status=str(getattr(result, "status", "unknown")),
            files=files,
            tool_calls=tuple(str(getattr(tool, "tool_name", "")) for tool in tool_results),
            citations=tuple(getattr(result, "citations", ()) or ()),
            usage=AgentUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                tool_calls=len(tool_results),
                latency_ms=latency_ms,
                logical_input_tokens=_sum_optional(
                    tuple(usage.logical_input_tokens for usage in usages)
                ),
                uncached_input_tokens=_sum_optional(
                    tuple(usage.uncached_input_tokens for usage in usages)
                ),
                cache_read_input_tokens=_sum_optional(
                    tuple(usage.cache_read_input_tokens for usage in usages)
                ),
                cache_write_input_tokens=_sum_optional(
                    tuple(usage.cache_write_input_tokens for usage in usages)
                ),
                usage_source=_usage_source(usages),
            ),
            diagnostics=tuple(getattr(result, "runtime_diagnostics", ()) or ()),
            run_id=str(getattr(result, "run_id", "")),
            thread_id=str(getattr(result, "thread_id", "")),
            raw=result,
        )


def _sum_optional(values: tuple[int | None, ...]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _usage_source(usages: tuple[Any, ...]) -> str | None:
    values = tuple(
        str(usage.usage_source)
        for usage in usages
        if usage.usage_source is not None
    )
    if not values:
        return None
    return values[0] if len(set(values)) == 1 else "mixed"
