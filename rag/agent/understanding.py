"""Task-level understanding that sits above retrieval intent parsing."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from rag.agent.schema import AgentTaskRequest, TaskUnderstanding
from rag.assembly import ChatCapabilityBinding
from rag.schema.query import RetrievalSignals

_TASK_UNDERSTANDING_PROMPT = """You are the task-understanding module for an evidence-grounded analysis agent.
Return exactly one JSON object and nothing else.
The output must describe the task-level objective and deliverable, not retrieval tactics.
Use enum values exactly:
- task_type: lookup | single_doc_qa | comparison | synthesis | timeline | research
Return JSON matching this schema:
{
  "task_type": "research",
  "deliverable_type": "analysis_report",
  "decomposition_required": true,
  "needs_external_evidence": false,
  "needs_comparison": false,
  "needs_timeline": false,
  "success_criteria": []
}
If uncertain, keep the output conservative and avoid inventing requirements.
"""


class TaskUnderstandingDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    llm_provider: str | None = None
    llm_model: str | None = None
    llm_latency_ms: float | None = None
    llm_raw_response: str | None = None
    llm_parsed_result: TaskUnderstanding | None = None
    final_understanding: TaskUnderstanding
    fallback_used: bool = False
    fallback_reason: str | None = None
    retrieval_signals_debug: dict[str, object] = {}


class TaskUnderstandingService:
    def __init__(
        self,
        *,
        chat_bindings: Sequence[ChatCapabilityBinding] = (),
        enable_llm: bool = True,
    ) -> None:
        self._chat_bindings = tuple(chat_bindings)
        self._enable_llm = enable_llm
        self.last_diagnostics: TaskUnderstandingDiagnostics | None = None

    def analyze(
        self,
        request: AgentTaskRequest,
        *,
        retrieval_signals: RetrievalSignals | None = None,
    ) -> TaskUnderstanding:
        understanding, diagnostics = self.analyze_with_diagnostics(
            request, retrieval_signals=retrieval_signals
        )
        self.last_diagnostics = diagnostics
        return understanding

    def analyze_with_diagnostics(
        self,
        request: AgentTaskRequest,
        *,
        retrieval_signals: RetrievalSignals | None = None,
    ) -> tuple[TaskUnderstanding, TaskUnderstandingDiagnostics]:
        signals = retrieval_signals or RetrievalSignals()
        llm_result, provider, model, latency_ms, raw_response, fallback_reason = (
            self._understand_with_llm(request=request, retrieval_signals=signals)
        )
        final_understanding = llm_result or self._fallback_understanding(
            request=request, retrieval_signals=signals
        )
        diagnostics = TaskUnderstandingDiagnostics(
            llm_provider=provider,
            llm_model=model,
            llm_latency_ms=latency_ms,
            llm_raw_response=raw_response,
            llm_parsed_result=llm_result,
            final_understanding=final_understanding,
            fallback_used=llm_result is None,
            fallback_reason=fallback_reason,
            retrieval_signals_debug=signals.model_dump(mode="json"),
        )
        self.last_diagnostics = diagnostics
        return final_understanding, diagnostics

    def diagnostics_payload(self) -> dict[str, object]:
        if self.last_diagnostics is None:
            return {}
        return self.last_diagnostics.model_dump(mode="json")

    def _understand_with_llm(
        self,
        *,
        request: AgentTaskRequest,
        retrieval_signals: RetrievalSignals,
    ) -> tuple[TaskUnderstanding | None, str | None, str | None, float | None, str | None, str | None]:
        if not self._enable_llm:
            return None, None, None, None, None, "llm_disabled"
        if not self._chat_bindings:
            return None, None, None, None, None, "no_chat_binding"
        prompt = self._build_prompt(request=request, retrieval_signals=retrieval_signals)
        fallback_reason = "llm_unavailable"
        for binding in self._chat_bindings:
            started = time.perf_counter()
            try:
                raw_response = binding.chat(prompt)
            except Exception as exc:
                fallback_reason = f"chat_failed:{binding.provider_name}:{exc}"
                continue
            latency_ms = (time.perf_counter() - started) * 1000.0
            parsed = self._parse_llm_response(raw_response)
            if parsed is None:
                fallback_reason = f"invalid_json:{binding.provider_name}"
                continue
            return parsed, binding.provider_name, binding.model_name, latency_ms, raw_response, None
        return None, None, None, None, None, fallback_reason

    @staticmethod
    def _build_prompt(
        *, request: AgentTaskRequest, retrieval_signals: RetrievalSignals
    ) -> str:
        return (
            f"{_TASK_UNDERSTANDING_PROMPT}\n"
            f"User query: {request.user_query}\n"
            f"Task goal: {request.task_goal}\n"
            f"Expected output: {request.expected_output}\n"
            f"Allow web: {str(request.allow_web).lower()}\n"
            f"Retrieval hint context: {retrieval_signals.model_dump_json()}\n"
            "JSON only."
        )

    @classmethod
    def _parse_llm_response(cls, response: str) -> TaskUnderstanding | None:
        candidate = _extract_json_object(response)
        if candidate is None:
            return None
        try:
            payload = json.loads(candidate)
            return TaskUnderstanding.model_validate(payload)
        except Exception:
            return None

    @staticmethod
    def _fallback_understanding(
        *,
        request: AgentTaskRequest,
        retrieval_signals: RetrievalSignals,
    ) -> TaskUnderstanding:
        task_type = "research" if retrieval_signals.allow_graph_expansion else "lookup"
        needs_timeline = "timeline" in request.user_query.lower()
        needs_comparison = "compare" in request.user_query.lower()
        needs_external_evidence = request.allow_web and not request.source_scope and retrieval_signals.allow_graph_expansion
        deliverable_type = request.expected_output or (
            "decision_report"
            if needs_comparison and "recommend" in request.user_query.lower()
            else "analysis_report"
        )
        decomposition_required = retrieval_signals.allow_graph_expansion or request.max_subtasks > 1
        success_criteria = [
            "Cover the main evidence dimensions of the task.",
            "Preserve uncertainty and unsupported areas explicitly.",
        ]
        if needs_comparison:
            success_criteria.append("Compare alternatives with grounded evidence.")
        if needs_timeline:
            success_criteria.append("Present chronology in an evidence-grounded order.")
        return TaskUnderstanding(
            task_type=task_type,
            deliverable_type=deliverable_type,
            decomposition_required=decomposition_required,
            needs_external_evidence=needs_external_evidence,
            needs_comparison=needs_comparison,
            needs_timeline=needs_timeline,
            success_criteria=success_criteria,
        )


def _extract_json_object(response: str) -> str | None:
    stripped = response.strip()
    if not stripped:
        return None
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return stripped[start : end + 1]


__all__ = [
    "TaskUnderstandingDiagnostics",
    "TaskUnderstandingService",
]
