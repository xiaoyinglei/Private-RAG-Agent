"""Subtask execution loop that delegates evidence retrieval to RetrievalService."""

from __future__ import annotations

from rag.agent.critic import EvidenceCritic
from rag.agent.schema import (
    AgentTaskRequest,
    CriticAction,
    ExecutionStepTrace,
    SubTask,
    SubTaskResult,
    SubTaskStatus,
)
from rag.retrieval.models import RetrievalResult
from rag.schema.runtime import AccessPolicy


class AgentExecutor:
    """Execute a single subtask through retrieval, criticism, and bounded retries."""

    def __init__(
        self,
        *,
        retrieval_service: object,
        critic: EvidenceCritic | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._critic = critic or EvidenceCritic()

    def execute_subtask(
        self,
        *,
        request: AgentTaskRequest,
        subtask: SubTask,
        access_policy: AccessPolicy,
    ) -> SubTaskResult:
        traces: list[ExecutionStepTrace] = []
        last_retrieval: RetrievalResult | None = None
        latest_missing_dimensions: list[str] = []

        for attempt_index in range(1, request.retry_budget + 2):
            retrieval_query = self.prepare_subtask_query(
                request=request,
                subtask=subtask,
                attempt_index=attempt_index,
                missing_dimensions=latest_missing_dimensions,
            )
            selected_mode = self._select_mode(subtask=subtask, attempt_index=attempt_index)
            retrieval = self.execute_retrieval_attempt(
                retrieval_query=retrieval_query,
                selected_mode=selected_mode,
                request=request,
                access_policy=access_policy,
            )
            last_retrieval = retrieval
            assessment = self._critic.assess(
                subtask=subtask,
                retrieval=retrieval,
                attempt_index=attempt_index,
                retry_budget_remaining=max(0, request.retry_budget - attempt_index + 1),
                allow_web=request.allow_web,
            )
            latest_missing_dimensions = assessment.missing_dimensions
            trace = self.collect_trace(
                subtask=subtask,
                retrieval_query=retrieval_query,
                selected_mode=selected_mode,
                retrieval=retrieval,
                assessment=assessment,
                attempt_index=attempt_index,
            )
            traces.append(trace)
            if assessment.recommended_action is CriticAction.ACCEPT:
                return self.finalize_subtask_result(
                    subtask=subtask,
                    retrieval=retrieval,
                    traces=traces,
                    status=SubTaskStatus.COMPLETED,
                    missing_dimensions=assessment.missing_dimensions,
                )
            if attempt_index > request.retry_budget:
                break
            if assessment.recommended_action is CriticAction.ABSTAIN:
                return self.finalize_subtask_result(
                    subtask=subtask,
                    retrieval=retrieval,
                    traces=traces,
                    status=SubTaskStatus.ABSTAINED,
                    missing_dimensions=assessment.missing_dimensions or [subtask.objective],
                )
            self.apply_retry_strategy(assessment.recommended_action)

        assert last_retrieval is not None
        return self.finalize_subtask_result(
            subtask=subtask,
            retrieval=last_retrieval,
            traces=traces,
            status=SubTaskStatus.RETRY_EXHAUSTED,
            missing_dimensions=latest_missing_dimensions or [subtask.objective],
        )

    @staticmethod
    def prepare_subtask_query(
        *,
        request: AgentTaskRequest,
        subtask: SubTask,
        attempt_index: int,
        missing_dimensions: list[str],
    ) -> str:
        parts = [subtask.instruction.strip(), f"Task goal: {request.task_goal}"]
        if subtask.retrieval_hint:
            parts.append(f"Retrieval hint: {subtask.retrieval_hint}")
        if attempt_index > 1 and missing_dimensions:
            parts.append("Focus on the missing evidence dimensions: " + ", ".join(missing_dimensions))
        return "\n".join(part for part in parts if part)

    def execute_retrieval_attempt(
        self,
        *,
        retrieval_query: str,
        selected_mode: str,
        request: AgentTaskRequest,
        access_policy: AccessPolicy,
    ) -> RetrievalResult:
        retrieve = self._retrieval_service.retrieve
        return retrieve(
            retrieval_query,
            access_policy=access_policy,
            source_scope=request.source_scope,
            query_mode=selected_mode,
        )

    @staticmethod
    def collect_trace(
        *,
        subtask: SubTask,
        retrieval_query: str,
        selected_mode: str,
        retrieval: RetrievalResult,
        assessment,
        attempt_index: int,
    ) -> ExecutionStepTrace:
        notes = []
        if assessment.missing_dimensions:
            notes.append("Missing: " + ", ".join(assessment.missing_dimensions))
        if assessment.conflicts:
            notes.extend(assessment.conflicts)
        return ExecutionStepTrace(
            subtask_id=subtask.subtask_id,
            attempt_index=attempt_index,
            retrieval_query=retrieval_query,
            selected_mode=selected_mode,
            branch_hits=dict(retrieval.diagnostics.branch_hits),
            evidence_count=len(retrieval.evidence.all),
            evidence_sufficient=assessment.sufficient,
            action_taken=assessment.recommended_action,
            notes=notes,
        )

    @staticmethod
    def apply_retry_strategy(action: CriticAction) -> None:
        del action

    @staticmethod
    def finalize_subtask_result(
        *,
        subtask: SubTask,
        retrieval: RetrievalResult,
        traces: list[ExecutionStepTrace],
        status: SubTaskStatus,
        missing_dimensions: list[str],
    ) -> SubTaskResult:
        findings = []
        evidence_summary = []
        for item in retrieval.evidence.all[:3]:
            findings.append(item.text.strip())
            evidence_summary.append(f"{item.doc_id} | {item.citation_anchor}")
        assessment = traces[-1]
        unresolved_questions = [f"Still missing evidence for: {dimension}" for dimension in missing_dimensions]
        return SubTaskResult(
            subtask=subtask,
            status=status,
            findings=findings,
            evidence_summary=evidence_summary,
            evidence=list(retrieval.evidence.all),
            evidence_assessment=selfless_assessment_to_model(
                trace=assessment,
                retrieval=retrieval,
                missing_dimensions=missing_dimensions,
            ),
            traces=traces,
            unresolved_questions=unresolved_questions,
        )

    @staticmethod
    def _select_mode(*, subtask: SubTask, attempt_index: int) -> str:
        hint = f"{subtask.retrieval_hint} {subtask.objective}".lower()
        if any(token in hint for token in ("timeline", "metadata", "table", "figure", "section")):
            return "mix"
        if attempt_index > 1 and "compare" in hint:
            return "mix"
        return "naive"


def selfless_assessment_to_model(
    *,
    trace: ExecutionStepTrace,
    retrieval: RetrievalResult,
    missing_dimensions: list[str],
):
    from rag.agent.schema import EvidenceAssessment

    conflicts = [
        note for note in trace.notes if note.startswith("Conflicting evidence")
    ]
    return EvidenceAssessment(
        sufficient=trace.evidence_sufficient,
        confidence=min(1.0, 0.2 + (len(retrieval.evidence.all) * 0.15)),
        missing_dimensions=missing_dimensions,
        conflicts=conflicts,
        recommended_action=trace.action_taken,
    )


__all__ = ["AgentExecutor"]
