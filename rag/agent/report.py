"""Stable report rendering utilities for UI and benchmark consumers."""

from __future__ import annotations

from rag.agent.schema import (
    AgentFinalReport,
    EvidenceMapEntry,
    ExecutionSummary,
    ReportCitation,
    SubTaskResult,
    SubTaskStatus,
)
from rag.agent.state import AgentRunState


class AgentReportBuilder:
    """Build the fixed UI-facing report envelope from agent state."""

    def build(
        self,
        *,
        state: AgentRunState,
        executive_summary: str,
        key_findings: list[str],
        risks: list[str],
        unknowns: list[str],
        recommendations: list[str],
    ) -> AgentFinalReport:
        citations = self._citations(state.subtask_results)
        evidence_map = self._evidence_map(state.subtask_results, citations)
        return AgentFinalReport(
            executive_summary=executive_summary,
            key_findings=key_findings,
            evidence_map=evidence_map,
            risks=risks,
            unknowns=unknowns,
            recommendations=recommendations,
            citations=citations,
            execution_summary=self._execution_summary(state),
        )

    @staticmethod
    def _citations(results: list[SubTaskResult]) -> list[ReportCitation]:
        citations: list[ReportCitation] = []
        seen: set[str] = set()
        for result in results:
            for item in result.evidence:
                evidence_id = item.evidence_id
                if evidence_id in seen:
                    continue
                seen.add(evidence_id)
                citations.append(
                    ReportCitation(
                        citation_id=f"cit-{len(citations) + 1}",
                        subtask_id=result.subtask.subtask_id,
                        chunk_id=evidence_id,
                        evidence_id=evidence_id,
                        record_type=item.record_type or "unknown",
                        doc_id=item.doc_id,
                        file_name=item.file_name,
                        citation_anchor=item.citation_anchor,
                        section_path=list(item.section_path),
                        page_start=item.page_start,
                        page_end=item.page_end,
                        evidence_kind=item.evidence_kind,
                        benchmark_doc_id=item.benchmark_doc_id,
                        source_id=item.source_id,
                        source_type=item.source_type,
                    )
                )
        return citations

    @staticmethod
    def _evidence_map(results: list[SubTaskResult], citations: list[ReportCitation]) -> list[EvidenceMapEntry]:
        citation_index = {citation.evidence_id: citation.citation_id for citation in citations}
        entries: list[EvidenceMapEntry] = []
        for result in results:
            citation_ids = [
                citation_index[item.evidence_id]
                for item in result.evidence[:3]
                if item.evidence_id in citation_index
            ]
            for finding in result.findings[:2]:
                entries.append(
                    EvidenceMapEntry(
                        finding=finding,
                        subtask_id=result.subtask.subtask_id,
                        citation_ids=citation_ids,
                        confidence=result.evidence_assessment.confidence,
                    )
                )
        return entries

    @staticmethod
    def _execution_summary(state: AgentRunState) -> ExecutionSummary:
        return ExecutionSummary(
            subtasks_count=len(state.subtasks),
            completed_subtasks=sum(1 for item in state.subtask_results if item.status is SubTaskStatus.COMPLETED),
            retries_count=sum(max(0, len(item.traces) - 1) for item in state.subtask_results),
            web_used=state.web_used,
            unresolved_items_count=sum(len(item.unresolved_questions) for item in state.subtask_results),
        )


__all__ = ["AgentReportBuilder"]
