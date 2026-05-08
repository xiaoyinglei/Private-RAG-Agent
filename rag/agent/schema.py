"""Structured contracts for the agent orchestration layer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag.schema.query import AnswerCitation, EvidenceItem, MetadataFilters, PolicyHints, TaskType


class CriticAction(StrEnum):
    ACCEPT = "accept"
    RETRY_SAME_SCOPE = "retry_same_scope"
    RETRY_REWRITE_QUERY = "retry_rewrite_query"
    EXPAND_SCOPE = "expand_scope"
    ENABLE_WEB = "enable_web"
    ABSTAIN = "abstain"


class SubTaskStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    FAILED = "failed"
    RETRY_EXHAUSTED = "retry_exhausted"


class AgentTaskRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_query: str
    task_goal: str = "Produce an evidence-grounded analysis report."
    source_scope: list[str] = Field(default_factory=list)
    allow_web: bool = False
    expected_output: str = "structured_analysis_report"
    response_style: str = "formal"
    max_subtasks: int = Field(default=5, ge=1, le=8)
    retry_budget: int = Field(default=2, ge=0, le=6)


class TaskUnderstanding(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_type: TaskType = TaskType.RESEARCH
    deliverable_type: str = "analysis_report"
    decomposition_required: bool = True
    needs_external_evidence: bool = False
    needs_comparison: bool = False
    needs_timeline: bool = False
    success_criteria: list[str] = Field(default_factory=list)


class RetrievalIntent(BaseModel):
    model_config = ConfigDict(frozen=True)

    needs_special: bool = False
    needs_structure: bool = False
    needs_metadata: bool = False
    needs_graph_expansion: bool = False
    preferred_sections: list[str] = Field(default_factory=list)
    metadata_filters: MetadataFilters = Field(default_factory=MetadataFilters)
    policy_hints: PolicyHints = Field(default_factory=PolicyHints)


class SubTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    subtask_id: str
    objective: str
    instruction: str
    expected_evidence: list[str] = Field(default_factory=list)
    retrieval_hint: str = ""
    allow_web: bool = False
    stop_condition: str = ""
    priority: int = Field(default=1, ge=1, le=10)


class ExecutionStepTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    subtask_id: str
    attempt_index: int = Field(ge=1)
    retrieval_query: str
    selected_mode: str
    branch_hits: dict[str, int] = Field(default_factory=dict)
    evidence_count: int = Field(default=0, ge=0)
    evidence_sufficient: bool = False
    action_taken: CriticAction
    notes: list[str] = Field(default_factory=list)


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    sufficient: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_dimensions: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    recommended_action: CriticAction = CriticAction.ABSTAIN


class SubTaskResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    subtask: SubTask
    status: SubTaskStatus = SubTaskStatus.PENDING
    findings: list[str] = Field(default_factory=list)
    evidence_summary: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    evidence_assessment: EvidenceAssessment
    traces: list[ExecutionStepTrace] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class ReportCitation(AnswerCitation):
    model_config = ConfigDict(frozen=True)

    subtask_id: str
    chunk_id: str
    evidence_kind: str = "internal"


class EvidenceMapEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    finding: str
    subtask_id: str
    citation_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExecutionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    subtasks_count: int = Field(default=0, ge=0)
    completed_subtasks: int = Field(default=0, ge=0)
    retries_count: int = Field(default=0, ge=0)
    web_used: bool = False
    unresolved_items_count: int = Field(default=0, ge=0)


class AgentFinalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)
    evidence_map: list[EvidenceMapEntry] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    citations: list[ReportCitation] = Field(default_factory=list)
    execution_summary: ExecutionSummary = Field(default_factory=ExecutionSummary)


class AgentRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class AgentTraceEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    steps: list[ExecutionStepTrace] = Field(default_factory=list)


__all__ = [
    "AgentFinalReport",
    "AgentRunStatus",
    "AgentTaskRequest",
    "AgentTraceEnvelope",
    "CriticAction",
    "EvidenceAssessment",
    "EvidenceMapEntry",
    "ExecutionStepTrace",
    "ExecutionSummary",
    "ReportCitation",
    "RetrievalIntent",
    "SubTask",
    "SubTaskResult",
    "SubTaskStatus",
    "TaskUnderstanding",
]
