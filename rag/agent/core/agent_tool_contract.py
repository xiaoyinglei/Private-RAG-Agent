from __future__ import annotations

import inspect
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.delegation import (
    AgentAsToolExecutionError,
    AgentDelegationRequest,
    DelegatedAgentResult,
    DelegatedAgentRunner,
    ParentAgentContext,
)
from rag.schema.query import AnswerCitation


class AgentToolInput(BaseModel):
    """Agent-as-tool input containing only stable delegation fields."""

    task: str = Field(min_length=1, description="子任务描述")
    goal: str | None = Field(default=None, description="期望产出")
    context_summary: str | None = Field(default=None, description="父 Agent 传递的上下文摘要")
    required_outputs: list[str] = Field(
        default_factory=list,
        description="需要的产出类型，如 ['evidence', 'conclusion']",
    )
    constraints: list[str] = Field(default_factory=list, description="约束条件，如 'prefer_table', 'max_3_items'")


MAX_DELEGATED_EVIDENCE_REFS = 20
MAX_DELEGATED_CITATIONS = 20
MAX_DELEGATED_KEY_FACTS = 10
MAX_DELEGATED_FACT_CHARS = 200


class DelegatedEvidenceRef(BaseModel):
    """Compact, traceable child evidence reference returned to the parent loop."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(min_length=1)
    citation_id: str | None = None
    citation_anchor: str | None = None
    doc_id: int | None = None
    source: Literal["delegated_agent"] = "delegated_agent"

    @model_validator(mode="after")
    def require_traceable_locator(self) -> Self:
        if not self.citation_id and not self.citation_anchor:
            raise ValueError("delegated evidence reference requires a citation id or anchor")
        return self


class AgentToolOutput(BaseModel):
    """Agent-as-tool 脱水结果。不暴露完整 AgentRunResult，只返回关键信息。"""

    conclusion: str = Field(default="", description="子 Agent 的核心结论")
    key_facts: list[str] = Field(
        default_factory=list,
        max_length=MAX_DELEGATED_KEY_FACTS,
        description="关键事实列表",
    )
    evidence_refs: list[DelegatedEvidenceRef] = Field(
        default_factory=list,
        max_length=MAX_DELEGATED_EVIDENCE_REFS,
        description="可定位的子 Agent 证据引用",
    )
    citations: list[AnswerCitation] = Field(
        default_factory=list,
        max_length=MAX_DELEGATED_CITATIONS,
        description="子 Agent 引用元数据",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度 0-1")
    status: str = Field(default="unknown", description="子 Agent 最终状态")
    agent_name: str = Field(default="", description="执行的 Agent 类型")

    @classmethod
    def from_run_result(cls, result: DelegatedAgentResult, agent_name: str) -> AgentToolOutput:
        """从 AgentRunResult 提取脱水结果。"""
        return cls(
            conclusion=result.final_answer or "",
            key_facts=_extract_key_facts(result),
            evidence_refs=_extract_evidence_refs(result),
            citations=list(result.citations[:MAX_DELEGATED_CITATIONS]),
            confidence=_derive_confidence(result),
            status=result.status,
            agent_name=agent_name,
        )

    @classmethod
    def error_result(cls, agent_name: str, error_message: str, status: str = "failed") -> AgentToolOutput:
        """构造错误结果。"""
        return cls(
            conclusion=error_message,
            status=status,
            agent_name=agent_name,
        )


def _extract_key_facts(result: DelegatedAgentResult) -> list[str]:
    facts: list[str] = []
    for item in result.evidence:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            facts.append(text.strip()[:MAX_DELEGATED_FACT_CHARS])
        if len(facts) >= MAX_DELEGATED_KEY_FACTS:
            break
    return facts


def _extract_evidence_refs(result: DelegatedAgentResult) -> list[DelegatedEvidenceRef]:
    citations = list(result.citations[:MAX_DELEGATED_CITATIONS])
    citations_by_evidence = {citation.evidence_id: citation for citation in citations}
    refs: list[DelegatedEvidenceRef] = []
    evidence_ids: set[str] = set()
    for item in result.evidence:
        if len(refs) >= MAX_DELEGATED_EVIDENCE_REFS:
            break
        citation = citations_by_evidence.get(item.evidence_id)
        citation_anchor = item.citation_anchor.strip() if item.citation_anchor.strip() else None
        if citation is None and citation_anchor is None:
            continue
        refs.append(
            DelegatedEvidenceRef(
                evidence_id=item.evidence_id,
                citation_id=None if citation is None else citation.citation_id,
                citation_anchor=citation_anchor,
                doc_id=item.doc_id,
            )
        )
        evidence_ids.add(item.evidence_id)
    for citation in citations:
        if len(refs) >= MAX_DELEGATED_EVIDENCE_REFS:
            break
        if citation.evidence_id in evidence_ids:
            continue
        refs.append(
            DelegatedEvidenceRef(
                evidence_id=citation.evidence_id,
                citation_id=citation.citation_id,
                citation_anchor=citation.citation_anchor,
                doc_id=citation.doc_id,
            )
        )
    return refs


def _derive_confidence(result: DelegatedAgentResult) -> float:
    if result.status == "done":
        return 0.8 if bool(getattr(result, "groundedness_flag", False)) else 0.5
    return 0.1


class AgentAsToolAdapter:
    """Adapt a delegated agent runner to the normal ToolRunner callable shape."""

    def __init__(
        self,
        runner: DelegatedAgentRunner,
        agent_type: str,
        run_config: AgentRunConfig,
    ) -> None:
        self._runner = runner
        self._agent_type = agent_type
        self._run_config = run_config

    async def __call__(self, payload: BaseModel) -> AgentToolOutput:
        request = AgentToolInput.model_validate(payload)
        parent_state = ParentAgentContext(run_config=self._run_config)

        prompt = _build_delegation_prompt(request)
        delegation = AgentDelegationRequest(
            delegation_id=f"{self._agent_type}-tool-{uuid4().hex[:8]}",
            agent_type=self._agent_type,
            prompt=prompt,
        )

        try:
            raw_result = self._runner.run_delegated_task(
                request=delegation,
                parent_state=parent_state,
            )
            result = await raw_result if inspect.isawaitable(raw_result) else raw_result
        except Exception as exc:
            raise AgentAsToolExecutionError(
                self._agent_type,
                f"subagent execution failed: {exc}",
                status="failed",
                stop_reason=exc.__class__.__name__,
            ) from exc

        if result.status != "done":
            reason = result.stop_reason or result.status
            raise AgentAsToolExecutionError(
                self._agent_type,
                f"subagent returned {result.status}: {reason}",
                status=result.status,
                stop_reason=result.stop_reason,
            )

        return AgentToolOutput.from_run_result(result, self._agent_type)


def _build_delegation_prompt(payload: AgentToolInput) -> str:
    parts = [f"## Task\n{payload.task}"]
    if payload.goal:
        parts.append(f"## Goal\n{payload.goal}")
    if payload.context_summary:
        parts.append(f"## Context\n{payload.context_summary}")
    if payload.required_outputs:
        parts.append(f"## Required Outputs\n{', '.join(payload.required_outputs)}")
    if payload.constraints:
        parts.append("## Constraints\n" + "\n".join(f"- {constraint}" for constraint in payload.constraints))
    return "\n\n".join(parts)


__all__ = [
    "AgentAsToolAdapter",
    "AgentToolInput",
    "AgentToolOutput",
    "DelegatedEvidenceRef",
]
