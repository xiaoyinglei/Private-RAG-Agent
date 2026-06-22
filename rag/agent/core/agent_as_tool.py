from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.agent.core.context import AgentRunConfig, derive_child_config
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.delegation import (
    DEFAULT_DELEGATION_TOKEN_BUDGET,
    AgentAsToolExecutionError,
    AgentDelegationRequest,
    DelegatedAgentResult,
    DelegatedAgentRunner,
    ParentAgentContext,
)
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.runtime_ports import RetrievalHintProvider
from rag.agent.loop.runtime import ModelTurnProvider
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ExecutionCategory, ToolError, ToolPermissions, ToolSpec
from rag.schema.query import AnswerCitation

if TYPE_CHECKING:
    from rag.agent.service import AgentRunResult

# ── AgentToolSpec (existing) ────────────────────────────────────

@dataclass(frozen=True)
class AgentToolSpec:
    tool_spec: ToolSpec
    agent_definition: AgentDefinition
    inherits_context: bool = True


# ── Tool I/O schemas ───────────────────────────────────────────

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

# ── AgentAsToolAdapter ─────────────────────────────────────────

class AgentAsToolAdapter:
    """将 AgentAsToolRunner 适配为 ToolRunner = Callable[[BaseModel], BaseModel]。

    Request-scoped：每个 run_config 创建一个新实例，绑定到 runtime_tool_registry。
    禁止跨请求复用 adapter，避免 run_config 并发污染。
    """

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
        parent_state = ParentAgentContext(
            run_config=self._run_config,
        )

        prompt = _build_delegation_prompt(request)
        delegation = AgentDelegationRequest(
            delegation_id=f"{self._agent_type}-tool-{uuid4().hex[:8]}",
            agent_type=self._agent_type,
            prompt=prompt,
            estimated_tokens=DEFAULT_DELEGATION_TOKEN_BUDGET,
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
        parts.append("## Constraints\n" + "\n".join(f"- {c}" for c in payload.constraints))
    return "\n\n".join(parts)

# ── build_agent_tool_spec ──────────────────────────────────────

# Agent-as-tool 白名单：只有这些 agent_type 可以被包装为 tool
_AGENT_TOOL_WHITELIST: frozenset[str] = frozenset({
    "research",
    "compare",
    "factcheck",
    "synthesize",
})

# 禁止注册为 tool 的 agent_type（防止递归等安全问题）
_AGENT_TOOL_BLOCKLIST: frozenset[str] = frozenset({
    "orchestrator",
})


def build_agent_tool_spec(agent_definition: AgentDefinition) -> AgentToolSpec:
    """将 AgentDefinition 包装为 AgentToolSpec（含 ToolSpec）。

    白名单检查：只有 research/compare/factcheck/synthesize 可以注册。
    黑名单检查：orchestrator 禁止注册，避免递归调用。
    """
    agent_type = agent_definition.agent_type

    if agent_type in _AGENT_TOOL_BLOCKLIST:
        raise ValueError(
            f"Agent type {agent_type!r} is blocklisted from agent-as-tool registration."
        )

    if agent_type not in _AGENT_TOOL_WHITELIST:
        raise ValueError(
            f"Agent type {agent_type!r} is not in the agent-as-tool whitelist. "
            f"Allowed: {', '.join(sorted(_AGENT_TOOL_WHITELIST))}"
        )

    tool_name = f"agent_{agent_type}"
    tool_description = _make_tool_description(agent_definition)

    tool_spec = ToolSpec(
        name=tool_name,
        description=tool_description,
        input_model=AgentToolInput,
        output_model=AgentToolOutput,
        error_model=ToolError,
        permissions=ToolPermissions(
            read_db=True,
            embed=True,
            generate=True,
        ),
        execution_category=ExecutionCategory.READ,
        timeout_seconds=120.0,
        max_retries=0,
        work_budget_cost=2_000,
    )

    return AgentToolSpec(
        tool_spec=tool_spec,
        agent_definition=agent_definition,
        inherits_context=True,
    )


def _make_tool_description(definition: AgentDefinition) -> str:
    return (
        f"Delegate bounded work to the {definition.agent_type} agent. "
        f"{definition.description} "
        f"Use this when the task requires specialized {definition.agent_type} capabilities."
    )


class AgentAsToolRunner:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        agent_registry: AgentRegistry,
        model_turn_provider: ModelTurnProvider | None = None,
        retrieval_hint_provider: RetrievalHintProvider | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._agent_registry = agent_registry
        self._model_turn_provider = model_turn_provider
        self._retrieval_hint_provider = retrieval_hint_provider

    async def run_delegated_task(
        self,
        *,
        request: AgentDelegationRequest,
        parent_state: ParentAgentContext,
    ) -> AgentRunResult:
        from rag.agent.service import AgentService

        parent_config = parent_state["run_config"]
        child_definition = self._agent_registry.get(request.agent_type)
        child_config = derive_child_config(parent_config, child_definition)
        if request.estimated_tokens is not None:
            child_config = replace(child_config, budget_total=request.estimated_tokens)

        service = AgentService(
            definition=child_definition,
            tool_registry=self._tool_registry,
            model_turn_provider=self._model_turn_provider,
            retrieval_hint_provider=self._retrieval_hint_provider,
            subagent_runner=self,
        )
        return await service.run_with_config(
            task=request.prompt,
            run_config=child_config,
        )
