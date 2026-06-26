from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentRuntimePolicy, ModelSelectionPolicy, ToolPolicy
from rag.agent.core.delegation import ParentAgentContext
from rag.agent.loop.state import LoopState, create_loop_state
from rag.agent.memory.models import MemoryPolicy
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolResult, ToolSpec
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy

PARITY_SCENARIO_NAMES = (
    "approval_resume",
    "explicit_goal_spec",
    "message_compaction",
    "model_fallback",
    "multiple_tools",
    "plain_without_tools",
    "rag_grounding",
    "single_tool",
    "structured_output",
    "tool_retry",
)


class _TextInput(BaseModel):
    text: str


class _TextOutput(BaseModel):
    text: str


class _ComputationOutput(BaseModel):
    text: str
    operation: str
    expression: str


class _StructuredAnswer(BaseModel):
    answer: str
    confidence: float


@dataclass
class _DelegatedResult:
    status: str = "done"
    final_answer: str | None = "Child grounded conclusion."
    stop_reason: str | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    citations: list[AnswerCitation] = field(default_factory=list)
    groundedness_flag: bool = True


class _CapturingDelegatedRunner:
    def __init__(self) -> None:
        self.seen: list[dict[str, object]] = []

    def run_delegated_task(
        self,
        *,
        request: object,
        parent_state: ParentAgentContext,
    ) -> _DelegatedResult:
        run_config = parent_state["run_config"]
        self.seen.append(
            {
                "agent_type": request.agent_type,
                "estimated_tokens": request.estimated_tokens,
                "parent_budget_total": run_config.budget_total,
                "parent_max_depth": run_config.max_depth,
                "source_scope": list(run_config.source_scope),
            }
        )
        evidence = EvidenceItem(
            evidence_id="ev-child",
            doc_id=44,
            citation_anchor="child-policy#1",
            text="Child policy evidence.",
            score=0.88,
            retrieval_channels=["delegated"],
        )
        citation = AnswerCitation(
            citation_id="cit-child",
            evidence_id=evidence.evidence_id,
            record_type="section",
            citation_anchor=evidence.citation_anchor,
            doc_id=evidence.doc_id,
        )
        return _DelegatedResult(evidence=[evidence], citations=[citation])


class _StructuredFinalizer:
    def finalize(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: LoopState,
        candidate_text: str,
    ) -> _StructuredAnswer:
        del definition, state
        return _StructuredAnswer(answer=candidate_text, confidence=0.91)


def _config(
    run_id: str,
    *,
    tool_policy: ToolPolicy | None = None,
    memory_policy: MemoryPolicy | None = None,
    max_depth: int = 2,
    source_scope: tuple[str, ...] = (),
) -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=20_000,
        work_budget_total=20_000,
        max_depth=max_depth,
        source_scope=source_scope,
        access_policy=AccessPolicy.default(),
        tool_policy=tool_policy or ToolPolicy(),
        memory_policy=memory_policy or MemoryPolicy(),
    )


def _state(
    run_id: str,
    *,
    task: str,
    pending_tool_calls: list[ToolCallPlan] | None = None,
    definition: AgentRuntimePolicy | None = None,
    tool_policy: ToolPolicy | None = None,
    memory_policy: MemoryPolicy | None = None,
    messages: list[HumanMessage] | None = None,
    max_depth: int = 2,
    source_scope: tuple[str, ...] = (),
) -> LoopState:
    del definition
    config = _config(
        run_id,
        tool_policy=tool_policy,
        memory_policy=memory_policy,
        max_depth=max_depth,
        source_scope=source_scope,
    )
    RunRegistry.remove(run_id)
    RunRegistry.get_or_create(config)
    return create_loop_state(
        task=task,
        run_config=config,
        pending_tool_calls=pending_tool_calls or (),
        messages=messages or (),
    )


def _text_spec(
    name: str,
    *,
    idempotent: bool = False,
    max_retries: int = 0,
    concurrency_safe: bool = False,
    requires_confirmation: bool = False,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} parity tool",
        input_model=_TextInput,
        output_model=_TextOutput,
        error_model=ToolError,
        permissions=ToolPermissions(write_db=requires_confirmation),
        timeout_seconds=1.0,
        max_retries=max_retries,
        idempotent=idempotent,
        concurrency_safe=concurrency_safe,
        requires_confirmation=requires_confirmation,
    )


def _definition(
    name: str,
    allowed_tools: list[str],
    *,
    output_model: type[BaseModel] | None = None,
    model_selection: ModelSelectionPolicy | None = None,
) -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        agent_type=name,
        description=f"{name} parity definition",
        system_prompt="Use typed tools and preserve evidence.",
        allowed_tools=allowed_tools,
        output_model=output_model,
        max_iterations=6,
        model_selection=model_selection or ModelSelectionPolicy(),
        tool_policy=ToolPolicy(max_parallel_calls=4),
    )
