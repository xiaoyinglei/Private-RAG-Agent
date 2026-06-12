from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.core.agent_as_tool import (
    AgentAsToolAdapter,
    AgentToolInput,
    build_agent_tool_spec,
)
from rag.agent.core.checkpointing import agent_checkpoint_serde
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy, ToolPolicy
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.core.llm_registry import (
    ModelRegistry,
    ResolvedModel,
    UnknownModelAliasError,
)
from rag.agent.core.output_models import ValidatedFinalOutput
from rag.agent.goal_runtime import GoalDeliverable, GoalSpec
from rag.agent.graphs.base import build_agent_graph
from rag.agent.memory.models import MemoryPolicy
from rag.agent.memory.store import WorkspaceMemoryStore
from rag.agent.service import AgentService
from rag.agent.state import AgentState, ToolCallPlan, create_agent_state
from rag.agent.tools.rag_answer_tools import RAGSearchAnswerOutput, rag_search_answer
from rag.agent.tools.rag_tools import SearchOutput, rerank, vector_search
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolResult, ToolSpec
from rag.agent.workspace import WorkspaceRuntime
from rag.schema.query import AnswerCitation, EvidenceItem, RetrievalSignals
from rag.schema.runtime import AccessPolicy
from tests.agent.parity.normalize import normalize_legacy_state

LEGACY_SCENARIO_NAMES = (
    "approval_resume",
    "child_agent",
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


class _ScenarioRunner(Protocol):
    def __call__(self) -> Awaitable[dict[str, object]]: ...


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
        parent_state: AgentState,
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
        definition: AgentDefinition,
        state: AgentState,
        candidate_text: str,
    ) -> _StructuredAnswer:
        del definition, state
        return _StructuredAnswer(answer=candidate_text, confidence=0.91)


class _SchemaGenerator:
    def generate_structured(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        **kwargs: object,
    ) -> BaseModel:
        del prompt, kwargs
        if schema.__name__ == "GoalContractHint":
            return schema.model_validate(
                {
                    "deliverable_kinds": ["answer"],
                    "reason": "Fallback goal contract.",
                }
            )
        if schema.__name__ == "RetrievalHintDecision":
            return schema.model_validate(
                {
                    "reason": "fallback_hint",
                    "retrieval_signals": {
                        "quoted_terms": ["fallback"],
                    },
                }
            )
        return schema.model_validate(
            {
                "action": "pause",
                "thought": "fallback model requests clarification",
                "confidence": 0.5,
                "needs_user_input": "Fallback model needs clarification.",
            }
        )


class _FallbackRegistry(ModelRegistry):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._resolved = ResolvedModel(
            generator=_SchemaGenerator(),
            kwargs={},
        )

    @property
    def default_model(self) -> str:
        return "missing"

    @property
    def fallback_model(self) -> str:
        return "fallback"

    def resolve(self, alias: str) -> ResolvedModel:
        self.calls.append(alias)
        if alias == "missing":
            raise UnknownModelAliasError("missing")
        if alias == "fallback":
            return self._resolved
        raise UnknownModelAliasError(alias)


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
    definition: AgentDefinition | None = None,
    goal_spec: GoalSpec | None = None,
    tool_policy: ToolPolicy | None = None,
    memory_policy: MemoryPolicy | None = None,
    messages: list[HumanMessage] | None = None,
    max_depth: int = 2,
    source_scope: tuple[str, ...] = (),
) -> AgentState:
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
    return create_agent_state(
        task=task,
        run_config=config,
        pending_tool_calls=pending_tool_calls,
        messages=messages,
        goal_spec=goal_spec,
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
        permissions=ToolPermissions(
            write_db=requires_confirmation,
        ),
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
) -> AgentDefinition:
    return AgentDefinition(
        agent_type=name,
        description=f"{name} parity definition",
        system_prompt="Use typed tools and preserve evidence.",
        allowed_tools=allowed_tools,
        output_model=output_model,
        max_iterations=6,
        model_selection=model_selection or ModelSelectionPolicy(),
        tool_policy=ToolPolicy(max_parallel_calls=4),
    )


async def _run_graph(
    *,
    definition: AgentDefinition,
    registry: ToolRegistry,
    state: AgentState,
    checkpointer: MemorySaver | None = None,
    output_finalizer: object | None = None,
    model_registry: ModelRegistry | None = None,
) -> AgentState:
    from rag.agent.core.compiler import GraphCompiler

    if model_registry is None:
        graph = build_agent_graph(
            definition=definition,
            tool_registry=registry,
            output_finalizer=output_finalizer,  # type: ignore[arg-type]
            checkpointer=checkpointer,
        )
    else:
        graph = GraphCompiler(
            tool_registry=registry,
            model_registry=model_registry,
            checkpointer=checkpointer,
        ).compile(definition)
    return await graph.ainvoke(
        state,
        config={"configurable": {"thread_id": state["run_config"].thread_id}},
    )


async def _plain_without_tools() -> dict[str, object]:
    run_id = "parity-plain"
    state = _state(run_id, task="Answer directly without tools.")
    result = await _run_graph(
        definition=_definition("plain", []),
        registry=ToolRegistry(),
        state=state,
    )
    RunRegistry.remove(run_id)
    return normalize_legacy_state(result)


async def _single_tool() -> dict[str, object]:
    run_id = "parity-single"
    registry = ToolRegistry()
    registry.register(
        _text_spec("answer_tool"),
        runner=lambda payload: _TextOutput(text=f"answer:{payload.text}"),
    )
    call = ToolCallPlan(
        tool_call_id="tc-single",
        tool_name="answer_tool",
        arguments={"text": "policy"},
    )
    result = await _run_graph(
        definition=_definition("single", ["answer_tool"]),
        registry=registry,
        state=_state(
            run_id,
            task="Answer with one tool.",
            pending_tool_calls=[call],
        ),
    )
    RunRegistry.remove(run_id)
    return normalize_legacy_state(result)


async def _multiple_tools() -> dict[str, object]:
    run_id = "parity-multiple"
    registry = ToolRegistry()
    registry.register(
        _text_spec("echo_tool", concurrency_safe=True),
        runner=lambda payload: _TextOutput(text=f"echo:{payload.text}"),
    )
    compute_spec = ToolSpec(
        name="calculate_tool",
        description="Calculate a deterministic expression.",
        input_model=_TextInput,
        output_model=_ComputationOutput,
        error_model=ToolError,
        permissions=ToolPermissions(),
        timeout_seconds=1.0,
        idempotent=True,
        concurrency_safe=True,
    )
    registry.register(
        compute_spec,
        runner=lambda payload: _ComputationOutput(
            text="42",
            operation="sum",
            expression=payload.text,
        ),
    )
    calls = [
        ToolCallPlan(
            tool_call_id="tc-echo",
            tool_name="echo_tool",
            arguments={"text": "first"},
        ),
        ToolCallPlan(
            tool_call_id="tc-calculate",
            tool_name="calculate_tool",
            arguments={"text": "40 + 2"},
        ),
    ]
    result = await _run_graph(
        definition=_definition(
            "multiple",
            ["echo_tool", "calculate_tool"],
        ),
        registry=registry,
        state=_state(
            run_id,
            task="Use multiple tools and calculate.",
            pending_tool_calls=calls,
        ),
    )
    RunRegistry.remove(run_id)
    return normalize_legacy_state(result)


async def _rag_grounding() -> dict[str, object]:
    run_id = "parity-rag"
    evidence = EvidenceItem(
        evidence_id="ev-policy",
        doc_id=7,
        citation_anchor="policy#leave",
        text="Employees receive 15 days of annual leave.",
        score=0.94,
        record_type="section",
        retrieval_channels=["vector", "rerank"],
        retrieval_family="hybrid",
    )
    citation = AnswerCitation(
        citation_id="cit-policy",
        evidence_id=evidence.evidence_id,
        record_type="section",
        citation_anchor=evidence.citation_anchor,
        doc_id=evidence.doc_id,
    )
    registry = ToolRegistry()
    registry.register(
        vector_search,
        runner=lambda payload: SearchOutput(
            items=[
                {
                    **evidence.model_dump(mode="json"),
                    "rerank_score": 0.96,
                }
            ]
        ),
    )
    registry.register(
        rerank,
        runner=lambda payload: SearchOutput(
            items=[
                {
                    **evidence.model_dump(mode="json"),
                    "rerank_score": 0.99,
                }
            ]
        ),
    )
    registry.register(
        rag_search_answer,
        runner=lambda payload: RAGSearchAnswerOutput(
            text="Employees receive 15 days of annual leave.",
            evidence=[evidence],
            citations=[citation],
            groundedness_flag=True,
        ),
    )
    calls = [
        ToolCallPlan(
            tool_call_id="tc-vector",
            tool_name="vector_search",
            arguments={"query": "annual leave", "top_k": 4},
        ),
        ToolCallPlan(
            tool_call_id="tc-rerank",
            tool_name="rerank",
            arguments={"query": "annual leave", "top_k": 4},
        ),
        ToolCallPlan(
            tool_call_id="tc-rag-answer",
            tool_name="rag_search_answer",
            arguments={"query": "annual leave", "top_k": 4},
        ),
    ]
    state = _state(
        run_id,
        task='Answer the "annual leave" policy with citations.',
        pending_tool_calls=calls,
        tool_policy=ToolPolicy(max_parallel_calls=2),
    )
    state["retrieval_signals"] = RetrievalSignals(
        quoted_terms=["annual leave"],
        allow_graph_expansion=True,
    )
    result = await _run_graph(
        definition=AgentDefinition(
            agent_type="rag",
            description="RAG parity definition",
            system_prompt="Preserve retrieval metadata.",
            allowed_tools=["vector_search", "rerank", "rag_search_answer"],
            max_iterations=6,
            tool_policy=ToolPolicy(max_parallel_calls=2),
        ),
        registry=registry,
        state=state,
    )
    RunRegistry.remove(run_id)
    return normalize_legacy_state(result)


async def _approval_resume() -> dict[str, object]:
    run_id = "parity-approval"
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        _text_spec("write_tool", requires_confirmation=True),
        runner=lambda payload: (
            calls.append(payload.text)
            or _TextOutput(text=f"wrote:{payload.text}")
        ),
    )
    call = ToolCallPlan(
        tool_call_id="tc-write",
        tool_name="write_tool",
        arguments={"text": "approved"},
    )
    state = _state(
        run_id,
        task="Run an approved mutation.",
        pending_tool_calls=[call],
    )
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    graph = build_agent_graph(
        definition=_definition("approval", ["write_tool"]),
        tool_registry=registry,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": run_id}}
    paused = await graph.ainvoke(state, config=config)
    request = paused["human_input_request"]
    resumed = await graph.ainvoke(
        Command(
            resume=HumanInputResponse(
                request_id=request.request_id,
                decision="allow_once",
                approved_tool_call_ids=[call.tool_call_id],
            ).model_dump(mode="json")
        ),
        config=config,
    )
    RunRegistry.remove(run_id)
    return {
        "paused": normalize_legacy_state(paused),
        "resumed": normalize_legacy_state(
            resumed,
            observed={"runner_calls": calls},
        ),
    }


async def _tool_retry() -> dict[str, object]:
    run_id = "parity-retry"
    attempts: list[str] = []

    def runner(payload: _TextInput) -> _TextOutput:
        attempts.append(payload.text)
        if len(attempts) == 1:
            raise RuntimeError("transient failure")
        return _TextOutput(text=f"recovered:{payload.text}")

    registry = ToolRegistry()
    registry.register(
        _text_spec(
            "retry_tool",
            idempotent=True,
            max_retries=1,
        ),
        runner=runner,
    )
    call = ToolCallPlan(
        tool_call_id="tc-retry",
        tool_name="retry_tool",
        arguments={"text": "retry"},
    )
    result = await _run_graph(
        definition=_definition("retry", ["retry_tool"]),
        registry=registry,
        state=_state(
            run_id,
            task="Recover from a transient tool error.",
            pending_tool_calls=[call],
        ),
    )
    RunRegistry.remove(run_id)
    return normalize_legacy_state(
        result,
        observed={"attempts": attempts},
    )


async def _message_compaction() -> dict[str, object]:
    run_id = "parity-compaction"
    policy = MemoryPolicy(
        message_compaction_min_count=4,
        max_message_tail_count=2,
    )
    with TemporaryDirectory(prefix="agent-parity-memory-") as directory:
        workspace = WorkspaceRuntime(
            root=Path(directory) / "workspace",
            is_temporary=True,
        )
        workspace.initialize()
        store = WorkspaceMemoryStore(workspace=workspace, policy=policy)
        service = AgentService(
            definition=_definition("compaction", []),
            tool_registry=ToolRegistry(),
        )
        state = service.initial_state_from_config(
            task="Compact prior messages.",
            run_config=_config(run_id, memory_policy=policy),
            messages=[
                HumanMessage(
                    content=f"history {index}",
                    id=f"msg-{index}",
                )
                for index in range(6)
            ],
            memory_store=store,
        )
        result = await _run_graph(
            definition=_definition("compaction", []),
            registry=ToolRegistry(),
            state=state,
        )
    RunRegistry.remove(run_id)
    return normalize_legacy_state(result)


async def _structured_output() -> dict[str, object]:
    run_id = "parity-structured"
    registry = ToolRegistry()
    registry.register(
        _text_spec("answer_tool"),
        runner=lambda payload: _TextOutput(text=f"structured:{payload.text}"),
    )
    call = ToolCallPlan(
        tool_call_id="tc-structured",
        tool_name="answer_tool",
        arguments={"text": "policy"},
    )
    result = await _run_graph(
        definition=_definition(
            "structured",
            ["answer_tool"],
            output_model=_StructuredAnswer,
        ),
        registry=registry,
        state=_state(
            run_id,
            task="Return a structured answer.",
            pending_tool_calls=[call],
        ),
        output_finalizer=_StructuredFinalizer(),
    )
    assert isinstance(result.get("final_output"), ValidatedFinalOutput)
    RunRegistry.remove(run_id)
    return normalize_legacy_state(result)


async def _explicit_goal_spec() -> dict[str, object]:
    run_id = "parity-goal"
    registry = ToolRegistry()
    registry.register(
        _text_spec("answer_tool"),
        runner=lambda payload: _TextOutput(text=f"answer:{payload.text}"),
    )
    goal = GoalSpec(
        original_query="Answer with evidence.",
        deliverables=[
            GoalDeliverable(
                deliverable_id="answer",
                kind="answer",
                acceptance_rule="non_empty_answer",
            ),
            GoalDeliverable(
                deliverable_id="evidence",
                kind="evidence",
                acceptance_rule="traceable_evidence",
            ),
        ],
    )
    call = ToolCallPlan(
        tool_call_id="tc-goal-answer",
        tool_name="answer_tool",
        arguments={"text": "without evidence"},
    )
    result = await _run_graph(
        definition=_definition("goal", ["answer_tool"]),
        registry=registry,
        state=_state(
            run_id,
            task="Answer with evidence.",
            pending_tool_calls=[call],
            goal_spec=goal,
        ),
    )
    RunRegistry.remove(run_id)
    return normalize_legacy_state(result)


async def _model_fallback() -> dict[str, object]:
    run_id = "parity-model-fallback"
    registry = _FallbackRegistry()
    result = await _run_graph(
        definition=_definition(
            "model_fallback",
            [],
            model_selection=ModelSelectionPolicy(
                retrieval_hint_model="missing",
                tool_decision_model="missing",
            ),
        ),
        registry=ToolRegistry(),
        state=_state(run_id, task='Use the "fallback" model.'),
        model_registry=registry,
    )
    RunRegistry.remove(run_id)
    return normalize_legacy_state(
        result,
        observed={"model_resolutions": registry.calls},
    )


async def _child_agent() -> dict[str, object]:
    run_id = "parity-child"
    delegated_runner = _CapturingDelegatedRunner()
    registry = ToolRegistry()
    agent_tool = build_agent_tool_spec(RESEARCH_AGENT)
    registry.register(
        agent_tool.tool_spec,
        runner=AgentAsToolAdapter(
            runner=delegated_runner,
            agent_type="research",
            run_config=_config(
                run_id,
                max_depth=2,
                source_scope=("doc-parent",),
            ),
        ),
    )
    call = ToolCallPlan(
        tool_call_id="tc-child",
        tool_name="agent_research",
        arguments=AgentToolInput(
            task="Find the child policy.",
            goal="Return a grounded conclusion.",
            required_outputs=["evidence", "conclusion"],
        ).model_dump(mode="json"),
    )
    result = await _run_graph(
        definition=_definition("child", ["agent_research"]),
        registry=registry,
        state=_state(
            run_id,
            task="Delegate a bounded research task.",
            pending_tool_calls=[call],
            max_depth=2,
            source_scope=("doc-parent",),
        ),
    )
    RunRegistry.remove(run_id)
    return normalize_legacy_state(
        result,
        observed={"delegations": delegated_runner.seen},
    )


_SCENARIOS: dict[str, _ScenarioRunner] = {
    "approval_resume": _approval_resume,
    "child_agent": _child_agent,
    "explicit_goal_spec": _explicit_goal_spec,
    "message_compaction": _message_compaction,
    "model_fallback": _model_fallback,
    "multiple_tools": _multiple_tools,
    "plain_without_tools": _plain_without_tools,
    "rag_grounding": _rag_grounding,
    "single_tool": _single_tool,
    "structured_output": _structured_output,
    "tool_retry": _tool_retry,
}


async def run_legacy_scenarios() -> dict[str, object]:
    return {
        name: await _SCENARIOS[name]()
        for name in LEGACY_SCENARIO_NAMES
    }
