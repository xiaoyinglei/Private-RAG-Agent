from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from rag.agent.builtin.generic import GENERIC_AGENT
from rag.agent.compat.goal_contract import GoalDeliverable, GoalSpec
from rag.agent.core.agent_as_tool import (
    AgentAsToolAdapter,
    AgentToolInput,
    build_agent_tool_spec,
)
from rag.agent.core.checkpointing import (
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
)
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.finalization import FinishCandidateBuilder
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.core.tool_execution import ToolExecutionService
from rag.agent.loop.runtime import (
    AgentLoop,
    LoopEventSink,
    ModelTurnEnvelope,
)
from rag.agent.loop.state import (
    LoopState,
    LoopTransition,
    ModelTurnDraft,
)
from rag.agent.loop.stop_hooks import StopHookRunner, build_stop_hooks
from rag.agent.memory.compactor import LoopContextCompactor
from rag.agent.memory.models import MemoryPolicy
from rag.agent.memory.store import WorkspaceMemoryStore
from rag.agent.state import ToolCallPlan
from rag.agent.tools.rag_answer_tools import (
    RAGSearchAnswerOutput,
    rag_search_answer,
)
from rag.agent.tools.rag_tools import SearchOutput, rerank, vector_search
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.agent.workspace import WorkspaceRuntime
from rag.schema.query import AnswerCitation, EvidenceItem, RetrievalSignals
from tests.agent.parity.fixtures import (
    PARITY_SCENARIO_NAMES,
    _CapturingDelegatedRunner,
    _ComputationOutput,
    _config,
    _definition,
    _state,
    _StructuredAnswer,
    _StructuredFinalizer,
    _text_spec,
    _TextInput,
    _TextOutput,
)
from tests.agent.parity.normalize import normalize_loop_state


class _CandidateProvider:
    def __init__(
        self,
        *,
        direct_answer: str | None = None,
        pause_reason: str | None = None,
        pause_after_feedback: str | None = None,
        fallback: bool = False,
    ) -> None:
        self._direct_answer = direct_answer
        self._pause_reason = pause_reason
        self._pause_after_feedback = pause_after_feedback
        self._fallback = fallback

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft | ModelTurnEnvelope:
        del definition, budget_remaining
        if state["stop_hook_feedback"] and self._pause_after_feedback:
            return ModelTurnDraft(
                action="pause",
                pause_reason=self._pause_after_feedback,
            )
        if self._pause_reason is not None:
            draft = ModelTurnDraft(
                action="pause",
                pause_reason=self._pause_reason,
            )
        else:
            candidate = self._direct_answer
            if candidate is None:
                msg = "PR2: answer_candidates no longer written to LoopState; provide direct_answer"
                raise ValueError(msg)
            draft = ModelTurnDraft(
                action="finish",
                final_answer=candidate,
            )
        if not self._fallback:
            return draft
        return ModelTurnEnvelope(
            draft=draft,
            transitions=(
                LoopTransition(
                    reason="fallback",
                    iteration=state["iteration"],
                    detail={"provider": "fallback"},
                ),
            ),
        )


@dataclass
class _EventRecorder(LoopEventSink):
    reasons: list[str] = field(default_factory=list)

    async def emit(self, transition: LoopTransition) -> None:
        self.reasons.append(transition.reason)


async def _run_loop(
    *,
    definition: AgentRuntimePolicy,
    registry: ToolRegistry,
    state: LoopState,
    provider: _CandidateProvider,
    goal_spec: GoalSpec | None = None,
    output_finalizer: object | None = None,
    memory_store: WorkspaceMemoryStore | None = None,
    event_recorder: _EventRecorder | None = None,
) -> tuple[LoopState, LangGraphCheckpointStore]:
    loop_state = state
    if memory_store is not None:
        from rag.agent.core.context import RunRegistry

        RunRegistry.get(loop_state["run_config"].run_id).memory_store = memory_store
    checkpoint_store = LangGraphCheckpointStore(
        MemorySaver(serde=agent_checkpoint_serde()),
        run_config=loop_state["run_config"],
    )
    hooks = build_stop_hooks(
        definition=definition,
        output_finalizer=output_finalizer,  # type: ignore[arg-type]
        goal_spec=goal_spec,
    )
    loop = AgentLoop(
        definition=definition,
        model_provider=provider,
        context_manager=LoopContextCompactor(store=memory_store),
        tool_runner=ToolExecutionService(
            tool_registry=registry,
            record_writer=checkpoint_store,
        ),
        checkpoint_store=checkpoint_store,
        stop_hook_runner=StopHookRunner(
            hooks=hooks,
            max_blocks=definition.max_stop_hook_blocks,
        ),
        finish_candidate_builder=FinishCandidateBuilder(),
        event_sink=event_recorder,
    )
    return await loop.run(loop_state), checkpoint_store


async def _plain_without_tools() -> dict[str, object]:
    run_id = "loop-parity-plain"
    result, _ = await _run_loop(
        definition=_definition("plain", []),
        registry=ToolRegistry(),
        state=_state(
            run_id,
            task="Answer directly without tools.",
        ),
        provider=_CandidateProvider(direct_answer="Direct answer."),
    )
    return normalize_loop_state(result)


async def _single_tool() -> dict[str, object]:
    run_id = "loop-parity-single"
    registry = ToolRegistry()
    registry.register(
        _text_spec("answer_tool"),
        runner=lambda payload: _TextOutput(text=f"answer:{cast(_TextInput, payload).text}"),
    )
    call = ToolCallPlan(
        tool_call_id="tc-single",
        tool_name="answer_tool",
        arguments={"text": "policy"},
    )
    result, _ = await _run_loop(
        definition=_definition("single", ["answer_tool"]),
        registry=registry,
        state=_state(
            run_id,
            task="Answer with one tool.",
            pending_tool_calls=[call],
        ),
        provider=_CandidateProvider(direct_answer="answer:policy"),
    )
    return normalize_loop_state(result)


async def _multiple_tools() -> dict[str, object]:
    run_id = "loop-parity-multiple"
    registry = ToolRegistry()
    registry.register(
        _text_spec("echo_tool", concurrency_safe=True),
        runner=lambda payload: _TextOutput(text=f"echo:{cast(_TextInput, payload).text}"),
    )
    registry.register(
        ToolSpec(
            name="calculate_tool",
            description="Calculate a deterministic expression.",
            input_model=_TextInput,
            output_model=_ComputationOutput,
            error_model=ToolError,
            permissions=ToolPermissions(),
            timeout_seconds=1.0,
            idempotent=True,
            concurrency_safe=True,
        ),
        runner=lambda payload: _ComputationOutput(
            text="42",
            operation="sum",
            expression=cast(_TextInput, payload).text,
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
    result, _ = await _run_loop(
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
        provider=_CandidateProvider(direct_answer="42"),
    )
    return normalize_loop_state(result)


async def _rag_grounding() -> dict[str, object]:
    run_id = "loop-parity-rag"
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
    )
    state["retrieval_signals"] = RetrievalSignals(
        quoted_terms=["annual leave"],
        allow_graph_expansion=True,
    )
    result, _ = await _run_loop(
        definition=_definition(
            "rag",
            ["vector_search", "rerank", "rag_search_answer"],
        ),
        registry=registry,
        state=state,
        provider=_CandidateProvider(direct_answer="Employees receive 15 days of annual leave."),
    )
    return normalize_loop_state(result)


async def _approval_resume() -> dict[str, object]:
    run_id = "loop-parity-approval"
    calls: list[str] = []
    registry = ToolRegistry()

    def write_runner(payload: BaseModel) -> _TextOutput:
        text = cast(_TextInput, payload).text
        calls.append(text)
        return _TextOutput(text=f"wrote:{text}")

    registry.register(
        _text_spec("write_tool", requires_confirmation=True),
        runner=write_runner,
    )
    call = ToolCallPlan(
        tool_call_id="tc-write",
        tool_name="write_tool",
        arguments={"text": "approved"},
    )
    result, store = await _run_loop(
        definition=_definition("approval", ["write_tool"]),
        registry=registry,
        state=_state(
            run_id,
            task="Run an approved mutation.",
            pending_tool_calls=[call],
        ),
        provider=_CandidateProvider(),
    )
    paused = normalize_loop_state(result)
    request = result["approval_request"]
    assert request is not None
    resumed_state = await store.apply_human_response(
        HumanInputResponse(
            request_id=request.request_id,
            decision="allow_once",
            approved_tool_call_ids=[call.tool_call_id],
        )
    )
    resumed, _ = await _run_loop(
        definition=_definition("approval", ["write_tool"]),
        registry=registry,
        state=resumed_state,
        provider=_CandidateProvider(direct_answer="wrote:approved"),
    )
    return {
        "paused": paused,
        "resumed": normalize_loop_state(
            resumed,
            observed={"runner_calls": calls},
        ),
    }


async def _tool_retry() -> dict[str, object]:
    run_id = "loop-parity-retry"
    attempts: list[str] = []

    def runner(payload: object) -> _TextOutput:
        text = payload.text  # type: ignore[attr-defined]
        attempts.append(text)
        if len(attempts) == 1:
            raise RuntimeError("transient failure")
        return _TextOutput(text=f"recovered:{text}")

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
    result, _ = await _run_loop(
        definition=_definition("retry", ["retry_tool"]),
        registry=registry,
        state=_state(
            run_id,
            task="Recover from a transient tool error.",
            pending_tool_calls=[call],
        ),
        provider=_CandidateProvider(direct_answer="recovered:retry"),
    )
    return normalize_loop_state(
        result,
        observed={"attempts": attempts},
    )


async def _message_compaction() -> dict[str, object]:
    run_id = "loop-parity-compaction"
    policy = MemoryPolicy(
        message_compaction_min_count=4,
        max_message_tail_count=2,
    )
    with TemporaryDirectory(prefix="loop-agent-parity-memory-") as directory:
        workspace = WorkspaceRuntime(
            root=Path(directory) / "workspace",
            is_temporary=True,
        )
        workspace.initialize()
        store = WorkspaceMemoryStore(workspace=workspace, policy=policy)
        result, _ = await _run_loop(
            definition=_definition("compaction", []),
            registry=ToolRegistry(),
            state=_state(
                run_id,
                task="Compact prior messages.",
                memory_policy=policy,
                messages=[
                    HumanMessage(
                        content=f"history {index}",
                        id=f"msg-{index}",
                    )
                    for index in range(6)
                ],
            ),
            provider=_CandidateProvider(pause_reason="No model answer is configured."),
            memory_store=store,
        )
        return normalize_loop_state(result)


async def _structured_output() -> dict[str, object]:
    run_id = "loop-parity-structured"
    registry = ToolRegistry()
    registry.register(
        _text_spec("answer_tool"),
        runner=lambda payload: _TextOutput(text=f"structured:{cast(_TextInput, payload).text}"),
    )
    call = ToolCallPlan(
        tool_call_id="tc-structured",
        tool_name="answer_tool",
        arguments={"text": "policy"},
    )
    result, _ = await _run_loop(
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
        provider=_CandidateProvider(direct_answer="structured:policy"),
        output_finalizer=_StructuredFinalizer(),
    )
    return normalize_loop_state(result)


async def _explicit_goal_spec() -> dict[str, object]:
    run_id = "loop-parity-goal"
    registry = ToolRegistry()
    registry.register(
        _text_spec("answer_tool"),
        runner=lambda payload: _TextOutput(text=f"answer:{cast(_TextInput, payload).text}"),
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
    result, _ = await _run_loop(
        definition=_definition("goal", ["answer_tool"]),
        registry=registry,
        state=_state(
            run_id,
            task="Answer with evidence.",
            pending_tool_calls=[call],
        ),
        provider=_CandidateProvider(
            direct_answer="answer:without evidence",
            pause_after_feedback=("Explicit goal contract still requires traceable evidence."),
        ),
        goal_spec=goal,
    )
    return normalize_loop_state(result)


async def _model_fallback() -> dict[str, object]:
    run_id = "loop-parity-model-fallback"
    events = _EventRecorder()
    result, _ = await _run_loop(
        definition=_definition("model_fallback", []),
        registry=ToolRegistry(),
        state=_state(
            run_id,
            task='Use the "fallback" model.',
        ),
        provider=_CandidateProvider(
            pause_reason="Fallback model needs clarification.",
            fallback=True,
        ),
        event_recorder=events,
    )
    return normalize_loop_state(
        result,
        observed={
            "model_resolutions": ["missing"],
            "transition_reasons": events.reasons,
        },
    )


async def _child_agent() -> dict[str, object]:
    run_id = "loop-parity-child"
    delegated_runner = _CapturingDelegatedRunner()
    registry = ToolRegistry()
    agent_tool = build_agent_tool_spec(GENERIC_AGENT)
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
    result, _ = await _run_loop(
        definition=_definition("child", ["agent_research"]),
        registry=registry,
        state=_state(
            run_id,
            task="Delegate a bounded research task.",
            pending_tool_calls=[call],
            max_depth=2,
            source_scope=("doc-parent",),
        ),
        provider=_CandidateProvider(),
    )
    return normalize_loop_state(
        result,
        observed={"delegations": delegated_runner.seen},
    )


_SCENARIOS = {
    "approval_resume": _approval_resume,
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


async def run_loop_scenarios() -> dict[str, object]:
    return {name: await _SCENARIOS[name]() for name in PARITY_SCENARIO_NAMES}
