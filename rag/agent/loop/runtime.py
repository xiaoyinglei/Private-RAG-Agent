from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Sequence
from inspect import isawaitable
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from rag.agent.capabilities.catalog import CORE_TOOLS, DeferredToolStore, ToolCatalog
from rag.agent.capabilities.context import iteration_var
from rag.agent.core.checkpointing import (
    CheckpointStore,
    _migrate_discovery_candidates,
    _migrate_discovery_events,
)
from rag.agent.core.context import RunRegistry
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.tools.spec import ToolResult
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.finalization import (
    FinishCandidateBuilder,
    FinishCandidateBuildError,
)
from rag.agent.core.human_input import HumanInputRequest
from rag.agent.core.llm_context import AgentLLMContextOverflowError
from rag.agent.core.observations import ObservationBatch, ObservationExtractor
from rag.agent.core.output_models import ValidatedFinalOutput
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.core.tool_execution import (
    ToolBatchRequest,
    ToolBatchResult,
)
from rag.agent.loop.state import (
    LoopPause,
    LoopState,
    LoopTerminal,
    LoopTransition,
    LoopTransitionReason,
    ModelTurn,
    ModelTurnDraft,
    PendingToolCall,
    append_loop_diagnostic,
    replace_latest_transition,
)
from rag.agent.loop.stop_hooks import StopHookOutcome, StopHookRunner
from rag.agent.loop.substate import DeferredToolState, MemoryState, PlanState
from rag.agent.memory.compactor import LoopCompactionResult
from rag.agent.planning import MAX_PLAN_EVENTS, PlanEvent, PlanTracker

logger = logging.getLogger(__name__)

# ── B2c: tool classification for metrics ──
_NATIVE_TOOL_SET = frozenset({
    "read_file", "write_file", "search_text", "apply_patch",
    "run_command", "run_python", "list_files", "update_plan",
    "task", "tool_search", "activate_tools", "tool_repl",
})
_DEFERRED_TOOL_SET = frozenset({
    "search_knowledge", "search_assets", "llm_generate",
    "llm_summarize", "llm_compare", "structured_probe",
})
from rag.agent.streaming.events import StreamEvent
from rag.agent.streaming.sink import StreamEventSink
from rag.agent.tools.observation import ToolExecutionObservation
from rag.providers.llm_gateway import LLMContextOverflowError


class ModelTurnEnvelope(BaseModel):
    """Optional provider metadata surrounding one accepted draft."""

    model_config = ConfigDict(frozen=True)

    draft: ModelTurnDraft
    transitions: tuple[LoopTransition, ...] = ()


class ModelTurnProvider(Protocol):
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
    ) -> ModelTurnDraft | ModelTurnEnvelope: ...


class LoopContextManager(Protocol):
    def prepare(
        self,
        state: LoopState,
    ) -> LoopCompactionResult | Awaitable[LoopCompactionResult]: ...


class LoopToolRunner(Protocol):
    def execute_batch(
        self,
        request: ToolBatchRequest,
        *,
        state: LoopState | None,
        definition: AgentDefinition | None = None,
    ) -> ToolBatchResult | Awaitable[ToolBatchResult]: ...


class LoopEventSink(Protocol):
    async def emit(self, transition: LoopTransition) -> None: ...


class NullLoopEventSink:
    async def emit(self, transition: LoopTransition) -> None:
        del transition


class AgentLoop:
    """Claude-like single-agent kernel implemented as an ordinary while loop."""

    def __init__(
        self,
        *,
        definition: AgentDefinition,
        model_provider: ModelTurnProvider,
        context_manager: LoopContextManager,
        tool_runner: LoopToolRunner,
        checkpoint_store: CheckpointStore,
        stop_hook_runner: StopHookRunner,
        finish_candidate_builder: FinishCandidateBuilder,
        event_sink: LoopEventSink | None = None,
        stream_sink: StreamEventSink | None = None,
        observation_extractor: ObservationExtractor | None = None,
        plan_tracker: PlanTracker | None = None,
        max_model_retries: int = 1,
        catalog: ToolCatalog | None = None,
        deferred_store: DeferredToolStore | None = None,
    ) -> None:
        if max_model_retries < 0:
            raise ValueError("max_model_retries must be non-negative")
        self._definition = definition
        self._model_provider = model_provider
        self._context_manager = context_manager
        self._tool_runner = tool_runner
        self._checkpoint_store = checkpoint_store
        self._stop_hook_runner = stop_hook_runner
        self._finish_candidate_builder = finish_candidate_builder
        self._event_sink = event_sink or NullLoopEventSink()
        self._stream_sink = stream_sink
        # 把 stream_sink 注入到 model provider（如果支持）
        if stream_sink is not None and hasattr(model_provider, "_stream_sink"):
            model_provider._stream_sink = stream_sink
        self._observation_extractor = observation_extractor or ObservationExtractor()
        self._observed_tool_call_ids: set[str] = set()
        self._plan_tracker = plan_tracker or PlanTracker()
        self._max_model_retries = max_model_retries
        self._catalog = catalog
        self._deferred_store = deferred_store

    async def _emit_stream(self, event: Any) -> None:
        """Emit a stream event if sink is configured."""
        if self._stream_sink is not None:
            await self._stream_sink.emit(event)

    async def run_streaming(self, state: LoopState) -> AsyncGenerator[StreamEvent, None]:
        """流式运行 AgentLoop，yield 每一个 StreamEvent。

        如果 run() 异常，会 yield 一个 ERROR 事件然后关闭 sink。
        消费者不会挂死。

        用法：
            async for event in agent_loop.run_streaming(state):
                handle(event)
        """
        from rag.agent.streaming.sink import QueueStreamEventSink

        sink = QueueStreamEventSink()
        sink_targets: list[tuple[Any, StreamEventSink | None]] = [(self, self._stream_sink)]
        self._stream_sink = sink
        for target in (self._model_provider, self._tool_runner):
            if hasattr(target, "_stream_sink"):
                target_with_sink = cast(Any, target)
                sink_targets.append((target_with_sink, target_with_sink._stream_sink))
                target_with_sink._stream_sink = sink

        run_task: asyncio.Task[LoopState] | None = None
        try:
            run_task = asyncio.create_task(self.run(state))

            # Always close the queue when run() returns, including normal
            # completion paths that already emitted LOOP_END.
            def _on_run_done(t: asyncio.Task[LoopState]) -> None:
                del t
                asyncio.create_task(sink.close())

            run_task.add_done_callback(_on_run_done)

            # 消费事件
            async for event in sink.stream():
                yield event

            # stream 正常结束，等 run 完成（检查异常）
            await run_task

        except BaseException:
            # 消费者被取消或出错，确保 sink 关闭
            await sink.close()
            if run_task is not None and not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except (asyncio.CancelledError, Exception):
                    pass
            raise

        finally:
            for target, original_sink in sink_targets:
                target._stream_sink = original_sink

    async def run(self, state: LoopState) -> LoopState:
        if state["status"] != "running":
            return state
        handles = RunRegistry.get_or_create(state["run_config"])
        if state["agent_plan"] is None:
            plan, events = self._plan_tracker.initialize_task(task=state["task"])
            state["agent_plan"] = plan
            state["plan_state"] = PlanState(
                agent_plan=plan,
                plan_events=list(state.get("plan_events", [])),
            )
            self._append_plan_events(state, events)
        await self._checkpoint_store.save_snapshot(
            state,
            reason="loop_start",
        )

        consecutive_model_failures = 0
        while state["status"] == "running":
            # ── 流式事件：turn 开始 ──
            await self._emit_stream(
                _stream_turn_start(
                    run_id=state["run_config"].run_id,
                    turn=state["iteration"] + 1,
                )
            )
            if state["pending_tool_calls"]:
                token = iteration_var.set(state["iteration"])
                try:
                    paused = await self._execute_pending_tools(state)
                finally:
                    iteration_var.reset(token)
                if paused:
                    return state
                continue

            if state["iteration"] >= self._definition.max_iterations:
                await self._fail(
                    state,
                    stop_reason="max_iterations",
                    error=("Agent loop reached the configured maximum number of model turns."),
                    transition_reason="max_iterations",
                    checkpoint_reason="max_iterations",
                )
                return state

            try:
                compaction = await _await_value(self._context_manager.prepare(state))
            except Exception as exc:
                await self._fail(
                    state,
                    stop_reason="context_compaction_failed",
                    error=str(exc) or type(exc).__name__,
                    transition_reason="failed",
                    checkpoint_reason="context_compaction_failed",
                )
                return state
            if compaction.changed:
                transition = state["latest_transition"] or LoopTransition(
                    reason="compaction",
                    iteration=state["iteration"],
                    detail={
                        "channels": list(compaction.channels),
                        "warnings": list(compaction.warnings),
                    },
                )
                replace_latest_transition(state, transition)
                await self._emit_and_save(
                    state,
                    transition,
                    checkpoint_reason="compaction",
                )
                # ── 流式事件：压缩完成 ──
                await self._emit_stream(
                    _stream_compact_layer(
                        channels=list(compaction.channels),
                        warnings=list(compaction.warnings),
                        run_id=state["run_config"].run_id,
                        turn=state["iteration"],
                    )
                )

            remaining = await handles.budget_ledger.remaining()
            if remaining <= 0:
                await self._fail(
                    state,
                    stop_reason="budget_exhausted",
                    error="The model token budget is exhausted.",
                    transition_reason="failed",
                    checkpoint_reason="budget_exhausted",
                )
                return state

            state["iteration"] += 1
            try:
                provided = await self._model_provider.next_turn(
                    state,
                    definition=self._definition,
                    budget_remaining=remaining,
                )
                envelope = provided if isinstance(provided, ModelTurnEnvelope) else ModelTurnEnvelope(draft=provided)
                for provider_transition in envelope.transitions:
                    transition = provider_transition.model_copy(update={"iteration": state["iteration"]})
                    await self._set_transition(
                        state,
                        transition,
                        checkpoint_reason=(f"provider_{provider_transition.reason}"),
                    )
                turn = await self._finish_candidate_builder.build(
                    envelope.draft,
                    state=state,
                )
            except (AgentLLMContextOverflowError, LLMContextOverflowError) as exc:
                if not state.get("reactive_compact_used"):
                    state["reactive_compact_used"] = True
                    ms = state.get("memory_state")
                    if isinstance(ms, MemoryState):
                        state["memory_state"] = ms.model_copy(update={"reactive_compact_used": True})
                    reactive_compact = getattr(
                        self._context_manager,
                        "reactive_compact",
                        None,
                    )
                    if reactive_compact is not None:
                        try:
                            compaction = await _await_value(reactive_compact(state))
                        except Exception as compact_exc:
                            await self._fail(
                                state,
                                stop_reason="context_compaction_failed",
                                error=str(compact_exc) or type(compact_exc).__name__,
                                transition_reason="failed",
                                checkpoint_reason="context_compaction_failed",
                            )
                            return state
                        append_loop_diagnostic(
                            state,
                            RuntimeDiagnostic.from_exception(
                                code="context_overflow_recovered",
                                component="agent_loop",
                                error=exc,
                                severity="warning",
                            ),
                        )
                        if compaction.changed:
                            state["iteration"] = max(0, state["iteration"] - 1)
                            transition = LoopTransition(
                                reason="compaction",
                                iteration=state["iteration"],
                                detail={
                                    "mode": "reactive",
                                    "channels": list(compaction.channels),
                                    "warnings": list(compaction.warnings),
                                },
                            )
                            await self._set_transition(
                                state,
                                transition,
                                checkpoint_reason="reactive_compaction",
                            )
                            await self._emit_stream(
                                _stream_compact_layer(
                                    channels=list(compaction.channels),
                                    warnings=list(compaction.warnings),
                                    run_id=state["run_config"].run_id,
                                    turn=state["iteration"],
                                )
                            )
                            continue
                append_loop_diagnostic(
                    state,
                    RuntimeDiagnostic.from_exception(
                        code="context_overflow",
                        component="agent_loop",
                        error=exc,
                        severity="error",
                    ),
                )
                await self._fail(
                    state,
                    stop_reason="context_overflow",
                    error=str(exc) or type(exc).__name__,
                    transition_reason="failed",
                    checkpoint_reason="context_overflow",
                )
                return state
            except (FinishCandidateBuildError, ValidationError, ValueError) as exc:
                if isinstance(exc, FinishCandidateBuildError) and _has_tool_error(state):
                    append_loop_diagnostic(
                        state,
                        RuntimeDiagnostic.from_exception(
                            code="tool_error",
                            component="agent_loop",
                            error=exc,
                            severity="error",
                        ),
                    )
                    await self._fail(
                        state,
                        stop_reason="tool_error",
                        error=_latest_tool_error_message(state),
                        transition_reason="failed",
                        checkpoint_reason="tool_error",
                    )
                    return state
                append_loop_diagnostic(
                    state,
                    RuntimeDiagnostic.from_exception(
                        code="invalid_model_turn",
                        component="agent_loop",
                        error=exc,
                        severity="error",
                    ),
                )
                if consecutive_model_failures < self._max_model_retries:
                    consecutive_model_failures += 1
                    # ── 流式事件：恢复重试 ──
                    await self._emit_stream(
                        _stream_recovery(
                            strategy="model_retry",
                            detail=f"attempt={consecutive_model_failures}, error={str(exc)[:200]}",
                            run_id=state["run_config"].run_id,
                            turn=state["iteration"],
                        )
                    )
                    await self._transition(
                        state,
                        reason="retry",
                        detail={
                            "component": "model",
                            "attempt": consecutive_model_failures,
                            "error": str(exc),
                        },
                        checkpoint_reason="model_retry",
                    )
                    continue
                await self._fail(
                    state,
                    stop_reason="invalid_model_turn",
                    error=str(exc) or type(exc).__name__,
                    transition_reason="failed",
                    checkpoint_reason="invalid_model_turn",
                )
                return state
            except Exception as exc:
                append_loop_diagnostic(
                    state,
                    RuntimeDiagnostic.from_exception(
                        code="model_provider_failed",
                        component="agent_loop",
                        error=exc,
                        severity="error",
                    ),
                )
                if consecutive_model_failures < self._max_model_retries:
                    consecutive_model_failures += 1
                    # ── 流式事件：恢复重试 ──
                    await self._emit_stream(
                        _stream_recovery(
                            strategy="model_retry",
                            detail=f"attempt={consecutive_model_failures}, error={str(exc)[:200]}",
                            run_id=state["run_config"].run_id,
                            turn=state["iteration"],
                        )
                    )
                    await self._transition(
                        state,
                        reason="retry",
                        detail={
                            "component": "model",
                            "attempt": consecutive_model_failures,
                            "error": str(exc) or type(exc).__name__,
                        },
                        checkpoint_reason="model_retry",
                    )
                    continue
                await self._fail(
                    state,
                    stop_reason="model_provider_failed",
                    error=str(exc) or type(exc).__name__,
                    transition_reason="failed",
                    checkpoint_reason="model_provider_failed",
                )
                return state

            consecutive_model_failures = 0
            state["last_model_turn"] = turn
            if turn.action == "execute":
                state["tool_call_ledger"].append_plans(
                    turn.tool_calls,
                    turn=state["iteration"],
                )
                state["pending_tool_calls"] = [PendingToolCall(plan=call, status="pending") for call in turn.tool_calls]
                self._record_plan_decision(state, turn)
                # Trim ledger after new pending state: active = just-created calls
                state["tool_call_ledger"].trim(
                    active_tool_call_ids={p.tool_call_id for p in state["pending_tool_calls"]},
                )
            await self._transition(
                state,
                reason="next_turn",
                detail={"action": turn.action},
                checkpoint_reason="model_turn",
            )

            if turn.action == "execute":
                await self._transition(
                    state,
                    reason="tool_execution",
                    detail={
                        "phase": "scheduled",
                        "tool_call_ids": [call.tool_call_id for call in turn.tool_calls],
                    },
                    checkpoint_reason="tool_calls_scheduled",
                )
                # ── 流式事件：turn 结束（工具调度） ──
                await self._emit_stream(
                    _stream_turn_end(
                        run_id=state["run_config"].run_id,
                        turn=state["iteration"],
                        stop_reason="tool_use",
                    )
                )
                continue
            if turn.action == "pause":
                # ── 流式事件：turn 结束（暂停） ──
                await self._emit_stream(
                    _stream_turn_end(
                        run_id=state["run_config"].run_id,
                        turn=state["iteration"],
                        stop_reason="pause",
                    )
                )
                await self._pause(
                    state,
                    reason=cast(str, turn.pause_reason),
                    request=None,
                    checkpoint_reason="model_pause",
                    transition_reason="paused",
                )
                return state

            # ── 流式事件：turn 结束（完成） ──
            await self._emit_stream(
                _stream_turn_end(
                    run_id=state["run_config"].run_id,
                    turn=state["iteration"],
                    stop_reason="end_turn",
                )
            )
            finished = await self._evaluate_finish(state, turn)
            if finished:
                return state

        # ── 流式事件：loop 结束（正常退出 while） ──
        await self._emit_stream(
            _stream_loop_end(
                run_id=state["run_config"].run_id,
                reason=state.get("terminal", {}) and state["terminal"].stop_reason
                if state.get("terminal")
                else "loop_exited",
                total_turns=state["iteration"],
            )
        )
        return state

    def _sync_discovery_to_state(self, state: LoopState) -> None:
        """Sync deferred store state to LoopState discovery_* fields + DeferredToolState."""
        if self._deferred_store is not None:
            self._deferred_store.sync_to_state(cast(dict[Any, Any], state))
            # Backward compat alias
            state["active_deferred_tools"] = list(self._deferred_store.active_names())
            # ── PR1 dual-write: typed DeferredToolState ──
            state["deferred_tool_state"] = DeferredToolState(
                active_tools=list(state.get("discovery_active_tools", [])),
                active_tool_iterations=dict(state.get("discovery_active_tool_iterations", {})),
                last_candidates=_migrate_discovery_candidates(state.get("discovery_last_candidates", [])),
                last_search_query=str(state.get("discovery_last_search_query", "")),
                search_history=_migrate_discovery_events(state.get("discovery_search_history", [])),
                pinned_tools=list(state.get("discovery_pinned_tools", [])),
                capability_diagnostics=list(state.get("capability_diagnostics", [])),
            )

    async def _execute_pending_tools(self, state: LoopState) -> bool:
        # ── 流式事件：工具开始执行 ──
        run_id = state["run_config"].run_id
        turn = state["iteration"]
        for call in state["pending_tool_calls"]:
            await self._emit_stream(
                _stream_tool_use_start(
                    tool_name=call.tool_name,
                    tool_id=call.tool_call_id,
                    run_id=run_id,
                    turn=turn,
                )
            )

        result = await _await_value(
            self._tool_runner.execute_batch(
                ToolBatchRequest(
                    calls=tuple(call.plan for call in state["pending_tool_calls"]),
                    run_config=state["run_config"],
                    allowed_tools=frozenset(self._definition.allowed_tools),
                    approved_tool_call_ids=tuple(state["approved_tool_call_ids"]),
                    denied_tool_call_ids=tuple(state["denied_tool_call_ids"]),
                    execution_records=state["tool_execution_records"],
                ),
                state=state,
                definition=self._definition,
            )
        )
        state["run_config"] = result.run_config
        state["pending_tool_calls"] = [
            PendingToolCall(plan=call, status="pending") for call in result.pending_tool_calls
        ]
        state["tool_execution_records"] = {
            call_id: record.model_copy(deep=True) for call_id, record in result.execution_records.items()
        }
        new_results = list(result.tool_results)
        state["tool_results"] = _merge_keyed(
            state["tool_results"],
            new_results,
        )

        # ── 流式事件：工具执行结果 ──
        for tool_result in new_results:
            if tool_result.status == "error":
                await self._emit_stream(
                    _stream_tool_use_error(
                        tool_id=tool_result.tool_call_id,
                        error=tool_result.error.message if tool_result.error else "Unknown error",
                        run_id=run_id,
                        turn=turn,
                    )
                )
            else:
                await self._emit_stream(
                    _stream_tool_use_result(
                        tool_name=tool_result.tool_name,
                        tool_id=tool_result.tool_call_id,
                        result=str(tool_result.output)[:500] if tool_result.output else "",
                        run_id=run_id,
                        turn=turn,
                    )
                )

        if result.context_budget is not None:
            state["context_budget"] = result.context_budget
            ms = state.get("memory_state")
            if isinstance(ms, MemoryState):
                state["memory_state"] = ms.model_copy(update={"context_budget": result.context_budget})

        batch = self._observation_extractor.extract(
            new_results,
            seen_tool_call_ids=list(self._observed_tool_call_ids),
        )
        self._observed_tool_call_ids.update(observation.tool_call_id for observation in batch.structured_observations)
        self._merge_observations(state, batch)
        self._record_plan_observations(state, batch)

        for tool_result in new_results:
            if tool_result.retry_count <= 0:
                continue
            await self._transition(
                state,
                reason="retry",
                detail={
                    "component": "tool",
                    "tool_call_id": tool_result.tool_call_id,
                    "retry_count": tool_result.retry_count,
                },
                checkpoint_reason="tool_retry",
            )

        if result.status in {"paused", "reconciliation_required"}:
            request = result.human_input_request
            reason = request.question if request is not None else result.decision_reason or result.status
            state["approval_request"] = request
            # Sync active set even on pause (tool_search may have activated)
            self._sync_discovery_to_state(state)
            await self._pause(
                state,
                reason=reason,
                request=request,
                checkpoint_reason="tool_pause",
                transition_reason="approval_required",
            )
            return True

        # Sync active deferred tools after tool execution
        self._sync_discovery_to_state(state)

        # Process declarative tool batch (B2b: code-as-tool)
        batch_plans = self._process_tool_batch(state)
        if batch_plans:
            state["pending_tool_calls"].extend(batch_plans)

        await self._transition(
            state,
            reason="tool_execution",
            detail={
                "phase": "recorded",
                "result_count": len(new_results),
                "pending_count": len(state["pending_tool_calls"]),
                "skipped_completed_tool_call_ids": list(result.skipped_completed_tool_call_ids),
            },
            checkpoint_reason="tool_results_recorded",
        )

        # Trim ledger: keep entries for pending + just-completed calls
        active_ids: set[str] = set()
        for p in state["pending_tool_calls"]:
            active_ids.add(p.tool_call_id)
        for tr in new_results:
            active_ids.add(tr.tool_call_id)
        state["tool_call_ledger"].trim(active_tool_call_ids=active_ids)

        # ── B2c: tool call metrics ──
        self._record_metrics(state, new_results)

        return False

    def _record_metrics(
        self,
        state: LoopState,
        new_results: list[ToolResult],
    ) -> None:
        """B2c: record lightweight tool call counters.

        Appends a RuntimeDiagnostic with summary metrics.  The counters
        track the three calling modes (native, deferred, MCP) and
        approval behaviour.  Not persisted — rebuilt each run.
        """
        native = sum(1 for tr in new_results
                     if tr.tool_name in _NATIVE_TOOL_SET)
        deferred = sum(1 for tr in new_results
                       if tr.tool_name in _DEFERRED_TOOL_SET)
        mcp = sum(1 for tr in new_results
                  if tr.tool_name.startswith("mcp__"))
        native_err = sum(1 for tr in new_results
                         if tr.tool_name in _NATIVE_TOOL_SET and tr.status == "error")
        mcp_err = sum(1 for tr in new_results
                      if tr.tool_name.startswith("mcp__") and tr.status == "error")
        native_lat = sum(tr.latency_ms for tr in new_results
                         if tr.tool_name in _NATIVE_TOOL_SET)
        mcp_lat = sum(tr.latency_ms for tr in new_results
                      if tr.tool_name.startswith("mcp__"))

        msg = (
            f"native={native}/{native_err}err/{native_lat:.0f}ms "
            f"deferred={deferred} "
            f"mcp={mcp}/{mcp_err}err/{mcp_lat:.0f}ms"
        )
        state["runtime_diagnostics"] = [*state["runtime_diagnostics"],
            RuntimeDiagnostic(
                code="tool_call_metrics",
                component="AgentLoop",
                message=msg[:500],
                severity="warning",
                degraded=False,
            )][-20:]

    def _process_tool_batch(self, state: LoopState) -> list[PendingToolCall]:
        """Check for declarative tool batch (tool_calls.jsonl).

        Returns PendingToolCall list for validated declarations.
        The batch file is cleaned up after reading.
        """
        import os as _os

        scratch_dir = _os.environ.get("AGENT_SCRATCH_DIR")
        if not scratch_dir:
            return []
        from pathlib import Path

        batch_file = Path(scratch_dir) / "tool_calls.jsonl"
        if not batch_file.exists():
            return []

        from rag.agent.core.tool_batch_reader import clean_batch_file, read_tool_batch

        declarations = read_tool_batch(scratch_dir)
        if not declarations:
            clean_batch_file(scratch_dir)
            return []

        allowed = self._definition.allowed_tools
        new_pending: list[PendingToolCall] = []
        for decl in declarations:
            name = decl.tool_name
            # Basic validation: activation + allowed_tools
            if name not in allowed:
                logger.warning(f"Batch tool '{name}' not in allowed_tools — skipped")
                continue
            if name not in CORE_TOOLS and (
                self._deferred_store is None or not self._deferred_store.is_active(name)
            ):
                logger.warning(f"Batch tool '{name}' not activated — skipped")
                continue
            plan = ToolCallPlan(
                tool_call_id=f"batch__{name}__{decl.line_number}",
                tool_name=name,
                arguments=decl.arguments,
            )
            new_pending.append(PendingToolCall(plan=plan, status="pending"))

        if new_pending:
            logger.info(
                f"Batch: {len(new_pending)} tool(s) queued from tool_calls.jsonl"
            )
        clean_batch_file(scratch_dir)
        return new_pending

    async def _evaluate_finish(
        self,
        state: LoopState,
        turn: ModelTurn,
    ) -> bool:
        candidate = cast(str, turn.final_answer)
        outcome = await self._stop_hook_runner.evaluate(
            state=state,
            candidate=candidate,
        )
        if outcome.blocked:
            await self._transition(
                state,
                reason="stop_hook_blocked",
                detail={
                    "code": outcome.code,
                    "message": outcome.message or "",
                },
                checkpoint_reason="stop_hook_blocked",
            )
            return False
        if outcome.halted:
            await self._fail(
                state,
                stop_reason=outcome.code,
                error=outcome.message or outcome.code,
                transition_reason="failed",
                checkpoint_reason="stop_hook_halt",
                final_output=outcome.final_output,
            )
            return True
        await self._complete(
            state,
            candidate=candidate,
            outcome=outcome,
        )
        return True

    async def _complete(
        self,
        state: LoopState,
        *,
        candidate: str,
        outcome: StopHookOutcome,
    ) -> None:
        state["status"] = "completed"
        state["final_answer"] = candidate
        state["final_output"] = outcome.final_output
        state["pause"] = None
        state["terminal"] = LoopTerminal(
            status="completed",
            stop_reason=outcome.code,
            final_answer=candidate,
            final_output=outcome.final_output,
        )
        plan, events = self._plan_tracker.record_completion(state["agent_plan"])
        state["agent_plan"] = plan
        state["plan_state"] = PlanState(
            agent_plan=plan,
            plan_events=list(state.get("plan_events", [])),
        )
        self._append_plan_events(state, events)
        await self._transition(
            state,
            reason="finished",
            detail={"stop_code": outcome.code},
            checkpoint_reason="terminal_completed",
        )
        await self._emit_stream(
            _stream_loop_end(
                run_id=state["run_config"].run_id,
                reason=outcome.code,
                total_turns=state["iteration"],
            )
        )

    async def _fail(
        self,
        state: LoopState,
        *,
        stop_reason: str,
        error: str,
        transition_reason: LoopTransitionReason,
        checkpoint_reason: str,
        final_output: ValidatedFinalOutput | None = None,
    ) -> None:
        state["status"] = "failed"
        state["pause"] = None
        state["terminal"] = LoopTerminal(
            status="failed",
            stop_reason=stop_reason,
            final_output=final_output,
            error=error,
        )
        plan, events = self._plan_tracker.record_completion(
            state["agent_plan"],
            blocked=True,
        )
        state["agent_plan"] = plan
        state["plan_state"] = PlanState(
            agent_plan=plan,
            plan_events=list(state.get("plan_events", [])),
        )
        self._append_plan_events(state, events)
        await self._transition(
            state,
            reason=transition_reason,
            detail={
                "stop_reason": stop_reason,
                "error": error,
            },
            checkpoint_reason=checkpoint_reason,
        )
        # ── 流式事件：loop 结束 ──
        await self._emit_stream(
            _stream_loop_end(
                run_id=state["run_config"].run_id,
                reason=stop_reason,
                total_turns=state["iteration"],
            )
        )

    async def _pause(
        self,
        state: LoopState,
        *,
        reason: str,
        request: HumanInputRequest | None,
        checkpoint_reason: str,
        transition_reason: LoopTransitionReason,
    ) -> None:
        state["status"] = "paused"
        state["pause"] = LoopPause(
            reason=reason,
            request=request,
        )
        await self._transition(
            state,
            reason=transition_reason,
            detail={
                "reason": reason,
                "request_kind": (getattr(request, "kind", None) if request is not None else None),
            },
            checkpoint_reason=checkpoint_reason,
        )
        await self._emit_stream(
            _stream_loop_end(
                run_id=state["run_config"].run_id,
                reason=str(transition_reason),
                total_turns=state["iteration"],
            )
        )

    async def _transition(
        self,
        state: LoopState,
        *,
        reason: LoopTransitionReason,
        detail: dict[str, object],
        checkpoint_reason: str,
    ) -> None:
        transition = LoopTransition(
            reason=reason,
            iteration=state["iteration"],
            detail=detail,
        )
        await self._set_transition(
            state,
            transition,
            checkpoint_reason=checkpoint_reason,
        )

    async def _set_transition(
        self,
        state: LoopState,
        transition: LoopTransition,
        *,
        checkpoint_reason: str,
    ) -> None:
        replace_latest_transition(state, transition)
        await self._emit_and_save(
            state,
            transition,
            checkpoint_reason=checkpoint_reason,
        )

    async def _emit_and_save(
        self,
        state: LoopState,
        transition: LoopTransition,
        *,
        checkpoint_reason: str,
    ) -> None:
        await self._event_sink.emit(transition)
        await self._checkpoint_store.save_snapshot(
            state,
            reason=checkpoint_reason,
        )

    def _record_plan_decision(
        self,
        state: LoopState,
        turn: ModelTurn,
    ) -> None:
        plan = state["agent_plan"]
        if plan is None:
            return
        updated, events = self._plan_tracker.record_decision_progress(
            plan,
            tool_call_ids=[call.tool_call_id for call in turn.tool_calls],
            tool_names=[call.tool_name for call in turn.tool_calls],
        )
        state["agent_plan"] = updated
        state["plan_state"] = PlanState(
            agent_plan=updated,
            plan_events=list(state.get("plan_events", [])),
        )
        self._append_plan_events(state, events)

    def _record_plan_observations(
        self,
        state: LoopState,
        batch: ObservationBatch,
    ) -> None:
        typed_observations = [
            ToolExecutionObservation(
                tool_call_id=obs.tool_call_id,
                tool_name=obs.tool_name,
                status=obs.status,
                related_step_ids=list(getattr(obs, "related_step_ids", []) or []),
                metadata=dict(getattr(obs, "metadata", {}) or {}),
            )
            for obs in batch.structured_observations
        ]
        plan, events = self._plan_tracker.record_observation_progress(
            plan=state["agent_plan"],
            observations=typed_observations,
        )
        if plan is not None:
            state["agent_plan"] = plan
            state["plan_events"] = [*state["plan_events"], *events][-MAX_PLAN_EVENTS:]
            # PR1 dual-write
            if hasattr(state["plan_state"], "model_copy"):
                state["plan_state"] = state["plan_state"].model_copy(
                    update={"agent_plan": plan, "plan_events": list(state["plan_events"])}
                )

    @staticmethod
    def _append_plan_events(
        state: LoopState,
        events: Sequence[PlanEvent],
    ) -> None:
        state["plan_events"] = [
            *state["plan_events"],
            *events,
        ][-MAX_PLAN_EVENTS:]
        state["plan_state"] = PlanState(
            agent_plan=state.get("agent_plan"),
            plan_events=list(state["plan_events"]),
        )

    @staticmethod
    def _merge_observations(
        state: LoopState,
        batch: ObservationBatch,
    ) -> None:
        # PR2: ObservationExtractor no longer writes tool-semantic fields to LoopState.
        # Tool output rendering is handled by ToolOutputFormatter via ContextBuilder.
        return


async def _await_value[T](value: T | Awaitable[T]) -> T:
    if isawaitable(value):
        return await value
    return value


def _merge_keyed[T](existing: list[T], additions: list[T]) -> list[T]:
    merged: dict[str, T] = {}
    for item in [*existing, *additions]:
        key = _item_key(item)
        merged.pop(key, None)
        merged[key] = item
    return list(merged.values())


def _item_key(item: object) -> str:
    key = getattr(item, "key", None)
    if isinstance(key, str) and key:
        return key
    for attribute in (
        "tool_call_id",
        "source_tool_call_id",
        "unit_id",
        "evidence_id",
        "citation_id",
    ):
        value = getattr(item, attribute, None)
        if value is not None:
            return f"{attribute}:{value}"
    if isinstance(item, dict):
        return repr(sorted(item.items()))
    return repr(item)


def _has_tool_error(state: LoopState) -> bool:
    return any(result.status == "error" for result in state["tool_results"])


def _latest_tool_error_message(state: LoopState) -> str:
    for result in reversed(state["tool_results"]):
        if result.status != "error":
            continue
        if result.error is not None:
            return result.error.message
        return f"Tool {result.tool_name} failed."
    return "Tool execution failed."


# ── 流式事件 helper ──────────────────────────────────────


def _stream_turn_start(*, run_id: str, turn: int) -> Any:
    from rag.agent.streaming.events import EventType, StreamEvent, next_seq

    return StreamEvent(
        type=EventType.TURN_START,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
    )


def _stream_turn_end(*, run_id: str, turn: int, stop_reason: str) -> Any:
    from rag.agent.streaming.events import EventType, StreamEvent, next_seq

    return StreamEvent(
        type=EventType.TURN_END,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        data={"stop_reason": stop_reason},
    )


def _stream_loop_end(*, run_id: str, reason: str, total_turns: int) -> Any:
    from rag.agent.streaming.events import EventType, StreamEvent, next_seq

    return StreamEvent(
        type=EventType.LOOP_END,
        run_id=run_id,
        seq=next_seq(),
        data={"reason": reason, "total_turns": total_turns},
    )


def _stream_tool_use_start(*, tool_name: str, tool_id: str, run_id: str, turn: int) -> Any:
    from rag.agent.streaming.events import EventType, StreamEvent, next_seq

    return StreamEvent(
        type=EventType.TOOL_USE_START,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        span_id=f"tool:{tool_id}",
        data={"tool_name": tool_name, "tool_id": tool_id},
    )


def _stream_tool_use_result(*, tool_name: str, tool_id: str, result: str, run_id: str, turn: int) -> Any:
    from rag.agent.streaming.events import EventType, StreamEvent, next_seq

    return StreamEvent(
        type=EventType.TOOL_USE_RESULT,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        span_id=f"tool:{tool_id}",
        data={
            "tool_name": tool_name,
            "tool_id": tool_id,
            "result": result,
        },
    )


def _stream_tool_use_error(*, tool_id: str, error: str, run_id: str, turn: int) -> Any:
    from rag.agent.streaming.events import EventType, StreamEvent, next_seq

    return StreamEvent(
        type=EventType.TOOL_USE_ERROR,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        span_id=f"tool:{tool_id}",
        data={"tool_id": tool_id, "error": error},
    )


def _stream_compact_layer(*, channels: list[str], warnings: list[str], run_id: str, turn: int) -> Any:
    from rag.agent.streaming.events import EventType, StreamEvent, next_seq

    return StreamEvent(
        type=EventType.COMPACT_LAYER,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        data={
            "channels": channels,
            "warnings": warnings,
        },
    )


def _stream_recovery(*, strategy: str, detail: str, run_id: str, turn: int) -> Any:
    from rag.agent.streaming.events import EventType, StreamEvent, next_seq

    return StreamEvent(
        type=EventType.RECOVERY,
        run_id=run_id,
        turn=turn,
        seq=next_seq(),
        data={"strategy": strategy, "detail": detail},
    )


__all__ = [
    "AgentLoop",
    "LoopContextManager",
    "LoopEventSink",
    "LoopToolRunner",
    "ModelTurnEnvelope",
    "ModelTurnProvider",
    "NullLoopEventSink",
]
