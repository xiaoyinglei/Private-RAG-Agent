from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_runtime import Agent
from agent_runtime.runtime import builder as runtime_builder
from rag.agent import service as service_module
from rag.agent.core.checkpointing import (
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
)
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.messages import ModelMessage
from rag.agent.loop.runtime import ModelTurnEnvelope
from rag.agent.loop.state import LoopState, ModelTurnDraft
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.sessions import (
    RuntimeBinding,
    SessionBusyError,
    SessionNotFoundError,
    SessionStore,
    TurnNotFoundError,
    TurnStateError,
    TurnStatus,
)
from rag.agent.tools.registry import ToolRegistry
from rag.agent.workspace import open_workspace


def test_run_request_generates_a_uuid_turn_distinct_from_session() -> None:
    session_id = str(uuid4())

    config = AgentRunRequest(
        task="hello",
        session_id=session_id,
    ).to_run_config(AgentRuntimePolicy.test_factory())

    assert str(UUID(config.run_id)) == config.run_id
    assert config.run_id != session_id
    assert config.session_id == session_id
    assert config.thread_id == config.run_id


def test_agent_chat_returns_public_session_and_turn_ids(
    monkeypatch,
) -> None:
    requests: list[AgentRunRequest] = []

    class _Service:
        async def chat(self, request: AgentRunRequest) -> object:
            requests.append(request)
            return SimpleNamespace(
                run_id=request.run_id,
                thread_id=request.thread_id,
                session_id=request.session_id or str(uuid4()),
                status="done",
                final_answer="hello back",
                tool_results=(),
                model_call_records=(),
                citations=(),
                runtime_diagnostics=(),
            )

    monkeypatch.setattr(
        runtime_builder,
        "build_agent_service",
        lambda *_args, **_kwargs: _Service(),
    )

    result = Agent().chat("hello")

    assert result.answer == "hello back"
    assert str(UUID(result.session_id)) == result.session_id
    assert str(UUID(result.turn_id)) == result.turn_id
    assert result.session_id != result.turn_id
    assert requests[0].session_id is None
    assert requests[0].run_id == result.turn_id
    assert requests[0].thread_id == result.turn_id


def test_agent_run_starts_one_session_and_one_turn(monkeypatch) -> None:
    requests: list[AgentRunRequest] = []

    class _Service:
        async def chat(self, request: AgentRunRequest) -> object:
            requests.append(request)
            return SimpleNamespace(
                run_id=request.run_id,
                thread_id=request.thread_id,
                session_id=request.session_id or str(uuid4()),
                status="done",
                final_answer="one shot",
                tool_results=(),
                model_call_records=(),
                citations=(),
                runtime_diagnostics=(),
            )

    monkeypatch.setattr(
        runtime_builder,
        "build_agent_service",
        lambda *_args, **_kwargs: _Service(),
    )

    result = Agent().run("hello")

    assert str(UUID(result.session_id)) == result.session_id
    assert str(UUID(result.turn_id)) == result.turn_id
    assert result.session_id != result.turn_id
    assert requests[0].session_id is None


def test_agent_chat_assembles_persisted_runtime_binding(
    tmp_path,
    monkeypatch,
) -> None:
    built: list[dict[str, object]] = []

    class _Service:
        async def chat(self, request: AgentRunRequest) -> object:
            return SimpleNamespace(
                run_id=request.run_id,
                thread_id=request.thread_id,
                session_id=request.session_id,
                status="done",
                final_answer="bound",
                tool_results=(),
                model_call_records=(),
                citations=(),
                runtime_diagnostics=(),
            )

    def build_service(runtime, **kwargs):
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)

    Agent(
        model="qwen3_5_9b_mlx_4bit",
        checkpoint_db=tmp_path / "agent.sqlite",
        workspace_path=tmp_path,
        knowledge=("team-docs",),
    ).chat("hello")

    assert isinstance(built[0]["session_store"], SessionStore)
    binding = built[0]["runtime_binding"]
    assert isinstance(binding, RuntimeBinding)
    assert binding.model_alias == "qwen3_5_9b_mlx_4bit"
    assert binding.workspace_path == str(tmp_path.resolve())
    assert binding.knowledge == ("team-docs",)
    assert built[0]["checkpointer"] is not None


def test_runtime_binding_persists_the_resolved_current_model() -> None:
    agent = Agent()
    agent._model_control_plane = SimpleNamespace(
        current_model=lambda: SimpleNamespace(id="groq_gpt_oss_120b")
    )

    binding = agent._runtime_binding()

    assert binding.model_alias == "groq_gpt_oss_120b"


def test_cross_process_chat_loads_metadata_before_runtime_assembly(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "agent.sqlite"
    persisted_workspace = tmp_path / "persisted-workspace"
    persisted_workspace.mkdir()
    wrong_workspace = tmp_path / "wrong-workspace"
    wrong_workspace.mkdir()
    binding = RuntimeBinding(
        agent_type="generic",
        model_alias="qwen3_5_9b_mlx_4bit",
        workspace_path=str(persisted_workspace),
    )
    store = SessionStore(database)
    session = store.create_session(binding)
    turn = store.begin_turn(session.session_id, "first")
    store.sync_turn_messages(
        turn.turn_id,
        (
            ModelMessage(role="user", content="first"),
            ModelMessage(role="assistant", content="done"),
        ),
    )
    store.mark_terminal(turn.turn_id, TurnStatus.COMPLETED)
    store.close()
    built: list[dict[str, object]] = []

    class _Service:
        async def chat(self, request: AgentRunRequest) -> object:
            return SimpleNamespace(
                run_id=request.run_id,
                thread_id=request.thread_id,
                session_id=request.session_id,
                status="done",
                final_answer="continued",
                tool_results=(),
                model_call_records=(),
                citations=(),
                runtime_diagnostics=(),
            )

    def build_service(runtime, **kwargs):
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)

    Agent(
        model="groq_default",
        checkpoint_db=database,
        workspace_path=wrong_workspace,
    ).chat("second", session_id=session.session_id)

    assert built[0]["model_alias"] == binding.model_alias
    assert built[0]["runtime_binding"] == binding
    runtime = built[0]["runtime"]
    assert runtime.root == persisted_workspace.resolve()


def test_cross_process_resume_loads_turn_metadata_before_runtime_assembly(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "agent.sqlite"
    persisted_workspace = tmp_path / "persisted-workspace"
    persisted_workspace.mkdir()
    binding = RuntimeBinding(
        model_alias="qwen3_5_9b_mlx_4bit",
        workspace_path=str(persisted_workspace),
    )
    store = SessionStore(database)
    session = store.create_session(binding)
    turn = store.begin_turn(session.session_id, "write it")
    store.mark_paused(turn.turn_id)
    store.close()
    built: list[dict[str, object]] = []
    resumed: list[tuple[str, str, str | None]] = []

    class _Service:
        async def resume_turn(
            self,
            *,
            turn_id: str,
            action: str,
            user_input: str | None,
        ) -> object:
            resumed.append((turn_id, action, user_input))
            return SimpleNamespace(
                run_id=turn_id,
                thread_id=turn_id,
                session_id=session.session_id,
                status="done",
                final_answer="resumed",
                tool_results=(),
                model_call_records=(),
                citations=(),
                runtime_diagnostics=(),
            )

    def build_service(runtime, **kwargs):
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)

    result = Agent(
        model="groq_default",
        checkpoint_db=database,
        workspace_path=tmp_path / "wrong",
    ).resume(turn.turn_id, "continue", user_input="approved details")

    assert result.session_id == session.session_id
    assert result.turn_id == turn.turn_id
    assert resumed == [(turn.turn_id, "continue", "approved details")]
    assert built[0]["model_alias"] == binding.model_alias
    assert built[0]["runtime_binding"] == binding
    assert built[0]["runtime"].root == persisted_workspace.resolve()


def test_completed_turn_resume_fails_before_runtime_assembly(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "agent.sqlite"
    store = SessionStore(database)
    session = store.create_session(RuntimeBinding())
    turn = store.begin_turn(session.session_id, "done")
    store.mark_terminal(turn.turn_id, TurnStatus.COMPLETED)
    store.close()
    built = False

    def build_service(*_args, **_kwargs):
        nonlocal built
        built = True
        raise AssertionError("runtime must not be assembled")

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)

    with pytest.raises(TurnStateError, match="completed"):
        Agent(checkpoint_db=database).resume(turn.turn_id, "continue")

    assert built is False


def test_paused_session_chat_fails_before_runtime_assembly(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "agent.sqlite"
    store = SessionStore(database)
    session = store.create_session(RuntimeBinding())
    turn = store.begin_turn(session.session_id, "waiting")
    store.mark_paused(turn.turn_id)
    store.close()
    built = False

    def build_service(*_args, **_kwargs):
        nonlocal built
        built = True
        raise AssertionError("runtime must not be assembled")

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)

    with pytest.raises(SessionBusyError, match=turn.turn_id):
        Agent(checkpoint_db=database).chat(
            "must fail",
            session_id=session.session_id,
        )

    assert built is False


def test_missing_session_or_turn_fails_before_runtime_assembly(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "agent.sqlite"
    built = False

    def build_service(*_args, **_kwargs):
        nonlocal built
        built = True
        raise AssertionError("runtime must not be assembled")

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)
    facade = Agent(checkpoint_db=database)

    with pytest.raises(SessionNotFoundError, match="Session not found"):
        facade.chat("hello", session_id=str(uuid4()))
    with pytest.raises(TurnNotFoundError, match="Turn not found"):
        facade.resume(str(uuid4()), "continue")

    assert built is False


@pytest.mark.anyio
async def test_service_chat_creates_a_new_turn_with_canonical_history(
    tmp_path,
) -> None:
    observed: list[tuple[str, tuple[ModelMessage, ...]]] = []

    class _Provider:
        async def next_turn(
            self,
            state: LoopState,
            *,
            definition: AgentRuntimePolicy,
            budget_remaining: int,
        ) -> ModelTurnEnvelope:
            del definition, budget_remaining
            observed.append(
                (state["task"], tuple(state["canonical_transcript"]))
            )
            answer = "remembered" if len(observed) == 1 else "alpha"
            return ModelTurnEnvelope(
                draft=ModelTurnDraft(
                    action="finish",
                    final_answer=answer,
                ),
                assistant_message=ModelMessage(
                    role="assistant",
                    content=answer,
                ),
            )

    store = SessionStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    definition = AgentRuntimePolicy.test_factory()
    binding = RuntimeBinding(
        workspace_path=str(tmp_path),
        model_alias="qwen3_5_9b_mlx_4bit",
    )
    service = AgentService(
        definition=definition,
        tool_registry=ToolRegistry(),
        model_turn_provider=_Provider(),
        checkpointer=checkpointer,
        workspace=open_workspace(tmp_path),
        session_store=store,
        runtime_binding=binding,
    )

    first = await service.chat(AgentRunRequest(task="remember alpha"))
    second = await service.chat(
        AgentRunRequest(
            task="what did I ask you to remember?",
            session_id=first.session_id,
        )
    )

    assert first.session_id == second.session_id
    assert first.run_id != second.run_id
    assert observed == [
        ("remember alpha", ()),
        (
            "remember alpha",
            (
                ModelMessage(role="assistant", content="remembered"),
                ModelMessage(
                    role="user",
                    content="what did I ask you to remember?",
                ),
            ),
        ),
    ]
    assert store.history(first.session_id) == (
        ModelMessage(role="user", content="remember alpha"),
        ModelMessage(role="assistant", content="remembered"),
        ModelMessage(
            role="user",
            content="what did I ask you to remember?",
        ),
        ModelMessage(role="assistant", content="alpha"),
    )
    checkpoint = LangGraphCheckpointStore(
        checkpointer,
        run_config=AgentRunRequest(
            task="unused",
            session_id=second.session_id,
            run_id=second.run_id,
        ).to_run_config(definition),
    )
    restored = await checkpoint.load_latest()
    assert restored is not None
    assert restored["canonical_transcript"] == [
        ModelMessage(
            role="user",
            content="what did I ask you to remember?",
        ),
        ModelMessage(role="assistant", content="alpha"),
    ]


@pytest.mark.anyio
async def test_chat_preparation_failure_marks_turn_failed_and_unblocks_session(
    tmp_path,
) -> None:
    class _FinishingProvider:
        async def next_turn(
            self,
            state: LoopState,
            *,
            definition: AgentRuntimePolicy,
            budget_remaining: int,
        ) -> ModelTurnEnvelope:
            del state, definition, budget_remaining
            return ModelTurnEnvelope(
                draft=ModelTurnDraft(
                    action="finish",
                    final_answer="recovered",
                ),
                assistant_message=ModelMessage(
                    role="assistant",
                    content="recovered",
                ),
            )

    store = SessionStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    definition = AgentRuntimePolicy.test_factory()
    session = store.create_session(RuntimeBinding(workspace_path=str(tmp_path)))
    failed_turn_id = str(uuid4())
    service = AgentService(
        definition=definition,
        tool_registry=ToolRegistry(),
        model_turn_provider=_FinishingProvider(),
        checkpointer=checkpointer,
        workspace=open_workspace(tmp_path),
        session_store=store,
        runtime_binding=session.runtime,
    )

    with pytest.raises(FileNotFoundError, match="Source file not found"):
        await service.chat(
            AgentRunRequest(
                task="read the missing file",
                session_id=session.session_id,
                run_id=failed_turn_id,
                input_files=[str(tmp_path / "missing.txt")],
            )
        )

    assert store.get_turn(failed_turn_id).status is TurnStatus.FAILED
    assert store.get_session(session.session_id).active_turn_id is None

    next_turn = await service.chat(
        AgentRunRequest(
            task="continue after failure",
            session_id=session.session_id,
        )
    )
    assert next_turn.status == "done"
    assert next_turn.run_id != failed_turn_id


@pytest.mark.anyio
async def test_session_without_explicit_workspace_persists_one_for_later_turns(
    tmp_path,
) -> None:
    class _FinishingProvider:
        async def next_turn(
            self,
            state: LoopState,
            *,
            definition: AgentRuntimePolicy,
            budget_remaining: int,
        ) -> ModelTurnEnvelope:
            del state, definition, budget_remaining
            return ModelTurnEnvelope(
                draft=ModelTurnDraft(action="finish", final_answer="done"),
                assistant_message=ModelMessage(
                    role="assistant",
                    content="done",
                ),
            )

    database = tmp_path / "agent.sqlite"
    store = SessionStore(database)
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    first_service = AgentService(
        definition=AgentRuntimePolicy.test_factory(),
        tool_registry=ToolRegistry(),
        model_turn_provider=_FinishingProvider(),
        checkpointer=checkpointer,
        session_store=store,
        runtime_binding=RuntimeBinding(),
    )

    first = await first_service.chat(AgentRunRequest(task="first"))
    session = store.get_session(first.session_id)

    assert session.runtime.workspace_path is not None
    workspace = Path(session.runtime.workspace_path)
    assert workspace.is_dir()
    assert Path(first.workspace_path or "") == workspace

    restored_service = AgentService(
        definition=AgentRuntimePolicy.test_factory(),
        tool_registry=ToolRegistry(),
        model_turn_provider=_FinishingProvider(),
        checkpointer=checkpointer,
        session_store=store,
        runtime_binding=session.runtime,
    )
    second = await restored_service.chat(
        AgentRunRequest(task="second", session_id=first.session_id)
    )

    assert second.workspace_path == first.workspace_path
    assert store.get_turn(second.run_id).runtime.workspace_path == str(workspace)


@pytest.mark.anyio
async def test_service_streaming_chat_uses_session_and_turn_lifecycle(
    tmp_path,
) -> None:
    class _FinishingProvider:
        async def next_turn(
            self,
            state: LoopState,
            *,
            definition: AgentRuntimePolicy,
            budget_remaining: int,
        ) -> ModelTurnEnvelope:
            del state, definition, budget_remaining
            return ModelTurnEnvelope(
                draft=ModelTurnDraft(
                    action="finish",
                    final_answer="streamed",
                ),
                assistant_message=ModelMessage(
                    role="assistant",
                    content="streamed",
                ),
            )

    store = SessionStore(tmp_path / "agent.sqlite")
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(),
        tool_registry=ToolRegistry(),
        model_turn_provider=_FinishingProvider(),
        checkpointer=MemorySaver(serde=agent_checkpoint_serde()),
        workspace=open_workspace(tmp_path),
        session_store=store,
        runtime_binding=RuntimeBinding(workspace_path=str(tmp_path)),
    )

    events = [
        event
        async for event in service.chat_streaming(
            AgentRunRequest(task="stream this")
        )
    ]

    assert events
    session_ids = {event.session_id for event in events}
    turn_ids = {event.turn_id for event in events}
    assert len(session_ids) == 1
    assert len(turn_ids) == 1
    session_id = session_ids.pop()
    turn_id = turn_ids.pop()
    assert str(UUID(session_id)) == session_id
    assert str(UUID(turn_id)) == turn_id
    assert session_id != turn_id
    assert store.get_turn(turn_id).status is TurnStatus.COMPLETED
    assert store.history(session_id) == (
        ModelMessage(role="user", content="stream this"),
        ModelMessage(role="assistant", content="streamed"),
    )


@pytest.mark.anyio
async def test_closing_stream_marks_the_unfinished_turn_interrupted(
    tmp_path,
) -> None:
    class _BlockingProvider:
        async def next_turn(
            self,
            state: LoopState,
            *,
            definition: AgentRuntimePolicy,
            budget_remaining: int,
        ) -> ModelTurnEnvelope:
            del state, definition, budget_remaining
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    store = SessionStore(tmp_path / "agent.sqlite")
    session = store.create_session(RuntimeBinding(workspace_path=str(tmp_path)))
    turn_id = str(uuid4())
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(),
        tool_registry=ToolRegistry(),
        model_turn_provider=_BlockingProvider(),
        checkpointer=MemorySaver(serde=agent_checkpoint_serde()),
        workspace=open_workspace(tmp_path),
        session_store=store,
        runtime_binding=session.runtime,
    )
    stream = service.chat_streaming(
        AgentRunRequest(
            task="keep working",
            session_id=session.session_id,
            run_id=turn_id,
        )
    )

    first_event = await anext(stream)
    assert first_event.turn_id == turn_id
    await stream.aclose()

    assert store.get_turn(turn_id).status is TurnStatus.INTERRUPTED
    assert store.get_session(session.session_id).active_turn_id == turn_id


@pytest.mark.anyio
async def test_running_turn_renews_its_lease_during_a_long_model_call(
    tmp_path,
    monkeypatch,
) -> None:
    started = asyncio.Event()

    class _BlockingProvider:
        async def next_turn(
            self,
            state: LoopState,
            *,
            definition: AgentRuntimePolicy,
            budget_remaining: int,
        ) -> ModelTurnEnvelope:
            del state, definition, budget_remaining
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    monkeypatch.setattr(service_module, "_TURN_LEASE_SECONDS", 0.05)
    monkeypatch.setattr(
        service_module,
        "_TURN_LEASE_HEARTBEAT_SECONDS",
        0.01,
    )
    store = SessionStore(tmp_path / "agent.sqlite")
    session = store.create_session(RuntimeBinding(workspace_path=str(tmp_path)))
    turn_id = str(uuid4())
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(),
        tool_registry=ToolRegistry(),
        model_turn_provider=_BlockingProvider(),
        checkpointer=MemorySaver(serde=agent_checkpoint_serde()),
        workspace=open_workspace(tmp_path),
        session_store=store,
        runtime_binding=session.runtime,
    )
    running = asyncio.create_task(
        service.chat(
            AgentRunRequest(
                task="long model call",
                session_id=session.session_id,
                run_id=turn_id,
            )
        )
    )
    await started.wait()
    before = store.get_turn(turn_id).lease_expires_at
    await asyncio.sleep(0.035)
    after = store.get_turn(turn_id).lease_expires_at
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert before is not None and after is not None
    assert after > before
    assert store.get_turn(turn_id).status is TurnStatus.INTERRUPTED
