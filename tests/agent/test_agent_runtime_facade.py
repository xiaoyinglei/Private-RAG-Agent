from __future__ import annotations

import asyncio
import re
import shlex
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from agent_runtime import Agent, AgentResult, AgentUsage
from agent_runtime import agent as agent_module
from agent_runtime.knowledge_providers.rag import LazyRAGKnowledgeProvider
from agent_runtime.models import ModelControlPlane
from agent_runtime.runtime import builder as runtime_builder
from rag.agent import cli as agent_cli
from rag.agent.cli import agent_app
from rag.agent.core.runtime_diagnostics import AgentLatencyProfile
from rag.agent.service import AgentRunResult
from rag.agent.tools.executor import ToolExecutor
from rag.agent.tools.integrations.knowledge import KnowledgeSearchInput
from rag.agent.tools.permissions import ToolExecutionContext
from rag.agent.tools.tool import ToolCall, ToolCallOrigin

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(output: str) -> str:
    return _ANSI_RE.sub("", output)


def test_agent_runtime_exports_sdk_facade() -> None:
    assert Agent is not None
    assert AgentResult is not None
    assert AgentUsage is not None


def test_agent_facade_run_maps_public_request_to_internal_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, Any]] = []
    requests: list[Any] = []

    def fail_rag_runtime(**_: object) -> object:
        raise AssertionError("Agent() without knowledge must not initialize RAG")

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            requests.append(request)
            return AgentRunResult(
                run_id=request.run_id,
                thread_id=request.thread_id,
                status="done",
                final_answer="facade answer",
            )

    def build_service(runtime: object, **kwargs: object) -> _Service:
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_optional_rag_runtime", fail_rag_runtime)
    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)
    monkeypatch.setattr(
        agent_cli,
        "_build_agent_service",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Agent SDK must not build services through rag.agent.cli")
        ),
    )

    result = Agent(model="qwen3_14b_mlx_4bit").run(
        "summarize",
        files=["README.md"],
        run_id="sdk-run",
        max_tokens_total=1234,
    )

    assert result.answer == "facade answer"
    assert result.status == "done"
    assert result.files == ("README.md",)
    assert isinstance(built[0]["model_control_plane"], ModelControlPlane)
    assert isinstance(built[0]["startup_ms"], float)
    assert built[0]["startup_ms"] >= 0
    assert built == [
        {
            "runtime": None,
            "checkpoint_db": None,
            "agent_type": "generic",
            "model_alias": "qwen3_14b_mlx_4bit",
            "model_control_plane": built[0]["model_control_plane"],
            "runtime_diagnostics": (),
            "knowledge_runner": None,
            "mcp_tools": (),
            "skill_tools": (),
            "subagent_tools": (),
            "skill_runtime": None,
            "stream_sink": None,
            "startup_ms": built[0]["startup_ms"],
        }
    ]
    assert len(requests) == 1
    request = requests[0]
    assert request.task == "summarize"
    assert request.run_id == "sdk-run"
    assert request.thread_id == "sdk-run"
    assert request.llm_budget_total == 1234
    assert request.input_files == ["README.md"]
    assert request.tools is None
    assert request.disabled_tools == ()
    assert request.allow_write_tools is False
    assert request.allow_execute_tools is False
    assert request.allow_discovery_tools is None


def test_agent_facade_run_passes_explicit_single_runtime_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Any] = []

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            requests.append(request)
            return AgentRunResult(
                run_id=request.run_id,
                thread_id=request.thread_id,
                status="done",
                final_answer="configured tools",
            )

    monkeypatch.setattr(runtime_builder, "build_agent_service", lambda *_args, **_kwargs: _Service())

    with pytest.warns(DeprecationWarning, match="tool selection options"):
        Agent().run(
            "Find AgentService in this repository.",
            run_id="sdk-tools",
            tools=["search_text", "read_file", "run_command"],
            disabled_tools=["read_file"],
            allow_execute_tools=True,
        )

    assert len(requests) == 1
    request = requests[0]
    assert request.tools == ("search_text", "read_file", "run_command")
    assert request.disabled_tools == ("read_file",)
    assert request.allow_write_tools is False
    assert request.allow_execute_tools is True
    assert request.allow_discovery_tools is False


def test_agent_facade_binds_public_workspace_to_builder_and_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[object] = []
    requests: list[Any] = []

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            requests.append(request)
            return AgentRunResult(
                run_id=request.run_id,
                thread_id=request.thread_id,
                status="done",
                final_answer="workspace bound",
                workspace_path=request.workspace_path,
            )

    def build_service(runtime: object, **_kwargs: object) -> _Service:
        built.append(runtime)
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)

    result = Agent(workspace_path=tmp_path / "workspace").run(
        "Inspect this workspace.",
        run_id="workspace-run",
    )

    assert result.answer == "workspace bound"
    assert built[0].root == (tmp_path / "workspace").resolve()
    assert requests[0].workspace_path == str((tmp_path / "workspace").resolve())


def test_agent_facade_assembles_workspace_skills_into_fixed_gateways(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    skill_dir = workspace / ".agents" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code\n---\nReview carefully.\n",
        encoding="utf-8",
    )
    built: list[dict[str, Any]] = []

    class _Service:
        pass

    def build_service(runtime: object, **kwargs: object) -> _Service:
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)

    Agent(workspace_path=workspace)._build_service()

    assert [
        tool.definition.name for tool in built[0]["skill_tools"]
    ] == ["invoke_skill", "materialize_skill_asset"]
    assert [
        tool.definition.name for tool in built[0]["subagent_tools"]
    ] == ["task"]
    skill_runtime = built[0]["skill_runtime"]
    assert skill_runtime.model_invocable_skill_ids == ("project:review",)


@pytest.mark.anyio
async def test_product_subagent_projection_runs_a_bounded_child_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, Any]] = []
    closed: list[bool] = []

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            assert request.max_depth == 0
            return AgentRunResult(
                run_id="child-run",
                thread_id="child-run",
                status="done",
                final_answer="child answer",
            )

        async def aclose(self) -> None:
            closed.append(True)

    def build_service(runtime: object, **kwargs: object) -> _Service:
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)
    Agent(workspace_path=tmp_path)._build_service()
    [tool] = built[0]["subagent_tools"]
    call = ToolCall(
        tool_call_id="call_task",
        tool_name="task",
        arguments={"task": "inspect one file", "max_turns": 2},
        origin=ToolCallOrigin(
            request_id="request_task",
            toolset_revision="tools_task",
            exposed_tool_names=("task",),
        ),
    )

    execution = await ToolExecutor({"task": tool}).execute(
        call,
        context=ToolExecutionContext(
            approved_tool_call_ids=frozenset({"call_task"})
        ),
    )

    assert execution.result.is_error is False
    assert execution.result.structured_content["conclusion"] == "child answer"
    assert len(built) == 2
    assert closed == [True]


def test_agent_facade_registers_knowledge_runner_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[dict[str, Any]] = []

    def fail_rag_runtime(**_: object) -> object:
        raise AssertionError("Knowledge provider must initialize RAG only when the tool is called")

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            return AgentRunResult(
                run_id=request.run_id,
                thread_id=request.thread_id,
                status="done",
                final_answer="knowledge runner registered",
            )

    def build_service(runtime: object, **kwargs: object) -> _Service:
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_optional_rag_runtime", fail_rag_runtime)
    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)

    result = Agent(model="qwen3_14b_mlx_4bit", knowledge=["company_docs"]).run(
        "lookup policy",
        run_id="knowledge-run",
    )

    assert result.answer == "knowledge runner registered"
    assert built[0]["runtime"] is None
    assert built[0]["knowledge_runner"] is not None
    assert "knowledge_asset_runner" not in built[0]


def test_agent_facade_closes_service_after_run(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            return AgentRunResult(
                run_id=request.run_id,
                thread_id=request.thread_id,
                status="done",
                final_answer="closed",
            )

        async def aclose(self) -> None:
            closed.append(True)

    monkeypatch.setattr(runtime_builder, "build_agent_service", lambda *_args, **_kwargs: _Service())

    result = Agent(model="qwen3_14b_mlx_4bit").run("close service", run_id="close-service")

    assert result.answer == "closed"
    assert closed == [True]


@pytest.mark.anyio
async def test_agent_stream_explicit_close_releases_one_shot_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    class _Service:
        async def run_streaming(self, request: Any):
            del request
            yield "first"
            yield "second"

        async def aclose(self) -> None:
            closed.append(True)

    monkeypatch.setattr(
        runtime_builder,
        "build_agent_service",
        lambda *_args, **_kwargs: _Service(),
    )

    stream = Agent().stream("stream task", run_id="stream-close")
    assert await anext(stream) == "first"
    await stream.aclose()

    assert closed == [True]


@pytest.mark.anyio
async def test_agent_runtime_close_has_a_bounded_grace_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_cancelled: list[bool] = []

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            return AgentRunResult(
                run_id=request.run_id,
                thread_id=request.thread_id,
                status="done",
                final_answer="done",
            )

        async def aclose(self) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                close_cancelled.append(True)

    monkeypatch.setattr(agent_module, "_RUNTIME_CLOSE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(
        runtime_builder,
        "build_agent_service",
        lambda *_args, **_kwargs: _Service(),
    )

    result = await Agent().arun("bounded close", run_id="bounded-close")

    assert result.answer == "done"
    assert close_cancelled == [True]


def test_agent_result_usage_uses_latency_profile_total() -> None:
    raw = AgentRunResult(
        run_id="sdk-profile",
        thread_id="sdk-profile",
        status="done",
        final_answer="profiled",
        latency_profile=AgentLatencyProfile(
            total_ms=42.0,
            tool_latency_ms=5.0,
        ),
    )

    result = AgentResult.from_internal(raw)

    assert result.usage.latency_ms == 42.0


@pytest.mark.anyio
async def test_lazy_knowledge_provider_search_uses_typed_final_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, object]] = []

    class _Runtime:
        def query(self, query: str, *, options: object) -> object:
            seen.append((query, options))
            return SimpleNamespace(
                answer=SimpleNamespace(
                    answer_text="Revenue increased.",
                    citations=[
                        SimpleNamespace(
                            citation_anchor="report#revenue",
                            citation_id="citation-1",
                        )
                    ],
                    groundedness_flag=True,
                    insufficient_evidence_flag=False,
                ),
                retrieval=SimpleNamespace(
                    evidence=SimpleNamespace(
                        all=[
                            SimpleNamespace(
                                evidence_id="evidence-1",
                                doc_id=11,
                                citation_anchor="report#revenue",
                                text="Revenue increased by 12%.",
                                score=0.94,
                                source_type="section",
                                file_name="report.pdf",
                            )
                        ]
                    )
                ),
            )

    runtime = _Runtime()

    def build_runtime(**_: object) -> tuple[object, tuple[object, ...]]:
        return runtime, ()

    monkeypatch.setattr(runtime_builder, "build_optional_rag_runtime", build_runtime)

    provider = LazyRAGKnowledgeProvider()
    result = await provider.search_knowledge(
        KnowledgeSearchInput(query="revenue", top_k=1),
        execution_context=cast(Any, object()),
    )

    assert result.total_found == 1
    assert result.answer_text == "Revenue increased."
    assert result.results[0].evidence_id == "evidence-1"
    assert result.citations == ["report#revenue"]
    assert seen[0][0] == "revenue"
    assert seen[0][1].top_k == 1


def test_agent_run_cli_delegates_to_agent_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _Facade:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("init", kwargs))
            self.workspace_path = kwargs["workspace_path"]

        @asynccontextmanager
        async def _open_product_runtime(self, **_kwargs: object):
            class _Service:
                async def run(self, request: Any) -> AgentRunResult:
                    calls.append(("run", {"request": request}))
                    return AgentRunResult(
                        run_id=request.run_id,
                        thread_id=request.thread_id,
                        status="done",
                        final_answer="cli facade answer",
                    )

            yield _Service()

    def fail_rag_runtime(**_: object) -> object:
        raise AssertionError("CLI run without --knowledge must not initialize RAG")

    monkeypatch.setattr(agent_cli, "_create_agent_facade", lambda **kwargs: _Facade(**kwargs))
    monkeypatch.setattr(runtime_builder, "build_optional_rag_runtime", fail_rag_runtime)

    result = CliRunner().invoke(
        agent_app,
        [
            "run",
            "hello",
            "--model",
            "qwen3_14b_4bit",
            "--file",
            str(Path("README.md")),
            "--run-id",
            "cli-run",
        ],
        env={"COLUMNS": "240"},
    )

    assert result.exit_code == 0, result.output
    assert "cli facade answer" in result.output
    init = calls[0][1]
    assert init["checkpoint_db"] == Path(".rag/agent_checkpoints.sqlite")
    assert init["workspace_path"] == Path.cwd()
    request = calls[1][1]["request"]
    assert request.task == "hello"
    assert request.input_files == ["README.md"]
    assert request.run_id == "cli-run"
    assert request.allow_discovery_tools is None


def test_agent_run_cli_passes_explicit_tool_surface_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _Facade:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("init", kwargs))
            self.workspace_path = kwargs["workspace_path"]

        @asynccontextmanager
        async def _open_product_runtime(self, **_kwargs: object):
            class _Service:
                async def run(self, request: Any) -> AgentRunResult:
                    calls.append(("run", {"request": request}))
                    return AgentRunResult(
                        run_id=request.run_id,
                        thread_id=request.thread_id,
                        status="done",
                        final_answer="cli tools",
                    )

            yield _Service()

    monkeypatch.setattr(agent_cli, "_create_agent_facade", lambda **kwargs: _Facade(**kwargs))

    result = CliRunner().invoke(
        agent_app,
        [
            "run",
            "Find AgentService in this repository.",
            "--run-id",
            "cli-tools",
            "--tool",
            "search_text",
            "--tool",
            "read_file",
            "--disable-tool",
            "read_file",
            "--allow-execute-tools",
        ],
        env={"COLUMNS": "240"},
    )

    assert result.exit_code == 0, result.output
    request = calls[1][1]["request"]
    assert request.task == "Find AgentService in this repository."
    assert request.tools == ("search_text", "read_file")
    assert request.disabled_tools == ("read_file",)
    assert request.allow_execute_tools is True
    assert request.allow_discovery_tools is False


def test_noninteractive_pause_prints_complete_resume_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project workspace"
    checkpoint = tmp_path / "state files" / "agent.sqlite"

    class _Facade:
        workspace_path = workspace

        @asynccontextmanager
        async def _open_product_runtime(self, **_kwargs: object):
            class _Service:
                async def run(self, request: Any) -> AgentRunResult:
                    return AgentRunResult(
                        run_id=request.run_id,
                        thread_id=request.thread_id,
                        status="paused",
                        needs_user_input="approve command",
                        workspace_path=str(workspace),
                    )

            yield _Service()

    monkeypatch.setattr(
        agent_cli,
        "_create_agent_facade",
        lambda **_kwargs: _Facade(),
    )

    result = CliRunner().invoke(
        agent_app,
        [
            "run",
            "execute task",
            "--run-id",
            "paused-run",
            "--model",
            "test-model",
            "--knowledge",
            "company docs",
            "--checkpoint-db",
            str(checkpoint),
            "--non-interactive",
        ],
        env={"COLUMNS": "240"},
    )

    expected = shlex.join(
        [
            "agent",
            "resume",
            "paused-run",
            "--agent",
            "generic",
            "--checkpoint-db",
            str(checkpoint),
            "--model",
            "test-model",
            "--knowledge",
            "company docs",
            "--workspace-path",
            str(workspace),
        ]
    )
    assert result.exit_code == 2
    assert expected in _strip_ansi(result.output)


def test_agent_run_help_matches_public_api_surface() -> None:
    result = CliRunner().invoke(
        agent_app,
        ["run", "--help"],
        env={"COLUMNS": "240"},
        terminal_width=240,
        color=False,
    )

    assert result.exit_code == 0
    output = _strip_ansi(result.output)
    assert "--model" in output
    assert "--file" in output
    assert "--knowledge" in output
    assert "--tool" not in output
    assert "--disable-tool" not in output
    assert "--input-file" in output
    assert "--budget" not in output
    assert "--embedding-model" not in output
    assert "--reranker-model" not in output
    assert "--storage-root" not in output
