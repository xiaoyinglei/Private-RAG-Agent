from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.runtime.builder import (
    CLI_AGENT_CHOICES,
    build_agent_service,
    build_optional_rag_runtime,
    resolve_auto_rag_config,
    resolve_cli_agent_definition,
)
from rag.agent.builtin import create_builtin_agent_registry
from rag.agent.cli import (
    _build_agent_service,
    _CLIToolEventDisplay,
    _display_result,
)
from rag.agent.planning import AgentPlan, PlanEvent, PlanStep
from rag.agent.service import AgentRunRequest, AgentRunResult
from rag.agent.streaming.events import (
    EventType,
    StreamEvent,
    recovery_event,
    text_delta,
    tool_use_error,
    tool_use_progress,
    tool_use_result,
    tool_use_start,
)
from rag.agent.tools.builtins import RESIDENT_CODING_TOOL_NAMES
from rag.agent.tools.integrations.knowledge import KnowledgeSearchOutput
from rag.agent.tools.integrations.mcp import (
    MCPToolDescriptor,
    create_mcp_tools,
)
from rag.agent.tools.integrations.skills import create_skill_tools
from rag.agent.tools.tool import ToolResult
from rag.agent.workspace import open_workspace


class _ModelRegistry:
    default_model = "fake"

    def resolve_for_node(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError("model resolution is not needed for assembly")


def test_cli_supports_only_the_product_generic_agent() -> None:
    registry = create_builtin_agent_registry()

    assert CLI_AGENT_CHOICES == ("generic",)
    assert resolve_cli_agent_definition(registry, "generic").agent_type == (
        "generic"
    )
    with pytest.raises(ValueError, match="supported CLI agent"):
        resolve_cli_agent_definition(registry, "research")


def test_builder_assembles_default_six_tools_in_product_order() -> None:
    service = build_agent_service(
        None,
        model_control_plane=_ModelRegistry(),  # type: ignore[arg-type]
    )

    assert tuple(service._tool_snapshot) == RESIDENT_CODING_TOOL_NAMES
    assert service._tool_executor._tools is service._tool_snapshot
    state = service.initial_state(AgentRunRequest(task="Inspect repository."))
    assert tuple(state["resident_tool_names"]) == RESIDENT_CODING_TOOL_NAMES


def test_builder_binds_coding_tools_to_the_supplied_workspace(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "visible.txt").write_text("visible", encoding="utf-8")

    service = build_agent_service(
        workspace,
        model_control_plane=_ModelRegistry(),  # type: ignore[arg-type]
    )
    tool = service._tool_snapshot["list_files"]
    output = tool.run(tool.validate_input({}))

    assert service._workspace is workspace
    assert any(entry.name == "visible.txt" for entry in output.entries)


@pytest.mark.anyio
async def test_configured_knowledge_is_a_resident_extension() -> None:
    async def search(_payload: object, **_kwargs: object) -> object:
        return KnowledgeSearchOutput(
            answer_text="configured knowledge",
            total_found=0,
        )

    service = build_agent_service(
        None,
        model_control_plane=_ModelRegistry(),  # type: ignore[arg-type]
        knowledge_runner=search,  # type: ignore[arg-type]
    )
    state = service.initial_state(AgentRunRequest(task="Search docs."))

    assert tuple(service._tool_snapshot) == (
        *RESIDENT_CODING_TOOL_NAMES,
        "search_knowledge",
    )
    assert state["resident_tool_names"] == [
        *RESIDENT_CODING_TOOL_NAMES,
        "search_knowledge",
    ]


def test_cli_wrapper_uses_the_same_builder_snapshot() -> None:
    service = _build_agent_service(
        None,
        model_control_plane=_ModelRegistry(),  # type: ignore[arg-type]
    )

    assert tuple(service._tool_snapshot) == RESIDENT_CODING_TOOL_NAMES


def test_cli_shows_called_tool_names_without_verbose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _display_result(
        AgentRunResult(
            run_id="visible-call",
            thread_id="visible-call",
            status="done",
            tool_results=[
                ToolResult(
                    tool_call_id="call_search",
                    tool_name="search_text",
                )
            ],
        ),
        verbose=False,
    )

    assert "✓ search_text" in capsys.readouterr().out


def test_cli_does_not_repeat_an_answer_that_was_already_streamed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _display_result(
        AgentRunResult(
            run_id="streamed-answer",
            thread_id="streamed-answer",
            status="done",
            final_answer="already visible",
        ),
        verbose=False,
        answer_streamed=True,
    )

    assert "already visible" not in capsys.readouterr().out


def test_cli_shows_the_persisted_update_plan_without_verbose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = AgentPlan(
        objective="Ship durable plans.",
        revision=3,
        active_step_id="step_verify",
        steps=[
            PlanStep(
                step_id="step_store",
                title="Store the plan",
                status="completed",
            ),
            PlanStep(
                step_id="step_verify",
                title="Verify CLI exposure",
                status="in_progress",
            ),
        ],
    )
    event = PlanEvent(
        event_id="plan_event_cli",
        event_type="llm_update",
        plan_revision=plan.revision,
        message="Applied update_plan tool update.",
    )

    _display_result(
        AgentRunResult(
            run_id="visible-plan",
            thread_id="visible-plan",
            status="paused",
            plan=plan,
            plan_events=[event],
        ),
        verbose=False,
    )

    output = capsys.readouterr().out
    assert "计划 (revision 3)" in output
    assert "✓ Store the plan" in output
    assert "→ Verify CLI exposure" in output


@pytest.mark.anyio
async def test_cli_displays_canonical_tool_start_with_bounded_preview(
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _CLIToolEventDisplay().emit(
        tool_use_start(
            "read_file",
            "call_read",
            input_preview="path='src/service.py'",
        )
    )

    assert "→ read_file: path='src/service.py'" in capsys.readouterr().out


@pytest.mark.anyio
async def test_cli_displays_one_start_for_resumed_tool_call(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = _CLIToolEventDisplay()
    event = tool_use_start(
        "apply_patch",
        "call_patch",
        input_preview="file_path='notes.txt'",
    )

    await display.emit(event)
    await display.emit(event)

    assert capsys.readouterr().out.count("→ apply_patch") == 1


@pytest.mark.anyio
async def test_cli_streams_text_deltas_without_inserting_newlines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = _CLIToolEventDisplay()

    await display.emit(text_delta("hello"))
    await display.emit(text_delta(" world"))

    assert capsys.readouterr().out == "hello world"
    assert display.answer_streamed is True

    display.begin_turn()

    assert display.answer_streamed is False


@pytest.mark.anyio
async def test_cli_displays_correlated_tool_lifecycle_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = _CLIToolEventDisplay()
    start = tool_use_start(
        "read_file",
        "call_read",
        input_preview="path='src/service.py'",
    )
    result = tool_use_result(
        "read_file",
        "call_read",
        {"path": "src/service.py", "size_bytes": 420},
    )

    await display.emit(start)
    await display.emit(tool_use_progress("call_read", "reading", percent=50))
    await display.emit(result)
    await display.emit(result)

    output = capsys.readouterr().out
    assert "→ read_file: path='src/service.py'" in output
    assert "… read_file: reading (50%)" in output
    assert "✓ read_file:" in output
    assert "size_bytes" in output
    assert output.count("✓ read_file:") == 1


@pytest.mark.anyio
async def test_cli_displays_correlated_tool_error_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = _CLIToolEventDisplay()
    error = tool_use_error("call_read", "file not found")

    await display.emit(tool_use_start("read_file", "call_read"))
    await display.emit(error)
    await display.emit(error)

    output = capsys.readouterr().out
    assert "✗ read_file: file not found" in output
    assert output.count("✗ read_file:") == 1


@pytest.mark.anyio
async def test_cli_displays_patch_diff_from_existing_result_event(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = _CLIToolEventDisplay()
    event = StreamEvent(
        type=EventType.TOOL_USE_RESULT,
        data={
            "tool_name": "apply_patch",
            "tool_id": "call_patch",
            "result": {"replaced": True},
            "details": {
                "file_path": "src/example.py",
                "diff": (
                    "--- a/src/example.py\n"
                    "+++ b/src/example.py\n"
                    "@@ -1 +1 @@\n"
                    "-old\n"
                    "+new"
                ),
                "diff_truncated": False,
            },
        },
    )

    await display.emit(event)

    output = capsys.readouterr().out
    assert "✓ apply_patch:" in output
    assert "--- a/src/example.py" in output
    assert "+++ b/src/example.py" in output
    assert "-old" in output
    assert "+new" in output


@pytest.mark.anyio
async def test_cli_displays_plan_and_recovery_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = _CLIToolEventDisplay()
    plan_event = StreamEvent(
        type=EventType.PLAN_UPDATED,
        data={
            "plan": {
                "revision": 2,
                "steps": [
                    {"title": "Inspect source", "status": "completed"},
                    {"title": "Wire CLI", "status": "in_progress"},
                ],
            }
        },
    )

    await display.emit(plan_event)
    await display.emit(plan_event)
    await display.emit(recovery_event("model_retry", "attempt 2 of 3"))

    output = capsys.readouterr().out
    assert output.count("计划 (revision 2)") == 1
    assert "✓ Inspect source" in output
    assert "→ Wire CLI" in output
    assert "↻ 恢复: model_retry — attempt 2 of 3" in output


def test_builder_passes_canonical_events_to_cli_display() -> None:
    display = _CLIToolEventDisplay()

    service = build_agent_service(
        None,
        model_control_plane=_ModelRegistry(),  # type: ignore[arg-type]
        stream_sink=display,
    )

    assert service._stream_sink is display


def test_builder_installs_hidden_factory_outputs_with_find_tools() -> None:
    mcp_tools = create_mcp_tools(
        (
            MCPToolDescriptor(
                server_name="docs",
                tool_name="search",
                description="Search external documentation.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                read_only_hint=True,
            ),
        ),
        lambda _server, _tool, _arguments: {"text": "found"},
    )

    service = build_agent_service(
        None,
        model_control_plane=_ModelRegistry(),  # type: ignore[arg-type]
        mcp_tools=mcp_tools,
    )

    assert tuple(service._tool_snapshot) == (
        *RESIDENT_CODING_TOOL_NAMES,
        "mcp__docs__search",
        "find_tools",
    )
    automatic_state = service.initial_state(AgentRunRequest(task="Inspect docs."))
    disabled_state = service.initial_state(
        AgentRunRequest(
            task="Inspect docs.",
            allow_discovery_tools=False,
        )
    )
    assert tuple(automatic_state["resident_tool_names"]) == (
        *RESIDENT_CODING_TOOL_NAMES,
        "find_tools",
    )
    assert automatic_state["allow_discovery_tools"] is True
    assert tuple(disabled_state["resident_tool_names"]) == (
        RESIDENT_CODING_TOOL_NAMES
    )
    assert disabled_state["allow_discovery_tools"] is False
    assert automatic_state["active_tool_names"] == []


def test_builder_makes_skill_gateways_resident_when_skills_are_available(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    skill_tools = create_skill_tools(
        workspace,
        invoke_skill=lambda _arguments: {"success": False, "name": "missing"},
        active_skill_root=lambda _skill_id: None,
    )

    service = build_agent_service(
        workspace,
        model_control_plane=_ModelRegistry(),  # type: ignore[arg-type]
        skill_tools=skill_tools,
    )
    state = service.initial_state(AgentRunRequest(task="Use an installed skill."))

    assert tuple(state["resident_tool_names"]) == (
        *RESIDENT_CODING_TOOL_NAMES,
        "invoke_skill",
        "materialize_skill_asset",
    )
    assert "find_tools" not in state["resident_tool_names"]


def test_auto_rag_config_prefers_explicit_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_VECTOR_BACKEND", "sqlite")
    resolved = resolve_auto_rag_config(
        storage_root=Path("custom-rag"),
        vector_backend="milvus",
        vector_dsn="explicit-dsn",
        vector_namespace="explicit-ns",
        vector_collection_prefix="explicit-prefix",
    )

    assert resolved.storage_root == Path("custom-rag")
    assert resolved.vector_backend == "sqlite"
    assert resolved.vector_dsn == "explicit-dsn"
    assert resolved.explicit is True


def test_optional_rag_runtime_is_lazy_when_not_explicit() -> None:
    runtime, diagnostics = build_optional_rag_runtime(
        storage_root=Path(".rag"),
        model_alias=None,
        embedding_model_alias=None,
        reranker_model_alias=None,
        vector_backend="milvus",
        vector_dsn=None,
        vector_namespace=None,
        vector_collection_prefix=None,
        explicit=False,
    )

    assert runtime is None
    assert diagnostics == ()
