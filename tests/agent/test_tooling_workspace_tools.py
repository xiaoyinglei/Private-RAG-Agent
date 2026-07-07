from __future__ import annotations

import asyncio
from pathlib import Path

from rag.agent.tooling import (
    ToolCall,
    ToolExecutor,
    ToolRegistry,
    ToolSurfacePolicy,
    ToolSurfaceRequest,
    install_minimal_workspace_tools,
)
from rag.agent.workspace import WorkspaceRuntime


def _workspace(tmp_path: Path) -> WorkspaceRuntime:
    workspace = WorkspaceRuntime(root=tmp_path, is_temporary=False)
    workspace.initialize()
    return workspace


def test_minimal_workspace_tools_can_be_selectively_surfaced(tmp_path: Path) -> None:
    registry = ToolRegistry()
    install_minimal_workspace_tools(registry, _workspace(tmp_path))

    direct_answer = ToolSurfacePolicy().decide(
        registry,
        ToolSurfaceRequest(force_empty=True),
    )
    search_request = ToolSurfacePolicy().decide(
        registry,
        ToolSurfaceRequest(
            requested_tool_names=["search_text", "read_file", "list_files"],
        ),
    )

    assert direct_answer.sent_schema_names == []
    assert [spec.name for spec in search_request.visible_tools] == [
        "search_text",
        "read_file",
        "list_files",
    ]
    assert "tool_search" not in search_request.sent_schema_names
    assert "activate_tools" not in search_request.sent_schema_names


def test_missing_file_returns_structured_recoverable_error(tmp_path: Path) -> None:
    registry = ToolRegistry()
    install_minimal_workspace_tools(registry, _workspace(tmp_path))
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="call_1",
                name="read_file",
                arguments={"path": "README_DOES_NOT_EXIST.md"},
            ),
            sent_schema_names=["read_file"],
        )
    )

    assert result.ok is False
    assert result.recoverable is True
    assert result.error_code == "file_not_found"
    assert "README_DOES_NOT_EXIST.md" in result.content


def test_read_file_returns_structured_data_and_meta(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace.scratch / "note.txt").write_text("hello", encoding="utf-8")
    registry = ToolRegistry()
    install_minimal_workspace_tools(registry, workspace)
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="call_1",
                name="read_file",
                arguments={"path": "scratch/note.txt"},
            ),
            sent_schema_names=["read_file"],
        )
    )

    assert result.ok is True
    assert result.data["path"] == "scratch/note.txt"
    assert result.data["content"] == "hello"
    assert result.data["truncated"] is False
    assert result.data["size_bytes"] == 5
    assert result.data["_meta"]["truncated"] is False
    assert executor.traces[-1].output_size_bytes == result.data["_meta"]["size_bytes"]


def test_search_text_returns_structured_data_and_meta(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace.scratch / "note.txt").write_text("alpha\nneedle\n", encoding="utf-8")
    registry = ToolRegistry()
    install_minimal_workspace_tools(registry, workspace)
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="call_1",
                name="search_text",
                arguments={"pattern": "needle", "path": "scratch"},
            ),
            sent_schema_names=["search_text"],
        )
    )

    assert result.ok is True
    assert result.data["matches"][0]["file_path"] == "scratch/note.txt"
    assert result.data["matches"][0]["line_content"] == "needle"
    assert result.data["total_matches"] == 1
    assert result.data["truncated"] is False
    assert result.data["_meta"]["size_bytes"] > 0


def test_run_command_uses_allowlist_and_records_output_trace(tmp_path: Path) -> None:
    registry = ToolRegistry()
    install_minimal_workspace_tools(
        registry,
        _workspace(tmp_path),
        allowed_commands={"echo"},
    )
    executor = ToolExecutor(registry, allow_execute_tools=True)

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="call_1",
                name="run_command",
                arguments={
                    "command": "echo hello",
                    "working_dir": ".",
                    "timeout_seconds": 3,
                },
            ),
            sent_schema_names=["run_command"],
        )
    )

    assert result.ok is True
    assert result.data["stdout"].strip() == "hello"
    assert result.data["timed_out"] is False
    assert result.data["_meta"]["truncated"] is False
    assert executor.traces[-1].tool_name == "run_command"
    assert executor.traces[-1].status == "ok"
    assert executor.traces[-1].output_size_bytes == result.data["_meta"]["size_bytes"]


def test_run_python_returns_structured_data_and_meta(tmp_path: Path) -> None:
    registry = ToolRegistry()
    workspace = _workspace(tmp_path)
    (workspace.scratch / "hello.py").write_text("print('hello')", encoding="utf-8")
    install_minimal_workspace_tools(registry, workspace)
    executor = ToolExecutor(registry, allow_execute_tools=True)

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="call_1",
                name="run_python",
                arguments={"script_path": "scratch/hello.py", "timeout_seconds": 5},
            ),
            sent_schema_names=["run_python"],
        )
    )

    assert result.ok is True
    assert result.data["ok"] is True
    assert result.data["stdout"].strip() == "hello"
    assert result.data["stdout_truncated"] is False
    assert result.data["_meta"]["truncated"] is False
    assert executor.traces[-1].can_use_tool_decision == "allow"


def test_write_file_returns_structured_data_and_meta(tmp_path: Path) -> None:
    registry = ToolRegistry()
    workspace = _workspace(tmp_path)
    install_minimal_workspace_tools(registry, workspace)
    executor = ToolExecutor(registry, allow_write_tools=True)

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="call_1",
                name="write_file",
                arguments={"path": "scratch/out.txt", "content": "saved"},
            ),
            sent_schema_names=["write_file"],
        )
    )

    assert result.ok is True
    assert result.data["path"] == "scratch/out.txt"
    assert result.data["size_bytes"] == 5
    assert result.data["_meta"]["truncated"] is False
    assert (workspace.scratch / "out.txt").read_text(encoding="utf-8") == "saved"


def test_run_command_rejects_non_allowlisted_command(tmp_path: Path) -> None:
    registry = ToolRegistry()
    install_minimal_workspace_tools(
        registry,
        _workspace(tmp_path),
        allowed_commands={"echo"},
    )
    executor = ToolExecutor(registry, allow_execute_tools=True)

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="call_1",
                name="run_command",
                arguments={"command": "pwd", "working_dir": "."},
            ),
            sent_schema_names=["run_command"],
        )
    )

    assert result.ok is False
    assert result.recoverable is True
    assert result.error_code == "command_not_allowed"
    assert executor.traces[-1].can_use_tool_decision == "allow"


def test_run_command_requires_entry_execute_allow_flag(tmp_path: Path) -> None:
    registry = ToolRegistry()
    install_minimal_workspace_tools(
        registry,
        _workspace(tmp_path),
        allowed_commands={"echo"},
    )
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="call_1",
                name="run_command",
                arguments={"command": "echo hello", "working_dir": "."},
            ),
            sent_schema_names=["run_command"],
        )
    )

    assert result.ok is False
    assert result.recoverable is True
    assert result.error_code == "permission_required"
    assert result.data["can_use_tool"]["decision"] == "ask"
    assert executor.traces[-1].can_use_tool_decision == "ask"
