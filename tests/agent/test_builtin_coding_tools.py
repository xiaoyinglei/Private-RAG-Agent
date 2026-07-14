from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from rag.agent.tools.builtins import (
    RESIDENT_CODING_TOOL_NAMES,
    create_resident_coding_tools,
)
from rag.agent.tools.builtins import search as search_module
from rag.agent.tools.executor import ToolExecution, ToolExecutor
from rag.agent.tools.permissions import ToolExecutionContext
from rag.agent.tools.tool import Tool, ToolCall, ToolCallOrigin
from rag.agent.workspace import WorkspaceRuntime, open_workspace


def _origin(tool_name: str) -> ToolCallOrigin:
    return ToolCallOrigin(
        request_id="req_builtin",
        toolset_revision="tools_builtin_v1",
        exposed_tool_names=(tool_name,),
    )


async def _execute(
    tool: Tool,
    arguments: Mapping[str, Any],
    *,
    workspace: WorkspaceRuntime,
) -> ToolExecution:
    call = ToolCall(
        tool_call_id=f"call_{tool.definition.name}",
        tool_name=tool.definition.name,
        arguments=arguments,
        origin=_origin(tool.definition.name),
    )
    return await ToolExecutor({tool.definition.name: tool}).execute(
        call,
        context=ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
            allow_write_tools=True,
            allow_execute_tools=True,
        ),
    )


def _tools_by_name(
    workspace: WorkspaceRuntime,
    *,
    updates: list[Mapping[str, Any]] | None = None,
) -> dict[str, Tool]:
    captured = updates if updates is not None else []

    def update_plan(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        captured.append(arguments)
        return {
            "accepted": True,
            "revision": len(captured),
            "message": "plan updated",
        }

    tools = create_resident_coding_tools(
        workspace,
        plan_updater=update_plan,
    )
    return {tool.definition.name: tool for tool in tools}


def test_resident_coding_tool_baseline_is_exact_and_ordered(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path, create=True)

    tools = create_resident_coding_tools(
        workspace,
        plan_updater=lambda _arguments: {
            "accepted": True,
            "revision": 1,
            "message": "ok",
        },
    )

    assert RESIDENT_CODING_TOOL_NAMES == (
        "list_files",
        "search_text",
        "read_file",
        "apply_patch",
        "run_command",
        "update_plan",
    )
    assert tuple(tool.definition.name for tool in tools) == RESIDENT_CODING_TOOL_NAMES
    assert all(isinstance(tool, Tool) for tool in tools)
    assert len({tool.definition.name for tool in tools}) == len(tools)
    assert "write_file" not in {tool.definition.name for tool in tools}
    for tool in tools:
        assert len(tool.definition.description) >= 80
        properties = tool.definition.input_schema.get("properties", {})
        assert isinstance(properties, Mapping)
        assert properties
        assert all(
            isinstance(schema, Mapping) and schema.get("description")
            for schema in properties.values()
        )


@pytest.mark.anyio
async def test_filesystem_tools_list_read_patch_and_expose_changes_immediately(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    source = workspace.root / "src"
    source.mkdir()
    target = source / "example.py"
    target.write_text("before\nneedle_one()\nafter\n", encoding="utf-8")
    tools = _tools_by_name(workspace)

    listed = await _execute(
        tools["list_files"],
        {"path": "src", "glob": "*.py", "limit": 10},
        workspace=workspace,
    )
    read_before = await _execute(
        tools["read_file"],
        {"path": "src/example.py", "max_bytes": 1000},
        workspace=workspace,
    )
    patched = await _execute(
        tools["apply_patch"],
        {
            "file_path": "src/example.py",
            "old_string": "needle_one",
            "new_string": "fresh_symbol",
        },
        workspace=workspace,
    )
    old_search = await _execute(
        tools["search_text"],
        {"pattern": "needle_one", "path": "src", "glob": "*.py"},
        workspace=workspace,
    )
    new_search = await _execute(
        tools["search_text"],
        {"pattern": "fresh_symbol", "path": "src", "glob": "*.py"},
        workspace=workspace,
    )

    assert listed.result.is_error is False
    assert listed.result.structured_content is not None
    assert [entry["path"] for entry in listed.result.structured_content["entries"]] == [
        "src/example.py"
    ]
    assert read_before.result.structured_content is not None
    assert read_before.result.structured_content["content"] == (
        "before\nneedle_one()\nafter\n"
    )
    assert patched.result.structured_content is not None
    assert patched.result.structured_content["replaced"] is True
    assert old_search.result.structured_content is not None
    assert old_search.result.structured_content["matches"] == ()
    assert new_search.result.structured_content is not None
    assert new_search.result.structured_content["total_matches"] == 1
    assert target.read_text(encoding="utf-8") == (
        "before\nfresh_symbol()\nafter\n"
    )


@pytest.mark.anyio
async def test_read_file_default_window_bounds_model_context(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    (workspace.root / "large.txt").write_text("x" * 20_000, encoding="utf-8")

    execution = await _execute(
        _tools_by_name(workspace)["read_file"],
        {"path": "large.txt"},
        workspace=workspace,
    )

    assert execution.result.is_error is False
    assert execution.result.structured_content is not None
    assert len(execution.result.structured_content["content"]) == 16_000
    assert execution.result.structured_content["truncated"] is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("arguments", "error_code"),
    [
        (
            {
                "file_path": "missing.txt",
                "old_string": "before",
                "new_string": "after",
            },
            "file_not_found",
        ),
        (
            {
                "file_path": "notes.txt",
                "old_string": "absent",
                "new_string": "after",
            },
            "old_string_not_found",
        ),
        (
            {
                "file_path": "notes.txt",
                "old_string": "same",
                "new_string": "after",
            },
            "old_string_not_unique",
        ),
    ],
)
async def test_apply_patch_non_effect_is_a_canonical_tool_error(
    tmp_path: Path,
    arguments: Mapping[str, Any],
    error_code: str,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    (workspace.root / "notes.txt").write_text("same same", encoding="utf-8")

    execution = await _execute(
        _tools_by_name(workspace)["apply_patch"],
        arguments,
        workspace=workspace,
    )

    assert execution.result.is_error is True
    assert execution.result.error_code == error_code
    assert execution.result.error_message
    assert execution.result.structured_content is not None
    assert execution.result.structured_content["replaced"] is False


@pytest.mark.anyio
async def test_search_text_supports_literal_regex_path_glob_context_and_limits(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    source = workspace.root / "src"
    source.mkdir()
    (source / "one.py").write_text(
        "context before\nneedle_alpha()\ncontext after\n",
        encoding="utf-8",
    )
    (source / "two.py").write_text("needle_beta()\n", encoding="utf-8")
    (source / "notes.md").write_text("needle_markdown\n", encoding="utf-8")
    search = _tools_by_name(workspace)["search_text"]

    literal = await _execute(
        search,
        {
            "pattern": "needle_alpha",
            "path": "src/one.py",
            "glob": "*.py",
            "context_lines": 1,
            "max_results": 10,
        },
        workspace=workspace,
    )
    regex = await _execute(
        search,
        {
            "pattern": r"needle_[a-z]+\(",
            "path": "src",
            "glob": "*.py",
            "regex": True,
            "max_results": 10,
        },
        workspace=workspace,
    )
    limited = await _execute(
        search,
        {
            "pattern": "needle_",
            "path": "src",
            "glob": "*.py",
            "max_results": 1,
        },
        workspace=workspace,
    )
    invalid_regex = await _execute(
        search,
        {"pattern": "(", "path": "src", "regex": True},
        workspace=workspace,
    )

    assert literal.result.structured_content is not None
    literal_matches = literal.result.structured_content["matches"]
    assert len(literal_matches) == 1
    assert literal_matches[0]["file_path"] == "src/one.py"
    assert literal_matches[0]["line_number"] == 2
    assert literal_matches[0]["match_start"] == 0
    assert literal_matches[0]["context_before"] == ("context before",)
    assert literal_matches[0]["context_after"] == ("context after",)

    assert regex.result.structured_content is not None
    assert [
        match["file_path"] for match in regex.result.structured_content["matches"]
    ] == ["src/one.py", "src/two.py"]
    assert limited.result.structured_content is not None
    assert limited.result.structured_content["total_matches"] == 1
    assert limited.result.structured_content["truncated"] is True
    assert invalid_regex.result.error_code == "invalid_arguments"


@pytest.mark.anyio
async def test_search_text_does_not_follow_symlinks_or_escape_workspace(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    source = workspace.root / "src"
    source.mkdir()
    (source / "inside.py").write_text("safe_marker\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("outside_marker\n", encoding="utf-8")
    try:
        (source / "external.py").symlink_to(outside)
        (workspace.root / "linked_src").symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    search = _tools_by_name(workspace)["search_text"]

    safe = await _execute(
        search,
        {"pattern": "marker", "path": ".", "glob": "*.py"},
        workspace=workspace,
    )
    escaped = await _execute(
        search,
        {"pattern": "outside", "path": "../outside.py"},
        workspace=workspace,
    )

    assert safe.result.structured_content is not None
    assert [
        match["file_path"] for match in safe.result.structured_content["matches"]
    ] == ["src/inside.py"]
    assert escaped.result.error_code == "workspace_escape"


@pytest.mark.anyio
async def test_update_plan_uses_the_injected_state_callback(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path, create=True)
    updates: list[Mapping[str, Any]] = []
    update_plan = _tools_by_name(workspace, updates=updates)["update_plan"]

    execution = await _execute(
        update_plan,
        {
            "explanation": "Show the next implementation checkpoint.",
            "plan": [
                {"step": "Implement the resident tools", "status": "in_progress"},
                {"step": "Run focused tests", "status": "pending"},
            ],
        },
        workspace=workspace,
    )

    assert execution.result.is_error is False
    assert execution.result.structured_content == {
        "accepted": True,
        "revision": 1,
        "message": "plan updated",
    }
    assert len(updates) == 1
    assert updates[0]["plan"][0]["step"] == "Implement the resident tools"


@pytest.mark.anyio
async def test_run_command_returns_bounded_structured_process_output(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    tools = _tools_by_name(workspace)

    execution = await _execute(
        tools["run_command"],
        {
            "command": "printf 'hello'; printf 'warning' >&2",
            "working_dir": ".",
            "timeout_seconds": 1,
        },
        workspace=workspace,
    )

    assert execution.result.is_error is False
    assert execution.result.structured_content is not None
    assert execution.result.structured_content["stdout"] == "hello"
    assert execution.result.structured_content["stderr"] == "warning"
    assert execution.result.structured_content["exit_code"] == 0
    assert execution.result.structured_content["timed_out"] is False


def test_search_builtin_has_no_embedding_or_retrieval_import_dependency() -> None:
    module_path = Path(search_module.__file__ or "")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    forbidden_prefixes = (
        "rag.retrieval",
        "rag.ingestion",
        "chromadb",
        "faiss",
        "sentence_transformers",
    )
    assert not any(
        module.startswith(forbidden_prefixes) for module in imported_modules
    )
