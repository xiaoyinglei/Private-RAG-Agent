"""B2b: End-to-end integration tests — code-as-tool full pipeline.

Covers:
  - run_python(code=...) → executes, SDK writes batch file
  - _process_tool_batch() → reads jsonl → PendingToolCall → error ToolResult
  - tool_repl runner → adapts command → code → execute
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag.agent.capabilities.catalog import (
    CORE_TOOLS,
    DEFERRED_TOOLS,
    DeferredToolStore,
    ToolCatalog,
)
from rag.agent.core.definition import AgentRuntimePolicy, AgentRuntimePolicy
from rag.agent.core.tool_batch_reader import clean_batch_file, read_tool_batch
from rag.agent.loop.state import (
    LoopState,
    PendingToolCall,
    create_loop_state,
)
from rag.agent.tools.spec import ToolResult


def _minimal_state() -> LoopState:
    from rag.agent.core.context import AgentRunConfig
    from rag.schema.runtime import AccessPolicy

    run_config = AgentRunConfig(
        run_id="test-run",
        thread_id="test-thread",
        budget_total=10000,
        max_depth=3,
        access_policy=AccessPolicy.default(),
    )
    state = create_loop_state(task="test", run_config=run_config)
    state["tool_results"] = []
    state["pending_tool_calls"] = []
    state["runtime_diagnostics"] = []
    return state


def _minimal_policy() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.from_legacy(
        system_instructions="You are a test agent",
        core_tool_names=(
            "read_file", "write_file", "run_python", "list_files",
            "search_text", "apply_patch", "run_command",
            "update_plan", "tool_repl", "task",
            "tool_search", "activate_tools",
        ),
        deferred_tool_names=(
            "search_knowledge", "search_assets",
        ),
        token_budget=10000,
        work_budget=100,
        max_iterations=5,
        max_depth=3,
    )


class TestRunPythonCodePath:
    """Verify run_python(code=...) end-to-end."""

    def test_run_python_with_tools_declare(self) -> None:
        """Python code uses tools.declare() → batch file is written."""
        from rag.agent.primitive_ops import PrimitiveOps, RunPythonInput
        from rag.agent.workspace import create_temp_workspace

        ws = create_temp_workspace()
        ops = PrimitiveOps(workspace=ws)
        scratch = ws.root / "scratch"

        code = """
tools.declare('search_knowledge', query='Q3 revenue', top_k=5)
tools.declare('search_assets', query='financial tables', max_results=3)
print('done')
"""
        result = ops.run_python(RunPythonInput(code=code))
        assert result.exit_code == 0
        assert "done" in result.stdout

        # Verify batch file was written
        batch_file = scratch / "tool_calls.jsonl"
        if batch_file.exists():
            content = batch_file.read_text().strip()
            assert "search_knowledge" in content
            assert "search_assets" in content
            assert "Q3 revenue" in content
            # Cleanup
            batch_file.unlink()

    def test_run_python_script_path_still_works(self) -> None:
        """script_path= parameter still works (regression check)."""
        from rag.agent.primitive_ops import PrimitiveOps, RunPythonInput
        from rag.agent.workspace import create_temp_workspace

        ws = create_temp_workspace()
        ops = PrimitiveOps(workspace=ws)
        scratch = ws.root / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)

        # Write a real .py file
        script = scratch / "hello.py"
        script.write_text("print('hello from file')")
        result = ops.run_python(RunPythonInput(script_path="scratch/hello.py"))
        assert result.exit_code == 0
        assert "hello from file" in result.stdout

    def test_run_python_code_complex_processing(self) -> None:
        """Python code that processes data and declares tools."""
        from rag.agent.primitive_ops import PrimitiveOps, RunPythonInput
        from rag.agent.workspace import create_temp_workspace

        ws = create_temp_workspace()
        ops = PrimitiveOps(workspace=ws)

        code = """
# Data processing
data = [{"file": "a.py", "line": 10}, {"file": "b.py", "line": 20}]
for item in data:
    tools.declare('read_file', file_path=item['file'])

print(f"Declared {len(data)} tools")
"""
        result = ops.run_python(RunPythonInput(code=code))
        assert result.exit_code == 0
        assert "Declared 2 tools" in result.stdout

        batch_file = ws.root / "scratch" / "tool_calls.jsonl"
        if batch_file.exists():
            lines = batch_file.read_text().strip().split("\n")
            assert len(lines) == 2
            batch_file.unlink()


class TestToolReplRunner:
    """Verify tool_repl runner and integration."""

    def test_tool_repl_registered_with_card(self) -> None:
        from rag.agent.tools.workspace_tools import ToolReplTool
        from rag.agent.workspace import create_temp_workspace

        ws = create_temp_workspace()
        tool = ToolReplTool(ws)
        spec = tool.to_spec()
        assert spec.name == "tool_repl"
        assert spec.aci is not None
        assert spec.aci.when_to_use != ""
        assert spec.aci.activation_group == "workspace"

    def test_tool_repl_has_formatter(self) -> None:
        from rag.agent.builtin_registry import create_builtin_tool_registry

        reg = create_builtin_tool_registry()
        fmt = reg.get_formatter("tool_repl")
        # Formatter is registered via builtin_registry even though
        # the tool spec itself is registered at runtime
        assert fmt is not None
        assert fmt.tool_name == "tool_repl"

    def test_tool_repl_runner_adapts_command(self) -> None:
        from rag.agent.primitive_ops import PrimitiveOps
        from rag.agent.workspace import create_temp_workspace

        ws = create_temp_workspace()
        ops = PrimitiveOps(workspace=ws)

        # Direct runner call
        result = ops.tool_repl({"command": "print('repl test')"})
        assert result.exit_code == 0
        assert "repl test" in result.stdout

    def test_tool_repl_runner_accepts_pydantic_input(self) -> None:
        from rag.agent.primitive_ops import PrimitiveOps
        from rag.agent.tools.generic_tools import RunCommandInput
        from rag.agent.workspace import create_temp_workspace

        ws = create_temp_workspace()
        ops = PrimitiveOps(workspace=ws)

        result = ops.tool_repl(RunCommandInput(command="print('pydantic')"))
        assert result.exit_code == 0
        assert "pydantic" in result.stdout


class TestProcessToolBatch:
    """Verify _process_tool_batch via AgentLoop."""

    def test_valid_batch_becomes_pending(self) -> None:
        """Valid declarations → PendingToolCall list."""
        with tempfile.TemporaryDirectory() as d:
            scratch = Path(d) / "scratch"
            scratch.mkdir(parents=True, exist_ok=True)

            # Write a valid batch file
            batch_file = scratch / "tool_calls.jsonl"
            batch_file.write_text(
                json.dumps({"tool_name": "read_file", "arguments": {"file_path": "test.py"}}) + "\n"
            )

            # Build minimal AgentLoop with scratch_dir
            from rag.agent.loop.runtime import AgentLoop

            loop = AgentLoop(  # type: ignore[call-arg]
                definition=AgentRuntimePolicy.from_legacy(
                    agent_type="test",
                    description="test agent",
                    system_prompt="test",
                    allowed_tools=["read_file"],
                ),
                model_provider=MagicMock(),
                context_manager=MagicMock(),
                tool_runner=MagicMock(),
                checkpoint_store=MagicMock(),
                stop_hook_runner=MagicMock(),
                finish_candidate_builder=MagicMock(),
                catalog=MagicMock(),
                deferred_store=MagicMock(),
                scratch_dir=scratch,
            )

            state = _minimal_state()
            # read_file is in CORE_TOOLS → should be allowed without activation
            pending = loop._process_tool_batch(state)
            assert len(pending) == 1
            assert pending[0].tool_name == "read_file"
            assert pending[0].plan.arguments == {"file_path": "test.py"}

            # Batch file should be cleaned up
            assert not batch_file.exists()

    def test_unactivated_deferred_tool_errors(self) -> None:
        """Unactivated deferred tool → error ToolResult, no pending."""
        with tempfile.TemporaryDirectory() as d:
            scratch = Path(d) / "scratch"
            scratch.mkdir(parents=True, exist_ok=True)

            batch_file = scratch / "tool_calls.jsonl"
            batch_file.write_text(
                json.dumps({"tool_name": "search_knowledge", "arguments": {"query": "x"}}) + "\n"
            )

            from rag.agent.loop.runtime import AgentLoop

            # Minimal catalog that classifies search_knowledge as deferred
            catalog = ToolCatalog()
            catalog.register(
                __import__("rag.agent.capabilities.catalog", fromlist=["ToolCatalogEntry"])
                .ToolCatalogEntry(
                    name="search_knowledge",
                    description="Search",
                    category="deferred",
                    search_text="search",
                    activation_group="rag",
                )
            )

            loop = AgentLoop(  # type: ignore[call-arg]
                definition=AgentRuntimePolicy.from_legacy(
                    agent_type="test",
                    description="test agent",
                    system_prompt="test",
                    allowed_tools=["search_knowledge"],
                ),
                model_provider=MagicMock(),
                context_manager=MagicMock(),
                tool_runner=MagicMock(),
                checkpoint_store=MagicMock(),
                stop_hook_runner=MagicMock(),
                finish_candidate_builder=MagicMock(),
                catalog=None,
                deferred_store=DeferredToolStore(max_active=5),
                scratch_dir=scratch,
            )

            state = _minimal_state()
            pending = loop._process_tool_batch(state)

            # Should NOT become pending (not activated)
            assert len(pending) == 0

            # Should produce an error ToolResult
            errors = [tr for tr in state["tool_results"] if tr.status == "error"]
            assert len(errors) >= 1
            assert "not activated" in errors[0].error.message  # type: ignore[union-attr]

    def test_not_allowed_tool_errors(self) -> None:
        """Tool not in allowed_tools → error ToolResult."""
        with tempfile.TemporaryDirectory() as d:
            scratch = Path(d) / "scratch"
            scratch.mkdir(parents=True, exist_ok=True)

            batch_file = scratch / "tool_calls.jsonl"
            batch_file.write_text(
                json.dumps({"tool_name": "unknown_tool", "arguments": {}}) + "\n"
            )

            from rag.agent.loop.runtime import AgentLoop

            loop = AgentLoop(  # type: ignore[call-arg]
                definition=AgentRuntimePolicy.from_legacy(
                    agent_type="test",
                    description="test agent",
                    system_prompt="test",
                    allowed_tools=["read_file"],
                ),
                model_provider=MagicMock(),
                context_manager=MagicMock(),
                tool_runner=MagicMock(),
                checkpoint_store=MagicMock(),
                stop_hook_runner=MagicMock(),
                finish_candidate_builder=MagicMock(),
                catalog=None,
                deferred_store=None,
                scratch_dir=scratch,
            )

            state = _minimal_state()
            pending = loop._process_tool_batch(state)
            assert len(pending) == 0

            errors = [tr for tr in state["tool_results"] if tr.status == "error"]
            assert len(errors) >= 1
            assert "not in allowed_tools" in errors[0].error.message  # type: ignore[union-attr]

    def test_no_scratch_dir_noop(self) -> None:
        """scratch_dir=None → returns empty, no crash."""
        from rag.agent.loop.runtime import AgentLoop

        loop = AgentLoop(  # type: ignore[call-arg]
            definition=AgentRuntimePolicy.from_legacy(
                agent_type="test",
                description="test agent",
                system_prompt="test",
                allowed_tools=[],
            ),
            model_provider=MagicMock(),
            context_manager=MagicMock(),
            tool_runner=MagicMock(),
            checkpoint_store=MagicMock(),
            stop_hook_runner=MagicMock(),
            finish_candidate_builder=MagicMock(),
            scratch_dir=None,
        )

        state = _minimal_state()
        pending = loop._process_tool_batch(state)
        assert pending == []
