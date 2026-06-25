"""B2b: Declarative tool batch — integration tests with real SDK and reader."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from rag.agent.core.tool_batch_reader import (
    clean_batch_file,
    read_tool_batch,
)


class TestToolBatchReader:
    """Read and parse tool_calls.jsonl using real reader module."""

    def test_read_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            decls = read_tool_batch(d)
            assert decls == []

    def test_read_valid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            batch_file = Path(d) / "tool_calls.jsonl"
            batch_file.write_text(
                '{"tool_name":"search_knowledge","arguments":{"query":"test","top_k":3}}\n'
                '{"tool_name":"search_text","arguments":{"pattern":"TODO","path":"."}}\n'
            )
            decls = read_tool_batch(d)
            assert len(decls) == 2
            assert decls[0].tool_name == "search_knowledge"
            assert decls[0].arguments == {"query": "test", "top_k": 3}

    def test_respects_max_batch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            batch_file = Path(d) / "tool_calls.jsonl"
            lines = [json.dumps({"tool_name": f"t{i}", "arguments": {}}) for i in range(20)]
            batch_file.write_text("\n".join(lines))
            decls = read_tool_batch(d, max_batch=5)
            assert len(decls) == 5

    def test_skips_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            batch_file = Path(d) / "tool_calls.jsonl"
            batch_file.write_text('not json\n{"tool_name":"valid","arguments":{}}\n')
            decls = read_tool_batch(d)
            assert len(decls) == 1

    def test_clean_batch_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            batch_file = Path(d) / "tool_calls.jsonl"
            batch_file.write_text("test")
            clean_batch_file(d)
            assert not batch_file.exists()


class TestRealToolSDK:
    """Test the actual tool_sdk.py module."""

    def test_sdk_module_loads(self) -> None:
        from rag.agent.tools.tool_sdk import tools

        assert hasattr(tools, "declare")
        assert hasattr(tools, "list_available")

    def test_tools_declare_writes_jsonl(self) -> None:
        """tools.declare() writes to tool_calls.jsonl."""
        with tempfile.TemporaryDirectory() as d:
            os.environ["AGENT_SCRATCH_DIR"] = d
            try:
                # Reset module-level counter by re-importing
                from importlib import reload

                from rag.agent.tools import tool_sdk

                reload(tool_sdk)

                result = tool_sdk.tools.declare("search_knowledge", query="test", top_k=5)
                assert result.get("declared") == "search_knowledge"

                batch_file = Path(d) / "tool_calls.jsonl"
                assert batch_file.exists()
                content = batch_file.read_text().strip()
                assert "search_knowledge" in content
                assert "test" in content
            finally:
                os.environ.pop("AGENT_SCRATCH_DIR", None)


class TestRunPythonCodePath:
    """Verify run_python(code=...) actually works."""

    def test_run_python_code_executes(self) -> None:
        from rag.agent.primitive_ops import PrimitiveOps, RunPythonInput
        from rag.agent.workspace import create_temp_workspace

        ws = create_temp_workspace()
        ops = PrimitiveOps(workspace=ws)

        result = ops.run_python(RunPythonInput(code="print('hello from code path')"))
        assert result.exit_code == 0
        assert "hello from code path" in result.stdout

    def test_run_python_code_with_sdk(self) -> None:
        from rag.agent.primitive_ops import PrimitiveOps, RunPythonInput
        from rag.agent.workspace import create_temp_workspace

        ws = create_temp_workspace()
        ops = PrimitiveOps(workspace=ws)

        code = """
import os
# The SDK is prepended — tools should be available
print("SDK test")
batch_path = os.path.join(os.environ.get('AGENT_SCRATCH_DIR', '.'), 'tool_calls.jsonl')
print(f"batch_path={batch_path}")
"""
        result = ops.run_python(RunPythonInput(code=code))
        assert result.exit_code == 0
        assert "SDK test" in result.stdout
