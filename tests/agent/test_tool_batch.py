"""B2b: Declarative tool batch — tool_sdk + tool_batch_reader tests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from rag.agent.capabilities.catalog import DeferredToolStore
from rag.agent.core.tool_batch_reader import (
    clean_batch_file,
    read_tool_batch,
    validate_declaration,
    ToolBatchDeclaration,
)
from rag.agent.tools.registry import ToolRegistry


class TestToolBatchReader:
    """Read and parse tool_calls.jsonl."""

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
            assert decls[1].tool_name == "search_text"

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
            assert decls[0].tool_name == "valid"

    def test_clean_batch_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            batch_file = Path(d) / "tool_calls.jsonl"
            batch_file.write_text("test")
            clean_batch_file(d)
            assert not batch_file.exists()


class TestToolSDK:
    """The tools.declare() function in the SDK preamble."""

    def test_tools_declare_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            os.environ["AGENT_SCRATCH_DIR"] = d
            os.environ["AGENT_MAX_BATCH_SIZE"] = "10"
            try:
                # Use the inline SDK fallback (no file)
                sdk = """
import json, os, pathlib
class _TD:
    def __init__(self): self._c=0
    def declare(self, n, **a):
        if self._c>=int(os.environ.get('AGENT_MAX_BATCH_SIZE','10')): return {'d':False}
        b=os.path.join(os.environ.get('AGENT_SCRATCH_DIR','.'),'tool_calls.jsonl')
        pathlib.Path(b).parent.mkdir(parents=True,exist_ok=True)
        with open(b,'a') as f: f.write(json.dumps({'tool_name':n,'arguments':a},default=str)+'\\n')
        self._c+=1; return {'declared':n,'seq':self._c}
    def list_available(self): return os.environ.get('AGENT_AVAILABLE_TOOLS','').split(',')
tools = _TD()
"""
                exec(sdk)
                from types import SimpleNamespace

                # tools is now in local scope... actually no, exec'd in isolated scope
                # Let's just test the file was written
                ns: dict = {}
                exec(sdk, ns)
                tools_obj = ns["tools"]
                result = tools_obj.declare("search_knowledge", query="test", top_k=5)
                assert result["declared"] == "search_knowledge"

                batch_file = Path(d) / "tool_calls.jsonl"
                assert batch_file.exists()
                content = batch_file.read_text().strip()
                assert "search_knowledge" in content
                assert "test" in content
            finally:
                os.environ.pop("AGENT_SCRATCH_DIR", None)
                os.environ.pop("AGENT_MAX_BATCH_SIZE", None)

    def test_batch_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            os.environ["AGENT_SCRATCH_DIR"] = d
            os.environ["AGENT_MAX_BATCH_SIZE"] = "2"
            try:
                sdk = """
import json, os, pathlib
class _TD:
    def __init__(self): self._c=0
    def declare(self, n, **a):
        if self._c>=int(os.environ.get('AGENT_MAX_BATCH_SIZE','10')): return {'d':False}
        b=os.path.join(os.environ.get('AGENT_SCRATCH_DIR','.'),'tool_calls.jsonl')
        pathlib.Path(b).parent.mkdir(parents=True,exist_ok=True)
        with open(b,'a') as f: f.write(json.dumps({'tool_name':n,'arguments':a},default=str)+'\\n')
        self._c+=1; return {'declared':n,'seq':self._c}
tools = _TD()
"""
                ns: dict = {}
                exec(sdk, ns)
                t = ns["tools"]
                assert t.declare("a")["declared"] == "a"
                assert t.declare("b")["declared"] == "b"
                result = t.declare("c")
                # Fallback SDK uses short 'd' key; full SDK uses 'declared'
                assert not result.get("declared", result.get("d", True))
            finally:
                os.environ.pop("AGENT_SCRATCH_DIR", None)
                os.environ.pop("AGENT_MAX_BATCH_SIZE", None)
