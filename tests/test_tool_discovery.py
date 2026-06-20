"""Provider-Agnostic Tool Discovery Protocol — closed-loop tests.

Tests verify the full discovery lifecycle:
1. deferred tools not in initial tools=
2. tool_search returns candidates
3. activate_tools makes tools available next turn
4. cannot activate unsearched tools
5. runtime does not auto-activate based on user text
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.capabilities.catalog import (
    CORE_TOOLS,
    DEFERRED_TOOLS,
    DeferredToolStore,
    SearchCandidate,
    ToolCatalog,
    ToolCatalogEntry,
    flatten_schema,
    resolve_visible_tools,
)
from rag.agent.capabilities.tool_search import (
    ActivateToolsInput,
    ActivateToolsOutput,
    ToolSearchInput,
    ToolSearchOutput,
    execute_activate_tools,
    execute_tool_search,
)
from rag.agent.tools.spec import ToolPermissions, ToolSpec


# ── Helpers ──


class DummyInput(BaseModel):
    query: str = ""


class DummyOutput(BaseModel):
    results: list = []


def _make_spec(name: str, description: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_model=DummyInput,
        output_model=DummyOutput,
        error_model=DummyOutput,
        permissions=ToolPermissions(),
        timeout_seconds=5,
    )


def _build_test_catalog() -> ToolCatalog:
    """Build a catalog with core and deferred tools."""
    catalog = ToolCatalog()

    # Core tools (always visible)
    for name in ["tool_search", "activate_tools", "read_file", "write_file"]:
        catalog.register(ToolCatalogEntry(
            name=name,
            description=f"{name} tool",
            category="core",
            search_text=name,
        ))

    # Deferred tools (hidden until activated)
    catalog.register(ToolCatalogEntry(
        name="excel_analyze",
        description="Read and analyze xlsx workbooks",
        category="deferred",
        search_text="excel_analyze read and analyze xlsx workbooks file_path string path to xlsx workbook sheet_name string sheet to analyze",
        schema_text="file_path string path to xlsx workbook sheet_name string sheet to analyze",
        examples=("Analyze workbook sales.xlsx",),
        tags=("spreadsheet", "excel"),
    ))
    catalog.register(ToolCatalogEntry(
        name="vector_search",
        description="Search documents using vector similarity",
        category="deferred",
        search_text="vector_search search documents using vector similarity query string search",
        schema_text="query string",
    ))
    catalog.register(ToolCatalogEntry(
        name="pdf_extract",
        description="Extract text and tables from PDF files",
        category="deferred",
        search_text="pdf_extract extract text and tables from pdf files file_path string path to pdf",
        schema_text="file_path string path to pdf",
    ))

    return catalog


# ── Test 1: deferred tools not in initial tools= ──


class TestDeferredNotVisible:
    def test_deferred_tools_not_in_initial_visible(self):
        """deferred 工具初始不出现在可见工具列表中。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=10)

        allowed = [
            "tool_search", "activate_tools", "read_file", "write_file",
            "excel_analyze", "vector_search", "pdf_extract",
        ]
        visible = resolve_visible_tools(allowed, catalog=catalog, store=store)

        # Core tools are visible
        assert "tool_search" in visible
        assert "read_file" in visible
        # Deferred tools are NOT visible
        assert "excel_analyze" not in visible
        assert "vector_search" not in visible
        assert "pdf_extract" not in visible


# ── Test 2: tool_search returns candidates ──


class TestToolSearchReturnsCandidates:
    def test_search_returns_candidates_with_reason(self):
        """tool_search 返回候选列表，包含 name/description/reason。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=10)

        result = execute_tool_search(
            "analyze excel spreadsheet",
            catalog=catalog,
            store=store,
        )

        assert isinstance(result, ToolSearchOutput)
        assert len(result.candidates) > 0
        # excel_analyze should be the top match
        assert result.candidates[0].name == "excel_analyze"
        assert result.candidates[0].reason  # reason is non-empty
        assert "excel" in result.candidates[0].reason or "analyze" in result.candidates[0].reason

    def test_search_stores_pending_candidates(self):
        """tool_search 将候选存入 store 的 pending_candidates。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=10)

        execute_tool_search("analyze excel", catalog=catalog, store=store)

        assert store.is_pending("excel_analyze")
        assert not store.is_active("excel_analyze")


# ── Test 3: activate_tools makes tools available next turn ──


class TestActivateToolsAddsToNextTurn:
    def test_activate_makes_tool_visible(self):
        """activate_tools 后，工具进入可见列表（模拟下一轮 tools=）。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=10)

        # Step 1: search
        execute_tool_search("analyze excel", catalog=catalog, store=store)

        # Step 2: activate
        result = execute_activate_tools(
            ["excel_analyze"],
            catalog=catalog,
            store=store,
            allowed_tools=["tool_search", "activate_tools", "excel_analyze"],
            deny_tools=frozenset(),
            iteration=1,
        )

        assert "excel_analyze" in result.activated
        assert store.is_active("excel_analyze")

        # Step 3: resolve visible — excel_analyze now appears
        allowed = ["tool_search", "activate_tools", "excel_analyze", "vector_search"]
        visible = resolve_visible_tools(allowed, catalog=catalog, store=store)
        assert "excel_analyze" in visible
        # vector_search was NOT activated
        assert "vector_search" not in visible


# ── Test 4: cannot activate unsearched tools ──


class TestCannotActivateUnsearched:
    def test_activate_rejects_unsearched_tool(self):
        """未通过 tool_search 搜索过的工具不能被 activate_tools 激活。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=10)

        # Try to activate without searching first
        result = execute_activate_tools(
            ["excel_analyze"],
            catalog=catalog,
            store=store,
            allowed_tools=["excel_analyze"],
            deny_tools=frozenset(),
            iteration=1,
        )

        assert "excel_analyze" in result.not_in_candidates
        assert "excel_analyze" not in result.activated
        assert not store.is_active("excel_analyze")

    def test_activate_rejects_tool_not_in_last_search(self):
        """搜索 A 后不能激活 B（如果 B 不在搜索结果中）。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=10)

        # Search for excel only
        execute_tool_search("analyze excel", catalog=catalog, store=store)

        # Try to activate pdf_extract (not in search results)
        result = execute_activate_tools(
            ["pdf_extract"],
            catalog=catalog,
            store=store,
            allowed_tools=["pdf_extract"],
            deny_tools=frozenset(),
            iteration=1,
        )

        assert "pdf_extract" in result.not_in_candidates


# ── Test 5: runtime does not auto-activate ──


class TestRuntimeDoesNotAutoActivate:
    def test_search_does_not_activate(self):
        """tool_search 不自动激活任何工具——只有 activate_tools 才能激活。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=10)

        # Search returns candidates
        result = execute_tool_search("analyze excel", catalog=catalog, store=store)
        assert len(result.candidates) > 0

        # But NO tool is activated
        assert store.active_names() == []
        assert not store.is_active("excel_analyze")

    def test_activate_requires_explicit_names(self):
        """activate_tools 必须由模型显式提供工具名，不能自动推断。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=10)

        # Search
        execute_tool_search("analyze excel", catalog=catalog, store=store)

        # Activate with empty list — nothing happens
        result = execute_activate_tools(
            [],
            catalog=catalog,
            store=store,
            allowed_tools=[],
            deny_tools=frozenset(),
            iteration=1,
        )
        assert result.activated == ()
        assert store.active_names() == []


# ── Additional edge cases ──


class TestEdgeCases:
    def test_deny_tools_blocks_activation(self):
        """deny_tools 中的工具不能被激活。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=10)

        execute_tool_search("analyze excel", catalog=catalog, store=store)

        result = execute_activate_tools(
            ["excel_analyze"],
            catalog=catalog,
            store=store,
            allowed_tools=["excel_analyze"],
            deny_tools=frozenset({"excel_analyze"}),
            iteration=1,
        )

        assert "excel_analyze" in result.denied
        assert not store.is_active("excel_analyze")

    def test_not_in_allowed_tools_blocks_activation(self):
        """不在 allowed_tools 中的工具不能被激活。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=10)

        execute_tool_search("analyze excel", catalog=catalog, store=store)

        result = execute_activate_tools(
            ["excel_analyze"],
            catalog=catalog,
            store=store,
            allowed_tools=[],  # excel_analyze not in allowed
            deny_tools=frozenset(),
            iteration=1,
        )

        assert "excel_analyze" in result.denied

    def test_max_active_limit(self):
        """激活数量不能超过 max_active。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=1)

        # Search for excel first, activate it
        execute_tool_search("analyze excel", catalog=catalog, store=store)
        r1 = execute_activate_tools(
            ["excel_analyze"],
            catalog=catalog, store=store,
            allowed_tools=["excel_analyze", "vector_search"],
            deny_tools=frozenset(),
            iteration=1,
        )
        assert "excel_analyze" in r1.activated

        # Now search for vector and try to activate — hits max_active=1
        execute_tool_search("search vector similarity", catalog=catalog, store=store)
        r2 = execute_activate_tools(
            ["vector_search"],
            catalog=catalog, store=store,
            allowed_tools=["excel_analyze", "vector_search"],
            deny_tools=frozenset(),
            iteration=2,
        )
        assert "vector_search" in r2.denied

    def test_chinese_search(self):
        """中文查询能正确搜索工具。"""
        catalog = _build_test_catalog()
        catalog.register(ToolCatalogEntry(
            name="excel分析",
            description="分析Excel工作簿数据",
            category="deferred",
            search_text="excel分析 分析excel工作簿数据 file_path string 路径",
        ))

        results = catalog.search("分析Excel")
        names = [r.name for r in results]
        # Should find at least one of the excel tools
        assert "excel_analyze" in names or "excel分析" in names

    def test_flatten_schema_extracts_params(self):
        """flatten_schema 正确提取参数名和描述。"""
        schema = {
            "properties": {
                "file_path": {"type": "string", "description": "Path to workbook"},
                "sheet_name": {"type": "string", "description": "Sheet name"},
            },
            "required": ["file_path"],
        }
        text = flatten_schema(schema)
        assert "file_path" in text
        assert "Path to workbook" in text
        assert "required" in text

    def test_flatten_schema_handles_additional_properties(self):
        """flatten_schema 处理 additionalProperties。"""
        schema = {
            "properties": {"name": {"type": "string"}},
            "additionalProperties": {"type": "string", "description": "Extra fields"},
        }
        text = flatten_schema(schema)
        assert "Extra fields" in text

    def test_store_sync_roundtrip(self):
        """DeferredToolStore 状态可以通过 LoopState 正确往返。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=10)

        execute_tool_search("analyze excel", catalog=catalog, store=store)
        store.activate("excel_analyze", iteration=5)
        store.pin("excel_analyze")

        # Sync to state
        state: dict = {}
        store.sync_to_state(state)

        # Verify state structure
        assert state["discovery_active_tools"] == ["excel_analyze"]
        assert state["discovery_active_tool_iterations"] == {"excel_analyze": 5}
        assert state["discovery_pinned_tools"] == ["excel_analyze"]

        # Restore from state
        store2 = DeferredToolStore(max_active=10)
        store2.sync_from_state(state)

        assert store2.is_active("excel_analyze")
        assert store2.is_pending("excel_analyze")
        # Iteration preserved
        refs = store2.active_refs()
        assert len(refs) == 1
        assert refs[0].activated_at_iteration == 5
        # Pinned preserved
        assert "excel_analyze" in store2._pinned

    def test_already_active_returns_correctly(self):
        """重复激活已激活的工具返回 already_active。"""
        catalog = _build_test_catalog()
        store = DeferredToolStore(max_active=10)

        execute_tool_search("analyze excel", catalog=catalog, store=store)
        r1 = execute_activate_tools(
            ["excel_analyze"],
            catalog=catalog, store=store,
            allowed_tools=["excel_analyze"],
            deny_tools=frozenset(),
            iteration=1,
        )
        assert "excel_analyze" in r1.activated

        # Second activation — already_active
        r2 = execute_activate_tools(
            ["excel_analyze"],
            catalog=catalog, store=store,
            allowed_tools=["excel_analyze"],
            deny_tools=frozenset(),
            iteration=2,
        )
        assert "excel_analyze" in r2.already_active
        assert "excel_analyze" not in r2.activated

    def test_pin_prevents_eviction(self):
        """pinned 工具不会被 LRU 驱逐。"""
        store = DeferredToolStore(max_active=1)

        # Create two pending candidates
        store.set_pending_candidates("test", [
            SearchCandidate(name="tool_a", description="A", reason="match"),
            SearchCandidate(name="tool_b", description="B", reason="match"),
        ])

        # Activate tool_a and pin it
        store.activate("tool_a", iteration=1)
        store.pin("tool_a")

        # Activate tool_b — tool_a should NOT be evicted because it's pinned
        # This should raise because all slots are pinned
        with pytest.raises(RuntimeError, match="all 1 slots are pinned"):
            store.activate("tool_b", iteration=2)

        assert store.is_active("tool_a")
        assert not store.is_active("tool_b")

    def test_internal_tools_excluded(self):
        """internal 工具在 resolve_visible_tools 中被排除。"""
        catalog = _build_test_catalog()
        catalog.register(ToolCatalogEntry(
            name="audit_log",
            description="Internal audit logging",
            category="internal",
            search_text="audit_log internal audit logging",
        ))
        store = DeferredToolStore(max_active=10)

        visible = resolve_visible_tools(
            ["tool_search", "excel_analyze", "audit_log"],
            catalog=catalog,
            store=store,
        )
        assert "tool_search" in visible
        assert "excel_analyze" not in visible  # deferred, not activated
        assert "audit_log" not in visible  # internal, never visible
