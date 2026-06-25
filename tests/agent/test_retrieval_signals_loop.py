from __future__ import annotations

import pytest

from rag.agent.tools.rag_tools import RAG_SIGNAL_AWARE_TOOLS, SearchInput
from rag.schema.query import RetrievalSignals

# ── RAG tool schema ──


class TestSearchInputSchema:
    def test_accepts_retrieval_signals(self) -> None:
        signals = RetrievalSignals(
            special_targets=["table"],
            allow_graph_expansion=True,
        )
        inp = SearchInput(
            query="test",
            retrieval_signals=signals,
        )
        assert inp.retrieval_signals is not None
        assert inp.retrieval_signals.special_targets == ["table"]

    def test_defaults_to_none(self) -> None:
        inp = SearchInput(query="test", top_k=5)
        assert inp.retrieval_signals is None


# ── RAG_SIGNAL_AWARE_TOOLS whitelist ──


class TestRAGSignalAwareTools:
    def test_contains_all_rag_tools(self) -> None:
        assert "vector_search" in RAG_SIGNAL_AWARE_TOOLS
        assert "keyword_search" in RAG_SIGNAL_AWARE_TOOLS
        assert "grounding" in RAG_SIGNAL_AWARE_TOOLS
        assert "rerank" in RAG_SIGNAL_AWARE_TOOLS
        assert "graph_expand" in RAG_SIGNAL_AWARE_TOOLS

    def test_does_not_contain_llm_tools(self) -> None:
        assert "llm_summarize" not in RAG_SIGNAL_AWARE_TOOLS
        assert "llm_generate" not in RAG_SIGNAL_AWARE_TOOLS


# ── No routing mapping in Agent layer ──


class TestNoRoutingMapping:
    def test_llm_providers_module_has_no_routing_from_signals(self) -> None:
        import rag.agent.core.llm_providers as m

        assert not hasattr(m, "_routing_from_signals")

    def test_llm_providers_has_no_runtime_mode_import(self) -> None:
        import rag.agent.core.llm_providers as m

        src = m.__dict__.get("__file__", "")
        if not src:
            return
        from pathlib import Path

        text = Path(str(src)).read_text()
        assert "RuntimeMode" not in text
        assert "runtime_mode" not in text.lower()

    def test_tool_execution_has_no_routing_from_signals(self) -> None:
        import inspect

        import rag.agent.core.tool_execution as m

        src = inspect.getsource(m)
        assert "_routing_from_signals" not in src


class TestQueryOptionsToRetrievalServiceSignalFlow:
    """验证 QueryOptions.retrieval_signals → RetrievalService → engine 的完整链路"""

    def test_query_options_carries_retrieval_signals(self) -> None:
        from rag.retrieval import QueryOptions

        signals = RetrievalSignals(special_targets=["table"], quoted_terms=["报销"])
        opts = QueryOptions(retrieval_signals=signals)
        assert opts.retrieval_signals is not None
        assert opts.retrieval_signals.special_targets == ["table"]
        assert opts.retrieval_signals.quoted_terms == ["报销"]

    def test_query_options_signals_default_to_none(self) -> None:
        from rag.retrieval import QueryOptions

        opts = QueryOptions()
        assert opts.retrieval_signals is None
        assert opts.retrieval_signals_debug == {}

    def test_aretrieve_payload_reads_signals_from_query_options(self) -> None:
        """aretrieve_payload 不再硬编码 RetrievalSignals()"""
        import inspect

        import rag.retrieval.orchestrator as m

        src = inspect.getsource(m.RetrievalService.aretrieve_payload)
        # 确认不再有硬编码的 RetrievalSignals()
        assert "retrieval_signals=RetrievalSignals()" not in src
        # 确认从 query_options 读取 signals
        assert "query_options.retrieval_signals" in src
        # 确认有 fallback 逻辑
        assert "signals or RetrievalSignals()" in src

    def test_async_rag_tool_runner_passes_signals_in_query_options(self) -> None:
        """AsyncRAGToolRunner._via_aretrieve_payload 构造 QueryOptions 时包含 retrieval_signals"""
        import inspect

        import rag.agent.tools.rag_tool_runner as m

        src = inspect.getsource(m.AsyncRAGToolRunner._query_options)
        assert "retrieval_signals=" in src
        assert "retrieval_signals_debug=" in src
        assert '"signals_source": "agent_tool_input"' in src

    def test_rag_answer_runner_passes_signals(self) -> None:
        """RAGSearchAnswerRunner.answer 在 QueryOptions 中传递 retrieval_signals"""
        import inspect

        import rag.agent.tools.rag_answer_tools as m

        src = inspect.getsource(m.RAGSearchAnswerRunner.answer)
        assert '"retrieval_signals": _answer_path_retrieval_signals(payload.retrieval_signals)' in src
        assert '"signals_source": "agent_tool_input"' in src
        assert '"answer_path_special_targets_skipped"' in src

    def test_aretrieve_payload_signals_plumbing(self) -> None:
        """验证 aretrieve_payload 的信号解析逻辑（不构造完整 service）"""
        from rag.retrieval import QueryOptions

        # 模拟信号解析逻辑（与 aretrieve_payload 一致）
        def _resolve(qo):
            s = qo.retrieval_signals if qo else None
            return s or RetrievalSignals()

        # 有 signals 时用传入的
        signals = RetrievalSignals(special_targets=["table"], quoted_terms=["报销"])
        qo = QueryOptions(retrieval_signals=signals)
        assert _resolve(qo).special_targets == ["table"]

        # 无 signals 时 fallback 到空
        qo_none = QueryOptions()
        resolved = _resolve(qo_none)
        assert isinstance(resolved, RetrievalSignals)
        assert resolved.special_targets == []
        assert resolved.quoted_terms == []

        # query_options=None 也不崩
        assert isinstance(_resolve(None), RetrievalSignals)

    @pytest.mark.anyio
    async def test_l3_l4_bypass_payload_preserves_signals_debug(self) -> None:
        """L3/L4 payload diagnostics 要保留 QueryOptions 中的 retrieval_signals_debug。"""
        from rag.retrieval import QueryOptions
        from rag.retrieval.l3_l4_engine import L3L4RetrievalEngine
        from rag.retrieval.runtime_coordinator import RoutingDecision
        from rag.schema.runtime import AccessPolicy, RuntimeMode

        signals = RetrievalSignals(special_targets=["table"], quoted_terms=["报销"])
        debug = {
            "signals_source": "agent_tool_input",
            "special_targets": ["table"],
            "quoted_terms": ["报销"],
        }
        payload = await object.__new__(L3L4RetrievalEngine).arun(
            "test",
            access_policy=AccessPolicy.default(),
            retrieval_signals=signals,
            decision=RoutingDecision(runtime_mode=RuntimeMode.FAST),
            query_options=QueryOptions(
                retrieval_profile="bypass",
                retrieval_signals=signals,
                retrieval_signals_debug=debug,
            ),
        )

        assert payload.retrieval_signals is signals
        assert payload.retrieval_signals_debug == debug
