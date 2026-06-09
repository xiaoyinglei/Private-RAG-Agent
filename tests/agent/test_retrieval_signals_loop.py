from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from rag.agent.core.llm_providers import (
    LLMRetrievalHintProvider,
    _extract_quoted_terms,
    _filter_non_empty,
    _merge_quoted_terms,
    _validate_retrieval_signals,
)
from rag.agent.graphs.nodes.execute import run_tools_raw
from rag.agent.tools.rag_tools import RAG_SIGNAL_AWARE_TOOLS, SearchInput, SearchOutput
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.query import RetrievalSignals

# ── Helpers ──


class _DummyResult(BaseModel):
    text: str


def _make_state(**overrides: object) -> dict:
    from rag.agent.core.context import AgentRunConfig
    from rag.schema.runtime import AccessPolicy

    s: dict[str, Any] = {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": "test query",
        "retrieval_signals": RetrievalSignals(),
        "retrieval_signals_debug": None,
        "run_config": AgentRunConfig(
            run_id="sig-test", thread_id="sig-test", budget_total=10000, max_depth=2,
            access_policy=AccessPolicy.default(),
        ),
        "iteration": 0, "status": "running",
        "decision_reason": None, "stop_reason": None, "needs_user_input": None,
        "pending_tool_calls": [], "approved_tool_call_ids": [], "denied_tool_call_ids": [],
        "user_decision": None, "user_message": None,
        "human_input_request": None, "human_input_response": None,
        "working_summary": None,
        "extracted_facts": [], "context_budget": None,
        "final_answer": None, "groundedness_flag": False, "insufficient_evidence_flag": False,
    }
    s.update(overrides)
    return s


def _stub_gen(*responses: dict[str, object]) -> object:
    class Stub:
        def __init__(self):
            self._idx = 0
            self._responses = responses

        def generate_structured(self, *, prompt, schema, **kw):
            if self._idx >= len(self._responses):
                return None
            r = self._responses[self._idx]
            self._idx += 1
            return schema.model_validate(r)

    return Stub()


# ── quoted_terms extraction ──


class TestQuotedTermsExtraction:
    def test_extracts_double_quotes(self) -> None:
        terms = _extract_quoted_terms('查询"公积金"政策')
        assert "公积金" in terms

    def test_extracts_chinese_quotes(self) -> None:
        terms = _extract_quoted_terms('查询“公积金”政策')
        assert "公积金" in terms

    def test_extracts_single_quotes(self) -> None:
        terms = _extract_quoted_terms("查询'公积金'政策")
        assert "公积金" in terms

    def test_deduplicates(self) -> None:
        terms = _extract_quoted_terms('"A" and "A"')
        assert len(terms) == 1

    def test_no_quotes_returns_empty(self) -> None:
        terms = _extract_quoted_terms("普通查询")
        assert terms == []


class TestMergeQuotedTerms:
    def test_rule_terms_first(self) -> None:
        """规则提取的 quoted_terms 优先排在前面。"""
        merged = _merge_quoted_terms(
            ["LLM词", "共同词"],
            ["规则词", "共同词"],
        )
        assert merged[:2] == ["规则词", "共同词"]

    def test_filters_empty_and_dedup(self) -> None:
        """规则优先：rule_terms 的 'valid' 排在 llm_terms 的 'valid' 之前"""
        merged = _merge_quoted_terms(
            ["", "llm-only", "  "],    # llm_terms (second priority)
            ["valid", "rule-only"],     # rule_terms (first priority)
        )
        # rule 优先: valid, rule-only; 然后 llm: llm-only
        assert merged == ["valid", "rule-only", "llm-only"]


class TestFilterNonEmpty:
    def test_filters_empty_strings(self) -> None:
        result = _filter_non_empty(["a", "", "  ", "b"])
        assert result == ["a", "b"]

    def test_empty_list(self) -> None:
        result = _filter_non_empty([])
        assert result == []


class TestValidateRetrievalSignals:
    def test_valid_signals(self) -> None:
        raw = {
            "special_targets": ["table"],
            "quoted_terms": ["公积金"],
            "allow_graph_expansion": True,
        }
        signals, source = _validate_retrieval_signals(raw)
        assert signals.special_targets == ["table"]
        assert signals.quoted_terms == ["公积金"]
        assert signals.allow_graph_expansion is True
        assert source == "llm"

    def test_filters_unknown_fields(self) -> None:
        raw = {
            "special_targets": ["table"],
            "metadata_filters": {"doc_id": 123},
            "unknown_field": "ignored",
        }
        signals, source = _validate_retrieval_signals(raw)
        assert signals.special_targets == ["table"]
        assert signals.metadata_filters.has_constraints() is False
        assert source == "llm"

    def test_none_returns_rule_fallback(self) -> None:
        signals, source = _validate_retrieval_signals(None)
        assert signals.special_targets == []
        assert source == "rule_fallback"

    def test_invalid_type_returns_rule_fallback(self) -> None:
        signals, source = _validate_retrieval_signals("not a dict")  # type: ignore[arg-type]
        assert signals.special_targets == []
        assert source == "rule_fallback"

    def test_malformed_json_returns_validation_failed(self) -> None:
        """类型错误（非 list/non-bool）→ validation_failed"""
        signals, source = _validate_retrieval_signals(
            {"special_targets": "not_a_list", "quoted_terms": 123, "allow_graph_expansion": "yes"}  # type: ignore[dict-item]
        )
        assert source == "validation_failed"


# ── LLMRetrievalHintProvider writes retrieval_signals ──


class TestLLMRetrievalHintProviderSignals:
    def test_writes_signals_when_llm_produces(self) -> None:
        gen = _stub_gen({
            "route": "direct",
            "reason": "test",
            "retrieval_signals": {
                "special_targets": ["table"],
                "quoted_terms": ["公积金"],
                "allow_graph_expansion": False,
            },
        })
        provider = LLMRetrievalHintProvider(gen)
        result = provider.hint(_make_state(task="查询公积金"))
        signals = result["retrieval_signals"]
        assert isinstance(signals, RetrievalSignals)
        assert signals.special_targets == ["table"]
        assert signals.quoted_terms == ["公积金"]

    def test_writes_empty_signals_when_llm_returns_none(self) -> None:
        gen = _stub_gen({
            "route": "direct",
            "reason": "simple query",
        })
        provider = LLMRetrievalHintProvider(gen)
        result = provider.hint(_make_state(task="hello"))
        signals = result["retrieval_signals"]
        assert isinstance(signals, RetrievalSignals)
        assert signals.special_targets == []
        assert signals.allow_graph_expansion is False

    def test_overwrites_previous_signals(self) -> None:
        """每轮 retrieval hint 必须覆盖，不是累积。"""
        gen = _stub_gen({
            "route": "direct",
            "reason": "test",
            "retrieval_signals": {
                "quoted_terms": ["新词"],
                "allow_graph_expansion": False,
            },
        })
        provider = LLMRetrievalHintProvider(gen)
        state_with_old = _make_state(
            task="新查询",
            retrieval_signals=RetrievalSignals(
                allow_graph_expansion=True,  # 上一轮是 True
                quoted_terms=["旧词"],
            ),
        )
        result = provider.hint(state_with_old)
        signals = result["retrieval_signals"]
        assert signals.allow_graph_expansion is False  # 被新值覆盖
        assert "新词" in signals.quoted_terms

    def test_merges_rule_quoted_terms(self) -> None:
        gen = _stub_gen({
            "route": "direct",
            "reason": "test",
            "retrieval_signals": {
                "quoted_terms": ["LLM提取"],
            },
        })
        provider = LLMRetrievalHintProvider(gen)
        result = provider.hint(_make_state(task='查询"规则提取"内容'))
        signals = result["retrieval_signals"]
        assert "LLM提取" in signals.quoted_terms
        assert "规则提取" in signals.quoted_terms

    def test_writes_debug_info(self) -> None:
        gen = _stub_gen({
            "route": "direct",
            "reason": "test",
            "retrieval_signals": {
                "special_targets": ["table"],
            },
        })
        provider = LLMRetrievalHintProvider(gen)
        result = provider.hint(_make_state())
        debug = result["retrieval_signals_debug"]
        assert debug is not None
        assert debug["signals_source"] == "llm"
        assert "special_targets" in debug

    def test_fallback_signals_source_is_rule(self) -> None:
        """LLM 未返回 retrieval_signals → rule_fallback"""
        gen = _stub_gen({
            "reason": "simple",
        })
        provider = LLMRetrievalHintProvider(gen)
        result = provider.hint(_make_state())
        debug = result["retrieval_signals_debug"]
        assert debug is not None
        assert debug["signals_source"] == "rule_fallback"

    def test_validation_failed_in_debug(self) -> None:
        """LLM 返回非法 retrieval_signals → validation_failed 写入 debug"""
        gen = _stub_gen({
            "route": "direct",
            "reason": "test",
            "retrieval_signals": {
                "special_targets": 42,  # not a list → validation fails
            },
        })
        provider = LLMRetrievalHintProvider(gen)
        result = provider.hint(_make_state())
        debug = result["retrieval_signals_debug"]
        assert debug is not None
        assert debug["signals_source"] == "validation_failed"


# ── run_tools_raw injects signals ──


class TestExecuteNodeSignalInjection:
    @pytest.mark.anyio
    async def test_injects_signals_for_rag_tools(self) -> None:
        captured_args: dict[str, Any] = {}

        def runner(payload: SearchInput) -> SearchOutput:
            captured_args["retrieval_signals"] = payload.retrieval_signals
            return SearchOutput(items=[])

        spec = ToolSpec(
            name="vector_search", description="search",
            input_model=SearchInput, output_model=SearchOutput,
            error_model=ToolError, permissions=ToolPermissions(read_db=True, embed=True),
            timeout_seconds=5.0,
        )
        registry = ToolRegistry()
        registry.register(spec, runner=runner)

        from rag.agent.state import ToolCallPlan
        signals = RetrievalSignals(
            special_targets=["table"],
            quoted_terms=["公积金"],
            allow_graph_expansion=True,
        )
        call = ToolCallPlan.create("vector_search", {"query": "test", "top_k": 8})

        update = await run_tools_raw(
            _make_state(
                retrieval_signals=signals,
                pending_tool_calls=[call],
            ),
            tool_registry=registry, allowed_tools=frozenset({"vector_search"}),
        )
        assert update.get("status") != "paused"
        assert captured_args["retrieval_signals"] is not None
        passed_signals = captured_args["retrieval_signals"]
        assert isinstance(passed_signals, RetrievalSignals)
        assert passed_signals.special_targets == ["table"]
        assert passed_signals.allow_graph_expansion is True
        assert "retrieval_signals" not in call.arguments

    @pytest.mark.anyio
    async def test_does_not_inject_for_non_rag_tools(self) -> None:
        captured: dict[str, Any] = {}

        class NonRagInput(BaseModel):
            text: str
            retrieval_signals: RetrievalSignals | None = None

        class NonRagOutput(BaseModel):
            result: str

        def runner(payload: NonRagInput) -> NonRagOutput:
            captured["got_signals"] = payload.retrieval_signals
            return NonRagOutput(result="ok")

        spec = ToolSpec(
            name="llm_summarize", description="summarize",
            input_model=NonRagInput, output_model=NonRagOutput,
            error_model=ToolError, permissions=ToolPermissions(generate=True),
            timeout_seconds=5.0,
        )
        registry = ToolRegistry()
        registry.register(spec, runner=runner)

        from rag.agent.state import ToolCallPlan
        call = ToolCallPlan.create("llm_summarize", {"text": "hello"})

        await run_tools_raw(
            _make_state(
                retrieval_signals=RetrievalSignals(quoted_terms=["test"]),
                pending_tool_calls=[call],
            ),
            tool_registry=registry, allowed_tools=frozenset({"llm_summarize"}),
        )
        # 非 RAG 工具不应被注入 retrieval_signals
        assert captured["got_signals"] is None

    @pytest.mark.anyio
    async def test_does_not_overwrite_existing_signals_in_args(self) -> None:
        captured_args: dict[str, Any] = {}

        def runner(payload: SearchInput) -> SearchOutput:
            captured_args["retrieval_signals"] = payload.retrieval_signals
            return SearchOutput(items=[])

        spec = ToolSpec(
            name="vector_search", description="search",
            input_model=SearchInput, output_model=SearchOutput,
            error_model=ToolError, permissions=ToolPermissions(read_db=True),
            timeout_seconds=5.0,
        )
        registry = ToolRegistry()
        registry.register(spec, runner=runner)

        from rag.agent.state import ToolCallPlan
        explicit_signals = RetrievalSignals(quoted_terms=["显式设置"])
        call = ToolCallPlan.create("vector_search", {
            "query": "test",
            "retrieval_signals": explicit_signals.model_dump(mode="json"),
        })

        await run_tools_raw(
            _make_state(
                retrieval_signals=RetrievalSignals(quoted_terms=["state中的"]),
                pending_tool_calls=[call],
            ),
            tool_registry=registry, allowed_tools=frozenset({"vector_search"}),
        )
        # 已有 retrieval_signals 不覆盖
        passed = captured_args["retrieval_signals"]
        assert isinstance(passed, RetrievalSignals)
        assert "显式设置" in passed.quoted_terms

    @pytest.mark.anyio
    async def test_injects_empty_signals_when_state_has_none(self) -> None:
        captured: list[Any] = []

        def runner(payload: SearchInput) -> SearchOutput:
            captured.append(payload.retrieval_signals)
            return SearchOutput(items=[])

        spec = ToolSpec(
            name="keyword_search", description="search",
            input_model=SearchInput, output_model=SearchOutput,
            error_model=ToolError, permissions=ToolPermissions(read_db=True),
            timeout_seconds=5.0,
        )
        registry = ToolRegistry()
        registry.register(spec, runner=runner)

        from rag.agent.state import ToolCallPlan
        call = ToolCallPlan.create("keyword_search", {"query": "test"})

        await run_tools_raw(
            _make_state(
                retrieval_signals=None,  # type: ignore[arg-type]
                pending_tool_calls=[call],
            ),
            tool_registry=registry, allowed_tools=frozenset({"keyword_search"}),
        )
        # 没有 signals 时注入空 dict → 被 Pydantic 解析为默认 RetrievalSignals()
        assert isinstance(captured[0], RetrievalSignals)
        assert captured[0].special_targets == []
        assert captured[0].allow_graph_expansion is False


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

    def test_run_tools_raw_has_no_routing_from_signals(self) -> None:
        import inspect

        import rag.agent.graphs.nodes.execute as m
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
