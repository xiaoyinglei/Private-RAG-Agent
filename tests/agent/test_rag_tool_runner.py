from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from rag.agent.core.context import AgentRunConfig
from rag.agent.tools.rag_tool_runner import (
    AsyncRAGToolRunner,
    RAGToolRunnerNotConfiguredError,
    _evidence_to_output,
)
from rag.agent.tools.rag_tools import SearchInput
from rag.agent.tools.registry import ToolExecutionContext
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy, RuntimeMode

# ── Fake evidence ──


class _FakeEvidenceItem:
    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_evidence(n: int = 1) -> list[_FakeEvidenceItem]:
    return [
        _FakeEvidenceItem(
            evidence_id=f"ev{n}",
            doc_id=n * 10,
            source_id=n * 100,
            citation_anchor=f"sec-{n}",
            text=f"text {n}",
            score=0.9,
            file_name=f"doc_{n}.md",
            source_type="markdown",
            record_type="section_summary",
            section_path=["Ch1", f"Sec{n}"],
            retrieval_channels=["vector"],
            page_start=n,
            page_end=n + 1,
            benchmark_doc_id=f"bdoc_{n}",
        )
        for n in range(1, n + 1)
    ]


# ── Fake retrieval service ──


@dataclass
class _FakeRetrievalService:
    evidence: Any = field(default_factory=list)
    fail_next: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def aretrieve_payload(self, query: str, *, access_policy: Any, query_options: Any = None) -> Any:
        self.calls.append(
            {
                "query": query,
                "access_policy": access_policy,
                "query_options": query_options,
            }
        )
        if self.fail_next:
            raise RuntimeError("simulated retrieval failure")
        return _FakePayload(self.evidence)


@dataclass
class _FakeEvidenceBundle:
    internal: list[Any] = field(default_factory=list)
    external: list[Any] = field(default_factory=list)
    graph: list[Any] = field(default_factory=list)


@dataclass
class _FakePayload:
    evidence: _FakeEvidenceBundle = field(default_factory=_FakeEvidenceBundle)


def _execution_context(
    *,
    access_policy: AccessPolicy | None = None,
    run_id: str = "rag-tool-runner",
) -> ToolExecutionContext:
    return ToolExecutionContext(
        run_config=AgentRunConfig(
            run_id=run_id,
            thread_id=run_id,
            budget_total=1000,
            max_depth=1,
            access_policy=access_policy or AccessPolicy.default(),
        )
    )


# ── _evidence_to_output ──


class TestEvidenceToOutput:
    def test_retains_key_fields(self) -> None:
        evidence = _make_evidence(1)
        output = _evidence_to_output(evidence)
        item = output.items[0]
        assert item["text"] == "text 1"
        assert item["score"] == 0.9
        assert item["doc_id"] == 10
        assert item["source_id"] == 100
        assert item["citation_anchor"] == "sec-1"
        assert item["file_name"] == "doc_1.md"
        assert item["section_path"] == ["Ch1", "Sec1"]
        assert item["retrieval_channels"] == ["vector"]
        assert item["page_start"] == 1
        assert item["page_end"] == 2

    def test_empty_evidence(self) -> None:
        output = _evidence_to_output([])
        assert output.items == []


# ── AsyncRAGToolRunner ──


class TestAsyncRAGToolRunner:
    def test_search_input_rejects_access_policy_override(self) -> None:
        with pytest.raises(ValidationError, match="access_policy"):
            SearchInput.model_validate(
                {
                    "query": "test",
                    "access_policy": AccessPolicy.default().model_dump(mode="json"),
                }
            )

    @pytest.mark.anyio
    async def test_aretrieve_payload_primary_path(self) -> None:
        """验证 aretrieve_payload 是第一优先级。"""
        bundle = _FakeEvidenceBundle(internal=_make_evidence(2))
        svc = _FakeRetrievalService(evidence=bundle)

        runner = AsyncRAGToolRunner(retrieval_service=svc)
        output = await runner.retrieve_evidence(
            SearchInput(query="test"),
            _execution_context(),
        )
        assert len(output.items) == 2
        assert output.items[0]["text"] == "text 1"
        assert output.items[1]["doc_id"] == 20
        assert svc.calls[0]["query_options"] is not None

    @pytest.mark.anyio
    async def test_aretrieve_payload_fail_does_not_fallback(self) -> None:
        """能力存在但执行失败 → fail loud，不 fallback。"""
        svc = _FakeRetrievalService(evidence=_FakeEvidenceBundle(), fail_next=True)

        runner = AsyncRAGToolRunner(retrieval_service=svc, allow_sync_fallback=False)
        with pytest.raises(RuntimeError, match="simulated retrieval failure"):
            await runner.retrieve_evidence(
                SearchInput(query="test"),
                _execution_context(),
            )

    @pytest.mark.anyio
    async def test_to_thread_fallback_when_no_async(self) -> None:
        """无 aretrieve_payload 时 fallback 到 runtime.query()。"""
        calls: list[str] = []

        class _FakeRuntime:
            access_policy = AccessPolicy.default()

            def query(self, *args: Any, **kwargs: Any) -> Any:
                calls.append("query")
                return _FakeQueryResult(evidence=_make_evidence(1))

        @dataclass
        class _FakeQueryResult:
            evidence: list[Any]

        runtime = _FakeRuntime()
        runner = AsyncRAGToolRunner(runtime=runtime)
        output = await runner.retrieve_evidence(
            SearchInput(query="test"),
            _execution_context(),
        )
        assert "query" in calls
        assert len(output.items) == 1

    @pytest.mark.anyio
    async def test_aquery_fallback_preserves_retrieval_signals(self) -> None:
        """aquery fallback 不能静默丢弃 Agent 产出的 retrieval_signals。"""
        captured: dict[str, Any] = {}
        signals = RetrievalSignals(special_targets=["table"], quoted_terms=["报销"])

        class _FakeRuntime:
            async def aquery(self, query: str, *, options: Any) -> Any:
                captured["query"] = query
                captured["options"] = options
                return _FakeQueryResult(evidence=_make_evidence(1))

        @dataclass
        class _FakeQueryResult:
            evidence: list[Any]

        runner = AsyncRAGToolRunner(runtime=_FakeRuntime())
        output = await runner.retrieve_evidence(
            SearchInput(query="test", retrieval_signals=signals),
            _execution_context(),
        )

        assert len(output.items) == 1
        assert captured["query"] == "test"
        options = captured["options"]
        assert options.retrieval_signals is signals
        assert options.retrieval_signals_debug["signals_source"] == "agent_tool_input"
        assert options.retrieval_signals_debug["special_targets"] == ["table"]

    @pytest.mark.anyio
    async def test_fail_closed_when_nothing_configured(self) -> None:
        """无 runtime、无 retrieval_service → fail closed。"""
        runner = AsyncRAGToolRunner()
        with pytest.raises(RAGToolRunnerNotConfiguredError, match="not configured"):
            await runner.retrieve_evidence(
                SearchInput(query="test"),
                _execution_context(),
            )

    @pytest.mark.anyio
    async def test_no_sync_fallback_when_disabled(self) -> None:
        """allow_sync_fallback=False + 无 async → fail closed。"""
        runner = AsyncRAGToolRunner(allow_sync_fallback=False)
        with pytest.raises(RAGToolRunnerNotConfiguredError, match="not configured"):
            await runner.retrieve_evidence(
                SearchInput(query="test"),
                _execution_context(),
            )

    @pytest.mark.anyio
    async def test_access_policy_comes_from_execution_context(self) -> None:
        access_policy = AccessPolicy(
            allowed_runtimes=frozenset({RuntimeMode.FAST})
        )
        svc = _FakeRetrievalService(evidence=_FakeEvidenceBundle())
        runner = AsyncRAGToolRunner(retrieval_service=svc)

        await runner.retrieve_evidence(
            SearchInput(query="test"),
            _execution_context(access_policy=access_policy),
        )

        assert svc.calls[0]["access_policy"] is access_policy
        assert svc.calls[0]["query_options"].access_policy is access_policy

    @pytest.mark.anyio
    async def test_shared_runner_keeps_concurrent_run_policies_isolated(self) -> None:
        fast_policy = AccessPolicy(
            allowed_runtimes=frozenset({RuntimeMode.FAST})
        )
        deep_policy = AccessPolicy(
            allowed_runtimes=frozenset({RuntimeMode.DEEP})
        )
        svc = _FakeRetrievalService(evidence=_FakeEvidenceBundle())
        runner = AsyncRAGToolRunner(retrieval_service=svc)

        await asyncio.gather(
            runner.retrieve_evidence(
                SearchInput(query="fast"),
                _execution_context(access_policy=fast_policy, run_id="fast-run"),
            ),
            runner.retrieve_evidence(
                SearchInput(query="deep"),
                _execution_context(access_policy=deep_policy, run_id="deep-run"),
            ),
        )

        policies_by_query = {
            call["query"]: call["access_policy"]
            for call in svc.calls
        }
        assert policies_by_query == {
            "fast": fast_policy,
            "deep": deep_policy,
        }
