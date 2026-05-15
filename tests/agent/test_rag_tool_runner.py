from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from rag.agent.tools.rag_tool_runner import (
    AsyncRAGToolRunner,
    RAGToolRunnerNotConfiguredError,
    _evidence_to_output,
)
from rag.agent.tools.rag_tools import SearchInput
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy

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
    @pytest.mark.anyio
    async def test_aretrieve_payload_primary_path(self) -> None:
        """验证 aretrieve_payload 是第一优先级。"""
        bundle = _FakeEvidenceBundle(internal=_make_evidence(2))
        svc = _FakeRetrievalService(evidence=bundle)

        runner = AsyncRAGToolRunner(retrieval_service=svc)
        output = await runner.retrieve_evidence(
            SearchInput(query="test")
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
            await runner.retrieve_evidence(SearchInput(query="test"))

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
        output = await runner.retrieve_evidence(SearchInput(query="test"))
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
            SearchInput(query="test", retrieval_signals=signals)
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
            await runner.retrieve_evidence(SearchInput(query="test"))

    @pytest.mark.anyio
    async def test_no_sync_fallback_when_disabled(self) -> None:
        """allow_sync_fallback=False + 无 async → fail closed。"""
        runner = AsyncRAGToolRunner(allow_sync_fallback=False)
        with pytest.raises(RAGToolRunnerNotConfiguredError, match="not configured"):
            await runner.retrieve_evidence(SearchInput(query="test"))

    def test_access_policy_priority(self) -> None:
        """access_policy 来源优先级验证。"""
        payload_ap = AccessPolicy(local_only=True)
        runner_ap = AccessPolicy(local_only=False)

        # 无 runtime 时用 runner policy
        runner = AsyncRAGToolRunner(access_policy=runner_ap)
        ap = runner._resolve_access_policy(SearchInput(query="test"))
        assert ap is runner_ap

        # payload 优先：通过额外属性注入
        class _SearchInputWithPolicy(SearchInput):
            access_policy: AccessPolicy | None = None

        ap = runner._resolve_access_policy(
            _SearchInputWithPolicy(query="test", access_policy=payload_ap)
        )
        assert ap is payload_ap
