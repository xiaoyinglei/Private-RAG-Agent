from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from rag.agent.tools.rag_tools import SearchInput, SearchOutput
from rag.schema.runtime import AccessPolicy


class RAGToolRunnerNotConfiguredError(RuntimeError):
    """没有可用的 retrieval_service、aquery 或 query fallback。"""


@dataclass
class AsyncRAGToolRunner:
    """async-first RAG 工具 runner。主路径走 RetrievalService.aretrieve_payload()。

    调用优先级：
    1. retrieval_service.aretrieve_payload()
    2. runtime.aquery()（如果存在）
    3. asyncio.to_thread(runtime.query, ...)（仅 fallback）
    """

    runtime: Any | None = None
    retrieval_service: Any | None = None
    access_policy: AccessPolicy | None = None
    max_context_tokens: int = 4096
    allow_sync_fallback: bool = True

    # ── Public API ──

    async def retrieve_evidence(self, payload: SearchInput) -> SearchOutput:
        """执行 RAG 检索，返回 SearchOutput。"""
        ap = self._resolve_access_policy(payload)

        # 能力不存在 → fallback，不能 fail loud
        has_async = self.retrieval_service is not None and callable(
            getattr(self.retrieval_service, "aretrieve_payload", None)
        )

        # 优先级 1: aretrieve_payload()
        if has_async:
            return await self._via_aretrieve_payload(payload, ap)

        # 优先级 2: runtime.aquery()
        if self.runtime is not None and callable(getattr(self.runtime, "aquery", None)):
            return await self._via_aquery(payload, ap)

        # 优先级 3: to_thread(runtime.query)
        if self.allow_sync_fallback and self.runtime is not None and callable(
            getattr(self.runtime, "query", None)
        ):
            return await self._via_to_thread(payload)

        raise RAGToolRunnerNotConfiguredError(
            "RAG tool runner is not configured. "
            "Please initialize RAGRuntime or configure retrieval_service."
        )

    # ── Internal ──

    async def _via_aretrieve_payload(
        self, payload: SearchInput, access_policy: AccessPolicy
    ) -> SearchOutput:
        from rag.retrieval import QueryOptions

        retrieval_signals = getattr(payload, "retrieval_signals", None)
        signals_debug: dict[str, object] = {}
        if retrieval_signals is not None:
            signals_debug = {
                "signals_source": "agent_tool_input",
                "special_targets": list(retrieval_signals.special_targets),
                "quoted_terms": list(retrieval_signals.quoted_terms),
            }
        query_options = QueryOptions(
            max_context_tokens=self.max_context_tokens,
            retrieval_signals=retrieval_signals,
            retrieval_signals_debug=signals_debug,
        )
        p = await self.retrieval_service.aretrieve_payload(
            payload.query,
            access_policy=access_policy,
            query_options=query_options,
        )
        # EvidenceBundle 有三个子列表：internal / external / graph
        bundle = p.evidence
        all_items = list(
            getattr(bundle, "internal", [])
            + getattr(bundle, "external", [])
            + getattr(bundle, "graph", [])
        )
        return _evidence_to_output(all_items)

    async def _via_aquery(self, payload: SearchInput, access_policy: AccessPolicy) -> SearchOutput:
        result = await self.runtime.aquery(
            payload.query,
            access_policy=access_policy,
        )
        if hasattr(result, "evidence"):
            return _evidence_to_output(result.evidence)
        if hasattr(result, "answer") and hasattr(result.answer, "answer_sections"):
            items: list[dict[str, object]] = []
            for section in result.answer.answer_sections:
                if getattr(section, "text", "").strip():
                    items.append({"text": section.text})
            return SearchOutput(items=items)
        return SearchOutput(items=[])

    async def _via_to_thread(self, payload: SearchInput) -> SearchOutput:
        from rag.retrieval import QueryOptions

        retrieval_signals = getattr(payload, "retrieval_signals", None)
        query_kwargs: dict[str, object] = {
            "max_context_tokens": self.max_context_tokens,
            "retrieval_signals": retrieval_signals,
        }
        if retrieval_signals is not None:
            query_kwargs["retrieval_signals_debug"] = {
                "signals_source": "agent_tool_input",
                "special_targets": list(retrieval_signals.special_targets),
                "quoted_terms": list(retrieval_signals.quoted_terms),
            }
        result = await asyncio.to_thread(
            self.runtime.query,
            payload.query,
            options=QueryOptions(**query_kwargs),
        )
        items: list[dict[str, object]] = []
        if hasattr(result, "evidence") and result.evidence:
            items = _evidence_to_output(result.evidence).items
        if not items and hasattr(result, "answer"):
            for section in getattr(result.answer, "answer_sections", []):
                text = getattr(section, "text", "")
                if text:
                    items.append({"text": text})
        return SearchOutput(items=items)

    def _resolve_access_policy(self, payload: SearchInput) -> AccessPolicy:
        if getattr(payload, "access_policy", None) is not None:
            return payload.access_policy
        if self.access_policy is not None:
            return self.access_policy
        if self.runtime is not None and getattr(self.runtime, "access_policy", None) is not None:
            return self.runtime.access_policy
        return AccessPolicy.default()


def _evidence_to_output(evidence: Any) -> SearchOutput:
    """将 EvidenceItem 列表转为 SearchOutput，保留 citation/source 元数据。"""
    items: list[dict[str, object]] = []
    for item in evidence:
        entry: dict[str, object] = {
            "text": getattr(item, "text", ""),
            "score": float(getattr(item, "score", 0.0)),
        }
        for field in (
            "evidence_id", "doc_id", "source_id", "citation_anchor",
            "file_name", "source_type", "record_type",
        ):
            value = getattr(item, field, None)
            if value is not None:
                entry[field] = value
        for field in ("section_path", "retrieval_channels"):
            value = getattr(item, field, None)
            if value:
                entry[field] = list(value)
        for field in ("page_start", "page_end", "benchmark_doc_id"):
            value = getattr(item, field, None)
            if value is not None:
                entry[field] = value
        items.append(entry)
    return SearchOutput(items=items)
