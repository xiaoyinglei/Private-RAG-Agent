from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from rag.agent.tools.rag_answer_tools import RAGSearchAnswerInput, RAGSearchAnswerRunner
from rag.agent.tools.rag_semantic_tools import (
    AssetResult,
    AssetSearchInput,
    AssetSearchOutput,
    KnowledgeResult,
    KnowledgeSearchInput,
    KnowledgeSearchOutput,
)
from rag.agent.tools.registry import ToolExecutionContext
from rag.storage.runtime_config import DEFAULT_VECTOR_BACKEND


@dataclass
class LazyRAGKnowledgeProvider:
    storage_root: Path = Path(".rag")
    model_alias: str | None = None
    embedding_model_alias: str | None = None
    reranker_model_alias: str | None = None
    vector_backend: str = DEFAULT_VECTOR_BACKEND
    vector_dsn: str | None = None
    vector_namespace: str | None = None
    vector_collection_prefix: str | None = None

    def __post_init__(self) -> None:
        self._runtime: Any | None = None
        self._runtime_context_entered = False
        self._diagnostics: tuple[Any, ...] = ()

    @property
    def diagnostics(self) -> tuple[Any, ...]:
        return self._diagnostics

    async def search_knowledge(
        self,
        payload: KnowledgeSearchInput,
        execution_context: ToolExecutionContext,
    ) -> KnowledgeSearchOutput:
        runtime = self._ensure_runtime()
        answer = await RAGSearchAnswerRunner(runtime=runtime).answer(
            RAGSearchAnswerInput(query=payload.query, top_k=payload.top_k),
            execution_context,
        )
        results = [
            KnowledgeResult(
                evidence_id=evidence.evidence_id,
                doc_id=str(evidence.doc_id),
                citation_anchor=evidence.citation_anchor,
                text=evidence.text,
                score=evidence.score,
                source_type=evidence.source_type or "",
                file_name=evidence.file_name or "",
            )
            for evidence in answer.evidence
        ]
        return KnowledgeSearchOutput(
            results=results,
            answer_text=answer.text,
            citations=[
                citation.citation_anchor or citation.citation_id
                for citation in answer.citations
            ],
            groundedness_flag=answer.groundedness_flag,
            insufficient_evidence=answer.insufficient_evidence,
            total_found=len(results),
        )

    async def search_assets(
        self,
        payload: AssetSearchInput,
        execution_context: ToolExecutionContext,
    ) -> AssetSearchOutput:
        del execution_context
        runtime = self._ensure_runtime()
        stores = getattr(runtime, "stores", None)
        metadata_repo = getattr(stores, "metadata_repo", None)
        object_store = getattr(stores, "object_store", None)
        if metadata_repo is None or object_store is None:
            raise RuntimeError("Knowledge asset search requires RAG metadata and object stores.")

        from rag.agent.tools.asset_tools import AssetInspectInput, AssetListInput, AssetToolRunner

        asset_runner = AssetToolRunner(
            metadata_repo=metadata_repo,
            object_store=object_store,
        )
        list_out = asset_runner.list_assets(
            AssetListInput(
                doc_id=payload.doc_id,
                asset_type=payload.asset_type,
                limit=payload.max_results,
            )
        )
        results: list[AssetResult] = []
        for asset in list_out.assets[:payload.max_results]:
            result = AssetResult(
                asset_id=asset.asset_id,
                doc_id=asset.doc_id,
                asset_type=asset.asset_type,
                sheet_name=asset.sheet_name,
                caption=asset.caption,
                columns=list(asset.columns or []),
                row_count=asset.row_count,
                column_count=asset.column_count,
            )
            if payload.include_preview and asset.asset_id:
                try:
                    preview = asset_runner.inspect_asset(
                        AssetInspectInput(asset_id=asset.asset_id, head_rows=3)
                    )
                except Exception:
                    preview = None
                if preview is not None:
                    if preview.head_rows:
                        result.preview_rows = preview.head_rows[:3]
                    result.analysis_capabilities = list(preview.analysis_capabilities or [])
            results.append(result)
        return AssetSearchOutput(
            assets=results,
            total_found=len(list_out.assets),
            truncated=len(list_out.assets) > len(results),
        )

    def close(self) -> None:
        runtime = self._runtime
        if runtime is None or not self._runtime_context_entered:
            return
        exit_method = getattr(runtime, "__exit__", None)
        if callable(exit_method):
            exit_method(None, None, None)
        self._runtime_context_entered = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    def _ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime

        from agent_runtime.runtime.builder import build_optional_rag_runtime

        runtime, diagnostics = build_optional_rag_runtime(
            storage_root=self.storage_root,
            model_alias=self.model_alias,
            embedding_model_alias=self.embedding_model_alias,
            reranker_model_alias=self.reranker_model_alias,
            vector_backend=self.vector_backend,
            vector_dsn=self.vector_dsn,
            vector_namespace=self.vector_namespace,
            vector_collection_prefix=self.vector_collection_prefix,
            explicit=True,
        )
        self._diagnostics = tuple(diagnostics)
        if runtime is None:
            raise RuntimeError(
                "Knowledge provider is not available. Configure --knowledge with "
                "an indexed RAG storage root before using search_knowledge."
            )

        enter_method = getattr(runtime, "__enter__", None)
        if callable(enter_method):
            runtime = enter_method()
            self._runtime_context_entered = True
        self._runtime = runtime
        return runtime
