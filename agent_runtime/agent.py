from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from agent_runtime.knowledge_providers.rag import LazyRAGKnowledgeProvider
from agent_runtime.models import ModelControlPlane, ModelSpec
from agent_runtime.result import AgentResult
from rag.agent.tools.registry import ContextualToolRunner
from rag.storage.runtime_config import DEFAULT_VECTOR_BACKEND


class Agent:
    def __init__(
        self,
        *,
        model: str | None = None,
        agent_type: str = "generic",
        checkpoint_db: Path | None = None,
        model_session_path: Path | None = None,
        knowledge: tuple[str, ...] | list[str] | None = None,
        rag_storage_root: Path = Path(".rag"),
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        vector_backend: str = DEFAULT_VECTOR_BACKEND,
        vector_dsn: str | None = None,
        vector_namespace: str | None = None,
        vector_collection_prefix: str | None = None,
    ) -> None:
        self.model = model
        self.agent_type = agent_type
        self.checkpoint_db = checkpoint_db
        self.model_session_path = model_session_path
        self.knowledge = tuple(knowledge or ())
        self.rag_storage_root = rag_storage_root
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.vector_backend = vector_backend
        self.vector_dsn = vector_dsn
        self.vector_namespace = vector_namespace
        self.vector_collection_prefix = vector_collection_prefix
        self._model_control_plane: ModelControlPlane | None = None

    def models(self) -> list[ModelSpec]:
        return self._get_model_control_plane().list_models()

    def current_model(self) -> ModelSpec:
        return self._get_model_control_plane().current_model()

    def switch_model(self, model_id: str) -> ModelSpec:
        return self._get_model_control_plane().switch_model(
            model_id,
            requested_by="user",
            persist=self.model_session_path is not None,
        )

    def run(
        self,
        task: str,
        *,
        files: list[str] | tuple[str, ...] | None = None,
        run_id: str | None = None,
        max_tokens_total: int | None = None,
    ) -> AgentResult:
        return asyncio.run(
            self.arun(
                task,
                files=files,
                run_id=run_id,
                max_tokens_total=max_tokens_total,
            )
        )

    async def arun(
        self,
        task: str,
        *,
        files: list[str] | tuple[str, ...] | None = None,
        run_id: str | None = None,
        max_tokens_total: int | None = None,
    ) -> AgentResult:
        from rag.agent.service import AgentRunRequest

        service, provider = self._build_service()
        effective_run_id = run_id or f"run_{id(service):x}"
        try:
            raw = await service.run(
                AgentRunRequest(
                    task=task,
                    run_id=effective_run_id,
                    thread_id=effective_run_id,
                    llm_budget_total=max_tokens_total,
                    input_files=list(files or ()),
                )
            )
            return AgentResult.from_internal(
                raw,
                files=tuple(files or ()),
            )
        finally:
            close_method = getattr(service, "aclose", None)
            if callable(close_method):
                await close_method()
            if provider is not None:
                provider.close()

    async def stream(
        self,
        task: str,
        *,
        files: list[str] | tuple[str, ...] | None = None,
        run_id: str | None = None,
        max_tokens_total: int | None = None,
    ) -> AsyncIterator[Any]:
        from rag.agent.service import AgentRunRequest

        service, provider = self._build_service()
        effective_run_id = run_id or f"run_{id(service):x}"
        try:
            request = AgentRunRequest(
                task=task,
                run_id=effective_run_id,
                thread_id=effective_run_id,
                llm_budget_total=max_tokens_total,
                input_files=list(files or ()),
            )
            async for event in service.run_streaming(request):
                yield event
        finally:
            close_method = getattr(service, "aclose", None)
            if callable(close_method):
                await close_method()
            if provider is not None:
                provider.close()

    def _build_service(self) -> tuple[Any, LazyRAGKnowledgeProvider | None]:
        from rag.agent.cli import _build_agent_service
        from rag.utils.text import load_env_file

        load_env_file()
        try:
            model_control_plane = self._get_model_control_plane()
        except Exception:
            if self.model is not None:
                raise
            model_control_plane = None
        provider: LazyRAGKnowledgeProvider | None = None
        knowledge_runner = None
        knowledge_asset_runner = None
        if self.knowledge:
            provider = LazyRAGKnowledgeProvider(
                storage_root=self.rag_storage_root,
                model_alias=self.model,
                embedding_model_alias=self.embedding_model,
                reranker_model_alias=self.reranker_model,
                vector_backend=self.vector_backend,
                vector_dsn=self.vector_dsn,
                vector_namespace=self.vector_namespace,
                vector_collection_prefix=self.vector_collection_prefix,
            )
            knowledge_runner = cast(ContextualToolRunner, provider.search_knowledge)
            knowledge_asset_runner = cast(ContextualToolRunner, provider.search_assets)

        service = _build_agent_service(
            None,
            checkpoint_db=self.checkpoint_db,
            agent_type=self.agent_type,
            model_alias=self.model,
            model_control_plane=model_control_plane,
            runtime_diagnostics=(),
            knowledge_runner=knowledge_runner,
            knowledge_asset_runner=knowledge_asset_runner,
        )
        return service, provider

    def _get_model_control_plane(self) -> ModelControlPlane:
        if self._model_control_plane is None:
            self._model_control_plane = ModelControlPlane.from_env(
                initial_model_id=self.model,
                session_path=self.model_session_path,
            )
        return self._model_control_plane
