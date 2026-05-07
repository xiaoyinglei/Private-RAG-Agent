from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rag.assembly import (
    AssemblyDiagnostics,
    AssemblyRequest,
    CapabilityAssemblyService,
    CapabilityBundle,
    CapabilityCatalog,
    CapabilityRequirements,
    ChatCapabilityBinding,
    ProviderConfig,
    TokenAccountingService,
    TokenizerContract,
)
from rag.assembly.support import build_provider
from rag.ingest.parsers import (
    DoclingParserRepo,
    ExcelParserRepo,
    ExtractionDispatcher,
    ImageParserRepo,
    PptxParserRepo,
    create_default_ocr_repo,
)
from rag.ingest.pipeline import (
    BatchIngestItemResult,
    BatchIngestResult,
    DirectContentItem,
    IngestPipeline,
    IngestPipelineResult,
    IngestRequest,
)
from rag.ingest.retrievalsummarizer import RetrievalSummarizer
from rag.ingest.section_refiner import SectionRefiner
from rag.ingest.table_executor import ComputeResult, TableExecutor
from rag.utils.guard import RateLimitExceeded
from rag.providers.citation_formatter import CitationFormatter
from rag.providers.generation import AnswerGenerationService, AnswerGenerator, GeneratorBinding
from rag.retrieval.analysis import QueryUnderstandingService, RoutingService
from rag.retrieval.authorization_service import AuthorizationService
from rag.retrieval.context import (
    ContextPromptBuilder,
    ContextPromptBuildResult,
    ContextTruncationResult,
    EvidenceTruncator,
)
from rag.retrieval.evidence import ContextEvidenceMerger, EvidenceService
from rag.retrieval.graph import GraphExpansionService, SearchBackedRetrievalFactory
from rag.retrieval.grounding_service import GroundingService
from rag.retrieval.models import (
    BuiltContext,
    PublicQueryResult,
    QueryOptions,
    RAGQueryResult,
    RetrievalProfile,
)
from rag.retrieval.orchestrator import RetrievalService
from rag.retrieval.planning_graph import PlanningGraph
from rag.retrieval.rerank_service import ModelBackedRerankService
from rag.retrieval.runtime_coordinator import (
    RuntimeCoordinator,
    build_retrieval_diagnostics,
    to_retrieval_result,
)
from rag.retrieval.synthesis_service import SynthesisService
from rag.schema.core import Document, ProcessingStateRecord, Source, SourceType, StorageTier
from rag.schema.graph import GraphEdge, GraphNode
from rag.schema.query import EvidenceItem
from rag.schema.runtime import AccessPolicy, CacheEntry, ProviderAttempt, VisualDescriptionRepo
from rag.storage import StorageBundle, StorageConfig
from rag.storage.data_contract_service import DataContractService
from rag.storage.index_sync_worker import IndexSyncWorker
from rag.storage.storage_lifecycle_service import StorageLifecycleService
from rag.storage.storage_lifecycle_worker import StorageLifecycleWorker
from rag.utils.telemetry import TelemetryService

_RUNTIME_CONTRACT_NAMESPACE = "rag_runtime"
_RUNTIME_CONTRACT_KEY = "core_contract_v1"
DEFAULT_SUMMARY_PROVIDER_KIND = "local-hf"
DEFAULT_SUMMARY_MODEL = "Qwen/Qwen3-8B-MLX-4bit"
DEFAULT_SUMMARY_BACKEND = "mlx"


def _supports_data_contract_metadata_repo(repo: object) -> bool:
    required = (
        "find_document_by_hash",
        "increment_document_reference_count",
        "save_source",
        "save_document",
        "save_section",
        "save_asset",
        "get_document",
        "list_documents",
        "get_section",
        "list_sections",
        "get_asset",
        "list_assets",
        "get_layout_meta_cache",
        "save_processing_state",
        "get_processing_state",
        "list_processing_states",
        "delete_processing_state",
        "deactivate_document",
        "set_document_index_state",
        "set_document_storage_tier",
    )
    return all(callable(getattr(repo, name, None)) for name in required)


def _supports_summary_index_repo(repo: object) -> bool:
    required = ("search", "delete", "upsert_record", "upsert_records")
    return all(callable(getattr(repo, name, None)) for name in required)


def _supports_storage_lifecycle_repo(repo: object) -> bool:
    required = (
        "get_document",
        "list_documents",
        "get_processing_state",
        "save_processing_state",
        "list_processing_states",
        "set_document_storage_tier",
    )
    return all(callable(getattr(repo, name, None)) for name in required)


class _ChatGeneratorAdapter:
    def __init__(self, binding: object | None) -> None:
        self._binding = binding

    @property
    def provider_name(self) -> str | None:
        value = getattr(self._binding, "provider_name", None)
        return value if isinstance(value, str) and value else None

    @property
    def model_name(self) -> str | None:
        value = getattr(self._binding, "model_name", None)
        return value if isinstance(value, str) and value else None

    def generate_text(self, *, prompt: str) -> str:
        chat = getattr(self._binding, "chat", None)
        if not callable(chat):
            raise RuntimeError("chat generation capability is not configured")
        return str(chat(prompt))

    def generate_structured(self, *, prompt: str, schema: type[Any], **kwargs: Any) -> Any:
        backend = getattr(self._binding, "backend", None)
        generate_structured = getattr(backend, "generate_structured", None)
        if callable(generate_structured):
            return generate_structured(prompt=prompt, schema=schema, **kwargs)
        return schema.model_validate_json(self.generate_text(prompt=prompt))


class _LazySummaryGeneratorAdapter:
    def __init__(
        self,
        *,
        provider_builder: Callable[[ProviderConfig], object] = build_provider,
        provider_kind: str = DEFAULT_SUMMARY_PROVIDER_KIND,
        model: str | None = DEFAULT_SUMMARY_MODEL,
        model_path: str | None = None,
        backend: str | None = DEFAULT_SUMMARY_BACKEND,
    ) -> None:
        self._provider_builder = provider_builder
        self._provider_kind = provider_kind
        self._model = model
        self._model_path = model_path
        self._backend = backend
        self._adapter: _ChatGeneratorAdapter | None = None

    @property
    def provider_name(self) -> str | None:
        return self._adapter.provider_name if self._adapter is not None else self._provider_kind

    @property
    def model_name(self) -> str | None:
        if self._adapter is not None and self._adapter.model_name:
            return self._adapter.model_name
        return self._model or self._model_path

    def _ensure_adapter(self) -> _ChatGeneratorAdapter:
        if self._adapter is None:
            provider_config = ProviderConfig(
                provider_kind=self._provider_kind,
                location="local",
                chat_model=self._model,
                chat_model_path=self._model_path,
                chat_backend=self._backend,
            )
            provider = self._provider_builder(provider_config)
            binding = ChatCapabilityBinding(provider, location=provider_config.location)
            self._adapter = _ChatGeneratorAdapter(binding)
        return self._adapter

    def generate_text(self, *, prompt: str) -> str:
        return self._ensure_adapter().generate_text(prompt=prompt)

    def generate_structured(self, *, prompt: str, schema: type[Any], **kwargs: Any) -> Any:
        return self._ensure_adapter().generate_structured(prompt=prompt, schema=schema, **kwargs)


def _generator_bindings_from_chat_bindings(chat_bindings: Sequence[object]) -> tuple[GeneratorBinding, ...]:
    bindings: list[GeneratorBinding] = []
    for binding in chat_bindings:
        location = str(getattr(binding, "location", "") or "local")
        bindings.append(
            GeneratorBinding(
                backend=_ChatGeneratorAdapter(binding),
                provider_name=str(getattr(binding, "provider_name", "chat")),
                model_name=getattr(binding, "model_name", None),
                location="cloud" if location == "cloud" else "local",
            )
        )
    return tuple(bindings)


class _InstrumentedReranker:
    def __init__(self, rerank_service: object) -> None:
        self._rerank_service = rerank_service
        self.provider_name = getattr(rerank_service, "provider_name", "formal-rerank")
        self.rerank_model_name = getattr(rerank_service, "rerank_model_name", "unconfigured-reranker")
        self.last_provider: str | None = self.provider_name
        self.last_attempts: list[ProviderAttempt] = []

    def __call__(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]:
        return self.rerank(query, documents, **kwargs)

    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]:
        attempt = ProviderAttempt(
            stage="rerank",
            capability="rerank",
            provider=self.provider_name,
            location="core",
            model=self.rerank_model_name,
            status="success",
        )
        rerank = getattr(self._rerank_service, "rerank", None)
        if not callable(rerank):
            self.last_attempts = [attempt.model_copy(update={"status": "failed", "error": "rerank not supported"})]
            raise RuntimeError("rerank not supported")
        try:
            result = list(rerank(query, documents, **kwargs))
        except RuntimeError as exc:
            self.last_attempts = [attempt.model_copy(update={"status": "failed", "error": str(exc)})]
            raise
        self.provider_name = getattr(self._rerank_service, "provider_name", self.provider_name)
        self.rerank_model_name = getattr(self._rerank_service, "rerank_model_name", self.rerank_model_name)
        self.last_provider = self.provider_name
        self.last_attempts = [
            attempt.model_copy(update={"provider": self.provider_name, "model": self.rerank_model_name})
        ]
        return result


@dataclass(slots=True, frozen=True)
class _LLMFallbackResult:
    answer: object = field(default_factory=lambda: type("_FallbackAnswer", (), {"answer_text": "抱歉，回答生成服务暂时不可用，请稍后重试。"})())
    provider: str = "fallback"
    model: str = "none"
    attempts: int = 1


@dataclass(slots=True)
class _QueryPipeline:
    retrieval: RetrievalService
    context_merger: ContextEvidenceMerger
    grounding_service: GroundingService | object
    truncator: EvidenceTruncator
    prompt_builder: ContextPromptBuilder
    answer_generator: AnswerGenerator
    synthesis_service: SynthesisService | object | None = None
    authorization_service: AuthorizationService | object | None = None
    table_executor: TableExecutor | object | None = None
    rate_limiter: object | None = None
    llm_circuit_breaker: object | None = None
    _compute_executed: bool = field(default=False, init=False, repr=False)

    _citation_formatter: CitationFormatter = field(default_factory=CitationFormatter, init=False, repr=False)

    @staticmethod
    def _run_async(awaitable: Any) -> Any:
        return RuntimeCoordinator().run_sync(awaitable)

    def _render_answer(self, generated: object) -> object:
        answer = getattr(generated, "answer", None)
        if answer is None:
            return generated
        formatted = self._citation_formatter.format(answer)
        rendered = answer.model_copy(update={"answer_text": formatted.answer_text})
        return replace(generated, answer=rendered)

    def _generate_with_breaker(self, awaitable: Any) -> Any:
        breaker = self.llm_circuit_breaker
        if breaker is not None and not breaker.allow():
            _logger = logging.getLogger("rag.runtime")
            _logger.warning("LLM circuit breaker open, returning fallback")
            return _LLMFallbackResult()
        try:
            result = self._run_async(awaitable)
        except Exception:
            if breaker is not None:
                breaker.on_failure()
            raise
        else:
            if breaker is not None:
                breaker.on_success()
            return result

    def run(
        self,
        query: str,
        *,
        options: QueryOptions,
    ) -> RAGQueryResult:
        if self.rate_limiter is not None:
            user_id = options.user_id or "anonymous"
            if not self.rate_limiter.allow(user_id=user_id):
                raise RateLimitExceeded(f"rate limit exceeded for user '{user_id}'")
        access_policy, source_scope = self._resolve_query_scope(options)
        retrieval_payload = self._retrieve_payload(
            query=query,
            access_policy=access_policy,
            source_scope=source_scope,
            options=options,
        )
        retrieval = (
            to_retrieval_result(retrieval_payload)
            if retrieval_payload is not None
            else self.retrieval.retrieve(
                query,
                access_policy=access_policy,
                source_scope=source_scope,
                execution_location_preference=options.execution_location_preference,
                query_options=options,
            )
        )
        if options.resolved_retrieval_profile is RetrievalProfile.BYPASS:
            prompt = self.prompt_builder.answer_generation_service.build_direct_prompt(
                query=query,
                response_type=options.response_type,
                user_prompt=options.user_prompt,
                conversation_history=options.conversation_history,
            )
            generated = self._generate_with_breaker(
                self.answer_generator.generate_direct(
                    query=query,
                    prompt=prompt,
                    access_policy=access_policy,
                    execution_location_preference=options.execution_location_preference,
                )
            )
            return RAGQueryResult(
                query=query,
                retrieval_profile=options.resolved_retrieval_profile.value,
                answer=self._render_answer(generated).answer,
                retrieval=retrieval,
                context=BuiltContext(
                    evidence=[],
                    token_budget=options.max_context_tokens,
                    token_count=self.prompt_builder.token_accounting.count(prompt),
                    truncated_count=0,
                    grounded_candidate="Bypass mode does not use retrieved evidence.",
                    prompt=prompt,
                ),
                generation_provider=generated.provider,
                generation_model=generated.model,
                generation_attempts=generated.attempts,
            )
        merged_evidence = self.context_merger.merge(retrieval_payload or retrieval)
        grounding_service = getattr(self, "grounding_service", None)
        if grounding_service is not None and callable(getattr(grounding_service, "ground", None)):
            merged_evidence = list(grounding_service.ground(query=query, evidence=merged_evidence))
        merged_evidence = self._section_diversity_filter(merged_evidence)
        synthesis_service = getattr(self, "synthesis_service", None)
        if synthesis_service is not None and callable(getattr(synthesis_service, "filter_evidence", None)):
            merged_evidence = list(
                synthesis_service.filter_evidence(
                    evidence=merged_evidence,
                    access_policy=access_policy,
                    user_id=options.user_id,
                )
            )
        total_budget = max(options.max_context_tokens, 1)
        evidence_budget = self.truncator.token_accounting.prompt_budget(total_budget)
        truncated, prompt_build = self._build_bounded_context(
            query=query,
            options=options,
            retrieval=retrieval,
            merged_evidence=merged_evidence,
            total_budget=total_budget,
            evidence_budget=evidence_budget,
        )
        context_evidence_items = [item.as_evidence_item() for item in truncated.evidence]
        generated = self._generate_with_breaker(
            self.answer_generator.generate(
                query=query,
                prompt=prompt_build.prompt,
                evidence_pack=context_evidence_items,
                grounded_candidate=prompt_build.grounded_candidate,
                runtime_mode=retrieval.decision.runtime_mode,
                access_policy=access_policy,
                execution_location_preference=options.execution_location_preference,
            )
        )
        generated, merged_evidence, truncated, prompt_build = self._maybe_execute_compute_loop(
            generated=generated,
            merged_evidence=merged_evidence,
            query=query,
            options=options,
            retrieval=retrieval,
            total_budget=total_budget,
            evidence_budget=evidence_budget,
            access_policy=access_policy,
        )
        return RAGQueryResult(
            query=query,
            retrieval_profile=options.resolved_retrieval_profile.value,
            answer=self._render_answer(generated).answer,
            retrieval=retrieval,
            context=BuiltContext(
                evidence=truncated.evidence,
                token_budget=total_budget,
                token_count=prompt_build.token_count,
                truncated_count=truncated.truncated_count,
                grounded_candidate=prompt_build.grounded_candidate,
                prompt=prompt_build.prompt,
            ),
            generation_provider=generated.provider,
            generation_model=generated.model,
            generation_attempts=generated.attempts,
        )

    def run_public(
        self,
        query: str,
        *,
        options: QueryOptions,
    ) -> PublicQueryResult:
        if self.rate_limiter is not None:
            user_id = options.user_id or "anonymous"
            if not self.rate_limiter.allow(user_id=user_id):
                raise RateLimitExceeded(f"rate limit exceeded for user '{user_id}'")
        access_policy, source_scope = self._resolve_query_scope(options)
        retrieval_payload = self._retrieve_payload(
            query=query,
            access_policy=access_policy,
            source_scope=source_scope,
            options=options,
        )
        retrieval = (
            to_retrieval_result(retrieval_payload)
            if retrieval_payload is not None
            else self.retrieval.retrieve(
                query,
                access_policy=access_policy,
                source_scope=source_scope,
                execution_location_preference=options.execution_location_preference,
                query_options=options,
            )
        )
        if options.resolved_retrieval_profile is RetrievalProfile.BYPASS:
            prompt = self.prompt_builder.answer_generation_service.build_direct_prompt(
                query=query,
                response_type=options.response_type,
                user_prompt=options.user_prompt,
                conversation_history=options.conversation_history,
            )
            generated = self._generate_with_breaker(
                self.answer_generator.generate_direct(
                    query=query,
                    prompt=prompt,
                    access_policy=access_policy,
                    execution_location_preference=options.execution_location_preference,
                )
            )
            return PublicQueryResult(
                query=query,
                retrieval_profile=options.resolved_retrieval_profile.value,
                answer=self._render_answer(generated).answer,
                context=BuiltContext(
                    evidence=[],
                    token_budget=options.max_context_tokens,
                    token_count=self.prompt_builder.token_accounting.count(prompt),
                    truncated_count=0,
                    grounded_candidate="Bypass mode does not use retrieved evidence.",
                    prompt=prompt,
                ),
                routing_decision=retrieval.decision.model_dump(mode="json"),
                retrieval_diagnostics=(
                    build_retrieval_diagnostics(retrieval_payload)
                    if retrieval_payload is not None
                    else retrieval.diagnostics
                ),
                retrieval_self_check=retrieval.self_check.model_dump(mode="json"),
                generation_provider=generated.provider,
                generation_model=generated.model,
                generation_attempts=generated.attempts,
            )

        merged_evidence = self.context_merger.merge(retrieval_payload or retrieval)
        grounding_service = getattr(self, "grounding_service", None)
        if grounding_service is not None and callable(getattr(grounding_service, "ground", None)):
            merged_evidence = list(grounding_service.ground(query=query, evidence=merged_evidence))
        merged_evidence = self._section_diversity_filter(merged_evidence)
        synthesis_service = getattr(self, "synthesis_service", None)
        if synthesis_service is not None and callable(getattr(synthesis_service, "filter_evidence", None)):
            merged_evidence = list(
                synthesis_service.filter_evidence(
                    evidence=merged_evidence,
                    access_policy=access_policy,
                    user_id=options.user_id,
                )
            )
        total_budget = max(options.max_context_tokens, 1)
        evidence_budget = self.truncator.token_accounting.prompt_budget(total_budget)
        truncated, prompt_build = self._build_bounded_context(
            query=query,
            options=options,
            retrieval=retrieval,
            merged_evidence=merged_evidence,
            total_budget=total_budget,
            evidence_budget=evidence_budget,
        )
        context_evidence_items = [item.as_evidence_item() for item in truncated.evidence]
        generated = self._generate_with_breaker(
            self.answer_generator.generate(
                query=query,
                prompt=prompt_build.prompt,
                evidence_pack=context_evidence_items,
                grounded_candidate=prompt_build.grounded_candidate,
                runtime_mode=retrieval.decision.runtime_mode,
                access_policy=access_policy,
                execution_location_preference=options.execution_location_preference,
            )
        )
        generated, merged_evidence, truncated, prompt_build = self._maybe_execute_compute_loop(
            generated=generated,
            merged_evidence=merged_evidence,
            query=query,
            options=options,
            retrieval=retrieval,
            total_budget=total_budget,
            evidence_budget=evidence_budget,
            access_policy=access_policy,
        )
        return PublicQueryResult(
            query=query,
            retrieval_profile=options.resolved_retrieval_profile.value,
            answer=self._render_answer(generated).answer,
            context=BuiltContext(
                evidence=truncated.evidence,
                token_budget=total_budget,
                token_count=prompt_build.token_count,
                truncated_count=truncated.truncated_count,
                grounded_candidate=prompt_build.grounded_candidate,
                prompt=prompt_build.prompt,
            ),
            routing_decision=retrieval.decision.model_dump(mode="json"),
            retrieval_diagnostics=(
                build_retrieval_diagnostics(retrieval_payload)
                if retrieval_payload is not None
                else retrieval.diagnostics
            ),
            retrieval_self_check=retrieval.self_check.model_dump(mode="json"),
            generation_provider=generated.provider,
            generation_model=generated.model,
            generation_attempts=generated.attempts,
        )

    def _resolve_query_scope(self, options: QueryOptions) -> tuple[AccessPolicy, tuple[str, ...]]:
        access_policy = options.access_policy
        source_scope = options.source_scope
        authorization_service = getattr(self, "authorization_service", None)
        if authorization_service is not None and callable(getattr(authorization_service, "resolve_query", None)):
            auth_context = authorization_service.resolve_query(
                user_id=options.user_id,
                access_policy=options.access_policy,
                source_scope=options.source_scope,
            )
            access_policy = auth_context.access_policy
            source_scope = auth_context.source_scope
        return access_policy, source_scope

    _COMPUTE_REQUEST_PATTERN: re.Pattern[str] | None = field(default=None, init=False, repr=False)

    def _maybe_execute_compute_loop(
        self,
        *,
        generated: object,
        merged_evidence: list[EvidenceItem],
        query: str,
        options: QueryOptions,
        retrieval: object,
        total_budget: int,
        evidence_budget: int,
        access_policy: object,
    ) -> tuple[object, list[EvidenceItem], ContextTruncationResult, ContextPromptBuildResult]:
        def _recontext(
            ev: list[EvidenceItem],
        ) -> tuple[list[EvidenceItem], ContextTruncationResult, ContextPromptBuildResult]:
            t, pb = self._build_bounded_context(
                query=query, options=options, retrieval=retrieval,
                merged_evidence=ev, total_budget=total_budget,
                evidence_budget=evidence_budget,
            )
            return ev, t, pb

        def _passthrough() -> tuple[object, list[EvidenceItem], ContextTruncationResult, ContextPromptBuildResult]:
            ev, t, pb = _recontext(merged_evidence)
            return generated, ev, t, pb

        if self._compute_executed:
            return _passthrough()

        executor = self.table_executor
        if executor is None or not hasattr(executor, "execute"):
            return _passthrough()

        answer_text = getattr(getattr(generated, "answer", None), "answer_text", None)
        if not answer_text or not isinstance(answer_text, str):
            return _passthrough()

        match = self._compute_request_re().search(answer_text)
        if match is None:
            return _passthrough()

        try:
            payload = json.loads(match.group(1))
            asset_id = int(payload.get("asset_id", 0))
            sql = str(payload.get("sql", "")).strip()
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            return _passthrough()

        if not sql or asset_id <= 0:
            return _passthrough()

        compute_result = executor.execute(asset_id=asset_id, sql=sql)
        self._compute_executed = True

        if compute_result is None:
            updated_evidence = self._strip_system_instructions(merged_evidence)
            ev, t, pb = _recontext(updated_evidence)
            return generated, ev, t, pb

        updated_evidence = self._inject_compute_result(
            merged_evidence, asset_id=asset_id, result=compute_result,
        )
        stripped_evidence = self._strip_system_instructions(updated_evidence)
        ev, t, pb = _recontext(stripped_evidence)
        context_evidence_items = [item.as_evidence_item() for item in t.evidence]
        regenerated = self._generate_with_breaker(
            self.answer_generator.generate(
                query=query,
                prompt=pb.prompt,
                evidence_pack=context_evidence_items,
                grounded_candidate=pb.grounded_candidate,
                runtime_mode=getattr(retrieval, "decision", None) and getattr(
                    getattr(retrieval, "decision", None), "runtime_mode", None
                ),
                access_policy=access_policy,
                execution_location_preference=options.execution_location_preference,
            )
        )
        return regenerated, ev, t, pb

    def _compute_request_re(self) -> re.Pattern[str]:
        if self._COMPUTE_REQUEST_PATTERN is None:
            self._COMPUTE_REQUEST_PATTERN = re.compile(
                r"<compute_request>\s*(\{.*?\})\s*</compute_request>", re.DOTALL
            )
        return self._COMPUTE_REQUEST_PATTERN

    @staticmethod
    def _inject_compute_result(
        evidence: list[EvidenceItem],
        *,
        asset_id: int,
        result: ComputeResult,
    ) -> list[EvidenceItem]:
        anchor_pattern = re.compile(rf"\[TABLE_COMPUTE_ONLY:asset_id={asset_id}\]")
        updated: list[EvidenceItem] = []
        replaced = False
        for item in evidence:
            if not replaced and anchor_pattern.search(item.text):
                new_text = anchor_pattern.sub(result.markdown, item.text, count=1)
                updated.append(
                    item.model_copy(
                        update={
                            "text": new_text,
                            "score": 1.0,
                            "retrieval_channels": [*item.retrieval_channels, "compute_result"],
                        }
                    )
                )
                replaced = True
            else:
                updated.append(item)
        if not replaced:
            if evidence:
                result_item = evidence[0].model_copy(
                    update={
                        "evidence_id": f"compute_result:{asset_id}",
                        "text": result.markdown,
                        "score": 1.0,
                        "retrieval_channels": ["compute_result"],
                    }
                )
            else:
                result_item = EvidenceItem(
                    evidence_id=f"compute_result:{asset_id}",
                    doc_id=0,
                    citation_anchor=f"table@{asset_id}",
                    text=result.markdown,
                    score=1.0,
                    retrieval_channels=["compute_result"],
                )
            updated.append(result_item)
        return updated

    @staticmethod
    def _section_diversity_filter(evidence: list[EvidenceItem], *, max_per_section: int = 2) -> list[EvidenceItem]:
        """post-grounding 章节多样性过滤：同章最多保留 max_per_section 条。"""
        if len(evidence) <= max_per_section:
            return evidence
        grouped: dict[str, list[EvidenceItem]] = {}
        for item in evidence:
            path = getattr(item, "section_path", None)
            if path and len(path) >= 2:
                key = f"{item.doc_id}:{path[0]}:{path[1]}"
            else:
                key = f"{item.doc_id}:{getattr(item, 'evidence_id', id(item))}"
            grouped.setdefault(key, []).append(item)
        result: list[EvidenceItem] = []
        for key, group in grouped.items():
            sorted_group = sorted(group, key=lambda x: getattr(x, "score", 0), reverse=True)
            result.extend(sorted_group[:max_per_section])
        return result

    @staticmethod
    def _strip_system_instructions(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        instruction_pattern = re.compile(
            r"<system_instruction>.*?</system_instruction>", re.DOTALL
        )
        replacement = (
            "[SYSTEM_NOTIFICATION] The data analysis query has been executed successfully. "
            "The results are injected below. You are STRICTLY FORBIDDEN from requesting "
            "further computations. Synthesize the final answer directly."
        )
        updated: list[EvidenceItem] = []
        for item in evidence:
            new_text = instruction_pattern.sub(replacement, item.text)
            if new_text != item.text:
                updated.append(item.model_copy(update={"text": new_text}))
            else:
                updated.append(item)
        return updated

    def _retrieve_payload(
        self,
        *,
        query: str,
        access_policy: AccessPolicy,
        source_scope: tuple[str, ...],
        options: QueryOptions,
    ) -> object | None:
        retrieve_payload = getattr(self.retrieval, "retrieve_payload", None)
        if not callable(retrieve_payload):
            return None
        return retrieve_payload(
            query,
            access_policy=access_policy,
            source_scope=source_scope,
            execution_location_preference=options.execution_location_preference,
            query_options=options,
        )

    def _build_bounded_context(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: object,
        merged_evidence: list[EvidenceItem],
        total_budget: int,
        evidence_budget: int,
    ) -> tuple[ContextTruncationResult, ContextPromptBuildResult]:
        current_budget = max(evidence_budget, 1)
        truncated = self._truncate_evidence(merged_evidence, budget=current_budget, options=options)
        truncated, prompt_build, current_budget = self._shrink_to_budget(
            query=query,
            options=options,
            retrieval=retrieval,
            merged_evidence=merged_evidence,
            total_budget=total_budget,
            current_budget=current_budget,
            truncated=truncated,
            prompt_variants=(("full", options.user_prompt, options.conversation_history),),
        )
        if prompt_build.token_count > total_budget:
            truncated, prompt_build, _current_budget = self._shrink_to_budget(
                query=query,
                options=options,
                retrieval=retrieval,
                merged_evidence=merged_evidence,
                total_budget=total_budget,
                current_budget=current_budget,
                truncated=truncated,
                prompt_variants=(
                    ("compact", options.user_prompt, options.conversation_history),
                    ("compact", options.user_prompt, ()),
                    ("compact", None, ()),
                    ("minimal", None, ()),
                ),
            )
        return truncated, prompt_build

    def _shrink_to_budget(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: object,
        merged_evidence: list[EvidenceItem],
        total_budget: int,
        current_budget: int,
        truncated: ContextTruncationResult,
        prompt_variants: Sequence[tuple[str, str | None, Sequence[tuple[str, str]]]],
    ) -> tuple[ContextTruncationResult, ContextPromptBuildResult, int]:
        prompt_build = self._build_prompt_variants(
            query=query,
            options=options,
            retrieval=retrieval,
            truncated=truncated,
            prompt_variants=prompt_variants,
        )
        while prompt_build.token_count > total_budget and truncated.evidence and current_budget > 1:
            overflow = prompt_build.token_count - total_budget
            next_budget = max(current_budget - max(overflow, 1), 1)
            retruncated = self._truncate_evidence(merged_evidence, budget=next_budget, options=options)
            if (
                retruncated.token_count >= truncated.token_count
                and len(retruncated.evidence) >= len(truncated.evidence)
            ):
                break
            truncated = retruncated
            current_budget = next_budget
            prompt_build = self._build_prompt_variants(
                query=query,
                options=options,
                retrieval=retrieval,
                truncated=truncated,
                prompt_variants=prompt_variants,
            )
        return truncated, prompt_build, current_budget

    def _truncate_evidence(
        self,
        merged_evidence: list[EvidenceItem],
        *,
        budget: int,
        options: QueryOptions,
    ) -> ContextTruncationResult:
        max_evidence_items = options.resolved_max_evidence_items
        if options.answer_context_top_k is not None:
            max_evidence_items = min(max_evidence_items, max(options.answer_context_top_k, 1))
        return self.truncator.truncate(
            merged_evidence,
            token_budget=budget,
            max_evidence_items=max_evidence_items,
            retrieval_profile=options.resolved_retrieval_profile.value,
        )

    def _build_prompt_variants(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: object,
        truncated: ContextTruncationResult,
        prompt_variants: Sequence[tuple[str, str | None, Sequence[tuple[str, str]]]],
    ) -> ContextPromptBuildResult:
        last_prompt: ContextPromptBuildResult | None = None
        for prompt_style, user_prompt, conversation_history in prompt_variants:
            last_prompt = self._build_prompt_from_truncation(
                query=query,
                options=options,
                retrieval=retrieval,
                truncated=truncated,
                prompt_style=prompt_style,
                user_prompt=user_prompt,
                conversation_history=conversation_history,
            )
            if last_prompt.token_count <= options.max_context_tokens:
                return last_prompt
        assert last_prompt is not None
        clipped_prompt = self.prompt_builder.token_accounting.clip(
            last_prompt.prompt,
            options.max_context_tokens,
        )
        return ContextPromptBuildResult(
            grounded_candidate=last_prompt.grounded_candidate,
            prompt=clipped_prompt,
            token_count=self.prompt_builder.token_accounting.count(clipped_prompt),
        )

    def _build_prompt_from_truncation(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: object,
        truncated: ContextTruncationResult,
        prompt_style: str,
        user_prompt: str | None,
        conversation_history: Sequence[tuple[str, str]],
    ) -> ContextPromptBuildResult:
        context_evidence_items = [item.as_evidence_item() for item in truncated.evidence]
        grounded_candidate = self.answer_generator.grounded_candidate(
            query,
            context_evidence_items,
            query_understanding=retrieval.diagnostics.query_understanding,
        )
        return self.prompt_builder.build(
            query=query,
            grounded_candidate=grounded_candidate,
            evidence=truncated.evidence,
            runtime_mode=retrieval.decision.runtime_mode,
            response_type=options.response_type,
            user_prompt=user_prompt,
            conversation_history=conversation_history,
            prompt_style=prompt_style,
        )


@dataclass(frozen=True, slots=True)
class RuntimeDeleteResult:
    deleted_doc_ids: list[int]
    deleted_source_ids: list[int]
    deleted_vector_count: int


@dataclass(frozen=True, slots=True)
class RuntimeRebuildResult:
    rebuilt_doc_ids: list[int]
    results: list[IngestPipelineResult]


@dataclass(slots=True)
class RAGRuntime:
    storage: StorageConfig
    request: AssemblyRequest = field(default_factory=AssemblyRequest)
    assembly_service: CapabilityAssemblyService = field(default_factory=CapabilityAssemblyService, repr=False)
    telemetry_service: TelemetryService | None = None
    vlm_repo: VisualDescriptionRepo | None = None
    capability_bundle: CapabilityBundle = field(init=False, repr=False)
    token_contract: TokenizerContract = field(init=False, repr=False)
    token_accounting: TokenAccountingService = field(init=False, repr=False)
    stores: StorageBundle = field(init=False)
    ingest_pipeline: IngestPipeline = field(init=False, repr=False)
    retrieval_service: RetrievalService = field(init=False, repr=False)
    agent_service: object | None = field(init=False, default=None, repr=False)
    query_pipeline: _QueryPipeline = field(init=False, repr=False)
    index_sync_worker: IndexSyncWorker | None = field(init=False, default=None, repr=False)
    storage_lifecycle_worker: StorageLifecycleWorker | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.capability_bundle = self.assembly_service.assemble_request(self.request)
        self.token_contract = self.capability_bundle.token_contract
        self.token_accounting = self.capability_bundle.token_accounting
        self.stores = self.storage.build()
        self._register_or_validate_runtime_contract()
        self._build_pipelines()

    @classmethod
    def from_request(
        cls,
        *,
        storage: StorageConfig,
        request: AssemblyRequest,
        assembly_service: CapabilityAssemblyService | None = None,
        telemetry_service: TelemetryService | None = None,
        vlm_repo: VisualDescriptionRepo | None = None,
    ) -> RAGRuntime:
        return cls(
            storage=storage,
            request=request,
            assembly_service=assembly_service or CapabilityAssemblyService(),
            telemetry_service=telemetry_service,
            vlm_repo=vlm_repo,
        )

    @classmethod
    def from_profile(
        cls,
        *,
        storage: StorageConfig,
        profile_id: str,
        requirements: CapabilityRequirements | None = None,
        assembly_service: CapabilityAssemblyService | None = None,
        telemetry_service: TelemetryService | None = None,
        vlm_repo: VisualDescriptionRepo | None = None,
    ) -> RAGRuntime:
        service = assembly_service or CapabilityAssemblyService()
        request = service.request_for_profile(profile_id, requirements=requirements)
        return cls.from_request(
            storage=storage,
            request=request,
            assembly_service=service,
            telemetry_service=telemetry_service,
            vlm_repo=vlm_repo,
        )

    @property
    def diagnostics(self) -> AssemblyDiagnostics:
        return self.capability_bundle.diagnostics

    @property
    def catalog(self) -> CapabilityCatalog:
        return self.assembly_service.catalog_from_environment(config=self.request.config)

    @property
    def runtime_contract_payload(self) -> dict[str, str | int | bool]:
        return self.capability_bundle.runtime_contract_payload

    def configure_summary_generator(
        self,
        *,
        provider_kind: str,
        model: str | None = None,
        model_path: str | None = None,
        backend: str | None = None,
    ) -> None:
        provider_config = ProviderConfig(
            provider_kind=provider_kind,
            location="local",
            chat_model=model,
            chat_model_path=model_path,
            chat_backend=backend,
        )
        provider = build_provider(provider_config)
        binding = ChatCapabilityBinding(provider, location=provider_config.location)
        self.ingest_pipeline.configure_summarizer(
            RetrievalSummarizer(
                llm_client=_ChatGeneratorAdapter(binding),
                token_accounting=self.token_accounting,
            )
        )

    @property
    def selected_profile_id(self) -> str | None:
        return self.capability_bundle.selected_profile_id

    def diagnostics_payload(self) -> dict[str, object]:
        return {
            "status": self.diagnostics.status,
            "selected_profile_id": self.selected_profile_id,
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                }
                for issue in self.diagnostics.issues
            ],
            "decisions": [
                {
                    "capability": decision.capability,
                    "source": decision.source,
                    "provider_kind": decision.provider_kind,
                    "provider_name": decision.provider_name,
                    "model_name": decision.model_name,
                    "location": decision.location,
                    "reason": decision.reason,
                    "selected": decision.selected,
                }
                for decision in self.diagnostics.decisions
            ],
            "runtime_contract": self.runtime_contract_payload,
        }

    def close(self) -> None:
        close_grounding = getattr(self.query_pipeline.grounding_service, "close", None)
        if callable(close_grounding):
            close_grounding()
        self.stores.close()

    def __enter__(self) -> RAGRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def insert(self, request: IngestRequest | None = None, /, **kwargs: Any) -> IngestPipelineResult:
        self._register_or_validate_runtime_contract()
        result = self.ingest_pipeline.run(self._coerce_ingest_request(request, **kwargs))
        self.process_pending_index_sync(max_tasks=1)
        self.process_pending_storage_lifecycle(max_tasks=1)
        return result

    def insert_many(
        self,
        requests: list[IngestRequest],
        *,
        continue_on_error: bool = False,
    ) -> BatchIngestResult:
        self._register_or_validate_runtime_contract()
        if not requests:
            return BatchIngestResult(results=[])

        if not continue_on_error:
            batch_results = self.ingest_pipeline.run_many(requests)
            self.process_pending_index_sync(max_tasks=max(1, len(batch_results)))
            self.process_pending_storage_lifecycle(max_tasks=1)
            return BatchIngestResult(
                results=[
                    BatchIngestItemResult(request=request, result=result)
                    for request, result in zip(requests, batch_results, strict=True)
                ]
            )

        results: list[BatchIngestItemResult] = []
        for request in requests:
            try:
                result = self.insert(request)
            except Exception as exc:
                if not continue_on_error:
                    raise
                results.append(BatchIngestItemResult(request=request, error=str(exc)))
                continue
            results.append(BatchIngestItemResult(request=request, result=result))
        return BatchIngestResult(results=results)

    def insert_content_list(
        self,
        items: list[object],
        *,
        continue_on_error: bool = False,
    ) -> BatchIngestResult:
        requests = [
            self._coerce_direct_content_item(item, index=index)
            for index, item in enumerate(items, start=1)
        ]
        return self.insert_many(requests, continue_on_error=continue_on_error)

    def query(
        self,
        *args: Any,
        options: QueryOptions | None = None,
        **kwargs: Any,
    ) -> RAGQueryResult:
        query_text = self._coerce_query_text(*args, **kwargs)
        self._register_or_validate_runtime_contract()
        self.process_pending_index_sync(max_tasks=2)
        self.process_pending_storage_lifecycle(max_tasks=1)
        return self.query_pipeline.run(query_text, options=self._normalize_query_options(options))

    def query_public(
        self,
        *args: Any,
        options: QueryOptions | None = None,
        **kwargs: Any,
    ) -> PublicQueryResult:
        query_text = self._coerce_query_text(*args, **kwargs)
        self._register_or_validate_runtime_contract()
        self.process_pending_index_sync(max_tasks=2)
        self.process_pending_storage_lifecycle(max_tasks=1)
        return self.query_pipeline.run_public(query_text, options=self._normalize_query_options(options))

    def analyze_task(
        self,
        task: str | object,
        /,
        **kwargs: Any,
    ) -> object:
        self._register_or_validate_runtime_contract()
        agent_task_request_cls = self._agent_task_request_class()
        if isinstance(task, agent_task_request_cls):
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise TypeError(f"analyze_task request was provided both positionally and by keyword: {unexpected}")
            request = task
        else:
            if not isinstance(task, str) or not task.strip():
                raise TypeError("analyze_task requires a non-empty task string or AgentTaskRequest")
            request = agent_task_request_cls(user_query=task, **kwargs)
        agent_service = self._ensure_agent_service()
        return agent_service.run_task(
            request,
            access_policy=AccessPolicy.default(),
        )

    def delete(
        self,
        *,
        doc_id: int | str | None = None,
        source_id: int | str | None = None,
        location: str | None = None,
    ) -> RuntimeDeleteResult:
        self._register_or_validate_runtime_contract()
        data_contract_service = self._build_data_contract_service()
        if data_contract_service is None:
            raise RuntimeError("delete requires the v1 data contract metadata and summary index repositories")

        documents = self._resolve_documents_for_operation(
            doc_id=doc_id,
            source_id=source_id,
            location=location,
            active_only=True,
        )
        deleted_doc_ids: list[int] = []
        deleted_source_ids: list[int] = []
        deleted_vector_count = 0
        for document in documents:
            deleted_vector_count += self._count_existing_summary_vectors(document)
            self._save_runtime_processing_state(
                doc_id=document.doc_id,
                source_id=document.source_id,
                stage="delete",
                status="deleting",
            )
            data_contract_service.deactivate_document(document.doc_id)
            self.stores.metadata_repo.set_document_storage_tier(document.doc_id, storage_tier=StorageTier.COLD)
            self._save_runtime_processing_state(
                doc_id=document.doc_id,
                source_id=document.source_id,
                stage="delete",
                status="deleted",
            )
            deleted_doc_ids.append(document.doc_id)
            if document.source_id not in deleted_source_ids:
                deleted_source_ids.append(document.source_id)
        return RuntimeDeleteResult(
            deleted_doc_ids=deleted_doc_ids,
            deleted_source_ids=deleted_source_ids,
            deleted_vector_count=deleted_vector_count,
        )

    def rebuild(
        self,
        *,
        doc_id: int | str | None = None,
        source_id: int | str | None = None,
        location: str | None = None,
    ) -> RuntimeRebuildResult:
        self._register_or_validate_runtime_contract()
        documents = self._resolve_documents_for_operation(
            doc_id=doc_id,
            source_id=source_id,
            location=location,
            active_only=False,
        )
        rebuilt_doc_ids: list[int] = []
        results: list[IngestPipelineResult] = []
        for document in documents:
            source = self._source_for_document(document)
            self._save_runtime_processing_state(
                doc_id=document.doc_id,
                source_id=document.source_id,
                stage="rebuild",
                status="rebuilding",
            )
            try:
                request = self._rebuild_request_from_source(source=source, document=document)
            except Exception as exc:
                message = f"No rebuildable source payload available for doc_id={document.doc_id}: {exc}"
                self.stores.metadata_repo.set_document_index_state(
                    document.doc_id,
                    is_indexed=False,
                    index_ready=False,
                    last_index_error=message,
                )
                self._save_runtime_processing_state(
                    doc_id=document.doc_id,
                    source_id=document.source_id,
                    stage="rebuild",
                    status="failed",
                    error_message=message,
                )
                raise ValueError(message) from exc
            result = self.insert(request)
            rebuilt_doc_ids.append(result.doc_id)
            results.append(result)
        return RuntimeRebuildResult(rebuilt_doc_ids=rebuilt_doc_ids, results=results)

    def process_pending_index_sync(self, *, max_tasks: int = 1, lease_seconds: int = 60) -> int:
        worker = self.index_sync_worker
        if worker is None or max_tasks <= 0:
            return 0
        processed = worker.run_until_idle(max_tasks=max_tasks, lease_seconds=lease_seconds)
        return len(processed)

    def process_pending_storage_lifecycle(self, *, max_tasks: int = 1, lease_seconds: int = 60) -> int:
        worker = self.storage_lifecycle_worker
        if worker is None or max_tasks <= 0:
            return 0
        worker.service.enqueue_due_documents(limit=max_tasks)
        processed = worker.run_until_idle(max_tasks=max_tasks, lease_seconds=lease_seconds)
        return len(processed)

    def upsert_node(self, node: GraphNode, *, evidence_ids: list[str] | None = None) -> GraphNode:
        self.stores.graph_repo.save_node(node)
        if evidence_ids:
            self.stores.graph_repo.merge_node_evidence(node.node_id, evidence_ids)
        return node

    def upsert_edge(self, edge: GraphEdge, *, candidate: bool = False) -> GraphEdge:
        if candidate:
            self.stores.graph_repo.save_candidate_edge(edge)
        else:
            self.stores.graph_repo.save_edge(edge)
        return edge

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.stores.graph_repo.get_node(node_id)

    def list_nodes(self, *, node_type: str | None = None) -> list[GraphNode]:
        return self.stores.graph_repo.list_nodes(node_type=node_type)

    def delete_node(self, node_id: str) -> None:
        self.stores.graph_repo.delete_node(node_id)

    def get_edge(self, edge_id: str, *, include_candidates: bool = False) -> GraphEdge | None:
        return self.stores.graph_repo.get_edge(edge_id, include_candidates=include_candidates)

    def list_edges(self) -> list[GraphEdge]:
        return self.stores.graph_repo.list_edges()

    def delete_edge(self, edge_id: str, *, include_candidates: bool = True) -> None:
        self.stores.graph_repo.delete_edge(edge_id, include_candidates=include_candidates)

    def insert_custom_kg(
        self,
        *,
        nodes: list[GraphNode] | None = None,
        edges: list[GraphEdge] | None = None,
    ) -> dict[str, int]:
        for node in nodes or []:
            self.upsert_node(node)
        for edge in edges or []:
            self.upsert_edge(edge)
        return {
            "node_count": len(nodes or []),
            "edge_count": len(edges or []),
        }

    def _build_pipelines(self) -> None:
        from rag.utils.guard import CircuitBreaker, CircuitConfig, RateLimiter

        ocr_repo = create_default_ocr_repo()
        s3_breaker = CircuitBreaker("s3", CircuitConfig(failure_threshold=5, cooldown_seconds=30.0))
        llm_breaker = CircuitBreaker("llm", CircuitConfig(failure_threshold=3, cooldown_seconds=60.0))
        query_rate_limiter = RateLimiter()
        data_contract_service = self._build_data_contract_service()
        self.index_sync_worker = self._build_index_sync_worker(data_contract_service)
        self.storage_lifecycle_worker = self._build_storage_lifecycle_worker(data_contract_service)
        embedding_binding = (
            self.capability_bundle.embedding_bindings[0] if self.capability_bundle.embedding_bindings else None
        )
        if embedding_binding is None:
            raise RuntimeError("new ingest pipeline requires an embedding capability")
        if not _supports_summary_index_repo(self.stores.vector_repo):
            raise RuntimeError("new ingest pipeline requires a summary index repo with upsert_record support")
        dispatcher = ExtractionDispatcher(
            docling_parser=DoclingParserRepo(self.vlm_repo),
            excel_parser=ExcelParserRepo(),
            pptx_parser=PptxParserRepo(),
            image_parser=ImageParserRepo(ocr_repo),
        )
        self.ingest_pipeline = IngestPipeline(
            dispatcher=dispatcher,
            summarizer=RetrievalSummarizer(
                llm_client=_LazySummaryGeneratorAdapter(
                    provider_builder=self.assembly_service._build_provider,
                ),
                token_accounting=self.token_accounting,
            ),
            embedder=embedding_binding,
            metadata_repo=self.stores.metadata_repo,
            summary_repo=self.stores.vector_repo,
            object_store=self.stores.object_store,
            embedding_model_id=embedding_binding.model_name or embedding_binding.provider_name,
            section_refiner=SectionRefiner(token_accounting=self.token_accounting),
        )
        self.retrieval_service = self._build_retrieval_service()
        answer_generation_service = AnswerGenerationService()
        self.query_pipeline = _QueryPipeline(
            retrieval=self.retrieval_service,
            context_merger=ContextEvidenceMerger(),
            grounding_service=GroundingService(
                metadata_repo=self.stores.metadata_repo,
                object_store=self.stores.object_store,
                token_accounting=self.token_accounting,
                rerank_binding=(
                    self.capability_bundle.rerank_bindings[0]
                    if self.capability_bundle.rerank_bindings
                    else None
                ),
                s3_circuit_breaker=s3_breaker,
            ),
            synthesis_service=SynthesisService(
                metadata_repo=self.stores.metadata_repo,
                authorization_service=AuthorizationService(resolver=self.stores.metadata_repo),
            ),
            truncator=EvidenceTruncator(token_accounting=self.token_accounting),
            prompt_builder=ContextPromptBuilder(
                answer_generation_service=answer_generation_service,
                token_accounting=self.token_accounting,
            ),
            answer_generator=AnswerGenerator(
                answer_generation_service=answer_generation_service,
                generators=_generator_bindings_from_chat_bindings(self.capability_bundle.chat_bindings),
            ),
            authorization_service=AuthorizationService(resolver=self.stores.metadata_repo),
            table_executor=TableExecutor(
                object_store=self.stores.object_store,
                metadata_repo=self.stores.metadata_repo,
            ),
            rate_limiter=query_rate_limiter,
            llm_circuit_breaker=llm_breaker,
        )

    def _build_data_contract_service(self) -> DataContractService | None:
        if not _supports_data_contract_metadata_repo(self.stores.metadata_repo):
            return None
        if not _supports_summary_index_repo(self.stores.vector_repo):
            return None
        embedder = self.capability_bundle.embedding_bindings[0] if self.capability_bundle.embedding_bindings else None
        return DataContractService(
            metadata_repo=self.stores.metadata_repo,
            milvus_repo=self.stores.vector_repo,
            embedder=embedder,
        )

    @staticmethod
    def _build_index_sync_worker(data_contract_service: DataContractService | None) -> IndexSyncWorker | None:
        if data_contract_service is None or data_contract_service.index_sync_service is None:
            return None
        return IndexSyncWorker(
            index_sync_service=data_contract_service.index_sync_service,
            data_contract_service=data_contract_service,
        )

    def _build_storage_lifecycle_worker(
        self,
        data_contract_service: DataContractService | None,
    ) -> StorageLifecycleWorker | None:
        if data_contract_service is None:
            return None
        if not _supports_storage_lifecycle_repo(self.stores.metadata_repo):
            return None
        return StorageLifecycleWorker(
            service=StorageLifecycleService(
                metadata_repo=self.stores.metadata_repo,
                data_contract_service=data_contract_service,
            )
        )

    def _build_retrieval_service(self) -> RetrievalService:
        bundle = self.capability_bundle
        query_understanding_service = QueryUnderstandingService(chat_bindings=bundle.chat_bindings)
        retrieval_factory = SearchBackedRetrievalFactory(
            metadata_repo=self.stores.metadata_repo,
            graph_repo=self.stores.graph_repo,
        )
        planning_graph = PlanningGraph(
            metadata_scope_resolver=self.stores.metadata_repo,
            use_summary_hybrid_paths=True,
        )
        reranker_service = (
            ModelBackedRerankService(
                binding=bundle.rerank_bindings[0],
            )
            if bundle.rerank_bindings
            else None
        )
        instrumented_reranker = None if reranker_service is None else _InstrumentedReranker(reranker_service)
        return RetrievalService(
            vector_retriever=retrieval_factory.vector_retriever_from_repo(
                self.stores.vector_repo,
                bundle.embedding_bindings,
            ),
            local_retriever=(lambda _query, _scope, _understanding: []),
            global_retriever=(lambda _query, _scope, _understanding: []),
            section_retriever=(lambda _query, _scope, _understanding: []),
            special_retriever=retrieval_factory.special_retriever_from_repo(
                self.stores.vector_repo,
                bundle.embedding_bindings,
            ),
            metadata_retriever=(lambda _query, _scope, _understanding: []),
            graph_expander=retrieval_factory.graph_expander,
            web_retriever=retrieval_factory.web_retriever,
            reranker=instrumented_reranker,
            routing_service=RoutingService(),
            query_understanding_service=query_understanding_service,
            evidence_service=EvidenceService(),
            graph_expansion_service=GraphExpansionService(),
            telemetry_service=self.telemetry_service,
            metadata_scope_resolver=self.stores.metadata_repo,
            planning_graph=planning_graph,
        )

    def _build_agent_service(self, *, answer_generation_service: AnswerGenerationService) -> object:
        from rag.agent import AnalysisAgentService
        from rag.agent.critic import EvidenceCritic
        from rag.agent.executor import AgentExecutor
        from rag.agent.planner import AgentPlanner
        from rag.agent.synthesizer import AgentSynthesizer
        from rag.agent.understanding import TaskUnderstandingService

        bundle = self.capability_bundle
        task_understanding_service = TaskUnderstandingService(
            chat_bindings=bundle.chat_bindings,
            query_understanding_service=self.retrieval_service.query_understanding_service,
        )
        return AnalysisAgentService(
            task_understanding_service=task_understanding_service,
            planner=AgentPlanner(enable_llm=False),
            executor=AgentExecutor(
                retrieval_service=self.retrieval_service,
                critic=EvidenceCritic(),
            ),
            synthesizer=AgentSynthesizer(
                answer_generator=AnswerGenerator(
                    answer_generation_service=answer_generation_service,
                    chat_bindings=self.capability_bundle.chat_bindings,
                ),
            ),
        )

    def _ensure_agent_service(self) -> object:
        agent_service = self.agent_service
        if agent_service is None:
            agent_service = self._build_agent_service(answer_generation_service=AnswerGenerationService())
            self.agent_service = agent_service
        return agent_service

    @staticmethod
    def _agent_task_request_class() -> type[object]:
        from rag.agent import AgentTaskRequest

        return AgentTaskRequest

    def _register_or_validate_runtime_contract(self) -> None:
        payload = dict(self.capability_bundle.runtime_contract_payload)
        existing = self.stores.cache_repo.get_cache_entry(_RUNTIME_CONTRACT_KEY, namespace=_RUNTIME_CONTRACT_NAMESPACE)
        stored_payload = existing.payload if existing is not None and isinstance(existing.payload, dict) else None
        governance = self.assembly_service.govern_runtime_contract(
            bundle=self.capability_bundle,
            stored_payload=stored_payload,
        )
        if governance.should_persist:
            self.stores.cache_repo.save_cache_entry(
                CacheEntry(
                    namespace=_RUNTIME_CONTRACT_NAMESPACE,
                    cache_key=_RUNTIME_CONTRACT_KEY,
                    payload=payload,
                )
            )
            return
        governance.raise_for_invalid()

    def _normalize_query_options(self, options: QueryOptions | None) -> QueryOptions:
        if options is None:
            return QueryOptions(max_context_tokens=self.token_contract.max_context_tokens)
        if (
            options.max_context_tokens == QueryOptions().max_context_tokens
            and self.token_contract.max_context_tokens != QueryOptions().max_context_tokens
        ):
            return replace(options, max_context_tokens=self.token_contract.max_context_tokens)
        return options

    def _resolve_documents_for_operation(
        self,
        *,
        doc_id: int | str | None,
        source_id: int | str | None,
        location: str | None,
        active_only: bool,
    ) -> list[Document]:
        if doc_id is None and source_id is None and (location is None or not location.strip()):
            raise TypeError("doc_id, source_id, or location is required")

        documents: list[Document] = []
        if doc_id is not None:
            document = self.stores.metadata_repo.get_document(self._coerce_int_identifier(doc_id, "doc_id"))
            if document is not None and (not active_only or document.is_active):
                documents.append(document)
        if source_id is not None:
            documents.extend(
                self.stores.metadata_repo.list_documents(
                    source_id=self._coerce_int_identifier(source_id, "source_id"),
                    active_only=active_only,
                )
            )
        if location is not None and location.strip():
            for source in self.stores.metadata_repo.list_sources(location=location):
                documents.extend(
                    self.stores.metadata_repo.list_documents(
                        source_id=source.source_id,
                        active_only=active_only,
                    )
                )

        unique: dict[int, Document] = {}
        for document in documents:
            unique[document.doc_id] = document
        return list(unique.values())

    @staticmethod
    def _coerce_int_identifier(value: int | str, name: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be an integer") from exc

    def _source_for_document(self, document: Document) -> Source:
        source = self.stores.metadata_repo.get_source(document.source_id)
        if source is None:
            raise ValueError(f"source not found for doc_id={document.doc_id}")
        return source

    def _count_existing_summary_vectors(self, document: Document) -> int:
        count = 0
        if self.stores.vector_repo.get_entry(str(document.doc_id), item_kind="doc_summary") is not None:
            count += 1
        for section in self.stores.metadata_repo.list_sections(doc_id=document.doc_id):
            if self.stores.vector_repo.get_entry(str(section.section_id), item_kind="section_summary") is not None:
                count += 1
        for asset in self.stores.metadata_repo.list_assets(doc_id=document.doc_id):
            if self.stores.vector_repo.get_entry(str(asset.asset_id), item_kind="asset_summary") is not None:
                count += 1
        return count

    def _rebuild_request_from_source(self, *, source: Source, document: Document) -> IngestRequest:
        object_key = source.object_key
        if object_key is None or not object_key.strip():
            raise ValueError("source.object_key is empty")
        if not self.stores.object_store.exists(object_key):
            raise ValueError(f"object_store key does not exist: {object_key}")
        raw_bytes = self.stores.object_store.read_bytes(object_key)
        metadata = {str(key): str(value) for key, value in document.metadata_json.items()}
        source_type = SourceType(source.source_type)
        kwargs: dict[str, object] = {
            "location": source.location,
            "source_type": source_type,
            "owner": source.owner_id or "user",
            "title": document.title,
            "metadata": metadata,
        }
        if source_type in {
            SourceType.PLAIN_TEXT,
            SourceType.PASTED_TEXT,
            SourceType.BROWSER_CLIP,
            SourceType.WEB,
        }:
            kwargs["content_text"] = raw_bytes.decode("utf-8", errors="replace")
        else:
            kwargs["raw_bytes"] = raw_bytes
        return IngestRequest(**kwargs)

    def _save_runtime_processing_state(
        self,
        *,
        doc_id: int,
        source_id: int,
        stage: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        self.stores.metadata_repo.save_processing_state(
            ProcessingStateRecord(
                doc_id=doc_id,
                source_id=source_id,
                stage=stage,
                status=status,
                attempts=1,
                priority="normal",
                worker_id=None,
                lease_expires_at=None,
                error_message=error_message,
                metadata_json={},
            )
        )

    @staticmethod
    def _coerce_query_text(*args: Any, **kwargs: Any) -> str:
        query_text = kwargs.pop("query", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if args:
            if len(args) != 1:
                raise TypeError("query accepts exactly one positional query string")
            if query_text is not None:
                raise TypeError("query was provided both positionally and by keyword")
            query_text = args[0]
        if not isinstance(query_text, str) or not query_text.strip():
            raise TypeError("query requires a non-empty string")
        return query_text

    @staticmethod
    def _coerce_ingest_request(request: IngestRequest | None = None, /, **kwargs: Any) -> IngestRequest:
        if request is not None:
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise TypeError(f"insert request was provided both positionally and by keyword: {unexpected}")
            return request
        normalized_kwargs = {"owner": "user", **kwargs}
        if "file_path" in normalized_kwargs and normalized_kwargs["file_path"] is not None:
            normalized_kwargs["file_path"] = Path(normalized_kwargs["file_path"])
        return IngestRequest(**normalized_kwargs)

    @staticmethod
    def _coerce_direct_content_item(item: object, *, index: int) -> IngestRequest:
        if isinstance(item, IngestRequest):
            return item
        if isinstance(item, DirectContentItem):
            content = item.content
            if isinstance(content, Path):
                return IngestRequest(
                    location=item.location or str(content),
                    source_type=item.source_type,
                    owner=item.owner,
                    title=item.title,
                    file_path=content,
                    metadata=dict(item.metadata),
                )
            if isinstance(content, bytes):
                return IngestRequest(
                    location=item.location,
                    source_type=item.source_type,
                    owner=item.owner,
                    title=item.title,
                    raw_bytes=content,
                    metadata=dict(item.metadata),
                )
            return IngestRequest(
                location=item.location,
                source_type=item.source_type,
                owner=item.owner,
                title=item.title,
                content_text=str(content),
                metadata=dict(item.metadata),
            )
        if isinstance(item, Path):
            return IngestRequest(
                location=str(item),
                source_type="plain_text",
                owner="user",
                file_path=item,
            )
        if isinstance(item, str):
            return IngestRequest(
                location=f"memory://content-{index}",
                source_type="plain_text",
                owner="user",
                content_text=item,
            )
        if isinstance(item, dict):
            payload = dict(item)
            if "file_path" in payload and payload["file_path"] is not None:
                payload["file_path"] = Path(payload["file_path"])
            return IngestRequest(**payload)
        raise TypeError(f"unsupported content item type: {type(item).__name__}")

__all__ = ["RAGRuntime", "_RUNTIME_CONTRACT_KEY", "_RUNTIME_CONTRACT_NAMESPACE"]
