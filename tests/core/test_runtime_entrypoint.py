from __future__ import annotations

import pytest

import rag.schema.query as query_schema
from rag import AssemblyRequest, RAGRuntime, StorageConfig
from rag.assembly import AssemblyConfig, CapabilityAssemblyService, CapabilityRequirements, ProviderConfig
from rag.retrieval import QueryOptions
from rag.retrieval.models import PublicQueryResult, RetrievalResult
from rag.schema.query import EvidenceItem, GroundingTarget


class _FakeProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.provider_name = config.provider_kind
        self.embedding_model_name = config.embedding_model
        self.chat_model_name = config.chat_model
        self.rerank_model_name = config.rerank_model
        self.is_embed_configured = bool(config.embedding_model)
        self.is_chat_configured = bool(config.chat_model)
        self.is_rerank_configured = bool(config.rerank_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def chat(self, prompt: str) -> str:
        return f"chat::{prompt}"

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        del query
        return list(range(len(candidates)))


def _assembly_service(monkeypatch: pytest.MonkeyPatch) -> CapabilityAssemblyService:
    service = CapabilityAssemblyService(env_path=".env.test-unused")
    monkeypatch.setattr(service, "_load_env", lambda: None)
    monkeypatch.setattr(
        service,
        "_compatibility_config_from_environment",
        lambda: (
            AssemblyConfig(
                profiles=(
                    ProviderConfig(
                        profile_id="openai-compatible",
                        provider_kind="openai-compatible",
                        api_key="cloud-key",
                        base_url="https://example.com/v1",
                        chat_model="cloud-chat",
                        embedding_model="cloud-embed",
                    ),
                    ProviderConfig(
                        profile_id="ollama",
                        provider_kind="ollama",
                        base_url="http://localhost:11434",
                        chat_model="local-chat",
                        embedding_model="local-embed",
                    ),
                    ProviderConfig(
                        profile_id="local-bge",
                        provider_kind="local-bge",
                        embedding_model="bge-m3",
                        rerank_model="bge-reranker-v2-m3",
                    ),
                )
            ),
            {},
        ),
    )
    monkeypatch.setattr(service, "_build_provider", lambda config: _FakeProvider(config))
    return service


def _empty_assembly_service(monkeypatch: pytest.MonkeyPatch) -> CapabilityAssemblyService:
    service = CapabilityAssemblyService(env_path=".env.test-unused")
    monkeypatch.setattr(service, "_load_env", lambda: None)
    monkeypatch.setattr(service, "_compatibility_config_from_environment", lambda: (AssemblyConfig(), {}))
    return service


def test_runtime_catalog_lists_compatibility_provider_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _assembly_service(monkeypatch)
    runtime = RAGRuntime.from_request(
        storage=StorageConfig.in_memory(),
        request=AssemblyRequest(),
        assembly_service=service,
    )
    try:
        profile_ids = {profile.profile_id for profile in runtime.catalog.profiles}
    finally:
        runtime.close()

    assert {"openai-compatible", "ollama", "local-bge"} <= profile_ids


def test_runtime_without_chat_binding_uses_visible_summary_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _empty_assembly_service(monkeypatch)
    runtime = RAGRuntime.from_request(
        storage=StorageConfig.in_memory(),
        request=AssemblyRequest(),
        assembly_service=service,
    )
    try:
        assert runtime.capability_bundle.chat_bindings == ()
        result = runtime.insert(
            source_type="plain_text",
            location="memory://runtime-no-chat",
            owner="test",
            content_text="Alpha Engine handles ingestion. Beta Service depends on Alpha Engine.",
        )
        entry = runtime.stores.vector_repo.get_entry(str(result.doc_id), item_kind="doc_summary")
    finally:
        runtime.close()

    assert entry is not None
    assert '"method": "fallback"' in entry.metadata["metadata_json"]
    assert '"fallback_reason": "RuntimeError"' in entry.metadata["metadata_json"]


def test_runtime_summary_generator_follows_chat_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summary generator 的模型来自 capability bundle 的 chat binding，不再独立配置。"""
    service = _assembly_service(monkeypatch)
    runtime = RAGRuntime.from_request(
        storage=StorageConfig.in_memory(),
        request=AssemblyRequest(requirements=CapabilityRequirements(require_chat=True)),
        assembly_service=service,
    )
    try:
        info = runtime.ingest_pipeline._summarizer.generator_info()
        chat_binding = runtime.capability_bundle.chat_bindings[0]
        assert info["model_name"] == chat_binding.model_name
    finally:
        runtime.close()


def test_public_retrieval_result_excludes_old_preservation_contract() -> None:
    assert "preservation_suggestion" not in PublicQueryResult.model_fields
    assert "preservation_suggestion" not in RetrievalResult.model_fields
    assert not hasattr(query_schema, "ArtifactType")
    assert not hasattr(query_schema, "PreservationSuggestion")


def test_runtime_no_longer_exposes_agent_analysis_facade() -> None:
    assert not hasattr(RAGRuntime, "analyze_task")


def test_runtime_from_request_round_trips_and_exposes_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _assembly_service(monkeypatch)
    runtime = RAGRuntime.from_request(
        storage=StorageConfig.in_memory(),
        request=AssemblyRequest(
            requirements=CapabilityRequirements(require_chat=True, default_context_tokens=1024),
        ),
        assembly_service=service,
    )
    retrieval_payload = None
    try:
        runtime.insert(
            source_type="plain_text",
            location="memory://runtime-profile",
            owner="test",
            content_text="Alpha Engine handles ingestion. Beta Service depends on Alpha Engine.",
        )
        result = runtime.query(
            "What does Alpha Engine handle?",
            options=QueryOptions(retrieval_profile="auto"),
        )
        retrieval_payload = runtime.retrieval_service.last_payload
    finally:
        runtime.close()

    assert runtime.diagnostics.status == "valid"
    diagnostics_payload = runtime.diagnostics_payload()
    assert "selected_profile_id" not in diagnostics_payload
    assert all(decision.capability != "assembly_profile" for decision in runtime.diagnostics.decisions)
    assert result.answer.answer_text
    assert result.context.evidence
    assert retrieval_payload is not None
    payload_evidence = retrieval_payload.evidence.all
    assert payload_evidence
    assert all(isinstance(item, EvidenceItem) for item in payload_evidence)
    assert all("chunk_id" not in item.model_dump() for item in payload_evidence)
    assert all(
        item.grounding_target is None or isinstance(item.grounding_target, GroundingTarget)
        for item in payload_evidence
    )
    assert all("chunk_id" not in item.model_dump() for item in result.context.evidence)
    assert all(
        item.grounding_target is None or isinstance(item.grounding_target, GroundingTarget)
        for item in result.context.evidence
    )


def test_runtime_from_request_uses_new_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _assembly_service(monkeypatch)
    request = AssemblyRequest()
    runtime = RAGRuntime.from_request(
        storage=StorageConfig.in_memory(),
        request=request,
        assembly_service=service,
    )
    try:
        runtime.insert(
            source_type="plain_text",
            location="memory://runtime-request",
            owner="test",
            content_text="Assembly request is the recommended entrypoint for runtime construction.",
        )
        result = runtime.query(
            "What is the recommended entrypoint for runtime construction?",
            options=QueryOptions(retrieval_profile="auto"),
        )
    finally:
        runtime.close()

    assert runtime.diagnostics.status == "valid"
    assert runtime.runtime_contract_payload["embedding_model_name"] == "cloud-embed"
    assert result.answer.answer_text
