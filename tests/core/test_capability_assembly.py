from __future__ import annotations

import pytest

from rag.assembly import (
    AssemblyConfig,
    AssemblyOverrides,
    AssemblyRequest,
    CapabilityAssemblyService,
    CapabilityRequirements,
    ProviderConfig,
    RuntimeContractGovernance,
    TokenizerConfig,
)


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


def _isolated_service(
    monkeypatch: pytest.MonkeyPatch,
    *,
    compatibility_config: AssemblyConfig | None = None,
) -> CapabilityAssemblyService:
    service = CapabilityAssemblyService(env_path=".env.test-unused")
    monkeypatch.setattr(service, "_load_env", lambda: None)
    monkeypatch.setattr(
        service,
        "_compatibility_config_from_environment",
        lambda: (compatibility_config or AssemblyConfig(), {}),
    )
    monkeypatch.setattr(service, "_build_provider", lambda config: _FakeProvider(config))
    return service


def test_capability_assembly_assembles_capabilities_from_structured_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _isolated_service(monkeypatch)
    bundle = service.assemble_request(
        AssemblyRequest(
            requirements=CapabilityRequirements(default_context_tokens=1536),
            overrides=AssemblyOverrides(
                embedding=ProviderConfig(
                    provider_kind="fake-core",
                    embedding_model="fake-embed-model",
                ),
                chat=ProviderConfig(
                    provider_kind="fake-core",
                    chat_model="fake-chat-model",
                ),
                rerank=ProviderConfig(
                    provider_kind="fake-core",
                    rerank_model="fake-rerank-model",
                ),
            ),
        )
    )

    assert len(bundle.embedding_bindings) == 1
    assert bundle.embedding_bindings[0].provider_name == "fake-core"
    assert bundle.embedding_bindings[0].model_name == "fake-embed-model"
    assert len(bundle.chat_bindings) == 1
    assert bundle.chat_bindings[0].provider_name == "fake-core"
    assert len(bundle.rerank_bindings) == 1
    assert bundle.rerank_bindings[0].model_name == "fake-rerank-model"
    assert bundle.token_contract.embedding_model_name == "fake-embed-model"
    assert bundle.runtime_contract_payload["embedding_model_name"] == "fake-embed-model"
    assert bundle.token_contract.max_context_tokens == 1536


def test_assembly_prefers_explicit_over_profile_config_and_compat_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = AssemblyConfig(
        profiles=(
            ProviderConfig(
                profile_id="compat-openai",
                provider_kind="openai-compatible",
                api_key="compat-key",
                base_url="https://example.com/v1",
                embedding_model="compat-embed",
                chat_model="compat-chat",
            ),
        )
    )
    service = _isolated_service(monkeypatch, compatibility_config=compatibility)

    bundle = service.assemble_request(
        AssemblyRequest(
            profile_id="profile-selected",
            config=AssemblyConfig(
                default_profile_id="config-default",
                profiles=(
                    ProviderConfig(
                        profile_id="profile-selected",
                        provider_kind="profile-provider",
                        embedding_model="profile-embed",
                        chat_model="profile-chat",
                    ),
                    ProviderConfig(
                        profile_id="config-default",
                        provider_kind="config-provider",
                        embedding_model="config-embed",
                        chat_model="config-chat",
                    ),
                ),
            ),
            overrides=AssemblyOverrides(
                embedding=ProviderConfig(
                    profile_id="explicit-embedding",
                    provider_kind="explicit-provider",
                    embedding_model="explicit-embed",
                ),
                chat=ProviderConfig(
                    profile_id="explicit-chat",
                    provider_kind="explicit-provider",
                    chat_model="explicit-chat",
                ),
            ),
        )
    )

    assert bundle.status == "valid"
    assert bundle.embedding_bindings[0].provider_name == "explicit-provider"
    assert bundle.embedding_bindings[0].model_name == "explicit-embed"
    assert bundle.chat_bindings[0].provider_name == "explicit-provider"
    assert bundle.chat_bindings[0].model_name == "explicit-chat"
    embedding_decision = next(
        decision
        for decision in bundle.diagnostics.decisions
        if decision.capability == "embedding" and decision.selected
    )
    chat_decision = next(
        decision
        for decision in bundle.diagnostics.decisions
        if decision.capability == "chat" and decision.selected
    )
    assert embedding_decision.source == "explicit"
    assert chat_decision.source == "explicit"
    assert bundle.token_contract.embedding_model_name == "explicit-embed"


def test_assembly_prefers_profile_over_config_and_compat_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = AssemblyConfig(
        profiles=(
            ProviderConfig(
                profile_id="compat-openai",
                provider_kind="openai-compatible",
                api_key="compat-key",
                base_url="https://example.com/v1",
                embedding_model="compat-embed",
                chat_model="compat-chat",
            ),
        )
    )
    service = _isolated_service(monkeypatch, compatibility_config=compatibility)

    bundle = service.assemble_request(
        AssemblyRequest(
            profile_id="profile-selected",
            config=AssemblyConfig(
                default_profile_id="config-default",
                profiles=(
                    ProviderConfig(
                        profile_id="profile-selected",
                        provider_kind="profile-provider",
                        embedding_model="profile-embed",
                        chat_model="profile-chat",
                    ),
                    ProviderConfig(
                        profile_id="config-default",
                        provider_kind="config-provider",
                        embedding_model="config-embed",
                        chat_model="config-chat",
                    ),
                ),
            ),
        )
    )

    assert bundle.status == "valid"
    assert bundle.embedding_bindings[0].provider_name == "profile-provider"
    assert bundle.embedding_bindings[0].model_name == "profile-embed"
    assert bundle.chat_bindings[0].provider_name == "profile-provider"
    assert bundle.chat_bindings[0].model_name == "profile-chat"
    assert bundle.selected_profile_id == "profile-selected"
    embedding_decision = next(
        decision
        for decision in bundle.diagnostics.decisions
        if decision.capability == "embedding" and decision.selected
    )
    assert embedding_decision.source == "profile"


def test_assembly_prefers_structured_config_over_compat_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = AssemblyConfig(
        profiles=(
            ProviderConfig(
                profile_id="compat-openai",
                provider_kind="openai-compatible",
                api_key="compat-key",
                base_url="https://example.com/v1",
                embedding_model="compat-embed",
                chat_model="compat-chat",
            ),
        )
    )
    service = _isolated_service(monkeypatch, compatibility_config=compatibility)

    bundle = service.assemble_request(
        AssemblyRequest(
            config=AssemblyConfig(
                embedding=ProviderConfig(
                    provider_kind="config-provider",
                    embedding_model="config-embed",
                ),
                chat=ProviderConfig(
                    provider_kind="config-provider",
                    chat_model="config-chat",
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_model_name="config-tokenizer",
                    chunking_tokenizer_model_name="config-chunker",
                ),
            )
        )
    )

    assert bundle.status == "valid"
    assert bundle.embedding_bindings[0].provider_name == "config-provider"
    assert bundle.embedding_bindings[0].model_name == "config-embed"
    assert bundle.chat_bindings[0].provider_name == "config-provider"
    assert bundle.chat_bindings[0].model_name == "config-chat"
    assert bundle.token_contract.tokenizer_model_name == "config-tokenizer"
    assert bundle.token_contract.chunking_tokenizer_model_name == "config-chunker"
    embedding_decision = next(
        decision
        for decision in bundle.diagnostics.decisions
        if decision.capability == "embedding" and decision.selected
    )
    assert embedding_decision.source == "config"


def test_assembly_uses_compat_env_then_default_with_structured_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _isolated_service(
        monkeypatch,
        compatibility_config=AssemblyConfig(
            profiles=(
                ProviderConfig(
                    profile_id="compat-openai",
                    provider_kind="openai-compatible",
                    api_key="compat-key",
                    base_url="https://example.com/v1",
                    embedding_model="compat-embed",
                    chat_model="compat-chat",
                ),
            )
        ),
    )
    compat_bundle = service.assemble_request(AssemblyRequest())

    assert compat_bundle.status == "valid"
    assert compat_bundle.embedding_bindings[0].provider_name == "openai-compatible"
    assert compat_bundle.embedding_bindings[0].model_name == "compat-embed"
    compat_decision = next(
        decision
        for decision in compat_bundle.diagnostics.decisions
        if decision.capability == "embedding" and decision.selected
    )
    assert compat_decision.source == "compat_env"

    fallback_service = _isolated_service(monkeypatch)
    fallback_bundle = fallback_service.evaluate_request(AssemblyRequest())

    assert fallback_bundle.status == "degraded"
    assert fallback_bundle.embedding_bindings[0].provider_name == "fallback"
    fallback_decision = next(
        decision
        for decision in fallback_bundle.diagnostics.decisions
        if decision.capability == "embedding" and decision.selected
    )
    assert fallback_decision.source == "default"
    assert any(issue.code == "fallback_embedding_selected" for issue in fallback_bundle.diagnostics.issues)


def test_assembly_reports_degraded_and_invalid_outcomes_for_missing_required_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _isolated_service(
        monkeypatch,
        compatibility_config=AssemblyConfig(
            profiles=(
                ProviderConfig(
                    profile_id="embed-only",
                    provider_kind="embed-only-provider",
                    embedding_model="embed-only-model",
                ),
            )
        ),
    )

    degraded_bundle = service.evaluate_request(
        AssemblyRequest(
            requirements=CapabilityRequirements(require_chat=True, allow_degraded=True),
        )
    )
    assert degraded_bundle.status == "degraded"
    assert not degraded_bundle.chat_bindings
    assert any(issue.code == "missing_required_chat" for issue in degraded_bundle.diagnostics.issues)

    invalid_bundle = service.evaluate_request(
        AssemblyRequest(
            requirements=CapabilityRequirements(require_chat=True, allow_degraded=False),
        )
    )
    assert invalid_bundle.status == "invalid"
    with pytest.raises(RuntimeError, match="No usable chat capability could be assembled"):
        invalid_bundle.diagnostics.raise_for_invalid()


def test_assembly_raises_for_invalid_request_and_runtime_contract_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _isolated_service(
        monkeypatch,
        compatibility_config=AssemblyConfig(
            profiles=(
                ProviderConfig(
                    profile_id="embed-only",
                    provider_kind="embed-only-provider",
                    embedding_model="embed-only-model",
                ),
            )
        ),
    )

    with pytest.raises(RuntimeError, match="No usable chat capability could be assembled"):
        service.assemble_request(
            AssemblyRequest(
                requirements=CapabilityRequirements(require_chat=True, allow_degraded=False),
            )
        )

    valid_bundle = service.assemble_request(AssemblyRequest())
    governance = service.govern_runtime_contract(
        bundle=valid_bundle,
        stored_payload={
            **valid_bundle.runtime_contract_payload,
            "embedding_model_name": "other-embed-model",
        },
    )
    assert isinstance(governance, RuntimeContractGovernance)
    assert governance.status == "invalid"
    assert "embedding_model_name" in governance.mismatches
    with pytest.raises(RuntimeError, match="runtime contract does not match"):
        governance.raise_for_invalid()


def test_capability_assembly_uses_legacy_env_profile_for_openai_compatible_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_BASE_URL",
        "GEMINI_API_KEY",
        "GEMINI_CHAT_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "GEMINI_BASE_URL",
        "OLLAMA_BASE_URL",
        "OLLAMA_CHAT_MODEL",
        "OLLAMA_EMBEDDING_MODEL",
        "PKP_OLLAMA__BASE_URL",
        "PKP_OLLAMA__CHAT_MODEL",
        "PKP_OLLAMA__EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PKP_OPENAI__API_KEY", "test-key")
    monkeypatch.setenv("PKP_OPENAI__MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("PKP_OPENAI__EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("PKP_OPENAI__BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

    service = CapabilityAssemblyService(env_path=".env.test-unused")
    monkeypatch.setattr(service, "_load_env", lambda: None)
    monkeypatch.setattr(service, "_build_provider", lambda config: _FakeProvider(config))
    catalog = service.catalog_from_environment()
    bundle = service.assemble_request(
        AssemblyRequest(
            profile_id="openai-compatible",
            requirements=CapabilityRequirements(),
        )
    )

    assert any(profile.profile_id == "openai-compatible" for profile in catalog.profiles)
    assert bundle.chat_bindings[0].provider_name == "openai-compatible"
    assert bundle.chat_bindings[0].model_name == "gemini-2.5-pro"
    assert bundle.embedding_bindings[0].model_name == "gemini-embedding-001"
    assert bundle.token_contract.embedding_model_name == "gemini-embedding-001"


def test_capability_assembly_rejects_embedding_contract_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _isolated_service(
        monkeypatch,
        compatibility_config=AssemblyConfig(
            tokenizer=TokenizerConfig(
                embedding_model_name="locked-model",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Configured embedding model does not match"):
        service.assemble_request(
            AssemblyRequest(
                overrides=AssemblyOverrides(
                    embedding=ProviderConfig(
                        provider_kind="fake-core",
                        embedding_model="fake-embed-model",
                    )
                )
            )
        )


def test_capability_assembly_builds_env_reranker_outside_business_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAG_RERANK_MODEL_PATH", raising=False)
    monkeypatch.delenv("PKP_LOCAL_BGE__RERANK_MODEL_PATH", raising=False)
    monkeypatch.setenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    service = CapabilityAssemblyService(env_path=".env.test-unused")
    monkeypatch.setattr(service, "_load_env", lambda: None)
    bundle = service.assemble_request(AssemblyRequest(requirements=CapabilityRequirements()))

    assert bundle.rerank_bindings
    assert bundle.rerank_bindings[0].provider_name == "local-bge"
    assert bundle.rerank_bindings[0].model_name == "BAAI/bge-reranker-v2-m3"
