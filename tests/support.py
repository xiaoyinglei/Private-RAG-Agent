from __future__ import annotations

from pathlib import Path

from rag import CapabilityRequirements, RAGRuntime, StorageConfig
from rag.assembly import AssemblyConfig, CapabilityAssemblyService, CapabilityBundle
from rag.retrieval.models import QueryOptions
from rag.schema.runtime import OcrVisionRepo, VisualDescriptionRepo, WebFetchRepo


def _isolated_assembly_service() -> CapabilityAssemblyService:
    service = CapabilityAssemblyService(env_path=".env.test-unused")
    service._load_env = lambda: None  # type: ignore[method-assign]
    service._compatibility_config_from_environment = lambda: (AssemblyConfig(), {})  # type: ignore[method-assign]
    return service


def make_capability_bundle(
    *,
    require_chat: bool = False,
    assembly_service: CapabilityAssemblyService | None = None,
) -> CapabilityBundle:
    service = assembly_service or _isolated_assembly_service()
    request = service.request_for_profile(
        "test_minimal",
        requirements=CapabilityRequirements(
            require_chat=require_chat,
            allow_degraded=True,
            default_context_tokens=QueryOptions().max_context_tokens,
        ),
    )
    return service.assemble_request(request)


def make_runtime(
    *,
    storage: StorageConfig | None = None,
    require_chat: bool = False,
    assembly_service: CapabilityAssemblyService | None = None,
) -> RAGRuntime:
    service = assembly_service or _isolated_assembly_service()
    request = service.request_for_profile(
        "test_minimal",
        requirements=CapabilityRequirements(
            require_chat=require_chat,
            allow_degraded=True,
            default_context_tokens=QueryOptions().max_context_tokens,
        ),
    )
    return RAGRuntime.from_request(
        storage=storage or StorageConfig.in_memory(),
        request=request,
        assembly_service=service,
    )


def make_ingest_service(
    root: Path,
    *,
    capability_bundle: CapabilityBundle | None = None,
    ocr_repo: OcrVisionRepo | None = None,
    vlm_repo: VisualDescriptionRepo | None = None,
    web_fetch_repo: WebFetchRepo | None = None,
) -> object:
    from rag.ingest.pipeline import IngestService

    return IngestService.create_in_memory(
        root,
        capability_bundle=capability_bundle or make_capability_bundle(),
        ocr_repo=ocr_repo,
        vlm_repo=vlm_repo,
        web_fetch_repo=web_fetch_repo,
    )
