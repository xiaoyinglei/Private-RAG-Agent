from __future__ import annotations

from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageConfig
from rag.assembly import AssemblyConfig, CapabilityAssemblyService
from rag.benchmarks import RetrievalBenchmarkEvaluator, prepared_document_to_ingest_request
from rag.retrieval.models import QueryOptions
from rag.schema.runtime import AccessPolicy


class _CapturingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index + 1), 0.0, 0.0] for index, _text in enumerate(texts)]


def _make_runtime() -> RAGRuntime:
    service = CapabilityAssemblyService(env_path=".env.test-unused")
    service._load_env = lambda: None  # type: ignore[method-assign]
    service._compatibility_config_from_environment = lambda: (AssemblyConfig(), {})  # type: ignore[method-assign]
    request = AssemblyRequest(
        requirements=CapabilityRequirements(
            require_chat=False,
            allow_degraded=True,
            default_context_tokens=QueryOptions().max_context_tokens,
        ),
    )
    return RAGRuntime.from_request(
        storage=StorageConfig.in_memory(),
        request=request,
        assembly_service=service,
    )


def test_prepared_benchmark_document_persists_new_contract_metadata_and_char_range() -> None:
    request = prepared_document_to_ingest_request(
        {
            "doc_id": "medical-doc-1",
            "title": "Medical Benchmark Doc",
            "text": "Alpha treatment reduces symptoms.",
            "source_type": "plain_text",
            "metadata": {"language": "en"},
        },
        dataset="medical_retrieval",
    )

    assert request.metadata["benchmark_doc_id"] == "medical-doc-1"

    runtime = _make_runtime()
    try:
        result = runtime.insert(request)
        source = runtime.stores.metadata_repo.get_source(result.source_id)
        document = runtime.stores.metadata_repo.get_document(result.doc_id)
        sections = runtime.stores.metadata_repo.list_sections(doc_id=result.doc_id)
    finally:
        runtime.close()

    assert source is not None
    assert source.object_key
    assert document is not None
    assert document.metadata_json["benchmark_doc_id"] == "medical-doc-1"
    assert len(sections) == 1
    section = sections[0]
    assert section.metadata_json["benchmark_doc_id"] == "medical-doc-1"
    assert section.char_range_start == 0
    assert section.char_range_end == len("Alpha treatment reduces symptoms.")


def test_runtime_ingest_refines_long_plain_text_sections_on_new_contract() -> None:
    text = " ".join(f"alpha{index:03d}" for index in range(620))
    runtime = _make_runtime()
    try:
        result = runtime.insert(
            prepared_document_to_ingest_request(
                {
                    "doc_id": "medical-doc-long-section",
                    "title": "Medical Long Section Doc",
                    "text": text,
                    "source_type": "plain_text",
                    "metadata": {"language": "en"},
                },
                dataset="medical_retrieval",
            )
        )
        sections = runtime.stores.metadata_repo.list_sections(doc_id=result.doc_id)
        visible_text = runtime.stores.object_store.read_bytes(sections[0].visible_text_key).decode("utf-8")
    finally:
        runtime.close()

    assert len(sections) > 1
    assert result.section_count == len(sections)
    assert [section.order_index for section in sections] == list(range(len(sections)))
    assert all(section.metadata_json["refine_strategy"] == "token_window" for section in sections)
    assert all(section.metadata_json["refined_from_section_order"] == "0" for section in sections)
    for section in sections:
        assert visible_text[section.char_range_start : section.char_range_end]


def test_ingest_pipeline_embeds_doc_and_section_summaries_independently() -> None:
    embedder = _CapturingEmbedder()
    runtime = _make_runtime()
    runtime.ingest_pipeline._embedder = embedder
    try:
        result = runtime.insert(
            prepared_document_to_ingest_request(
                {
                    "doc_id": "medical-doc-batch",
                    "title": "Medical Batch Doc",
                    "text": "Alpha treatment reduces symptoms.",
                    "source_type": "plain_text",
                    "metadata": {"language": "en"},
                },
                dataset="medical_retrieval",
            )
        )
        sections = runtime.stores.metadata_repo.list_sections(doc_id=result.doc_id)
        section_entry = runtime.stores.vector_repo.get_entry(
            str(sections[0].section_id),
            item_kind="section_summary",
        )
        doc_entry = runtime.stores.vector_repo.get_entry(str(result.doc_id), item_kind="doc_summary")
    finally:
        runtime.close()

    assert result.indexed_object_count == 2
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 2
    assert doc_entry is not None
    assert section_entry is not None
    assert doc_entry.vector != section_entry.vector


def test_runtime_batch_ingest_and_benchmark_validation_use_summary_contract() -> None:
    requests = [
        prepared_document_to_ingest_request(
            {
                "doc_id": "medical-doc-1",
                "title": "Medical Benchmark Doc 1",
                "text": "Alpha treatment reduces symptoms.",
                "source_type": "plain_text",
                "metadata": {"language": "en"},
            },
            dataset="medical_retrieval",
        ),
        prepared_document_to_ingest_request(
            {
                "doc_id": "medical-doc-2",
                "title": "Medical Benchmark Doc 2",
                "text": "Beta therapy improves follow-up outcomes.",
                "source_type": "plain_text",
                "metadata": {"language": "en"},
            },
            dataset="medical_retrieval",
        ),
    ]

    runtime = _make_runtime()
    try:
        batch = runtime.insert_many(requests)
        embedding_model_id = runtime.capability_bundle.embedding_bindings[0].model_name
        evaluator = RetrievalBenchmarkEvaluator(
            runtime=runtime,
            dataset="medical_retrieval",
            split="mini",
            retrieval_profile="auto",
            top_k=10,
            evidence_top_k=20,
            rerank_enabled=False,
        )
        evaluator._validate_retrieval_index()
        documents = runtime.stores.metadata_repo.list_documents()
        sections = [
            section
            for document in documents
            for section in runtime.stores.metadata_repo.list_sections(doc_id=document.doc_id)
        ]
        section_entries = [
            runtime.stores.vector_repo.get_entry(str(section.section_id), item_kind="section_summary")
            for section in sections
        ]
        section_vectors = runtime.stores.vector_repo.count_vectors(item_kind="section_summary")
        doc_vectors = runtime.stores.vector_repo.count_vectors(item_kind="doc_summary")
    finally:
        runtime.close()

    assert batch.success_count == 2
    assert batch.failure_count == 0
    assert batch.indexed_object_count >= 4
    assert len(sections) == 2
    assert {document.embedding_model_id for document in documents} == {embedding_model_id}
    assert all(entry is not None for entry in section_entries)
    assert {entry.metadata["embedding_model_id"] for entry in section_entries if entry is not None} == {
        embedding_model_id
    }
    assert section_vectors == 2
    assert doc_vectors == 2


def test_runtime_insert_many_batches_embeddings_across_documents() -> None:
    embedder = _CapturingEmbedder()
    requests = [
        prepared_document_to_ingest_request(
            {
                "doc_id": f"medical-doc-batch-{index}",
                "title": f"Medical Batch Doc {index}",
                "text": f"Alpha treatment {index} reduces symptoms.",
                "source_type": "plain_text",
                "metadata": {"language": "en"},
            },
            dataset="medical_retrieval",
        )
        for index in range(2)
    ]

    runtime = _make_runtime()
    runtime.ingest_pipeline._embedder = embedder
    try:
        batch = runtime.insert_many(requests)
        section_vectors = runtime.stores.vector_repo.count_vectors(item_kind="section_summary")
        doc_vectors = runtime.stores.vector_repo.count_vectors(item_kind="doc_summary")
    finally:
        runtime.close()

    assert batch.success_count == 2
    assert batch.indexed_object_count == 4
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 4
    assert section_vectors == 2
    assert doc_vectors == 2


def test_summary_retrieval_preserves_grounding_target_for_l5() -> None:
    runtime = _make_runtime()
    try:
        runtime.insert(
            prepared_document_to_ingest_request(
                {
                    "doc_id": "medical-doc-grounding",
                    "title": "Grounding Doc",
                    "text": "Alpha treatment reduces symptoms.",
                    "source_type": "plain_text",
                    "metadata": {"language": "en"},
                },
                dataset="medical_retrieval",
            )
        )
        payload = runtime.retrieval_service.retrieve_payload(
            "What reduces symptoms?",
            access_policy=AccessPolicy.default(),
            query_options=QueryOptions(retrieval_profile="auto", enable_rerank=False),
        )
    finally:
        runtime.close()

    evidence = payload.evidence.all
    assert evidence
    assert evidence[0].grounding_target is not None
    assert evidence[0].grounding_target.kind == "section"
    assert evidence[0].grounding_target.section_id is not None
