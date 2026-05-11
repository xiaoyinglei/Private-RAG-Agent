from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pytest

import rag.benchmarks as benchmarks
from rag.benchmarks import (
    MEDICAL_RETRIEVAL_DATASET,
    BenchmarkIngestResult,
    BenchmarkRunSummary,
    RetrievalBenchmarkEvaluator,
    append_baseline_row,
    benchmark_access_policy,
    build_prepared_mini_subset,
    build_runtime_for_benchmark,
    compute_doc_ranking_metrics,
    default_benchmark_paths,
    download_fiqa,
    download_public_benchmark,
    ensure_benchmark_layout,
    ingest_prepared_documents,
    load_baseline_rows,
    prepared_document_to_ingest_request,
    write_dataset_baseline_summary,
)
from rag.retrieval.evidence import EvidenceBundle, SelfCheckResult
from rag.retrieval.runtime_coordinator import RoutingDecision
from rag.retrieval.models import RetrievalResult

from rag.schema.runtime import ExternalRetrievalPolicy, Residency, RuntimeMode
from tests.support import make_runtime


def test_prepared_document_ingest_preserves_benchmark_chunk_metadata() -> None:
    runtime = make_runtime()
    try:
        result = runtime.insert(
            prepared_document_to_ingest_request(
                {
                    "doc_id": "fiqa-doc-1",
                    "title": "FiQA Benchmark Doc",
                    "text": "Alpha engine supports benchmark retrieval evaluation.",
                    "source_type": "plain_text",
                    "metadata": {
                        "dataset": "fiqa",
                        "benchmark": True,
                    },
                },
                dataset="fiqa",
            )
        )
        sections = runtime.stores.metadata_repo.list_sections(doc_id=result.doc_id)
    finally:
        runtime.close()

    assert result.section_count == 1
    assert sections
    for section in sections:
        assert section.metadata_json["benchmark_dataset"] == "fiqa"
        assert section.metadata_json["benchmark_doc_id"] == "fiqa-doc-1"
        assert section.metadata_json["parent_doc_id"] == "fiqa-doc-1"


def test_prepared_document_batch_ingest_uses_formal_insert_many_path() -> None:
    runtime = make_runtime()
    try:
        result = runtime.insert_many(
            [
                prepared_document_to_ingest_request(
                    {
                        "doc_id": "fiqa-doc-1",
                        "title": "FiQA Benchmark Doc 1",
                        "text": "Alpha engine supports benchmark retrieval evaluation.",
                        "source_type": "plain_text",
                        "metadata": {"dataset": "fiqa", "benchmark": True},
                    },
                    dataset="fiqa",
                ),
                prepared_document_to_ingest_request(
                    {
                        "doc_id": "fiqa-doc-2",
                        "title": "FiQA Benchmark Doc 2",
                        "text": "Beta service compares risk and return.",
                        "source_type": "plain_text",
                        "metadata": {"dataset": "fiqa", "benchmark": True},
                    },
                    dataset="fiqa",
                ),
            ]
        )
        all_sections = [
            section
            for item in result.results
            if item.result is not None
            for section in runtime.stores.metadata_repo.list_sections(doc_id=item.result.doc_id)
        ]
    finally:
        runtime.close()

    assert result.success_count == 2
    assert result.failure_count == 0
    assert all_sections
    for section in all_sections:
        assert section.metadata_json["benchmark_dataset"] == "fiqa"
        assert section.metadata_json["benchmark_doc_id"].startswith("fiqa-doc-")


def test_prepared_document_batch_ingest_marks_documents_query_visible() -> None:
    runtime = make_runtime()
    try:
        result = runtime.insert_many(
            [
                prepared_document_to_ingest_request(
                    {
                        "doc_id": "fiqa-doc-ready-1",
                        "title": "FiQA Benchmark Ready Doc 1",
                        "text": "Alpha engine supports benchmark retrieval evaluation.",
                        "source_type": "plain_text",
                        "metadata": {"dataset": "fiqa", "benchmark": True},
                    },
                    dataset="fiqa",
                ),
                prepared_document_to_ingest_request(
                    {
                        "doc_id": "fiqa-doc-ready-2",
                        "title": "FiQA Benchmark Ready Doc 2",
                        "text": "Beta service compares risk and return.",
                        "source_type": "plain_text",
                        "metadata": {"dataset": "fiqa", "benchmark": True},
                    },
                    dataset="fiqa",
                ),
            ]
        )

        persisted_documents = [
            runtime.stores.metadata_repo.get_document(item.result.doc_id)
            for item in result.results
            if item.result is not None
        ]
    finally:
        runtime.close()

    assert result.success_count == 2
    assert all(document is not None for document in persisted_documents)
    assert all(document.is_indexed is True for document in persisted_documents if document is not None)
    assert all(document.index_ready is True for document in persisted_documents if document is not None)


def test_compute_doc_ranking_metrics_uses_document_level_relevance() -> None:
    metrics = compute_doc_ranking_metrics(
        predicted_doc_ids=["doc-x", "doc-b", "doc-a"],
        gold_relevances={"doc-a": 3, "doc-b": 1},
        top_k=10,
    )

    assert metrics["hit_at_10"] == 1
    assert metrics["recall_at_10"] == 1.0
    assert metrics["mrr_at_10"] == 0.5
    assert metrics["ndcg_at_10"] == pytest.approx(0.54134, abs=1e-6)


def test_benchmark_ingest_result_reports_throughput() -> None:
    result = BenchmarkIngestResult(
        dataset="fiqa",
        request_count=100,
        success_count=100,
        duplicate_count=0,
        failure_count=0,
        indexed_object_count=250,
        elapsed_ms=20_000.0,
    )

    assert result.docs_per_second == pytest.approx(5.0)
    assert result.indexed_objects_per_second == pytest.approx(12.5)
    assert result.as_json()["docs_per_second"] == pytest.approx(5.0)
    assert result.as_json()["indexed_objects_per_second"] == pytest.approx(12.5)


def test_benchmark_run_summary_reports_query_throughput() -> None:
    summary = BenchmarkRunSummary(
        run_id="run-1",
        dataset="medical_retrieval",
        split="dev",
        query_count=300,
        top_k=10,
        evidence_top_k=20,
        embedding_model="BAAI/bge-m3",
        retrieval_profile="fast",
        rerank_enabled=True,
        profile_id="local_full",
        recall_at_10=0.7,
        mrr_at_10=0.6,
        ndcg_at_10=0.65,
        avg_latency_ms=2500.0,
        p95_latency_ms=3200.0,
    )

    assert summary.queries_per_second == pytest.approx(0.4)
    assert summary.baseline_row()["queries_per_second"] == pytest.approx(0.4)
    assert summary.as_json()["queries_per_second"] == pytest.approx(0.4)


def test_append_baseline_row_upgrades_existing_csv_header(tmp_path: Path) -> None:
    path = tmp_path / "baseline.csv"
    path.write_text(
        "run_id,dataset,query_count,top_k,embedding_model,retrieval_profile,rerank_enabled,"
        "Recall@10,MRR@10,NDCG@10,avg_latency_ms,p95_latency_ms\n"
        "old-run,medical_retrieval,300,10,BAAI/bge-m3,fast,True,0.7,0.6,0.65,2500.0,3200.0\n",
        encoding="utf-8",
    )

    append_baseline_row(
        path,
        BenchmarkRunSummary(
            run_id="new-run",
            dataset="medical_retrieval",
            split="dev",
            query_count=300,
            top_k=10,
            evidence_top_k=20,
            embedding_model="BAAI/bge-m3",
            retrieval_profile="fast",
            rerank_enabled=True,
            profile_id="local_full",
            recall_at_10=0.71,
            mrr_at_10=0.61,
            ndcg_at_10=0.66,
            avg_latency_ms=2400.0,
            p95_latency_ms=3100.0,
        ),
    )

    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 2
    assert "queries_per_second" in rows[0]
    assert rows[0]["queries_per_second"] == ""
    assert rows[1]["queries_per_second"] == "0.417"


def test_load_baseline_rows_coerces_common_types(tmp_path: Path) -> None:
    path = tmp_path / "baseline.csv"
    path.write_text(
        "run_id,dataset,query_count,top_k,embedding_model,retrieval_profile,"
        "rerank_enabled,Recall@10,MRR@10,NDCG@10,avg_latency_ms,p95_latency_ms,queries_per_second\n"
        "run-1,medical_retrieval,300,10,BAAI/bge-m3,fast,True,0.7,0.6,0.65,2500.0,3200.0,0.4\n",
        encoding="utf-8",
    )

    rows = load_baseline_rows(path)

    assert rows == [
        {
            "run_id": "run-1",
            "dataset": "medical_retrieval",
            "query_count": 300,
            "top_k": 10,
            "embedding_model": "BAAI/bge-m3",
            "retrieval_profile": "fast",
            "rerank_enabled": True,
            "Recall@10": 0.7,
            "MRR@10": 0.6,
            "NDCG@10": 0.65,
            "avg_latency_ms": 2500.0,
            "p95_latency_ms": 3200.0,
            "queries_per_second": 0.4,
        }
    ]


def test_write_dataset_baseline_summary_includes_priority_deltas(tmp_path: Path) -> None:
    path = tmp_path / "medical_retrieval.json"

    write_dataset_baseline_summary(
        path,
        dataset="medical_retrieval",
        reference_run_id="sqlite-run",
        latency_priority_run_id="milvus-bge-run",
        quality_priority_run_id="milvus-qwen-run",
        baselines=[
            {
                "run_id": "sqlite-run",
                "embedding_model": "BAAI/bge-m3",
                "vector_backend": "sqlite",
                "Recall@10": 0.776667,
                "MRR@10": 0.690972,
                "NDCG@10": 0.712199,
                "avg_latency_ms": 2472.225,
            },
            {
                "run_id": "milvus-bge-run",
                "embedding_model": "BAAI/bge-m3",
                "vector_backend": "milvus",
                "Recall@10": 0.67,
                "MRR@10": 0.588259,
                "NDCG@10": 0.608173,
                "avg_latency_ms": 563.793,
                "queries_per_second": 1.774,
            },
            {
                "run_id": "milvus-qwen-run",
                "embedding_model": "qwen3-embedding:8b",
                "vector_backend": "milvus",
                "Recall@10": 0.82,
                "MRR@10": 0.705854,
                "NDCG@10": 0.733644,
                "avg_latency_ms": 695.559,
                "queries_per_second": 1.438,
            },
        ],
    )

    payload = benchmarks.json.loads(path.read_text(encoding="utf-8"))

    assert payload["dataset"] == "medical_retrieval"
    assert payload["reference_run_id"] == "sqlite-run"
    assert payload["latency_priority_run_id"] == "milvus-bge-run"
    assert payload["quality_priority_run_id"] == "milvus-qwen-run"
    assert payload["baseline_count"] == 3
    assert payload["comparison"]["latency_priority_vs_reference"]["avg_latency_ms_delta"] == -1908.432
    assert payload["comparison"]["quality_priority_vs_reference"]["Recall@10_delta"] == 0.043333
    assert payload["comparison"]["latency_priority_vs_reference"]["queries_per_second_delta"] == pytest.approx(
        1.37, abs=1e-3
    )


def test_benchmark_access_policy_forces_local_retrieval_only() -> None:
    policy = benchmark_access_policy()

    assert policy.residency is Residency.LOCAL_REQUIRED
    assert policy.external_retrieval is ExternalRetrievalPolicy.DENY
    assert policy.local_only is True


def test_default_benchmark_paths_use_visible_index_directory() -> None:
    paths = default_benchmark_paths("fiqa")

    assert paths.raw_dir == Path("data/benchmarks/fiqa/raw")
    assert paths.prepared_dir == Path("data/benchmarks/fiqa/prepared/full")
    assert paths.eval_dir == Path("data/benchmarks/fiqa/eval/retrieval/full")
    assert paths.index_dir == Path("data/benchmarks/fiqa/index/full")
    assert paths.prepared_variant_dir("mini") == Path("data/benchmarks/fiqa/prepared/mini")
    assert paths.index_variant_dir("mini") == Path("data/benchmarks/fiqa/index/mini")


def test_ensure_benchmark_layout_creates_full_and_mini_directories(tmp_path) -> None:
    paths = ensure_benchmark_layout(
        benchmarks.BenchmarkPaths(
            dataset_dir=tmp_path / "medical_retrieval",
            raw_dir=tmp_path / "medical_retrieval" / "raw",
            prepared_root=tmp_path / "medical_retrieval" / "prepared",
            index_root=tmp_path / "medical_retrieval" / "index",
            eval_root=tmp_path / "medical_retrieval" / "eval",
            subsets_root=tmp_path / "medical_retrieval" / "subsets",
        )
    )

    assert paths.raw_dir.is_dir()
    assert paths.prepared_variant_dir("full").is_dir()
    assert paths.prepared_variant_dir("mini").is_dir()
    assert paths.index_variant_dir("full").is_dir()
    assert paths.index_variant_dir("mini").is_dir()
    assert paths.eval_variant_dir("retrieval", "full").is_dir()
    assert paths.eval_variant_dir("retrieval", "mini").is_dir()


def test_build_runtime_for_benchmark_creates_storage_root(tmp_path) -> None:
    storage_root = tmp_path / "benchmarks" / "fiqa" / "index"
    assert not storage_root.exists()

    runtime = build_runtime_for_benchmark(
        storage_root=storage_root,
        profile_id="test_minimal",
        require_chat=False,
        require_rerank=False,
        vector_backend="sqlite",
    )
    try:
        assert storage_root.exists()
        assert storage_root.is_dir()
    finally:
        runtime.close()


def test_benchmark_storage_config_defaults_to_milvus_vectors(tmp_path: Path) -> None:
    storage = benchmarks._benchmark_storage_config(root=tmp_path / "index")

    assert storage.vectors is not None
    assert storage.vectors.backend == "milvus"
    assert storage.vectors.dsn == "http://127.0.0.1:19530"


def test_build_runtime_for_benchmark_passes_milvus_vector_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.capability_bundle = type("_Bundle", (), {"embedding_bindings": []})()

        def close(self) -> None:
            return None

    def _fake_from_request(*, storage, request, assembly_service):
        captured["storage"] = storage
        captured["request"] = request
        captured["assembly_service"] = assembly_service
        return _FakeRuntime()

    monkeypatch.setattr(benchmarks.RAGRuntime, "from_request", staticmethod(_fake_from_request))

    runtime = build_runtime_for_benchmark(
        storage_root=tmp_path / "benchmarks" / "medical_retrieval" / "index",
        profile_id="test_minimal",
        require_chat=False,
        require_rerank=False,
        vector_backend="milvus",
        vector_dsn="http://127.0.0.1:19530",
        vector_namespace="rag_benchmarks",
        vector_collection_prefix="medical_retrieval_mini",
    )
    try:
        storage = captured["storage"]
        assert isinstance(storage, benchmarks.StorageConfig)
        assert storage.vectors is not None
        assert storage.vectors.backend == "milvus"
        assert storage.vectors.dsn == "http://127.0.0.1:19530"
        assert storage.vectors.namespace == "rag_benchmarks"
        assert storage.vectors.collection == "medical_retrieval_mini"
    finally:
        runtime.close()


def test_build_runtime_for_benchmark_accepts_ollama_embedding_override(tmp_path) -> None:
    runtime = build_runtime_for_benchmark(
        storage_root=tmp_path / "benchmarks" / "medical_retrieval" / "index",
        profile_id="local_full",
        require_chat=False,
        require_rerank=True,
        embedding_provider_kind="ollama",
        embedding_model="qwen3-embedding:8b",
        vector_backend="sqlite",
    )
    try:
        binding = runtime.capability_bundle.embedding_bindings[0]
        assert binding.provider_name == "ollama"
        assert binding.model_name == "qwen3-embedding:8b"
        assert runtime.capability_bundle.rerank_bindings[0].provider_name == "local-bge"
    finally:
        runtime.close()


def test_build_runtime_for_benchmark_accepts_chat_and_rerank_overrides(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.capability_bundle = type("_Bundle", (), {"embedding_bindings": []})()

        def close(self) -> None:
            return None

    def _fake_from_request(*, storage, request, assembly_service):
        captured["storage"] = storage
        captured["request"] = request
        captured["assembly_service"] = assembly_service
        return _FakeRuntime()

    monkeypatch.setattr(benchmarks.RAGRuntime, "from_request", staticmethod(_fake_from_request))

    runtime = build_runtime_for_benchmark(
        storage_root=tmp_path / "benchmarks" / "medical_retrieval" / "index",
        profile_id="local_full",
        require_chat=True,
        require_rerank=True,
        embedding_provider_kind="ollama",
        embedding_model="qwen3-embedding:8b",
        chat_model="Qwen3-14B-4bit",
        rerank_model="Qwen/Qwen3-Reranker-4B",
    )
    try:
        request = captured["request"]
        overrides = request.overrides
        assert overrides is not None
        assert overrides.embedding is not None
        assert overrides.embedding.provider_kind == "ollama"
        assert overrides.embedding.embedding_model == "qwen3-embedding:8b"
        assert overrides.chat is not None
        assert overrides.chat.provider_kind == "ollama"
        assert overrides.chat.chat_model == "Qwen3-14B-4bit"
        assert overrides.rerank is not None
        assert overrides.rerank.provider_kind == "local-bge"
        assert overrides.rerank.rerank_model == "Qwen/Qwen3-Reranker-4B"
    finally:
        runtime.close()


def test_build_runtime_for_benchmark_accepts_local_hf_chat_overrides(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.capability_bundle = type("_Bundle", (), {"embedding_bindings": []})()

        def close(self) -> None:
            return None

    def _fake_from_request(*, storage, request, assembly_service):
        captured["storage"] = storage
        captured["request"] = request
        captured["assembly_service"] = assembly_service
        return _FakeRuntime()

    monkeypatch.setattr(benchmarks.RAGRuntime, "from_request", staticmethod(_fake_from_request))

    runtime = build_runtime_for_benchmark(
        storage_root=tmp_path / "benchmarks" / "medical_retrieval" / "index",
        profile_id="local_full",
        require_chat=True,
        require_rerank=False,
        chat_provider_kind="local-hf",
        chat_model_path="/models/Qwen3-14B-4bit",
        chat_backend="mlx",
    )
    try:
        request = captured["request"]
        overrides = request.overrides
        assert overrides is not None
        assert overrides.chat is not None
        assert overrides.chat.provider_kind == "local-hf"
        assert overrides.chat.chat_model_path == "/models/Qwen3-14B-4bit"
        assert overrides.chat.chat_backend == "mlx"
    finally:
        runtime.close()


def test_build_runtime_for_benchmark_configures_summary_model_without_requiring_chat(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.capability_bundle = type("_Bundle", (), {"embedding_bindings": []})()

        def configure_summary_generator(self, **kwargs: object) -> None:
            captured["summary_generator"] = kwargs

        def close(self) -> None:
            return None

    def _fake_from_request(*, storage, request, assembly_service):
        captured["storage"] = storage
        captured["request"] = request
        captured["assembly_service"] = assembly_service
        return _FakeRuntime()

    monkeypatch.setattr(benchmarks.RAGRuntime, "from_request", staticmethod(_fake_from_request))

    runtime = build_runtime_for_benchmark(
        storage_root=tmp_path / "benchmarks" / "medical_retrieval" / "index",
        profile_id="local_full",
        require_chat=False,
        require_rerank=False,
        summary_provider_kind="local-hf",
        summary_model_path="/models/Qwen3-8B-MLX-4bit",
        summary_backend="mlx",
    )
    try:
        request = captured["request"]
        assert request.requirements.require_chat is False
        assert request.overrides is None or request.overrides.chat is None
        assert captured["summary_generator"] == {
            "provider_kind": "local-hf",
            "model": None,
            "model_path": "/models/Qwen3-8B-MLX-4bit",
            "backend": "mlx",
        }
    finally:
        runtime.close()


def test_build_runtime_for_benchmark_rerank_override_clears_stale_rerank_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_RERANK_MODEL_PATH", "/tmp/old-bge-reranker")

    runtime = build_runtime_for_benchmark(
        storage_root=tmp_path / "benchmarks" / "medical_retrieval" / "index",
        profile_id="local_full",
        require_chat=False,
        require_rerank=True,
        rerank_model="Qwen/Qwen3-Reranker-4B",
        vector_backend="sqlite",
    )
    try:
        binding = runtime.capability_bundle.rerank_bindings[0]
        assert binding.model_name == "Qwen/Qwen3-Reranker-4B"
        provider = binding.backend
        assert provider.rerank_model_name == "Qwen/Qwen3-Reranker-4B"
    finally:
        runtime.close()


def test_download_fiqa_redownloads_when_cached_zip_is_corrupt(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "fiqa.zip").write_text("not-a-zip", encoding="utf-8")

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("fiqa/corpus.jsonl", '{"_id":"doc-1","title":"Doc 1","text":"hello"}\n')
        zf.writestr("fiqa/queries.jsonl", '{"_id":"q-1","text":"hello"}\n')
        zf.writestr("fiqa/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq-1\tdoc-1\t1\n")
    payload = archive.getvalue()

    class _FakeResponse:
        headers = {"Content-Length": str(len(payload))}

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self):
            yield payload

    class _FakeStream:
        def __enter__(self) -> _FakeResponse:
            return _FakeResponse()

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(benchmarks.httpx, "stream", lambda *args, **kwargs: _FakeStream())

    result = download_fiqa(raw_dir, force=False)

    assert result.archive_path.exists()
    assert result.corpus_path.read_text(encoding="utf-8").strip()
    assert result.queries_path.read_text(encoding="utf-8").strip()
    assert result.qrels_path.read_text(encoding="utf-8").strip()


def test_download_public_benchmark_routes_medical_retrieval(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "raw"

    def _fake_download(target: Path, *, force: bool = False):
        target.mkdir(parents=True, exist_ok=True)
        corpus_path = target / "corpus.jsonl"
        queries_path = target / "queries.jsonl"
        qrels_path = target / "qrels" / "dev.jsonl"
        qrels_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_path.write_text('{"id":"d1","text":"alpha"}\n', encoding="utf-8")
        queries_path.write_text('{"id":"q1","text":"what is alpha"}\n', encoding="utf-8")
        qrels_path.write_text('{"qid":"q1","pid":"d1","score":1}\n', encoding="utf-8")
        return benchmarks.BenchmarkDownloadResult(
            dataset=MEDICAL_RETRIEVAL_DATASET,
            archive_path=target,
            corpus_path=corpus_path,
            queries_path=queries_path,
            qrels_path=qrels_path,
        )

    monkeypatch.setattr(benchmarks, "download_medical_retrieval", _fake_download)
    result = download_public_benchmark(MEDICAL_RETRIEVAL_DATASET, raw_dir)

    assert result.dataset == MEDICAL_RETRIEVAL_DATASET
    assert result.corpus_path.exists()
    assert result.queries_path.exists()
    assert result.qrels_path.exists()


def test_build_prepared_mini_subset_keeps_gold_docs_and_requested_scale(tmp_path) -> None:
    source_dir = tmp_path / "prepared" / "full"
    target_dir = tmp_path / "prepared" / "mini"
    source_dir.mkdir(parents=True)
    _write_jsonl(
        source_dir / "documents.jsonl",
        [
            {"doc_id": f"d{i}", "title": f"Doc {i}", "text": f"text {i}", "source_type": "plain_text", "metadata": {}}
            for i in range(1, 11)
        ],
    )
    _write_jsonl(
        source_dir / "queries.jsonl",
        [
            {"query_id": f"q{i}", "query_text": f"query {i}"}
            for i in range(1, 6)
        ],
    )
    _write_jsonl(
        source_dir / "qrels.jsonl",
        [
            {"query_id": "q1", "doc_id": "d1", "relevance": 1},
            {"query_id": "q2", "doc_id": "d2", "relevance": 1},
            {"query_id": "q3", "doc_id": "d3", "relevance": 1},
            {"query_id": "q4", "doc_id": "d4", "relevance": 1},
            {"query_id": "q5", "doc_id": "d5", "relevance": 1},
        ],
    )

    result = build_prepared_mini_subset(
        dataset=MEDICAL_RETRIEVAL_DATASET,
        source_prepared_dir=source_dir,
        target_prepared_dir=target_dir,
        query_count=3,
        target_doc_count=6,
        seed=7,
    )

    mini_docs = list(benchmarks.iter_jsonl(result.documents_path))
    mini_qrels = list(benchmarks.iter_jsonl(result.qrels_path))
    mini_queries = list(benchmarks.iter_jsonl(result.queries_path))

    assert result.document_count == 6
    assert result.query_count == 3
    assert result.qrel_count == 3
    assert {row["doc_id"] for row in mini_docs}.issuperset({row["doc_id"] for row in mini_qrels})
    assert len(mini_queries) == 3


def test_ingest_prepared_documents_streaming_uses_formal_pipeline(tmp_path) -> None:
    runtime = make_runtime()
    documents_path = tmp_path / "documents.jsonl"
    _write_jsonl(
        documents_path,
        [
            {
                "doc_id": "fiqa-doc-1",
                "title": "Doc 1",
                "text": "alpha benchmark text",
                "source_type": "plain_text",
                "metadata": {"dataset": "fiqa", "benchmark": True},
            },
            {
                "doc_id": "fiqa-doc-2",
                "title": "Doc 2",
                "text": "beta benchmark text",
                "source_type": "plain_text",
                "metadata": {"dataset": "fiqa", "benchmark": True},
            },
        ],
    )
    try:
        result = ingest_prepared_documents(
            runtime,
            dataset="fiqa",
            documents_path=documents_path,
            batch_size=1,
            streaming=True,
        )
    finally:
        runtime.close()

    assert result.success_count == 2
    assert result.failure_count == 0
    assert result.request_count == 2
    assert result.indexed_object_count >= 2


def test_benchmark_evaluator_reads_benchmark_doc_ids_from_retrieval_result(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = make_runtime()
    try:
        runtime.insert(
            prepared_document_to_ingest_request(
                {
                    "doc_id": "fiqa-doc-seed",
                    "title": "seed",
                    "text": "seed benchmark document",
                    "source_type": "plain_text",
                    "metadata": {"dataset": "fiqa", "benchmark": True},
                },
                dataset="fiqa",
            )
        )
        queries_path = tmp_path / "queries.jsonl"
        qrels_path = tmp_path / "qrels.jsonl"
        eval_dir = tmp_path / "eval"
        _write_jsonl(
            queries_path,
            [
                {"query_id": "q-1", "query_text": "alpha benchmark question"},
            ],
        )
        _write_jsonl(
            qrels_path,
            [
                {"query_id": "q-1", "doc_id": "fiqa-doc-1", "relevance": 1},
            ],
        )

        class _StubRetrievalService:
            def retrieve(self, *args, **kwargs) -> RetrievalResult:
                return RetrievalResult(
                    decision=RoutingDecision(
                        runtime_mode=RuntimeMode.FAST,
                    ),
                    evidence=EvidenceBundle(),
                    self_check=SelfCheckResult(
                        retrieve_more=False,
                        evidence_sufficient=True,
                        claim_supported=True,
                    ),
                    reranked_evidence_ids=["evidence-1", "evidence-2"],
                    reranked_benchmark_doc_ids=["fiqa-doc-1", "fiqa-doc-2"],
                )

        runtime.retrieval_service = _StubRetrievalService()  # type: ignore[assignment]

        summary = RetrievalBenchmarkEvaluator(
            runtime=runtime,
            dataset="fiqa",
            split="test",
            retrieval_profile="fast",
            top_k=10,
            evidence_top_k=10,
            rerank_enabled=False,
        ).evaluate(
            queries_path=queries_path,
            qrels_path=qrels_path,
            eval_dir=eval_dir,
        )
    finally:
        runtime.close()

    assert summary.query_count == 1
    per_query = list(benchmarks.iter_jsonl(eval_dir / "per_query.jsonl"))
    assert per_query[0]["predicted_doc_ids"] == ["fiqa-doc-1", "fiqa-doc-2"]


def test_benchmark_evaluator_appends_run_history_instead_of_overwriting(tmp_path) -> None:
    runtime = make_runtime()
    try:
        runtime.insert(
            prepared_document_to_ingest_request(
                {
                    "doc_id": "fiqa-doc-seed",
                    "title": "seed",
                    "text": "seed benchmark document",
                    "source_type": "plain_text",
                    "metadata": {"dataset": "fiqa", "benchmark": True},
                },
                dataset="fiqa",
            )
        )
        queries_path = tmp_path / "queries.jsonl"
        qrels_path = tmp_path / "qrels.jsonl"
        eval_dir = tmp_path / "eval"
        _write_jsonl(queries_path, [{"query_id": "q-1", "query_text": "alpha benchmark question"}])
        _write_jsonl(qrels_path, [{"query_id": "q-1", "doc_id": "fiqa-doc-1", "relevance": 1}])

        class _StubRetrievalService:
            def retrieve(self, *args, **kwargs) -> RetrievalResult:
                return RetrievalResult(
                    decision=RoutingDecision(
                        runtime_mode=RuntimeMode.FAST,
                    ),
                    evidence=EvidenceBundle(),
                    self_check=SelfCheckResult(
                        retrieve_more=False,
                        evidence_sufficient=True,
                        claim_supported=True,
                    ),
                    reranked_evidence_ids=["evidence-1", "evidence-2"],
                    reranked_benchmark_doc_ids=["fiqa-doc-1", "fiqa-doc-2"],
                )

        runtime.retrieval_service = _StubRetrievalService()  # type: ignore[assignment]

        first = RetrievalBenchmarkEvaluator(
            runtime=runtime,
            dataset="fiqa",
            split="test",
            retrieval_profile="fast",
            top_k=10,
            evidence_top_k=10,
            rerank_enabled=False,
        ).evaluate(
            queries_path=queries_path,
            qrels_path=qrels_path,
            eval_dir=eval_dir,
        )
        second = RetrievalBenchmarkEvaluator(
            runtime=runtime,
            dataset="fiqa",
            split="test",
            retrieval_profile="fast",
            top_k=10,
            evidence_top_k=10,
            rerank_enabled=False,
        ).evaluate(
            queries_path=queries_path,
            qrels_path=qrels_path,
            eval_dir=eval_dir,
        )
    finally:
        runtime.close()

    assert first.run_id != second.run_id

    baseline_rows = list(csv.DictReader((eval_dir / "baseline.csv").open("r", encoding="utf-8")))
    assert len(baseline_rows) == 2

    history_rows = list(benchmarks.iter_jsonl(eval_dir / "run_history.jsonl"))
    assert [row["run_id"] for row in history_rows] == [first.run_id, second.run_id]

    per_query_rows = list(benchmarks.iter_jsonl(eval_dir / "per_query.jsonl"))
    assert len(per_query_rows) == 2
    assert [row["run_id"] for row in per_query_rows] == [first.run_id, second.run_id]

    assert (eval_dir / "runs" / first.run_id / "per_query.jsonl").exists()
    assert (eval_dir / "runs" / first.run_id / "run_summary.json").exists()
    assert (eval_dir / "runs" / second.run_id / "per_query.jsonl").exists()
    assert (eval_dir / "runs" / second.run_id / "run_summary.json").exists()


def test_benchmark_evaluator_fails_fast_on_empty_storage(tmp_path) -> None:
    runtime = make_runtime()
    try:
        queries_path = tmp_path / "queries.jsonl"
        qrels_path = tmp_path / "qrels.jsonl"
        eval_dir = tmp_path / "eval"
        _write_jsonl(
            queries_path,
            [
                {"query_id": "q-1", "query_text": "alpha benchmark question"},
            ],
        )
        _write_jsonl(
            qrels_path,
            [
                {"query_id": "q-1", "doc_id": "fiqa-doc-1", "relevance": 1},
            ],
        )

        with pytest.raises(RuntimeError, match="benchmark index is empty"):
            RetrievalBenchmarkEvaluator(
                runtime=runtime,
                dataset="fiqa",
                split="test",
                retrieval_profile="fast",
                top_k=10,
                evidence_top_k=10,
                rerank_enabled=False,
            ).evaluate(
                queries_path=queries_path,
                qrels_path=qrels_path,
                eval_dir=eval_dir,
            )
    finally:
        runtime.close()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{benchmarks.json.dumps(row, ensure_ascii=False)}\n" for row in rows),
        encoding="utf-8",
    )
