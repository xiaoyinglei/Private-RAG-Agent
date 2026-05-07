from __future__ import annotations

from pathlib import Path

from scripts import ingest_public_benchmark


class _FakePaths:
    def __init__(self, root: Path) -> None:
        self._root = root

    def prepared_variant_dir(self, variant: str) -> Path:
        path = self._root / "prepared" / variant
        path.mkdir(parents=True, exist_ok=True)
        (path / "documents.jsonl").write_text("", encoding="utf-8")
        return path

    def index_variant_dir(self, variant: str) -> Path:
        path = self._root / "index" / variant
        path.mkdir(parents=True, exist_ok=True)
        return path


class _FakeRuntime:
    selected_profile_id = "local_full"

    def close(self) -> None:
        return None


class _FakeIngestResult:
    def as_json(self) -> dict[str, object]:
        return {
            "dataset": "medical_retrieval",
            "request_count": 1,
            "success_count": 1,
            "duplicate_count": 0,
            "failure_count": 0,
            "indexed_object_count": 1,
            "elapsed_ms": 10.0,
            "docs_per_second": 100.0,
            "indexed_objects_per_second": 100.0,
        }


def test_ingest_script_requires_chat_when_graph_extraction_enabled(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured: dict[str, object] = {}

    def _fake_build_runtime_for_benchmark(**kwargs):
        captured.update(kwargs)
        return _FakeRuntime()

    monkeypatch.setattr(
        ingest_public_benchmark,
        "default_benchmark_paths",
        lambda dataset: _FakePaths(tmp_path),
    )
    monkeypatch.setattr(ingest_public_benchmark, "ensure_benchmark_layout", lambda paths: paths)
    monkeypatch.setattr(ingest_public_benchmark, "build_runtime_for_benchmark", _fake_build_runtime_for_benchmark)
    monkeypatch.setattr(
        ingest_public_benchmark,
        "configure_runtime_embedding",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        ingest_public_benchmark,
        "ingest_prepared_documents",
        lambda *args, **kwargs: _FakeIngestResult(),
    )
    monkeypatch.setattr(ingest_public_benchmark, "runtime_embedding_stats", lambda runtime: None)

    exit_code = ingest_public_benchmark.main(
        [
            "--dataset",
            "medical_retrieval",
            "--variant",
            "mini",
            "--profile",
            "local_full",
        ]
    )

    assert exit_code == 0
    assert captured["require_chat"] is True
    assert captured["vector_backend"] == "milvus"
    assert captured["vector_dsn"] == "http://127.0.0.1:19530"
    payload = capsys.readouterr().out
    assert '"skip_graph_extraction": false' in payload


def test_ingest_script_passes_chat_model_override_for_graph_extraction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_build_runtime_for_benchmark(**kwargs):
        captured.update(kwargs)
        return _FakeRuntime()

    monkeypatch.setattr(
        ingest_public_benchmark,
        "default_benchmark_paths",
        lambda dataset: _FakePaths(tmp_path),
    )
    monkeypatch.setattr(ingest_public_benchmark, "ensure_benchmark_layout", lambda paths: paths)
    monkeypatch.setattr(ingest_public_benchmark, "build_runtime_for_benchmark", _fake_build_runtime_for_benchmark)
    monkeypatch.setattr(
        ingest_public_benchmark,
        "configure_runtime_embedding",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        ingest_public_benchmark,
        "ingest_prepared_documents",
        lambda *args, **kwargs: _FakeIngestResult(),
    )
    monkeypatch.setattr(ingest_public_benchmark, "runtime_embedding_stats", lambda runtime: None)

    exit_code = ingest_public_benchmark.main(
        [
            "--dataset",
            "medical_retrieval",
            "--variant",
            "mini",
            "--profile",
            "local_full",
            "--chat-provider",
            "local-hf",
            "--chat-model",
            "Qwen3-14B-4bit",
            "--chat-model-path",
            "/models/Qwen3-14B-4bit",
            "--chat-backend",
            "mlx",
        ]
    )

    assert exit_code == 0
    assert captured["require_chat"] is True
    assert captured["chat_provider_kind"] == "local-hf"
    assert captured["chat_model"] == "Qwen3-14B-4bit"
    assert captured["chat_model_path"] == "/models/Qwen3-14B-4bit"
    assert captured["chat_backend"] == "mlx"
