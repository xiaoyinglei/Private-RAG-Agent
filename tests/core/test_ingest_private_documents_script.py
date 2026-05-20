from __future__ import annotations

from types import SimpleNamespace

import scripts.ingest_private_documents as ingest_private_documents


class _FakeSummaryGenerator:
    def generator_info(self) -> dict[str, object]:
        return {
            "provider_name": "test-provider",
            "model_name": "test-model",
        }


class _FakeVectorRepo:
    def count_vectors(self, *, item_kind: str) -> int:
        del item_kind
        return 0


class _FakeRuntime:
    def __init__(self) -> None:
        self.capability_bundle = SimpleNamespace(embedding_bindings=[])
        self.ingest_pipeline = SimpleNamespace(_summarizer=_FakeSummaryGenerator())
        self.stores = SimpleNamespace(vector_repo=_FakeVectorRepo())

    def insert_many(self, requests: list[object], *, continue_on_error: bool = False) -> object:
        del continue_on_error
        return SimpleNamespace(
            success_count=len(requests),
            failure_count=0,
            indexed_object_count=0,
            results=[
                SimpleNamespace(
                    error=None,
                    request=SimpleNamespace(location="ignored"),
                )
                for _ in requests
            ],
        )

    def close(self) -> None:
        return None


def test_private_ingest_requires_summary_chat_by_default(tmp_path, monkeypatch) -> None:
    source = tmp_path / "policy.txt"
    source.write_text("policy text", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_build_runtime_for_benchmark(**kwargs: object) -> _FakeRuntime:
        captured.update(kwargs)
        return _FakeRuntime()

    monkeypatch.setattr(
        ingest_private_documents,
        "build_runtime_for_benchmark",
        fake_build_runtime_for_benchmark,
    )

    assert ingest_private_documents.main(["--input", str(source)]) == 0

    assert captured["require_chat"] is True
    assert captured["strict_summary_generation"] is True


def test_private_ingest_can_explicitly_disable_summary_generation(tmp_path, monkeypatch) -> None:
    source = tmp_path / "policy.txt"
    source.write_text("policy text", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_build_runtime_for_benchmark(**kwargs: object) -> _FakeRuntime:
        captured.update(kwargs)
        return _FakeRuntime()

    monkeypatch.setattr(
        ingest_private_documents,
        "build_runtime_for_benchmark",
        fake_build_runtime_for_benchmark,
    )

    assert ingest_private_documents.main(["--input", str(source), "--summary-provider", "none"]) == 0

    assert captured["require_chat"] is False
    assert captured["strict_summary_generation"] is False

