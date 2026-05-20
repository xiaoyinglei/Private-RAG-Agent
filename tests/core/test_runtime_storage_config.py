from __future__ import annotations

from pathlib import Path

from rag.storage.runtime_config import runtime_storage_config


def test_runtime_storage_config_defaults_milvus_dsn_when_blank(tmp_path: Path) -> None:
    storage = runtime_storage_config(
        tmp_path / "index",
        vector_backend="milvus",
        vector_dsn=" ",
    )

    assert storage.vectors is not None
    assert storage.vectors.backend == "milvus"
    assert storage.vectors.dsn == "http://127.0.0.1:19530"
