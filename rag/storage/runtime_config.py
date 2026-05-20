from __future__ import annotations

from pathlib import Path

from rag.storage import StorageComponentConfig, StorageConfig

DEFAULT_VECTOR_BACKEND = "milvus"
DEFAULT_VECTOR_DSN = "http://127.0.0.1:19530"


def runtime_storage_config(
    storage_root: Path,
    *,
    vector_backend: str = DEFAULT_VECTOR_BACKEND,
    vector_dsn: str | None = None,
    vector_namespace: str | None = None,
    vector_collection_prefix: str | None = None,
) -> StorageConfig:
    backend = (vector_backend or DEFAULT_VECTOR_BACKEND).strip().lower()
    dsn = _normalize_optional_string(vector_dsn)
    if backend == "milvus" and dsn is None:
        dsn = DEFAULT_VECTOR_DSN
    return StorageConfig(
        root=storage_root,
        vectors=StorageComponentConfig(
            backend=backend,
            dsn=dsn,
            namespace=_normalize_optional_string(vector_namespace),
            collection=_normalize_optional_string(vector_collection_prefix),
        ),
    )


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
