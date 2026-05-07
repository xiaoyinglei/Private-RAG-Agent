from __future__ import annotations

import os
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn, cast

from rag.schema.runtime import CacheRepo, GraphRepo, MetadataRepo, ObjectStore, VectorRepo
from rag.storage.data_contract_service import DataContractService


@dataclass(frozen=True, slots=True)
class StorageComponentConfig:
    backend: str
    dsn: str | None = None
    root: str | PathLike[str] | Path | None = None
    namespace: str | None = None
    collection: str | None = None
    bucket: str | None = None
    options: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class StorageBundle:
    root: Path
    metadata_repo: MetadataRepo
    vector_repo: VectorRepo
    graph_repo: GraphRepo
    cache_repo: CacheRepo
    object_store: ObjectStore
    _closeables: tuple[object, ...] = field(default_factory=tuple, repr=False)
    _ephemeral_root: TemporaryDirectory[str] | None = field(default=None, repr=False)

    def close(self) -> None:
        closed_ids: set[int] = set()
        for closeable in self._closeables:
            if id(closeable) in closed_ids:
                continue
            closed_ids.add(id(closeable))
            close = getattr(closeable, "close", None)
            if callable(close):
                close()
        if self._ephemeral_root is not None:
            self._ephemeral_root.cleanup()


@dataclass(frozen=True, slots=True)
class StorageConfig:
    backend: str = "sqlite"
    root: str | PathLike[str] | Path | None = None
    metadata: StorageComponentConfig | None = None
    vectors: StorageComponentConfig | None = None
    graph: StorageComponentConfig | None = None
    cache: StorageComponentConfig | None = None
    object_store: StorageComponentConfig | None = None

    @classmethod
    def in_memory(cls, root: str | PathLike[str] | Path | None = None) -> StorageConfig:
        return cls(backend="in_memory", root=root)

    def build(self) -> StorageBundle:
        ephemeral_root: TemporaryDirectory[str] | None = None
        if self.backend == "in_memory" and self.root is None:
            ephemeral_root = TemporaryDirectory(prefix="rag-runtime-")
            root = Path(ephemeral_root.name)
        else:
            root = Path(self.root) if self.root is not None else Path(".rag")
        root.mkdir(parents=True, exist_ok=True)

        metadata_config = self._component_config(
            self.metadata,
            default_backend="sqlite" if self.backend in {"sqlite", "in_memory"} else "postgres",
        )
        vectors_config = self._component_config(
            self.vectors,
            default_backend="sqlite" if self.backend in {"sqlite", "in_memory"} else "milvus",
        )
        graph_config = self._component_config(
            self.graph,
            default_backend="null",
        )
        cache_config = self._component_config(
            self.cache,
            default_backend="metadata" if self.backend in {"sqlite", "in_memory"} else "redis",
        )
        object_config = self._component_config(
            self.object_store,
            default_backend="local" if self.backend in {"sqlite", "in_memory"} else "s3",
        )
        metadata_config = self._inherit_component_dsn(metadata_config, fallback=None)
        vectors_config = self._inherit_component_dsn(vectors_config, fallback=metadata_config.dsn)

        metadata_repo = self._build_metadata_repo(metadata_config, root)
        vector_repo = self._build_vector_repo(vectors_config, root)
        graph_repo = self._build_graph_repo(graph_config, root)
        cache_repo = self._build_cache_repo(cache_config, root, metadata_repo=metadata_repo)
        object_store = self._build_object_store(object_config, root)

        closeables = (
            metadata_repo,
            vector_repo,
            graph_repo,
            cache_repo,
            object_store,
        )

        return StorageBundle(
            root=root,
            metadata_repo=metadata_repo,
            vector_repo=vector_repo,
            graph_repo=graph_repo,
            cache_repo=cache_repo,
            object_store=object_store,
            _closeables=closeables,
            _ephemeral_root=ephemeral_root,
        )

    @staticmethod
    def _component_config(
        component: StorageComponentConfig | None,
        *,
        default_backend: str,
    ) -> StorageComponentConfig:
        if component is not None:
            return component
        return StorageComponentConfig(backend=default_backend)

    @staticmethod
    def _component_path(root: Path, component: StorageComponentConfig, filename: str) -> Path:
        if component.root is None:
            return root / filename
        candidate = Path(component.root)
        if candidate.suffix:
            return candidate
        return candidate / filename

    @staticmethod
    def _inherit_component_dsn(component: StorageComponentConfig, *, fallback: str | None) -> StorageComponentConfig:
        if component.dsn or fallback is None:
            return component
        return StorageComponentConfig(
            backend=component.backend,
            dsn=fallback,
            root=component.root,
            namespace=component.namespace,
            collection=component.collection,
            bucket=component.bucket,
            options=dict(component.options),
        )

    def _build_metadata_repo(self, component: StorageComponentConfig, root: Path) -> MetadataRepo:
        backend = component.backend.lower()
        if backend in {"sqlite", "in_memory"}:
            from rag.storage.repositories.sqlite_metadata_repo import SQLiteMetadataRepo

            return cast(MetadataRepo, SQLiteMetadataRepo(self._component_path(root, component, "metadata.sqlite3")))
        if backend in {"postgres", "postgresql"}:
            from rag.storage.repositories.postgres_metadata_repo import PostgresMetadataRepo

            return cast(
                MetadataRepo,
                PostgresMetadataRepo(
                    self._require_dsn(component, env_names=("RAG_METADATA_DSN", "RAG_POSTGRES_DSN")),
                    schema=component.namespace or "public",
                ),
            )
        self._unsupported_component(component, "Unsupported metadata backend.")

    def _build_vector_repo(self, component: StorageComponentConfig, root: Path) -> VectorRepo:
        backend = component.backend.lower()
        if backend in {"sqlite", "in_memory"}:
            from rag.storage.search_backends.sqlite_vector_repo import SQLiteVectorRepo

            return cast(VectorRepo, SQLiteVectorRepo(self._component_path(root, component, "vectors.sqlite3")))
        if backend == "milvus":
            from rag.storage.search_backends.milvus_vector_repo import MilvusVectorRepo

            return cast(
                VectorRepo,
                MilvusVectorRepo(
                    self._require_dsn(component, env_names=("RAG_MILVUS_URI", "RAG_VECTOR_DSN")),
                    token=self._env_or_option(component, "token", env_names=("RAG_MILVUS_TOKEN",)),
                    db_name=component.namespace
                    or self._env_or_option(component, "database", env_names=("RAG_MILVUS_DB",)),
                    collection_prefix=component.collection or "rag_vectors",
                ),
            )
        self._unsupported_component(component, "Unsupported vector backend.")

    def _build_graph_repo(self, component: StorageComponentConfig, root: Path) -> GraphRepo:
        del root
        backend = component.backend.lower()
        if backend == "null":
            from rag.storage.graph_backends.null_graph_repo import NullGraphRepo

            return cast(GraphRepo, NullGraphRepo())
        self._unsupported_component(component, "Graph backends must be rebuilt on the EvidenceItem contract.")

    def _build_cache_repo(
        self,
        component: StorageComponentConfig,
        root: Path,
        *,
        metadata_repo: MetadataRepo,
    ) -> CacheRepo:
        backend = component.backend.lower()
        if backend in {"metadata", "sqlite", "in_memory"}:
            return self._require_cache_repo(metadata_repo)
        if backend == "redis":
            from rag.storage.repositories.redis_cache_repo import RedisCacheRepo

            return cast(
                CacheRepo,
                RedisCacheRepo(
                    self._require_dsn(component, env_names=("RAG_CACHE_DSN", "RAG_REDIS_URL")),
                    key_prefix=component.namespace or "rag-cache",
                ),
            )
        self._unsupported_component(component, "Unsupported cache backend.")

    @staticmethod
    def _require_cache_repo(repo: object) -> CacheRepo:
        required = (
            "save_cache_entry",
            "get_cache_entry",
            "list_cache_entries",
            "delete_cache_entry",
            "purge_expired_cache_entries",
        )
        missing = [name for name in required if not callable(getattr(repo, name, None))]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"selected metadata-backed cache requires cache capability: {joined}")
        return cast(CacheRepo, repo)

    def _build_object_store(self, component: StorageComponentConfig, root: Path) -> ObjectStore:
        backend = component.backend.lower()
        if backend in {"local", "sqlite", "in_memory"}:
            from rag.storage.repositories.file_object_store import FileObjectStore

            return cast(ObjectStore, FileObjectStore(self._component_path(root, component, "objects")))
        if backend in {"s3", "minio"}:
            from rag.storage.repositories.s3_object_store import S3ObjectStore

            bucket = component.bucket or self._env_or_option(component, "bucket", env_names=("RAG_OBJECT_BUCKET",))
            if not bucket:
                self._unsupported_component(component, "S3/MinIO object store requires bucket configuration.")
            endpoint_url = None
            if backend == "minio":
                endpoint_url = self._require_dsn(component, env_names=("RAG_OBJECT_ENDPOINT", "RAG_MINIO_ENDPOINT"))
            elif component.dsn:
                endpoint_url = component.dsn
            return cast(
                ObjectStore,
                S3ObjectStore(
                    bucket=bucket,
                    endpoint_url=endpoint_url,
                    prefix=component.collection or component.namespace or "",
                    region_name=self._env_or_option(component, "region", env_names=("AWS_REGION", "RAG_OBJECT_REGION")),
                    access_key_id=self._env_or_option(
                        component,
                        "access_key_id",
                        env_names=("AWS_ACCESS_KEY_ID", "MINIO_ROOT_USER", "RAG_OBJECT_ACCESS_KEY_ID"),
                    ),
                    secret_access_key=self._env_or_option(
                        component,
                        "secret_access_key",
                        env_names=("AWS_SECRET_ACCESS_KEY", "MINIO_ROOT_PASSWORD", "RAG_OBJECT_SECRET_ACCESS_KEY"),
                    ),
                    session_token=self._env_or_option(component, "session_token", env_names=("AWS_SESSION_TOKEN",)),
                ),
            )
        self._unsupported_component(component, "Unsupported object backend.")

    @staticmethod
    def _env_or_option(
        component: StorageComponentConfig,
        option_key: str,
        *,
        env_names: tuple[str, ...],
    ) -> str | None:
        option_value = component.options.get(option_key)
        if option_value:
            return option_value
        for env_name in env_names:
            env_value = os.environ.get(env_name)
            if env_value:
                return env_value
        return None

    def _require_dsn(self, component: StorageComponentConfig, *, env_names: tuple[str, ...]) -> str:
        if component.dsn:
            return component.dsn
        for env_name in env_names:
            env_value = os.environ.get(env_name)
            if env_value:
                return env_value
        self._unsupported_component(
            component,
            "Component backend requires a DSN/URI. "
            f"Set one explicitly or via one of: {', '.join(env_names)}.",
        )

    @staticmethod
    def _unsupported_component(component: StorageComponentConfig, message: str) -> NoReturn:
        backend = component.backend
        dsn_hint = f" dsn={component.dsn!r}" if component.dsn else ""
        raise RuntimeError(f"{message} backend={backend!r}{dsn_hint}")


__all__ = [
    "StorageBundle",
    "StorageComponentConfig",
    "StorageConfig",
    "DataContractService",
]
