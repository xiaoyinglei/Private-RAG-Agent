from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from sentence_transformers import SentenceTransformer

from rag.providers.huggingface.hf_utils import (
    _load_flagembedding_module,
    resolve_local_model_reference,
    suppress_backend_fast_tokenizer_padding_warning,
)
from rag.schema.model_protocols import Embedder


class HuggingFaceEmbedder(Embedder):
    """
    本地 Hugging Face 向量化专员。

    适用于：
    - sentence-transformers
    - BGE / GTE / e5 等 embedding 模型
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        *,
        device: str | None = "mps",
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self._model_name_or_path = model_name_or_path
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if device is not None:
            model_kwargs["device"] = device

        self._model = SentenceTransformer(
            model_name_or_path,
            **model_kwargs,
        )

    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            return []

        batch_size = int(kwargs.pop("batch_size", self._batch_size))
        normalize_embeddings = bool(
            kwargs.pop("normalize_embeddings", self._normalize_embeddings)
        )

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        embeddings = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        vectors = embeddings.tolist()

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"HuggingFace embedding count mismatch: expected {len(texts)}, got {len(vectors)}"
            )

        return vectors

    @property
    def model_name_or_path(self) -> str:
        return self._model_name_or_path

    @property
    def provider_name(self) -> str:
        return "huggingface"

    @property
    def embedding_model_name(self) -> str:
        return self._model_name_or_path


class BgeM3Embedder(Embedder):
    """
    BGE-M3 model adapter.

    This is model access only: dense embedding plus BGE-M3 sparse lexical
    weights for the L4 dual-mode sparse path.
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        *,
        model_path: str | None = None,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 1024,
        normalize_embeddings: bool = True,
        use_fp16: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._model_name_or_path = model_name_or_path
        self._model_ref = resolve_local_model_reference(model_name_or_path, model_path)
        self._device = self._resolve_device(device)
        self._batch_size = batch_size
        self._max_length = max_length
        self._normalize_embeddings = normalize_embeddings
        self._use_fp16 = use_fp16
        self._backend: object | None = None

    @property
    def provider_name(self) -> str:
        return "huggingface"

    @property
    def embedding_model_name(self) -> str:
        return self._model_name_or_path

    @property
    def model_name_or_path(self) -> str:
        return self._model_ref

    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        if not texts:
            return []
        batch_size = int(kwargs.get("batch_size", self._batch_size))
        payload = self._encode(texts, batch_size=batch_size, return_dense=True, return_sparse=False)
        vectors = payload.get("dense_vecs") if isinstance(payload, dict) else payload
        if vectors is None:
            raise RuntimeError("BGE-M3 embedding backend returned no dense vectors")
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query_sparse(self, texts: Sequence[str]) -> list[dict[int, float]]:
        if not texts:
            return []
        payload = self._encode(texts, batch_size=self._batch_size, return_dense=False, return_sparse=True)
        sparse_payload = payload.get("lexical_weights") if isinstance(payload, dict) else None
        if sparse_payload is None and isinstance(payload, dict):
            sparse_payload = self._first_non_empty_payload(payload, ("sparse_vecs", "sparse_weights"))
        if sparse_payload is None:
            raise RuntimeError("BGE-M3 embedding backend returned no sparse vectors")
        return [self._normalize_sparse_payload(item) for item in sparse_payload]

    def _encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        return_dense: bool,
        return_sparse: bool,
    ) -> dict[str, Any]:
        backend = self._load_backend()
        encoder = cast(Any, backend)
        dense_payloads: list[Any] = []
        sparse_payloads: list[Any] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            payload = encoder.encode(
                batch,
                batch_size=batch_size,
                max_length=self._max_length,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
            )
            if isinstance(payload, dict):
                if return_dense:
                    dense_payload = self._first_non_empty_payload(payload, ("dense_vecs",))
                    if dense_payload is not None:
                        dense_payloads.extend(dense_payload)
                if return_sparse:
                    sparse_payload = self._first_non_empty_payload(
                        payload,
                        ("lexical_weights", "sparse_vecs", "sparse_weights"),
                    )
                    if sparse_payload is not None:
                        sparse_payloads.extend(sparse_payload)
            elif return_dense:
                dense_payloads.extend(payload)
        return {"dense_vecs": dense_payloads, "lexical_weights": sparse_payloads}

    def _load_backend(self) -> object:
        if self._backend is not None:
            return self._backend
        module = _load_flagembedding_module()
        encoder_cls = getattr(module, "BGEM3FlagModel", None)
        if encoder_cls is None:
            raise RuntimeError("FlagEmbedding BGEM3FlagModel is unavailable")
        backend = encoder_cls(
            self._model_ref,
            normalize_embeddings=self._normalize_embeddings,
            use_fp16=self._use_fp16,
            devices=self._device,
            batch_size=self._batch_size,
            query_max_length=self._max_length,
            passage_max_length=self._max_length,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        self._backend = suppress_backend_fast_tokenizer_padding_warning(backend)
        return self._backend

    @staticmethod
    def _normalize_sparse_payload(payload: object) -> dict[int, float]:
        if isinstance(payload, dict):
            normalized: dict[int, float] = {}
            for key, value in payload.items():
                try:
                    normalized[int(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            if normalized:
                return normalized
        if isinstance(payload, list):
            normalized = {}
            for item in payload:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    continue
                try:
                    normalized[int(item[0])] = float(item[1])
                except (TypeError, ValueError):
                    continue
            if normalized:
                return normalized
        raise RuntimeError(f"Unsupported sparse embedding payload: {type(payload)!r}")

    @staticmethod
    def _first_non_empty_payload(payload: dict[str, Any], keys: Sequence[str]) -> object | None:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                if len(value) == 0:  # type: ignore[arg-type]
                    continue
            except TypeError:
                pass
            return value
        return None

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if isinstance(device, str) and device.strip():
            normalized = device.strip().lower()
            if normalized != "auto":
                return device.strip()
        try:
            import torch
        except Exception:  # pragma: no cover
            return "cpu"
        if getattr(torch.cuda, "is_available", None) and torch.cuda.is_available():
            return "cuda:0"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"


__all__ = ["BgeM3Embedder", "HuggingFaceEmbedder"]
