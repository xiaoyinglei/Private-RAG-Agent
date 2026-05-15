from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag.providers.huggingface.hf_utils import (
    _infer_flagembedding_reranker_model_class,
    _load_flagembedding_module,
    resolve_local_model_reference,
    suppress_backend_fast_tokenizer_padding_warning,
)
from rag.schema.model_protocols import Reranker


class FlagEmbeddingReranker(Reranker):
    """
    Hugging Face / FlagEmbedding rerank model adapter.

    This layer only loads a local rerank model and exposes
    rerank(query, documents) -> scores. Candidate cleanup, fusion,
    diagnostics, and exit decisions live in retrieval.rerank_service.
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-v2-m3",
        *,
        model_path: str | None = None,
        batch_size: int = 8,
        max_length: int = 1024,
        use_fp16: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
        devices: str | None = None,
        normalize: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._model_name_or_path = model_name_or_path
        self._model_path = model_path
        self._batch_size = batch_size
        self._max_length = max_length
        self._use_fp16 = use_fp16
        self._local_files_only = local_files_only
        self._trust_remote_code = trust_remote_code
        self._devices = devices
        self._normalize = normalize
        self._backend: object | None = None

    @property
    def provider_name(self) -> str:
        return "flagembedding"

    @property
    def rerank_model_name(self) -> str:
        return resolve_local_model_reference(self._model_name_or_path, self._model_path)

    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]:
        if not documents:
            return []
        batch_size = int(kwargs.get("batch_size", self._batch_size))
        max_length = int(kwargs.get("max_length", self._max_length))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        compute_score = getattr(self._load_backend(), "compute_score", None)
        if not callable(compute_score):
            raise RuntimeError("FlagEmbedding reranker backend does not expose compute_score")

        pairs = [[query, document] for document in documents]
        raw_scores = compute_score(pairs, batch_size=batch_size, max_length=max_length)
        return self._coerce_scores(raw_scores, expected=len(documents))

    def _load_backend(self) -> object:
        if self._backend is not None:
            return self._backend

        module = _load_flagembedding_module()
        model_ref = self.rerank_model_name
        model_class = _infer_flagembedding_reranker_model_class(model_ref)
        class_names = (
            ("FlagAutoReranker", "FlagLLMReranker", "LayerWiseFlagLLMReranker")
            if model_class == "decoder-only-base"
            else ("FlagReranker", "FlagAutoReranker")
        )

        for class_name in class_names:
            backend_cls = getattr(module, class_name, None)
            if backend_cls is None:
                continue
            self._backend = suppress_backend_fast_tokenizer_padding_warning(
                self._instantiate_backend(backend_cls, model_ref, model_class=model_class)
            )
            return self._backend

        raise RuntimeError(f"FlagEmbedding reranker class unavailable for {model_ref}")

    def _instantiate_backend(self, backend_cls: object, model_ref: str, *, model_class: str) -> object:
        from_finetuned = getattr(backend_cls, "from_finetuned", None)
        kwargs: dict[str, Any] = {
            "use_fp16": self._use_fp16,
            "local_files_only": self._local_files_only,
            "trust_remote_code": self._trust_remote_code,
            "model_class": model_class,
            "batch_size": self._batch_size,
            "max_length": self._max_length,
            "normalize": self._normalize,
        }
        if self._devices:
            kwargs["devices"] = self._devices
        if callable(from_finetuned):
            return self._call_factory(from_finetuned, model_ref, kwargs)
        return self._call_factory(backend_cls, model_ref, kwargs)

    @staticmethod
    def _call_factory(factory: object, model_ref: str, kwargs: dict[str, Any]) -> object:
        call = factory
        unsupported_order = (
            "local_files_only",
            "normalize",
            "max_length",
            "batch_size",
            "devices",
            "model_class",
            "trust_remote_code",
        )
        current = dict(kwargs)
        while True:
            try:
                return call(model_ref, **current)  # type: ignore[operator]
            except TypeError:
                removed = False
                for key in unsupported_order:
                    if key in current:
                        current.pop(key, None)
                        removed = True
                        break
                if not removed:
                    raise

    @staticmethod
    def _coerce_scores(raw_scores: object, *, expected: int) -> list[float]:
        tolist = getattr(raw_scores, "tolist", None)
        if callable(tolist):
            raw_scores = tolist()
        if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, str | bytes):
            raise RuntimeError(f"Unsupported rerank score payload: {type(raw_scores)!r}")
        scores = [float(score) for score in raw_scores]
        if len(scores) != expected:
            raise RuntimeError(f"Rerank score count mismatch: expected {expected}, got {len(scores)}")
        return scores


__all__ = ["FlagEmbeddingReranker"]
