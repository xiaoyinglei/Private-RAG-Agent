from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from rag.schema.model_protocols import Embedder


class MLXEmbeddingModelError(RuntimeError):
    """模型不暴露 hidden states 或输出疑似 logits。"""


class MLXEmbedder(Embedder):
    """Apple Silicon / MLX 本地 dense embedding provider —— 实验性。

    使用 mlx_lm 加载模型，通过 model.model（transformer trunk）提取 hidden states。
    默认 last_token + L2 normalize。第一版只做 dense，不涉及 sparse。

    注意：切换 embedding 模型后必须重建 Milvus collection（dense vector 维度可能变化）。
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        batch_size: int = 8,
        pooling: Literal["last_token", "mean"] = "last_token",
        normalize: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
        tokenizer_config: dict[str, Any] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if pooling not in ("last_token", "mean"):
            raise ValueError(f"pooling must be 'last_token' or 'mean', got {pooling!r}")

        self._model_name_or_path = model_name_or_path
        self._batch_size = batch_size
        self._pooling: Literal["last_token", "mean"] = pooling
        self._normalize_enabled = normalize
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._dim: int | None = None

        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                "mlx_lm is not installed. Install mlx-lm before using MLXEmbedder."
            ) from exc

        try:
            loaded = cast(
                Any,
                load(
                model_name_or_path,
                tokenizer_config=tokenizer_config or {},
                ),
            )
            self._model = loaded[0]
            self._tokenizer = loaded[1]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MLX embedding model '{model_name_or_path}': {exc}"
            ) from exc

        transformer = getattr(self._model, "model", None)
        if transformer is None:
            raise MLXEmbeddingModelError(
                "This MLX model does not expose hidden states required for embedding. "
                "The model must have a .model attribute (transformer trunk)."
            )
        self._transformer = cast(Any, transformer)

    # ── Public API ──

    @property
    def embedding_model_name(self) -> str:
        return self._model_name_or_path

    @property
    def dimension(self) -> int | None:
        return self._dim

    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        mode: Literal["query", "document"] = kwargs.pop("mode", "document")
        return self._embed(texts, mode=mode, **kwargs)

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, mode="query")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, mode="document")

    def close(self) -> None:
        pass

    # ── Internal ──

    def _embed(
        self,
        texts: Sequence[str],
        *,
        mode: Literal["query", "document"],
        **kwargs: Any,
    ) -> list[list[float]]:
        if not texts:
            return []

        batch_size = int(kwargs.pop("batch_size", self._batch_size))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        prefix = self._query_prefix if mode == "query" else self._document_prefix
        prefixed = [f"{prefix}{text}" for text in texts]

        all_vectors: list[list[float]] = []
        for start in range(0, len(prefixed), batch_size):
            batch = prefixed[start : start + batch_size]
            all_vectors.extend(self._embed_batch(batch))

        if len(all_vectors) != len(texts):
            raise RuntimeError(
                f"MLX embedding count mismatch: expected {len(texts)}, got {len(all_vectors)}"
            )
        return all_vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import mlx.core as mx

        encoded = self._encode(texts)
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        hidden = self._extract_hidden_states(input_ids, attention_mask)
        self._validate_dimension(hidden)

        if self._pooling == "last_token":
            pooled = self._last_token_pool(hidden, attention_mask)
        else:
            pooled = self._mean_pool(hidden, attention_mask)

        if self._normalize_enabled:
            pooled = self._l2_normalize(pooled)

        vectors = pooled.tolist()
        return [[float(v) for v in vec] for vec in vectors]

    def _encode(self, texts: list[str]) -> Any:
        """编码文本。优先 call tokenizer，不支持时 fallback 到内部 _tokenizer。"""
        try:
            tokenizer = cast(Any, self._tokenizer)
            return tokenizer(
                texts, padding=True, truncation=True, return_tensors="np",
            )
        except TypeError:
            inner = getattr(self._tokenizer, "_tokenizer", None)
            if inner is None:
                raise
            inner_tokenizer = cast(Any, inner)
            return inner_tokenizer(texts, padding=True, truncation=True, return_tensors="np")

    def _extract_hidden_states(self, input_ids: Any, attention_mask: Any) -> Any:
        """提取 hidden states。优先传 attention_mask，不支持时 fallback。"""
        output = self._try_transformer(input_ids, attention_mask)

        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, dict) and "last_hidden_state" in output:
            hidden = output["last_hidden_state"]
        elif isinstance(output, (tuple, list)):
            hidden = output[0]
        else:
            hidden = output

        self._reject_logits(hidden)
        return hidden

    def _try_transformer(self, input_ids: Any, attention_mask: Any) -> Any:
        try:
            return self._transformer(input_ids, attention_mask=attention_mask)
        except TypeError:
            return self._transformer(input_ids)

    def _reject_logits(self, hidden: Any) -> None:
        import mlx.core as mx

        if not isinstance(hidden, mx.array) or hidden.ndim != 3:
            return

        last_dim = int(hidden.shape[-1])
        if last_dim > 20000:
            raise MLXEmbeddingModelError(
                f"Output looks like logits (last_dim={last_dim}, likely vocab size). "
                "This MLX model does not expose hidden states required for embedding. "
                "Expected hidden_size (768-8192), got logits dimensions."
            )

    def _validate_dimension(self, hidden: Any) -> None:
        import mlx.core as mx

        if not isinstance(hidden, mx.array):
            raise MLXEmbeddingModelError(
                f"Expected mx.array from hidden states, got {type(hidden).__name__}. "
                "This MLX model does not expose hidden states required for embedding."
            )
        if hidden.ndim != 3:
            raise MLXEmbeddingModelError(
                f"Expected 3D hidden states (batch, seq_len, dim), got shape {hidden.shape}"
            )

        dim = int(hidden.shape[-1])
        if self._dim is None:
            self._dim = dim
        elif self._dim != dim:
            raise RuntimeError(
                f"MLX embedding dimension changed: expected {self._dim}, got {dim}"
            )

    @staticmethod
    def _last_token_pool(hidden: Any, attention_mask: Any) -> Any:
        import mlx.core as mx

        seq_lens = mx.maximum(
            mx.sum(attention_mask, axis=1).astype(mx.int32) - 1,
            mx.array(0, dtype=mx.int32),
        )
        batch_indices = mx.arange(hidden.shape[0])
        return hidden[batch_indices, seq_lens]

    @staticmethod
    def _mean_pool(hidden: Any, attention_mask: Any) -> Any:
        import mlx.core as mx

        mask = mx.expand_dims(attention_mask.astype(mx.float32), axis=-1)
        masked = hidden * mask
        summed = mx.sum(masked, axis=1)
        counts = mx.sum(mask, axis=1)
        return summed / mx.maximum(counts, mx.array(1.0, dtype=mx.float32))

    @staticmethod
    def _l2_normalize(vectors: Any) -> Any:
        import mlx.core as mx

        norms = mx.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / mx.maximum(norms, mx.array(1e-12, dtype=mx.float32))
