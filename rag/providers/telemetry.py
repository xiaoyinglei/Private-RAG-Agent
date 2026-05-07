from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from rag.schema.model_protocols import Embedder, Generator, Reranker

T = TypeVar("T", bound=BaseModel)

_LOGGER = logging.getLogger(__name__)


class TelemetryEmbedder(Embedder):
    """Embedding 能力的监控装饰器。"""

    def __init__(self, inner: Embedder, provider_name: str) -> None:
        self._inner = inner
        self._name = provider_name

        self.call_count = 0
        self.total_texts = 0
        self.total_duration_ms = 0.0
        self.error_count = 0

    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        start_time = time.perf_counter()
        batch_size = len(texts)

        try:
            vectors = self._inner.embed(texts, **kwargs)
        except Exception:
            duration = (time.perf_counter() - start_time) * 1000.0
            self.call_count += 1
            self.total_texts += batch_size
            self.total_duration_ms += duration
            self.error_count += 1

            _LOGGER.exception(
                "[%s] Embed failed | Batch: %d | Latency: %.2fms | Calls: %d | Errors: %d",
                self._name,
                batch_size,
                duration,
                self.call_count,
                self.error_count,
            )
            raise

        duration = (time.perf_counter() - start_time) * 1000.0
        self.call_count += 1
        self.total_texts += batch_size
        self.total_duration_ms += duration

        _LOGGER.info(
            "[%s] Embed ok | Batch: %d | Latency: %.2fms | Calls: %d | Total Texts: %d",
            self._name,
            batch_size,
            duration,
            self.call_count,
            self.total_texts,
        )
        return vectors


class TelemetryGenerator(Generator):
    """文本生成能力的监控装饰器。"""

    def __init__(self, inner: Generator, provider_name: str) -> None:
        self._inner = inner
        self._name = provider_name

        self.call_count = 0
        self.text_call_count = 0
        self.structured_call_count = 0
        self.total_duration_ms = 0.0
        self.error_count = 0

    def generate_text(self, *, prompt: str, **kwargs: Any) -> str:
        start_time = time.perf_counter()

        try:
            result = self._inner.generate_text(prompt=prompt, **kwargs)
        except Exception:
            duration = (time.perf_counter() - start_time) * 1000.0
            self.call_count += 1
            self.text_call_count += 1
            self.total_duration_ms += duration
            self.error_count += 1

            _LOGGER.exception(
                "[%s] TextGen failed | Latency: %.2fms | Calls: %d | Errors: %d",
                self._name,
                duration,
                self.call_count,
                self.error_count,
            )
            raise

        duration = (time.perf_counter() - start_time) * 1000.0
        self.call_count += 1
        self.text_call_count += 1
        self.total_duration_ms += duration

        _LOGGER.info(
            "[%s] TextGen ok | Latency: %.2fms | Calls: %d | Text Calls: %d",
            self._name,
            duration,
            self.call_count,
            self.text_call_count,
        )
        return result

    def generate_structured(self, *, prompt: str, schema: type[T], **kwargs: Any) -> T:
        start_time = time.perf_counter()

        try:
            result = self._inner.generate_structured(prompt=prompt, schema=schema, **kwargs)
        except Exception:
            duration = (time.perf_counter() - start_time) * 1000.0
            self.call_count += 1
            self.structured_call_count += 1
            self.total_duration_ms += duration
            self.error_count += 1

            _LOGGER.exception(
                "[%s] StructuredGen failed | Schema: %s | Latency: %.2fms | Calls: %d | Errors: %d",
                self._name,
                getattr(schema, "__name__", str(schema)),
                duration,
                self.call_count,
                self.error_count,
            )
            raise

        duration = (time.perf_counter() - start_time) * 1000.0
        self.call_count += 1
        self.structured_call_count += 1
        self.total_duration_ms += duration

        _LOGGER.info(
            "[%s] StructuredGen ok | Schema: %s | Latency: %.2fms | Calls: %d | Structured Calls: %d",
            self._name,
            getattr(schema, "__name__", str(schema)),
            duration,
            self.call_count,
            self.structured_call_count,
        )
        return result


class TelemetryReranker(Reranker):
    """重排序能力的监控装饰器。"""

    def __init__(self, inner: Reranker, provider_name: str) -> None:
        self._inner = inner
        self._name = provider_name

        self.call_count = 0
        self.total_documents = 0
        self.total_duration_ms = 0.0
        self.error_count = 0

    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]:
        start_time = time.perf_counter()
        doc_count = len(documents)

        try:
            scores = self._inner.rerank(query, documents, **kwargs)
        except Exception:
            duration = (time.perf_counter() - start_time) * 1000.0
            self.call_count += 1
            self.total_documents += doc_count
            self.total_duration_ms += duration
            self.error_count += 1

            _LOGGER.exception(
                "[%s] Rerank failed | Docs: %d | Latency: %.2fms | Calls: %d | Errors: %d",
                self._name,
                doc_count,
                duration,
                self.call_count,
                self.error_count,
            )
            raise

        duration = (time.perf_counter() - start_time) * 1000.0
        self.call_count += 1
        self.total_documents += doc_count
        self.total_duration_ms += duration

        _LOGGER.info(
            "[%s] Rerank ok | Docs: %d | Latency: %.2fms | Calls: %d | Total Docs: %d",
            self._name,
            doc_count,
            duration,
            self.call_count,
            self.total_documents,
        )
        return scores


__all__ = [
    "TelemetryEmbedder",
    "TelemetryGenerator",
    "TelemetryReranker",
]