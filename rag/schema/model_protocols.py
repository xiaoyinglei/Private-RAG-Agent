from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from rag.schema.core import OcrResult


T = TypeVar("T", bound=BaseModel)


class Generator(Protocol):
    def generate_text(self, *, prompt: str, **kwargs: Any) -> str: ...
    def generate_structured(self, *, prompt: str, schema: type[T], **kwargs: Any) -> T: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]: ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]: ...


class OcrVisionRepo(Protocol):
    def extract(self, image_path: Path, **kwargs: Any) -> OcrResult: ...


class VisualDescriptionRepo(Protocol):
    def describe_visual(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/png",
        prompt: str | None = None,
        **kwargs: Any,
    ) -> str: ...
