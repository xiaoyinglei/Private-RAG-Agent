from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from rag.schema.model_protocols import OcrVisionRepo
from rag.schema.core import OcrRegion, OcrResult
from rag.ingest.parsers.util import normalize_whitespace

try:
    from ocrmac import ocrmac as ocrmac_module
except Exception:  # pragma: no cover - optional dependency
    ocrmac_module = None

class DeterministicOcrVisionRepo(OcrVisionRepo):
    def __init__(self, mapping: dict[str, OcrResult] | None = None) -> None:
        self._mapping = mapping or {}

    def extract(self, image_path: Path) -> OcrResult:
        if image_path.as_posix() in self._mapping:
            return self._mapping[image_path.as_posix()]

        with Image.open(image_path) as image:
            semantics = f"{image.width}x{image.height} {image.mode} image"
        return OcrResult(visible_text="", visual_semantics=semantics, regions=[])

class OCRMacVisionRepo(OcrVisionRepo):
    def __init__(
        self,
        *,
        language_preferences: tuple[str, ...] = ("zh-Hans", "zh-Hant", "en-US"),
        min_confidence: float | None = None,
        fallback_repo: OcrVisionRepo | None = None,
    ) -> None:
        self._language_preferences = tuple(language_preferences)
        self._min_confidence = min_confidence
        self._fallback_repo = fallback_repo or DeterministicOcrVisionRepo()

    def extract(self, image_path: Path) -> OcrResult:
        with Image.open(image_path) as image:
            width = image.width
            height = image.height

        if sys.platform != "darwin" or ocrmac_module is None:
            return self._fallback_repo.extract(image_path)

        try:
            recognizer = ocrmac_module.OCR(
                str(image_path),
                language_preference=list(self._language_preferences),
            )
            raw_results = recognizer.recognize()
        except Exception:
            return self._fallback_repo.extract(image_path)

        lines: list[str] = []
        regions: list[OcrRegion] = []
        for raw in raw_results:
            text, confidence, bbox = self._parse_record(raw)
            normalized = normalize_whitespace(text)
            if not normalized:
                continue
            if self._min_confidence is not None and confidence < self._min_confidence:
                continue
            if not lines or lines[-1] != normalized:
                lines.append(normalized)
            regions.append(
                OcrRegion(
                    text=normalized,
                    bbox=self._normalize_bbox(bbox, width=width, height=height),
                )
            )

        if not lines:
            return self._fallback_repo.extract(image_path)

        visible_text = "\n".join(lines)
        preview = " | ".join(lines[:6])
        return OcrResult(
            visible_text=visible_text,
            visual_semantics=f"Image containing text: {preview}",
            regions=regions,
        )

    @staticmethod
    def _parse_record(raw: object) -> tuple[str, float, object]:
        if isinstance(raw, tuple) and len(raw) >= 3:
            text = str(raw[0])
            confidence = float(raw[1])
            bbox = raw[2]
            return text, confidence, bbox
        return str(raw), 1.0, None

    @staticmethod
    def _normalize_bbox(
        bbox: object,
        *,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            return None
        x, y, box_width, box_height = (float(value) for value in bbox)
        left = max(0, min(width, round(x * width)))
        top = max(0, min(height, round((1.0 - y - box_height) * height)))
        right = max(left, min(width, round((x + box_width) * width)))
        bottom = max(top, min(height, round((1.0 - y) * height)))
        return left, top, right, bottom


class CallableOcrVisionRepo(OcrVisionRepo):
    def __init__(self, extract_fn: Callable[[Path], OcrResult]) -> None:
        self._extract_fn = extract_fn

    def extract(self, image_path: Path) -> OcrResult:
        return self._extract_fn(image_path)


def create_default_ocr_repo() -> OcrVisionRepo:
    if sys.platform == "darwin" and ocrmac_module is not None:
        return OCRMacVisionRepo()
    return DeterministicOcrVisionRepo()

__all__ = ["OCRMacVisionRepo","create_default_ocr_repo"]
