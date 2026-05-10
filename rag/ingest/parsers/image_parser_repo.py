from __future__ import annotations

from pathlib import Path
from typing import cast

from PIL import Image

from rag.ingest.parsers.util import default_title_from_location, normalize_whitespace, slugify
from rag.schema.core import ParsedDocument, ParsedElement, ParsedSection, SourceType
from rag.schema.model_protocols import OcrVisionRepo


class ImageParserRepo:
    def __init__(self, ocr_repo: OcrVisionRepo) -> None:
        self._ocr_repo = ocr_repo

    def parse(
        self,
        image_path: Path,
        *,
        location: str,
        source_type: SourceType,
        title: str | None = None,
        owner: str = "user",
    ) -> ParsedDocument:
        document_title = title or default_title_from_location(location)
        ocr_result = self._ocr_repo.extract(image_path)
        
        # 提取底层图像物理元数据
        with Image.open(image_path) as image:
            image_metadata = {
                "image_width": str(image.width),
                "image_height": str(image.height),
                "image_mode": image.mode,
                "source_type": SourceType.IMAGE.value,
                "location": location,
            }
            
        normalized_visible_text = normalize_whitespace(ocr_result.visible_text)
        visible_text = normalized_visible_text or document_title
        
        elements = [
            ParsedElement(
                element_id=f"{slugify(document_title)}-ocr-{index}",
                kind="ocr_region",
                text=normalize_whitespace(region.text),
                toc_path=(document_title,),
                page_no=1,
                bbox=(
                    None
                    if region.bbox is None
                    else cast(
                        tuple[float, float, float, float],
                        tuple(float(value) for value in region.bbox),
                    )
                ),
                metadata={"source_type": SourceType.IMAGE.value, "region_index": str(index)},
            )
            for index, region in enumerate(ocr_result.regions)
            if normalize_whitespace(region.text)
        ]

        section = ParsedSection(
            toc_path=(document_title,),
            heading_level=1,
            page_range=(1, 1),
            order_index=0,
            text=visible_text,
            char_range_start=0,
            char_range_end=len(visible_text),
            anchor_hint=slugify(document_title),
            metadata=image_metadata,
        )
        start = section.char_range_start
        end = section.char_range_end
        if start != 0 or end != len(visible_text):
            raise ValueError(
                f"image section span mismatch: start={start}, end={end}, visible_len={len(visible_text)}"
            )
        if visible_text[start:end] != section.text:
            raise ValueError(
                "image section text/span mismatch"
            )
        return ParsedDocument(
            title=document_title,
            source_type=SourceType.IMAGE,
            authors=[owner],
            language=None,
            sections=[section],
            visible_text=visible_text,
            visual_semantics=ocr_result.visual_semantics,
            elements=elements,
            page_count=1,
            metadata=image_metadata,
        )
