from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    MarkdownFormatOption,
    PdfFormatOption,
    WordFormatOption,
)
from docling_core.types.doc.document import PictureItem, TableItem
from docling_core.types.doc.labels import DocItemLabel

from rag.ingest.asset_anchors import asset_anchor
from rag.ingest.parsers.util import default_title_from_location, normalize_whitespace, slugify
from rag.schema.core import DocumentType, ParsedDocument, ParsedElement, ParsedSection, SourceType
from rag.schema.model_protocols import VisualDescriptionRepo

_DOCLING_TABLE_IMAGE_DEPRECATION = (
    r"This field is deprecated\. Use `generate_page_images=True` and call `TableItem\.get_image\(\)` "
    r"to extract table images from page images\."
)


class DoclingParserRepo:
    _MARKDOWN_SUFFIXES = {".md", ".markdown"}

    def __init__(
        self,
        vlm_repo: VisualDescriptionRepo | None = None,
    ) -> None:
        self._vlm_repo = vlm_repo
        self._converter = self._build_converter()

    def _build_converter(self) -> DocumentConverter:
        enable_visual = self._vlm_repo is not None

        pdf_options = PdfPipelineOptions(
            do_ocr=True,  
            do_table_structure=True,
            generate_picture_images=enable_visual, 
            generate_page_images=False, 
            do_picture_classification=False,
            do_picture_description=False,
        )

        return DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.MD,
                InputFormat.DOCX,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                InputFormat.MD: MarkdownFormatOption(),
                InputFormat.DOCX: WordFormatOption(),
            },
        )



    def parse(
        self,
        file_path: Path,
        *,
        location: str,
        source_type: SourceType,
        title: str | None = None,
        owner: str = "user",
    ) -> ParsedDocument:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=_DOCLING_TABLE_IMAGE_DEPRECATION,
                    category=DeprecationWarning,
                    module=r"docling\.pipeline\.standard_pdf_pipeline",
                )
                result = self._converter.convert(file_path)
        except Exception as exc: 
            raise ValueError(f"Docling failed to parse {location}: {exc}") from exc

        doc = result.document
        item_records = list(doc.iterate_items(traverse_pictures=True))
        lookup: dict[str, Any] = {str(getattr(item, "self_ref", "")): item for item, _ in item_records}
        resolved_title = title or self._resolve_title(
            item_records,
            fallback=doc.name or default_title_from_location(location),
        )
        sections, elements, visible_text = self._extract_sections_and_elements(
            doc=doc,
            item_records=item_records,
            lookup=lookup,
            document_title=resolved_title,
            source_type=source_type,
        )

        page_numbers = [element.page_no for element in elements if element.page_no is not None]
        page_count = max(page_numbers) if page_numbers else None
        
        return ParsedDocument(
            title=resolved_title,
            source_type=source_type,
            doc_type=DocumentType.UNKNOWN,
            authors=[owner],
            language=None, 
            sections=sections,
            visible_text=visible_text,
            visual_semantics=None,
            elements=elements,
            page_count=page_count,
            doc_model=doc,
            metadata={"location": location, "source_type": source_type.value},
        )

    def _extract_sections_and_elements(
        self,
        *,
        doc: Any,
        item_records: Sequence[tuple[Any, int]],
        lookup: Mapping[str, Any],
        document_title: str,
        source_type: SourceType,
    ) -> tuple[list[ParsedSection], list[ParsedElement], str]:
        sections: list[ParsedSection] = []
        elements: list[ParsedElement] = []
        visible_parts: list[str] = []
        visible_cursor = 0
        visible_section_separator = "\n\n"
        heading_stack: list[str] = []
        current_section_path: tuple[str, ...] = (document_title,)
        current_heading_level = 1
        current_text_parts: list[str] = []
        current_page_numbers: list[int] = []
        current_anchor_hint = slugify(document_title)
        section_order = 0
        element_counts: defaultdict[str, int] = defaultdict(int)

        def flush_section() -> None:
            nonlocal current_text_parts, current_page_numbers, section_order, visible_cursor

            text = self._section_text_from_parts(current_text_parts)
            if not text:
                return

            page_range = None
            if current_page_numbers:
                page_range = (min(current_page_numbers), max(current_page_numbers))

            # 关键：在写入 visible_parts 的同时记录 span
            if visible_parts:
                visible_parts.append(visible_section_separator)
                visible_cursor += len(visible_section_separator)

            char_start = visible_cursor
            visible_parts.append(text)
            visible_cursor += len(text)
            char_end = visible_cursor

            sections.append(
                ParsedSection(
                    toc_path=current_section_path,
                    heading_level=current_heading_level,
                    page_range=page_range,
                    order_index=section_order,
                    text=text,
                    char_range_start=char_start,
                    char_range_end=char_end,
                    anchor_hint=(
                        f"page-{page_range[0]}"
                        if source_type is SourceType.PDF and page_range is not None
                        else current_anchor_hint
                    ),
                    metadata={},
                )
            )

            current_text_parts = []
            current_page_numbers = []
            section_order += 1

        for item, _ in item_records:
            label = self._label_value(item)
            if label == DocItemLabel.TITLE.value:
                title_text = normalize_whitespace(getattr(item, "text", ""))
                if title_text:
                    current_section_path = (title_text,)
                    current_anchor_hint = slugify(title_text)
                continue
            if label == DocItemLabel.SECTION_HEADER.value:
                flush_section()
                heading_text = normalize_whitespace(getattr(item, "text", ""))
                level = max(1, int(getattr(item, "level", 1) or 1))
                heading_stack[:] = heading_stack[: max(level - 1, 0)] + [heading_text]
                current_section_path = (document_title, *heading_stack)
                current_heading_level = level
                current_anchor_hint = slugify("-".join(current_section_path))
                elements.append(
                    ParsedElement(
                        element_id=self._next_element_id(element_counts, "section_header", current_anchor_hint),
                        kind="section_header",
                        text=heading_text,
                        toc_path=current_section_path,
                        heading_level=level,
                        page_no=self._page_no(item),
                        bbox=self._bbox(item),
                    )
                )
                continue

            if isinstance(item, TableItem):
                try:
                    table_text = self._normalize_asset_text(item.export_to_markdown(doc))
                except TypeError:
                    table_text = self._normalize_asset_text(item.export_to_markdown())
                
                table_description = self._visual_description(
                    doc=doc,
                    item=item,
                    prompt="Describe the table layout and any visually encoded patterns useful for retrieval.",
                )
                element_id = self._next_element_id(element_counts, "table", current_anchor_hint)
                
                if table_text:
                    current_text_parts.append(asset_anchor(element_id))
                
                elements.append(
                    ParsedElement(
                        element_id=element_id,
                        kind="table",
                        text=table_text,
                        toc_path=current_section_path,
                        page_no=self._page_no(item),
                        bbox=self._bbox(item),
                        metadata={} if table_description is None else {"visual_description": table_description},
                    )
                )
                continue

            if isinstance(item, PictureItem):
                caption_text = self._caption_text(item=item, lookup=lookup)
                
                figure_description = self._visual_description(
                    doc=doc,
                    item=item,
                    prompt=(
                        "Describe the picture content, chart structure, and notable visual signals "
                        "useful for retrieval."
                    ),
                )
                
                figure_text = normalize_whitespace(caption_text or "figure")
                if figure_text and figure_text != "figure":
                    current_text_parts.append(figure_text)
                
                elements.append(
                    ParsedElement(
                        element_id=self._next_element_id(element_counts, "figure", current_anchor_hint),
                        kind="figure",
                        text=figure_text,
                        toc_path=current_section_path,
                        page_no=self._page_no(item),
                        bbox=self._bbox(item),
                        metadata={
                            "caption": caption_text,
                            **({} if figure_description is None else {"visual_description": figure_description}),
                        },
                    )
                )
                
                if figure_description:
                    elements.append(
                        ParsedElement(
                            element_id=self._next_element_id(element_counts, "image_summary", current_anchor_hint),
                            kind="image_summary",
                            text=figure_description,
                            toc_path=current_section_path,
                            page_no=self._page_no(item),
                            bbox=self._bbox(item),
                            metadata={"derived_from": "figure"},
                        )
                    )
                continue

            text = self._normalize_text_block(getattr(item, "text", ""))
            if not text:
                continue
            if label == DocItemLabel.CAPTION.value:
                elements.append(
                    ParsedElement(
                        element_id=self._next_element_id(element_counts, "caption", current_anchor_hint),
                        kind="caption",
                        text=text,
                        toc_path=current_section_path,
                        page_no=self._page_no(item),
                        bbox=self._bbox(item),
                    )
                )
                continue
            if label in {
                DocItemLabel.PAGE_HEADER.value,
                DocItemLabel.PAGE_FOOTER.value,
                DocItemLabel.FOOTNOTE.value,
            }:
                continue

            page_no = self._page_no(item)
            
            current_text_parts.append(text)
            if page_no is not None:
                current_page_numbers.append(page_no)
            elements.append(
                ParsedElement(
                    element_id=self._next_element_id(element_counts, "text", current_anchor_hint),
                    kind="text",
                    text=text,
                    toc_path=current_section_path,
                    heading_level=current_heading_level,
                    page_no=page_no,
                    bbox=self._bbox(item),
                    metadata={"label": label},
                )
            )

        flush_section()
        if not sections:
            root_text = self._section_text_from_parts(
                element.text for element in elements if element.kind == "text"
            ) or document_title
            sections = [
                ParsedSection(
                    toc_path=(document_title,),
                    heading_level=1,
                    page_range=None,
                    order_index=0,
                    text=root_text,
                    char_range_start=0,
                    char_range_end=len(root_text),
                    anchor_hint=slugify(document_title),
                    metadata={},
                )
            ]
            visible_parts = [root_text]
            visible_cursor = len(root_text)
                
        visible_text = "".join(visible_parts)

        for idx, section in enumerate(sections):
            start = section.char_range_start
            end = section.char_range_end
            if start is None or end is None:
                raise ValueError(f"section[{idx}] missing char range")
            if start < 0 or end <= start or end > len(visible_text):
                raise ValueError(
                    f"section[{idx}] invalid char range: start={start}, end={end}, visible_len={len(visible_text)}"
                )
            if visible_text[start:end] != section.text:
                raise ValueError(
                    f"section[{idx}] text/span mismatch: "
                    f"expected={section.text!r}, actual={visible_text[start:end]!r}"
                )

        return sections, elements, visible_text

    @staticmethod
    def _resolve_title(item_records: Sequence[tuple[Any, int]], *, fallback: str) -> str:
        for item, _ in item_records:
            if DoclingParserRepo._label_value(item) == DocItemLabel.TITLE.value:
                text = normalize_whitespace(getattr(item, "text", ""))
                if text:
                    return text
        return normalize_whitespace(fallback) or "document"

    @staticmethod
    def _normalize_text_block(text: object) -> str:
        lines = [
            normalize_whitespace(line)
            for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        ]
        return "\n".join(line for line in lines if line).strip()

    @staticmethod
    def _section_text_from_parts(parts: Iterable[object]) -> str:
        normalized_parts = [str(part).strip() for part in parts if str(part or "").strip()]
        return "\n\n".join(normalized_parts).strip()

    @staticmethod
    def _label_value(item: object) -> str:
        label = getattr(item, "label", "")
        return label.value if hasattr(label, "value") else str(label)

    @staticmethod
    def _page_no(item: object) -> int | None:
        provenance = getattr(item, "prov", None) or []
        if not provenance:
            return None
        return int(getattr(provenance[0], "page_no", 0) or 0) or None

    @staticmethod
    def _bbox(item: object) -> tuple[float, float, float, float] | None:
        provenance = getattr(item, "prov", None) or []
        if not provenance:
            return None
        bbox = getattr(provenance[0], "bbox", None)
        if bbox is None:
            return None
        return (
            float(getattr(bbox, "l", 0.0)),
            float(getattr(bbox, "t", 0.0)),
            float(getattr(bbox, "r", 0.0)),
            float(getattr(bbox, "b", 0.0)),
        )

    @staticmethod
    def _caption_text(*, item: PictureItem, lookup: Mapping[str, Any]) -> str:
        caption_texts: list[str] = []
        for reference in getattr(item, "captions", None) or []:
            ref = getattr(reference, "cref", None) or str(reference)
            caption_item = lookup.get(ref)
            if caption_item is None:
                continue
            text = normalize_whitespace(getattr(caption_item, "text", ""))
            if text:
                caption_texts.append(text)
        return normalize_whitespace(" ".join(caption_texts))

    @staticmethod
    def _normalize_asset_text(text: object) -> str:
        return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    @staticmethod
    def _next_element_id(counts: defaultdict[str, int], kind: str, prefix: str) -> str:
        counts[kind] += 1
        return f"{prefix}-{kind}-{counts[kind]}"

    def _visual_description(self, *, doc: Any, item: object, prompt: str) -> str | None:
        if self._vlm_repo is None:
            return None
        get_image = getattr(item, "get_image", None)
        if not callable(get_image):
            return None
        try:
            image = get_image(doc)
        except Exception:
            return None
        if image is None:
            return None
        buffer = BytesIO()
        try:
            image.save(buffer, format="PNG")
        except Exception:
            return None
        try:
            description = normalize_whitespace(
                self._vlm_repo.describe_visual(
                    buffer.getvalue(),
                    mime_type="image/png",
                    prompt=prompt,
                )
            )
        except Exception:
            return None
        return description or None
