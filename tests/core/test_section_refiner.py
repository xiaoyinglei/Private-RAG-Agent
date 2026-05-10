from __future__ import annotations

from rag.assembly import TokenAccountingService, TokenizerContract
from rag.ingest.asset_anchors import asset_anchor
from rag.ingest.section_refiner import SectionRefiner
from rag.schema.core import ParsedDocument, ParsedSection, SourceType


def _token_accounting(*, chunk_size: int = 24, overlap: int = 4) -> TokenAccountingService:
    return TokenAccountingService(
        TokenizerContract(
            embedding_model_name="test-embedding",
            tokenizer_model_name="test-tokenizer",
            chunking_tokenizer_model_name="test-tokenizer",
            tokenizer_backend="simple",
            chunk_token_size=chunk_size,
            chunk_overlap_tokens=overlap,
            local_files_only=True,
        )
    )


class _CharOffsetTokenAccounting:
    def __init__(self, *, chunk_size: int = 10, overlap: int = 0) -> None:
        self.contract = TokenizerContract(
            embedding_model_name="test-embedding",
            tokenizer_model_name="test-tokenizer",
            chunking_tokenizer_model_name="test-tokenizer",
            tokenizer_backend="simple",
            chunk_token_size=chunk_size,
            chunk_overlap_tokens=overlap,
            local_files_only=True,
        )

    def _offset_spans(self, text: str) -> list[tuple[int, int]]:
        return [(index, index + 1) for index in range(len(text))]

    def count(self, text: str) -> int:
        return len(text)


def test_section_refiner_splits_oversized_sections_without_losing_raw_spans() -> None:
    text = " ".join(f"alpha{i:03d}" for i in range(70))
    parsed = ParsedDocument(
        title="Generic Long Document",
        source_type=SourceType.PLAIN_TEXT,

        authors=["tester"],
        language="en",
        visible_text=text,
        sections=[
            ParsedSection(
                toc_path=("Generic Long Document",),
                heading_level=1,
                page_range=None,
                order_index=0,
                text=text,
                char_range_start=0,
                char_range_end=len(text),
                anchor_hint="generic-long-document",
            )
        ],
    )

    refined = SectionRefiner(token_accounting=_token_accounting(chunk_size=24, overlap=4)).refine(parsed)

    assert len(refined.sections) > 1
    assert refined.visible_text == parsed.visible_text
    assert [section.order_index for section in refined.sections] == list(range(len(refined.sections)))
    assert all(section.metadata["refined_from_section_order"] == "0" for section in refined.sections)
    assert all(section.metadata["refine_strategy"] == "token_window" for section in refined.sections)
    for section in refined.sections:
        assert section.text == refined.visible_text[section.char_range_start : section.char_range_end]
        assert _token_accounting(chunk_size=24, overlap=4).count(section.text) <= 24


def test_section_refiner_keeps_existing_small_sections_unchanged() -> None:
    first = "Alpha content"
    second = "Beta content"
    visible_text = f"{first}\n\n{second}"
    parsed = ParsedDocument(
        title="Structured",
        source_type=SourceType.MARKDOWN,

        authors=["tester"],
        language="en",
        visible_text=visible_text,
        sections=[
            ParsedSection(
                toc_path=("Structured", "Alpha"),
                heading_level=2,
                page_range=None,
                order_index=0,
                text=first,
                char_range_start=0,
                char_range_end=len(first),
            ),
            ParsedSection(
                toc_path=("Structured", "Beta"),
                heading_level=2,
                page_range=None,
                order_index=1,
                text=second,
                char_range_start=len(first) + 2,
                char_range_end=len(visible_text),
            ),
        ],
    )

    refined = SectionRefiner(token_accounting=_token_accounting(chunk_size=24, overlap=4)).refine(parsed)

    assert refined.sections == parsed.sections


def test_section_refiner_preserves_asset_anchor_when_token_offsets_split_anchor() -> None:
    anchor = asset_anchor("table-1")
    text = f"aaaa{anchor}bbbb"
    parsed = ParsedDocument(
        title="Anchored",
        source_type=SourceType.PLAIN_TEXT,

        authors=["tester"],
        language="en",
        visible_text=text,
        sections=[
            ParsedSection(
                toc_path=("Anchored",),
                heading_level=1,
                page_range=None,
                order_index=0,
                text=text,
                char_range_start=0,
                char_range_end=len(text),
                anchor_hint="anchored",
            )
        ],
    )

    refined = SectionRefiner(
        token_accounting=_CharOffsetTokenAccounting(chunk_size=10, overlap=0)  # type: ignore[arg-type]
    ).refine(parsed)

    sections_with_anchor = [section for section in refined.sections if anchor in section.text]
    assert len(sections_with_anchor) == 1
    assert sections_with_anchor[0].text == refined.visible_text[
        sections_with_anchor[0].char_range_start : sections_with_anchor[0].char_range_end
    ]
