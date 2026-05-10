from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.assembly import TokenAccountingService, TokenizerContract
from rag.ingest.asset_anchors import asset_anchor
from rag.ingest.parsers.docling_parser_repo import DoclingParserRepo
from rag.ingest.pipeline import IngestPipeline, IngestRequest
from rag.ingest.retrievalsummarizer import RetrievalSummaryResult
from rag.ingest.section_refiner import SectionRefiner
from rag.schema.core import (
    AssetRecord,
    Document,
    LayoutMetaCacheRecord,
    ParsedDocument,
    ParsedElement,
    ParsedSection,
    ProcessingStateRecord,
    SectionRecord,
    Source,
    SourceType,
)
from rag.storage.repositories.file_object_store import FileObjectStore


class _Dispatcher:
    def __init__(self, parsed_doc: ParsedDocument) -> None:
        self.parsed_doc = parsed_doc

    def route_and_parse(self, **_: Any) -> ParsedDocument:
        return self.parsed_doc


class _Summarizer:
    def __init__(self) -> None:
        self.asset_inputs: list[str] = []
        self.doc_inputs: list[tuple[list[str], list[str]]] = []

    def summarize_section_with_metadata(self, section: ParsedSection, document_title: str) -> RetrievalSummaryResult:
        return RetrievalSummaryResult(
            text=f"{document_title}: {section.text}",
            method="test",
            provider_name=None,
            model_name=None,
        )

    def summarize_asset_with_metadata(
        self,
        *,
        asset_type: str,
        asset_text: str,
        document_title: str,
        toc_path: tuple[str, ...] | list[str],
        caption: str | None = None,
    ) -> RetrievalSummaryResult:
        del toc_path, caption
        self.asset_inputs.append(asset_text)
        return RetrievalSummaryResult(
            text=f"{document_title} {asset_type} asset summary: {asset_text}",
            method="test",
            provider_name=None,
            model_name=None,
        )

    def summarize_doc_with_metadata(
        self,
        *,
        document_title: str,
        section_summaries: list[str] | tuple[str, ...],
        asset_summaries: list[str] | tuple[str, ...] = (),
    ) -> RetrievalSummaryResult:
        self.doc_inputs.append((list(section_summaries), list(asset_summaries)))
        return RetrievalSummaryResult(
            text=f"{document_title} doc summary from {' / '.join(section_summaries)}",
            method="test_doc_reduce",
            provider_name=None,
            model_name=None,
        )

    def generator_info(self) -> dict[str, str | int | None]:
        return {}


class _Embedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


class _SummaryRepo:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def upsert_records(self, items: list[tuple[Any, list[float]]]) -> None:
        self.records.extend(record for record, _ in items)


class _MetadataRepo:
    def __init__(self) -> None:
        self.sources: list[Source] = []
        self.documents: list[Document] = []
        self.sections: list[SectionRecord] = []
        self.assets: list[AssetRecord] = []
        self.processing_states: list[ProcessingStateRecord] = []
        self.layout_records: list[LayoutMetaCacheRecord] = []
        self._source_id = 0
        self._doc_id = 0
        self._section_id = 0
        self._asset_id = 0

    def transaction(self):
        return nullcontext()

    def save_source(self, source: Source) -> Source:
        if source.source_id:
            saved = source
        else:
            self._source_id += 1
            saved = source.model_copy(update={"source_id": self._source_id})
        self.sources.append(saved)
        return saved

    def save_document(self, document: Document) -> Document:
        self._doc_id += 1
        saved = document.model_copy(update={"doc_id": self._doc_id, "version_group_id": self._doc_id})
        self.documents.append(saved)
        return saved

    def save_section(self, section: SectionRecord) -> SectionRecord:
        if section.section_id > 0:
            saved = section
            self.sections = [
                saved if existing.section_id == section.section_id else existing
                for existing in self.sections
            ]
            if all(existing.section_id != section.section_id for existing in self.sections):
                self.sections.append(saved)
            return saved
        self._section_id += 1
        saved = section.model_copy(update={"section_id": self._section_id})
        self.sections.append(saved)
        return saved

    def save_asset(self, asset: AssetRecord) -> AssetRecord:
        self._asset_id += 1
        saved = asset.model_copy(update={"asset_id": self._asset_id})
        self.assets.append(saved)
        return saved

    def save_layout_meta_cache(self, record: LayoutMetaCacheRecord) -> LayoutMetaCacheRecord:
        self.layout_records.append(record)
        return record

    def save_processing_state(self, record: ProcessingStateRecord) -> ProcessingStateRecord:
        self.processing_states.append(record)
        return record

    def set_document_index_state(
        self,
        doc_id: int,
        *,
        is_indexed: bool | None = None,
        index_ready: bool | None = None,
        indexed_at: datetime | None = None,
        last_index_error: str | None = None,
    ) -> Document:
        del is_indexed, index_ready, indexed_at, last_index_error
        return self.documents[-1].model_copy(update={"doc_id": doc_id})


def _token_accounting(*, chunk_size: int = 8) -> TokenAccountingService:
    return TokenAccountingService(
        TokenizerContract(
            embedding_model_name="test-embedding",
            tokenizer_model_name="test-tokenizer",
            chunking_tokenizer_model_name="test-tokenizer",
            tokenizer_backend="simple",
            chunk_token_size=chunk_size,
            chunk_overlap_tokens=0,
            local_files_only=True,
        )
    )


def _section_text(section: SectionRecord, object_store: FileObjectStore) -> str:
    visible_text = object_store.read_bytes(section.visible_text_key).decode("utf-8")
    return visible_text[section.char_range_start : section.char_range_end]


def test_docling_section_text_preserves_block_boundaries_and_asset_anchor() -> None:
    block = DoclingParserRepo._normalize_text_block("1. 适用范围  \n  2. 审批要求")
    text = DoclingParserRepo._section_text_from_parts(
        [block, asset_anchor("table-1"), "后续说明"]
    )

    assert text == "1. 适用范围\n2. 审批要求\n\n[ASSET_ANCHOR:table-1]\n\n后续说明"


def test_asset_anchor_binds_table_to_refined_section_and_indexes_asset_text(tmp_path: Path) -> None:
    table_anchor = "[ASSET_ANCHOR:table-1]"
    preface = " ".join(f"preface{i}" for i in range(18))
    tail = " ".join(f"tail{i}" for i in range(8))
    section_text = f"{preface} {table_anchor} {tail}"
    table_markdown = "| Column A | Column B |\n|---|---|\n| travel limit | 500 |"
    parsed_doc = ParsedDocument(
        title="Policy",
        source_type=SourceType.DOCX,

        authors=["tester"],
        language="zh",
        visible_text=section_text,
        sections=[
            ParsedSection(
                toc_path=("Policy", "Travel"),
                heading_level=2,
                page_range=(1, 1),
                order_index=0,
                text=section_text,
                char_range_start=0,
                char_range_end=len(section_text),
                anchor_hint="policy-travel",
            )
        ],
        elements=[
            ParsedElement(
                element_id="table-1",
                kind="table",
                text=table_markdown,
                toc_path=("Policy", "Travel"),
                page_no=1,
            )
        ],
        page_count=1,
    )

    source_path = tmp_path / "policy.docx"
    source_path.write_bytes(b"fake docx bytes")
    object_store = FileObjectStore(tmp_path / "objects")
    metadata_repo = _MetadataRepo()
    summary_repo = _SummaryRepo()

    pipeline = IngestPipeline(
        dispatcher=_Dispatcher(parsed_doc),  # type: ignore[arg-type]
        summarizer=_Summarizer(),  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
        metadata_repo=metadata_repo,  # type: ignore[arg-type]
        summary_repo=summary_repo,  # type: ignore[arg-type]
        object_store=object_store,
        section_refiner=SectionRefiner(token_accounting=_token_accounting(chunk_size=8)),
    )

    pipeline.run(
        IngestRequest(
            location=str(source_path),
            file_path=source_path,
            source_type=SourceType.DOCX,
            owner="tester",
            title="Policy",
        )
    )

    assert len(metadata_repo.assets) == 1
    saved_asset = metadata_repo.assets[0]
    anchor_sections = [
        section
        for section in metadata_repo.sections
        if table_anchor in _section_text(section, object_store)
    ]
    assert len(anchor_sections) == 1
    anchor_section = anchor_sections[0]
    assert saved_asset.section_id == anchor_section.section_id
    assert saved_asset.neighbor_section_id == anchor_section.section_id
    assert anchor_section.has_table is True
    assert anchor_section.neighbor_asset_count == 1
    refined_sections = [
        section
        for section in metadata_repo.sections
        if section.metadata_json.get("refine_strategy") == "token_window"
    ]
    assert refined_sections
    assert {section.parent_section_id for section in refined_sections} == {refined_sections[0].section_id}
    assert [section.metadata_json["window_index"] for section in refined_sections] == list(range(len(refined_sections)))
    assert all(section.metadata_json["document_title"] == "Policy" for section in refined_sections)
    stored = object_store.read_bytes(saved_asset.storage_key)
    assert len(stored) > 0
    assert stored[:4] == b"PAR1"

    asset_summaries = [record for record in summary_repo.records if record.__class__.__name__ == "AssetSummaryRecord"]
    assert len(asset_summaries) == 1
    assert "Column A" in asset_summaries[0].summary_text
    assert "travel limit" in asset_summaries[0].summary_text


def test_large_table_assets_store_table_policy_and_sample_summary_input(tmp_path: Path) -> None:
    table_anchor = "[ASSET_ANCHOR:large-table]"
    rows = "\n".join(f"| row{index} | dept{index % 3} | {index * 100} |" for index in range(800))
    table_markdown = f"| Name | Department | Amount |\n|---|---|---|\n{rows}"
    parsed_doc = ParsedDocument(
        title="Policy",
        source_type=SourceType.DOCX,

        authors=["tester"],
        language="zh",
        visible_text=f"Large table follows {table_anchor}",
        sections=[
            ParsedSection(
                toc_path=("Policy", "Large Table"),
                heading_level=2,
                page_range=(1, 1),
                order_index=0,
                text=f"Large table follows {table_anchor}",
                char_range_start=0,
                char_range_end=len(f"Large table follows {table_anchor}"),
                anchor_hint="policy-large-table",
            )
        ],
        elements=[
            ParsedElement(
                element_id="large-table",
                kind="table",
                text=table_markdown,
                toc_path=("Policy", "Large Table"),
                page_no=1,
            )
        ],
        page_count=1,
    )

    source_path = tmp_path / "policy.docx"
    source_path.write_bytes(b"fake docx bytes")
    object_store = FileObjectStore(tmp_path / "objects")
    metadata_repo = _MetadataRepo()
    summary_repo = _SummaryRepo()
    summarizer = _Summarizer()

    pipeline = IngestPipeline(
        dispatcher=_Dispatcher(parsed_doc),  # type: ignore[arg-type]
        summarizer=summarizer,  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
        metadata_repo=metadata_repo,  # type: ignore[arg-type]
        summary_repo=summary_repo,  # type: ignore[arg-type]
        object_store=object_store,
        section_refiner=SectionRefiner(token_accounting=_token_accounting(chunk_size=64)),
    )

    pipeline.run(
        IngestRequest(
            location=str(source_path),
            file_path=source_path,
            source_type=SourceType.DOCX,
            owner="tester",
            title="Policy",
        )
    )

    saved_asset = metadata_repo.assets[0]
    assert saved_asset.row_count == 800
    assert saved_asset.column_count == 3
    assert saved_asset.metadata_json["row_count"] == 800
    assert saved_asset.metadata_json["column_count"] == 3
    assert saved_asset.metadata_json["table_policy"] == "compute_only"
    assert int(saved_asset.metadata_json["estimated_tokens"]) > 800
    stored = object_store.read_bytes(saved_asset.storage_key)
    assert len(stored) > 0
    assert stored[:4] == b"PAR1"

    assert len(summarizer.asset_inputs) == 1
    sampled_input = summarizer.asset_inputs[0]
    assert "Table columns: Name | Department | Amount" in sampled_input
    assert "Field types:" in sampled_input
    assert "row0" in sampled_input
    assert "row799" not in sampled_input


def test_compute_only_excel_asset_uses_raw_workbook_storage_and_top_level_table_fields(tmp_path: Path) -> None:
    table_anchor = "[ASSET_ANCHOR:ledger-table]"
    sample_text = "\n".join(
        [
            "Table policy: compute_only",
            "Table shape: rows=800, columns=3, estimated_tokens=20000",
            "Table columns: Name | Department | Amount",
            "Field types: Name=text; Department=enum(dept0, dept1, dept2); Amount=number(min=0, max=79900)",
            "Sample rows:",
            "| Name | Department | Amount |",
            "|---|---|---|",
            "| row0 | dept0 | 0 |",
            "| row1 | dept1 | 100 |",
        ]
    )
    parsed_doc = ParsedDocument(
        title="Ledger",
        source_type=SourceType.XLSX,

        authors=["tester"],
        language="zh",
        visible_text=f"Ledger table {table_anchor}",
        sections=[
            ParsedSection(
                toc_path=("Ledger", "Sheet1"),
                heading_level=2,
                page_range=(1, 1),
                order_index=0,
                text=f"Ledger table {table_anchor}",
                char_range_start=0,
                char_range_end=len(f"Ledger table {table_anchor}"),
                anchor_hint="ledger-sheet1",
            )
        ],
        elements=[
            ParsedElement(
                element_id="ledger-table",
                kind="table",
                text=sample_text,
                toc_path=("Ledger", "Sheet1"),
                page_no=1,
                metadata={
                    "source_type": "xlsx",
                    "sheet_name": "Sheet1",
                    "row_count": 800,
                    "column_count": 3,
                    "estimated_tokens": 20_000,
                    "table_policy": "compute_only",
                    "asset_summary_sample": sample_text,
                    "sample_rows": [{"Name": "row0", "Department": "dept0", "Amount": "0"}],
                    "schema": [
                        {"name": "Name", "type": "text"},
                        {"name": "Department", "type": "enum(dept0, dept1, dept2)"},
                        {"name": "Amount", "type": "number(min=0, max=79900)"},
                    ],
                },
            )
        ],
        page_count=1,
    )

    source_path = tmp_path / "ledger.xlsx"
    source_bytes = b"fake xlsx bytes"
    source_path.write_bytes(source_bytes)
    object_store = FileObjectStore(tmp_path / "objects")
    metadata_repo = _MetadataRepo()

    pipeline = IngestPipeline(
        dispatcher=_Dispatcher(parsed_doc),  # type: ignore[arg-type]
        summarizer=_Summarizer(),  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
        metadata_repo=metadata_repo,  # type: ignore[arg-type]
        summary_repo=_SummaryRepo(),  # type: ignore[arg-type]
        object_store=object_store,
        section_refiner=SectionRefiner(token_accounting=_token_accounting(chunk_size=64)),
    )

    pipeline.run(
        IngestRequest(
            location=str(source_path),
            file_path=source_path,
            source_type=SourceType.XLSX,
            owner="tester",
            title="Ledger",
        )
    )

    saved_asset = metadata_repo.assets[0]
    assert saved_asset.sheet_name == "Sheet1"
    assert saved_asset.row_count == 800
    assert saved_asset.column_count == 3
    assert saved_asset.sample_rows == [{"Name": "row0", "Department": "dept0", "Amount": "0"}]
    assert saved_asset.schema[2]["name"] == "Amount"
    stored = object_store.read_bytes(saved_asset.storage_key)
    assert len(stored) > 0
    assert stored[:4] == b"PAR1"


def test_doc_summary_reduces_generated_section_and_asset_summaries(tmp_path: Path) -> None:
    first_text = "first policy section with reimbursement limits"
    second_text = "second policy section with approval workflow"
    table_anchor = "[ASSET_ANCHOR:table-1]"
    table_markdown = "| Item | Limit |\n|---|---|\n| Hotel | 500 |"
    visible_text = f"{first_text}\n\n{second_text} {table_anchor}"
    parsed_doc = ParsedDocument(
        title="Policy",
        source_type=SourceType.DOCX,

        authors=["tester"],
        language="zh",
        visible_text=visible_text,
        sections=[
            ParsedSection(
                toc_path=("Policy", "First"),
                heading_level=2,
                page_range=(1, 1),
                order_index=0,
                text=first_text,
                char_range_start=0,
                char_range_end=len(first_text),
                anchor_hint="policy-first",
            ),
            ParsedSection(
                toc_path=("Policy", "Second"),
                heading_level=2,
                page_range=(1, 1),
                order_index=1,
                text=f"{second_text} {table_anchor}",
                char_range_start=len(first_text) + 2,
                char_range_end=len(visible_text),
                anchor_hint="policy-second",
            ),
        ],
        elements=[
            ParsedElement(
                element_id="table-1",
                kind="table",
                text=table_markdown,
                toc_path=("Policy", "Second"),
                page_no=1,
            )
        ],
        page_count=1,
    )
    source_path = tmp_path / "policy.docx"
    source_path.write_bytes(b"fake docx bytes")
    object_store = FileObjectStore(tmp_path / "objects")
    metadata_repo = _MetadataRepo()
    summary_repo = _SummaryRepo()
    summarizer = _Summarizer()

    pipeline = IngestPipeline(
        dispatcher=_Dispatcher(parsed_doc),  # type: ignore[arg-type]
        summarizer=summarizer,  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
        metadata_repo=metadata_repo,  # type: ignore[arg-type]
        summary_repo=summary_repo,  # type: ignore[arg-type]
        object_store=object_store,
        section_refiner=SectionRefiner(token_accounting=_token_accounting(chunk_size=128)),
    )

    pipeline.run(
        IngestRequest(
            location=str(source_path),
            file_path=source_path,
            source_type=SourceType.DOCX,
            owner="tester",
            title="Policy",
        )
    )

    doc_records = [record for record in summary_repo.records if record.__class__.__name__ == "DocSummaryRecord"]
    assert len(doc_records) == 1
    assert summarizer.doc_inputs == [
        (
            [
                f"Policy: {first_text}",
                f"Policy: {second_text} {table_anchor}",
            ],
            [f"Policy table asset summary: {summarizer.asset_inputs[0]}"],
        )
    ]
    assert doc_records[0].summary_text.startswith("Policy doc summary from Policy: first policy section")
