from __future__ import annotations

from pathlib import Path

import pandas as pd

from rag.ingest.asset_anchors import asset_anchor
from rag.ingest.header_detector import HeaderKind, detect_header
from rag.ingest.parsers.util import default_title_from_location, normalize_whitespace, slugify
from rag.ingest.table_sampler import profile_table_data
from rag.schema.core import ParsedDocument, ParsedElement, ParsedSection, SourceType

_MIN_DETECT_CONFIDENCE = 0.5


class ExcelParserRepo:
    """
    专门针对 Excel (XLSX/XLS) 格式的解析特种兵。
    核心使命：拒绝视觉渲染，直接读取底层数据，保住极其脆弱的二维表格关系。
    """

    def parse(
        self,
        file_path: Path,
        *,
        location: str,
        source_type: SourceType,
        title: str | None = None,
        owner: str = "user",
    ) -> ParsedDocument:
        doc_title = title or default_title_from_location(location)

        try:
            # 读取所有 sheet
            sheet_dict = pd.read_excel(file_path, sheet_name=None)
        except Exception as exc:
            raise ValueError(f"Excel parser failed to read {location}: {exc}") from exc

        sections: list[ParsedSection] = []
        all_elements: list[ParsedElement] = []
        visible_text_parts: list[str] = []
        visible_cursor = 0
        visible_section_separator = "\n\n"

        # 维护一个真实的有效章节计数器
        valid_section_count = 0
        total_sheets = len(sheet_dict)

        for sheet_idx, (sheet_name, df) in enumerate(sheet_dict.items()):
            # 1. 物理清洗
            df = df.dropna(how="all").dropna(axis=1, how="all")
            df = df.fillna("")

            if df.empty:
                continue

            clean_sheet_name = normalize_whitespace(sheet_name)
            element_id = f"{slugify(doc_title)}-sheet-{sheet_idx}-table"

            # 2. 用 header=None 重读原 sheet，走通用表头检测
            raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            raw_df = raw_df.dropna(how="all").dropna(axis=1, how="all").fillna("")
            header_result = detect_header(raw_df)

            columns: list[str]
            rows: list[list[str]]
            if header_result.confidence >= _MIN_DETECT_CONFIDENCE and header_result.header_kind != HeaderKind.NONE:
                columns = [
                    normalize_whitespace(col) or f"column_{i + 1}"
                    for i, col in enumerate(header_result.normalized_columns)
                ]
                data_start = header_result.data_start_row
                rows = [
                    [self._cell_text(raw_df.iat[r, c]) for c in range(raw_df.shape[1])]
                    for r in range(data_start, raw_df.shape[0])
                ]
            else:
                columns = [
                    normalize_whitespace(str(column)) or f"column_{column_index + 1}"
                    for column_index, column in enumerate(df.columns)
                ]
                rows = [
                    [self._cell_text(value) for value in row]
                    for row in df.itertuples(index=False, name=None)
                ]

            rows = [row for row in rows if any(self._cell_text(v).strip() for v in row)]
            table_profile = profile_table_data(columns=columns, rows=rows)
            asset_text = table_profile.summary_sample

            # 3. 封装 Element
            all_elements.append(
                ParsedElement(
                    element_id=element_id,
                    kind="table",
                    text=asset_text,
                    toc_path=(doc_title, clean_sheet_name),
                    page_no=sheet_idx + 1,
                    metadata={
                        "source_type": SourceType.XLSX.value,
                        "sheet_name": clean_sheet_name,
                        "row_count": table_profile.row_count,
                        "column_count": table_profile.column_count,
                        "estimated_tokens": table_profile.estimated_tokens,
                        "table_policy": table_profile.table_policy,
                        "asset_summary_sample": table_profile.summary_sample,
                        "asset_text_preview": table_profile.preview_text,
                        "sample_rows": table_profile.sample_rows,
                        "schema": table_profile.schema,
                        "header_kind": header_result.header_kind.name,
                        "header_confidence": header_result.confidence,
                    },
                )
            )

            section_text = f"## Sheet: {clean_sheet_name}\n\n{asset_anchor(element_id)}"

            if visible_text_parts:
                visible_text_parts.append(visible_section_separator)
                visible_cursor += len(visible_section_separator)

            char_range_start = visible_cursor
            visible_text_parts.append(section_text)
            visible_cursor += len(section_text)
            char_range_end = visible_cursor

            # 4. 封装 Section
            sections.append(
                ParsedSection(
                    toc_path=(doc_title, clean_sheet_name),
                    heading_level=2,
                    page_range=(sheet_idx + 1, sheet_idx + 1),
                    order_index=valid_section_count,
                    text=section_text,
                    char_range_start=char_range_start,
                    char_range_end=char_range_end,
                    anchor_hint=slugify(f"sheet-{clean_sheet_name}"),
                    metadata={"sheet_name": clean_sheet_name},
                )
            )

            valid_section_count += 1
        
        visible_text = "".join(visible_text_parts)

        for idx, section in enumerate(sections):
            start = section.char_range_start
            end = section.char_range_end
            if start is None or end is None:
                raise ValueError(f"xlsx section[{idx}] missing char range")
            if start < 0 or end <= start or end > len(visible_text):
                raise ValueError(
                    f"xlsx section[{idx}] invalid char range: start={start}, end={end}, visible_len={len(visible_text)}"
                )
            if visible_text[start:end] != section.text:
                raise ValueError(
                    f"xlsx section[{idx}] text/span mismatch: "
                    f"expected={section.text!r}, actual={visible_text[start:end]!r}"
                )

        return ParsedDocument(
            title=doc_title,
            source_type=SourceType.XLSX,
            authors=[owner],
            language=None,
            sections=sections,
            visible_text=visible_text,
            visual_semantics=None,
            elements=all_elements,
            page_count=total_sheets,  # 记录物理总数，无论是否为空
            metadata={
                "location": location,
                "source_type": SourceType.XLSX.value,
                "valid_sections": str(valid_section_count),
            },
        )

    @staticmethod
    def _cell_text(value: object) -> str:
        text = "" if value is None else str(value).strip()
        return normalize_whitespace(text)
