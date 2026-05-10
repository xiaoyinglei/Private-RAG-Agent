from __future__ import annotations

from bs4 import BeautifulSoup

from rag.ingest.parsers.util import default_title_from_location, normalize_whitespace, slugify
from rag.schema.core import ParsedDocument, ParsedSection, SourceType


class WebParserRepo:
    def parse(
        self,
        html: str,
        *,
        location: str,
        title: str | None = None,
        owner: str = "user",
    ) -> ParsedDocument:
        soup = BeautifulSoup(html, "html.parser")
        document_title = title
        if not document_title and soup.title is not None:
            document_title = normalize_whitespace(soup.title.get_text(" ", strip=True))
        if not document_title:
            first_heading = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"])
            document_title = (
                normalize_whitespace(first_heading.get_text(" ", strip=True))
                if first_heading is not None
                else default_title_from_location(location)
            )

        sections: list[ParsedSection] = []
        heading_stack: list[str] = [document_title]
        current_lines: list[str] = []
        current_order = 0
        current_heading_level: int | None = 1
        visible_parts: list[str] = []
        visible_cursor = 0
        visible_section_separator = " "

        def flush_section() -> None:
            nonlocal current_lines, current_order, visible_cursor
            if not current_lines:
                return
            text = normalize_whitespace("\n".join(current_lines))
            if not text:
                current_lines = []
                return
            if visible_parts:
                visible_parts.append(visible_section_separator)
                visible_cursor += len(visible_section_separator)
            char_range_start = visible_cursor
            visible_parts.append(text)
            visible_cursor += len(text)
            char_range_end = visible_cursor
            sections.append(
                ParsedSection(
                    toc_path=tuple(heading_stack),
                    heading_level=current_heading_level,
                    page_range=None,
                    order_index=current_order,
                    text=text,
                    char_range_start=char_range_start,
                    char_range_end=char_range_end,
                    anchor_hint=slugify(" ".join(heading_stack)),
                )
            )
            current_lines = []
            current_order += 1

        container = soup.article or soup.body or soup
        for tag in container.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p"],
            recursive=True,
        ):
            if tag.name is None:
                continue
            if tag.name.startswith("h"):
                heading_level = int(tag.name[1])
                heading_text = normalize_whitespace(tag.get_text(" ", strip=True))
                flush_section()
                heading_stack = heading_stack[: max(heading_level - 1, 0)] + [heading_text]
                current_heading_level = heading_level
                continue
            paragraph = normalize_whitespace(tag.get_text(" ", strip=True))
            if paragraph:
                current_lines.append(paragraph)

        flush_section()

        if not sections:
            root_text = normalize_whitespace(soup.get_text(" ", strip=True)) or document_title
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
                )
            ]
            visible_parts = [root_text]

        visible_text = "".join(visible_parts)
        return ParsedDocument(
            title=document_title,
            source_type=SourceType.WEB,
            authors=[owner],
            language="en",
            sections=sections,
            visible_text=visible_text,
            metadata={"location": location, "source_type": "web"},
        )
