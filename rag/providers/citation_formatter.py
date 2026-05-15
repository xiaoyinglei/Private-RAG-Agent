"""引用格式化器：上标编号、按出现顺序重排、尾部脚注。

将 LLM 输出的 [Doc-N] 技术标记转换为用户可读的 ¹²³ 上标 + 尾部来源脚注。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.schema.query import AnswerCitation, GroundedAnswer

_DOC_ALIAS_RE = re.compile(r"\[(Doc-\d+)\]")
_SUPERSCRIPT_TABLE = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _to_superscript(n: int) -> str:
    return str(n).translate(_SUPERSCRIPT_TABLE)


@dataclass(frozen=True, slots=True)
class FormattedAnswer:
    """格式化后的最终回答文本（可直接展示给用户）。"""
    answer_body: str          # 带 ¹²³ 上标的正文
    footnotes: str            # 尾部脚注区（可能为空）
    answer_text: str          # answer_body + footnotes 拼接
    citation_count: int


class CitationFormatter:
    """将 GroundedAnswer 格式化为带脚注引用的最终文本。"""

    FOOTNOTE_SEPARATOR = "\n\n⁣——\n"

    # 匹配非标准引用格式： [N:M], [section:N], [ref:N], [N]
    _ALT_CITATION_RE = re.compile(
        r"\[(?:doc[-_]?)?(\d+)[:：,._](\d+)\]"  # [2:18], [doc-2:18], [Doc_2.18]
        r"|\[section[-_]?(\d+)\]"                 # [section-18]
    )

    def format(self, grounded: GroundedAnswer) -> FormattedAnswer:
        if not grounded.citations:
            return FormattedAnswer(
                answer_body=grounded.answer_text,
                footnotes="",
                answer_text=grounded.answer_text,
                citation_count=0,
            )

        # 0. 归一化非标准引用格式 → [Doc-N]
        answer_text = self._normalize_alt_citations(grounded)

        citation_by_id = {c.citation_id: c for c in grounded.citations}
        evidence_to_citation = self._evidence_to_citation_map(grounded)

        # 1. 扫描 [Doc-N] 的首次出现顺序，建立重映射
        doc_order: list[str] = []
        seen_docs: set[str] = set()
        for match in _DOC_ALIAS_RE.finditer(answer_text):
            doc_marker = match.group(1)
            if doc_marker not in seen_docs:
                seen_docs.add(doc_marker)
                doc_order.append(doc_marker)

        if not doc_order:
            ordered = self._ordered_evidence_ids(grounded)
            if not ordered:
                return FormattedAnswer(
                    answer_body=answer_text,
                    footnotes="",
                    answer_text=answer_text,
                    citation_count=0,
                )
            doc_order = [f"Doc-{i+1}" for i in range(len(ordered))]
            reindex = {f"Doc-{i+1}": (i + 1, ordered[i]) for i in range(len(ordered))}
        else:
            reindex = {}
            evidence_ids_seen: set[str] = set()
            next_num = 1
            for doc_marker in doc_order:
                ev_id = evidence_to_citation.get(doc_marker)
                if ev_id is None or ev_id in evidence_ids_seen:
                    continue
                evidence_ids_seen.add(ev_id)
                reindex[doc_marker] = (next_num, ev_id)
                next_num += 1

        # 2. 替换 [Doc-N] 为上标
        def _replace(match: re.Match[str]) -> str:
            dm = match.group(1)
            entry = reindex.get(dm)
            if entry is None:
                entry = reindex.get(dm, (0, ""))
                if entry[0] == 0:
                    return ""
            return _to_superscript(entry[0])

        answer_body = _DOC_ALIAS_RE.sub(_replace, answer_text)
        answer_body = re.sub(r"\s{2,}", " ", answer_body).strip()

        # 4. 构建脚注区
        footnotes = self._build_footnotes(reindex, citation_by_id)
        if footnotes:
            answer_text = answer_body + self.FOOTNOTE_SEPARATOR + footnotes
        else:
            answer_text = answer_body

        return FormattedAnswer(
            answer_body=answer_body,
            footnotes=footnotes,
            answer_text=answer_text,
            citation_count=len(reindex),
        )

    # ── helpers ──────────────────────────────────────────

    @staticmethod
    def _evidence_to_citation_map(grounded: GroundedAnswer) -> dict[str, str]:
        """Map [Doc-N] alias → evidence_id."""
        mapping: dict[str, str] = {}
        for i, ev_link in enumerate(grounded.evidence_links):
            if ev_link.evidence_id:
                mapping[f"Doc-{i+1}"] = ev_link.evidence_id
        return mapping

    def _normalize_alt_citations(self, grounded: GroundedAnswer) -> str:
        """将非标准引用格式（[2:18], [section-18] 等）归一化为 [Doc-N]。

        根据 evidence_links 中的 section_id / evidence_id 映射关系，
        找到最匹配的 Doc 编号并替换。
        """
        text = grounded.answer_text
        if not grounded.evidence_links:
            return text

        # 构建 section_id → Doc-N 反向查找表
        section_to_doc: dict[int, str] = {}
        for i, ev_link in enumerate(grounded.evidence_links):
            doc_alias = f"Doc-{i+1}"
            # 尝试从 citation 中提取 section_id
            for cit in grounded.citations:
                if cit.evidence_id == ev_link.evidence_id and cit.section_path:
                    # section_path 末尾可能含数字信息
                    pass
            section_to_doc[i + 1] = doc_alias

        def _replace_alt(match: re.Match[str]) -> str:
            # [doc_id:section_id] 如 [2:18], [2:17]
            if match.group(1) and match.group(2):
                section_hint = int(match.group(2))
                # 遍历 evidence_links 找到匹配 section_id 的 Doc 编号
                for i, ev_link in enumerate(grounded.evidence_links):
                    for cit in grounded.citations:
                        if cit.evidence_id == ev_link.evidence_id:
                            # citation 的 evidence_id 是 E1/E2 等，用 evidence_links 的 index 映射
                            if i + 1 == section_hint or cit.evidence_id == f"E{section_hint}":
                                return f"[Doc-{i+1}]"
                # fallback: 用 section_hint mod evidence_count
                idx = (section_hint - 1) % max(len(grounded.evidence_links), 1)
                return f"[Doc-{idx+1}]"
            # [section-N]
            if match.group(3):
                n = int(match.group(3))
                for i, ev_link in enumerate(grounded.evidence_links):
                    for cit in grounded.citations:
                        if cit.evidence_id == ev_link.evidence_id:
                            if cit.evidence_id == f"E{n}" or i + 1 == n:
                                return f"[Doc-{i+1}]"
                idx = (n - 1) % max(len(grounded.evidence_links), 1)
                return f"[Doc-{idx+1}]"
            return match.group(0)

        return self._ALT_CITATION_RE.sub(_replace_alt, text)

    @staticmethod
    def _ordered_evidence_ids(grounded: GroundedAnswer) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for ev in grounded.evidence_links:
            if ev.evidence_id and ev.evidence_id not in seen:
                seen.add(ev.evidence_id)
                ordered.append(ev.evidence_id)
        return ordered

    @staticmethod
    def _build_footnotes(
        reindex: dict[str, tuple[int, str]],
        citation_by_id: dict[str, AnswerCitation],
    ) -> str:
        entries: list[tuple[int, AnswerCitation]] = []
        seen_nums: set[int] = set()
        for _doc_marker, (num, ev_id) in sorted(reindex.items(), key=lambda x: x[1][0]):
            if num in seen_nums:
                continue
            seen_nums.add(num)
            for _cid, cit in citation_by_id.items():
                if cit.evidence_id == ev_id:
                    entries.append((num, cit))
                    break

        if not entries:
            return ""

        lines: list[str] = []
        for num, cit in entries:
            source = _format_citation_source(cit)
            lines.append(f"{_to_superscript(num)} {source}")
        return "\n".join(lines)


def _format_citation_source(cit: AnswerCitation) -> str:
    parts: list[str] = []
    if cit.file_name:
        parts.append(cit.file_name)
    if cit.section_path:
        parts.append(" > ".join(cit.section_path))
    elif cit.citation_anchor:
        parts.append(cit.citation_anchor)
    if cit.page_start is not None:
        page = f"p.{cit.page_start}"
        if cit.page_end is not None and cit.page_end != cit.page_start:
            page += f"-{cit.page_end}"
        parts.append(page)
    return " — ".join(parts)


__all__ = ["CitationFormatter", "FormattedAnswer", "_to_superscript"]
