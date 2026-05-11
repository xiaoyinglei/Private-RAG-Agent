from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.schema.query import (
    EvidenceItem,
    RetrievalSignals,
    TaskType,
)
from rag.schema.runtime import AccessPolicy, RuntimeMode


class CandidateLike(Protocol):
    doc_id: int | str
    benchmark_doc_id: str | None
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str
    source_id: int | str | None
    section_path: Sequence[str]
    item_id: str


def classify_retrieval_family(
    *,
    evidence_kind: str,
    record_type: str | None,
    retrieval_channels: Sequence[str] = (),
) -> str:
    channels = {channel for channel in retrieval_channels if channel}
    if evidence_kind == "external":
        return "external"
    if (record_type or "").startswith("asset") or channels & {"special", "section", "metadata"}:
        return "multimodal"
    if evidence_kind == "graph" or channels & {"local", "global"}:
        return "kg"
    has_vector = "vector" in channels
    has_sparse = "sparse" in channels
    if has_vector and has_sparse:
        return "hybrid"
    if has_sparse:
        return "sparse"
    if has_vector:
        return "vector"
    return "kg" if evidence_kind == "graph" else "vector"


class EvidenceThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    fast_min_evidence_items: int = 2
    fast_min_sections: int = 1
    deep_min_evidence_items: int = 4
    deep_min_supporting_units: int = 2


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    internal: list[EvidenceItem] = Field(default_factory=list)
    external: list[EvidenceItem] = Field(default_factory=list)
    graph: list[EvidenceItem] = Field(default_factory=list)

    @property
    def all(self) -> list[EvidenceItem]:
        return [*self.internal, *self.external, *self.graph]


class SelfCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieve_more: bool
    evidence_sufficient: bool
    claim_supported: bool


class EvidenceService:
    def __init__(self, thresholds: EvidenceThresholds | None = None) -> None:
        self._thresholds = thresholds or EvidenceThresholds()

    @staticmethod
    def _safe_int(value: object, *, default: int | None = None) -> int | None:
        if value in {None, ""}:
            return default
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _candidate_source_scope(candidate: CandidateLike) -> set[str]:
        scope = {str(candidate.doc_id)}
        if candidate.source_id:
            scope.add(str(candidate.source_id))
        return scope

    @staticmethod
    def _candidate_access_policy(candidate: object) -> AccessPolicy | None:
        policy = getattr(candidate, "effective_access_policy", None)
        if isinstance(policy, AccessPolicy):
            return policy
        return None

    @staticmethod
    def _is_candidate_external(candidate: object) -> bool:
        return getattr(candidate, "source_kind", "internal") == "external"

    @staticmethod
    def _is_candidate_graph(candidate: object) -> bool:
        return getattr(candidate, "source_kind", "internal") == "graph"

    def filter_candidates(
        self,
        candidates: Sequence[CandidateLike],
        *,
        source_scope: Sequence[str],
        access_policy: AccessPolicy,
        runtime_mode: RuntimeMode,
        retrieval_signals: RetrievalSignals | None = None,
    ) -> list[CandidateLike]:
        allowed_scope = set(source_scope)
        filtered: list[CandidateLike] = []
        for candidate in candidates:
            if allowed_scope and not self._candidate_source_scope(candidate) & allowed_scope:
                continue
            if self._is_candidate_external(candidate) and access_policy.external_retrieval.value != "allow":
                continue
            candidate_policy = self._candidate_access_policy(candidate)
            if candidate_policy is not None:
                if runtime_mode not in candidate_policy.allowed_runtimes:
                    continue
                if not (candidate_policy.allowed_locations & access_policy.allowed_locations):
                    continue
            if retrieval_signals is not None and not self._matches_explicit_constraints(
                candidate, retrieval_signals
            ):
                continue
            filtered.append(candidate)
        return filtered

    @staticmethod
    def _matches_explicit_constraints(candidate: CandidateLike, retrieval_signals: RetrievalSignals) -> bool:
        if getattr(candidate, "source_kind", "internal") == "external":
            return True
        if not retrieval_signals.has_constraints():
            return True
        if not EvidenceService._matches_metadata_constraints(candidate, retrieval_signals):
            return False
        if not EvidenceService._matches_structure_constraints(candidate, retrieval_signals):
            return False
        return True

    @staticmethod
    def _matches_metadata_constraints(candidate: CandidateLike, retrieval_signals: RetrievalSignals) -> bool:
        metadata_filters = retrieval_signals.metadata_filters
        candidate_metadata = getattr(candidate, "metadata", {}) or {}
        candidate_source_type = getattr(candidate, "source_type", None) or candidate_metadata.get("source_type")
        if metadata_filters.source_types and candidate_source_type is not None:
            if candidate_source_type not in metadata_filters.source_types:
                return False

        candidate_file_name = getattr(candidate, "file_name", None)
        if metadata_filters.file_names and candidate_file_name is not None:
            if candidate_file_name not in metadata_filters.file_names:
                return False
        if metadata_filters.document_titles and candidate_file_name is not None:
            if candidate_file_name not in metadata_filters.document_titles:
                return False

        if not metadata_filters.page_numbers and not metadata_filters.page_ranges:
            return True
        candidate_pages = EvidenceService._candidate_pages(candidate)
        if not candidate_pages:
            return True
        if metadata_filters.page_numbers and not candidate_pages & set(metadata_filters.page_numbers):
            if not metadata_filters.page_ranges:
                return False
        if metadata_filters.page_ranges and not any(
            any(page_range.start <= page <= page_range.end for page in candidate_pages)
            for page_range in metadata_filters.page_ranges
        ):
            if not metadata_filters.page_numbers:
                return False
            if not candidate_pages & set(metadata_filters.page_numbers):
                return False
        return True

    @staticmethod
    def _matches_structure_constraints(candidate: CandidateLike, retrieval_signals: RetrievalSignals) -> bool:
        constraints = retrieval_signals.structure_constraints
        if not constraints.requires_structure_match:
            return True
        section_path = tuple(getattr(candidate, "section_path", ()) or ())
        if not section_path:
            return True
        section_text = " ".join(section_path).lower()
        match_terms = {term.lower() for term in constraints.focus_terms}
        if not match_terms:
            return True
        matched = any(term in section_text for term in match_terms)
        if constraints.prefer_heading_match:
            return matched
        return matched

    @staticmethod
    def _candidate_pages(candidate: CandidateLike) -> set[int]:
        metadata = getattr(candidate, "metadata", {}) or {}
        pages: set[int] = set()
        page_no = metadata.get("page_no")
        if isinstance(page_no, str) and page_no.isdigit():
            pages.add(int(page_no))
        page_start = getattr(candidate, "page_start", None)
        page_end = getattr(candidate, "page_end", None)
        if isinstance(page_start, int) and isinstance(page_end, int):
            pages.update(range(page_start, page_end + 1))
        elif isinstance(page_start, int):
            pages.add(page_start)
        elif isinstance(page_end, int):
            pages.add(page_end)
        return pages

    @staticmethod
    def _to_evidence_item(candidate: CandidateLike) -> EvidenceItem:
        evidence_kind = getattr(candidate, "source_kind", "internal")
        if evidence_kind not in {"internal", "external", "graph"}:
            evidence_kind = "internal"
        retrieval_channels = list(getattr(candidate, "retrieval_channels", []) or [])
        record_type = (
            getattr(candidate, "record_type", None)
            or getattr(getattr(candidate, "grounding_target", None), "kind", None)
        )
        retrieval_family = classify_retrieval_family(
            evidence_kind=evidence_kind,
            record_type=None if record_type is None else str(record_type),
            retrieval_channels=retrieval_channels,
        )
        return EvidenceItem(
            evidence_id=str(candidate.item_id),
            doc_id=EvidenceService._safe_int(candidate.doc_id, default=0) or 0,
            benchmark_doc_id=getattr(candidate, "benchmark_doc_id", None),
            source_id=EvidenceService._safe_int(getattr(candidate, "source_id", None), default=None),
            citation_anchor=str(candidate.citation_anchor),
            text=str(candidate.text),
            score=float(candidate.score),
            evidence_kind=evidence_kind,
            record_type=None if record_type is None else str(record_type),
            file_name=getattr(candidate, "file_name", None),
            section_path=list(getattr(candidate, "section_path", ()) or ()),
            page_start=getattr(candidate, "page_start", None),
            page_end=getattr(candidate, "page_end", None),
            source_type=getattr(candidate, "source_type", None),
            retrieval_channels=retrieval_channels,
            retrieval_family=retrieval_family,
            grounding_target=getattr(candidate, "grounding_target", None),
        )

    def assemble_bundle(self, candidates: Sequence[CandidateLike]) -> EvidenceBundle:
        internal: list[EvidenceItem] = []
        external: list[EvidenceItem] = []
        graph: list[EvidenceItem] = []
        for candidate in candidates:
            item = self._to_evidence_item(candidate)
            if item.evidence_kind == "external":
                external.append(item)
            elif item.evidence_kind == "graph":
                graph.append(item)
            else:
                internal.append(item)
        return EvidenceBundle(internal=internal, external=external, graph=graph)

    def evaluate_self_check(
        self,
        *,
        bundle: EvidenceBundle,
        task_type: TaskType,
        runtime_mode: RuntimeMode,
    ) -> SelfCheckResult:
        internal = bundle.internal
        section_keys = {item.citation_anchor if item.citation_anchor else item.evidence_id for item in internal}
        doc_ids = {item.doc_id for item in internal}

        if runtime_mode is RuntimeMode.FAST or task_type in {TaskType.LOOKUP, TaskType.SINGLE_DOC_QA}:
            evidence_sufficient = (
                len(internal) >= self._thresholds.fast_min_evidence_items
                and len(section_keys) >= self._thresholds.fast_min_sections
            )
        else:
            evidence_sufficient = len(internal) >= self._thresholds.deep_min_evidence_items and (
                len(doc_ids) >= self._thresholds.deep_min_supporting_units
                or len(section_keys) >= self._thresholds.deep_min_supporting_units
            )

        claim_supported = evidence_sufficient and bool(internal)
        retrieve_more = not evidence_sufficient
        return SelfCheckResult(
            retrieve_more=retrieve_more,
            evidence_sufficient=evidence_sufficient,
            claim_supported=claim_supported,
        )

    @staticmethod
    def evidence_counts(bundle: EvidenceBundle) -> Counter[str]:
        return Counter(item.evidence_kind for item in bundle.all)


class ContextEvidenceMerger:
    def merge(self, retrieval: object) -> list[EvidenceItem]:
        evidence = retrieval.evidence
        reranked_evidence_ids = list(getattr(retrieval, "reranked_evidence_ids", []) or [])
        if not reranked_evidence_ids:
            reranked_evidence_ids = [
                candidate.item_id for candidate in list(getattr(retrieval, "clean_items", []) or [])
            ]

        internal_by_id = {item.evidence_id: item for item in evidence.internal}
        ordered_internal = [
            internal_by_id[evidence_id] for evidence_id in reranked_evidence_ids if evidence_id in internal_by_id
        ]
        seen_internal = {item.evidence_id for item in ordered_internal}
        ordered_internal.extend(item for item in evidence.internal if item.evidence_id not in seen_internal)

        merged: list[EvidenceItem] = []
        merged_by_evidence_id: dict[str, EvidenceItem] = {}
        ordered_evidence_ids: list[str] = []

        for item in [*ordered_internal, *evidence.graph]:
            existing = merged_by_evidence_id.get(item.evidence_id)
            if existing is None:
                merged_by_evidence_id[item.evidence_id] = item
                ordered_evidence_ids.append(item.evidence_id)
                continue
            merged_by_evidence_id[item.evidence_id] = self._merge_duplicate_item(existing, item)

        merged.extend(merged_by_evidence_id[evidence_id] for evidence_id in ordered_evidence_ids)

        seen_external: set[str] = set()
        for item in evidence.external:
            if item.evidence_id in seen_external:
                continue
            seen_external.add(item.evidence_id)
            merged.append(item)
        return merged

    @staticmethod
    def _merge_duplicate_item(existing: EvidenceItem, incoming: EvidenceItem) -> EvidenceItem:
        preferred = existing
        secondary = incoming
        if existing.evidence_kind != "internal" and incoming.evidence_kind == "internal":
            preferred = incoming
            secondary = existing

        merged_kind = (
            "internal" if "internal" in {existing.evidence_kind, incoming.evidence_kind} else preferred.evidence_kind
        )
        merged_text = preferred.text if len(preferred.text) >= len(secondary.text) else secondary.text
        merged_channels = list(dict.fromkeys([*existing.retrieval_channels, *incoming.retrieval_channels]))
        merged_family = classify_retrieval_family(
            evidence_kind=merged_kind,
            record_type=preferred.record_type or secondary.record_type,
            retrieval_channels=merged_channels,
        )

        return preferred.model_copy(
            update={
                "evidence_kind": merged_kind,
                "score": max(float(existing.score), float(incoming.score)),
                "text": merged_text,
                "section_path": preferred.section_path or secondary.section_path,
                "file_name": preferred.file_name or secondary.file_name,
                "source_id": preferred.source_id or secondary.source_id,
                "record_type": preferred.record_type or secondary.record_type,
                "source_type": preferred.source_type or secondary.source_type,
                "page_start": preferred.page_start if preferred.page_start is not None else secondary.page_start,
                "page_end": preferred.page_end if preferred.page_end is not None else secondary.page_end,
                "retrieval_channels": merged_channels,
                "retrieval_family": merged_family,
            }
        )


__all__ = [
    "CandidateLike",
    "ContextEvidenceMerger",
    "EvidenceBundle",
    "EvidenceService",
    "EvidenceThresholds",
    "SelfCheckResult",
    "classify_retrieval_family",
]
