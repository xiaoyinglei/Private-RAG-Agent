"""Evidence criticism and retry recommendation logic for agent subtasks."""

from __future__ import annotations

from itertools import combinations

from rag.agent.schema import CriticAction, EvidenceAssessment, SubTask
from rag.retrieval.models import RetrievalResult
from rag.utils.text import keyword_overlap, search_terms

_NEGATION_MARKERS = (" not ", " no ", " does not ", " cannot ", " without ", "未", "不")


class EvidenceCritic:
    """Judge whether a subtask has enough evidence and recommend the next action."""

    def __init__(self, *, enable_llm: bool = False) -> None:
        self._enable_llm = enable_llm

    def assess(
        self,
        *,
        subtask: SubTask,
        retrieval: RetrievalResult,
        attempt_index: int,
        retry_budget_remaining: int,
        allow_web: bool,
    ) -> EvidenceAssessment:
        evidence = retrieval.evidence.all
        evidence_text = " ".join(item.text for item in evidence)
        missing_dimensions = [
            dimension
            for dimension in subtask.expected_evidence
            if keyword_overlap(search_terms(dimension), evidence_text) == 0
        ]
        conflicts = self._detect_conflicts(retrieval)
        sufficient = retrieval.self_check.evidence_sufficient and not missing_dimensions and not conflicts
        confidence = self._confidence(
            evidence_count=len(evidence),
            sufficient=sufficient,
            missing_dimensions=missing_dimensions,
            conflicts=conflicts,
        )
        recommended_action = self._recommended_action(
            sufficient=sufficient,
            evidence_count=len(evidence),
            missing_dimensions=missing_dimensions,
            conflicts=conflicts,
            retry_budget_remaining=retry_budget_remaining,
            allow_web=allow_web and subtask.allow_web,
            retrieve_more=retrieval.self_check.retrieve_more,
            attempt_index=attempt_index,
        )
        return EvidenceAssessment(
            sufficient=sufficient,
            confidence=confidence,
            missing_dimensions=missing_dimensions,
            conflicts=conflicts,
            recommended_action=recommended_action,
        )

    @staticmethod
    def _confidence(
        *,
        evidence_count: int,
        sufficient: bool,
        missing_dimensions: list[str],
        conflicts: list[str],
    ) -> float:
        score = min(0.6, evidence_count * 0.15)
        if sufficient:
            score += 0.3
        score -= min(0.3, len(missing_dimensions) * 0.1)
        if conflicts:
            score -= 0.25
        return max(0.0, min(1.0, score))

    @staticmethod
    def _recommended_action(
        *,
        sufficient: bool,
        evidence_count: int,
        missing_dimensions: list[str],
        conflicts: list[str],
        retry_budget_remaining: int,
        allow_web: bool,
        retrieve_more: bool,
        attempt_index: int,
    ) -> CriticAction:
        del attempt_index
        if sufficient:
            return CriticAction.ACCEPT
        if conflicts and retry_budget_remaining <= 0:
            return CriticAction.ABSTAIN
        if retry_budget_remaining <= 0:
            return CriticAction.ABSTAIN
        if allow_web and retrieve_more and evidence_count == 0:
            return CriticAction.ENABLE_WEB
        if evidence_count == 0 or missing_dimensions:
            return CriticAction.RETRY_REWRITE_QUERY
        if retrieve_more:
            return CriticAction.RETRY_SAME_SCOPE
        return CriticAction.ABSTAIN

    @classmethod
    def _detect_conflicts(cls, retrieval: RetrievalResult) -> list[str]:
        conflicts: list[str] = []
        evidence = retrieval.evidence.internal
        for left, right in combinations(evidence, 2):
            left_text = f" {left.text.strip().lower()} "
            right_text = f" {right.text.strip().lower()} "
            left_negated = any(marker in left_text for marker in _NEGATION_MARKERS)
            right_negated = any(marker in right_text for marker in _NEGATION_MARKERS)
            if left_negated == right_negated:
                continue
            overlap = keyword_overlap(search_terms(left.text), right.text)
            if overlap == 0:
                continue
            conflicts.append(f"Conflicting evidence between {left.evidence_id} and {right.evidence_id}.")
        return conflicts


__all__ = ["EvidenceCritic"]
