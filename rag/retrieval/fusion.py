from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag.retrieval.evidence import CandidateLike
from rag.retrieval.models import FusedCandidateView


@dataclass(frozen=True)
class FusedCandidate:
    candidate: CandidateLike
    fused_score: float
    rank: int
    supporting_branches: int
    branch_scores: dict[str, float]


@dataclass(slots=True)
class ReciprocalRankFusion:
    rank_constant: int = 60
    alpha: float = 0.65

    def fuse(
        self,
        *,
        query: str,
        retrieval_profile: object,
        branches: Sequence[tuple[str, Sequence[CandidateLike]]],
        alpha: float | None = None,
    ) -> list[CandidateLike]:
        del query, retrieval_profile
        blend = self._normalized_alpha(alpha)
        branch_weights = self._branch_weights(branches, alpha=blend)
        fused: dict[str, FusedCandidate] = {}
        for branch_name, branch in branches:
            branch_weight = branch_weights.get(branch_name, 1.0)
            for index, candidate in enumerate(branch, start=1):
                score = branch_weight * (1.0 / (self.rank_constant + index))
                candidate_id = candidate.item_id
                existing = fused.get(candidate_id)
                branch_scores = {branch_name: max(float(candidate.score), 0.0)}
                if existing is None:
                    fused[candidate_id] = FusedCandidate(
                        candidate=candidate,
                        fused_score=score,
                        rank=index,
                        supporting_branches=1,
                        branch_scores=branch_scores,
                    )
                    continue
                merged_scores = dict(existing.branch_scores)
                merged_scores.update(branch_scores)
                fused[candidate_id] = FusedCandidate(
                    candidate=existing.candidate,
                    fused_score=existing.fused_score + score,
                    rank=min(existing.rank, index),
                    supporting_branches=existing.supporting_branches + 1,
                    branch_scores=merged_scores,
                )

        ordered = sorted(
            fused.values(),
            key=lambda item: (-item.fused_score, -item.supporting_branches, item.rank, item.candidate.item_id),
        )
        return [self._to_view(item, index) for index, item in enumerate(ordered, start=1)]

    def _normalized_alpha(self, alpha: float | None) -> float:
        if alpha is None:
            alpha = self.alpha
        return max(0.0, min(float(alpha), 1.0))

    @classmethod
    def _branch_weights(
        cls,
        branches: Sequence[tuple[str, Sequence[CandidateLike]]],
        *,
        alpha: float,
    ) -> dict[str, float]:
        branch_names = [branch_name for branch_name, branch in branches if branch]
        if len(branch_names) <= 1:
            return {branch_name: 1.0 for branch_name in branch_names}
        return {branch_name: 1.0 / len(branch_names) for branch_name in branch_names}

    @staticmethod
    def _to_view(item: FusedCandidate, unified_rank: int) -> FusedCandidateView:
        final_score = item.fused_score
        return FusedCandidateView(
            evidence_id=item.candidate.item_id,
            doc_id=item.candidate.doc_id,
            benchmark_doc_id=getattr(item.candidate, "benchmark_doc_id", None),
            text=item.candidate.text,
            citation_anchor=item.candidate.citation_anchor,
            score=final_score,
            rank=item.rank,
            source_kind=item.candidate.source_kind,
            source_id=item.candidate.source_id,
            section_path=tuple(item.candidate.section_path),
            effective_access_policy=getattr(item.candidate, "effective_access_policy", None),
            metadata=getattr(item.candidate, "metadata", None),
            record_type=getattr(item.candidate, "record_type", None),
            retrieval_channels=sorted(
                {*item.branch_scores, *(getattr(item.candidate, "retrieval_channels", None) or ())}
            ),
            dense_score=item.branch_scores.get("vector"),
            sparse_score=item.branch_scores.get("section") or item.branch_scores.get("metadata"),
            special_score=item.branch_scores.get("special"),
            structure_score=item.branch_scores.get("section"),
            metadata_score=item.branch_scores.get("metadata"),
            fusion_score=final_score,
            rrf_score=final_score,
            unified_rank=unified_rank,
            grounding_target=getattr(item.candidate, "grounding_target", None),
        )


__all__ = ["FusedCandidate", "ReciprocalRankFusion"]
