from __future__ import annotations

from rag.providers.generation import AnswerGenerationService, AnswerSectionPayload, StructuredAnswerPayload
from rag.schema.query import EvidenceItem, GroundingTarget


def test_answer_generation_removes_fabricated_doc_aliases() -> None:
    service = AnswerGenerationService()
    evidence = [
        EvidenceItem(
            evidence_id="E1",
            doc_id=1,
            source_id=2,
            citation_anchor="Architecture",
            text="Alpha Engine handles ingestion requests.",
            score=0.95,
            record_type="section",
            grounding_target=GroundingTarget(
                kind="section",
                doc_id=1,
                source_id=2,
                section_id=7,
            ),
        )
    ]

    answer = service.answer_from_structured_payload(
        query="What does Alpha Engine handle?",
        evidence_pack=evidence,
        grounded_candidate="Alpha Engine handles ingestion requests.",
        payload=StructuredAnswerPayload(
            answer_text="Alpha Engine handles ingestion requests. [Doc-99]",
            answer_sections=[
                AnswerSectionPayload(
                    title="Direct",
                    text="Alpha Engine handles ingestion requests. [Doc-99]",
                    evidence_ids=["E1"],
                )
            ],
        ),
        trust_evidence_pack=True,
    )

    assert "[Doc-99]" not in answer.answer_text
    assert "[Doc-99]" not in answer.answer_sections[0].text
    assert "[1:7]" in answer.answer_text
    assert "[1:7]" in answer.answer_sections[0].text
