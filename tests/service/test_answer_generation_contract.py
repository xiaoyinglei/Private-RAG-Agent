from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from rag.providers.generation import AnswerGenerationService, AnswerSectionPayload, StructuredAnswerPayload
from rag.providers.generation import AnswerGenerator, GeneratorBinding
from rag.schema.query import EvidenceItem, GroundingTarget
from rag.schema.runtime import AccessPolicy, RuntimeMode


class _TextOnlyJsonGenerator:
    def __init__(self, output: str) -> None:
        self._output = output

    def generate_structured(self, *, prompt: str, schema: type[Any], **kwargs: Any) -> Any:
        del prompt, schema, kwargs
        raise RuntimeError("structured generation is not supported")

    def generate_text(self, *, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        return self._output


def _structured_answer_json(*, fenced: bool) -> str:
    payload = {
        "answer_text": "Alpha Engine handles ingestion requests. [Doc-1]",
        "answer_sections": [
            {
                "title": "Direct",
                "text": "Alpha Engine handles ingestion requests. [Doc-1]",
                "evidence_ids": ["E1"],
            }
        ],
        "insufficient_evidence_flag": False,
    }
    raw = json.dumps(payload)
    return f"```json\n{raw}\n```" if fenced else raw


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


@pytest.mark.parametrize("fenced", [False, True])
def test_answer_generator_parses_json_text_fallback_after_structured_failure(fenced: bool) -> None:
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
    generator = AnswerGenerator(
        generators=(
            GeneratorBinding(
                backend=_TextOnlyJsonGenerator(_structured_answer_json(fenced=fenced)),
                provider_name="local-openai-compatible",
                model_name="qwen3-8b",
                location="local",
            ),
        )
    )

    result = asyncio.run(
        generator.generate(
            query="What does Alpha Engine handle?",
            prompt="Return JSON",
            evidence_pack=evidence,
            grounded_candidate="Alpha Engine handles ingestion requests.",
            runtime_mode=RuntimeMode.FAST,
            access_policy=AccessPolicy.default(),
        )
    )

    assert result.attempts[0].status == "failed"
    assert "structured_generation_failed" in (result.attempts[0].error or "")
    assert result.answer.answer_text == "Alpha Engine handles ingestion requests. [1:7]"
    assert not result.answer.answer_text.lstrip().startswith("{")
    assert result.answer.answer_sections[0].title == "Direct"
    assert result.answer.answer_sections[0].text == "Alpha Engine handles ingestion requests. [1:7]"
    assert result.answer.answer_sections[0].evidence_ids == ["E1"]
    assert result.answer.groundedness_flag is True
    assert result.answer.evidence_links[0].answer_excerpt == "Alpha Engine handles ingestion requests. [1:7]"
