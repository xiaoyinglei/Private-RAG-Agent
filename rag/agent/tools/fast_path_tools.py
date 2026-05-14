from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.query import AnswerCitation, EvidenceItem, RetrievalSignals


class RAGSearchAnswerInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    top_k: int = Field(default=8, gt=0)
    retrieval_signals: RetrievalSignals | None = Field(default=None)


class RAGSearchAnswerOutput(BaseModel):
    text: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    groundedness_flag: bool = False
    insufficient_evidence: bool = False


rag_search_answer = ToolSpec(
    name="rag_search_answer",
    description="Run the fast RAG query-and-answer path and return grounded answer text.",
    input_model=RAGSearchAnswerInput,
    output_model=RAGSearchAnswerOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True, embed=True, generate=True),
    timeout_seconds=45.0,
    max_retries=0,
    token_budget_cost=3000,
)


ALL_FAST_PATH_TOOLS = [rag_search_answer]


@dataclass
class RAGSearchAnswerRunner:
    runtime: Any
    max_context_tokens: int = 12000

    async def answer(self, payload: RAGSearchAnswerInput) -> RAGSearchAnswerOutput:
        from rag.retrieval import QueryOptions

        options_kwargs: dict[str, object] = {
            "retrieval_profile": "fast",
            "top_k": payload.top_k,
            "max_context_tokens": self.max_context_tokens,
        }
        if access_policy := getattr(self.runtime, "access_policy", None):
            options_kwargs["access_policy"] = access_policy
        result = await asyncio.to_thread(
            self.runtime.query_public,
            payload.query,
            options=QueryOptions(**options_kwargs),
        )
        answer = result.answer
        context = getattr(result, "context", None)
        return RAGSearchAnswerOutput(
            text=answer.answer_text,
            evidence=_context_evidence(context),
            citations=list(getattr(answer, "citations", [])),
            groundedness_flag=bool(getattr(answer, "groundedness_flag", False)),
            insufficient_evidence=bool(
                getattr(answer, "insufficient_evidence_flag", False)
            ),
        )


def _context_evidence(context: object | None) -> list[EvidenceItem]:
    evidence = list(getattr(context, "evidence", []) or [])
    items: list[EvidenceItem] = []
    for item in evidence:
        if hasattr(item, "as_evidence_item"):
            item = item.as_evidence_item()
        items.append(EvidenceItem.model_validate(item))
    return items
