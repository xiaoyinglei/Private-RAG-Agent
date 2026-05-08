from __future__ import annotations

from langgraph.graph.message import BaseMessage
from pydantic import BaseModel, ConfigDict, Field


class WorkingSummary(BaseModel):
    summary: str
    covered_message_ids: list[str]
    updated_at: str
    token_count: int


class ExtractedFact(BaseModel):
    fact_id: str
    text: str
    source_message_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    stale: bool = False


class ContextBudgetSnapshot(BaseModel):
    max_context_tokens: int
    system_tokens: int = 0
    evidence_tokens: int = 0
    working_memory_tokens: int = 0
    recalled_memory_tokens: int = 0
    message_tail_tokens: int = 0
    tool_result_tokens: int = 0


class WorkingMemoryDehydration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    working_summary: WorkingSummary | None
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    tail_messages: list[BaseMessage] = Field(default_factory=list)
    context_budget: ContextBudgetSnapshot | None = None
