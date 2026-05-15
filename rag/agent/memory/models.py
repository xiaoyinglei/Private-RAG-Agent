from __future__ import annotations

from typing import Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field

ContextSectionName = Literal[
    "system",
    "policy_hints",
    "task",
    "evidence",
    "working_memory",
    "historical_hints",
    "message_tail",
    "tool_results",
    "open_decisions",
]


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


class ContextSection(BaseModel):
    name: ContextSectionName
    content: str
    token_count: int = Field(ge=0)
    required: bool = False


class InjectedContext(BaseModel):
    sections: list[ContextSection]
    context_budget: ContextBudgetSnapshot

    def section(self, name: ContextSectionName) -> ContextSection:
        for section in self.sections:
            if section.name == name:
                return section
        raise KeyError(f"context section not found: {name}")

    def as_text(self) -> str:
        return "\n\n".join(
            f"[{section.name}]\n{section.content}" for section in self.sections
        )


class WorkingMemoryDehydration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    working_summary: WorkingSummary | None
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    tail_messages: list[BaseMessage] = Field(default_factory=list)
    context_budget: ContextBudgetSnapshot | None = None
