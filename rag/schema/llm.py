from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class LLMCallStage(StrEnum):
    GOAL_CONTRACT = "goal_contract"
    RETRIEVAL_HINT = "retrieval_hint"
    TOOL_DECISION = "tool_decision"
    RETRIEVAL_SUMMARY = "retrieval_summary"
    LLM_SUMMARIZE = "llm_summarize"
    LLM_COMPARE = "llm_compare"
    LLM_GENERATE = "llm_generate"
    RAG_ANSWER = "rag_answer"
    FINAL_SYNTHESIS = "final_synthesis"


class LLMStageBudget(BaseModel):
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    safety_margin_tokens: int = Field(default=512, ge=0)


class LLMUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    source: Literal["provider", "tokenizer_estimate"]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class LLMProviderResult[T]:
    value: T
    usage: LLMUsage | None = None


@dataclass(frozen=True, slots=True)
class LLMCallResult[T]:
    value: T
    usage: LLMUsage
    stage: LLMCallStage


DEFAULT_LLM_STAGE_BUDGETS: dict[LLMCallStage, LLMStageBudget] = {
    LLMCallStage.GOAL_CONTRACT: LLMStageBudget(
        max_input_tokens=3_000,
        max_output_tokens=512,
        safety_margin_tokens=256,
    ),
    LLMCallStage.RETRIEVAL_HINT: LLMStageBudget(
        max_input_tokens=2_000,
        max_output_tokens=256,
    ),
    LLMCallStage.TOOL_DECISION: LLMStageBudget(
        max_input_tokens=12_000,
        max_output_tokens=2_048,
    ),
    LLMCallStage.RETRIEVAL_SUMMARY: LLMStageBudget(
        max_input_tokens=6_000,
        max_output_tokens=1_024,
    ),
    LLMCallStage.LLM_SUMMARIZE: LLMStageBudget(
        max_input_tokens=16_000,
        max_output_tokens=2_048,
    ),
    LLMCallStage.LLM_COMPARE: LLMStageBudget(
        max_input_tokens=20_000,
        max_output_tokens=3_072,
    ),
    LLMCallStage.LLM_GENERATE: LLMStageBudget(
        max_input_tokens=16_000,
        max_output_tokens=3_072,
    ),
    LLMCallStage.RAG_ANSWER: LLMStageBudget(
        max_input_tokens=16_000,
        max_output_tokens=4_096,
    ),
    LLMCallStage.FINAL_SYNTHESIS: LLMStageBudget(
        max_input_tokens=24_000,
        max_output_tokens=4_096,
    ),
}


def parse_llm_stage_budgets(
    raw: object,
) -> dict[LLMCallStage, LLMStageBudget]:
    parsed = {
        stage: budget.model_copy()
        for stage, budget in DEFAULT_LLM_STAGE_BUDGETS.items()
    }
    if not isinstance(raw, Mapping):
        return parsed
    for raw_stage, raw_budget in raw.items():
        try:
            stage = LLMCallStage(str(raw_stage))
        except ValueError:
            continue
        parsed[stage] = LLMStageBudget.model_validate(raw_budget)
    return parsed


__all__ = [
    "DEFAULT_LLM_STAGE_BUDGETS",
    "LLMCallResult",
    "LLMCallStage",
    "LLMProviderResult",
    "LLMStageBudget",
    "LLMUsage",
    "parse_llm_stage_budgets",
]
