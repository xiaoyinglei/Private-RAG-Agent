from __future__ import annotations

from pydantic import BaseModel, Field

from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec


class LLMGenerateInput(BaseModel):
    prompt: str
    context_sections: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)


class LLMSummarizeInput(BaseModel):
    task: str
    context_sections: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)


class LLMCompareInput(BaseModel):
    question: str
    left_context_sections: list[str] = Field(default_factory=list)
    right_context_sections: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)


class LLMTextOutput(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False


llm_generate = ToolSpec(
    name="llm_generate",
    description="Generate a grounded response from provided context. Does not retrieve evidence.",
    input_model=LLMGenerateInput,
    output_model=LLMTextOutput,
    error_model=ToolError,
    permissions=ToolPermissions(generate=True),
    timeout_seconds=30.0,
    max_retries=1,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=200,
)

llm_summarize = ToolSpec(
    name="llm_summarize",
    description="Summarize provided evidence and citations for a focused research task.",
    input_model=LLMSummarizeInput,
    output_model=LLMTextOutput,
    error_model=ToolError,
    permissions=ToolPermissions(generate=True),
    timeout_seconds=30.0,
    max_retries=1,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=200,
)

llm_compare = ToolSpec(
    name="llm_compare",
    description="Compare two sets of provided evidence. Does not retrieve evidence.",
    input_model=LLMCompareInput,
    output_model=LLMTextOutput,
    error_model=ToolError,
    permissions=ToolPermissions(generate=True),
    timeout_seconds=30.0,
    max_retries=1,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=250,
)

ALL_LLM_TOOLS = [llm_generate, llm_summarize, llm_compare]
