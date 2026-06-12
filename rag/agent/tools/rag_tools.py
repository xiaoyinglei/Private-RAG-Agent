from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.query import RetrievalSignals


class SearchInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    query: str
    top_k: int = 8
    retrieval_signals: RetrievalSignals | None = Field(default=None)


RAG_SIGNAL_AWARE_TOOLS = frozenset({
    "vector_search", "keyword_search", "grounding", "rerank", "graph_expand",
})


class SearchOutput(BaseModel):
    items: list[dict[str, object]]


vector_search = ToolSpec(
    name="vector_search",
    description="Semantic vector search across document summaries. Use for natural language queries.",
    input_model=SearchInput,
    output_model=SearchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True, embed=True),
    timeout_seconds=10.0,
    max_retries=1,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=500,
)

keyword_search = ToolSpec(
    name="keyword_search",
    description="Lexical/keyword search. Use for exact terms, document IDs, codes, dates.",
    input_model=SearchInput,
    output_model=SearchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True),
    timeout_seconds=5.0,
    max_retries=1,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=200,
)

grounding = ToolSpec(
    name="grounding",
    description="Read original document text at a precise location. Use to verify retrieved evidence.",
    input_model=SearchInput,
    output_model=SearchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True, read_object_store=True),
    timeout_seconds=15.0,
    max_retries=2,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=1000,
)

rerank = ToolSpec(
    name="rerank",
    description="Re-rank candidate evidence by relevance to the query. Use when evidence ordering matters.",
    input_model=SearchInput,
    output_model=SearchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True, embed=True, generate=True),
    timeout_seconds=10.0,
    max_retries=1,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=800,
)

graph_expand = ToolSpec(
    name="graph_expand",
    description="Expand retrieval via knowledge graph neighbors. Use for multi-hop or relational queries.",
    input_model=SearchInput,
    output_model=SearchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True),
    timeout_seconds=5.0,
    max_retries=1,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=300,
)

ALL_RAG_TOOLS = [vector_search, keyword_search, grounding, rerank, graph_expand]
