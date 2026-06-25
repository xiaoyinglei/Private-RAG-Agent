"""Semantic-level RAG tools — agent asks *what*, runner handles *how*.

Replaces 10 fine-grained pipeline tools (vector_search, keyword_search, grounding,
rerank, graph_expand, rag_search_answer, asset_list, asset_inspect, asset_read_slice,
asset_analyze) with 2 semantic tools that internally orchestrate the full pipeline.

Design principle: the agent should not care about retrieval pipeline ordering.
It says "search knowledge for X" or "find assets about Y" — the runner does the rest.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag.agent.tools.card import ToolCard
from rag.agent.tools.spec import ExecutionCategory, ToolError, ToolPermissions, ToolSpec


# ═══════════════════════════════════════════════════════════════════════════════
# search_knowledge — unified semantic knowledge retrieval
# ═══════════════════════════════════════════════════════════════════════════════


class KnowledgeSearchInput(BaseModel):
    """Input for search_knowledge — what to find in the knowledge base."""

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural language query. What do you want to know?",
    )
    constraints: str | None = Field(
        default=None,
        max_length=1000,
        description=(
            "Optional constraints: metadata filters, document types, date ranges, "
            "required fields. Expressed in natural language, e.g. "
            "'only 2024 annual reports' or 'pdf documents about compliance'."
        ),
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum number of results to return.",
    )


class KnowledgeResult(BaseModel):
    """A single knowledge result with evidence and citation."""

    evidence_id: str = ""
    doc_id: str = ""
    citation_anchor: str = ""
    text: str = ""
    score: float = 0.0
    source_type: str = ""
    file_name: str = ""


class KnowledgeGraphNeighbor(BaseModel):
    """A knowledge graph neighbor of a result entity."""

    entity_id: str = ""
    entity_label: str = ""
    relation: str = ""
    doc_id: str = ""


class KnowledgeSearchOutput(BaseModel):
    """Output from search_knowledge — evidence, answer, and graph neighbors."""

    results: list[KnowledgeResult] = Field(default_factory=list)
    answer_text: str = ""
    citations: list[str] = Field(default_factory=list)
    kg_neighbors: list[KnowledgeGraphNeighbor] = Field(default_factory=list)
    groundedness_flag: bool = False
    insufficient_evidence: bool = False
    total_found: int = 0


search_knowledge_spec = ToolSpec(
    name="search_knowledge",
    description=(
        "Search the knowledge base for evidence, facts, and insights. "
        "Returns ranked evidence items with citation anchors, an optional "
        "generated answer, and knowledge graph neighbors for related entities. "
        "Internally orchestrates vector search, keyword search, reranking, "
        "grounding, and graph expansion — you only need to say what to find."
    ),
    input_model=KnowledgeSearchInput,
    output_model=KnowledgeSearchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True, embed=True, generate=True),
    execution_category=ExecutionCategory.READ,
    timeout_seconds=45.0,
    max_retries=1,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=3000,
    aci=ToolCard(
        when_to_use=(
            "Use when you need to find information in the knowledge base: facts, "
            "document excerpts, evidence for claims, or answers to questions. "
            "This is the primary knowledge retrieval tool — use it instead of "
            "manually sequencing vector_search → rerank → grounding."
        ),
        when_not_to_use=(
            "Do not use for finding files in the workspace — use search_text. "
            "Do not use for finding structured assets (tables, charts) — use search_assets. "
            "Do not use when you already have the evidence and just need to generate text — use llm_generate."
        ),
        preconditions=("knowledge base must be indexed",),
        required_context=("clear search query",),
        input_examples=(
            {"query": "What was Q3 2025 revenue?", "top_k": 5},
            {"query": "machine learning applications in healthcare", "constraints": "peer-reviewed, since 2024", "top_k": 10},
        ),
        output_examples=(
            "results=[{evidence_id:ev-1,score:0.95,text:'Q3 revenue was $5.2B...'}] citations=[{citation_id:c-1,anchor:[1]}]",
        ),
        output_cap_policy="truncate",
        failure_codes=("timeout", "index_unavailable", "insufficient_evidence"),
        retryable=True,
        user_recoverable=True,
        model_next_action=(
            "If insufficient_evidence: try a broader query, remove constraints, or use search_assets. "
            "If timeout: reduce top_k."
        ),
        selection_tags=("search", "retrieval", "knowledge", "evidence"),
        file_types=(),
        domains=("knowledge_base", "documents", "research"),
        activation_group="rag",
    ),
)

# ═══════════════════════════════════════════════════════════════════════════════
# search_assets — unified asset discovery and preview
# ═══════════════════════════════════════════════════════════════════════════════


class AssetSearchInput(BaseModel):
    """Input for search_assets — find structured assets in the knowledge base."""

    query: str = Field(
        min_length=1,
        max_length=1000,
        description="What kind of assets are you looking for? e.g. 'financial tables', 'charts', 'spreadsheets'.",
    )
    asset_type: str | None = Field(
        default=None,
        description="Optional filter: 'table', 'chart', 'image', 'spreadsheet', etc.",
    )
    doc_id: int | None = Field(
        default=None,
        description="Optional: limit search to a specific document.",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Maximum number of assets to return.",
    )
    include_preview: bool = Field(
        default=True,
        description="Include data preview (head rows) for each asset. Set False for faster results.",
    )


class AssetResult(BaseModel):
    """A single asset result with metadata and optional preview."""

    asset_id: int
    doc_id: int
    asset_type: str = ""
    sheet_name: str | None = None
    caption: str | None = None
    columns: list[str] = Field(default_factory=list)
    row_count: int | None = None
    column_count: int | None = None
    preview_rows: list[dict[str, str]] = Field(default_factory=list)
    analysis_capabilities: list[str] = Field(default_factory=list)


class AssetSearchOutput(BaseModel):
    """Output from search_assets — asset list with previews."""

    assets: list[AssetResult] = Field(default_factory=list)
    total_found: int = 0
    truncated: bool = False


search_assets_spec = ToolSpec(
    name="search_assets",
    description=(
        "Search for structured assets (tables, charts, spreadsheets) in the "
        "knowledge base. Returns asset metadata, columns, row counts, and "
        "data previews. Internally orchestrates asset listing, inspection, "
        "and slice reading — you only say what kind of assets to find."
    ),
    input_model=AssetSearchInput,
    output_model=AssetSearchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True, read_object_store=True),
    execution_category=ExecutionCategory.READ,
    timeout_seconds=30.0,
    max_retries=1,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=1500,
    aci=ToolCard(
        when_to_use=(
            "Use when you need to find tables, charts, spreadsheets, or other "
            "structured data in the knowledge base. Returns metadata AND data "
            "previews so you can decide which asset to dive deeper into. "
            "Use this instead of manually calling asset_list → asset_inspect → asset_read_slice."
        ),
        when_not_to_use=(
            "Do not use for textual knowledge search — use search_knowledge. "
            "Do not use for raw file access in the workspace — use read_file."
        ),
        preconditions=("knowledge base must have indexed assets",),
        required_context=("asset type or domain context",),
        input_examples=(
            {"query": "Q3 financial tables", "asset_type": "table", "max_results": 5},
            {"query": "compliance charts", "doc_id": 42},
        ),
        output_examples=(
            "assets=[{asset_id:1,asset_type:table,columns:[Date,Revenue,Profit],row_count:120,preview_rows:[...]}]",
        ),
        output_cap_policy="truncate",
        failure_codes=("timeout", "no_assets_found"),
        retryable=True,
        user_recoverable=True,
        model_next_action="Broaden the query or remove asset_type filter.",
        selection_tags=("asset", "table", "chart", "structured_data"),
        file_types=(),
        domains=("knowledge_base", "structured_data"),
        activation_group="rag",
    ),
)

# ═══════════════════════════════════════════════════════════════════════════════
# Module aggregates
# ═══════════════════════════════════════════════════════════════════════════════

ALL_RAG_SEMANTIC_TOOLS: list[ToolSpec] = [
    search_knowledge_spec,
    search_assets_spec,
]
