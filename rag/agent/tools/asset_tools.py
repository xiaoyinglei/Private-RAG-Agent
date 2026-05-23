from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.ingest.table_executor import TableExecutor
from rag.schema.core import AssetRecord


class _ObjectStore(Protocol):
    def read_byte_range(self, key: str, start: int, end: int) -> bytes: ...

    def path_for_key(self, key: str) -> str | None: ...


class _MetadataRepo(Protocol):
    def get_asset(self, asset_id: int) -> AssetRecord | None: ...

    def list_assets(
        self,
        *,
        doc_id: int | None = None,
        source_id: int | None = None,
        section_id: int | None = None,
    ) -> list[AssetRecord]: ...


MAX_ASSET_LIST_LIMIT = 50
MAX_ASSET_PREVIEW_ROWS = 20

AssetAnalysisOperation = Literal["dataframe_sql"]


class AssetListInput(BaseModel):
    doc_id: int | None = None
    source_id: int | None = None
    section_id: int | None = None
    asset_type: str | None = None
    limit: int = Field(default=20, ge=1, le=MAX_ASSET_LIST_LIMIT)


class AssetDescriptor(BaseModel):
    asset_id: int
    doc_id: int
    source_id: int | None = None
    section_id: int | None = None
    asset_type: str
    page_no: int | None = None
    element_ref: str | None = None
    sheet_name: str | None = None
    caption: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    columns: list[str] = Field(default_factory=list)
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    analysis_capabilities: list[str] = Field(default_factory=list)


class AssetListOutput(BaseModel):
    assets: list[AssetDescriptor]
    truncated: bool = False


class AssetInspectInput(BaseModel):
    asset_id: int
    head_rows: int = Field(default=8, ge=0, le=MAX_ASSET_PREVIEW_ROWS)
    tail_rows: int = Field(default=3, ge=0, le=MAX_ASSET_PREVIEW_ROWS)


class AssetInspectOutput(BaseModel):
    asset_id: int
    doc_id: int
    source_id: int | None = None
    section_id: int | None = None
    asset_type: str
    page_no: int | None = None
    element_ref: str | None = None
    caption: str | None = None
    analysis_capabilities: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    row_count: int | None = None
    column_count: int | None = None
    head_rows: list[dict[str, str]] = Field(default_factory=list)
    tail_rows: list[dict[str, str]] = Field(default_factory=list)


class AssetAnalyzeInput(BaseModel):
    asset_id: int
    operation: AssetAnalysisOperation
    query: str = Field(min_length=1, max_length=4000)


class AssetAnalyzeOutput(BaseModel):
    asset_id: int
    operation: AssetAnalysisOperation
    columns: list[str]
    rows: list[list[str]]
    raw_row_count: int
    elapsed_ms: float
    truncated: bool
    query: str
    markdown: str


class AssetToolRunner:
    """Generic indexed-asset access surface for agent analysis.

    The exposed tool API is asset-oriented. Format-specific readers live behind
    capability strings so the agent does not need one public tool per file type.
    """

    def __init__(self, *, metadata_repo: _MetadataRepo, object_store: _ObjectStore) -> None:
        self._metadata_repo = metadata_repo
        self._table_executor = TableExecutor(object_store=object_store, metadata_repo=metadata_repo)

    def list_assets(self, payload: AssetListInput) -> AssetListOutput:
        assets = self._metadata_repo.list_assets(
            doc_id=payload.doc_id,
            source_id=payload.source_id,
            section_id=payload.section_id,
        )
        if payload.asset_type:
            wanted = payload.asset_type.strip().lower()
            assets = [asset for asset in assets if str(asset.asset_type).lower() == wanted]
        limited = assets[: payload.limit]
        return AssetListOutput(
            assets=[_descriptor_from_asset(asset) for asset in limited],
            truncated=len(assets) > payload.limit,
        )

    def inspect_asset(self, payload: AssetInspectInput) -> AssetInspectOutput:
        asset = self._get_asset(payload.asset_id)
        capabilities = _analysis_capabilities(asset)
        output = AssetInspectOutput(
            asset_id=asset.asset_id,
            doc_id=asset.doc_id,
            source_id=asset.source_id,
            section_id=asset.section_id,
            asset_type=asset.asset_type,
            page_no=asset.page_no,
            element_ref=asset.element_ref,
            caption=asset.caption,
            analysis_capabilities=capabilities,
            columns=_column_names(asset),
            row_count=asset.row_count,
            column_count=asset.column_count,
        )

        if "dataframe_preview" not in capabilities:
            return output

        result = self._table_executor.inspect(
            asset_id=payload.asset_id,
            head_rows=payload.head_rows,
            tail_rows=payload.tail_rows,
        )
        if result is None:
            return output
        return output.model_copy(
            update={
                "columns": result.columns,
                "row_count": result.row_count,
                "column_count": result.column_count,
                "head_rows": result.head_rows,
                "tail_rows": result.tail_rows,
            }
        )

    def analyze_asset(self, payload: AssetAnalyzeInput) -> AssetAnalyzeOutput:
        asset = self._get_asset(payload.asset_id)
        capabilities = _analysis_capabilities(asset)
        if payload.operation not in capabilities:
            raise ValueError(
                f"asset_id={payload.asset_id} does not support analysis operation {payload.operation!r}"
            )

        if payload.operation == "dataframe_sql":
            result = self._table_executor.execute(asset_id=payload.asset_id, sql=payload.query)
            if result is None:
                raise RuntimeError(
                    "asset analysis failed or query was rejected; dataframe_sql only supports bounded SELECT"
                )
            return AssetAnalyzeOutput(
                asset_id=result.asset_id,
                operation=payload.operation,
                columns=result.columns,
                rows=result.rows,
                raw_row_count=result.raw_row_count,
                elapsed_ms=result.elapsed_ms,
                truncated=result.truncated,
                query=result.sql or payload.query.strip(),
                markdown=result.markdown,
            )
        raise ValueError(f"unsupported asset analysis operation: {payload.operation!r}")

    def _get_asset(self, asset_id: int) -> AssetRecord:
        asset = self._metadata_repo.get_asset(asset_id)
        if asset is None:
            raise ValueError(f"asset not found: asset_id={asset_id}")
        return asset


def _descriptor_from_asset(asset: AssetRecord) -> AssetDescriptor:
    return AssetDescriptor(
        asset_id=asset.asset_id,
        doc_id=asset.doc_id,
        source_id=asset.source_id,
        section_id=asset.section_id,
        asset_type=asset.asset_type,
        page_no=asset.page_no,
        element_ref=asset.element_ref,
        sheet_name=asset.sheet_name,
        caption=asset.caption,
        row_count=asset.row_count,
        column_count=asset.column_count,
        columns=_column_names(asset),
        sample_rows=list(asset.sample_rows[:3]),
        analysis_capabilities=_analysis_capabilities(asset),
    )


def _column_names(asset: AssetRecord) -> list[str]:
    names: list[str] = []
    for column in asset.table_schema:
        if not isinstance(column, dict):
            continue
        name = column.get("name") or column.get("column_name")
        if name is not None:
            names.append(str(name))
    return names


def _analysis_capabilities(asset: AssetRecord) -> list[str]:
    if _is_dataframe_asset(asset):
        return ["dataframe_preview", "dataframe_sql"]
    return []


def _is_dataframe_asset(asset: AssetRecord) -> bool:
    if str(asset.asset_type).lower() == "table":
        return True
    suffix = str(asset.storage_key or "").lower()
    return suffix.endswith((".xlsx", ".xls", ".parquet", ".csv"))


asset_list = ToolSpec(
    name="asset_list",
    description=(
        "List indexed source assets by document/source/section. Use after retrieval identifies a document "
        "or section and you need concrete asset ids plus supported analysis capabilities."
    ),
    input_model=AssetListInput,
    output_model=AssetListOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True),
    timeout_seconds=5.0,
    max_retries=1,
    token_budget_cost=300,
)

asset_inspect = ToolSpec(
    name="asset_inspect",
    description=(
        "Inspect one indexed asset through a bounded read-only preview. The output includes available "
        "analysis capabilities such as dataframe_preview/dataframe_sql when supported."
    ),
    input_model=AssetInspectInput,
    output_model=AssetInspectOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True, read_object_store=True),
    timeout_seconds=10.0,
    max_retries=1,
    token_budget_cost=900,
)

asset_analyze = ToolSpec(
    name="asset_analyze",
    description=(
        "Execute a bounded read-only analysis against an indexed asset using one of its advertised "
        "analysis capabilities. For dataframe_sql, query the asset as table name 'sheet' with SELECT."
    ),
    input_model=AssetAnalyzeInput,
    output_model=AssetAnalyzeOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True, read_object_store=True),
    timeout_seconds=10.0,
    max_retries=1,
    token_budget_cost=800,
)


ALL_ASSET_TOOLS = [asset_list, asset_inspect, asset_analyze]


__all__ = [
    "ALL_ASSET_TOOLS",
    "AssetAnalyzeInput",
    "AssetAnalyzeOutput",
    "AssetDescriptor",
    "AssetInspectInput",
    "AssetInspectOutput",
    "AssetListInput",
    "AssetListOutput",
    "AssetToolRunner",
    "asset_analyze",
    "asset_inspect",
    "asset_list",
]
