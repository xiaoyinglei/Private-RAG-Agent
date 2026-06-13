from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.asset_tools import (
    AssetAnalyzeInput,
    AssetInspectInput,
    AssetListInput,
    AssetReadSliceInput,
    AssetToolRunner,
)
from rag.agent.tools.rag_tool_runner import _evidence_to_output
from rag.schema.core import AssetRecord
from rag.schema.query import EvidenceItem, GroundingTarget


class _FakeObjectStore:
    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files

    def read_byte_range(self, key: str, start: int, end: int) -> bytes:
        return self._files.get(key, b"")

    def path_for_key(self, key: str) -> str | None:
        return None


class _FakeMetadataRepo:
    def __init__(self, assets: list[AssetRecord]) -> None:
        self._assets = {asset.asset_id: asset for asset in assets}

    def get_asset(self, asset_id: int) -> AssetRecord | None:
        return self._assets.get(asset_id)

    def list_assets(
        self,
        *,
        doc_id: int | None = None,
        source_id: int | None = None,
        section_id: int | None = None,
    ) -> list[AssetRecord]:
        assets = list(self._assets.values())
        if doc_id is not None:
            assets = [asset for asset in assets if asset.doc_id == doc_id]
        if source_id is not None:
            assets = [asset for asset in assets if asset.source_id == source_id]
        if section_id is not None:
            assets = [asset for asset in assets if asset.section_id == section_id]
        return assets


def _create_parquet(data: dict[str, list[object]]) -> bytes:
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)
    try:
        df.to_parquet(path)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def _asset(asset_id: int, *, storage_key: str = "daily.parquet") -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id,
        doc_id=7,
        source_id=3,
        section_id=5,
        asset_type="table",
        element_ref="table@p4",
        page_no=4,
        caption="日报",
        sheet_name="日报",
        row_count=3,
        column_count=2,
        sample_rows=[
            {"区域公司": "总计（不含一体化）", "日_日提货": "131.074462"},
            {"区域公司": "北方", "日_日提货": "19.22484"},
        ],
        table_schema=[
            {"name": "区域公司", "dtype": "object"},
            {"name": "日_日提货", "dtype": "float"},
        ],
        metadata_json={},
        content_hash="hash",
        storage_key=storage_key,
    )


def test_builtin_registry_exposes_generic_asset_tools_not_file_type_specific_tools() -> None:
    registry = create_builtin_tool_registry()
    tool_names = {spec.name for spec in registry.list_all()}

    assert {"asset_list", "asset_inspect", "asset_read_slice", "asset_analyze"} <= tool_names
    assert {"table_list", "table_inspect", "table_query_sql"}.isdisjoint(tool_names)
    assert {"asset_list", "asset_inspect", "asset_read_slice", "asset_analyze"} <= set(RESEARCH_AGENT.allowed_tools)


def test_asset_runner_lists_bounded_asset_metadata_and_capabilities() -> None:
    repo = _FakeMetadataRepo([_asset(14), _asset(15)])
    runner = AssetToolRunner(metadata_repo=repo, object_store=_FakeObjectStore({}))

    output = runner.list_assets(AssetListInput(doc_id=7, limit=1))

    assert len(output.assets) == 1
    asset = output.assets[0]
    assert asset.asset_id == 14
    assert asset.doc_id == 7
    assert asset.section_id == 5
    assert asset.page_no == 4
    assert asset.columns == ["区域公司", "日_日提货"]
    assert asset.analysis_capabilities == ["dataframe_preview", "dataframe_slice", "dataframe_sql"]
    assert output.truncated is True


def test_asset_runner_inspects_schema_head_tail_without_full_dump() -> None:
    parquet = _create_parquet(
        {
            "区域公司": ["总计（不含一体化）", "北方", "东北", "华东"],
            "日_日提货": ["131.074462", "19.22484", "6.307968", "29.137104"],
        }
    )
    repo = _FakeMetadataRepo([_asset(14)])
    runner = AssetToolRunner(
        metadata_repo=repo,
        object_store=_FakeObjectStore({"daily.parquet": parquet}),
    )

    output = runner.inspect_asset(AssetInspectInput(asset_id=14, head_rows=2, tail_rows=1))

    assert output.asset_id == 14
    assert output.asset_type == "table"
    assert output.sheet_name == "日报"
    assert output.analysis_capabilities == ["dataframe_preview", "dataframe_slice", "dataframe_sql"]
    assert output.columns == ["区域公司", "日_日提货"]
    assert output.row_count == 4
    assert output.head_rows == [
        {"区域公司": "总计（不含一体化）", "日_日提货": "131.074462"},
        {"区域公司": "北方", "日_日提货": "19.22484"},
    ]
    assert output.tail_rows == [{"区域公司": "华东", "日_日提货": "29.137104"}]


def test_asset_runner_executes_capability_based_read_only_analysis() -> None:
    parquet = _create_parquet(
        {
            "区域公司": ["总计（不含一体化）", "北方", "东北"],
            "日_日提货": ["131.074462", "19.22484", "6.307968"],
        }
    )
    repo = _FakeMetadataRepo([_asset(14)])
    runner = AssetToolRunner(
        metadata_repo=repo,
        object_store=_FakeObjectStore({"daily.parquet": parquet}),
    )

    output = runner.analyze_asset(
        AssetAnalyzeInput(
            asset_id=14,
            operation="dataframe_sql",
            query='SELECT SUM("日_日提货") AS total FROM sheet WHERE "区域公司" IN (\'北方\', \'东北\')',
        )
    )

    assert output.asset_id == 14
    assert output.operation == "dataframe_sql"
    assert output.columns == ["total"]
    assert output.rows == [["25.532808"]]
    assert output.raw_row_count == 1
    assert output.observation_only is False
    assert "TABLE_COMPUTE_RESULT:asset_id=14" in output.markdown


def test_asset_runner_marks_probe_analysis_as_observation_only() -> None:
    parquet = _create_parquet(
        {
            "区域公司": ["北方", "东北"],
            "日_日提货": ["19.22484", "6.307968"],
        }
    )
    runner = AssetToolRunner(
        metadata_repo=_FakeMetadataRepo([_asset(14)]),
        object_store=_FakeObjectStore({"daily.parquet": parquet}),
    )

    output = runner.analyze_asset(
        AssetAnalyzeInput(
            asset_id=14,
            operation="dataframe_sql",
            query='SELECT DISTINCT "区域公司" AS "__goal_probe__" FROM sheet',
        )
    )

    assert output.observation_only is True


def test_asset_runner_reads_bounded_table_slice_with_locator() -> None:
    parquet = _create_parquet(
        {
            "区域公司": ["总计（不含一体化）", "北方", "东北", "华东"],
            "日_日提货": ["131.074462", "19.22484", "6.307968", "29.137104"],
            "月_累计": ["1000", "200", "80", "300"],
        }
    )
    repo = _FakeMetadataRepo([_asset(14)])
    runner = AssetToolRunner(
        metadata_repo=repo,
        object_store=_FakeObjectStore({"daily.parquet": parquet}),
    )

    output = runner.read_slice(
        AssetReadSliceInput(
            asset_id=14,
            row_start=1,
            row_count=2,
            columns=["区域公司", "日_日提货"],
        )
    )

    assert output.asset_id == 14
    assert output.locator.asset_id == 14
    assert output.locator.doc_id == 7
    assert output.locator.sheet_name == "日报"
    assert output.locator.page_no == 4
    assert output.locator.row_start == 1
    assert output.locator.row_end_exclusive == 3
    assert output.locator.column_names == ["区域公司", "日_日提货"]
    assert output.columns == ["区域公司", "日_日提货"]
    assert output.rows == [
        {"区域公司": "北方", "日_日提货": "19.22484"},
        {"区域公司": "东北", "日_日提货": "6.307968"},
    ]
    assert output.truncated is False


def test_rag_search_output_preserves_asset_ids_for_followup_asset_tools() -> None:
    evidence = [
        EvidenceItem(
            evidence_id="e1",
            doc_id=7,
            source_id=3,
            citation_anchor="table@p4",
            text="[TABLE_ASSET asset_id=14] 日报",
            score=0.91,
            record_type="table",
            grounding_target=GroundingTarget(
                kind="asset",
                doc_id=7,
                source_id=3,
                section_id=5,
                asset_id=14,
                page_start=4,
                page_end=4,
            ),
        )
    ]

    output = _evidence_to_output(evidence)

    assert output.items[0]["asset_id"] == 14
    assert output.items[0]["section_id"] == 5
    assert output.items[0]["page_start"] == 4
