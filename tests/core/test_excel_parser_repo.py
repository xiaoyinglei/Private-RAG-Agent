from __future__ import annotations

import pandas as pd

from rag.ingest.asset_anchors import asset_anchor
from rag.ingest.parsers.excel_parser_repo import ExcelParserRepo
from rag.schema.core import SourceType


def test_excel_parser_schema_samples_large_sheet_without_full_markdown(tmp_path) -> None:
    source_path = tmp_path / "ledger.xlsx"
    rows = [{"Name": f"row{index}", "Department": f"dept{index % 3}", "Amount": index * 100} for index in range(800)]
    pd.DataFrame(rows).to_excel(source_path, sheet_name="Ledger", index=False)

    parsed = ExcelParserRepo().parse(
        source_path,
        location=str(source_path),
        source_type=SourceType.XLSX,
        title="Ledger",
        owner="tester",
    )

    assert len(parsed.sections) == 1
    assert len(parsed.elements) == 1
    element = parsed.elements[0]
    assert element.kind == "table"
    assert element.metadata["sheet_name"] == "Ledger"
    assert element.metadata["row_count"] == 800
    assert element.metadata["column_count"] == 3
    assert element.metadata["table_policy"] == "compute_only"
    assert element.metadata["sample_rows"][0] == {
        "Name": "row0",
        "Department": "dept0",
        "Amount": "0",
    }
    assert element.metadata["schema"][2]["name"] == "Amount"
    assert element.metadata["schema"][2]["type"].startswith("number")
    assert "Table columns: Name | Department | Amount" in element.text
    assert "row0" in element.text
    assert "row799" not in element.text
    assert asset_anchor(element.element_id) in parsed.sections[0].text
