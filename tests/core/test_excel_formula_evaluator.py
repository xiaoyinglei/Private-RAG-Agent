from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rag.ingest.excel_formula_evaluator import ExcelFormulaEvaluator


@dataclass
class _IterableOnlyWorksheet:
    rows: list[list[object]]

    @property
    def max_row(self) -> int:
        return len(self.rows)

    @property
    def max_column(self) -> int:
        return max((len(row) for row in self.rows), default=0)

    def iter_rows(
        self,
        *,
        min_row: int = 1,
        max_row: int | None = None,
        max_col: int | None = None,
        values_only: bool = False,
    ) -> list[tuple[object, ...]]:
        assert values_only is True
        row_limit = max_row or self.max_row
        col_limit = max_col or self.max_column
        result: list[tuple[object, ...]] = []
        for row in self.rows[min_row - 1 : row_limit]:
            result.append(tuple(row[index] if index < len(row) else None for index in range(col_limit)))
        return result

    def cell(self, *, row: int, column: int) -> Any:
        raise AssertionError(f"unexpected random cell access at row={row}, column={column}")


class _Workbook:
    def __init__(self, sheets: dict[str, _IterableOnlyWorksheet]) -> None:
        self._sheets = sheets

    def __getitem__(self, name: str) -> _IterableOnlyWorksheet:
        return self._sheets[name]


def test_formula_evaluator_uses_sheet_rows_instead_of_random_cell_access() -> None:
    value_book = _Workbook(
        {
            "Report": _IterableOnlyWorksheet(
                [
                    ["区域", "品类", "当日 提货", "查找值"],
                    ["北方", "石膏板", 19.2, None],
                    ["东北", "石膏板", 6.3, None],
                    ["合计", "石膏板", None, None],
                    ["东北查找", "石膏板", None, None],
                ]
            ),
            "Lookup": _IterableOnlyWorksheet(
                [
                    ["区域", "当日 提货"],
                    ["北方", 19.2],
                    ["东北", 6.3],
                ]
            ),
        }
    )
    formula_book = _Workbook(
        {
            "Report": _IterableOnlyWorksheet(
                [
                    ["区域", "品类", "当日 提货", "查找值"],
                    ["北方", "石膏板", 19.2, None],
                    ["东北", "石膏板", 6.3, None],
                    ["合计", "石膏板", "=SUM(C2:C3)", None],
                    ["东北查找", "石膏板", "=VLOOKUP(A3,Lookup!A:B,2,0)", None],
                ]
            ),
            "Lookup": _IterableOnlyWorksheet(
                [
                    ["区域", "当日 提货"],
                    ["北方", 19.2],
                    ["东北", 6.3],
                ]
            ),
        }
    )

    evaluator = ExcelFormulaEvaluator(value_workbook=value_book, formula_workbook=formula_book)

    assert evaluator.cell_value("Report", 4, 3) == pytest.approx(25.5)
    assert evaluator.cell_value("Report", 5, 3) == pytest.approx(6.3)
