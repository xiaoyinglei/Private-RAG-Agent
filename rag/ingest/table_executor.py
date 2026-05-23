from __future__ import annotations

import math
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from rag.ingest.table_semantics import deduplicate_table_columns


class _RangeReadableObjectStore(Protocol):
    def read_byte_range(self, key: str, start: int, end: int) -> bytes: ...

    def path_for_key(self, key: str) -> str | None: ...


class _AssetMetadataRepo(Protocol):
    def get_asset(self, asset_id: int) -> Any: ...


MAX_RESULT_ROWS = 100
MAX_SQL_TIMEOUT_SECONDS = 5.0
_TRAILING_SPARSE_TRIM_MIN_COLUMNS = 4
_TRAILING_SPARSE_ROW_MIN_FILL_RATIO = 0.6

_FORBIDDEN_SQL_PATTERNS = re.compile(
    r"\b(DROP|CREATE|INSERT|UPDATE|DELETE|ALTER|COPY|PRAGMA|ATTACH|DETACH|VACUUM|EXPORT|IMPORT)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ComputeResult:
    asset_id: int
    columns: list[str]
    rows: list[list[str]]
    raw_row_count: int
    elapsed_ms: float
    truncated: bool
    sql: str | None = None

    @property
    def markdown(self) -> str:
        parts: list[str] = []
        parts.append(f"[TABLE_COMPUTE_RESULT:asset_id={self.asset_id}]")
        parts.append(f"Computation executed in {self.elapsed_ms:.0f}ms. Returned {self.raw_row_count} rows.")
        if self.sql:
            parts.append(f"Executed SQL: {self.sql}")
        if self.truncated:
            parts.append(f"(Showing first {len(self.rows)} rows of {self.raw_row_count} total)")
        parts.append("")

        if self.columns and self.rows:
            parts.append("| " + " | ".join(self.columns) + " |")
            parts.append("|" + "|".join(["---"] * len(self.columns)) + "|")
            for row in self.rows:
                parts.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class TableInspectionResult:
    asset_id: int
    columns: list[str]
    row_count: int
    column_count: int
    head_rows: list[dict[str, str]]
    tail_rows: list[dict[str, str]]


class TableExecutor:
    def __init__(self, *, object_store: _RangeReadableObjectStore, metadata_repo: _AssetMetadataRepo) -> None:
        self._object_store = object_store
        self._metadata_repo = metadata_repo

    def execute(self, *, asset_id: int, sql: str) -> ComputeResult | None:
        if not self._validate_sql(sql):
            return None

        asset = self._metadata_repo.get_asset(asset_id)
        if asset is None:
            return None

        storage_key = str(getattr(asset, "storage_key", "") or "").strip()
        sheet_name = str(getattr(asset, "sheet_name", "") or "").strip() or None

        if not storage_key:
            return None

        local_path, is_temp = self._download_to_temp(storage_key)
        if local_path is None:
            return None

        try:
            return self._execute_on_file(local_path, sheet_name=sheet_name, sql=sql, asset_id=asset_id)
        except Exception:
            return None
        finally:
            if is_temp:
                try:
                    Path(local_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def inspect(
        self,
        *,
        asset_id: int,
        head_rows: int = 8,
        tail_rows: int = 3,
    ) -> TableInspectionResult | None:
        asset = self._metadata_repo.get_asset(asset_id)
        if asset is None:
            return None

        storage_key = str(getattr(asset, "storage_key", "") or "").strip()
        sheet_name = str(getattr(asset, "sheet_name", "") or "").strip() or None
        if not storage_key:
            return None

        local_path, is_temp = self._download_to_temp(storage_key)
        if local_path is None:
            return None

        try:
            df = self._load_dataframe_from_file(
                local_path,
                sheet_name=sheet_name,
                coerce_numeric=False,
            )
            if df is None:
                return None
            return TableInspectionResult(
                asset_id=asset_id,
                columns=[str(column) for column in df.columns],
                row_count=len(df),
                column_count=len(df.columns),
                head_rows=self._rows_to_records(df.head(max(0, head_rows))),
                tail_rows=self._rows_to_records(df.tail(max(0, tail_rows))) if tail_rows > 0 else [],
            )
        except Exception:
            return None
        finally:
            if is_temp:
                try:
                    Path(local_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def _download_to_temp(self, storage_key: str) -> tuple[str | None, bool]:
        cached_path = self._object_store.path_for_key(storage_key)
        if cached_path is not None and Path(cached_path).exists():
            return cached_path, False

        try:
            raw = self._object_store.read_byte_range(storage_key, 0, 2**31 - 1)
        except Exception:
            raw = b""
        if not raw:
            return None, False

        suffix = Path(storage_key).suffix or ".xlsx"
        try:
            fd, path = tempfile.mkstemp(suffix=suffix)
            with open(fd, "wb") as f:
                f.write(raw)
            return path, True
        except Exception:
            return None, False

    def _execute_on_file(
        self, file_path: str, *, sheet_name: str | None, sql: str, asset_id: int
    ) -> ComputeResult | None:
        import duckdb

        con = duckdb.connect(":memory:")
        try:
            con.execute("SET enable_external_access=false")
            con.execute("PRAGMA threads=2")
            con.execute("SET memory_limit='1GB'")

            df = self._load_dataframe_from_file(
                file_path,
                sheet_name=sheet_name,
                coerce_numeric=True,
            )
            if df is None or df.empty:
                return ComputeResult(
                    asset_id=asset_id,
                    columns=[],
                    rows=[],
                    raw_row_count=0,
                    elapsed_ms=0.0,
                    truncated=False,
                )
            con.register("sheet", df)

            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(con.execute, sql)
                try:
                    result = future.result(timeout=MAX_SQL_TIMEOUT_SECONDS)
                except FuturesTimeoutError:
                    return None

            elapsed = (time.perf_counter() - start) * 1000.0
            columns = [str(col[0]) for col in (result.description or [])]
            all_rows = [[str(cell) for cell in row] for row in result.fetchall()]
            raw_row_count = len(all_rows)
            truncated = raw_row_count > MAX_RESULT_ROWS
            display_rows = all_rows[:MAX_RESULT_ROWS]

            return ComputeResult(
                asset_id=asset_id,
                columns=columns,
                rows=display_rows,
                raw_row_count=raw_row_count,
                elapsed_ms=elapsed,
                truncated=truncated,
                sql=sql.strip(),
            )
        finally:
            con.close()

    @staticmethod
    def _validate_sql(sql: str) -> bool:
        if not sql or not sql.strip():
            return False
        normalized = sql.strip()
        if not normalized.upper().startswith("SELECT"):
            return False
        if _FORBIDDEN_SQL_PATTERNS.search(normalized):
            return False
        return True

    @staticmethod
    def _load_dataframe_from_file(
        file_path: str,
        *,
        sheet_name: str | None,
        coerce_numeric: bool,
    ) -> pd.DataFrame | None:
        is_parquet = str(file_path).endswith(".parquet")
        if is_parquet:
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        if not isinstance(df, pd.DataFrame):
            return None
        df = TableExecutor._normalize_dataframe_columns(df)
        if coerce_numeric:
            df = TableExecutor._coerce_numeric_columns(df)
        return TableExecutor._trim_trailing_sparse_rows(df)

    @staticmethod
    def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        normalized.columns = deduplicate_table_columns([str(column) for column in normalized.columns])
        return normalized

    @staticmethod
    def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        converted = df.copy()
        for column in converted.columns:
            series = converted[column]
            if pd.api.types.is_numeric_dtype(series):
                continue
            non_empty = series.dropna()
            if non_empty.empty:
                continue
            normalized = non_empty.astype(str).str.strip()
            normalized = normalized[normalized != ""]
            if normalized.empty:
                continue
            normalized = normalized.str.replace(",", "", regex=False)
            normalized = normalized.str.replace("，", "", regex=False)
            normalized = normalized.str.rstrip("%").str.strip()
            numeric = pd.to_numeric(normalized, errors="coerce")
            if numeric.notna().sum() < max(1, int(len(normalized) * 0.7)):
                continue
            full_normalized = series.astype(str).str.strip()
            full_normalized = full_normalized.str.replace(",", "", regex=False)
            full_normalized = full_normalized.str.replace("，", "", regex=False)
            full_normalized = full_normalized.str.rstrip("%").str.strip()
            converted[column] = pd.to_numeric(full_normalized, errors="coerce")
        return converted

    @staticmethod
    def _trim_trailing_sparse_rows(df: pd.DataFrame) -> pd.DataFrame:
        column_count = len(df.columns)
        if df.empty or column_count < _TRAILING_SPARSE_TRIM_MIN_COLUMNS:
            return df

        filled_counts = [TableExecutor._row_filled_cell_count(row) for _, row in df.iterrows()]
        max_filled_cells = max(filled_counts, default=0)
        if max_filled_cells <= 2:
            return df

        min_filled_cells = max(
            2,
            math.ceil(max_filled_cells * _TRAILING_SPARSE_ROW_MIN_FILL_RATIO) + 1,
        )
        last_data_index = len(df) - 1
        while last_data_index >= 0:
            if filled_counts[last_data_index] >= min_filled_cells:
                break
            last_data_index -= 1

        if last_data_index == len(df) - 1:
            return df
        if last_data_index < 0:
            return df
        return df.iloc[: last_data_index + 1].reset_index(drop=True)

    @staticmethod
    def _row_filled_cell_count(row: pd.Series) -> int:
        count = 0
        for value in row.tolist():
            if pd.isna(value):
                continue
            if str(value).strip():
                count += 1
        return count

    @staticmethod
    def _rows_to_records(df: pd.DataFrame) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for row in df.to_dict(orient="records"):
            record: dict[str, str] = {}
            for key, value in row.items():
                if pd.isna(value):
                    record[str(key)] = ""
                else:
                    record[str(key)] = str(value)
            records.append(record)
        return records


__all__ = [
    "ComputeResult",
    "TableExecutor",
    "TableInspectionResult",
    "MAX_RESULT_ROWS",
    "MAX_SQL_TIMEOUT_SECONDS",
]
