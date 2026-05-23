from __future__ import annotations

import re


def normalize_table_column_name(name: object) -> str:
    normalized = "" if name is None else str(name)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def deduplicate_table_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for index, raw_name in enumerate(columns):
        name = normalize_table_column_name(raw_name) or f"column_{index + 1}"
        deduped = name
        suffix = 2
        while deduped in seen:
            deduped = f"{name}_{suffix}"
            suffix += 1
        seen.add(deduped)
        result.append(deduped)
    return result


__all__ = [
    "deduplicate_table_columns",
    "normalize_table_column_name",
]
