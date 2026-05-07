from __future__ import annotations

from urllib.parse import quote, unquote

ASSET_ANCHOR_PREFIX = "[ASSET_ANCHOR:"
ASSET_ANCHOR_SUFFIX = "]"


def asset_anchor(element_ref: str) -> str:
    encoded = quote(element_ref, safe="-_.:")
    return f"{ASSET_ANCHOR_PREFIX}{encoded}{ASSET_ANCHOR_SUFFIX}"


def iter_asset_anchor_refs(text: str) -> list[str]:
    return [ref for _start, _end, ref in iter_asset_anchor_spans(text)]


def iter_asset_anchor_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    while True:
        start = text.find(ASSET_ANCHOR_PREFIX, cursor)
        if start < 0:
            return spans
        ref_start = start + len(ASSET_ANCHOR_PREFIX)
        end = text.find(ASSET_ANCHOR_SUFFIX, ref_start)
        if end < 0:
            return spans
        encoded_ref = text[ref_start:end].strip()
        if encoded_ref:
            spans.append((start, end + len(ASSET_ANCHOR_SUFFIX), unquote(encoded_ref)))
        cursor = end + len(ASSET_ANCHOR_SUFFIX)
