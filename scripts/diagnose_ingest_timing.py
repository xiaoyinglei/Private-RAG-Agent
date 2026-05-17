from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rag.ingest.parsers.excel_parser_repo import ExcelParserRepo
from rag.providers.mlx.embedder import MLXEmbedder
from rag.schema.core import SourceType

DEFAULT_EMBEDDING_MODEL = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure Excel parse and embedding timing for ingest diagnosis.")
    parser.add_argument("excel_path", type=Path, help="Path to the Excel file to parse.")
    parser.add_argument("--title", default="ingest timing sample")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    return parser


def _collect_texts(parsed: object) -> list[str]:
    sections = getattr(parsed, "sections", ())
    elements = getattr(parsed, "elements", ())
    title = getattr(parsed, "title", None)
    texts: list[str] = []
    for section in sections:
        texts.append(getattr(section, "text", None) or getattr(section, "anchor_hint", None) or "no text")
    for element in elements:
        texts.append(getattr(element, "text", None) or "no text")
    if title:
        texts.append(str(title))
    return texts


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    excel_path = args.excel_path.expanduser().resolve()
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file does not exist: {excel_path}")

    parser = ExcelParserRepo()
    parse_start = time.perf_counter()
    parsed = parser.parse(
        excel_path,
        location=str(excel_path),
        source_type=SourceType.XLSX,
        title=args.title,
    )
    parse_seconds = time.perf_counter() - parse_start

    load_start = time.perf_counter()
    embedder = MLXEmbedder(args.embedding_model)
    load_seconds = time.perf_counter() - load_start

    texts = _collect_texts(parsed)
    embed_start = time.perf_counter()
    vectors = embedder.embed(texts)
    embed_seconds = time.perf_counter() - embed_start

    payload = {
        "excel_path": str(excel_path),
        "embedding_model": args.embedding_model,
        "section_count": len(getattr(parsed, "sections", ())),
        "element_count": len(getattr(parsed, "elements", ())),
        "embedded_text_count": len(texts),
        "embedding_dimension": len(vectors[0]) if vectors else None,
        "timing_seconds": {
            "excel_parse": round(parse_seconds, 3),
            "embedding_model_load": round(load_seconds, 3),
            "embedding_compute": round(embed_seconds, 3),
            "total": round(parse_seconds + load_seconds + embed_seconds, 3),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
