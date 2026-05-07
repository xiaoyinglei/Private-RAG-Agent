from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag.storage import StorageConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export SectionRecord text from a local private index as JSONL.")
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--output", default="data/company_policy_sections.jsonl")
    return parser


def _section_text(object_store: Any, section: Any) -> str:
    key = section.visible_text_key
    start = int(section.byte_range_start)
    end = int(section.byte_range_end)
    return object_store.read_byte_range(key, start, end).decode("utf-8")


def _metadata(section: object) -> dict[str, Any]:
    value = getattr(section, "metadata_json", {})
    return dict(value) if isinstance(value, dict) else {}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stores = StorageConfig(root=Path(args.storage_root)).build()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        documents = stores.metadata_repo.list_documents(active_only=False)
        count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for document in documents:
                sections = stores.metadata_repo.list_sections(doc_id=document.doc_id)
                for section in sections:
                    metadata = _metadata(section)
                    payload = {
                        "doc_id": section.doc_id,
                        "source_id": section.source_id,
                        "title": document.title,
                        "section_id": section.section_id,
                        "parent_section_id": section.parent_section_id,
                        "order_index": section.order_index,
                        "toc_path": list(section.toc_path),
                        "char_range_start": section.char_range_start,
                        "char_range_end": section.char_range_end,
                        "window_index": metadata.get("window_index"),
                        "window_count": metadata.get("window_count"),
                        "metadata": metadata,
                        "text": _section_text(stores.object_store, section),
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    count += 1
        print(
            json.dumps(
                {
                    "storage_root": str(Path(args.storage_root)),
                    "output": str(output_path),
                    "document_count": len(documents),
                    "section_count": count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        stores.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
