from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path

from rag.benchmarks import (
    FIQA_DATASET,
    MEDICAL_RETRIEVAL_DATASET,
    default_benchmark_paths,
    ensure_benchmark_layout,
    profile_benchmark_ingest,
)


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_str_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Profile benchmark ingest throughput on a small prepared subset.")
    parser.add_argument("--dataset", default=FIQA_DATASET, choices=[FIQA_DATASET, MEDICAL_RETRIEVAL_DATASET])
    parser.add_argument("--variant", default="full", choices=["full", "mini"])
    parser.add_argument("--documents-path", default=None)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--doc-counts", type=_parse_int_list, default=[100], help="comma-separated, e.g. 100,500,1000")
    parser.add_argument(
        "--ingest-batch-sizes",
        type=_parse_int_list,
        default=[32],
        help="comma-separated outer ingest batch sizes, e.g. 16,32,64",
    )
    parser.add_argument(
        "--encode-batch-sizes",
        type=_parse_int_list,
        default=[8, 16, 32, 64],
        help="comma-separated embedding encode batch sizes, e.g. 8,16,32,64",
    )
    parser.add_argument(
        "--ingest-strategies",
        type=_parse_str_list,
        default=["stream"],
        help="comma-separated ingest strategies: stream,preload",
    )
    parser.add_argument("--chunk-token-sizes", type=_parse_int_list, default=[480])
    parser.add_argument("--chunk-overlap-tokens", type=_parse_int_list, default=[64])
    parser.add_argument("--embedding-device", default=None, help="mps / cpu / auto")
    parser.add_argument("--embedding-provider", default=None, choices=["local-bge", "ollama"])
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--storage-root-base", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--skip-graph-extraction", action="store_true")
    parser.add_argument("--log-embedding-calls", action="store_true")
    parser.add_argument("--show-backend-progress", action="store_true")
    args = parser.parse_args()

    paths = ensure_benchmark_layout(default_benchmark_paths(args.dataset), tasks=("retrieval", "ingest"))
    documents_path = (
        paths.prepared_variant_dir(args.variant) / "documents.jsonl"
        if args.documents_path is None
        else Path(args.documents_path)
    )
    storage_root_base = (
        paths.index_root / "profile" / args.variant
        if args.storage_root_base is None
        else Path(args.storage_root_base)
    )
    output_csv = (
        paths.eval_variant_dir("ingest", args.variant) / "ingest_profile.csv"
        if args.output_csv is None
        else Path(args.output_csv)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    storage_root_base.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "doc_count",
        "indexed_object_count",
        "ingest_strategy",
        "ingest_batch_size",
        "encode_batch_size",
        "chunk_token_size",
        "chunk_overlap_tokens",
        "embedding_model",
        "device",
        "total_elapsed_ms",
        "embedding_elapsed_ms",
        "docs_per_second",
        "indexed_objects_per_second",
        "storage_root",
    ]
    run_count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()
        for doc_count in args.doc_counts:
            for ingest_strategy in args.ingest_strategies:
                for chunk_token_size in args.chunk_token_sizes:
                    for chunk_overlap_tokens in args.chunk_overlap_tokens:
                        for ingest_batch_size in args.ingest_batch_sizes:
                            for encode_batch_size in args.encode_batch_sizes:
                                run_root = (
                                    storage_root_base
                                    / (
                                        f"docs{doc_count}-strategy{ingest_strategy}"
                                        f"-chunk{chunk_token_size}-overlap{chunk_overlap_tokens}"
                                        f"-ingest{ingest_batch_size}-encode{encode_batch_size}"
                                    )
                                )
                                if run_root.exists():
                                    shutil.rmtree(run_root)
                                result = profile_benchmark_ingest(
                                    dataset=args.dataset,
                                    profile_id=args.profile,
                                    documents_path=documents_path,
                                    storage_root=run_root,
                                    doc_limit=doc_count,
                                    ingest_batch_size=ingest_batch_size,
                                    encode_batch_size=encode_batch_size,
                                    ingest_strategy=ingest_strategy,
                                    embedding_device=args.embedding_device,
                                    chunk_token_size=chunk_token_size,
                                    chunk_overlap_tokens=chunk_overlap_tokens,
                                    skip_graph_extraction=args.skip_graph_extraction,
                                    log_embedding_calls=args.log_embedding_calls,
                                    show_backend_progress=args.show_backend_progress,
                                    embedding_provider_kind=args.embedding_provider,
                                    embedding_model=args.embedding_model,
                                    embedding_model_path=args.embedding_model_path,
                                )
                                payload = result.as_json()
                                payload["chunk_token_size"] = chunk_token_size
                                payload["chunk_overlap_tokens"] = chunk_overlap_tokens
                                payload["storage_root"] = str(run_root)
                                print(json.dumps(payload, ensure_ascii=False))
                                writer.writerow(payload)
                                handle.flush()
                                run_count += 1
    print(json.dumps({"output_csv": str(output_csv), "run_count": run_count}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
