from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rag.benchmarks import (
    FIQA_DATASET,
    MEDICAL_RETRIEVAL_DATASET,
    build_runtime_for_benchmark,
    configure_runtime_embedding,
    default_benchmark_paths,
    ensure_benchmark_layout,
    ingest_prepared_documents,
    runtime_embedding_stats,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest prepared benchmark documents through the formal ingest pipeline."
    )
    parser.add_argument("--dataset", default=FIQA_DATASET, choices=[FIQA_DATASET, MEDICAL_RETRIEVAL_DATASET])
    parser.add_argument("--variant", default="full", choices=["full", "mini"])
    parser.add_argument("--documents-path", default=None)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preload", action="store_true", help="Load all prepared requests before ingesting.")
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-provider", default=None, choices=["local-bge", "ollama"])
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--chat-provider", default=None, choices=["ollama", "openai-compatible", "local-hf"])
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--chat-model-path", default=None)
    parser.add_argument("--chat-backend", default=None, choices=["auto", "mlx", "transformers"])
    parser.add_argument("--summary-provider", default=None, choices=["ollama", "openai-compatible", "local-hf"])
    parser.add_argument("--summary-model", default=None)
    parser.add_argument("--summary-model-path", default=None)
    parser.add_argument("--summary-backend", default=None, choices=["auto", "mlx", "transformers"])
    parser.add_argument("--vector-backend", default="milvus", choices=["sqlite", "milvus", "pgvector"])
    parser.add_argument("--vector-dsn", default="http://127.0.0.1:19530")
    parser.add_argument("--vector-namespace", default=None)
    parser.add_argument("--vector-collection-prefix", default=None)
    parser.add_argument("--chunk-token-size", type=int, default=None)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=None)
    parser.add_argument("--log-embedding-calls", action="store_true")
    parser.add_argument("--show-backend-progress", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-graph-extraction", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = _build_parser()
    args = parser.parse_args(argv)

    paths = ensure_benchmark_layout(default_benchmark_paths(args.dataset))
    documents_path = (
        paths.prepared_variant_dir(args.variant) / "documents.jsonl"
        if args.documents_path is None
        else Path(args.documents_path)
    )
    storage_root = Path(args.storage_root) if args.storage_root else paths.index_variant_dir(args.variant)
    storage_root.mkdir(parents=True, exist_ok=True)

    runtime = build_runtime_for_benchmark(
        storage_root=storage_root,
        profile_id=args.profile,
        require_chat=not args.skip_graph_extraction,
        require_rerank=False,
        skip_graph_extraction=args.skip_graph_extraction,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        log_embedding_calls=args.log_embedding_calls,
        show_backend_progress=args.show_backend_progress,
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
        embedding_provider_kind=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_model_path=args.embedding_model_path,
        chat_provider_kind=args.chat_provider,
        chat_model=args.chat_model,
        chat_model_path=args.chat_model_path,
        chat_backend=args.chat_backend,
        summary_provider_kind=args.summary_provider,
        summary_model=args.summary_model,
        summary_model_path=args.summary_model_path,
        summary_backend=args.summary_backend,
        vector_backend=args.vector_backend,
        vector_dsn=args.vector_dsn,
        vector_namespace=args.vector_namespace,
        vector_collection_prefix=args.vector_collection_prefix,
    )
    try:
        embedding_info = configure_runtime_embedding(
            runtime,
            encode_batch_size=args.embedding_batch_size,
            device=args.embedding_device,
            log_embedding_calls=args.log_embedding_calls,
            show_backend_progress=args.show_backend_progress,
        )
        if embedding_info is not None:
            print(
                json.dumps(
                    {
                        "event": "embedding_runtime",
                        "embedding_model": embedding_info.model_name,
                        "device": embedding_info.device,
                        "encode_batch_size": embedding_info.encode_batch_size,
                        "ingest_batch_size": max(args.batch_size, 1),
                        "ingest_strategy": "preload" if args.preload else "stream",
                        "chunk_token_size": args.chunk_token_size,
                        "chunk_overlap_tokens": args.chunk_overlap_tokens,
                        "embedding_provider": args.embedding_provider,
                        "embedding_model_override": args.embedding_model,
                        "chat_provider": args.chat_provider,
                        "chat_model_override": args.chat_model,
                        "chat_model_path_override": args.chat_model_path,
                        "chat_backend_override": args.chat_backend,
                        "summary_provider": args.summary_provider,
                        "summary_model_override": args.summary_model,
                        "summary_model_path_override": args.summary_model_path,
                        "summary_backend_override": args.summary_backend,
                        "vector_backend": args.vector_backend,
                        "vector_namespace": args.vector_namespace,
                        "vector_collection_prefix": args.vector_collection_prefix,
                    },
                    ensure_ascii=False,
                )
            )
        result = ingest_prepared_documents(
            runtime,
            dataset=args.dataset,
            documents_path=documents_path,
            batch_size=max(args.batch_size, 1),
            continue_on_error=args.continue_on_error,
            streaming=not args.preload,
        )
        embedding_stats = runtime_embedding_stats(runtime)
        payload = result.as_json()
        payload.update(
            {
                "variant": args.variant,
                "storage_root": str(storage_root),
                "selected_profile_id": runtime.selected_profile_id,
                "skip_graph_extraction": args.skip_graph_extraction,
                "ingest_batch_size": max(args.batch_size, 1),
                "ingest_strategy": "preload" if args.preload else "stream",
                "chunk_token_size": args.chunk_token_size,
                "chunk_overlap_tokens": args.chunk_overlap_tokens,
                "embedding_provider": args.embedding_provider,
                "embedding_model_override": args.embedding_model,
                "chat_provider": args.chat_provider,
                "chat_model_override": args.chat_model,
                "chat_model_path_override": args.chat_model_path,
                "chat_backend_override": args.chat_backend,
                "summary_provider": args.summary_provider,
                "summary_model_override": args.summary_model,
                "summary_model_path_override": args.summary_model_path,
                "summary_backend_override": args.summary_backend,
                "vector_backend": args.vector_backend,
                "vector_namespace": args.vector_namespace,
                "vector_collection_prefix": args.vector_collection_prefix,
                "embedding_stats": embedding_stats,
            }
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
