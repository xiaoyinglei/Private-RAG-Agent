from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from rag.benchmarks import build_runtime_for_benchmark, runtime_embedding_stats
from rag.ingest.pipeline import IngestRequest
from rag.schema.core import SourceType

_SOURCE_TYPES_BY_SUFFIX = {
    ".pdf": SourceType.PDF,
    ".md": SourceType.MARKDOWN,
    ".markdown": SourceType.MARKDOWN,
    ".docx": SourceType.DOCX,
    ".pptx": SourceType.PPTX,
    ".xlsx": SourceType.XLSX,
    ".png": SourceType.IMAGE,
    ".jpg": SourceType.IMAGE,
    ".jpeg": SourceType.IMAGE,
    ".webp": SourceType.IMAGE,
    ".txt": SourceType.PLAIN_TEXT,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest private local documents through the formal parser -> section -> summary -> vector pipeline."
    )
    parser.add_argument("--input", required=True, help="File or directory containing private documents.")
    parser.add_argument("--storage-root", default="data/private_index_milvus")
    parser.add_argument("--owner", default="private")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--embedding-provider", default="ollama", choices=["local-bge", "ollama"])
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--chat-provider", default=None, choices=["ollama", "openai-compatible", "local-hf"])
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--chat-model-path", default=None)
    parser.add_argument("--chat-backend", default=None, choices=["auto", "mlx", "transformers"])
    parser.add_argument("--summary-provider", default=None, choices=["ollama", "openai-compatible", "local-hf"])
    parser.add_argument("--summary-model", default=None)
    parser.add_argument("--summary-model-path", default=None)
    parser.add_argument("--summary-backend", default=None, choices=["auto", "mlx", "transformers"])
    parser.add_argument("--vector-backend", default="milvus", choices=["milvus", "sqlite"])
    parser.add_argument("--vector-dsn", default=None)
    parser.add_argument("--vector-namespace", default=None)
    parser.add_argument("--vector-collection-prefix", default=None)
    parser.add_argument("--chunk-token-size", type=int, default=None)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser


def _iter_files(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_file():
        files = [input_path]
    else:
        pattern = "**/*" if recursive else "*"
        files = [path for path in input_path.glob(pattern) if path.is_file()]
    supported = [
        path
        for path in files
        if path.suffix.lower() in _SOURCE_TYPES_BY_SUFFIX and not path.name.startswith(".")
    ]
    return sorted(supported, key=lambda path: str(path))


def _request_for_file(path: Path, *, owner: str) -> IngestRequest:
    suffix = path.suffix.lower()
    source_type = _SOURCE_TYPES_BY_SUFFIX[suffix]
    return IngestRequest(
        location=str(path),
        source_type=source_type,
        owner=owner,
        title=path.stem,
        file_path=path,
        metadata={
            "dataset": "private",
            "private_corpus": "true",
            "original_file_name": path.name,
        },
    )


def _vector_counts(runtime) -> dict[str, int]:
    return {
        item_kind: int(runtime.stores.vector_repo.count_vectors(item_kind=item_kind))
        for item_kind in ("doc_summary", "section_summary", "asset_summary")
    }


def _summary_generator_info(runtime) -> dict[str, object] | None:
    summarizer = getattr(runtime.ingest_pipeline, "_summarizer", None)
    generator_info = getattr(summarizer, "generator_info", None)
    if not callable(generator_info):
        return None
    return dict(generator_info())


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")

    files = _iter_files(input_path, recursive=args.recursive)
    if not files:
        raise RuntimeError(f"no supported private documents found under: {input_path}")

    runtime = build_runtime_for_benchmark(
        storage_root=Path(args.storage_root),
        require_chat=False,
        require_rerank=False,
        skip_graph_extraction=True,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
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
        success_count = 0
        failure_count = 0
        indexed_object_count = 0
        errors: list[dict[str, str]] = []
        batch_size = max(args.batch_size, 1)
        requests = [_request_for_file(path, owner=args.owner) for path in files]

        with tqdm(total=len(requests), desc="Ingesting private documents", unit="doc") as progress:
            for start in range(0, len(requests), batch_size):
                batch = requests[start : start + batch_size]
                result = runtime.insert_many(batch, continue_on_error=args.continue_on_error)
                success_count += result.success_count
                failure_count += result.failure_count
                indexed_object_count += result.indexed_object_count
                for item in result.results:
                    if item.error is not None:
                        errors.append(
                            {
                                "location": item.request.location,
                                "error": item.error,
                            }
                        )
                progress.update(len(batch))
                progress.set_postfix(success=success_count, failure=failure_count)

        payload = {
            "input": str(input_path),
            "storage_root": str(Path(args.storage_root)),
            "vector_backend": args.vector_backend,
            "vector_namespace": args.vector_namespace,
            "vector_collection_prefix": args.vector_collection_prefix,
            "document_count": len(files),
            "success_count": success_count,
            "failure_count": failure_count,
            "indexed_object_count": indexed_object_count,
            "summary_vector_counts": _vector_counts(runtime),
            "summary_generator": _summary_generator_info(runtime),
            "embedding_stats": runtime_embedding_stats(runtime),
            "errors": errors,
        }
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
