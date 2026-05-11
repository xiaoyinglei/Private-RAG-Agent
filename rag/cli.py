from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import click
import typer
from pydantic import BaseModel

from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageComponentConfig, StorageConfig
from rag.agent.cli import agent_app
from rag.retrieval import QueryOptions, RetrievalProfile
from rag.schema.core import SourceType

app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(agent_app, name="agent")
DEFAULT_STORAGE_ROOT = Path(".rag")
DEFAULT_VECTOR_BACKEND = "milvus"
DEFAULT_VECTOR_DSN = "http://127.0.0.1:19530"
FIQA_DATASET = "fiqa"
MEDICAL_RETRIEVAL_DATASET = "medical_retrieval"
STORAGE_ROOT_OPTION = typer.Option("--storage-root")
SOURCE_TYPE_OPTION = typer.Option("--source-type")
LOCATION_OPTION = typer.Option("--location")
CONTENT_OPTION = typer.Option("--content")
TITLE_OPTION = typer.Option("--title")
OWNER_OPTION = typer.Option("--owner")
QUERY_OPTION = typer.Option("--query")
RETRIEVAL_PROFILE_OPTION = typer.Option(
    "--retrieval-profile",
    help="New retrieval profile: auto, fast, deep, or bypass.",
)
JSON_OPTION = typer.Option("--json")
DOC_ID_OPTION = typer.Option("--doc-id")
SOURCE_ID_OPTION = typer.Option("--source-id")
PROFILE_OPTION = typer.Option("--profile", help="Recommended assembly profile to use.")
DATASET_OPTION = typer.Option("--dataset", help="Public benchmark dataset.")


def _runtime(storage_root: Path, *, profile_id: str | None = None, require_chat: bool = False) -> RAGRuntime:
    request = (
        CapabilityRequirements(
            require_chat=require_chat,
            default_context_tokens=QueryOptions().max_context_tokens,
        )
    )
    if profile_id:
        return RAGRuntime.from_profile(
            storage=_default_storage_config(storage_root),
            profile_id=profile_id,
            requirements=request,
        )
    return RAGRuntime.from_request(
        storage=_default_storage_config(storage_root),
        request=AssemblyRequest(requirements=request),
    )


def _default_storage_config(storage_root: Path) -> StorageConfig:
    return StorageConfig(
        root=storage_root,
        vectors=StorageComponentConfig(
            backend=DEFAULT_VECTOR_BACKEND,
            dsn=DEFAULT_VECTOR_DSN,
        ),
    )


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        payload = {field.name: getattr(value, field.name) for field in fields(value)}
        for computed_name in ("indexed_object_count", "success_count", "failure_count"):
            computed_value = getattr(value, computed_name, None)
            if computed_value is not None:
                payload[computed_name] = computed_value
        return payload
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True, default=_json_default))


def _requires_content(source_type: SourceType) -> bool:
    return source_type in {SourceType.PLAIN_TEXT, SourceType.PASTED_TEXT, SourceType.BROWSER_CLIP}


@app.command()
def ingest(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    profile_id: Annotated[str | None, PROFILE_OPTION] = None,
    source_type: Annotated[SourceType | None, SOURCE_TYPE_OPTION] = None,
    location: Annotated[str | None, LOCATION_OPTION] = None,
    content: Annotated[str | None, CONTENT_OPTION] = None,
    title: Annotated[str | None, TITLE_OPTION] = None,
    owner: Annotated[str, OWNER_OPTION] = "user",
) -> None:
    if source_type is None:
        raise typer.BadParameter("--source-type is required")
    if location is None or not location.strip():
        raise typer.BadParameter("--location is required")
    if _requires_content(source_type) and content is None:
        raise typer.BadParameter("--content is required for text-based ingest")

    with _runtime(storage_root, profile_id=profile_id, require_chat=False) as runtime:
        result = runtime.insert(
            source_type=source_type.value,
            location=location,
            owner=owner,
            title=title,
            content_text=content,
        )
    _echo_json(result)


@app.command()
def query(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    profile_id: Annotated[str | None, PROFILE_OPTION] = None,
    query: Annotated[str | None, QUERY_OPTION] = None,
    retrieval_profile: Annotated[RetrievalProfile, RETRIEVAL_PROFILE_OPTION] = RetrievalProfile.AUTO,
    json_output: Annotated[bool, JSON_OPTION] = False,
) -> None:
    if query is None or not query.strip():
        raise typer.BadParameter("--query is required")
    with _runtime(storage_root, profile_id=profile_id, require_chat=False) as runtime:
        result = runtime.query_public(query, options=QueryOptions(retrieval_profile=retrieval_profile.value))
    if json_output:
        _echo_json(result)
        return
    typer.echo(result.answer.answer_text)


@app.command("analyze-task")
def analyze_task(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    profile_id: Annotated[str | None, PROFILE_OPTION] = None,
    query: Annotated[str | None, QUERY_OPTION] = None,
    json_output: Annotated[bool, JSON_OPTION] = False,
    allow_web: Annotated[bool, typer.Option("--allow-web/--no-allow-web")] = False,
    expected_output: Annotated[str, typer.Option("--expected-output")] = "structured_analysis_report",
    response_style: Annotated[str, typer.Option("--response-style")] = "formal",
    max_subtasks: Annotated[int, typer.Option("--max-subtasks")] = 5,
    retry_budget: Annotated[int, typer.Option("--retry-budget")] = 2,
) -> None:
    del (
        storage_root,
        profile_id,
        query,
        json_output,
        allow_web,
        expected_output,
        response_style,
        max_subtasks,
        retry_budget,
    )
    raise click.ClickException("analyze-task is disabled on the new runtime CLI; use `rag query`.")


@app.command()
def delete(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    profile_id: Annotated[str | None, PROFILE_OPTION] = None,
    doc_id: Annotated[str | None, DOC_ID_OPTION] = None,
    source_id: Annotated[str | None, SOURCE_ID_OPTION] = None,
    location: Annotated[str | None, LOCATION_OPTION] = None,
) -> None:
    if doc_id is None and source_id is None and (location is None or not location.strip()):
        raise typer.BadParameter("--doc-id, --source-id, or --location is required")
    with _runtime(storage_root, profile_id=profile_id, require_chat=False) as runtime:
        result = runtime.delete(doc_id=doc_id, source_id=source_id, location=location)
    _echo_json(result)


@app.command()
def rebuild(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    profile_id: Annotated[str | None, PROFILE_OPTION] = None,
    doc_id: Annotated[str | None, DOC_ID_OPTION] = None,
    source_id: Annotated[str | None, SOURCE_ID_OPTION] = None,
    location: Annotated[str | None, LOCATION_OPTION] = None,
) -> None:
    if doc_id is None and source_id is None and (location is None or not location.strip()):
        raise typer.BadParameter("--doc-id, --source-id, or --location is required")
    try:
        with _runtime(storage_root, profile_id=profile_id, require_chat=False) as runtime:
            result = runtime.rebuild(doc_id=doc_id, source_id=source_id, location=location)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_json(result)


@app.command("profiles")
def list_profiles(
    json_output: Annotated[bool, JSON_OPTION] = False,
) -> None:
    runtime = RAGRuntime.from_request(
        storage=StorageConfig.in_memory(),
        request=AssemblyRequest(
            requirements=CapabilityRequirements(
                require_chat=False,
                default_context_tokens=QueryOptions().max_context_tokens,
            )
        ),
    )
    try:
        catalog = runtime.catalog
        payload = [
            {
                "profile_id": profile.profile_id,
                "label": profile.label,
                "description": profile.description,
                "location": profile.location,
                "recommended_requirements": {
                    "require_embedding": profile.recommended_requirements.require_embedding,
                    "require_chat": profile.recommended_requirements.require_chat,
                    "require_rerank": profile.recommended_requirements.require_rerank,
                    "allow_degraded": profile.recommended_requirements.allow_degraded,
                },
            }
            for profile in catalog.assembly_profiles
        ]
        if json_output:
            _echo_json(payload)
            return
        for profile in payload:
            typer.echo(f"{profile['profile_id']}: {profile['label']} [{profile['location']}]")
            typer.echo(f"  {profile['description']}")
    finally:
        runtime.close()


@app.command("benchmark-download")
def benchmark_download(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    raw_dir: Annotated[Path | None, typer.Option("--raw-dir")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    from rag.benchmarks import default_benchmark_paths, download_public_benchmark, ensure_benchmark_layout

    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    result = download_public_benchmark(dataset, paths.raw_dir if raw_dir is None else raw_dir, force=force)
    _echo_json(result)


@app.command("benchmark-prepare")
def benchmark_prepare(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    raw_dir: Annotated[Path | None, typer.Option("--raw-dir")] = None,
    prepared_dir: Annotated[Path | None, typer.Option("--prepared-dir")] = None,
    split: Annotated[str | None, typer.Option("--split")] = None,
    no_mini: Annotated[bool, typer.Option("--no-mini")] = False,
    mini_query_count: Annotated[int | None, typer.Option("--mini-query-count")] = None,
    mini_doc_count: Annotated[int | None, typer.Option("--mini-doc-count")] = None,
) -> None:
    from rag.benchmarks import (
        benchmark_dataset_spec,
        default_benchmark_paths,
        ensure_benchmark_layout,
        prepare_public_benchmark,
    )

    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    spec = benchmark_dataset_spec(dataset)
    result = prepare_public_benchmark(
        dataset,
        paths.raw_dir if raw_dir is None else raw_dir,
        paths.prepared_root if prepared_dir is None else prepared_dir,
        split=split or spec.default_split,
        build_mini=not no_mini,
        mini_query_count=mini_query_count,
        mini_target_doc_count=mini_doc_count,
    )
    _echo_json(result)


@app.command("benchmark-ingest")
def benchmark_ingest(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    variant: Annotated[str, typer.Option("--variant")] = "full",
    profile_id: Annotated[str, PROFILE_OPTION] = "local_full",
    storage_root: Annotated[Path | None, STORAGE_ROOT_OPTION] = None,
    documents_path: Annotated[Path | None, typer.Option("--documents-path")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 64,
    embedding_provider: Annotated[str | None, typer.Option("--embedding-provider")] = None,
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    embedding_model_path: Annotated[str | None, typer.Option("--embedding-model-path")] = None,
    embedding_batch_size: Annotated[int | None, typer.Option("--embedding-batch-size")] = None,
    chat_provider: Annotated[str | None, typer.Option("--chat-provider")] = None,
    chat_model: Annotated[str | None, typer.Option("--chat-model")] = None,
    chat_model_path: Annotated[str | None, typer.Option("--chat-model-path")] = None,
    chat_backend: Annotated[str | None, typer.Option("--chat-backend")] = None,
    summary_provider: Annotated[str | None, typer.Option("--summary-provider")] = None,
    summary_model: Annotated[str | None, typer.Option("--summary-model")] = None,
    summary_model_path: Annotated[str | None, typer.Option("--summary-model-path")] = None,
    summary_backend: Annotated[str | None, typer.Option("--summary-backend")] = None,
    vector_backend: Annotated[str, typer.Option("--vector-backend")] = "milvus",
    vector_dsn: Annotated[str | None, typer.Option("--vector-dsn")] = None,
    vector_namespace: Annotated[str | None, typer.Option("--vector-namespace")] = None,
    vector_collection_prefix: Annotated[str | None, typer.Option("--vector-collection-prefix")] = None,
    continue_on_error: Annotated[bool, typer.Option("--continue-on-error")] = False,
    skip_graph_extraction: Annotated[bool, typer.Option("--skip-graph-extraction/--with-graph-extraction")] = True,
) -> None:
    from rag.benchmarks import (
        build_runtime_for_benchmark,
        default_benchmark_paths,
        ensure_benchmark_layout,
        ingest_prepared_documents,
    )

    if variant not in {"full", "mini"}:
        raise typer.BadParameter("variant must be one of: full, mini")
    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    runtime = build_runtime_for_benchmark(
        storage_root=storage_root or paths.index_variant_dir(variant),
        profile_id=profile_id,
        require_chat=not skip_graph_extraction,
        require_rerank=False,
        skip_graph_extraction=skip_graph_extraction,
        embedding_provider_kind=embedding_provider,
        embedding_model=embedding_model,
        embedding_model_path=embedding_model_path,
        embedding_batch_size=embedding_batch_size,
        chat_provider_kind=chat_provider,
        chat_model=chat_model,
        chat_model_path=chat_model_path,
        chat_backend=chat_backend,
        summary_provider_kind=summary_provider,
        summary_model=summary_model,
        summary_model_path=summary_model_path,
        summary_backend=summary_backend,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )
    try:
        result = ingest_prepared_documents(
            runtime,
            dataset=dataset,
            documents_path=documents_path or (paths.prepared_variant_dir(variant) / "documents.jsonl"),
            batch_size=max(batch_size, 1),
            continue_on_error=continue_on_error,
        )
        _echo_json(result)
    finally:
        runtime.close()


@app.command("benchmark-evaluate")
def benchmark_evaluate(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    variant: Annotated[str, typer.Option("--variant")] = "full",
    profile_id: Annotated[str, PROFILE_OPTION] = "local_full",
    storage_root: Annotated[Path | None, STORAGE_ROOT_OPTION] = None,
    queries_path: Annotated[Path | None, typer.Option("--queries-path")] = None,
    qrels_path: Annotated[Path | None, typer.Option("--qrels-path")] = None,
    eval_dir: Annotated[Path | None, typer.Option("--eval-dir")] = None,
    retrieval_profile: Annotated[RetrievalProfile, RETRIEVAL_PROFILE_OPTION] = RetrievalProfile.AUTO,
    top_k: Annotated[int, typer.Option("--top-k")] = 10,
    evidence_top_k: Annotated[int | None, typer.Option("--evidence-top-k")] = None,
    vector_backend: Annotated[str, typer.Option("--vector-backend")] = "milvus",
    vector_dsn: Annotated[str | None, typer.Option("--vector-dsn")] = None,
    vector_namespace: Annotated[str | None, typer.Option("--vector-namespace")] = None,
    vector_collection_prefix: Annotated[str | None, typer.Option("--vector-collection-prefix")] = None,
    rerank_enabled: Annotated[bool, typer.Option("--rerank/--no-rerank")] = True,
    embedding_provider: Annotated[str | None, typer.Option("--embedding-provider")] = None,
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    embedding_model_path: Annotated[str | None, typer.Option("--embedding-model-path")] = None,
    rerank_provider: Annotated[str | None, typer.Option("--rerank-provider")] = None,
    rerank_model: Annotated[str | None, typer.Option("--rerank-model")] = None,
    rerank_model_path: Annotated[str | None, typer.Option("--rerank-model-path")] = None,
    chat_provider: Annotated[str | None, typer.Option("--chat-provider")] = None,
    chat_model: Annotated[str | None, typer.Option("--chat-model")] = None,
    chat_model_path: Annotated[str | None, typer.Option("--chat-model-path")] = None,
    chat_backend: Annotated[str | None, typer.Option("--chat-backend")] = None,
    split: Annotated[str | None, typer.Option("--split")] = None,
) -> None:
    from rag.benchmarks import (
        RetrievalBenchmarkEvaluator,
        benchmark_access_policy,
        benchmark_dataset_spec,
        build_runtime_for_benchmark,
        default_benchmark_paths,
        ensure_benchmark_layout,
    )
    from rag.schema.runtime import ExecutionLocationPreference

    if variant not in {"full", "mini"}:
        raise typer.BadParameter("variant must be one of: full, mini")
    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    spec = benchmark_dataset_spec(dataset)
    top_k = max(top_k, 1)
    evidence_top_k = max(evidence_top_k or max(top_k * 4, 40), top_k)
    runtime = build_runtime_for_benchmark(
        storage_root=storage_root or paths.index_variant_dir(variant),
        profile_id=profile_id,
        require_chat=False,
        require_rerank=rerank_enabled,
        embedding_provider_kind=embedding_provider,
        embedding_model=embedding_model,
        embedding_model_path=embedding_model_path,
        rerank_provider_kind=rerank_provider,
        rerank_model=rerank_model,
        rerank_model_path=rerank_model_path,
        chat_provider_kind=chat_provider,
        chat_model=chat_model,
        chat_model_path=chat_model_path,
        chat_backend=chat_backend,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )
    try:
        summary = RetrievalBenchmarkEvaluator(
            runtime=runtime,
            dataset=dataset,
            split=split or spec.default_split,
            retrieval_profile=retrieval_profile.value,
            top_k=top_k,
            evidence_top_k=evidence_top_k,
            rerank_enabled=rerank_enabled,
            execution_location_preference=ExecutionLocationPreference.LOCAL_ONLY,
            access_policy=benchmark_access_policy(),
        ).evaluate(
            queries_path=queries_path or (paths.prepared_variant_dir(variant) / "queries.jsonl"),
            qrels_path=qrels_path or (paths.prepared_variant_dir(variant) / "qrels.jsonl"),
            eval_dir=eval_dir or paths.eval_variant_dir("retrieval", variant),
        )
        payload = summary.as_json()
        payload["variant"] = variant
        _echo_json(payload)
    finally:
        runtime.close()


__all__ = ["app", "main"]


def main() -> None:
    app()


if __name__ == "__main__":
    main()
