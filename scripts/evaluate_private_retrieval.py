from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag.benchmarks import build_runtime_for_benchmark
from rag.retrieval.models import QueryOptions
from rag.schema.runtime import AccessPolicy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate private section-level retrieval gold set.")
    parser.add_argument("--golden-path", default="data/eval_private/golden_eval_dataset.jsonl")
    parser.add_argument("--storage-root", default="data/company_policy_index_recut")
    parser.add_argument("--retrieval-profile", default="auto", choices=["fast", "auto", "deep", "asset"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieval-pool-k", type=int, default=20)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--include-unanswerable", action="store_true")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rerank-provider", default=None)
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--rerank-model-path", default=None)
    parser.add_argument("--embedding-provider", default=None, choices=["local-bge", "ollama"])
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--chunk-token-size", type=int, default=None)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=None)
    parser.add_argument("--vector-backend", default="milvus", choices=["sqlite", "milvus", "pgvector"])
    parser.add_argument("--vector-dsn", default="http://127.0.0.1:19530")
    parser.add_argument("--vector-namespace", default=None)
    parser.add_argument("--vector-collection-prefix", default=None)
    parser.add_argument("--output", default="data/eval_private/private_retrieval_eval.json")
    parser.add_argument("--misses-output", default="data/eval_private/private_retrieval_misses.jsonl")
    return parser


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _id(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _metadata(item: object) -> dict[str, object]:
    metadata = getattr(item, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _target(item: object) -> object | None:
    return getattr(item, "grounding_target", None)


def _doc_id(item: object) -> str | None:
    target = _target(item)
    return (
        _id(getattr(target, "doc_id", None))
        or _id(getattr(item, "doc_id", None))
        or _id(_metadata(item).get("doc_id"))
    )


def _record_type(item: object) -> str:
    metadata = _metadata(item)
    value = (
        getattr(item, "record_type", None)
        or getattr(item, "candidate_kind", None)
        or metadata.get("record_type")
        or metadata.get("candidate_kind")
        or metadata.get("item_kind")
        or ""
    )
    return str(value)


def _section_id(item: object) -> str | None:
    target = _target(item)
    metadata = _metadata(item)
    section_id = (
        _id(getattr(target, "section_id", None))
        or _id(getattr(item, "section_id", None))
        or _id(metadata.get("section_id"))
    )
    if section_id is not None:
        return section_id
    record_type = _record_type(item)
    item_id = _id(getattr(item, "item_id", None) or getattr(item, "evidence_id", None))
    if item_id is not None and "section" in record_type:
        return item_id
    return None


def _gold_ids(row: dict[str, Any], key: str) -> set[str]:
    values = {_id(row.get(key))}
    for evidence in row.get("evidence") or []:
        if isinstance(evidence, dict):
            values.add(_id(evidence.get(key)))
    return {value for value in values if value is not None}


def _rank_of_hit(predicted: list[str | None], gold: set[str], *, top_k: int) -> int | None:
    for rank, value in enumerate(predicted[:top_k], start=1):
        if value is not None and value in gold:
            return rank
    return None


def _section_lookup(metadata_repo: object) -> dict[str, object]:
    list_sections = getattr(metadata_repo, "list_sections", None)
    if not callable(list_sections):
        return {}
    try:
        sections = list_sections()
    except TypeError:
        return {}
    return {str(section.section_id): section for section in sections}


def _metadata_json(section: object) -> dict[str, object]:
    metadata = getattr(section, "metadata_json", None)
    return metadata if isinstance(metadata, dict) else {}


def _int_value(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _section_window_index(section: object) -> int | None:
    metadata = _metadata_json(section)
    for key in ("window_index", "refined_window_index"):
        value = _int_value(metadata.get(key))
        if value is not None:
            return value
    return None


def _same_section_group(left: object, right: object) -> bool:
    left_parent = _int_value(getattr(left, "parent_section_id", None))
    right_parent = _int_value(getattr(right, "parent_section_id", None))
    if left_parent is None or right_parent is None:
        return False
    return (
        left_parent == right_parent
        and _id(getattr(left, "doc_id", None)) == _id(getattr(right, "doc_id", None))
        and _id(getattr(left, "source_id", None)) == _id(getattr(right, "source_id", None))
        and tuple(getattr(left, "toc_path", ()) or ()) == tuple(getattr(right, "toc_path", ()) or ())
    )


def _is_parent_section_hit(predicted_section: object, gold_section: object) -> bool:
    if _id(getattr(predicted_section, "section_id", None)) == _id(getattr(gold_section, "section_id", None)):
        return True
    return _same_section_group(predicted_section, gold_section)


def _is_neighbor_section_hit(predicted_section: object, gold_section: object, *, radius: int) -> bool:
    if _id(getattr(predicted_section, "section_id", None)) == _id(getattr(gold_section, "section_id", None)):
        return True
    if not _same_section_group(predicted_section, gold_section):
        return False
    predicted_window = _section_window_index(predicted_section)
    gold_window = _section_window_index(gold_section)
    if predicted_window is None or gold_window is None:
        return False
    return abs(predicted_window - gold_window) <= max(radius, 0)


def _rank_of_section_relation_hit(
    predicted: list[str | None],
    gold: set[str],
    section_by_id: dict[str, object],
    *,
    top_k: int,
    relation: str,
    neighbor_radius: int,
) -> int | None:
    gold_sections = [section_by_id[section_id] for section_id in gold if section_id in section_by_id]
    if not gold_sections:
        return _rank_of_hit(predicted, gold, top_k=top_k)
    for rank, predicted_section_id in enumerate(predicted[:top_k], start=1):
        if predicted_section_id is None:
            continue
        predicted_section = section_by_id.get(predicted_section_id)
        if predicted_section is None:
            continue
        for gold_section in gold_sections:
            if relation == "parent" and _is_parent_section_hit(predicted_section, gold_section):
                return rank
            if relation == "neighbor" and _is_neighbor_section_hit(
                predicted_section,
                gold_section,
                radius=neighbor_radius,
            ):
                return rank
    return None


def _same_doc_prediction(predicted: list[str | None], gold_docs: set[str], section_by_id: dict[str, object]) -> bool:
    for section_id in predicted:
        if section_id is None:
            continue
        section = section_by_id.get(section_id)
        if section is not None and _id(getattr(section, "doc_id", None)) in gold_docs:
            return True
    return False


def _miss_category(
    *,
    doc_rank: int | None,
    parent_section_rank: int | None,
    neighbor_section_rank: int | None,
    predicted_sections: list[str | None],
    gold_docs: set[str],
    section_by_id: dict[str, object],
) -> str:
    if doc_rank is None:
        return "doc_miss"
    if neighbor_section_rank is not None:
        return "same_parent_neighbor"
    if parent_section_rank is not None:
        return "same_parent_non_neighbor"
    if _same_doc_prediction(predicted_sections, gold_docs, section_by_id):
        return "same_doc_other_section"
    return "section_miss_other"


def _hit_at(rank: int | None, k: int) -> int:
    return int(rank is not None and rank <= k)


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    golden_path = Path(args.golden_path)
    rows = _load_jsonl(golden_path)
    if not args.include_unanswerable:
        rows = [row for row in rows if bool(row.get("answerable", True))]
    if args.query_limit is not None:
        rows = rows[: max(args.query_limit, 0)]

    output_path = Path(args.output)
    misses_path = Path(args.misses_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    misses_path.parent.mkdir(parents=True, exist_ok=True)

    top_k = max(args.top_k, 1)
    retrieval_pool_k = max(args.retrieval_pool_k, top_k)
    options = QueryOptions(
        retrieval_profile=args.retrieval_profile,
        top_k=top_k,
        evidence_top_k=top_k,
        max_evidence_items=top_k,
        retrieval_pool_k=retrieval_pool_k,
        rerank_pool_k=retrieval_pool_k,
        enable_rerank=bool(args.rerank),
    )

    totals: dict[str, Any] = {
        "query_count": 0,
        "doc_mrr": 0.0,
        "section_mrr": 0.0,
        "parent_section_mrr": 0.0,
        "neighbor_section_mrr": 0.0,
        "doc_hit": defaultdict(int),
        "section_hit": defaultdict(int),
        "parent_section_hit": defaultdict(int),
        "neighbor_section_hit": defaultdict(int),
        "miss_category": defaultdict(int),
        "returned_candidate_count_total": 0,
        "returned_candidate_count_min": None,
        "returned_candidate_count_max": 0,
    }
    by_type: dict[str, dict[str, Any]] = {}
    details: list[dict[str, Any]] = []

    runtime = build_runtime_for_benchmark(
        storage_root=Path(args.storage_root),
        require_chat=False,
        require_rerank=bool(args.rerank),
        skip_graph_extraction=True,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        embedding_provider_kind=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_model_path=args.embedding_model_path,
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
        rerank_provider_kind=args.rerank_provider,
        rerank_model=args.rerank_model,
        rerank_model_path=args.rerank_model_path,
        vector_backend=args.vector_backend,
        vector_dsn=args.vector_dsn,
        vector_namespace=args.vector_namespace,
        vector_collection_prefix=args.vector_collection_prefix,
    )
    try:
        section_by_id = _section_lookup(runtime.stores.metadata_repo)
        for index, row in enumerate(rows, start=1):
            query = str(row.get("question") or "").strip()
            if not query:
                continue
            gold_docs = _gold_ids(row, "doc_id")
            gold_sections = _gold_ids(row, "section_id")
            payload = runtime.retrieval_service.retrieve_payload(
                query,
                access_policy=AccessPolicy.default(),
                query_options=options,
            )
            ranked_items = list(payload.clean_items or payload.evidence.all)
            predicted_docs = [_doc_id(item) for item in ranked_items]
            predicted_sections = [_section_id(item) for item in ranked_items]
            doc_rank = _rank_of_hit(predicted_docs, gold_docs, top_k=top_k)
            section_rank = _rank_of_hit(predicted_sections, gold_sections, top_k=top_k)
            parent_section_rank = _rank_of_section_relation_hit(
                predicted_sections,
                gold_sections,
                section_by_id,
                top_k=top_k,
                relation="parent",
                neighbor_radius=max(args.neighbor_radius, 0),
            )
            neighbor_section_rank = _rank_of_section_relation_hit(
                predicted_sections,
                gold_sections,
                section_by_id,
                top_k=top_k,
                relation="neighbor",
                neighbor_radius=max(args.neighbor_radius, 0),
            )
            question_type = str(row.get("question_type") or "unknown")
            miss_category = _miss_category(
                doc_rank=doc_rank,
                parent_section_rank=parent_section_rank,
                neighbor_section_rank=neighbor_section_rank,
                predicted_sections=predicted_sections,
                gold_docs=gold_docs,
                section_by_id=section_by_id,
            )

            bucket = by_type.setdefault(
                question_type,
                {
                    "query_count": 0,
                    "doc_mrr": 0.0,
                    "section_mrr": 0.0,
                    "parent_section_mrr": 0.0,
                    "neighbor_section_mrr": 0.0,
                    "doc_hit": defaultdict(int),
                    "section_hit": defaultdict(int),
                    "parent_section_hit": defaultdict(int),
                    "neighbor_section_hit": defaultdict(int),
                    "miss_category": defaultdict(int),
                    "returned_candidate_count_total": 0,
                    "returned_candidate_count_min": None,
                    "returned_candidate_count_max": 0,
                },
            )
            for target in (totals, bucket):
                target["query_count"] += 1
                target["returned_candidate_count_total"] += len(predicted_sections)
                target["returned_candidate_count_min"] = (
                    len(predicted_sections)
                    if target["returned_candidate_count_min"] is None
                    else min(int(target["returned_candidate_count_min"]), len(predicted_sections))
                )
                target["returned_candidate_count_max"] = max(
                    int(target["returned_candidate_count_max"]),
                    len(predicted_sections),
                )
                target["doc_mrr"] += 0.0 if doc_rank is None else 1.0 / doc_rank
                target["section_mrr"] += 0.0 if section_rank is None else 1.0 / section_rank
                target["parent_section_mrr"] += (
                    0.0 if parent_section_rank is None else 1.0 / parent_section_rank
                )
                target["neighbor_section_mrr"] += (
                    0.0 if neighbor_section_rank is None else 1.0 / neighbor_section_rank
                )
                if section_rank is None:
                    target["miss_category"][miss_category] += 1
                for k in (1, 3, 5, 10, 20):
                    if k <= top_k:
                        target["doc_hit"][k] += _hit_at(doc_rank, k)
                        target["section_hit"][k] += _hit_at(section_rank, k)
                        target["parent_section_hit"][k] += _hit_at(parent_section_rank, k)
                        target["neighbor_section_hit"][k] += _hit_at(neighbor_section_rank, k)

            details.append(
                {
                    "query_id": row.get("query_id"),
                    "question": query,
                    "question_type": question_type,
                    "gold_doc_ids": sorted(gold_docs),
                    "gold_section_ids": sorted(gold_sections),
                    "doc_hit_rank": doc_rank,
                    "section_hit_rank": section_rank,
                    "parent_section_hit_rank": parent_section_rank,
                    "neighbor_section_hit_rank": neighbor_section_rank,
                    "miss_category": miss_category if section_rank is None else None,
                    "predicted_doc_ids": predicted_docs[:top_k],
                    "predicted_section_ids": predicted_sections[:top_k],
                    "returned_candidate_count": len(predicted_sections),
                }
            )
            if index % 25 == 0:
                print(f"evaluated {index}/{len(rows)} queries")
    finally:
        runtime.close()

    def summarize(target: dict[str, Any]) -> dict[str, Any]:
        total = int(target["query_count"])
        payload: dict[str, Any] = {
            "query_count": total,
            "doc_mrr": _rate(float(target["doc_mrr"]), total),
            "section_mrr": _rate(float(target["section_mrr"]), total),
            "parent_section_mrr": _rate(float(target["parent_section_mrr"]), total),
            "neighbor_section_mrr": _rate(float(target["neighbor_section_mrr"]), total),
            "returned_candidate_count_avg": _rate(float(target["returned_candidate_count_total"]), total),
            "returned_candidate_count_min": target["returned_candidate_count_min"],
            "returned_candidate_count_max": target["returned_candidate_count_max"],
        }
        for k in (1, 3, 5, 10, 20):
            if k <= top_k:
                payload[f"doc_hit@{k}"] = _rate(int(target["doc_hit"][k]), total)
                payload[f"section_hit@{k}"] = _rate(int(target["section_hit"][k]), total)
                payload[f"parent_section_hit@{k}"] = _rate(int(target["parent_section_hit"][k]), total)
                payload[f"neighbor_section_hit@{k}"] = _rate(int(target["neighbor_section_hit"][k]), total)
        payload["section_miss_category"] = dict(sorted(target["miss_category"].items()))
        return payload

    summary = {
        "golden_path": str(golden_path),
        "storage_root": str(Path(args.storage_root)),
        "retrieval_profile": args.retrieval_profile,
        "top_k": top_k,
        "neighbor_radius": max(args.neighbor_radius, 0),
        "rerank": bool(args.rerank),
        "vector_backend": args.vector_backend,
        "vector_collection_prefix": args.vector_collection_prefix,
        "overall": summarize(totals),
        "by_question_type": {key: summarize(value) for key, value in sorted(by_type.items())},
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with misses_path.open("w", encoding="utf-8") as handle:
        for detail in details:
            if detail["section_hit_rank"] is None:
                handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote summary: {output_path}")
    print(f"wrote section misses: {misses_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
