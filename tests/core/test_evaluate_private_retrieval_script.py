from __future__ import annotations

from types import SimpleNamespace

from scripts.evaluate_private_retrieval import (
    _build_parser,
    _miss_category,
    _rank_of_section_relation_hit,
)


def test_private_retrieval_parser_accepts_index_tokenizer_contract() -> None:
    args = _build_parser().parse_args(
        [
            "--chunk-token-size",
            "800",
            "--chunk-overlap-tokens",
            "120",
        ]
    )

    assert args.chunk_token_size == 800
    assert args.chunk_overlap_tokens == 120


def test_private_retrieval_parser_accepts_neighbor_radius() -> None:
    args = _build_parser().parse_args(["--neighbor-radius", "2"])

    assert args.neighbor_radius == 2


def test_private_retrieval_parser_defaults_to_milvus() -> None:
    args = _build_parser().parse_args([])

    assert args.vector_backend == "milvus"
    assert args.vector_dsn == "http://127.0.0.1:19530"


def test_parent_and_neighbor_section_ranks_use_refined_window_boundary() -> None:
    section_by_id = {
        "10": SimpleNamespace(
            section_id=10,
            doc_id=42,
            source_id=9,
            parent_section_id=10,
            toc_path=["Policy"],
            metadata_json={"window_index": 0},
        ),
        "11": SimpleNamespace(
            section_id=11,
            doc_id=42,
            source_id=9,
            parent_section_id=10,
            toc_path=["Policy"],
            metadata_json={"window_index": 1},
        ),
        "13": SimpleNamespace(
            section_id=13,
            doc_id=42,
            source_id=9,
            parent_section_id=10,
            toc_path=["Policy"],
            metadata_json={"window_index": 3},
        ),
        "20": SimpleNamespace(
            section_id=20,
            doc_id=42,
            source_id=9,
            parent_section_id=20,
            toc_path=["Policy"],
            metadata_json={"window_index": 0},
        ),
    }

    predicted = ["20", "13", "11"]

    assert (
        _rank_of_section_relation_hit(
            predicted,
            {"10"},
            section_by_id,
            top_k=10,
            relation="parent",
            neighbor_radius=1,
        )
        == 2
    )
    assert (
        _rank_of_section_relation_hit(
            predicted,
            {"10"},
            section_by_id,
            top_k=10,
            relation="neighbor",
            neighbor_radius=1,
        )
        == 3
    )


def test_miss_category_distinguishes_neighbor_from_doc_miss() -> None:
    section_by_id = {
        "10": SimpleNamespace(
            section_id=10,
            doc_id=42,
            source_id=9,
            parent_section_id=10,
            toc_path=["Policy"],
            metadata_json={"window_index": 0},
        ),
        "11": SimpleNamespace(
            section_id=11,
            doc_id=42,
            source_id=9,
            parent_section_id=10,
            toc_path=["Policy"],
            metadata_json={"window_index": 1},
        ),
    }

    assert (
        _miss_category(
            doc_rank=1,
            parent_section_rank=1,
            neighbor_section_rank=1,
            predicted_sections=["11"],
            gold_docs={"42"},
            section_by_id=section_by_id,
        )
        == "same_parent_neighbor"
    )
    assert (
        _miss_category(
            doc_rank=None,
            parent_section_rank=None,
            neighbor_section_rank=None,
            predicted_sections=["11"],
            gold_docs={"42"},
            section_by_id=section_by_id,
        )
        == "doc_miss"
    )
