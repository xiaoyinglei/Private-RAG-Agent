from __future__ import annotations

from types import SimpleNamespace

from rag.retrieval.models import QueryOptions, RetrievalProfile
from rag.retrieval.planning_graph import ComplexityGate, PlanningGraph
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy


class _MetadataScopeResolver:
    def __init__(self) -> None:
        self.calls: list[str | int] = []

    def get_document(self, doc_id: str | int):
        self.calls.append(doc_id)
        return SimpleNamespace(
            department_id="finance",
            auth_tag="internal",
            tenant_id="tenant-a",
        )


def test_planning_graph_promotes_large_doc_whitelist_to_attribute_filter() -> None:
    resolver = _MetadataScopeResolver()
    graph = PlanningGraph(metadata_scope_resolver=resolver, whitelist_threshold=1000)

    plan = graph.plan(
        "Compare Alpha and Beta across finance documents",
        source_scope=[str(index) for index in range(1200)],
        access_policy=AccessPolicy.default(),
        retrieval_signals=RetrievalSignals(
            allow_graph_expansion=True,
        ),
        resolved_retrieval_profile=RetrievalProfile.AUTO,
        query_options=QueryOptions(),
    )

    assert plan.complexity_gate is ComplexityGate.COMPLEX
    assert plan.predicate_plan.strategy == "attribute_filter"
    assert plan.predicate_plan.attribute_filters == {"department_id": ("finance",)}
    assert plan.predicate_plan.expression == 'department_id in ["finance"]'
    assert plan.predicate_plan.overflowed is True
    assert plan.target_collections == ("section_summary", "doc_summary")


def test_planning_graph_summary_hybrid_profile_uses_vector_and_special_only() -> None:
    graph = PlanningGraph(use_summary_hybrid_paths=True)

    plan = graph.plan(
        "第2页表格里的系统架构是什么？",
        source_scope=["42"],
        access_policy=AccessPolicy.default(),
        retrieval_signals=RetrievalSignals(



            special_targets=["table"],
        ),
        resolved_retrieval_profile=RetrievalProfile.AUTO,
        query_options=QueryOptions(top_k=6),
    )

    assert [path.branch for path in plan.retrieval_paths] == ["vector", "special"]


def test_planning_graph_summary_hybrid_emits_explicit_operator_and_fallback_plans() -> None:
    graph = PlanningGraph(use_summary_hybrid_paths=True)

    plan = graph.plan(
        "Compare Alpha and Beta in the page-2 table",
        source_scope=["42"],
        access_policy=AccessPolicy.default(),
        retrieval_signals=RetrievalSignals(
            allow_graph_expansion=True,
            special_targets=["table"],
        ),
        resolved_retrieval_profile=RetrievalProfile.AUTO,
        query_options=QueryOptions(top_k=4),
    )

    assert plan.version_gate_enabled is True
    assert [step.name for step in plan.operator_plan[:4]] == [
        "VersionGate",
        "EntitlementFilter",
        "QueryRewrite",
        "PredicatePushdown",
    ]
    assert [stage.collection for stage in plan.branch_stage_plans[0].stages] == [
        "section_summary",
        "doc_summary",
    ]
    assert {(step.trigger, step.branch) for step in plan.fallback_plan} == {
        ("asset_fallback", "special"),
    }
    assert len(plan.query_subtasks) == 2


def test_planning_graph_complex_comparison_emits_query_decomposition_subtasks() -> None:
    graph = PlanningGraph(use_summary_hybrid_paths=True)

    plan = graph.plan(
        "Compare Alpha and Beta across finance documents",
        source_scope=["42"],
        access_policy=AccessPolicy.default(),
        retrieval_signals=RetrievalSignals(
            allow_graph_expansion=True,
        ),
        resolved_retrieval_profile=RetrievalProfile.AUTO,
        query_options=QueryOptions(top_k=4),
    )

    assert len(plan.query_subtasks) == 2
    assert plan.operator_plan[0].name == "VersionGate"
    assert any(step.name == "QueryDecomposition" for step in plan.operator_plan)


def test_complexity_gate_defaults_to_standard_without_graph_expansion() -> None:
    query = "alpha beta gamma delta epsilon"

    assert PlanningGraph._complexity_gate(query, RetrievalSignals()) is ComplexityGate.STANDARD


def test_complexity_gate_returns_complex_when_graph_expansion_enabled() -> None:
    assert (
        PlanningGraph._complexity_gate(
            "alpha beta",
            RetrievalSignals(allow_graph_expansion=True),
        )
        is ComplexityGate.COMPLEX
    )
