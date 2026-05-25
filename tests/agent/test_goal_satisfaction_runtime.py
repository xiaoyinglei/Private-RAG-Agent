from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag.agent.binding_providers import AssetContextBindingProvider
from rag.agent.builtin.research import RESEARCH_AGENT
from rag.agent.core.context import AgentRunConfig, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.goal_runtime import (
    AnswerCandidate,
    ContextBindingAssessor,
    ContextUnit,
    EvidenceRef,
    GoalGap,
    GoalInitializer,
    SatisfactionChecker,
    StateReducer,
)
from rag.agent.graphs.nodes.goal_runtime import controller_node
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.state import AgentState, ThinkOutput, ToolCallPlan
from rag.agent.tools.asset_tools import (
    AssetAnalyzeOutput,
    AssetDescriptor,
    AssetListOutput,
)
from rag.agent.tools.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.llm_tools import LLMTextOutput
from rag.agent.tools.rag_tools import SearchOutput
from rag.agent.tools.spec import ToolResult
from rag.schema.runtime import AccessPolicy


@dataclass
class _SummarizeRunner:
    text: str = "北方和东北日提货合计为 15.491928，出处为 table@p4。"

    def __call__(self, payload: object) -> LLMTextOutput:
        del payload
        return LLMTextOutput(
            text=self.text,
            evidence_ids=["compute_result:14"],
            citation_ids=["table@p4"],
        )


class _FailingDecisionProvider:
    def decide(
        self,
        state: AgentState,
        *,
        definition: AgentDefinition,
        budget_remaining: int,
        context: object,
    ) -> ThinkOutput:
        del state, definition, budget_remaining, context
        raise AssertionError("ToolDecisionProvider must not run after goal is satisfied")


@pytest.mark.anyio
async def test_runtime_finalizes_satisfied_goal_without_llm_decision() -> None:
    call = ToolCallPlan.create(
        "llm_summarize",
        {
            "task": "北方和东北日提货合计是多少？请给出处",
            "context_sections": ["tool-computed result with source"],
        },
    )
    service = AgentService(
        definition=RESEARCH_AGENT,
        tool_registry=create_builtin_tool_registry(
            runners={"llm_summarize": _SummarizeRunner()}
        ),
        tool_decision_provider=_FailingDecisionProvider(),
    )

    result = await service.run(
        AgentRunRequest(
            task="北方和东北日提货合计是多少？请给出处",
            run_id="goal-runtime-satisfied",
            thread_id="goal-runtime-satisfied",
            pending_tool_calls=[call],
        )
    )

    assert result.status == "done"
    assert result.final_answer == "北方和东北日提货合计为 15.491928，出处为 table@p4。"
    assert result.stop_reason == "goal_satisfied"
    assert result.tool_results[0].tool_name == "llm_summarize"


def test_goal_initializer_requires_binding_for_explicit_source_scope() -> None:
    goal = GoalInitializer().initialize(
        "在分区域分品牌 石膏板-26年表中，北方和东北当日销售额合计是多少？请给出处"
    )

    assert goal.constraints[0].constraint_type == "context_title"
    assert goal.constraints[0].expected_value == "分区域分品牌 石膏板-26年"
    assert goal.constraints[0].required is True
    assert [gap.gap_type for gap in goal.open_gaps()] == [
        "context_binding",
        "answer",
        "evidence",
    ]


def test_context_binding_assessor_selects_explicitly_matching_table_candidate() -> None:
    goal = GoalInitializer().initialize(
        "在分区域分品牌 石膏板-26年表中，北方和东北当日销售额合计是多少？请给出处"
    )
    asset_list_result = ToolResult(
        tool_call_id="tc-assets",
        tool_name="asset_list",
        status="ok",
        output=AssetListOutput(
            assets=[
                AssetDescriptor(
                    asset_id=13,
                    doc_id=2,
                    source_id=1,
                    asset_type="table",
                    sheet_name="模板（套公式） -龙骨",
                ),
                AssetDescriptor(
                    asset_id=15,
                    doc_id=2,
                    source_id=1,
                    asset_type="table",
                    sheet_name="分区域分品牌 石膏板-26年",
                ),
            ]
        ),
        latency_ms=0,
    )
    state = {
        "task": goal.original_query,
        "goal_spec": goal,
        "tool_results": [asset_list_result],
        "structured_observations": [],
        "answer_candidates": [],
        "evidence_refs": [],
        "context_bindings": [],
        "satisfied_requirements": [],
        "open_gaps": goal.open_gaps(),
        "conflicts": [],
        "no_progress_count": 0,
        "pending_tool_calls": [],
    }
    state.update(StateReducer().reduce_tool_results(state))

    assessor = ContextBindingAssessor(providers=[AssetContextBindingProvider()])
    [binding] = assessor.assess_bindings(state)

    assert binding.status == "satisfied"
    assert binding.unit_id == "asset:15"


def test_asset_list_observation_keeps_compact_asset_locators_and_context_unit() -> None:
    asset_list_result = ToolResult(
        tool_call_id="tc-assets",
        tool_name="asset_list",
        status="ok",
        output=AssetListOutput(
            assets=[
                AssetDescriptor(
                    asset_id=14,
                    doc_id=2,
                    source_id=1,
                    section_id=6,
                    asset_type="table",
                    sheet_name="2024-0317新增",
                    columns=["区域公司", "日_日提货", "月累计_月累计提货"],
                    sample_rows=[
                        {
                            "区域公司": "北方",
                            "日_日提货": "12.660888",
                            "月累计_月累计提货": "329.946283",
                        }
                    ],
                    analysis_capabilities=["dataframe_preview", "dataframe_sql"],
                )
            ]
        ),
        latency_ms=0,
    )
    state = {
        "task": "北方和东北日提货合计是多少？请给出处",
        "tool_results": [asset_list_result],
        "structured_observations": [],
        "answer_candidates": [],
        "evidence_refs": [],
    }

    update = StateReducer().reduce_tool_results(state)
    observation = update["structured_observations"][0]

    assert observation.locators == [
        {
            "asset_id": 14,
            "doc_id": 2,
            "source_id": 1,
            "section_id": 6,
            "asset_type": "table",
            "sheet_name": "2024-0317新增",
            "columns": ["区域公司", "日_日提货", "月累计_月累计提货"],
            "analysis_capabilities": ["dataframe_preview", "dataframe_sql"],
        }
    ]
    assert update["context_units"] == [
        ContextUnit(
            unit_id="asset:14",
            unit_type="table_asset",
            locator={
                "asset_id": 14,
                "doc_id": 2,
                "source_id": 1,
                "section_id": 6,
                "asset_type": "table",
                "sheet_name": "2024-0317新增",
                "columns": ["区域公司", "日_日提货", "月累计_月累计提货"],
                "analysis_capabilities": ["dataframe_preview", "dataframe_sql"],
            },
            preview={"columns": ["区域公司", "日_日提货", "月累计_月累计提货"]},
            content_ref="asset:14",
            capabilities=["asset_inspect", "dataframe_preview", "dataframe_sql"],
            metadata={"source_tool": "asset_list", "inspection_status": "listed"},
        )
    ]


def test_asset_analyze_observation_satisfies_answer_and_asset_evidence() -> None:
    goal = GoalInitializer().initialize("北方和东北日提货合计是多少？请给出处")
    analyze_result = ToolResult(
        tool_call_id="tc-analyze",
        tool_name="asset_analyze",
        status="ok",
        output=AssetAnalyzeOutput(
            asset_id=14,
            operation="dataframe_sql",
            columns=["日_日提货"],
            rows=[["15.491928"]],
            raw_row_count=1,
            elapsed_ms=1.0,
            truncated=False,
            query='SELECT SUM("日_日提货") AS "日_日提货" FROM sheet',
            markdown="| 日_日提货 |\n|---|\n| 15.491928 |",
        ),
        latency_ms=0,
    )
    state = {
        "task": goal.original_query,
        "goal_spec": goal,
        "tool_results": [analyze_result],
        "structured_observations": [],
        "answer_candidates": [],
        "evidence_refs": [],
        "satisfied_requirements": [],
        "open_gaps": goal.requirement_ids,
    }

    update = StateReducer().reduce_tool_results(state)

    assert update["satisfied_requirements"] == ["answer", "evidence"]
    assert update["open_gaps"] == []
    assert update["evidence_refs"][0].evidence_id == "asset:14"


def test_checker_keeps_explicit_source_gap_open_for_wrong_computed_asset() -> None:
    goal = GoalInitializer().initialize(
        "在分区域分品牌 石膏板-26年表中，北方和东北当日销售额合计是多少？请给出处"
    )
    wrong_unit = ContextUnit(
        unit_id="asset:13",
        unit_type="table_asset",
        locator={
            "asset_id": 13,
            "asset_type": "table",
            "sheet_name": "模板（套公式） -龙骨",
        },
    )
    analyze_result = ToolResult(
        tool_call_id="tc-analyze",
        tool_name="asset_analyze",
        status="ok",
        output=AssetAnalyzeOutput(
            asset_id=13,
            asset_type="table",
            sheet_name="模板（套公式） -龙骨",
            operation="dataframe_sql",
            columns=["日销售额"],
            rows=[["53.56971238938054"]],
            raw_row_count=1,
            elapsed_ms=1.0,
            truncated=False,
            query='SELECT SUM("日销售额") AS "日销售额" FROM sheet',
            markdown="| 日销售额 |\n|---|\n| 53.56971238938054 |",
        ),
        latency_ms=0,
    )
    state = {
        "task": goal.original_query,
        "goal_spec": goal,
        "tool_results": [analyze_result],
        "structured_observations": [],
        "context_units": [wrong_unit],
        "answer_candidates": [],
        "evidence_refs": [],
        "context_bindings": [],
        "satisfied_requirements": [],
        "open_gaps": goal.open_gaps(),
        "conflicts": [],
        "no_progress_count": 0,
        "pending_tool_calls": [],
    }
    state.update(StateReducer().reduce_tool_results(state))
    assessor = ContextBindingAssessor(providers=[AssetContextBindingProvider()])
    state["context_bindings"] = assessor.assess_bindings(state)

    report = SatisfactionChecker().check(state)

    assert report.is_done is False
    assert any(gap.gap_type == "context_binding" for gap in report.open_gaps)
    assert report.conflicts[0].conflict_id == "constraint:context-title-1:asset:13"


def test_text_search_observation_creates_context_unit_without_answer_candidate() -> None:
    goal = GoalInitializer().initialize("总结政策影响并给出处")
    result = ToolResult(
        tool_call_id="tc-text",
        tool_name="vector_search",
        status="ok",
        output=SearchOutput(
            items=[
                {
                    "text": "该政策降低了准入门槛。",
                    "doc_id": 8,
                    "section_id": 3,
                    "record_type": "chunk",
                    "citation_anchor": "policy#3",
                    "evidence_id": "ev-policy-3",
                    "score": 0.87,
                }
            ]
        ),
        latency_ms=0,
    )
    state = {
        "task": goal.original_query,
        "goal_spec": goal,
        "tool_results": [result],
        "structured_observations": [],
        "context_units": [],
        "answer_candidates": [],
        "evidence_refs": [],
        "satisfied_requirements": [],
        "open_gaps": goal.open_gaps(),
    }

    update = StateReducer().reduce_tool_results(state)

    [unit] = update["context_units"]
    assert unit.unit_type == "retrieved_chunk"
    assert unit.preview == "该政策降低了准入门槛。"
    assert unit.evidence_refs[0].evidence_id == "ev-policy-3"
    assert update["answer_candidates"] == []
    assert update["open_gaps"] == [
        GoalGap(gap_id="answer", gap_type="answer", description="Produce an answer."),
    ]


def test_satisfaction_checker_reports_gaps_without_choosing_asset_action() -> None:
    goal = GoalInitializer().initialize("北方和东北日提货合计是多少？请给出处")
    state = {
        "task": goal.original_query,
        "goal_spec": goal,
        "pending_tool_calls": [],
        "tool_results": [],
        "answer_candidates": [],
        "evidence_refs": [],
        "conflicts": [],
        "no_progress_count": 0,
        "context_units": [
            ContextUnit(
                unit_id="section:6",
                unit_type="document_section",
                locator={"doc_id": 2, "source_id": 1, "section_id": 6},
                preview="## Sheet [ASSET_ANCHOR:sheet-3-table]",
            )
        ],
    }

    report = SatisfactionChecker().check(state)

    assert [gap.gap_id for gap in report.open_gaps] == ["answer", "evidence"]
    assert not hasattr(report, "deterministic_next_action")


def test_controller_defers_asset_action_to_model_decision() -> None:
    config = AgentRunConfig(
        run_id="goal-runtime-provider",
        thread_id="goal-runtime-provider",
        budget_total=1000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )
    RuntimeRegistry.remove(config.run_id)
    RuntimeRegistry.get_or_create(config)
    goal = GoalInitializer().initialize("定位表格并给出处")
    state = {
        "run_config": config,
        "status": "running",
        "task": goal.original_query,
        "goal_spec": goal,
        "pending_tool_calls": [],
        "tool_results": [],
        "answer_candidates": [],
        "evidence_refs": [],
        "context_units": [
            ContextUnit(
                unit_id="retrieval:sheet-3",
                unit_type="document_section",
                locator={"doc_id": 2, "source_id": 1, "section_id": 6},
                preview="## Sheet [ASSET_ANCHOR:sheet-3-table]",
                capabilities=["asset_list"],
            )
        ],
        "conflicts": [],
        "no_progress_count": 0,
    }

    update = controller_node(
        state,  # type: ignore[arg-type]
        definition=RESEARCH_AGENT,
        has_tool_decision_provider=True,
    )

    assert update["controller_next"] == "llm_decide"
    assert "tool_action_proposals" not in update
    assert "pending_tool_calls" not in update
    RuntimeRegistry.remove(config.run_id)


def test_controller_clears_replaced_source_binding_conflict_after_correct_result() -> None:
    config = AgentRunConfig(
        run_id="goal-runtime-binding-recovery",
        thread_id="goal-runtime-binding-recovery",
        budget_total=1000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )
    RuntimeRegistry.remove(config.run_id)
    RuntimeRegistry.get_or_create(config)
    goal = GoalInitializer().initialize(
        "在分区域分品牌 石膏板-26年表中，北方和东北当日销售额合计是多少？请给出处"
    )
    base = {
        "run_config": config,
        "status": "running",
        "task": goal.original_query,
        "goal_spec": goal,
        "pending_tool_calls": [],
        "tool_results": [],
        "no_progress_count": 0,
    }
    wrong_update = controller_node(
        {
            **base,
            "answer_candidates": [
                AnswerCandidate(text="53.57", evidence_refs=[EvidenceRef(evidence_id="asset:13")])
            ],
            "evidence_refs": [EvidenceRef(evidence_id="asset:13")],
            "context_units": [
                ContextUnit(
                    unit_id="asset:13",
                    unit_type="table_asset",
                    locator={"asset_id": 13, "sheet_name": "模板（套公式） -龙骨"},
                )
            ],
            "context_bindings": [],
            "conflicts": [],
        },  # type: ignore[arg-type]
        definition=RESEARCH_AGENT,
        has_tool_decision_provider=False,
    )
    assert wrong_update["conflicts"]

    recovered_update = controller_node(
        {
            **base,
            "answer_candidates": [
                AnswerCandidate(text="131.18", evidence_refs=[EvidenceRef(evidence_id="asset:15")])
            ],
            "evidence_refs": [EvidenceRef(evidence_id="asset:15")],
            "context_units": [
                ContextUnit(
                    unit_id="asset:15",
                    unit_type="table_asset",
                    locator={"asset_id": 15, "sheet_name": "分区域分品牌 石膏板-26年"},
                )
            ],
            "context_bindings": wrong_update["context_bindings"],
            "conflicts": wrong_update["conflicts"],
        },  # type: ignore[arg-type]
        definition=RESEARCH_AGENT,
        has_tool_decision_provider=False,
    )

    assert recovered_update["stop_reason"] == "goal_satisfied"
    assert recovered_update["conflicts"] == []
    RuntimeRegistry.remove(config.run_id)
