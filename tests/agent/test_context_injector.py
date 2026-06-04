from __future__ import annotations

from langchain_core.messages import HumanMessage

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentDefinition
from rag.agent.goal_runtime import AnswerCandidate, ContextUnit, EvidenceRef, GoalSpec, StructuredObservation
from rag.agent.memory.injector import ContextInjector
from rag.agent.memory.models import ExtractedFact, WorkingSummary
from rag.agent.state import AgentState
from rag.agent.tools.llm_tools import LLMTextOutput
from rag.agent.tools.spec import ToolError, ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy


def _definition() -> AgentDefinition:
    return AgentDefinition(
        agent_type="research",
        description="Research agent",
        system_prompt="System prompt",
        allowed_tools=["search"],
    )


def _state() -> AgentState:
    return {
        "messages": [HumanMessage(content="recent tail", id="h-tail")],
        "evidence": [
            EvidenceItem(
                evidence_id="ev1",
                doc_id=1,
                citation_anchor="doc#1",
                text="Authoritative evidence text",
                score=0.91,
                record_type="section",
            )
        ],
        "citations": [
            AnswerCitation(
                citation_id="cit1",
                evidence_id="ev1",
                record_type="section",
                citation_anchor="doc#1",
            )
        ],
        "tool_results": [
            ToolResult(
                tool_call_id="tc1",
                tool_name="search",
                status="error",
                error=ToolError(code="tool_not_implemented", message="not wired", retryable=False),
                latency_ms=0,
            )
        ],
        "task": "Explain policy",
        "run_config": AgentRunConfig(
            run_id="ctx",
            thread_id="ctx",
            budget_total=1000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        ),
        "iteration": 0,
        "status": "running",
        "decision_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "working_summary": WorkingSummary(
            summary="Prior working summary",
            covered_message_ids=["h1"],
            updated_at="2026-05-08T00:00:00Z",
            token_count=3,
        ),
        "extracted_facts": [
            ExtractedFact(fact_id="f1", text="Memory fact", evidence_ids=["ev1"]),
        ],
        "context_budget": None,
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
        "goal_spec": None,
        "goal_requirements": [],
        "satisfied_requirements": [],
        "open_gaps": [],
        "evidence_refs": [],
        "answer_candidates": [],
        "computation_results": [],
        "structured_observations": [],
        "locators": [],
        "asset_refs": [],
        "conflicts": [],
        "no_progress_count": 0,
        "satisfaction_report": None,
        "controller_next": None,
    }


def test_context_sections_follow_spec_order() -> None:
    context = ContextInjector(max_context_tokens=1000).assemble(
        definition=_definition(),
        state=_state(),
    )

    names = [section.name for section in context.sections]
    assert names == ["system", "task", "evidence", "working_memory", "message_tail", "tool_results"]
    assert names.index("evidence") < names.index("working_memory")
    assert "ev1" in context.section("evidence").content
    assert "cit1" in context.section("evidence").content


def test_historical_hints_are_marked_non_authoritative() -> None:
    context = ContextInjector(max_context_tokens=1000).assemble(
        definition=_definition(),
        state=_state(),
        recalled_memories=["Old project preference"],
    )

    historical = context.section("historical_hints")
    assert "historical hints, not authoritative evidence" in historical.content
    assert "Old project preference" in historical.content


def test_budget_keeps_evidence_before_tail() -> None:
    state = _state()
    state["messages"] = [HumanMessage(content="tail " * 200, id="h-tail")]

    context = ContextInjector(max_context_tokens=18).assemble(
        definition=_definition(),
        state=state,
    )

    names = [section.name for section in context.sections]
    assert "system" in names
    assert "task" in names
    assert "evidence" in names
    assert "message_tail" not in names
    assert context.context_budget.evidence_tokens > 0


def test_context_budget_snapshot_counts_sections() -> None:
    context = ContextInjector(max_context_tokens=1000).assemble(
        definition=_definition(),
        state=_state(),
    )

    budget = context.context_budget
    assert budget.max_context_tokens == 1000
    assert budget.system_tokens > 0
    assert budget.evidence_tokens > 0
    assert budget.working_memory_tokens > 0
    assert budget.message_tail_tokens > 0
    assert budget.tool_result_tokens > 0


def test_context_uses_structured_observations_instead_of_large_raw_tool_outputs() -> None:
    state = _state()
    state["goal_spec"] = GoalSpec(
        original_query="北方和东北日提货合计是多少？请给出处",
        required_evidence=["citation"],
    )
    state["open_gaps"] = ["evidence"]
    state["satisfied_requirements"] = ["answer"]
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc-big",
            tool_name="asset_analyze",
            status="ok",
            output=LLMTextOutput(text="RAW_RESULT " * 1000),
            latency_ms=0,
        )
    ]
    state["structured_observations"] = [
        StructuredObservation(
            tool_call_id="tc-big",
            tool_name="asset_analyze",
            status="ok",
            answer_candidate=AnswerCandidate(
                text="北方和东北日提货合计为 15.491928。",
                evidence_refs=[EvidenceRef(citation_anchor="table@p4")],
            ),
            evidence_refs=[EvidenceRef(citation_anchor="table@p4")],
            raw_result_ref="tc-big",
            resolved_gaps=["answer"],
        )
    ]

    context = ContextInjector(max_context_tokens=1000).assemble(
        definition=_definition(),
        state=state,
    )

    tool_context = context.section("tool_results").content
    decisions_context = context.section("open_decisions").content
    assert "Structured tool observations" in tool_context
    assert "15.491928" in tool_context
    assert "RAW_RESULT" not in tool_context
    assert "open_gaps: evidence" in decisions_context
    assert "satisfied_requirements: answer" in decisions_context


def test_context_formats_asset_locators_compactly() -> None:
    state = _state()
    state["tool_results"] = []
    state["structured_observations"] = [
        StructuredObservation(
            tool_call_id="tc-assets",
            tool_name="asset_list",
            status="ok",
            locators=[
                {
                    "asset_id": asset_id,
                    "doc_id": 2,
                    "source_id": 1,
                    "section_id": asset_id - 8,
                    "asset_type": "table",
                    "sheet_name": sheet_name,
                    "columns": ["区域公司", "日_日提货", "月累计_月累计提货"],
                    "sample_rows": [{"payload": "SAMPLE_ROW_SHOULD_NOT_ENTER_CONTEXT" * 100}],
                    "analysis_capabilities": ["dataframe_preview", "dataframe_sql"],
                }
                for asset_id, sheet_name in [
                    (11, "日报调整记录"),
                    (12, "模板（套公式）-石膏板"),
                    (13, "模板（套公式） -龙骨"),
                    (14, "2024-0317新增"),
                    (15, "分区域分品牌 石膏板-26年"),
                    (16, "分区域分品牌 轻钢龙骨-26年"),
                    (17, "透视-销售台账 板"),
                    (18, "透视-销售台账 骨"),
                ]
            ],
            raw_result_ref="tc-assets",
        )
    ]

    context = ContextInjector(max_context_tokens=1000).assemble(
        definition=_definition(),
        state=state,
    )

    tool_context = context.section("tool_results").content
    assert "asset_id=14" in tool_context
    assert "sheet_name=2024-0317新增" in tool_context
    assert "日_日提货" in tool_context
    assert "asset_id=18" in tool_context
    assert "SAMPLE_ROW_SHOULD_NOT_ENTER_CONTEXT" not in tool_context


def test_context_formats_workspace_file_observations() -> None:
    state = _state()
    state["tool_results"] = []
    state["structured_observations"] = [
        StructuredObservation(
            tool_call_id="tc-list",
            tool_name="list_files",
            status="ok",
            context_units=[
                ContextUnit(
                    unit_id="workspace_file:input_files/sales.csv",
                    unit_type="workspace_file",
                    locator={
                        "path": "input_files/sales.csv",
                        "name": "sales.csv",
                        "size_bytes": 24,
                        "is_dir": False,
                        "source_tool": "list_files",
                    },
                    preview="input_files/sales.csv (24 bytes)",
                    capabilities=["read_file"],
                )
            ],
            raw_result_ref="tc-list",
        )
    ]

    context = ContextInjector(max_context_tokens=1000).assemble(
        definition=_definition(),
        state=state,
    )

    tool_context = context.section("tool_results").content
    assert "workspace_file:input_files/sales.csv" in tool_context
    assert "path=input_files/sales.csv" in tool_context
    assert "preview: input_files/sales.csv (24 bytes)" in tool_context
    assert "sample_rows" not in tool_context


def test_context_preserves_workspace_path_spacing() -> None:
    state = _state()
    state["tool_results"] = []
    path = "input_files/2026年石膏板分城市销售情况对标  区域双周会.xlsx"
    state["structured_observations"] = [
        StructuredObservation(
            tool_call_id="tc-list",
            tool_name="list_files",
            status="ok",
            context_units=[
                ContextUnit(
                    unit_id=f"workspace_file:{path}",
                    unit_type="workspace_file",
                    locator={
                        "path": path,
                        "name": "2026年石膏板分城市销售情况对标  区域双周会.xlsx",
                        "size_bytes": 202943,
                        "is_dir": False,
                        "source_tool": "list_files",
                    },
                    preview=f"{path} (202943 bytes)",
                    capabilities=["read_file"],
                )
            ],
            raw_result_ref="tc-list",
        )
    ]

    context = ContextInjector(max_context_tokens=1000).assemble(
        definition=_definition(),
        state=state,
    )

    tool_context = context.section("tool_results").content
    assert f"unit_id=workspace_file:{path}" in tool_context
    assert f"path={path}" in tool_context
    assert "对标  区域双周会" in tool_context
