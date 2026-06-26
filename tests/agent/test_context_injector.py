from __future__ import annotations

from langchain_core.messages import HumanMessage

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.loop.state import LoopState, create_loop_state
from rag.agent.memory.injector import ContextBuilder
from rag.agent.memory.models import ExternalizedToolOutput, ExtractedFact, MemoryRef, WorkingSummary
from rag.agent.planning import AgentPlan, PlanStep
from rag.agent.state import ToolCallPlan
from rag.agent.tools.llm_tools import LLMTextOutput
from rag.agent.tools.spec import ToolError, ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy


class _CharacterTokenAccounting:
    def count(self, text: str) -> int:
        return len(text)

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str:
        clipped = text[: max(token_budget, 0)]
        if add_ellipsis and len(clipped) < len(text) and token_budget >= 4:
            return clipped[: token_budget - 4].rstrip() + " ..."
        return clipped


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.from_legacy(
        agent_type="research",
        description="Research agent",
        system_prompt="System prompt",
        allowed_tools=["search"],
    )


def _state() -> LoopState:
    state = create_loop_state(
        task="Explain policy",
        run_config=AgentRunConfig(
            run_id="ctx",
            thread_id="ctx",
            budget_total=1000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
        ),
        messages=[HumanMessage(content="recent tail", id="h-tail")],
    )
    state["evidence"] = [
        EvidenceItem(
            evidence_id="ev1",
            doc_id=1,
            citation_anchor="doc#1",
            text="Authoritative evidence text",
            score=0.91,
            record_type="section",
        )
    ]
    state["citations"] = [
        AnswerCitation(
            citation_id="cit1",
            evidence_id="ev1",
            record_type="section",
            citation_anchor="doc#1",
        )
    ]
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc1",
            tool_name="search",
            status="error",
            error=ToolError(
                code="tool_not_implemented",
                message="not wired",
                retryable=False,
            ),
            latency_ms=0,
        )
    ]
    state["working_summary"] = WorkingSummary(
        summary="Prior working summary",
        covered_message_ids=["h1"],
        updated_at="2026-05-08T00:00:00Z",
        token_count=3,
    )
    state["extracted_facts"] = [
        ExtractedFact(
            fact_id="f1",
            text="Memory fact",
            evidence_ids=["ev1"],
        ),
    ]
    return state


def test_context_sections_follow_spec_order() -> None:
    context = ContextBuilder(max_context_tokens=1000).assemble_loop(
        definition=_definition(),
        state=_state(),
    )

    names = [section.name for section in context.sections]
    # Evidence section removed in PR2 — evidence data now lives in tool_results via formatters
    assert "system" in names
    assert "task" in names
    assert "tool_results" in names
    # ev1 still appears via extracted_facts in working_memory
    rendered = context.as_text()
    assert "ev1" in rendered
    assert "tool_call_id=tc1" in rendered
    assert "error_code=tool_not_implemented" in rendered


def test_historical_hints_are_marked_non_authoritative() -> None:
    context = ContextBuilder(max_context_tokens=1000).assemble_loop(
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

    context = ContextBuilder(max_context_tokens=18).assemble_loop(
        definition=_definition(),
        state=state,
    )

    names = [section.name for section in context.sections]
    assert "system" in names
    assert "task" in names
    # Evidence section removed in PR2 — no longer a separate section
    assert "message_tail" not in names
    assert context.context_budget.overflow is True
    assert "message_tail" in context.context_budget.dropped_sections


def test_budget_priority_keeps_pending_decisions_before_memory_and_tail() -> None:
    state = _state()
    state["messages"] = [HumanMessage(content="tail " * 200, id="h-tail")]
    from rag.agent.loop.state import PendingToolCall

    state["pending_tool_calls"] = [
        PendingToolCall(
            plan=ToolCallPlan.create("search", {"query": "policy"}),
            status="pending",
        )
    ]
    state["memory_refs"] = [
        MemoryRef(
            ref_id=f"mem_{index}",
            path=f".agent_memory/records/mem_{index}.json",
            summary="large run output",
            source_tool_call_id=f"tc-{index}",
            source_tool_name="run_python",
            size_bytes=5000,
        )
        for index in range(8)
    ]

    context = ContextBuilder(max_context_tokens=350).assemble_loop(
        definition=_definition(),
        state=state,
    )

    names = [section.name for section in context.sections]
    assert "open_decisions" in names
    assert "memory" in names
    if "message_tail" in names:
        assert "message_tail" in context.context_budget.summarized_sections
        assert "tail tail tail" in context.section("message_tail").content
        assert "sha256=" not in context.section("message_tail").content
    else:
        assert "message_tail" in context.context_budget.dropped_sections
    assert names.index("open_decisions") < names.index("memory")
    assert context.context_budget.memory_ref_count == 8


def test_context_injects_memory_summaries_and_refs_not_raw_payload() -> None:
    state = _state()
    state["structured_observations"] = []
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc-big",
            tool_name="run_python",
            status="ok",
            output=ExternalizedToolOutput(
                original_output_model="rag.agent.primitive_ops.RunPythonOutput",
                summary="run_python ok=True exit_code=0 stdout_preview=total=42",
                ref=MemoryRef(
                    ref_id="mem_big",
                    path=".agent_memory/records/mem_big.json",
                    summary="run_python ok=True exit_code=0 stdout_preview=total=42",
                    source_tool_call_id="tc-big",
                    source_tool_name="run_python",
                    size_bytes=9999,
                ),
            ),
            latency_ms=0,
        )
    ]
    state["memory_refs"] = [state["tool_results"][0].output.ref]  # type: ignore[union-attr]

    context = ContextBuilder(max_context_tokens=1000).assemble_loop(
        definition=_definition(),
        state=state,
    )

    memory_context = context.section("memory").content
    tool_context = context.section("tool_results").content
    assert "mem_big" in memory_context
    assert "total=42" in memory_context
    assert ".agent_memory/records/mem_big.json" not in memory_context
    assert "raw stdout" not in memory_context
    assert "ExternalizedToolOutput" not in tool_context
    assert "mem_big" in tool_context


def test_context_budget_snapshot_counts_sections() -> None:
    context = ContextBuilder(max_context_tokens=1000).assemble_loop(
        definition=_definition(),
        state=_state(),
    )

    budget = context.context_budget
    assert budget.max_context_tokens == 1000
    assert budget.system_tokens > 0
    # Evidence section removed in PR2 — evidence_tokens may be 0
    assert budget.working_memory_tokens > 0
    assert budget.message_tail_tokens > 0
    assert budget.tool_result_tokens > 0


def test_context_builder_uses_injected_model_token_accounting() -> None:
    accounting = _CharacterTokenAccounting()
    state = _state()
    state["evidence"] = []
    state["citations"] = []
    state["tool_results"] = []
    state["messages"] = []

    context = ContextBuilder(
        max_context_tokens=500,
        token_accounting=accounting,
    ).assemble_loop(
        definition=_definition(),
        state=state,
    )

    assert context.context_budget.used_context_tokens == accounting.count(context.as_text())
    assert context.section("system").token_count == accounting.count("[system]\nSystem prompt")


def test_required_section_overflow_never_replaces_real_content_with_hash() -> None:
    state = _state()
    state["evidence"] = []
    state["citations"] = []
    state["tool_results"] = []
    state["messages"] = []
    definition = AgentRuntimePolicy.from_legacy(
        agent_type="research",
        description="Research agent",
        system_prompt="SYSTEM_REAL_CONTENT",
        allowed_tools=[],
    )

    context = ContextBuilder(
        max_context_tokens=20,
        token_accounting=_CharacterTokenAccounting(),
        max_section_chars=10_000,
    ).assemble_loop(
        definition=definition,
        state=state,
    )

    assert context.context_budget.overflow is True
    assert "system" in context.context_budget.required_truncated
    assert all("sha256=" not in section.content for section in context.sections)
    assert all("system: compact" not in section.content for section in context.sections)
    if "system" in [section.name for section in context.sections]:
        assert context.section("system").content == "SYSTEM_REAL_CONTENT"


def test_optional_section_is_real_text_clipped_or_dropped() -> None:
    state = _state()
    state["evidence"] = []
    state["citations"] = []
    state["tool_results"] = []
    state["messages"] = [HumanMessage(content="OPTIONAL_REAL_TEXT " * 30, id="tail")]

    context = ContextBuilder(
        max_context_tokens=120,
        token_accounting=_CharacterTokenAccounting(),
        max_section_chars=10_000,
    ).assemble_loop(
        definition=_definition(),
        state=state,
    )

    names = [section.name for section in context.sections]
    if "message_tail" in names:
        content = context.section("message_tail").content
        assert "OPTIONAL_REAL_TEXT" in content
        assert "sha256=" not in content
        assert "message_tail: compact" not in content
    else:
        assert "message_tail" in context.context_budget.dropped_sections


def test_context_hard_budget_compacts_required_sections_without_overrun() -> None:
    state = _state()
    state["task"] = "TASK_RAW " * 400
    definition = AgentRuntimePolicy.from_legacy(
        agent_type="research",
        description="Research agent",
        system_prompt="SYSTEM_RAW " * 400,
        allowed_tools=["search"],
    )

    context = ContextBuilder(max_context_tokens=40, max_section_chars=10_000).assemble_loop(
        definition=definition,
        state=state,
    )

    assert sum(section.token_count for section in context.sections) <= 40
    assert context.context_budget.used_context_tokens <= 40
    assert context.context_budget.degraded is True
    assert "system" in context.context_budget.required_truncated
    assert "task" in context.context_budget.required_truncated


def test_context_overflow_marks_budget_when_minimal_snapshot_cannot_fit() -> None:
    state = _state()
    state["task"] = "irreducible task"
    definition = AgentRuntimePolicy.from_legacy(
        agent_type="research",
        description="Research agent",
        system_prompt="irreducible system",
        allowed_tools=["search"],
    )

    context = ContextBuilder(max_context_tokens=1, max_section_chars=10_000).assemble_loop(
        definition=definition,
        state=state,
    )

    assert context.context_budget.overflow is True
    assert "context_overflow" in context.context_budget.warnings
    assert sum(section.token_count for section in context.sections) <= 1


def test_context_renders_tool_results_via_fallback() -> None:
    """PR2: tool_results are rendered via fallback when no formatter is registered."""
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc-big",
            tool_name="asset_analyze",
            status="ok",
            output=LLMTextOutput(text="RAW_RESULT " * 10),
            latency_ms=0,
        )
    ]
    # structured_observations are no longer rendered by ContextBuilder (PR2)

    context = ContextBuilder(max_context_tokens=8000).assemble_loop(
        definition=_definition(),
        state=state,
    )

    tool_context = context.section("tool_results").content
    assert "RAW_RESULT" in tool_context


def test_context_includes_bounded_agent_plan_without_raw_scratchpad() -> None:
    state = _state()
    state["agent_plan"] = AgentPlan(
        objective="Analyze the spreadsheet and answer with evidence.",
        active_step_id="step_probe",
        steps=[
            PlanStep(
                step_id="step_probe",
                title="Probe workbook structure",
                status="in_progress",
                expected_tool_names=["structured_probe"],
                notes="Keep this bounded. " + ("RAW_SCRATCHPAD " * 200),
            )
        ],
        summary="Need structure before computation.",
    )

    context = ContextBuilder(max_context_tokens=1000).assemble_loop(
        definition=_definition(),
        state=state,
    )

    plan_context = context.section("plan").content
    assert "Current autonomous plan" in plan_context
    assert "active_step_id=step_probe" in plan_context
    assert "structured_probe" in plan_context
    assert plan_context.count("RAW_SCRATCHPAD") < 20


def test_context_formats_asset_locators_compactly() -> None:
    """PR2: asset locator data rendered by formatters instead of ContextBuilder."""
    from rag.agent.tools.formatters.rag_retrieval import VectorSearchFormatter
    from rag.agent.tools.rag_tools import SearchOutput

    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc-assets",
            tool_name="vector_search",
            status="ok",
            output=SearchOutput(
                items=[
                    {
                        "asset_id": asset_id,
                        "doc_id": 2,
                        "source_id": 1,
                        "section_id": asset_id - 8,
                        "asset_type": "table",
                        "sheet_name": sheet_name,
                        "columns": ["区域公司", "日_日提货", "月累计_月累计提货"],
                        "analysis_capabilities": ["dataframe_preview", "dataframe_sql"],
                        "score": 0.9,
                        "text": "SAMPLE_ROW_SHOULD_NOT_ENTER_CONTEXT" * 100,
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
                ]
            ),
            latency_ms=10.0,
        )
    ]

    context = ContextBuilder(
        max_context_tokens=8000,
        formatter_resolver=lambda name: VectorSearchFormatter() if name == "vector_search" else None,
    ).assemble_loop(
        definition=_definition(),
        state=state,
    )

    tool_context = context.section("tool_results").content
    assert "asset_id=14" in tool_context
    assert "sheet_name=2024-0317新增" in tool_context
    assert "asset_id=18" in tool_context


def test_context_formats_workspace_file_observations() -> None:
    """PR2: file observations rendered by ListFilesFormatter instead of ContextBuilder."""
    from rag.agent.primitive_ops import FileInfo, ListFilesOutput
    from rag.agent.tools.formatters.file_tools import ListFilesFormatter

    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc-list",
            tool_name="list_files",
            status="ok",
            output=ListFilesOutput(
                files=[
                    FileInfo(
                        name="sales.csv",
                        path="input_files/sales.csv",
                        size=24,
                        is_dir=False,
                        modified_at=1700000000.0,
                        mime_type="text/csv",
                        file_kind="text",
                        is_binary=False,
                        readable_as_text=True,
                        capabilities=["read_file"],
                    ),
                ]
            ),
            latency_ms=5.0,
        )
    ]

    context = ContextBuilder(
        max_context_tokens=8000,
        formatter_resolver=lambda name: ListFilesFormatter() if name == "list_files" else None,
    ).assemble_loop(
        definition=_definition(),
        state=state,
    )

    tool_context = context.section("tool_results").content
    assert "list_files results" in tool_context
    assert "input_files/sales.csv" in tool_context


def test_context_preserves_workspace_path_spacing() -> None:
    """PR2: path spacing preserved by ListFilesFormatter."""
    from rag.agent.primitive_ops import FileInfo, ListFilesOutput
    from rag.agent.tools.formatters.file_tools import ListFilesFormatter

    state = _state()
    path = "input_files/2026年石膏板分城市销售情况对标  区域双周会.xlsx"
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc-list",
            tool_name="list_files",
            status="ok",
            output=ListFilesOutput(
                files=[
                    FileInfo(
                        name="2026年石膏板分城市销售情况对标  区域双周会.xlsx",
                        path=path,
                        size=202943,
                        is_dir=False,
                        modified_at=1700000000.0,
                        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        file_kind="text",
                        is_binary=False,
                        readable_as_text=True,
                        capabilities=["read_file"],
                    ),
                ]
            ),
            latency_ms=5.0,
        )
    ]

    context = ContextBuilder(
        max_context_tokens=8000,
        formatter_resolver=lambda name: ListFilesFormatter() if name == "list_files" else None,
    ).assemble_loop(
        definition=_definition(),
        state=state,
    )

    tool_context = context.section("tool_results").content
    assert path in tool_context
    assert "对标" in tool_context
