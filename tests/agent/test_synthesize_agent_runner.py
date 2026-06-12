from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.builtin.synthesize import SYNTHESIZE_AGENT
from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.llm_context import AgentLLMContextOverflowError
from rag.agent.core.output_finalizer import OutputValidationExhaustedError
from rag.agent.core.output_models import ValidatedFinalOutput
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.subagent_runner import BuiltinSynthesisRunner
from rag.agent.goal_runtime import (
    AnswerCandidate,
    ComputationResult,
    ContextUnit,
    EvidenceRef,
    StructuredObservation,
)
from rag.agent.graphs.nodes.synthesize import build_answer
from rag.agent.primitive_ops import (
    CandidateHeaderRow,
    StructuredProbeOutput,
    StructuredTableProbe,
)
from rag.agent.service import AgentRunResult
from rag.agent.state import AgentState
from rag.agent.tools.llm_tools import LLMGenerateInput, LLMTextOutput
from rag.agent.tools.rag_answer_tools import RAGSearchAnswerOutput
from rag.agent.tools.spec import ToolResult
from rag.schema.llm import LLMCallStage
from rag.schema.query import AnswerCitation, EvidenceItem, RetrievalSignals
from rag.schema.runtime import AccessPolicy


class _StructuredFinalAnswer(BaseModel):
    answer: str
    confidence: float


def _state() -> AgentState:
    evidence = EvidenceItem(
        evidence_id="ev-child",
        doc_id=1,
        text="Grounded child evidence",
        score=0.9,
        citation_anchor="doc#1",
    )
    citation = AnswerCitation(
        citation_id="cit-child",
        evidence_id="ev-child",
        record_type="section",
        citation_anchor="doc#1",
    )
    run_config = AgentRunConfig(
        run_id="synthesis-parent",
        thread_id="synthesis-parent",
        budget_total=10000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )
    RunRegistry.remove(run_config.run_id)
    RunRegistry.get_or_create(run_config)
    return {
        "messages": [],
        "evidence": [evidence],
        "citations": [citation],
        "tool_results": [],
        "task": "Write final answer",
        "retrieval_signals": RetrievalSignals(),
        "retrieval_signals_debug": None,
        "run_config": run_config,
        "iteration": 0,
        "status": "done",
        "decision_reason": None,
        "stop_reason": "synthesize",
        "needs_user_input": None,
        "pending_tool_calls": [],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "user_message": None,
        "human_input_request": None,
        "human_input_response": None,
        "working_summary": None,
        "extracted_facts": [],
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
        "context_units": [],
        "locators": [],
        "asset_refs": [],
        "conflicts": [],
        "no_progress_count": 0,
        "satisfaction_report": None,
        "controller_next": None,
    }


@pytest.mark.anyio
async def test_build_answer_delegates_to_builtin_synthesize_agent() -> None:
    seen_payloads: list[LLMGenerateInput] = []

    def llm_generate(payload: LLMGenerateInput) -> LLMTextOutput:
        seen_payloads.append(payload)
        return LLMTextOutput(
            text="synthesized answer",
            evidence_ids=payload.evidence_ids,
            citation_ids=payload.citation_ids,
        )

    agent_registry = AgentRegistry()
    agent_registry.register(SYNTHESIZE_AGENT)
    service_factory = AgentServiceFactory(
        tool_registry=create_builtin_tool_registry(runners={"llm_generate": llm_generate}),
        model_registry=None,
    )
    synthesis_runner = BuiltinSynthesisRunner(
        agent_registry=agent_registry,
        service_factory=service_factory,
    )
    service_factory.bind_synthesis_runner(synthesis_runner)

    update = await build_answer(_state(), synthesis_runner=synthesis_runner)

    assert update["status"] == "done"
    assert update["final_answer"] == "synthesized answer"
    assert update["groundedness_flag"] is True
    assert update["insufficient_evidence_flag"] is False
    assert update["tool_results"][0].tool_name == "llm_generate"
    [payload] = seen_payloads
    assert payload.evidence_ids == ["ev-child"]
    assert payload.citation_ids == ["cit-child"]
    assert any("Grounded child evidence" in section for section in payload.context_sections)
    RunRegistry.remove("synthesis-parent")


@pytest.mark.anyio
async def test_build_answer_preserves_grounded_rag_search_answer() -> None:
    called = False

    def llm_generate(payload: LLMGenerateInput) -> LLMTextOutput:
        nonlocal called
        called = True
        return LLMTextOutput(text=f"rewritten: {payload.prompt}")

    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="call-rag",
            tool_name="rag_search_answer",
            status="ok",
            output=RAGSearchAnswerOutput(
                text="日提货总量是131.074462。 [1]",
                evidence=state["evidence"],
                citations=state["citations"],
                groundedness_flag=True,
                insufficient_evidence=False,
            ),
            latency_ms=10.0,
        )
    ]

    agent_registry = AgentRegistry()
    agent_registry.register(SYNTHESIZE_AGENT)
    service_factory = AgentServiceFactory(
        tool_registry=create_builtin_tool_registry(runners={"llm_generate": llm_generate}),
        model_registry=None,
    )
    synthesis_runner = BuiltinSynthesisRunner(
        agent_registry=agent_registry,
        service_factory=service_factory,
    )
    service_factory.bind_synthesis_runner(synthesis_runner)

    update = await build_answer(state, synthesis_runner=synthesis_runner)

    assert update["status"] == "done"
    assert update["final_answer"] == "日提货总量是131.074462。 [1]"
    assert update["groundedness_flag"] is True
    assert update["insufficient_evidence_flag"] is False
    assert called is False
    RunRegistry.remove("synthesis-parent")


@pytest.mark.anyio
async def test_synthesis_context_overflow_pauses_parent_without_fallback_completion() -> None:
    class _PausedSynthesisRunner:
        def run_synthesis(self, *, parent_state: AgentState) -> AgentRunResult:
            del parent_state
            return AgentRunResult(
                run_id="synthesis-child",
                thread_id="synthesis-child",
                status="paused",
                stop_reason="context_overflow",
                needs_user_input=(
                    "Required context does not fit the final synthesis model budget."
                ),
            )

    update = await build_answer(
        _state(),
        synthesis_runner=_PausedSynthesisRunner(),  # type: ignore[arg-type]
    )

    assert update["status"] == "paused"
    assert update["stop_reason"] == "context_overflow"
    assert "final synthesis model budget" in update["needs_user_input"]
    assert update.get("final_answer") is None
    RunRegistry.remove("synthesis-parent")


@pytest.mark.anyio
async def test_output_model_finalization_persists_validated_envelope() -> None:
    class _Finalizer:
        def __init__(self) -> None:
            self.candidates: list[str] = []

        def finalize(
            self,
            *,
            definition: AgentDefinition,
            state: AgentState,
            candidate_text: str,
        ) -> _StructuredFinalAnswer:
            del definition, state
            self.candidates.append(candidate_text)
            return _StructuredFinalAnswer(
                answer="validated structured answer",
                confidence=0.95,
            )

    definition = AgentDefinition(
        agent_type="structured",
        description="Structured",
        system_prompt="Return structured output.",
        allowed_tools=[],
        output_model=_StructuredFinalAnswer,
    )
    state = _state()
    state["answer_candidates"] = [
        AnswerCandidate(text="candidate answer")
    ]
    finalizer = _Finalizer()

    update = await build_answer(
        state,
        definition=definition,
        output_finalizer=finalizer,  # type: ignore[arg-type]
    )

    assert update["status"] == "done"
    assert update["final_answer"] == "validated structured answer"
    assert update["final_output"] == ValidatedFinalOutput(
        model_path=(
            f"{_StructuredFinalAnswer.__module__}."
            f"{_StructuredFinalAnswer.__qualname__}"
        ),
        data={
            "answer": "validated structured answer",
            "confidence": 0.95,
        },
    )
    assert finalizer.candidates == ["candidate answer"]
    RunRegistry.remove("synthesis-parent")


@pytest.mark.anyio
async def test_output_validation_retry_exhaustion_fails_without_final_answer() -> None:
    class _ExhaustedFinalizer:
        def finalize(
            self,
            *,
            definition: AgentDefinition,
            state: AgentState,
            candidate_text: str,
        ) -> BaseModel:
            del definition, state, candidate_text
            raise OutputValidationExhaustedError(
                attempts=3,
                validation_errors=[
                    {
                        "location": ["confidence"],
                        "message": "Field required",
                        "type": "missing",
                    }
                ],
            )

    definition = AgentDefinition(
        agent_type="structured",
        description="Structured",
        system_prompt="Return structured output.",
        allowed_tools=[],
        output_model=_StructuredFinalAnswer,
        output_validation_max_retries=2,
    )

    update = await build_answer(
        _state(),
        definition=definition,
        output_finalizer=_ExhaustedFinalizer(),  # type: ignore[arg-type]
    )

    assert update["status"] == "failed"
    assert update["stop_reason"] == "output_validation_failed"
    assert update["final_output"] is None
    assert update["final_answer"] is None
    assert update.get("controller_next") != "pause"
    RunRegistry.remove("synthesis-parent")


@pytest.mark.anyio
async def test_output_finalizer_context_overflow_remains_paused() -> None:
    class _OverflowFinalizer:
        def finalize(
            self,
            *,
            definition: AgentDefinition,
            state: AgentState,
            candidate_text: str,
        ) -> BaseModel:
            del definition, state, candidate_text
            from rag.agent.memory.models import ContextBudgetSnapshot

            raise AgentLLMContextOverflowError(
                stage=LLMCallStage.FINAL_SYNTHESIS,
                context_budget=ContextBudgetSnapshot(
                    max_context_tokens=10,
                    overflow=True,
                    degraded=True,
                    required_truncated=["task"],
                    warnings=["context_overflow"],
                ),
            )

    definition = AgentDefinition(
        agent_type="structured",
        description="Structured",
        system_prompt="Return structured output.",
        allowed_tools=[],
        output_model=_StructuredFinalAnswer,
    )

    update = await build_answer(
        _state(),
        definition=definition,
        output_finalizer=_OverflowFinalizer(),  # type: ignore[arg-type]
    )

    assert update["status"] == "paused"
    assert update["decision_reason"] == "context_overflow"
    assert update["final_output"] is None
    assert update["final_answer"] is None
    RunRegistry.remove("synthesis-parent")


@pytest.mark.anyio
async def test_synthesis_runner_uses_structured_observations_instead_of_raw_tool_outputs() -> None:
    seen_payloads: list[LLMGenerateInput] = []

    def llm_generate(payload: LLMGenerateInput) -> LLMTextOutput:
        seen_payloads.append(payload)
        return LLMTextOutput(text="final from structured observation")

    state = _state()
    state["evidence"] = []
    state["citations"] = []
    state["answer_candidates"] = [
        AnswerCandidate(
            text="北方和东北日提货合计为 15.491928。",
            evidence_refs=[EvidenceRef(evidence_id="compute_result:14", citation_id="table@p4")],
        )
    ]
    state["evidence_refs"] = [
        EvidenceRef(evidence_id="compute_result:14", citation_id="table@p4")
    ]
    state["structured_observations"] = [
        StructuredObservation(
            tool_call_id="tc-asset",
            tool_name="asset_analyze",
            status="ok",
            answer_candidate=state["answer_candidates"][0],
            evidence_refs=state["evidence_refs"],
            raw_result_ref="tc-asset",
            resolved_gaps=["answer", "evidence"],
        )
    ]
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc-asset",
            tool_name="asset_analyze",
            status="ok",
            output=LLMTextOutput(text="RAW_RESULT " * 1000),
            latency_ms=10.0,
        )
    ]

    agent_registry = AgentRegistry()
    agent_registry.register(SYNTHESIZE_AGENT)
    service_factory = AgentServiceFactory(
        tool_registry=create_builtin_tool_registry(runners={"llm_generate": llm_generate}),
        model_registry=None,
    )
    synthesis_runner = BuiltinSynthesisRunner(
        agent_registry=agent_registry,
        service_factory=service_factory,
    )
    service_factory.bind_synthesis_runner(synthesis_runner)

    update = await build_answer(state, synthesis_runner=synthesis_runner)

    assert update["status"] == "done"
    assert update["final_answer"] == "final from structured observation"
    [payload] = seen_payloads
    context_text = "\n".join(payload.context_sections)
    assert "Structured observations" in context_text
    assert "15.491928" in context_text
    assert "RAW_RESULT" not in context_text
    assert payload.evidence_ids == ["compute_result:14"]
    assert payload.citation_ids == ["table@p4"]
    RunRegistry.remove("synthesis-parent")


@pytest.mark.anyio
async def test_goal_satisfied_asset_analysis_finalizes_without_llm_rewrite() -> None:
    called = False

    def llm_generate(payload: LLMGenerateInput) -> LLMTextOutput:
        nonlocal called
        called = True
        return LLMTextOutput(text=f"hallucinated: {payload.prompt}")

    state = _state()
    state["status"] = "done"
    state["stop_reason"] = "goal_satisfied"
    state["task"] = "北方和东北日提货合计是多少？请给出处"
    state["answer_candidates"] = [
        AnswerCandidate(
            text="北方和东北日提货合计是多少？请给出处：15.491928000000001",
            source_tool_call_id="tc-analyze",
            source_tool_name="asset_analyze",
            evidence_refs=[EvidenceRef(evidence_id="asset:14", source="asset")],
        )
    ]
    state["computation_results"] = [
        ComputationResult(
            source_tool_call_id="tc-analyze",
            source_tool_name="asset_analyze",
            operation="dataframe_sql",
            value_preview="15.491928000000001",
            expression=(
                'SELECT SUM("日_日提货") AS "日_日提货" '
                "FROM sheet WHERE \"区域公司\" IN ('北方', '东北')"
            ),
        )
    ]
    state["context_units"] = [
        ContextUnit(
            unit_id="asset:14",
            unit_type="table_asset",
            locator={
                "asset_id": 14,
                "doc_id": 2,
                "source_id": 1,
                "section_id": 6,
                "sheet_name": "2024-0317新增",
                "page_no": 4,
                "element_ref": "02-日报生成版-20260522-sheet-3-table",
            },
            content_ref="asset:14",
        )
    ]
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc-analyze",
            tool_name="asset_analyze",
            status="ok",
            output=LLMTextOutput(text="RAW_RESULT_MUST_NOT_ENTER_FINAL_ANSWER"),
            latency_ms=10.0,
        )
    ]

    agent_registry = AgentRegistry()
    agent_registry.register(SYNTHESIZE_AGENT)
    service_factory = AgentServiceFactory(
        tool_registry=create_builtin_tool_registry(runners={"llm_generate": llm_generate}),
        model_registry=None,
    )
    synthesis_runner = BuiltinSynthesisRunner(
        agent_registry=agent_registry,
        service_factory=service_factory,
    )

    update = await build_answer(state, synthesis_runner=synthesis_runner)

    assert update["status"] == "done"
    assert "15.491928000000001" in update["final_answer"]
    assert "asset_id=14" in update["final_answer"]
    assert "sheet=2024-0317新增" in update["final_answer"]
    assert "SELECT SUM" in update["final_answer"]
    assert "RAW_RESULT_MUST_NOT_ENTER_FINAL_ANSWER" not in update["final_answer"]
    assert called is False
    RunRegistry.remove("synthesis-parent")


@pytest.mark.anyio
async def test_legacy_synthesis_summarizes_structured_probe_without_raw_json() -> None:
    state = _state()
    state["evidence"] = []
    state["citations"] = []
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc-probe",
            tool_name="structured_probe",
            status="ok",
            output=StructuredProbeOutput(
                path="input_files/report.xlsx",
                file_kind="binary",
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                tables=[
                    StructuredTableProbe(
                        table_index=0,
                        name="Sales",
                        used_range="A1:D5",
                        row_count=5,
                        column_count=4,
                        sample_rows=[
                            ["2026 sales report", None, None, None],
                            ["source: finance team", None, None, None],
                            ["region", "city", "amount\n（万㎡）", "price"],
                            ["north", "beijing", 10, 2.5],
                        ],
                        candidate_header_rows=[
                            CandidateHeaderRow(
                                row_index=3,
                                confidence=0.9,
                                reason="label-like row followed by data rows",
                            )
                        ],
                        data_start_row=4,
                    )
                ],
            ),
            latency_ms=10.0,
        )
    ]

    update = await build_answer(state, synthesis_runner=None)

    assert update["status"] == "done"
    assert update["final_answer"] is not None
    assert "表结构摘要" in update["final_answer"]
    assert "input_files/report.xlsx" in update["final_answer"]
    assert "Sales" in update["final_answer"]
    assert "A1:D5" in update["final_answer"]
    assert "候选表头行：第 3 行" in update["final_answer"]
    assert "数据起始行：第 4 行" in update["final_answer"]
    assert "关键字段：region, city, amount （万㎡）, price" in update["final_answer"]
    assert "{\"path\"" not in update["final_answer"]
    RunRegistry.remove("synthesis-parent")
