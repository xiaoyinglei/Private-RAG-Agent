from __future__ import annotations

import logging

from rag.agent.core.checkpointing import agent_checkpoint_serde
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import PendingToolCall, create_loop_state
from rag.agent.memory.models import ExternalizedToolOutput, MemoryRef
from rag.agent.planning import AgentPlan, PlanStep
from rag.agent.primitive_ops import FileInfo, ListFilesOutput
from rag.agent.tools.rag_answer_tools import RAGSearchAnswerOutput
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy


def test_agent_checkpoint_serde_restores_tool_result_without_unregistered_warning(
    caplog,
) -> None:
    result = ToolResult(
        tool_call_id="tc-list",
        tool_name="list_files",
        status="ok",
        output=ListFilesOutput(
            files=[
                FileInfo(
                    name="sales.csv",
                    path="input_files/sales.csv",
                    size=26,
                    is_dir=False,
                    modified_at=1.0,
                )
            ]
        ),
        latency_ms=0,
    )
    serde = agent_checkpoint_serde()

    with caplog.at_level(
        logging.WARNING,
        logger="langgraph.checkpoint.serde.jsonplus",
    ):
        restored = serde.loads_typed(serde.dumps_typed(result))

    assert isinstance(restored, ToolResult)
    assert restored.output == result.output
    assert "Deserializing unregistered type" not in caplog.text


def test_agent_checkpoint_serde_restores_agent_plan_without_unregistered_warning(
    caplog,
) -> None:
    plan = AgentPlan(
        objective="Answer with a bounded plan.",
        active_step_id="step_answer",
        steps=[
            PlanStep(
                step_id="step_answer",
                title="Produce the final answer",
                status="in_progress",
            )
        ],
    )
    serde = agent_checkpoint_serde()

    with caplog.at_level(
        logging.WARNING,
        logger="langgraph.checkpoint.serde.jsonplus",
    ):
        restored = serde.loads_typed(serde.dumps_typed(plan))

    assert isinstance(restored, AgentPlan)
    assert restored == plan
    assert "Deserializing unregistered type" not in caplog.text


def test_agent_checkpoint_serde_restores_externalized_tool_output_without_unregistered_warning(
    caplog,
) -> None:
    result = ToolResult(
        tool_call_id="tc-big",
        tool_name="run_python",
        status="ok",
        output=ExternalizedToolOutput(
            original_output_model="rag.agent.primitive_ops.RunPythonOutput",
            summary="run_python ok=True exit_code=0 stdout_preview=large",
            ref=MemoryRef(
                ref_id="mem_abc",
                path=".agent_memory/records/mem_abc.json",
                summary="run_python ok=True exit_code=0",
                source_tool_call_id="tc-big",
                source_tool_name="run_python",
                size_bytes=1024,
            ),
            status="available",
        ),
        latency_ms=0,
    )
    serde = agent_checkpoint_serde()

    with caplog.at_level(
        logging.WARNING,
        logger="langgraph.checkpoint.serde.jsonplus",
    ):
        restored = serde.loads_typed(serde.dumps_typed(result))

    assert isinstance(restored, ToolResult)
    assert isinstance(restored.output, ExternalizedToolOutput)
    assert restored.output.ref.ref_id == "mem_abc"
    assert "Deserializing unregistered type" not in caplog.text


def test_live_loop_state_serde_after_pr3_cleanup() -> None:
    """Live PR3 LoopState payload must serialize without deprecated state fields."""
    serde = agent_checkpoint_serde()

    plan = ToolCallPlan.create("rag_search_answer", {"query": "What is the capital of France?"})

    config = AgentRunConfig(
        run_id="test-serde-live",
        thread_id="test-serde-live",
        llm_budget_total=100,
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )

    result = ToolResult(
        tool_call_id=plan.tool_call_id,
        tool_name="rag_search_answer",
        status="ok",
        output=RAGSearchAnswerOutput(
            text="Paris is the capital of France.",
            evidence=[
                EvidenceItem(
                    evidence_id="ev_1",
                    doc_id=1,
                    citation_anchor="paris-capital",
                    text="Paris is the capital of France.",
                    score=0.95,
                ),
            ],
            citations=[
                AnswerCitation(
                    citation_id="cit_1",
                    evidence_id="ev_1",
                    record_type="section",
                    citation_anchor="paris-capital",
                    file_name="france_overview.pdf",
                ),
            ],
        ),
        latency_ms=0,
    )

    state = create_loop_state(task="What is the capital of France?", run_config=config)
    state["pending_tool_calls"] = [
        PendingToolCall(plan=plan, status="completed", summary="Paris is the capital of France."),
    ]
    state["tool_call_ledger"].append_plans([plan], turn=1)
    state["tool_results"] = [result]

    restored = serde.loads_typed(serde.dumps_typed(state))

    # Deprecated flat fields must not appear
    assert "evidence" not in restored
    assert "citations" not in restored

    # ToolResult output must survive with typed EvidenceItem inside
    assert restored["tool_results"][0].output.evidence[0].evidence_id == "ev_1"
    assert restored["tool_results"][0].output.citations[0].citation_id == "cit_1"

    # Ledger entries must survive
    assert len(restored["tool_call_ledger"].entries) == 1

    # PendingToolCall must survive
    assert restored["pending_tool_calls"][0].tool_call_id == plan.tool_call_id
    assert restored["pending_tool_calls"][0].status == "completed"
