from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from rag.agent.core.checkpointing import (
    aclose_agent_checkpointer,
    agent_checkpoint_serde,
    create_agent_checkpointer,
)
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.runtime_diagnostics import AgentLatencyProfile, ToolCallMetrics
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import PendingToolCall, create_loop_state
from rag.agent.memory.models import ExternalizedToolOutput, MemoryRef
from rag.agent.planning import AgentPlan, PlanStep
from rag.agent.tools.tool import ToolContentBlock, ToolResult
from rag.schema.runtime import AccessPolicy


def _round_trip_without_warning(
    value: object,
    caplog: pytest.LogCaptureFixture,
) -> object:
    serde = agent_checkpoint_serde()
    with caplog.at_level(
        logging.WARNING,
        logger="langgraph.checkpoint.serde.jsonplus",
    ):
        restored = serde.loads_typed(serde.dumps_typed(value))
    assert "Deserializing unregistered type" not in caplog.text
    return restored


def test_checkpoint_serde_restores_canonical_tool_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = ToolResult(
        tool_call_id="tc-list",
        tool_name="list_files",
        content=(ToolContentBlock(type="text", data={"text": "sales.csv"}),),
        structured_content={
            "entries": [
                {
                    "name": "sales.csv",
                    "path": "input_files/sales.csv",
                    "size": 26,
                    "is_dir": False,
                }
            ]
        },
        metadata={"latency_ms": 0.5},
    )

    restored = _round_trip_without_warning(result, caplog)

    assert isinstance(restored, ToolResult)
    assert restored.content == result.content
    assert restored.structured_content == result.structured_content
    assert restored.metadata == result.metadata


def test_checkpoint_serde_restores_agent_plan(
    caplog: pytest.LogCaptureFixture,
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

    restored = _round_trip_without_warning(plan, caplog)

    assert isinstance(restored, AgentPlan)
    assert restored == plan


def test_checkpoint_serde_restores_externalized_output_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    output = ExternalizedToolOutput(
        original_output_model="rag.agent.primitive_ops.RunCommandOutput",
        summary="run_command ok=True exit_code=0 stdout_preview=large",
        ref=MemoryRef(
            ref_id="mem_abc",
            path=".agent_memory/records/mem_abc.json",
            summary="run_command ok=True exit_code=0",
            source_tool_call_id="tc-big",
            source_tool_name="run_command",
            size_bytes=1024,
        ),
        status="available",
    )

    restored = _round_trip_without_warning(output, caplog)

    assert isinstance(restored, ExternalizedToolOutput)
    assert restored.ref.ref_id == "mem_abc"


@pytest.mark.parametrize(
    "value",
    [
        ToolCallMetrics(native_calls=2, native_latency_ms_total=15.5),
        AgentLatencyProfile(total_ms=12.5, model_latency_ms=4.0),
    ],
)
def test_checkpoint_serde_restores_runtime_diagnostics(
    value: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    restored = _round_trip_without_warning(value, caplog)

    assert restored == value


def test_sqlite_checkpointer_constructs_without_running_event_loop(
    tmp_path: Path,
) -> None:
    checkpointer = create_agent_checkpointer(
        tmp_path / "agent-checkpoints.sqlite"
    )

    assert checkpointer is not None
    asyncio.run(aclose_agent_checkpointer(checkpointer))


def test_sqlite_checkpointer_deletes_one_turn_thread(tmp_path: Path) -> None:
    async def exercise() -> None:
        checkpointer = create_agent_checkpointer(
            tmp_path / "agent-checkpoints.sqlite"
        )
        config = {"configurable": {"thread_id": "turn-delete", "checkpoint_ns": ""}}
        saved = await checkpointer.aput(
            config,
            empty_checkpoint(),
            {},
            {},
        )
        assert await checkpointer.aget_tuple(saved) is not None

        await checkpointer.adelete_thread("turn-delete")

        assert await checkpointer.aget_tuple(saved) is None
        await aclose_agent_checkpointer(checkpointer)

    asyncio.run(exercise())


def test_sqlite_checkpointer_delete_is_safe_before_first_checkpoint(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        checkpointer = create_agent_checkpointer(
            tmp_path / "agent-checkpoints.sqlite"
        )

        await checkpointer.adelete_thread("missing-turn")

        await aclose_agent_checkpointer(checkpointer)

    asyncio.run(exercise())


def test_live_loop_state_serde_preserves_final_tool_result() -> None:
    serde = agent_checkpoint_serde()
    plan = ToolCallPlan.create(
        "search_knowledge",
        {"query": "What is the capital of France?"},
    )
    config = AgentRunConfig(
        run_id="test-serde-live",
        thread_id="test-serde-live",
        llm_budget_total=100,
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )
    result = ToolResult(
        tool_call_id=plan.tool_call_id,
        tool_name="search_knowledge",
        structured_content={
            "answer_text": "Paris is the capital of France.",
            "results": [
                {
                    "evidence_id": "ev_1",
                    "doc_id": 1,
                    "citation_anchor": "paris-capital",
                    "text": "Paris is the capital of France.",
                    "score": 0.95,
                    "source_type": "section",
                    "file_name": "france.pdf",
                }
            ],
            "citations": ["paris-capital"],
            "groundedness_flag": True,
            "insufficient_evidence": False,
            "total_found": 1,
        },
    )
    state = create_loop_state(
        task="What is the capital of France?",
        run_config=config,
    )
    state["pending_tool_calls"] = [
        PendingToolCall(
            plan=plan,
            status="completed",
            summary="Paris is the capital of France.",
        )
    ]
    state["tool_call_ledger"].append_plans([plan], turn=1)
    state["tool_results"] = [result]

    restored = serde.loads_typed(serde.dumps_typed(state))

    assert "evidence" not in restored
    assert "citations" not in restored
    restored_result = restored["tool_results"][0]
    assert isinstance(restored_result, ToolResult)
    assert restored_result.structured_content["results"][0]["evidence_id"] == (
        "ev_1"
    )
    assert len(restored["tool_call_ledger"].entries) == 1
    assert restored["pending_tool_calls"][0].status == "completed"
