from __future__ import annotations

import logging

from rag.agent.core.checkpointing import agent_checkpoint_serde
from rag.agent.memory.models import ExternalizedToolOutput, MemoryRef
from rag.agent.planning import AgentPlan, PlanStep
from rag.agent.primitive_ops import FileInfo, ListFilesOutput
from rag.agent.tools.spec import ToolResult


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
