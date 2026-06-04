from __future__ import annotations

import logging

from rag.agent.core.checkpointing import agent_checkpoint_serde
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
