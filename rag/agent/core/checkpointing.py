from __future__ import annotations

from pathlib import Path

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

AGENT_CHECKPOINT_MSGPACK_ALLOWLIST: tuple[tuple[str, ...], ...] = (
    ("rag.agent.core.context", "AgentRunConfig"),
    ("rag.agent.core.definition", "ToolPolicy"),
    ("rag.agent.core.human_input", "HumanInputRequest"),
    ("rag.agent.core.human_input", "HumanInputResponse"),
    ("rag.agent.core.human_input", "ToolCallSummary"),
    ("rag.agent.goal_runtime", "AnswerCandidate"),
    ("rag.agent.goal_runtime", "ComputationResult"),
    ("rag.agent.goal_runtime", "ContextBinding"),
    ("rag.agent.goal_runtime", "ContextUnit"),
    ("rag.agent.goal_runtime", "EvidenceRef"),
    ("rag.agent.goal_runtime", "GoalConflict"),
    ("rag.agent.goal_runtime", "GoalConstraint"),
    ("rag.agent.goal_runtime", "GoalDeliverable"),
    ("rag.agent.goal_runtime", "GoalGap"),
    ("rag.agent.goal_runtime", "GoalSpec"),
    ("rag.agent.goal_runtime", "SatisfactionReport"),
    ("rag.agent.goal_runtime", "StructuredObservation"),
    ("rag.agent.memory.models", "ContextBudgetSnapshot"),
    ("rag.agent.memory.models", "EvictedStateItem"),
    ("rag.agent.memory.models", "ExtractedFact"),
    ("rag.agent.memory.models", "ExternalizedToolOutput"),
    ("rag.agent.memory.models", "MessageBatchPayload"),
    ("rag.agent.memory.models", "MemoryBudgetSnapshot"),
    ("rag.agent.memory.models", "MemoryPolicy"),
    ("rag.agent.memory.models", "MemoryRecord"),
    ("rag.agent.memory.models", "MemoryRef"),
    ("rag.agent.memory.models", "StateChannelReplacement"),
    ("rag.agent.memory.models", "ToolErrorDetailPayload"),
    ("rag.agent.memory.models", "WorkingSummary"),
    ("rag.agent.planning", "AgentPlan"),
    ("rag.agent.planning", "PlanEvent"),
    ("rag.agent.planning", "PlanStep"),
    ("rag.agent.planning", "PlanStepPatch"),
    ("rag.agent.planning", "PlanUpdate"),
    ("rag.agent.primitive_ops", "CandidateHeaderRow"),
    ("rag.agent.primitive_ops", "StructuredProbeOutput"),
    ("rag.agent.primitive_ops", "StructuredTableProbe"),
    ("rag.agent.state", "ThinkOutput"),
    ("rag.agent.state", "ToolCallPlan"),
    ("rag.agent.tools.spec", "ToolError"),
    ("rag.agent.tools.spec", "ToolResult"),
    ("rag.schema.query", "AnswerCitation"),
    ("rag.schema.query", "EvidenceItem"),
    ("rag.schema.query", "RetrievalSignals"),
    ("rag.schema.runtime", "AccessPolicy"),
    ("rag.schema.runtime", "RuntimeMode"),
)


def agent_checkpoint_serde() -> SerializerProtocol:
    return JsonPlusSerializer(
        allowed_msgpack_modules=AGENT_CHECKPOINT_MSGPACK_ALLOWLIST,
    )


def create_agent_checkpointer(checkpoint_db: Path | str | None) -> BaseCheckpointSaver[str]:
    if checkpoint_db is None:
        return MemorySaver(serde=agent_checkpoint_serde())

    path = Path(checkpoint_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    return AsyncSqliteSaver(
        aiosqlite.connect(str(path)),
        serde=agent_checkpoint_serde(),
    )


async def aclose_agent_checkpointer(checkpointer: BaseCheckpointSaver[str]) -> None:
    connection = getattr(checkpointer, "conn", None)
    if connection is not None and hasattr(connection, "close"):
        await connection.close()
