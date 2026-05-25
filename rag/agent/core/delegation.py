from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.state import AgentState
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem

DEFAULT_DELEGATION_TOKEN_BUDGET = 10000


class AgentDelegationRequest(BaseModel):
    """One bounded child-agent invocation issued by an ordinary agent tool."""

    model_config = ConfigDict(frozen=True)

    delegation_id: str
    agent_type: str
    prompt: str
    estimated_tokens: int | None = Field(default=DEFAULT_DELEGATION_TOKEN_BUDGET, gt=0)


class DelegatedAgentResult(Protocol):
    status: str
    final_answer: str | None
    stop_reason: str | None
    tool_results: list[ToolResult]
    evidence: list[EvidenceItem]
    citations: list[AnswerCitation]


class DelegatedAgentRunner(Protocol):
    def run_delegated_task(
        self,
        *,
        request: AgentDelegationRequest,
        parent_state: AgentState,
    ) -> DelegatedAgentResult | Awaitable[DelegatedAgentResult]: ...


__all__ = [
    "AgentDelegationRequest",
    "DEFAULT_DELEGATION_TOKEN_BUDGET",
    "DelegatedAgentResult",
    "DelegatedAgentRunner",
]
