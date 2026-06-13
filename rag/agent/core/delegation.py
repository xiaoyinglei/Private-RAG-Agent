from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from rag.agent.core.context import AgentRunConfig
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem

DEFAULT_DELEGATION_TOKEN_BUDGET = 10000


class AgentAsToolExecutionError(RuntimeError):
    def __init__(
        self,
        agent_name: str,
        message: str,
        *,
        status: str = "failed",
        stop_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.agent_name = agent_name
        self.status = status
        self.stop_reason = stop_reason


class ParentAgentContext(TypedDict):
    """Minimal parent data required to derive a bounded child run."""

    run_config: AgentRunConfig


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
        parent_state: ParentAgentContext,
    ) -> DelegatedAgentResult | Awaitable[DelegatedAgentResult]: ...


__all__ = [
    "AgentDelegationRequest",
    "AgentAsToolExecutionError",
    "DEFAULT_DELEGATION_TOKEN_BUDGET",
    "DelegatedAgentResult",
    "DelegatedAgentRunner",
    "ParentAgentContext",
]
