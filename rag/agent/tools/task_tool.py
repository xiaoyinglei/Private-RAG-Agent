"""Generic task tool — replaces role-specific agent_* delegation tools.

Creates an isolated child loop with the parent's runtime policy.
The child inherits tools but cannot recurse (task disabled by default).
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.delegation import (
    DEFAULT_DELEGATION_TOKEN_BUDGET,
    AgentDelegationRequest,
    DelegatedAgentResult,
    DelegatedAgentRunner,
    ParentAgentContext,
)
from rag.agent.tools.spec import ExecutionCategory, ToolError, ToolPermissions, ToolSpec

if TYPE_CHECKING:
    from rag.agent.core.definition import AgentRuntimePolicy
    from rag.agent.core.runtime_ports import RetrievalHintProvider
    from rag.agent.loop.runtime import ModelTurnProvider
    from rag.agent.tools.registry import ToolRegistry

# ── I/O schemas ──────────────────────────────────────────────────


class TaskInput(BaseModel):
    """Input for the generic task tool."""

    model_config = ConfigDict(frozen=True)

    task: str = Field(
        min_length=1,
        description="The task to execute in an isolated child loop.",
    )
    context_summary: str | None = Field(
        default=None,
        description="Bounded context from the parent to pass to the child.",
    )
    tool_query: str | None = Field(
        default=None,
        description=(
            "Hint for tool discovery (e.g. 'search documents', 'analyze table'). "
            "Not a permission grant — the child discovers tools within its policy."
        ),
    )
    token_budget: int | None = Field(
        default=None,
        gt=0,
        description="Override child token budget. Defaults to parent's delegation budget.",
    )


MAX_KEY_FACTS = 10
MAX_FACT_CHARS = 200
MAX_EVIDENCE_REFS = 20
MAX_CITATIONS = 20
TaskStatus = Literal["done", "failed", "paused"]


class TaskOutput(BaseModel):
    """Typed result from the child loop."""

    model_config = ConfigDict(frozen=True)

    conclusion: str = Field(default="", description="Child's core conclusion.")
    key_facts: list[str] = Field(
        default_factory=list,
        max_length=MAX_KEY_FACTS,
    )
    evidence_refs: list[dict[str, object]] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_REFS,
    )
    citations: list[dict[str, object]] = Field(
        default_factory=list,
        max_length=MAX_CITATIONS,
    )
    status: TaskStatus = "failed"
    child_run_id: str = ""
    stop_reason: str | None = None

    @classmethod
    def from_run_result(cls, result: DelegatedAgentResult) -> TaskOutput:
        return cls(
            conclusion=result.final_answer or "",
            key_facts=_extract_key_facts(result),
            evidence_refs=_extract_evidence_refs(result),
            citations=[
                c.model_dump(exclude_none=True)
                for c in result.citations[:MAX_CITATIONS]
            ],
            status=_coerce_status(result.status),
            child_run_id=result.run_id,
            stop_reason=result.stop_reason,
        )

    @classmethod
    def error_result(cls, error_message: str, *, status: TaskStatus = "failed") -> TaskOutput:
        return cls(conclusion=error_message, status=status)


def _coerce_status(status: str) -> TaskStatus:
    if status == "done":
        return "done"
    if status == "paused":
        return "paused"
    return "failed"


def _extract_key_facts(result: DelegatedAgentResult) -> list[str]:
    facts: list[str] = []
    for item in result.evidence:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            facts.append(text.strip()[:MAX_FACT_CHARS])
        if len(facts) >= MAX_KEY_FACTS:
            break
    return facts


def _extract_evidence_refs(result: DelegatedAgentResult) -> list[dict[str, object]]:
    refs: list[dict[str, object]] = []
    for item in result.evidence[:MAX_EVIDENCE_REFS]:
        ref: dict[str, object] = {"evidence_id": item.evidence_id}
        if item.doc_id is not None:
            ref["doc_id"] = item.doc_id
        if item.citation_anchor and item.citation_anchor.strip():
            ref["citation_anchor"] = item.citation_anchor.strip()
        refs.append(ref)
    return refs


# ── Task tool runner ─────────────────────────────────────────────


class TaskToolRunner:
    """Creates an isolated child loop for generic task delegation.

    The child inherits the parent's tool registry and model registry,
    but uses a derived run config with bounded budget and depth.
    The 'task' tool is disabled in the child by default to prevent
    recursive delegation.
    """

    def __init__(
        self,
        *,
        policy: AgentRuntimePolicy,
        tool_registry: ToolRegistry,
        model_turn_provider: ModelTurnProvider | None = None,
        retrieval_hint_provider: RetrievalHintProvider | None = None,
        delegated_runner: DelegatedAgentRunner | None = None,
    ) -> None:
        del tool_registry, model_turn_provider, retrieval_hint_provider
        self._policy = policy
        self._delegated_runner = delegated_runner

    async def run(
        self,
        payload: TaskInput,
        *,
        parent_config: AgentRunConfig,
    ) -> TaskOutput:
        if self._delegated_runner is None:
            return TaskOutput.error_result(
                "Task delegation runner is not configured.",
                status="failed",
            )

        # Build child prompt
        child_prompt = payload.task
        if payload.context_summary:
            child_prompt = f"{payload.task}\n\n## Context\n{payload.context_summary}"
        delegation = AgentDelegationRequest(
            delegation_id=f"task-{uuid4().hex[:8]}",
            agent_type="task_child",
            prompt=child_prompt,
            estimated_tokens=(
                payload.token_budget
                or self._policy.token_budget
                or DEFAULT_DELEGATION_TOKEN_BUDGET
            ),
        )
        parent_state = ParentAgentContext(run_config=parent_config)

        try:
            raw_result = self._delegated_runner.run_delegated_task(
                request=delegation,
                parent_state=parent_state,
            )
            result = await raw_result if inspect.isawaitable(raw_result) else raw_result
        except Exception as exc:
            return TaskOutput.error_result(
                f"Child execution failed: {exc}",
                status="failed",
            )

        return TaskOutput.from_run_result(result)


# ── Tool spec ────────────────────────────────────────────────────


task_tool_spec = ToolSpec(
    name="task",
    description=(
        "Execute a task in an isolated child loop. The child inherits your "
        "tools and can search, retrieve, and analyze data independently. "
        "Use this for bounded sub-tasks that benefit from context isolation. "
        "Returns a typed result with conclusion, key facts, and citations."
    ),
    input_model=TaskInput,
    output_model=TaskOutput,
    error_model=ToolError,
    permissions=ToolPermissions(
        read_db=True,
        embed=True,
        generate=True,
    ),
    execution_category=ExecutionCategory.READ,
    timeout_seconds=180.0,
    max_retries=0,
    work_budget_cost=2_000,
)


__all__ = [
    "TaskInput",
    "TaskOutput",
    "TaskToolRunner",
    "task_tool_spec",
]
