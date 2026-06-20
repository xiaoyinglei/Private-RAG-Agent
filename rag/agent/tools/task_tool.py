"""Generic task tool — replaces role-specific agent_* delegation tools.

Creates an isolated child loop with the parent's runtime policy.
The child inherits tools but cannot recurse (task disabled by default).
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.core.context import AgentRunConfig, derive_child_config
from rag.agent.core.delegation import (
    DEFAULT_DELEGATION_TOKEN_BUDGET,
    AgentDelegationRequest,
    ParentAgentContext,
)
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec

if TYPE_CHECKING:
    from rag.agent.core.definition import AgentRuntimePolicy
    from rag.agent.core.delegation import DelegatedAgentRunner
    from rag.agent.service import AgentRunResult

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
    status: Literal["done", "failed", "paused"] = "failed"
    child_run_id: str = ""
    stop_reason: str | None = None

    @classmethod
    def from_run_result(cls, result: AgentRunResult) -> TaskOutput:
        return cls(
            conclusion=result.final_answer or "",
            key_facts=_extract_key_facts(result),
            evidence_refs=_extract_evidence_refs(result),
            citations=[
                c.model_dump(exclude_none=True)
                for c in result.citations[:MAX_CITATIONS]
            ],
            status=result.status if result.status in ("done", "failed", "paused") else "failed",
            child_run_id=result.run_id,
            stop_reason=result.stop_reason,
        )

    @classmethod
    def error_result(cls, error_message: str, *, status: str = "failed") -> TaskOutput:
        return cls(conclusion=error_message, status=status)


def _extract_key_facts(result: AgentRunResult) -> list[str]:
    facts: list[str] = []
    for item in result.evidence:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            facts.append(text.strip()[:MAX_FACT_CHARS])
        if len(facts) >= MAX_KEY_FACTS:
            break
    return facts


def _extract_evidence_refs(result: AgentRunResult) -> list[dict[str, object]]:
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
    ) -> None:
        self._policy = policy
        self._tool_registry = tool_registry
        self._model_turn_provider = model_turn_provider
        self._retrieval_hint_provider = retrieval_hint_provider

    async def run(
        self,
        payload: TaskInput,
        *,
        parent_config: AgentRunConfig,
    ) -> TaskOutput:
        from rag.agent.service import AgentService

        # Derive child config: bounded budget and depth
        child_config = self._derive_child_config(parent_config, payload)

        # Build child prompt
        child_prompt = payload.task
        if payload.context_summary:
            child_prompt = f"{payload.task}\n\n## Context\n{payload.context_summary}"

        # Create child service with the same policy but no task tool
        child_policy = self._child_policy()
        service = AgentService(
            policy=child_policy,
            tool_registry=self._tool_registry,
            model_turn_provider=self._model_turn_provider,
            retrieval_hint_provider=self._retrieval_hint_provider,
            # No subagent_runner — child cannot delegate
        )

        try:
            result = await service.run_with_config(
                task=child_prompt,
                run_config=child_config,
            )
        except Exception as exc:
            return TaskOutput.error_result(
                f"Child execution failed: {exc}",
                status="failed",
            )

        return TaskOutput.from_run_result(result)

    def _derive_child_config(
        self,
        parent_config: AgentRunConfig,
        payload: TaskInput,
    ) -> AgentRunConfig:
        child_config = derive_child_config(parent_config, definition=None)
        if payload.token_budget is not None:
            child_config = replace(child_config, budget_total=payload.token_budget)
        return child_config

    def _child_policy(self) -> AgentRuntimePolicy:
        """Derive child policy: same tools, no task delegation."""
        return replace(
            self._policy,
            max_depth=max(self._policy.max_depth - 1, 0),
            # Remove 'task' from core tools to prevent recursion
            core_tool_names=tuple(
                t for t in self._policy.core_tool_names if t != "task"
            ),
        )


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
