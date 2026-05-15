from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from rag.agent.core.context import AgentRunConfig, derive_child_config
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.task import DEFAULT_SUBTASK_TOKEN_BUDGET, SubTaskNode
from rag.agent.graphs.nodes.evaluate import EvaluateDecisionProvider
from rag.agent.graphs.nodes.plan import PlanProvider
from rag.agent.graphs.nodes.route import RouteProvider
from rag.agent.state import AgentState
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec

if TYPE_CHECKING:
    from rag.agent.service import AgentRunResult

# ── AgentToolSpec (existing) ────────────────────────────────────

@dataclass(frozen=True)
class AgentToolSpec:
    tool_spec: ToolSpec
    agent_definition: AgentDefinition
    inherits_context: bool = True

# ── Tool I/O schemas ───────────────────────────────────────────

class AgentToolInput(BaseModel):
    """Agent-as-tool 输入。不暴露完整 SubTaskNode，只包含稳定业务字段。"""

    task: str = Field(min_length=1, description="子任务描述")
    goal: str | None = Field(default=None, description="期望产出")
    context_summary: str | None = Field(default=None, description="父 Agent 传递的上下文摘要")
    required_outputs: list[str] = Field(default_factory=list, description="需要的产出类型，如 ['evidence', 'conclusion']")
    constraints: list[str] = Field(default_factory=list, description="约束条件，如 'prefer_table', 'max_3_items'")


class AgentToolOutput(BaseModel):
    """Agent-as-tool 脱水结果。不暴露完整 AgentRunResult，只返回关键信息。"""

    conclusion: str = Field(default="", description="子 Agent 的核心结论")
    key_facts: list[str] = Field(default_factory=list, description="关键事实列表")
    evidence_refs: list[str] = Field(default_factory=list, description="证据引用（evidence_id 列表）")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度 0-1")
    status: str = Field(default="unknown", description="子 Agent 最终状态")
    agent_name: str = Field(default="", description="执行的 Agent 类型")

    @classmethod
    def from_run_result(cls, result: AgentRunResult, agent_name: str) -> AgentToolOutput:
        """从 AgentRunResult 提取脱水结果。"""
        return cls(
            conclusion=result.final_answer or "",
            key_facts=_extract_key_facts(result),
            evidence_refs=[item.evidence_id for item in result.evidence if getattr(item, "evidence_id", None)],
            confidence=_derive_confidence(result),
            status=result.status,
            agent_name=agent_name,
        )

    @classmethod
    def error_result(cls, agent_name: str, error_message: str, status: str = "failed") -> AgentToolOutput:
        """构造错误结果。"""
        return cls(
            conclusion=error_message,
            status=status,
            agent_name=agent_name,
        )


def _extract_key_facts(result: AgentRunResult) -> list[str]:
    facts: list[str] = []
    for item in result.evidence:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            facts.append(text.strip()[:200])
    return facts[:10]


def _derive_confidence(result: AgentRunResult) -> float:
    if result.status == "done":
        return 0.8 if result.groundedness_flag else 0.5
    return 0.1

# ── AgentAsToolAdapter ─────────────────────────────────────────

class AgentAsToolAdapter:
    """将 AgentAsToolRunner 适配为 ToolRunner = Callable[[BaseModel], BaseModel]。

    Request-scoped：每个 run_config 创建一个新实例，绑定到 runtime_tool_registry。
    禁止跨请求复用 adapter，避免 run_config 并发污染。
    """

    def __init__(
        self,
        runner: AgentAsToolRunner,
        agent_type: str,
        run_config: AgentRunConfig,
    ) -> None:
        self._runner = runner
        self._agent_type = agent_type
        self._run_config = run_config

    async def __call__(self, payload: AgentToolInput) -> AgentToolOutput:
        # 构造最小父上下文（不是完整 LangGraph state，仅用于 derive_child_config）
        # run_subtask 只访问 parent_state["run_config"]，不读取其他 state 字段
        parent_state: AgentState = {
            "run_config": self._run_config,
            "messages": [],
            "evidence": [],
            "citations": [],
            "tool_results": [],
            "task": payload.task,
            "retrieval_signals": None,
            "retrieval_signals_debug": None,
            "plan": None,
            "iteration": 0,
            "status": "running",
            "route_reason": None,
            "stop_reason": None,
            "needs_user_input": None,
            "pending_tool_calls": [],
            "approved_tool_call_ids": [],
            "denied_tool_call_ids": [],
            "user_decision": None,
            "user_message": None,
            "human_input_request": None,
            "human_input_response": None,
            "next_subtasks": None,
            "working_summary": None,
            "extracted_facts": [],
            "context_budget": None,
            "subtask_results": {},
            "terminal_subtasks": set(),
            "successful_subtasks": set(),
            "final_answer": None,
            "groundedness_flag": False,
            "insufficient_evidence_flag": False,
        }

        prompt = _build_subtask_prompt(payload)
        subtask = SubTaskNode(
            subtask_id=f"{self._agent_type}-tool-{uuid4().hex[:8]}",
            agent_type=self._agent_type,
            prompt=prompt,
            priority=1,
            estimated_tokens=DEFAULT_SUBTASK_TOKEN_BUDGET,
        )

        try:
            result = await self._runner.run_subtask(subtask=subtask, parent_state=parent_state)
        except Exception as exc:
            return AgentToolOutput.error_result(
                self._agent_type,
                f"subagent execution failed: {exc.__class__.__name__}",
                status="failed",
            )

        return AgentToolOutput.from_run_result(result, self._agent_type)


def _build_subtask_prompt(payload: AgentToolInput) -> str:
    parts = [f"## Task\n{payload.task}"]
    if payload.goal:
        parts.append(f"## Goal\n{payload.goal}")
    if payload.context_summary:
        parts.append(f"## Context\n{payload.context_summary}")
    if payload.required_outputs:
        parts.append(f"## Required Outputs\n{', '.join(payload.required_outputs)}")
    if payload.constraints:
        parts.append(f"## Constraints\n" + "\n".join(f"- {c}" for c in payload.constraints))
    return "\n\n".join(parts)

# ── build_agent_tool_spec ──────────────────────────────────────

# Agent-as-tool 白名单：只有这些 agent_type 可以被包装为 tool
_AGENT_TOOL_WHITELIST: frozenset[str] = frozenset({
    "research",
    "compare",
    "factcheck",
    "synthesize",
})

# 禁止注册为 tool 的 agent_type（防止递归等安全问题）
_AGENT_TOOL_BLOCKLIST: frozenset[str] = frozenset({
    "orchestrator",
})


def build_agent_tool_spec(agent_definition: AgentDefinition) -> AgentToolSpec:
    """将 AgentDefinition 包装为 AgentToolSpec（含 ToolSpec）。

    白名单检查：只有 research/compare/factcheck/synthesize 可以注册。
    黑名单检查：orchestrator 禁止注册，避免递归调用。
    """
    agent_type = agent_definition.agent_type

    if agent_type in _AGENT_TOOL_BLOCKLIST:
        raise ValueError(
            f"Agent type {agent_type!r} is blocklisted from agent-as-tool registration."
        )

    if agent_type not in _AGENT_TOOL_WHITELIST:
        raise ValueError(
            f"Agent type {agent_type!r} is not in the agent-as-tool whitelist. "
            f"Allowed: {', '.join(sorted(_AGENT_TOOL_WHITELIST))}"
        )

    tool_name = f"agent_{agent_type}"
    tool_description = _make_tool_description(agent_definition)

    tool_spec = ToolSpec(
        name=tool_name,
        description=tool_description,
        input_model=AgentToolInput,
        output_model=AgentToolOutput,
        error_model=ToolError,
        permissions=ToolPermissions(
            read_db=True,
            embed=True,
            generate=True,
        ),
        timeout_seconds=120.0,
        max_retries=0,
        token_budget_cost=agent_definition.estimated_token_budget,
    )

    return AgentToolSpec(
        tool_spec=tool_spec,
        agent_definition=agent_definition,
        inherits_context=True,
    )


def _make_tool_description(definition: AgentDefinition) -> str:
    return (
        f"Delegate a subtask to the {definition.agent_type} agent. "
        f"{definition.description} "
        f"Use this when the task requires specialized {definition.agent_type} capabilities."
    )


class AgentAsToolRunner:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        agent_registry: AgentRegistry,
        evaluate_decision_provider: EvaluateDecisionProvider | None = None,
        plan_provider: PlanProvider | None = None,
        route_provider: RouteProvider | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._agent_registry = agent_registry
        self._evaluate_decision_provider = evaluate_decision_provider
        self._plan_provider = plan_provider
        self._route_provider = route_provider

    async def run_subtask(
        self,
        *,
        subtask: SubTaskNode,
        parent_state: AgentState,
    ) -> AgentRunResult:
        from rag.agent.service import AgentService

        parent_config = parent_state["run_config"]
        child_definition = self._agent_registry.get(subtask.agent_type)
        child_config = derive_child_config(parent_config, child_definition)
        if subtask.estimated_tokens is not None:
            child_config = replace(child_config, budget_total=subtask.estimated_tokens)

        service = AgentService(
            definition=child_definition,
            tool_registry=self._tool_registry,
            evaluate_decision_provider=self._evaluate_decision_provider,
            plan_provider=self._plan_provider,
            route_provider=self._route_provider,
            subagent_runner=self,
        )
        return await service.run_with_config(
            task=subtask.prompt,
            run_config=child_config,
        )
