from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from rag.agent.capabilities.catalog import ToolCatalogFilter
from rag.schema.runtime import AccessPolicy


@dataclass(frozen=True)
class ModelSelectionPolicy:
    """每个 Agent 节点的模型选择策略。None = 使用 ModelRegistry.default_model。"""

    retrieval_hint_model: str | None = None
    tool_decision_model: str | None = None
    thinking: bool = True
    retrieval_hint_temperature: float = 0.0
    tool_decision_temperature: float = 0.0
    retrieval_hint_max_tokens: int | None = None
    tool_decision_max_tokens: int | None = None


@dataclass(frozen=True)
class ToolPolicy:
    max_parallel_calls: int = 4
    require_confirmation_for: frozenset[str] = field(default_factory=frozenset)
    deny_tools: frozenset[str] = field(default_factory=frozenset)
    # Opt-in only. Code execution is approval-gated by default even when a
    # tool is expected to run in a sandbox.
    auto_approve_sandboxed: bool = False


@dataclass(frozen=True)
class AgentRuntimePolicy:
    """Root runtime contract — replaces AgentRuntimePolicy as the primary entry point.

    ``allowed_tools`` is a derived convenience property: it returns
    ``core_tool_names`` plus all deferred tool names that pass the filter.
    Callers that only need the combined list can use it directly.
    """

    system_instructions: str
    core_tool_names: tuple[str, ...]
    deferred_tool_names: tuple[str, ...]
    max_iterations: int
    max_depth: int
    max_active_deferred_tools: int = 10
    tool_catalog_filter: ToolCatalogFilter = field(
        default_factory=ToolCatalogFilter,
    )
    access_policy_ceiling: AccessPolicy | None = None
    model_selection: ModelSelectionPolicy = field(
        default_factory=ModelSelectionPolicy,
    )
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    output_model: type[BaseModel] | None = None
    output_validation_max_retries: int = 2
    max_stop_hook_blocks: int = 3

    # ── MCP (external tool source) ──
    mcp_servers: tuple[str, ...] = ()      # enabled MCP server names
    mcp_allow_all_tools: bool = False      # explicitly allow ALL MCP tools (default: only allowlisted)

    # ── legacy compatibility metadata (not used for behavior) ──
    agent_type: str = "generic"
    description: str = ""

    @property
    def allowed_tools(self) -> list[str]:
        """Combined core + deferred tool names for backward compatibility."""
        return list(self.core_tool_names) + list(self.deferred_tool_names)

    def __post_init__(self) -> None:
        if self.output_validation_max_retries < 0:
            raise ValueError(
                "output_validation_max_retries must be non-negative"
            )
        if self.max_stop_hook_blocks < 1:
            raise ValueError("max_stop_hook_blocks must be positive")

    @classmethod
    def test_factory(
        cls,
        *,
        agent_type: str = "generic",
        description: str = "",
        system_prompt: str = "",
        allowed_tools: list[str] | None = None,
        access_policy: Any = None,
        model_selection: Any = None,
        output_model: type[BaseModel] | None = None,
        output_validation_max_retries: int = 2,
        max_stop_hook_blocks: int = 3,
        max_iterations: int = 10,
        max_depth: int = 2,
        tool_policy: Any = None,
    ) -> AgentRuntimePolicy:
        """Convenience factory — flat tool list + sensible defaults for tests."""
        from rag.agent.capabilities.catalog import CORE_TOOLS, DEFERRED_TOOLS

        tools = allowed_tools or []
        core = tuple(t for t in tools if t in CORE_TOOLS)
        deferred = tuple(t for t in tools if t in DEFERRED_TOOLS)
        core = core + tuple(t for t in tools if t not in CORE_TOOLS and t not in DEFERRED_TOOLS)

        return cls(
            system_instructions=system_prompt,
            core_tool_names=core,
            deferred_tool_names=deferred,
            max_iterations=max_iterations,
            max_depth=max_depth,
            agent_type=agent_type,
            description=description,
            access_policy_ceiling=access_policy,
            model_selection=model_selection or ModelSelectionPolicy(),
            tool_policy=tool_policy or ToolPolicy(),
            output_model=output_model,
            output_validation_max_retries=output_validation_max_retries,
            max_stop_hook_blocks=max_stop_hook_blocks,
        )


