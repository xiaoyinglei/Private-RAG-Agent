from __future__ import annotations

from dataclasses import dataclass, field

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
    # When True, tools with execute_code=True that run inside a sandbox
    # are auto-approved without user confirmation. Safety comes from the
    # sandbox boundary (restricted filesystem, no network, timeout),
    # not from asking "are you sure?" each time.
    # Mirrors Claude/GPT code interpreter: code runs automatically.
    auto_approve_sandboxed: bool = True


@dataclass(frozen=True)
class AgentRuntimePolicy:
    """Root runtime contract — replaces AgentDefinition as the primary entry point.

    ``allowed_tools`` is a derived convenience property: it returns
    ``core_tool_names`` plus all deferred tool names that pass the filter.
    Callers that only need the combined list can use it directly.
    """

    system_instructions: str
    core_tool_names: tuple[str, ...]
    deferred_tool_names: tuple[str, ...]
    token_budget: int
    work_budget: int
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


@dataclass(frozen=True)
class AgentDefinition:
    """Compatibility adapter — converts to AgentRuntimePolicy via ``to_runtime_policy()``.

    New callers should use AgentRuntimePolicy directly.
    """

    agent_type: str
    description: str
    system_prompt: str
    allowed_tools: list[str]
    access_policy: AccessPolicy | None = None
    estimated_token_budget: int = 8000
    estimated_work_budget: int = 20_000
    model_selection: ModelSelectionPolicy = field(default_factory=ModelSelectionPolicy)
    output_model: type[BaseModel] | None = None
    output_validation_max_retries: int = 2
    max_stop_hook_blocks: int = 3
    max_iterations: int = 10
    max_depth: int = 2
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)

    def __post_init__(self) -> None:
        if self.output_validation_max_retries < 0:
            raise ValueError(
                "output_validation_max_retries must be non-negative"
            )
        if self.max_stop_hook_blocks < 1:
            raise ValueError("max_stop_hook_blocks must be positive")

    def to_runtime_policy(self) -> AgentRuntimePolicy:
        """Convert legacy AgentDefinition to AgentRuntimePolicy."""
        from rag.agent.capabilities.catalog import CORE_TOOLS, DEFERRED_TOOLS

        core = tuple(
            t for t in self.allowed_tools if t in CORE_TOOLS
        )
        deferred = tuple(
            t for t in self.allowed_tools if t in DEFERRED_TOOLS
        )
        return AgentRuntimePolicy(
            system_instructions=self.system_prompt,
            core_tool_names=core,
            deferred_tool_names=deferred,
            token_budget=self.estimated_token_budget,
            work_budget=self.estimated_work_budget,
            max_iterations=self.max_iterations,
            max_depth=self.max_depth,
            access_policy_ceiling=self.access_policy,
            model_selection=self.model_selection,
            tool_policy=self.tool_policy,
            output_model=self.output_model,
            output_validation_max_retries=self.output_validation_max_retries,
            max_stop_hook_blocks=self.max_stop_hook_blocks,
            agent_type=self.agent_type,
            description=self.description,
        )

    # ── backward compat: expose runtime_policy as a property ──

    @property
    def runtime_policy(self) -> AgentRuntimePolicy:
        """Proxy to runtime_policy sub-fields for legacy callers."""
        return self.to_runtime_policy()
