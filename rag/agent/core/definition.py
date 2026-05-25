from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

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


@dataclass(frozen=True)
class AgentDefinition:
    agent_type: str
    description: str
    system_prompt: str
    allowed_tools: list[str]
    access_policy: AccessPolicy | None = None
    estimated_token_budget: int = 8000
    model_selection: ModelSelectionPolicy = field(default_factory=ModelSelectionPolicy)
    output_model: type[BaseModel] | None = None
    max_iterations: int = 10
    max_depth: int = 2
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
