from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from rag.schema.llm import (
    DEFAULT_LLM_STAGE_BUDGETS,
    LLMCallStage,
    LLMStageBudget,
)


class ModelProvider(StrEnum):
    """当前 Agent 模型配置真正支持的 provider。"""

    MLX = "mlx"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class ModelSpec(BaseModel):
    """单个模型声明：只允许填写当前已实现 provider 支持的模型。"""

    provider: ModelProvider
    model: str
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    base_url: str | None = None
    api_key_env: str | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)
    context_window_tokens: int = Field(default=32_768, gt=0)


class AgentModelsConfig(BaseModel):
    """Agent 模型配置：只声明可用模型，不绑定具体运行节点角色。"""

    version: int = 1
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    default_model: str
    fallback_model: str | None = None
    llm_stage_budgets: dict[LLMCallStage, LLMStageBudget] = Field(
        default_factory=lambda: {
            stage: budget.model_copy()
            for stage, budget in DEFAULT_LLM_STAGE_BUDGETS.items()
        }
    )

    @model_validator(mode="after")
    def validate_model_refs(self) -> AgentModelsConfig:
        if not self.models:
            raise ValueError("models must not be empty")

        if self.default_model not in self.models:
            raise ValueError(f"default_model not found in models: {self.default_model}")

        if self.fallback_model and self.fallback_model not in self.models:
            raise ValueError(f"fallback_model not found in models: {self.fallback_model}")

        return self
