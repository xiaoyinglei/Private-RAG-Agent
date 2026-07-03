"""Public Agent SDK facade."""

from typing import TYPE_CHECKING

from agent_runtime.models import (
    ModelCatalog,
    ModelControlPlane,
    ModelPolicy,
    ModelPolicyError,
    ModelRuntimeSpec,
    ModelSessionState,
    ModelSpec,
)
from agent_runtime.result import AgentResult, AgentUsage

if TYPE_CHECKING:
    from agent_runtime.agent import Agent


def __getattr__(name: str) -> object:
    if name == "Agent":
        from agent_runtime.agent import Agent

        return Agent
    raise AttributeError(f"module 'agent_runtime' has no attribute {name!r}")


__all__ = [
    "Agent",
    "AgentResult",
    "AgentUsage",
    "ModelCatalog",
    "ModelControlPlane",
    "ModelPolicy",
    "ModelPolicyError",
    "ModelRuntimeSpec",
    "ModelSessionState",
    "ModelSpec",
]
