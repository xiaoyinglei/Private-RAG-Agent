"""Real model smoke tests — verify the full agent loop works with a live LLM.

Uses DeepSeek (cheapest available model) for minimal end-to-end checks.
Requires DEEPSEEK_API_KEY in .env and network access.

Run:
    uv run pytest tests/agent/test_smoke_real_model.py -q -v
"""

from __future__ import annotations

import pytest


def _deepseek_service():
    from rag.agent.builtin.generic import GENERIC_AGENT
    from rag.agent.builtin_registry import create_builtin_tool_registry
    from rag.agent.core.agent_service_factory import AgentServiceFactory
    from rag.agent.core.llm_registry import ModelRegistry

    registry = ModelRegistry.from_env(default_model="deepseek_chat")
    tool_registry = create_builtin_tool_registry()
    factory = AgentServiceFactory(
        tool_registry=tool_registry,
        model_registry=registry,
    )
    return factory.create(GENERIC_AGENT)


@pytest.mark.anyio
class TestRealModelSmoke:
    async def test_hello(self) -> None:
        """Agent returns a simple text response via DeepSeek."""
        from rag.agent.service import AgentRunRequest

        svc = _deepseek_service()
        result = await svc.run(AgentRunRequest(
            task='Say exactly: "OK"',
            max_turns=10,
        ))

        assert result.status == "done", (
            f"status={result.status}, stop_reason={result.stop_reason}"
        )
        assert result.final_answer is not None
        assert "ok" in result.final_answer.lower()

    async def test_simple_math(self) -> None:
        """Model answers 2+2 correctly."""
        from rag.agent.service import AgentRunRequest

        svc = _deepseek_service()
        result = await svc.run(AgentRunRequest(
            task="What is 2 + 2? Answer with just the number.",
            max_turns=10,
        ))

        assert result.status == "done", (
            f"status={result.status}, stop_reason={result.stop_reason}"
        )
        assert result.final_answer is not None
        assert "4" in result.final_answer
