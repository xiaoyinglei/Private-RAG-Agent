from __future__ import annotations

from typing import Any

import pytest

from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.llm_context import AgentLLMContextOverflowError
from rag.agent.core.llm_registry import ResolvedModel
from rag.agent.core.llm_tool_runners import create_model_llm_tool_runners
from rag.agent.loop.state import LoopState, create_loop_state
from rag.agent.tools.llm_tools import (
    LLMCompareInput,
    LLMGenerateInput,
    LLMSummarizeInput,
)
from rag.agent.tools.registry import ToolExecutionContext
from rag.providers.llm_gateway import LLMGateway
from rag.schema.llm import (
    LLMCallStage,
    LLMProviderResult,
    LLMStageBudget,
    LLMUsage,
)
from rag.schema.runtime import AccessPolicy


class _WordTokenAccounting:
    def count(self, text: str) -> int:
        return len(text.split())


class _Generator:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.calls = 0

    def generate_text_with_usage(
        self,
        *,
        prompt: str,
        **kwargs: Any,
    ) -> LLMProviderResult[str]:
        del prompt
        self.calls += 1
        self.kwargs = kwargs
        return LLMProviderResult(
            value="bounded summary",
            usage=LLMUsage(
                input_tokens=8,
                output_tokens=3,
                source="provider",
            ),
        )


class _Registry:
    def __init__(self, *, max_input_tokens: int = 1_500) -> None:
        generator = _Generator()
        self.generator = generator
        self.resolved = ResolvedModel(
            generator=generator,
            kwargs={"max_tokens": 7, "temperature": 0.2},
            context_window_tokens=2_000,
            gateway=LLMGateway(
                generator=generator,
                token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
                model_context_tokens=2_000,
                stage_budgets={
                    stage: LLMStageBudget(
                        max_input_tokens=max_input_tokens,
                        max_output_tokens=100,
                        safety_margin_tokens=10,
                    )
                    for stage in (
                        LLMCallStage.LLM_SUMMARIZE,
                        LLMCallStage.LLM_GENERATE,
                        LLMCallStage.LLM_COMPARE,
                        LLMCallStage.FINAL_SYNTHESIS,
                    )
                },
            ),
        )

    def resolve_for_node(
        self,
        *,
        node_model: str | None,
        node_name: str,
    ) -> ResolvedModel:
        del node_model, node_name
        return self.resolved


def _definition() -> AgentDefinition:
    return AgentDefinition(
        agent_type="research",
        description="research",
        system_prompt="Use trusted context.",
        allowed_tools=["llm_generate", "llm_summarize", "llm_compare"],
    )


def _state(config: AgentRunConfig) -> LoopState:
    return create_loop_state(
        task="trusted task",
        run_config=config,
    )


@pytest.mark.anyio
async def test_llm_tool_runner_uses_gateway_and_run_ledger() -> None:
    config = AgentRunConfig(
        run_id="llm-tool-runner",
        thread_id="llm-tool-runner",
        budget_total=500,
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )
    RunRegistry.remove(config.run_id)
    handles = RunRegistry.get_or_create(config)
    registry = _Registry()
    runners = create_model_llm_tool_runners(registry)  # type: ignore[arg-type]

    result = await runners["llm_summarize"](
        LLMSummarizeInput(
            task="summarize",
            context_sections=["evidence text"],
        ),
        ToolExecutionContext(
            run_config=config,
            state=_state(config),
            definition=_definition(),
        ),
    )

    assert result.text == "bounded summary"
    assert registry.generator.kwargs["max_tokens"] == 100
    assert registry.generator.kwargs["temperature"] == 0.2
    assert await handles.budget_ledger.committed() == 11
    RunRegistry.remove(config.run_id)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool_name", "payload"),
    [
        (
            "llm_generate",
            LLMGenerateInput(
                prompt="generate",
                context_sections=["required context"],
            ),
        ),
        (
            "llm_summarize",
            LLMSummarizeInput(
                task="summarize",
                context_sections=["required context"],
            ),
        ),
        (
            "llm_compare",
            LLMCompareInput(
                question="compare",
                left_context_sections=["left required"],
                right_context_sections=["right required"],
            ),
        ),
    ],
)
async def test_llm_tool_required_overflow_never_calls_model(
    tool_name: str,
    payload: object,
) -> None:
    config = AgentRunConfig(
        run_id=f"{tool_name}-overflow",
        thread_id=f"{tool_name}-overflow",
        budget_total=500,
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )
    RunRegistry.remove(config.run_id)
    RunRegistry.get_or_create(config)
    registry = _Registry(max_input_tokens=1)
    runners = create_model_llm_tool_runners(registry)  # type: ignore[arg-type]

    with pytest.raises(AgentLLMContextOverflowError):
        await runners[tool_name](
            payload,  # type: ignore[arg-type]
            ToolExecutionContext(
                run_config=config,
                state=_state(config),
                definition=_definition(),
            ),
        )

    assert registry.generator.calls == 0
    RunRegistry.remove(config.run_id)
