from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from rag.agent.core.context import BudgetLedger
from rag.assembly.support import _OpenAICompatibleChatGenerator
from rag.providers.llm_gateway import LLMContextOverflowError, LLMGateway, StreamChunk
from rag.schema.llm import (
    LLMCallStage,
    LLMProviderResult,
    LLMStageBudget,
    LLMUsage,
)


class _WordTokenAccounting:
    def count(self, text: str) -> int:
        return len(text.split())


class _UsageAwareGenerator:
    def __init__(
        self,
        *,
        output: str = "final answer",
        usage: LLMUsage | None = None,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.usage = usage
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate_text_with_usage(
        self,
        *,
        prompt: str,
        **kwargs: Any,
    ) -> LLMProviderResult[str]:
        self.calls.append((prompt, kwargs))
        if self.error is not None:
            raise self.error
        return LLMProviderResult(value=self.output, usage=self.usage)


class _Decision(BaseModel):
    action: str


class _StructuredUsageAwareGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_structured_with_usage(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> LLMProviderResult[BaseModel]:
        del prompt
        self.calls.append(kwargs)
        return LLMProviderResult(
            value=schema.model_validate({"action": "execute"}),
            usage=LLMUsage(
                input_tokens=6,
                output_tokens=2,
                cached_input_tokens=3,
                reasoning_tokens=1,
                source="provider",
            ),
        )


class _StreamingGenerator:
    def stream_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[StreamChunk]:
        del messages, tools, kwargs
        return [
            StreamChunk(type="text_delta", content="one two"),
            StreamChunk(type="message_stop", stop_reason="end_turn"),
        ]


def _gateway(
    generator: object,
    *,
    max_input_tokens: int = 8,
    max_output_tokens: int = 4,
    model_context_tokens: int = 32,
) -> LLMGateway:
    return LLMGateway(
        generator=generator,
        token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
        model_context_tokens=model_context_tokens,
        stage_budgets={
            LLMCallStage.TOOL_DECISION: LLMStageBudget(
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                safety_margin_tokens=2,
            )
        },
    )


@pytest.mark.anyio
async def test_gateway_reserves_worst_case_and_commits_provider_usage() -> None:
    generator = _UsageAwareGenerator(
        usage=LLMUsage(input_tokens=3, output_tokens=2, source="provider")
    )
    ledger = BudgetLedger(total=20)

    result = await _gateway(generator).agenerate_text(
        stage=LLMCallStage.TOOL_DECISION,
        prompt="one two three",
        ledger=ledger,
        lease_id="decision:1",
    )

    assert result.value == "final answer"
    assert result.usage.source == "provider"
    assert result.usage.total_tokens == 5
    assert await ledger.committed() == 5
    assert await ledger.reserved() == 0
    assert generator.calls[0][1]["max_tokens"] == 4


@pytest.mark.anyio
async def test_gateway_estimates_usage_when_provider_does_not_report_it() -> None:
    generator = _UsageAwareGenerator(usage=None, output="four five")
    ledger = BudgetLedger(total=20)

    result = await _gateway(generator).agenerate_text(
        stage=LLMCallStage.TOOL_DECISION,
        prompt="one two three",
        ledger=ledger,
        lease_id="decision:2",
    )

    assert result.usage == LLMUsage(
        input_tokens=3,
        output_tokens=2,
        source="tokenizer_estimate",
    )
    assert await ledger.committed() == 5


@pytest.mark.anyio
async def test_gateway_refunds_reservation_when_provider_fails() -> None:
    generator = _UsageAwareGenerator(error=RuntimeError("provider unavailable"))
    ledger = BudgetLedger(total=20)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await _gateway(generator).agenerate_text(
            stage=LLMCallStage.TOOL_DECISION,
            prompt="one two three",
            ledger=ledger,
            lease_id="decision:3",
        )

    assert await ledger.committed() == 0
    assert await ledger.reserved() == 0
    assert await ledger.remaining() == 20


@pytest.mark.anyio
async def test_gateway_rejects_prompt_above_stage_input_budget() -> None:
    generator = _UsageAwareGenerator()
    ledger = BudgetLedger(total=100)

    with pytest.raises(LLMContextOverflowError) as exc_info:
        await _gateway(generator, max_input_tokens=3).agenerate_text(
            stage=LLMCallStage.TOOL_DECISION,
            prompt="one two three four",
            ledger=ledger,
            lease_id="decision:4",
        )

    assert exc_info.value.input_tokens == 4
    assert exc_info.value.max_input_tokens == 3
    assert generator.calls == []
    assert await ledger.remaining() == 100


@pytest.mark.anyio
async def test_gateway_respects_model_window_after_output_and_safety_reserve() -> None:
    generator = _UsageAwareGenerator()

    with pytest.raises(LLMContextOverflowError) as exc_info:
        await _gateway(
            generator,
            max_input_tokens=20,
            max_output_tokens=4,
            model_context_tokens=10,
        ).agenerate_text(
            stage=LLMCallStage.TOOL_DECISION,
            prompt="one two three four five",
            ledger=BudgetLedger(total=100),
            lease_id="decision:5",
        )

    assert exc_info.value.max_input_tokens == 4
    assert generator.calls == []


def test_gateway_exposes_effective_stage_budget_for_context_assembly() -> None:
    gateway = _gateway(
        _UsageAwareGenerator(),
        max_input_tokens=20,
        max_output_tokens=4,
        model_context_tokens=10,
    )

    budget = gateway.effective_stage_budget(LLMCallStage.TOOL_DECISION)

    assert budget.max_input_tokens == 4
    assert budget.max_output_tokens == 4


@pytest.mark.anyio
async def test_gateway_accounts_for_structured_generation_as_one_call() -> None:
    generator = _StructuredUsageAwareGenerator()
    ledger = BudgetLedger(total=30)

    result = await _gateway(
        generator,
        max_input_tokens=100,
        model_context_tokens=200,
    ).agenerate_structured(
        stage=LLMCallStage.TOOL_DECISION,
        prompt="choose action",
        schema=_Decision,
        ledger=ledger,
        lease_id="decision:structured",
    )

    assert result.value == _Decision(action="execute")
    assert result.usage.cached_input_tokens == 3
    assert result.usage.reasoning_tokens == 1
    assert result.usage.total_tokens == 8
    assert await ledger.committed() == 8
    assert generator.calls[0]["max_tokens"] == 4


@pytest.mark.anyio
async def test_streaming_gateway_commits_tokenizer_estimate_usage() -> None:
    ledger = BudgetLedger(total=20)

    chunks = [
        chunk
        async for chunk in _gateway(_StreamingGenerator()).astream_with_tools(
            stage=LLMCallStage.TOOL_DECISION,
            messages=[{"role": "user", "content": "hello prompt"}],
            tools=[],
            ledger=ledger,
            lease_id="decision:stream",
        )
    ]

    assert [chunk.type for chunk in chunks] == ["text_delta", "message_stop"]
    assert chunks[0].content == "one two"
    assert await ledger.committed() == 5
    assert await ledger.reserved() == 0


def test_openai_compatible_generator_preserves_response_usage() -> None:
    generator = _OpenAICompatibleChatGenerator.__new__(_OpenAICompatibleChatGenerator)
    generator.chat_model_name = "test-model"
    generator._base_url = "https://example.test/v1"
    generator._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="provider answer")
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=11,
                        completion_tokens=7,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=5),
                        completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
                    ),
                )
            )
        )
    )

    result = generator.generate_text_with_usage(prompt="question", max_tokens=20)

    assert result.value == "provider answer"
    assert result.usage == LLMUsage(
        input_tokens=11,
        output_tokens=7,
        cached_input_tokens=5,
        reasoning_tokens=3,
        source="provider",
    )
