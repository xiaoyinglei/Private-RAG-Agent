from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

from pydantic import BaseModel

from rag.schema.llm import (
    DEFAULT_LLM_STAGE_BUDGETS,
    LLMCallResult,
    LLMCallStage,
    LLMProviderResult,
    LLMStageBudget,
    LLMUsage,
)


class TokenAccounting(Protocol):
    def count(self, text: str) -> int: ...

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str: ...


class AsyncBudgetLedger(Protocol):
    async def reserve(self, lease_id: str, amount: int) -> bool: ...
    async def commit(self, lease_id: str, actual: int) -> int: ...
    async def refund(self, lease_id: str) -> int: ...


_ACTIVE_LLM_BUDGET_LEDGER: ContextVar[AsyncBudgetLedger | None] = ContextVar(
    "active_llm_budget_ledger",
    default=None,
)


def current_llm_budget_ledger() -> AsyncBudgetLedger | None:
    return _ACTIVE_LLM_BUDGET_LEDGER.get()


@contextmanager
def llm_budget_scope(
    ledger: AsyncBudgetLedger | None,
) -> Iterator[None]:
    token = _ACTIVE_LLM_BUDGET_LEDGER.set(ledger)
    try:
        yield
    finally:
        _ACTIVE_LLM_BUDGET_LEDGER.reset(token)


class LLMGatewayError(RuntimeError):
    pass


class LLMContextOverflowError(LLMGatewayError):
    def __init__(
        self,
        *,
        stage: LLMCallStage,
        input_tokens: int,
        max_input_tokens: int,
    ) -> None:
        super().__init__(
            f"{stage.value} input uses {input_tokens} tokens; "
            f"maximum is {max_input_tokens}"
        )
        self.stage = stage
        self.input_tokens = input_tokens
        self.max_input_tokens = max_input_tokens


class LLMBudgetExceededError(LLMGatewayError):
    def __init__(self, *, stage: LLMCallStage, required_tokens: int) -> None:
        super().__init__(
            f"Insufficient LLM token budget for {stage.value}; "
            f"required reservation is {required_tokens}"
        )
        self.stage = stage
        self.required_tokens = required_tokens


class LLMGateway:
    def __init__(
        self,
        *,
        generator: object,
        token_accounting: TokenAccounting,
        model_context_tokens: int,
        stage_budgets: Mapping[LLMCallStage, LLMStageBudget] | None = None,
    ) -> None:
        if model_context_tokens <= 0:
            raise ValueError("model_context_tokens must be positive")
        self._generator = generator
        self._token_accounting = token_accounting
        self._model_context_tokens = model_context_tokens
        self._stage_budgets = dict(stage_budgets or DEFAULT_LLM_STAGE_BUDGETS)

    @property
    def token_accounting(self) -> TokenAccounting:
        return self._token_accounting

    def stage_budget(self, stage: LLMCallStage) -> LLMStageBudget:
        return self._stage_budget(stage).model_copy()

    def effective_stage_budget(
        self,
        stage: LLMCallStage,
        *,
        kwargs: Mapping[str, Any] | None = None,
    ) -> LLMStageBudget:
        budget = self._stage_budget(stage)
        max_output_tokens = self._effective_max_output_tokens(
            budget,
            kwargs=kwargs,
        )
        max_input_tokens = min(
            budget.max_input_tokens,
            max(
                self._model_context_tokens
                - max_output_tokens
                - budget.safety_margin_tokens,
                0,
            ),
        )
        if max_input_tokens <= 0:
            raise ValueError(
                f"No input budget remains for {stage.value} after reserving "
                "output and safety margin"
            )
        return budget.model_copy(
            update={
                "max_input_tokens": max_input_tokens,
                "max_output_tokens": max_output_tokens,
            }
        )

    async def agenerate_text(
        self,
        *,
        stage: LLMCallStage,
        prompt: str,
        ledger: AsyncBudgetLedger | None = None,
        lease_id: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> LLMCallResult[str]:
        effective_ledger = ledger or current_llm_budget_ledger()
        _, call_kwargs, input_tokens, reservation = self._prepare_call(
            stage=stage,
            prompt=prompt,
            kwargs=kwargs,
        )
        effective_lease_id = lease_id or f"{stage.value}:{id(prompt)}"
        if effective_ledger is not None:
            reserved = await effective_ledger.reserve(effective_lease_id, reservation)
            if not reserved:
                raise LLMBudgetExceededError(
                    stage=stage,
                    required_tokens=reservation,
                )

        try:
            provider_result = await asyncio.to_thread(
                self._invoke_text,
                prompt,
                call_kwargs,
            )
        except Exception:
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise

        usage = provider_result.usage or LLMUsage(
            input_tokens=input_tokens,
            output_tokens=self._token_accounting.count(provider_result.value),
            source="tokenizer_estimate",
        )
        if effective_ledger is not None:
            await effective_ledger.commit(effective_lease_id, usage.total_tokens)
        return LLMCallResult(
            value=provider_result.value,
            usage=usage,
            stage=stage,
        )

    async def agenerate_structured[T: BaseModel](
        self,
        *,
        stage: LLMCallStage,
        prompt: str,
        schema: type[T],
        ledger: AsyncBudgetLedger | None = None,
        lease_id: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> LLMCallResult[T]:
        effective_ledger = ledger or current_llm_budget_ledger()
        accounted_prompt = structured_accounted_prompt(prompt, schema)
        budget, call_kwargs, input_tokens, reservation = self._prepare_call(
            stage=stage,
            prompt=accounted_prompt,
            kwargs=kwargs,
        )
        effective_lease_id = lease_id or f"{stage.value}:{id(prompt)}"
        if effective_ledger is not None:
            reserved = await effective_ledger.reserve(effective_lease_id, reservation)
            if not reserved:
                raise LLMBudgetExceededError(
                    stage=stage,
                    required_tokens=reservation,
                )

        try:
            provider_result = await asyncio.to_thread(
                self._invoke_structured,
                prompt,
                schema,
                call_kwargs,
            )
        except Exception:
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise

        usage = provider_result.usage or LLMUsage(
            input_tokens=input_tokens,
            output_tokens=self._token_accounting.count(
                provider_result.value.model_dump_json()
            ),
            source="tokenizer_estimate",
        )
        if effective_ledger is not None:
            await effective_ledger.commit(effective_lease_id, usage.total_tokens)
        return LLMCallResult(
            value=provider_result.value,
            usage=usage,
            stage=stage,
        )

    async def agenerate_with_tools(
        self,
        *,
        stage: LLMCallStage,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ledger: AsyncBudgetLedger | None = None,
        lease_id: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> LLMCallResult[Any]:
        """Native tool calling path with budget accounting.

        ``messages`` and ``tools`` are in OpenAI wire format.  Returns the
        raw provider response (caller parses via ``OpenAIAdapter``).
        Falls back to ``generate_text`` when the generator lacks
        ``generate_with_tools``.
        """
        effective_ledger = ledger or current_llm_budget_ledger()
        accounted_prompt = _account_messages(messages, tools)
        budget, call_kwargs, input_tokens, reservation = self._prepare_call(
            stage=stage,
            prompt=accounted_prompt,
            kwargs=kwargs,
        )
        effective_lease_id = lease_id or f"{stage.value}:tools:{id(messages)}"
        if effective_ledger is not None:
            reserved = await effective_ledger.reserve(effective_lease_id, reservation)
            if not reserved:
                raise LLMBudgetExceededError(
                    stage=stage,
                    required_tokens=reservation,
                )

        try:
            provider_result = await asyncio.to_thread(
                self._invoke_with_tools,
                messages,
                tools,
                call_kwargs,
            )
        except Exception:
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise

        usage = provider_result.usage or LLMUsage(
            input_tokens=input_tokens,
            output_tokens=0,  # unknown for raw responses
            source="tokenizer_estimate",
        )
        if effective_ledger is not None:
            await effective_ledger.commit(effective_lease_id, usage.total_tokens)
        return LLMCallResult(
            value=provider_result.value,
            usage=usage,
            stage=stage,
        )

    def generate_text(
        self,
        *,
        stage: LLMCallStage,
        prompt: str,
        kwargs: Mapping[str, Any] | None = None,
    ) -> LLMCallResult[str]:
        _, call_kwargs, input_tokens, _ = self._prepare_call(
            stage=stage,
            prompt=prompt,
            kwargs=kwargs,
        )
        provider_result = self._invoke_text(prompt, call_kwargs)
        usage = provider_result.usage or LLMUsage(
            input_tokens=input_tokens,
            output_tokens=self._token_accounting.count(provider_result.value),
            source="tokenizer_estimate",
        )
        return LLMCallResult(
            value=provider_result.value,
            usage=usage,
            stage=stage,
        )

    def generate_structured[T: BaseModel](
        self,
        *,
        stage: LLMCallStage,
        prompt: str,
        schema: type[T],
        kwargs: Mapping[str, Any] | None = None,
    ) -> LLMCallResult[T]:
        accounted_prompt = structured_accounted_prompt(prompt, schema)
        _, call_kwargs, input_tokens, _ = self._prepare_call(
            stage=stage,
            prompt=accounted_prompt,
            kwargs=kwargs,
        )
        provider_result = self._invoke_structured(
            prompt,
            schema,
            call_kwargs,
        )
        usage = provider_result.usage or LLMUsage(
            input_tokens=input_tokens,
            output_tokens=self._token_accounting.count(
                provider_result.value.model_dump_json()
            ),
            source="tokenizer_estimate",
        )
        return LLMCallResult(
            value=provider_result.value,
            usage=usage,
            stage=stage,
        )

    def _stage_budget(self, stage: LLMCallStage) -> LLMStageBudget:
        try:
            return self._stage_budgets[stage]
        except KeyError as exc:
            raise ValueError(f"No LLM stage budget configured for {stage.value}") from exc

    def _prepare_call(
        self,
        *,
        stage: LLMCallStage,
        prompt: str,
        kwargs: Mapping[str, Any] | None,
    ) -> tuple[LLMStageBudget, dict[str, Any], int, int]:
        budget = self.effective_stage_budget(stage, kwargs=kwargs)
        call_kwargs = dict(kwargs or {})
        max_output_tokens = budget.max_output_tokens
        call_kwargs["max_tokens"] = max_output_tokens

        max_input_tokens = budget.max_input_tokens
        input_tokens = self._token_accounting.count(prompt)
        if input_tokens > max_input_tokens:
            raise LLMContextOverflowError(
                stage=stage,
                input_tokens=input_tokens,
                max_input_tokens=max_input_tokens,
            )
        return (
            budget,
            call_kwargs,
            input_tokens,
            input_tokens + max_output_tokens,
        )

    @staticmethod
    def _effective_max_output_tokens(
        budget: LLMStageBudget,
        *,
        kwargs: Mapping[str, Any] | None,
    ) -> int:
        requested_output = (kwargs or {}).get("max_tokens")
        if isinstance(requested_output, int) and requested_output > 0:
            return min(budget.max_output_tokens, requested_output)
        return budget.max_output_tokens

    def _invoke_text(
        self,
        prompt: str,
        kwargs: dict[str, Any],
    ) -> LLMProviderResult[str]:
        with_usage = getattr(self._generator, "generate_text_with_usage", None)
        if callable(with_usage):
            result = with_usage(prompt=prompt, **kwargs)
            if isinstance(result, LLMProviderResult):
                return result
            raise TypeError("generate_text_with_usage must return LLMProviderResult")

        generate_text = getattr(self._generator, "generate_text", None)
        if callable(generate_text):
            return LLMProviderResult(value=str(generate_text(prompt=prompt, **kwargs)))

        chat = getattr(self._generator, "chat", None)
        if callable(chat):
            return LLMProviderResult(value=str(chat(prompt, **kwargs)))

        raise RuntimeError("Configured generator cannot generate text")

    def _invoke_structured[T: BaseModel](
        self,
        prompt: str,
        schema: type[T],
        kwargs: dict[str, Any],
    ) -> LLMProviderResult[T]:
        with_usage = getattr(self._generator, "generate_structured_with_usage", None)
        if callable(with_usage):
            result = with_usage(prompt=prompt, schema=schema, **kwargs)
            if isinstance(result, LLMProviderResult):
                return LLMProviderResult(
                    value=schema.model_validate(result.value),
                    usage=result.usage,
                )
            raise TypeError(
                "generate_structured_with_usage must return LLMProviderResult"
            )

        generate_structured = getattr(self._generator, "generate_structured", None)
        if callable(generate_structured):
            return LLMProviderResult(
                value=schema.model_validate(
                    generate_structured(prompt=prompt, schema=schema, **kwargs)
                )
            )
        raise RuntimeError("Configured generator cannot generate structured output")

    def _invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> LLMProviderResult[Any]:
        generate_with_tools = getattr(
            self._generator, "generate_with_tools", None
        )
        if callable(generate_with_tools):
            result = generate_with_tools(
                messages=messages, tools=tools, **kwargs
            )
            if isinstance(result, LLMProviderResult):
                return result
            raise TypeError("generate_with_tools must return LLMProviderResult")

        # Fallback: render messages as prompt, call generate_text
        prompt = _render_messages_as_prompt(messages)
        return self._invoke_text(prompt, kwargs)


def _render_messages_as_prompt(messages: list[dict[str, Any]]) -> str:
    """Render OpenAI messages as a flat prompt for fallback path."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "user":
            parts.append(f"[User]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        elif role == "tool":
            parts.append(f"[Tool Result: {msg.get('tool_call_id', '')}]\n{content}")
    return "\n\n".join(parts)


def _account_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    """Approximate token-countable text from messages + tools."""
    prompt = _render_messages_as_prompt(messages)
    if tools:
        tool_desc = "\n".join(
            t.get("function", {}).get("name", "") for t in tools
        )
        prompt += f"\n\n[Tools]\n{tool_desc}"
    return prompt


def structured_accounted_prompt(
    prompt: str,
    schema: type[BaseModel],
) -> str:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return f"{prompt}\n\nJSON schema:\n{schema_json}"


__all__ = [
    "LLMBudgetExceededError",
    "LLMContextOverflowError",
    "LLMGateway",
    "LLMGatewayError",
    "current_llm_budget_ledger",
    "llm_budget_scope",
    "structured_accounted_prompt",
]
