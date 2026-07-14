from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from openai.types.chat import ChatCompletion

type ChatCompletionFactory = Callable[..., ChatCompletion]


@pytest.fixture
def chat_completion_factory() -> ChatCompletionFactory:
    def create(
        *,
        content: str | None = "fixture answer",
        tool_calls: list[dict[str, Any]] | None = None,
        finish_reason: str = "stop",
        prompt_tokens: int = 11,
        completion_tokens: int = 3,
        cached_tokens: int | None = 5,
        reasoning_tokens: int | None = None,
    ) -> ChatCompletion:
        usage: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if cached_tokens is not None:
            usage["prompt_tokens_details"] = {
                "cached_tokens": cached_tokens,
            }
        if reasoning_tokens is not None:
            usage["completion_tokens_details"] = {
                "reasoning_tokens": reasoning_tokens,
            }
        return ChatCompletion.model_validate(
            {
                "id": "chatcmpl-provider-fixture",
                "created": 1,
                "model": "fixture-model",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": tool_calls,
                        },
                    }
                ],
                "usage": usage,
            }
        )

    return create
