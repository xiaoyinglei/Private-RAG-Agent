from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from rag.agent.core.messages import (
    ModelMessage,
    StopReason,
    ToolCall,
    ToolUseResult,
    canonical_json_text,
    snapshot_model_message,
)
from rag.agent.core.model_request import (
    ModelRequest,
    ToolChoice,
    ToolChoiceMode,
    canonical_hash,
    freeze_json_mapping,
)
from rag.agent.tools.tool import JsonValue, json_schema_output

OPENAI_WIRE_REVISION = "openai-compatible-chat-v1"
_RESERVED_PAYLOAD_FIELDS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "seed",
    }
)
_CACHE_CONTROLLED_FIELDS = frozenset(
    {
        "prompt_cache_key",
        "prompt_cache_retention",
    }
)
_SERIALIZER_OWNED_FIELDS = _RESERVED_PAYLOAD_FIELDS | _CACHE_CONTROLLED_FIELDS


@dataclass(frozen=True, slots=True)
class OpenAIWireRequest:
    payload: Mapping[str, JsonValue]
    serialized_json: str
    provider_wire_hash: str
    serializer_revision: str = OPENAI_WIRE_REVISION


def serialize_openai_request(
    request: ModelRequest,
    *,
    cache_key: str | None = None,
    cache_parameters: Mapping[str, JsonValue] | None = None,
    supported_cache_parameters: Collection[str] = (),
) -> OpenAIWireRequest:
    """Serialize one already-selected canonical request to chat wire data."""

    if not isinstance(request, ModelRequest):
        raise TypeError("request must be a ModelRequest")
    supported = _supported_parameter_names(supported_cache_parameters)
    payload: dict[str, JsonValue] = {
        "model": request.settings.model,
        "messages": tuple(_message_payload(message) for message in request.messages),
        "max_completion_tokens": request.settings.max_output_tokens,
        "temperature": float(request.settings.temperature),
        "parallel_tool_calls": request.settings.parallel_tool_calls,
    }
    if request.settings.top_p is not None:
        payload["top_p"] = float(request.settings.top_p)
    if request.settings.seed is not None:
        payload["seed"] = request.settings.seed
    if request.tools:
        payload["tools"] = tuple(
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_schema,
                },
            }
            for definition in request.tools
        )
        payload["tool_choice"] = _tool_choice_payload(request.tool_choice)
    elif request.tool_choice.mode is not ToolChoiceMode.NONE:
        payload["tool_choice"] = _tool_choice_payload(request.tool_choice)

    for key, value in request.settings.provider_options.items():
        if key in _SERIALIZER_OWNED_FIELDS:
            raise ValueError(f"provider option cannot override serializer-owned field: {key}")
        payload[key] = value

    if "prompt_cache_key" in supported:
        effective_cache_key = cache_key or request.prompt_revision
        if not isinstance(effective_cache_key, str) or not effective_cache_key:
            raise ValueError("cache_key must be a non-empty string")
        payload["prompt_cache_key"] = effective_cache_key

    if cache_parameters is not None:
        if not isinstance(cache_parameters, Mapping):
            raise TypeError("cache_parameters must be a mapping")
        for key, value in cache_parameters.items():
            if not isinstance(key, str) or not key:
                raise TypeError("cache parameter names must be non-empty strings")
            if key == "prompt_cache_key" or key not in supported:
                continue
            if key in _RESERVED_PAYLOAD_FIELDS or key in request.settings.provider_options:
                raise ValueError(f"cache parameter cannot override request field: {key}")
            payload[key] = value

    frozen_payload = freeze_json_mapping(payload)
    serialized = canonical_json_text(frozen_payload)
    wire_hash = "wire_" + canonical_hash(
        {
            "serializer_revision": OPENAI_WIRE_REVISION,
            "payload": frozen_payload,
        }
    )
    return OpenAIWireRequest(
        payload=frozen_payload,
        serialized_json=serialized,
        provider_wire_hash=wire_hash,
    )


def parse_openai_response(response: object) -> ToolUseResult:
    """Parse chat response data into the shared provider-neutral turn value."""

    choices = _field(response, "choices", default=None)
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        raise ValueError("response choices must be a non-empty sequence")
    if not choices:
        raise ValueError("response choices must not be empty")
    choice = choices[0]
    message = _field(choice, "message", default=None)
    if message is None:
        raise ValueError("response choice is missing message")
    raw_stop = _field(choice, "finish_reason", default="unknown")
    raw_stop_text = str(raw_stop or "unknown")
    if raw_stop_text in {"tool_calls", "tool_use"}:
        stop_reason = StopReason.TOOL_USE
    elif raw_stop_text == "length":
        stop_reason = StopReason.MAX_TOKENS
    else:
        stop_reason = StopReason.END_TURN

    raw_calls = _field(message, "tool_calls", default=())
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        raise ValueError("message tool_calls must be a sequence")
    calls: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls, start=1):
        function = _field(raw_call, "function", default=None)
        if function is None:
            raise ValueError("tool call is missing function")
        name = _field(function, "name", default="")
        if not isinstance(name, str) or not name:
            raise ValueError("tool call function name must be non-empty")
        raw_arguments = _field(function, "arguments", default={})
        arguments = _parse_arguments(raw_arguments)
        call_id = _field(raw_call, "id", default=None)
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{index}"
        calls.append(ToolCall(id=call_id, name=name, input=arguments))

    content = _field(message, "content", default="")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)
    return ToolUseResult(
        tool_calls=calls,
        text=content,
        stop_reason=stop_reason,
        raw_stop_reason=raw_stop_text,
    )


def _message_payload(message: ModelMessage) -> Mapping[str, JsonValue]:
    message = snapshot_model_message(message)
    if message.role in {"system", "user"}:
        return {"role": message.role, "content": message.content}
    if message.role == "context":
        return {"role": "system", "content": message.content}
    if message.role == "assistant":
        payload: dict[str, JsonValue] = {
            "role": "assistant",
            "content": message.content or None,
        }
        if message.tool_calls:
            payload["tool_calls"] = tuple(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": canonical_json_text(json_schema_output(None, call.input)),
                    },
                }
                for call in message.tool_calls
            )
        return payload
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }
    raise ValueError(f"unsupported model message role: {message.role}")


def _tool_choice_payload(choice: ToolChoice) -> JsonValue:
    if choice.mode is ToolChoiceMode.NAMED:
        assert choice.name is not None
        return {
            "type": "function",
            "function": {"name": choice.name},
        }
    return choice.mode.value


def _parse_arguments(raw: object) -> dict[str, object]:
    if isinstance(raw, Mapping):
        parsed: object = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"_raw": raw[:20_000]}
    else:
        parsed = {"_raw": str(raw)[:20_000]}
    if not isinstance(parsed, Mapping):
        parsed = {"_raw": parsed}
    frozen = json_schema_output(None, cast(JsonValue, parsed))
    if not isinstance(frozen, Mapping):
        raise TypeError("parsed tool arguments must be an object")
    return {key: _thaw_json(value) for key, value in frozen.items()}


def _thaw_json(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _supported_parameter_names(values: Collection[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise TypeError("supported_cache_parameters must be a collection of names")
    names: set[str] = set()
    for name in values:
        if not isinstance(name, str) or not name:
            raise TypeError("supported cache parameter names must be non-empty strings")
        names.add(name)
    return frozenset(names)


def _field(raw: object, name: str, *, default: object) -> object:
    if isinstance(raw, Mapping):
        return raw.get(name, default)
    return getattr(raw, name, default)


__all__ = [
    "OPENAI_WIRE_REVISION",
    "OpenAIWireRequest",
    "parse_openai_response",
    "serialize_openai_request",
]
