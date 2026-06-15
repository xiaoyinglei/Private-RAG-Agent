"""Prompt assembly and OpenAI-compatible tool schema translation.

Three responsibilities:
1. ``AgentMessageAssembler`` — section-based system prompt generation
2. ``OpenAIAdapter`` — ModelMessage/ToolSpec ↔ OpenAI wire format
3. ``PromptMessageRenderer`` — fallback text rendering for non-tool models
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from rag.agent.core.messages import (
    ModelMessage,
    StopReason,
    ToolCall,
    ToolUseResult,
)
from rag.agent.tools.spec import ToolSpec

if TYPE_CHECKING:
    from rag.agent.loop.state import LoopState


# ── System prompt sections ──


SYSTEM_PROMPT_DYNAMIC_BOUNDARY = (
    "\n\n--- DYNAMIC RUNTIME CONTEXT BELOW ---\n\n"
)


@dataclass(frozen=True)
class SystemPromptSection:
    """One block of the system prompt.

    ``cache_scope`` controls whether the section is part of the stable
    prefix (provider prompt cache friendly) or the dynamic suffix.
    """

    name: str
    content: str
    cache_scope: Literal["stable", "dynamic"] = "stable"


class AgentMessageAssembler:
    """Builds the system message from ordered sections.

    Stable sections are identical across turns → provider prompt cache
    hits.  Dynamic sections carry per-turn state (iteration, budget).
    The boundary marker separates the two regions.
    """

    def build_system_message(
        self,
        *,
        definition: Any,  # AgentDefinition
        state: LoopState,
        budget_remaining: int | None,
    ) -> ModelMessage:
        sections = [
            self._identity_section(definition),
            self._loop_contract_section(),
            self._tool_use_policy_section(),
            self._output_style_section(definition),
            self._workspace_policy_section(),
            SystemPromptSection(
                name="boundary",
                content=SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
                cache_scope="dynamic",
            ),
            self._runtime_state_section(
                state=state,
                budget_remaining=budget_remaining,
            ),
        ]
        content = "\n\n".join(s.content for s in sections)
        return ModelMessage(role="system", content=content)

    # ── stable sections ──

    @staticmethod
    def _identity_section(definition: Any) -> SystemPromptSection:
        agent_type = getattr(definition, "agent_type", "agent")
        description = getattr(definition, "description", "")
        system_prompt = getattr(definition, "system_prompt", "")
        parts = [f"You are {agent_type}."]
        if description:
            parts.append(description)
        if system_prompt:
            parts.append(system_prompt)
        return SystemPromptSection(
            name="identity",
            content="\n\n".join(parts),
            cache_scope="stable",
        )

    @staticmethod
    def _loop_contract_section() -> SystemPromptSection:
        return SystemPromptSection(
            name="loop_contract",
            content=(
                "You operate in a bounded agent loop.  Each turn you may:\n"
                "- Call one or more tools to advance the task.\n"
                "- Finish with a complete answer when trusted context suffices.\n"
                "- Pause only when external input or authorization is genuinely required.\n"
                "\n"
                "Do not repeat completed tool calls.  Read prior results before "
                "choosing a new call.  Preserve citation identifiers, evidence "
                "links, scores, and artifact paths in your answer."
            ),
            cache_scope="stable",
        )

    @staticmethod
    def _tool_use_policy_section() -> SystemPromptSection:
        return SystemPromptSection(
            name="tool_use_policy",
            content=(
                "Tool calls are structured invocations, not free-form text.  "
                "When you need a tool, request it through the tool calling "
                "mechanism provided.  Do not embed tool arguments in your "
                "text response.  Keep arguments bounded — do not place full "
                "documents, tables, or logs inside tool arguments."
            ),
            cache_scope="stable",
        )

    @staticmethod
    def _output_style_section(definition: Any) -> SystemPromptSection:
        return SystemPromptSection(
            name="output_style",
            content=(
                "Respond concisely.  Cite sources when available.  "
                "Do not fabricate references or evidence."
            ),
            cache_scope="stable",
        )

    @staticmethod
    def _workspace_policy_section() -> SystemPromptSection:
        return SystemPromptSection(
            name="workspace_policy",
            content=(
                "File operations are restricted to the workspace.  "
                "Write only to scratch/, artifacts/, reports/, or logs/.  "
                "Read operations are bounded by size limits."
            ),
            cache_scope="stable",
        )

    # ── dynamic sections ──

    @staticmethod
    def _runtime_state_section(
        *,
        state: dict[str, Any],
        budget_remaining: int | None,
    ) -> SystemPromptSection:
        iteration = state.get("iteration", 0)
        task = state.get("task", "")
        tool_results = state.get("tool_results", [])
        ok_count = sum(
            1 for r in tool_results if getattr(r, "status", None) == "ok"
        )
        error_count = sum(
            1 for r in tool_results if getattr(r, "status", None) == "error"
        )
        lines = [
            f"Task: {task}",
            f"Iteration: {iteration}",
        ]
        if budget_remaining is not None:
            lines.append(f"Remaining token budget: {budget_remaining}")
        lines.append(f"Tool results: {ok_count} successful, {error_count} failed")
        return SystemPromptSection(
            name="runtime_state",
            content="\n".join(lines),
            cache_scope="dynamic",
        )


# ── OpenAI adapter ──


class OpenAIAdapter:
    """Bidirectional translation between internal types and OpenAI wire format.

    All methods are static / stateless — no instance needed.
    """

    @staticmethod
    def tools(specs: list[ToolSpec]) -> list[dict[str, Any]]:
        """Convert ToolSpec list to OpenAI ``tools=`` parameter.

        Stable-sorted by name for deterministic cache keys.
        """
        ordered = sorted(specs, key=lambda s: s.name)
        result: list[dict[str, Any]] = []
        for spec in ordered:
            result.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_model.model_json_schema(),
                },
            })
        return result

    @staticmethod
    def messages(msgs: list[ModelMessage]) -> list[dict[str, Any]]:
        """Convert ModelMessage list to OpenAI messages format."""
        result: list[dict[str, Any]] = []
        for msg in msgs:
            if msg.role == "system":
                result.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                result.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or None,
                }
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(
                                    tc.input, ensure_ascii=False
                                ),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(entry)
            elif msg.role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id or "",
                    "content": msg.content,
                })
        return result

    @staticmethod
    def parse_tool_calls(response: Any) -> ToolUseResult:
        """Parse an OpenAI chat completion response into ToolUseResult.

        Includes fallbacks for:
        - Missing tool_call_id (generate one)
        - arguments as JSON string (parse into dict)
        - arguments as malformed JSON (wrap in ``{"_raw": ...}``)
        """
        choice = response.choices[0]
        message = choice.message
        raw_stop = getattr(choice, "finish_reason", None) or "unknown"

        # Normalize stop reason
        if raw_stop == "tool_calls" or raw_stop == "tool_use":
            stop_reason = StopReason.TOOL_USE
        elif raw_stop == "length":
            stop_reason = StopReason.MAX_TOKENS
        else:
            stop_reason = StopReason.END_TURN

        # Parse tool calls
        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        for tc in raw_tool_calls:
            tc_id = getattr(tc, "id", None) or f"tc_{uuid4().hex[:12]}"
            func = getattr(tc, "function", None)
            if func is None:
                continue
            name = getattr(func, "name", "") or ""
            raw_args = getattr(func, "arguments", "{}") or "{}"

            # Parse arguments — handle JSON string, dict, or malformed
            if isinstance(raw_args, dict):
                parsed_args = raw_args
            elif isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = {"_raw": raw_args}
            else:
                parsed_args = {"_raw": str(raw_args)}

            if not isinstance(parsed_args, dict):
                parsed_args = {"_raw": parsed_args}

            tool_calls.append(ToolCall(id=tc_id, name=name, input=parsed_args))

        text = message.content or ""

        return ToolUseResult(
            tool_calls=tool_calls,
            text=text,
            stop_reason=stop_reason,
            raw_stop_reason=str(raw_stop),
        )


# ── Prompt fallback renderer ──


class PromptMessageRenderer:
    """Render messages + tool specs as a single prompt string.

    Used as fallback when the model does not support native tool calling.
    Receives ``ToolSpec`` (internal type), not OpenAI api_tools dicts.
    """

    @staticmethod
    def render(
        messages: list[ModelMessage],
        tools: list[ToolSpec],
    ) -> str:
        parts: list[str] = []

        for msg in messages:
            if msg.role == "system":
                parts.append(f"[System]\n{msg.content}")
            elif msg.role == "user":
                parts.append(f"[User]\n{msg.content}")
            elif msg.role == "assistant":
                if msg.tool_calls:
                    calls_desc = "\n".join(
                        f"  - {tc.name}({json.dumps(tc.input, ensure_ascii=False)})"
                        for tc in msg.tool_calls
                    )
                    parts.append(
                        f"[Assistant - Tool Calls]\n{msg.content or ''}\n{calls_desc}"
                    )
                else:
                    parts.append(f"[Assistant]\n{msg.content}")
            elif msg.role == "tool":
                parts.append(
                    f"[Tool Result: {msg.tool_call_id}]\n{msg.content}"
                )

        if tools:
            tool_section = "\n".join(
                f"- {s.name}: {s.description}" for s in sorted(tools, key=lambda t: t.name)
            )
            parts.append(
                f"[Available Tools]\n{tool_section}\n\n"
                "To call a tool, respond with JSON:\n"
                '{{"tool_calls": [{{"tool_name": "...", "arguments": {{...}}}}]}}'
            )

        return "\n\n".join(parts)


__all__ = [
    "AgentMessageAssembler",
    "OpenAIAdapter",
    "PromptMessageRenderer",
    "SYSTEM_PROMPT_DYNAMIC_BOUNDARY",
    "SystemPromptSection",
]
