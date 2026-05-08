from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from langgraph.graph.message import BaseMessage

from rag.agent.memory.models import (
    ContextBudgetSnapshot,
    ExtractedFact,
    WorkingMemoryDehydration,
    WorkingSummary,
)
from rag.utils.text import text_unit_count


class WorkingMemoryDehydrator:
    """Deterministically compact old messages into bounded working memory.

    This component does not infer semantic facts from free text. It only carries
    forward explicitly supplied working-memory facts from message metadata.
    """

    def __init__(
        self,
        *,
        tail_message_count: int = 8,
        max_summary_chars: int = 4000,
        max_context_tokens: int = 0,
    ) -> None:
        if tail_message_count < 0:
            raise ValueError("tail_message_count must be non-negative")
        if max_summary_chars <= 0:
            raise ValueError("max_summary_chars must be positive")
        self._tail_message_count = tail_message_count
        self._max_summary_chars = max_summary_chars
        self._max_context_tokens = max_context_tokens

    def dehydrate(
        self,
        messages: Sequence[BaseMessage],
        *,
        now_iso: str | None = None,
    ) -> WorkingMemoryDehydration:
        indexed_messages = list(messages)
        tail_start = self._tail_start_index(indexed_messages)
        covered = indexed_messages[:tail_start]
        tail = indexed_messages[tail_start:]
        working_summary = self._build_summary(covered, now_iso=now_iso) if covered else None
        facts = self._extract_explicit_facts(covered)
        context_budget = ContextBudgetSnapshot(
            max_context_tokens=self._max_context_tokens,
            working_memory_tokens=0 if working_summary is None else working_summary.token_count,
            message_tail_tokens=sum(text_unit_count(self._message_text(message)) for message in tail),
        )
        return WorkingMemoryDehydration(
            working_summary=working_summary,
            extracted_facts=facts,
            tail_messages=tail,
            context_budget=context_budget,
        )

    def _tail_start_index(self, messages: list[BaseMessage]) -> int:
        if not messages:
            return 0
        if self._tail_message_count == 0:
            return len(messages)
        start = max(0, len(messages) - self._tail_message_count)
        return self._extend_tail_for_tool_pairs(messages, start)

    @staticmethod
    def _extend_tail_for_tool_pairs(messages: list[BaseMessage], start: int) -> int:
        required_tool_call_ids = {
            tool_call_id
            for message in messages[start:]
            if (tool_call_id := getattr(message, "tool_call_id", None))
        }
        if not required_tool_call_ids:
            return start
        earliest = start
        for index in range(start - 1, -1, -1):
            tool_calls = getattr(messages[index], "tool_calls", None) or []
            call_ids = {str(call.get("id")) for call in tool_calls if call.get("id")}
            if call_ids & required_tool_call_ids:
                earliest = index
                required_tool_call_ids -= call_ids
                if not required_tool_call_ids:
                    break
        return earliest

    def _build_summary(
        self,
        messages: list[BaseMessage],
        *,
        now_iso: str | None,
    ) -> WorkingSummary:
        lines = []
        for message in messages:
            message_id = self._message_id(message)
            role = getattr(message, "type", message.__class__.__name__)
            text = self._message_text(message).replace("\n", " ").strip()
            if text:
                lines.append(f"{message_id} [{role}]: {text}")
            else:
                lines.append(f"{message_id} [{role}]: <empty>")
        summary = self._truncate("\n".join(lines))
        return WorkingSummary(
            summary=summary,
            covered_message_ids=[self._message_id(message) for message in messages],
            updated_at=now_iso or datetime.now(UTC).isoformat(),
            token_count=text_unit_count(summary),
        )

    def _extract_explicit_facts(self, messages: list[BaseMessage]) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        seen: set[str] = set()
        for message in messages:
            message_id = self._message_id(message)
            for raw_fact in message.additional_kwargs.get("working_memory_facts", []):
                fact = self._coerce_fact(raw_fact, source_message_id=message_id)
                if fact.fact_id in seen:
                    continue
                seen.add(fact.fact_id)
                facts.append(fact)
        return facts

    def _coerce_fact(self, raw_fact: object, *, source_message_id: str) -> ExtractedFact:
        if isinstance(raw_fact, str):
            text = raw_fact.strip()
            return ExtractedFact(
                fact_id=self._fact_id(text),
                text=text,
                source_message_ids=[source_message_id],
            )
        if isinstance(raw_fact, dict):
            payload: dict[str, Any] = dict(raw_fact)
            existing_sources = list(payload.get("source_message_ids") or [])
            if source_message_id not in existing_sources:
                existing_sources.append(source_message_id)
            payload["source_message_ids"] = existing_sources
            if "fact_id" not in payload:
                payload["fact_id"] = self._fact_id(str(payload.get("text", "")))
            return ExtractedFact.model_validate(payload)
        raise ValueError(f"Unsupported working_memory_facts item: {type(raw_fact).__name__}")

    @staticmethod
    def _fact_id(text: str) -> str:
        return f"fact_{sha256(text.encode('utf-8')).hexdigest()[:16]}"

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_summary_chars:
            return text
        return text[: self._max_summary_chars].rstrip()

    @staticmethod
    def _message_id(message: BaseMessage) -> str:
        return message.id or f"message_{sha256(repr(message).encode('utf-8')).hexdigest()[:16]}"

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(part) for part in content)
        return str(content)
