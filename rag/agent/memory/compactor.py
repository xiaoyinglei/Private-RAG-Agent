from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Protocol, cast

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from rag.agent.memory.models import (
    ContextBudgetSnapshot,
    EvictedStateItem,
    ExternalizedToolOutput,
    ExtractedFact,
    MemoryBudgetSnapshot,
    MemoryPolicy,
    MemoryRef,
    MessageBatchPayload,
    StateChannelReplacement,
    ToolErrorDetailPayload,
    WorkingMemoryDraft,
    WorkingSummary,
)
from rag.agent.tools.spec import ToolResult
from rag.utils.text import text_unit_count

if TYPE_CHECKING:
    from rag.agent.loop.state import LoopState


class ToolOutputMemoryStore(Protocol):
    def write_tool_output(
        self,
        payload: BaseModel,
        *,
        summary: str,
        source_tool_call_id: str | None = None,
        source_tool_name: str | None = None,
        warnings: list[str] | None = None,
    ) -> MemoryRef: ...


@dataclass
class _ExternalizationMetadata:
    externalized_count: int = 0
    unavailable_count: int = 0
    memory_refs: list[MemoryRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class WorkingMemoryCompactor:
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

    def compact(
        self,
        messages: Sequence[BaseMessage],
        *,
        now_iso: str | None = None,
    ) -> WorkingMemoryDraft:
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
        return WorkingMemoryDraft(
            working_summary=working_summary,
            extracted_facts=facts,
            tail_messages=tail,
            context_budget=context_budget,
        )

    def dehydrate(
        self,
        messages: Sequence[BaseMessage],
        *,
        now_iso: str | None = None,
    ) -> WorkingMemoryDraft:
        return self.compact(messages, now_iso=now_iso)

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


class MessageCompactor:
    """Externalize old messages and keep a bounded deterministic message tail."""

    def __init__(
        self,
        *,
        policy: MemoryPolicy,
        store: ToolOutputMemoryStore | None = None,
    ) -> None:
        self._policy = policy
        self._store = store

    def compact_initial_state(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = [
            message for message in state.get("messages", []) if isinstance(message, BaseMessage)
        ]
        if len(messages) < self._policy.message_compaction_min_count:
            return state

        draft = WorkingMemoryCompactor(
            tail_message_count=self._policy.max_message_tail_count,
            max_summary_chars=self._policy.max_working_summary_chars,
            max_context_tokens=0,
        ).compact(messages)
        covered_ids = (
            set(draft.working_summary.covered_message_ids)
            if draft.working_summary is not None
            else set()
        )
        covered_messages = [
            message for message in messages if self._message_id(message) in covered_ids
        ]
        if not covered_messages:
            return state

        update: dict[str, Any] = {
            "messages": list(draft.tail_messages),
            "working_summary": self._merge_summary(
                state.get("working_summary"),
                draft.working_summary,
            ),
            "extracted_facts": self._bounded_facts(
                [
                    *[
                        fact
                        for fact in state.get("extracted_facts", [])
                        if isinstance(fact, ExtractedFact)
                    ],
                    *draft.extracted_facts,
                ]
            ),
        }
        warnings: list[str] = []
        ref = self._write_message_batch(covered_messages, warnings=warnings)
        if ref is not None:
            update["memory_refs"] = [
                *[ref for ref in state.get("memory_refs", []) if isinstance(ref, MemoryRef)],
                ref,
            ]
        if warnings:
            update["memory_warnings"] = [
                *[str(item) for item in state.get("memory_warnings", [])],
                *warnings,
            ]
        return {**state, **update}

    def compact_update(self, state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        combined = {**state, **update}
        compacted = self.compact_initial_state(combined)
        if compacted is combined:
            return update
        result = dict(update)
        for key in (
            "working_summary",
            "extracted_facts",
            "memory_refs",
            "memory_warnings",
        ):
            if compacted.get(key) != combined.get(key):
                result[key] = compacted.get(key)
        if compacted.get("messages") != combined.get("messages"):
            result["messages"] = [StateChannelReplacement(items=list(compacted["messages"]))]
        return result

    def _write_message_batch(
        self,
        messages: list[BaseMessage],
        *,
        warnings: list[str],
    ) -> MemoryRef | None:
        summary = (
            f"message_batch messages={len(messages)} "
            f"first_id={self._message_id(messages[0])} last_id={self._message_id(messages[-1])}"
        )
        payload = MessageBatchPayload(messages=messages)
        payload_chars = len(payload.model_dump_json())
        if payload_chars > self._policy.max_message_batch_chars:
            warnings.append("message_batch_truncated_to_policy")
            payload = MessageBatchPayload(messages=self._bounded_message_batch(messages))
        if self._store is None:
            warnings.append("memory_unavailable")
            return None
        try:
            return self._store.write_tool_output(
                payload,
                summary=summary,
                source_tool_name="message_compaction",
            )
        except Exception:
            warnings.append("message_compaction_failed")
            return None

    def _bounded_message_batch(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        bounded: list[BaseMessage] = []
        total = 0
        for message in reversed(messages):
            size = len(self._message_text(message))
            if bounded and total + size > self._policy.max_message_batch_chars:
                break
            bounded.append(message)
            total += size
        return list(reversed(bounded))

    def _merge_summary(
        self,
        existing: object,
        new_summary: WorkingSummary | None,
    ) -> WorkingSummary | None:
        if new_summary is None:
            return existing if isinstance(existing, WorkingSummary) else None
        if not isinstance(existing, WorkingSummary):
            return new_summary.model_copy(
                update={"summary": self._truncate_working_summary(new_summary.summary)}
            )
        merged_text = "\n".join(
            text for text in (existing.summary, new_summary.summary) if text.strip()
        )
        return WorkingSummary(
            summary=self._truncate_working_summary(merged_text),
            covered_message_ids=list(
                dict.fromkeys([*existing.covered_message_ids, *new_summary.covered_message_ids])
            ),
            updated_at=new_summary.updated_at,
            token_count=text_unit_count(self._truncate_working_summary(merged_text)),
        )

    def _bounded_facts(self, facts: list[ExtractedFact]) -> list[ExtractedFact]:
        by_id = {fact.fact_id: fact for fact in facts}
        return list(by_id.values())[-self._policy.max_extracted_facts :]

    def _truncate_working_summary(self, summary: str) -> str:
        if len(summary) <= self._policy.max_working_summary_chars:
            return summary
        return summary[: self._policy.max_working_summary_chars].rstrip() + " [truncated]"

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


class MemoryCompactor:
    """Deterministically externalize large tool outputs and cap long state channels."""

    _CAPPED_CHANNELS: dict[str, str] = {
        "tool_results": "max_tool_results",
        "structured_observations": "max_structured_observations",
        "context_units": "max_context_units",
        "answer_candidates": "max_answer_candidates",
        "computation_results": "max_computation_results",
        "evidence_refs": "max_evidence_refs",
        "evidence": "max_evidence_items",
        "citations": "max_citations",
        "locators": "max_locators",
        "memory_refs": "max_memory_refs",
        "plan_events": "max_plan_events",
    }

    def __init__(
        self,
        *,
        policy: MemoryPolicy,
        store: ToolOutputMemoryStore | None = None,
        loop_mode: bool = False,
    ) -> None:
        self._policy = policy
        self._store = store
        self._loop_mode = loop_mode

    def compact_update(
        self,
        state: dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        compacted = dict(update)
        warnings: list[str] = []
        externalized_count = 0
        unavailable_count = 0
        new_memory_refs: list[MemoryRef] = []
        raw_refs_by_tool_call_id: dict[str, MemoryRef] = {}
        evicted_items: list[EvictedStateItem] = []
        used_channel_counts: dict[str, int] = {}
        pinned_item_count = 0

        tool_results = self._combined_items(state, compacted, "tool_results")
        compacted_tool_results: list[ToolResult] = []
        tool_results_changed = False
        for result in tool_results:
            if not isinstance(result, ToolResult):
                compacted_tool_results.append(result)
                continue
            replacement, metadata = self._maybe_externalize_result(result)
            if replacement is result:
                compacted_tool_results.append(result)
                continue
            tool_results_changed = True
            compacted_tool_results.append(replacement)
            externalized_count += metadata.externalized_count
            unavailable_count += metadata.unavailable_count
            new_memory_refs.extend(metadata.memory_refs)
            warnings.extend(metadata.warnings)
            if ref := self._result_memory_ref(replacement):
                raw_refs_by_tool_call_id[replacement.tool_call_id] = ref

        dropped: dict[str, int] = {}
        pin_context = (
            self._pin_loop_context(
                state,
                compacted,
                new_memory_refs=new_memory_refs,
            )
            if self._loop_mode
            else self._pin_context(
                state,
                compacted,
                new_memory_refs=new_memory_refs,
            )
        )
        for channel, limit_attr in self._CAPPED_CHANNELS.items():
            channel_changed = False
            combined: list[Any]
            if channel == "tool_results":
                combined = compacted_tool_results
                channel_changed = tool_results_changed
            elif channel == "memory_refs" and new_memory_refs:
                combined = [
                    *[
                        ref
                        for ref in state.get("memory_refs", [])
                        if isinstance(ref, MemoryRef)
                    ],
                    *new_memory_refs,
                ]
                channel_changed = True
            else:
                combined = self._combined_items(state, compacted, channel)
            if channel == "structured_observations" and raw_refs_by_tool_call_id:
                combined, attached = self._attach_observation_raw_refs(
                    combined,
                    raw_refs_by_tool_call_id,
                )
                channel_changed = channel_changed or attached
            sanitized, sanitized_changed, sanitize_warnings = self._sanitize_channel_items(
                channel,
                combined,
            )
            if sanitized_changed:
                channel_changed = True
                warnings.extend(sanitize_warnings)
            limit = int(getattr(self._policy, limit_attr))
            bounded, channel_evictions, channel_pins = self._bounded_with_audit(
                channel,
                sanitized,
                limit=limit,
                pinned_keys=pin_context.get(channel, set()),
            )
            used_channel_counts[channel] = len(bounded)
            pinned_item_count += channel_pins
            if channel_evictions:
                channel_changed = True
                evicted_items.extend(channel_evictions)
                dropped[channel] = len(channel_evictions)
            if not channel_changed:
                continue
            if state.get(channel):
                compacted[channel] = [StateChannelReplacement(items=bounded)]
            else:
                compacted[channel] = bounded

        compacted["memory_budget"] = MemoryBudgetSnapshot(
            max_tool_output_chars=self._policy.max_tool_output_chars,
            externalized_record_count=externalized_count,
            unavailable_record_count=unavailable_count,
            memory_ref_count=len(self._combined_items(state, compacted, "memory_refs")),
            compacted_tool_result_count=externalized_count + unavailable_count,
            dropped_state_items=dropped,
            evicted_items=evicted_items,
            used_channel_counts=used_channel_counts,
            pinned_item_count=pinned_item_count,
            warnings=list(dict.fromkeys(warnings)),
        )
        if warnings:
            compacted["memory_warnings"] = list(dict.fromkeys(warnings))
        return compacted

    def summarize_tool_result(self, result: ToolResult) -> str:
        if result.status == "error":
            error = result.error
            if error is None:
                return f"{result.tool_name} error=<missing>"
            return (
                f"{result.tool_name} error_code={error.code} retryable={error.retryable} "
                f"message={self._one_line(error.message)}"
            )
        output = result.output
        if output is None:
            return f"{result.tool_name} output=<missing>"
        if result.tool_name == "run_python":
            return self._summarize_run_python(output)
        if result.tool_name == "structured_probe":
            return self._summarize_structured_probe(output)
        if result.tool_name == "list_files":
            return self._summarize_list_files(output)
        if result.tool_name == "read_file":
            return self._summarize_read_file(output)
        if result.tool_name == "asset_analyze":
            return self._summarize_asset_analyze(output)
        return self._summarize_unknown(output, tool_name=result.tool_name)

    def _maybe_externalize_result(self, result: ToolResult) -> tuple[ToolResult, _ExternalizationMetadata]:
        if result.status == "error":
            return self._maybe_externalize_error_detail(result)
        if result.output is None:
            return result, _ExternalizationMetadata()
        if isinstance(result.output, ExternalizedToolOutput):
            return result, _ExternalizationMetadata()
        output_json = result.output.model_dump_json()
        if len(output_json) <= self._policy.max_tool_output_chars:
            return result, _ExternalizationMetadata()

        summary = self._truncate_summary(self.summarize_tool_result(result))
        original_output_model = _model_path(result.output)
        warnings: list[str] = []
        if self._store is None:
            warnings.append("memory_unavailable")
            ref = MemoryRef(
                ref_id=f"unavailable_{result.tool_call_id}",
                path=f".agent_memory/records/unavailable_{result.tool_call_id}.json",
                summary=summary,
                source_tool_call_id=result.tool_call_id,
                source_tool_name=result.tool_name,
                status="unavailable",
                warnings=warnings,
            )
            replacement = result.model_copy(
                update={
                    "output": ExternalizedToolOutput(
                        original_output_model=original_output_model,
                        summary=summary,
                        ref=ref,
                        status="unavailable",
                        warnings=warnings,
                    )
                }
            )
            return replacement, _ExternalizationMetadata(
                unavailable_count=1,
                memory_refs=[ref],
                warnings=warnings,
            )

        try:
            ref = self._store.write_tool_output(
                result.output,
                summary=summary,
                source_tool_call_id=result.tool_call_id,
                source_tool_name=result.tool_name,
            )
            replacement = result.model_copy(
                update={
                    "output": ExternalizedToolOutput(
                        original_output_model=original_output_model,
                        summary=summary,
                        ref=ref,
                        status="available",
                    )
                }
            )
            return replacement, _ExternalizationMetadata(
                externalized_count=1,
                memory_refs=[ref],
            )
        except Exception as exc:
            warnings.append("memory_compaction_failed")
            ref = MemoryRef(
                ref_id=f"unavailable_{result.tool_call_id}",
                path=f".agent_memory/records/unavailable_{result.tool_call_id}.json",
                summary=summary,
                source_tool_call_id=result.tool_call_id,
                source_tool_name=result.tool_name,
                status="unavailable",
                warnings=warnings,
            )
            result_warnings = [*warnings, self._one_line(str(exc))]
            replacement = result.model_copy(
                update={
                    "output": ExternalizedToolOutput(
                        original_output_model=original_output_model,
                        summary=summary,
                        ref=ref,
                        status="unavailable",
                        warnings=result_warnings,
                    )
                }
            )
            return replacement, _ExternalizationMetadata(
                unavailable_count=1,
                memory_refs=[ref],
                warnings=result_warnings,
            )

    def _maybe_externalize_error_detail(
        self,
        result: ToolResult,
    ) -> tuple[ToolResult, _ExternalizationMetadata]:
        error = result.error
        if error is None or not error.detail:
            return result, _ExternalizationMetadata()
        if isinstance(error.detail.get("externalized_ref"), str):
            return result, _ExternalizationMetadata()
        detail_json = json.dumps(error.detail, ensure_ascii=False, default=str)
        if len(detail_json) <= self._policy.max_tool_output_chars:
            return result, _ExternalizationMetadata()

        summary = self._truncate_summary(
            self._metadata_summary(
                "tool_error_detail",
                tool_name=result.tool_name,
                error_code=error.code,
                retryable=error.retryable,
                detail_chars=len(detail_json),
                detail_keys=self._list_preview(list(error.detail.keys())),
            )
        )
        payload = ToolErrorDetailPayload(
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            detail=error.detail,
        )
        warnings: list[str] = []
        if self._store is None:
            warnings.append("memory_unavailable")
            ref = self._unavailable_ref(
                result,
                summary=summary,
                suffix="error_detail",
                warnings=warnings,
            )
            replacement = result.model_copy(
                update={
                    "error": error.model_copy(
                        update={
                            "detail": self._externalized_error_detail(
                                ref=ref,
                                summary=summary,
                                status="unavailable",
                            )
                        }
                    )
                }
            )
            return replacement, _ExternalizationMetadata(
                unavailable_count=1,
                memory_refs=[ref],
                warnings=warnings,
            )

        try:
            ref = self._store.write_tool_output(
                payload,
                summary=summary,
                source_tool_call_id=result.tool_call_id,
                source_tool_name=result.tool_name,
            )
            replacement = result.model_copy(
                update={
                    "error": error.model_copy(
                        update={
                            "detail": self._externalized_error_detail(
                                ref=ref,
                                summary=summary,
                                status="available",
                            )
                        }
                    )
                }
            )
            return replacement, _ExternalizationMetadata(
                externalized_count=1,
                memory_refs=[ref],
            )
        except Exception as exc:
            warnings.append("memory_compaction_failed")
            ref = self._unavailable_ref(
                result,
                summary=summary,
                suffix="error_detail",
                warnings=warnings,
            )
            result_warnings = [*warnings, self._one_line(str(exc))]
            replacement = result.model_copy(
                update={
                    "error": error.model_copy(
                        update={
                            "detail": {
                                **self._externalized_error_detail(
                                    ref=ref,
                                    summary=summary,
                                    status="unavailable",
                                ),
                                "warnings": result_warnings,
                            }
                        }
                    )
                }
            )
            return replacement, _ExternalizationMetadata(
                unavailable_count=1,
                memory_refs=[ref],
                warnings=result_warnings,
            )

    @staticmethod
    def _result_memory_ref(result: ToolResult) -> MemoryRef | None:
        output = result.output
        if isinstance(output, ExternalizedToolOutput):
            return output.ref
        error = result.error
        detail = None if error is None else error.detail
        if isinstance(detail, dict) and isinstance(detail.get("externalized_ref"), str):
            ref_id = str(detail["externalized_ref"])
            return MemoryRef(
                ref_id=ref_id,
                path=f".agent_memory/records/{ref_id}.json",
                summary=str(detail.get("summary", "")),
                source_tool_call_id=result.tool_call_id,
                source_tool_name=result.tool_name,
                status=cast(Any, detail.get("status", "available")),
            )
        return None

    @staticmethod
    def _attach_observation_raw_refs(
        observations: list[Any],
        refs_by_tool_call_id: dict[str, MemoryRef],
    ) -> tuple[list[Any], bool]:
        changed = False
        attached: list[Any] = []
        for observation in observations:
            tool_call_id = getattr(observation, "tool_call_id", None)
            ref = refs_by_tool_call_id.get(str(tool_call_id))
            if (
                ref is not None
                and getattr(observation, "raw_memory_ref", None) is None
                and hasattr(observation, "model_copy")
            ):
                attached.append(observation.model_copy(update={"raw_memory_ref": ref}))
                changed = True
            else:
                attached.append(observation)
        return attached, changed

    @staticmethod
    def _externalized_error_detail(
        *,
        ref: MemoryRef,
        summary: str,
        status: str,
    ) -> dict[str, object]:
        return {
            "externalized_ref": ref.ref_id,
            "summary": summary,
            "status": status,
        }

    @staticmethod
    def _unavailable_ref(
        result: ToolResult,
        *,
        summary: str,
        suffix: str,
        warnings: list[str],
    ) -> MemoryRef:
        ref_id = f"unavailable_{result.tool_call_id}_{suffix}"
        return MemoryRef(
            ref_id=ref_id,
            path=f".agent_memory/records/{ref_id}.json",
            summary=summary,
            source_tool_call_id=result.tool_call_id,
            source_tool_name=result.tool_name,
            status="unavailable",
            warnings=warnings,
        )

    def _bounded_with_audit(
        self,
        channel: str,
        items: list[Any],
        *,
        limit: int,
        pinned_keys: set[str],
    ) -> tuple[list[Any], list[EvictedStateItem], int]:
        if len(items) <= limit:
            pinned_count = sum(
                1 for item in items if self._is_pinned(channel, item, pinned_keys=pinned_keys)
            )
            return list(items), [], pinned_count

        selected: list[Any] = []
        selected_keys: set[str] = set()
        pinned_count = 0
        for item in items:
            key = _item_key(item)
            if self._is_pinned(channel, item, pinned_keys=pinned_keys):
                selected.append(item)
                selected_keys.add(key)
                pinned_count += 1

        for item in reversed(items):
            key = _item_key(item)
            if key in selected_keys:
                continue
            if len(selected) >= limit:
                break
            selected.append(item)
            selected_keys.add(key)

        kept_keys_in_order = {_item_key(item) for item in selected}
        kept = [item for item in items if _item_key(item) in kept_keys_in_order]
        kept_keys = {_item_key(item) for item in kept}
        evicted = [
            self._evicted_item(channel, item, reason="retention_limit")
            for item in items
            if _item_key(item) not in kept_keys
        ]
        return kept, evicted, min(pinned_count, len(kept))

    def _is_pinned(self, channel: str, item: Any, *, pinned_keys: set[str]) -> bool:
        if _must_preserve(item):
            return True
        key = _item_key(item)
        if key in pinned_keys:
            return True
        if channel == "memory_refs" and isinstance(item, MemoryRef):
            return item.ref_id in pinned_keys
        return False

    def _evicted_item(self, channel: str, item: Any, *, reason: str) -> EvictedStateItem:
        return EvictedStateItem(
            channel=cast(Any, channel),
            key=_item_key(item),
            reason=reason,
            summary=self._eviction_summary(item),
            source_tool_call_id=_source_tool_call_id(item),
            memory_ref_id=_memory_ref_id(item),
        )

    def _sanitize_channel_items(
        self,
        channel: str,
        items: list[Any],
    ) -> tuple[list[Any], bool, list[str]]:
        sanitized: list[Any] = []
        changed = False
        warnings: list[str] = []
        for item in items:
            replacement = self._sanitize_state_item(channel, item)
            if replacement is not item:
                changed = True
                warnings.append("raw_checkpoint_guard_sanitized")
            sanitized.append(replacement)
        return sanitized, changed, list(dict.fromkeys(warnings))

    def _sanitize_state_item(self, channel: str, item: Any) -> Any:
        if channel == "context_units":
            preview = getattr(item, "preview", None)
            sanitized_preview = self._sanitized_text(preview)
            if sanitized_preview is not preview and hasattr(item, "model_copy"):
                return item.model_copy(update={"preview": sanitized_preview})
        if channel == "answer_candidates":
            text = getattr(item, "text", None)
            sanitized_text = self._sanitized_text(text)
            if sanitized_text is not text and hasattr(item, "model_copy"):
                return item.model_copy(update={"text": sanitized_text})
        if channel == "computation_results":
            updates: dict[str, object] = {}
            for attr in ("value_preview", "expression"):
                value = getattr(item, attr, None)
                sanitized = self._sanitized_text(value)
                if sanitized is not value:
                    updates[attr] = sanitized
            if updates and hasattr(item, "model_copy"):
                return item.model_copy(update=updates)
        if channel == "structured_observations":
            return self._sanitize_observation(item)
        return item

    def _sanitize_observation(self, item: Any) -> Any:
        updates: dict[str, object] = {}
        answer = getattr(item, "answer_candidate", None)
        if answer is not None:
            text = getattr(answer, "text", None)
            sanitized = self._sanitized_text(text)
            if sanitized is not text and hasattr(answer, "model_copy"):
                updates["answer_candidate"] = answer.model_copy(update={"text": sanitized})
        context_units = getattr(item, "context_units", None)
        if isinstance(context_units, list):
            sanitized_units, changed, _ = self._sanitize_channel_items("context_units", context_units)
            if changed:
                updates["context_units"] = sanitized_units
        if updates and hasattr(item, "model_copy"):
            return item.model_copy(update=updates)
        return item

    def _sanitized_text(self, value: object) -> object:
        if not isinstance(value, str):
            return value
        if len(value) <= self._policy.max_tool_output_chars:
            return value
        return self._metadata_summary(
            "raw_checkpoint_guard_sanitized",
            original_chars=len(value),
            sha256=sha256(value.encode("utf-8")).hexdigest()[:16],
        )

    def _pin_context(
        self,
        state: dict[str, Any],
        update: dict[str, Any],
        *,
        new_memory_refs: list[MemoryRef],
    ) -> dict[str, set[str]]:
        combined = {**state, **update}
        pins: dict[str, set[str]] = {channel: set() for channel in self._CAPPED_CHANNELS}
        plan = combined.get("agent_plan")
        active_step = _active_plan_step(plan)
        active_tool_call_ids = set(getattr(active_step, "tool_call_ids", []) or [])
        active_evidence_refs = set(getattr(active_step, "evidence_refs", []) or [])

        for tool_call_id in active_tool_call_ids:
            key = f"tool_call_id:{tool_call_id}"
            for channel in (
                "tool_results",
                "structured_observations",
                "answer_candidates",
                "computation_results",
            ):
                pins[channel].add(key)
        for ref in active_evidence_refs:
            pins["evidence_refs"].add(str(ref))
            pins["evidence"].add(str(ref))
            pins["citations"].add(str(ref))

        for unit in self._combined_items(state, update, "context_units"):
            content_ref = getattr(unit, "content_ref", None)
            if isinstance(content_ref, str) and content_ref in active_tool_call_ids:
                pins["context_units"].add(_item_key(unit))
        for ref_id in self._referenced_memory_ref_ids(state, update, new_memory_refs):
            pins["memory_refs"].add(ref_id)
        return pins

    def _pin_loop_context(
        self,
        state: dict[str, Any],
        update: dict[str, Any],
        *,
        new_memory_refs: list[MemoryRef],
    ) -> dict[str, set[str]]:
        combined = {**state, **update}
        pins: dict[str, set[str]] = {
            channel: set() for channel in self._CAPPED_CHANNELS
        }
        active_tool_call_ids = {
            call.tool_call_id
            for call in combined.get("pending_tool_calls", [])
            if getattr(call, "tool_call_id", None)
        }
        approval_request = combined.get("approval_request")
        active_tool_call_ids.update(
            summary.tool_call_id
            for summary in getattr(approval_request, "tool_calls", []) or []
            if getattr(summary, "tool_call_id", None)
        )
        plan = combined.get("agent_plan")
        active_step = _active_plan_step(plan)
        active_tool_call_ids.update(
            str(tool_call_id)
            for tool_call_id in getattr(active_step, "tool_call_ids", []) or []
            if tool_call_id
        )

        for tool_call_id in active_tool_call_ids:
            key = f"tool_call_id:{tool_call_id}"
            for channel in (
                "tool_results",
                "structured_observations",
                "computation_results",
            ):
                pins[channel].add(key)
            pins["answer_candidates"].add(
                f"source_tool_call_id:{tool_call_id}"
            )

        evidence_refs: list[Any] = []
        for ref in getattr(active_step, "evidence_refs", []) or []:
            evidence_refs.append(ref)
        answer_candidates = self._combined_items(
            state,
            update,
            "answer_candidates",
        )
        if answer_candidates:
            current_candidate = answer_candidates[-1]
            pins["answer_candidates"].add(_item_key(current_candidate))
            evidence_refs.extend(
                getattr(current_candidate, "evidence_refs", []) or []
            )
        for observation in self._combined_items(
            state,
            update,
            "structured_observations",
        ):
            if getattr(observation, "tool_call_id", None) not in (
                active_tool_call_ids
            ):
                continue
            evidence_refs.extend(
                getattr(observation, "evidence_refs", []) or []
            )
        for ref in evidence_refs:
            pins["evidence_refs"].add(_item_key(ref))
            evidence_id = getattr(ref, "evidence_id", None)
            if evidence_id:
                pins["evidence"].add(f"evidence_id:{evidence_id}")
            citation_id = getattr(ref, "citation_id", None)
            if citation_id:
                pins["citations"].add(f"citation_id:{citation_id}")

        for unit in self._combined_items(state, update, "context_units"):
            content_ref = getattr(unit, "content_ref", None)
            if isinstance(content_ref, str) and content_ref in active_tool_call_ids:
                pins["context_units"].add(_item_key(unit))
        for ref_id in self._referenced_memory_ref_ids(
            state,
            update,
            new_memory_refs,
        ):
            pins["memory_refs"].add(ref_id)
        return pins

    def _referenced_memory_ref_ids(
        self,
        state: dict[str, Any],
        update: dict[str, Any],
        new_memory_refs: list[MemoryRef],
    ) -> set[str]:
        ref_ids = {ref.ref_id for ref in new_memory_refs}
        for result in self._combined_items(state, update, "tool_results"):
            output = getattr(result, "output", None)
            if isinstance(output, ExternalizedToolOutput):
                ref_ids.add(output.ref.ref_id)
            error = getattr(result, "error", None)
            detail = getattr(error, "detail", None)
            if isinstance(detail, dict) and isinstance(detail.get("externalized_ref"), str):
                ref_ids.add(str(detail["externalized_ref"]))
        for observation in self._combined_items(state, update, "structured_observations"):
            ref = getattr(observation, "raw_memory_ref", None)
            if isinstance(ref, MemoryRef):
                ref_ids.add(ref.ref_id)
        return ref_ids

    def _bounded(self, channel: str, items: list[Any]) -> list[Any]:
        limit = int(getattr(self._policy, self._CAPPED_CHANNELS[channel]))
        if len(items) <= limit:
            return list(items)
        return _bounded_recent(items, limit=limit)

    @staticmethod
    def _combined_items(
        state: dict[str, Any],
        update: dict[str, Any],
        channel: str,
    ) -> list[Any]:
        current = list(state.get(channel, []))
        incoming = list(update.get(channel, []))
        if len(incoming) == 1 and isinstance(incoming[0], StateChannelReplacement):
            return list(incoming[0].items)
        return [*current, *incoming]

    def _summarize_run_python(self, output: BaseModel) -> str:
        values = output.model_dump(mode="json")
        generated_files = values.get("generated_files") or []
        return self._metadata_summary(
            "run_python",
            ok=values.get("ok"),
            exit_code=values.get("exit_code"),
            duration_ms=values.get("duration_ms"),
            stdout_chars=len(str(values.get("stdout") or "")),
            stderr_chars=len(str(values.get("stderr") or "")),
            stdout_truncated=values.get("stdout_truncated"),
            stderr_truncated=values.get("stderr_truncated"),
            generated_files=self._list_preview(generated_files),
        )

    def _summarize_structured_probe(self, output: BaseModel) -> str:
        values = output.model_dump(mode="json")
        parts = [
            self._metadata_summary(
                "structured_probe",
                path=values.get("path"),
                file_kind=values.get("file_kind"),
                mime_type=values.get("mime_type"),
                tables=len(values.get("tables") or []),
                truncated=values.get("truncated"),
            )
        ]
        for table in (values.get("tables") or [])[:5]:
            if not isinstance(table, dict):
                continue
            header = None
            candidates = table.get("candidate_header_rows") or []
            if candidates and isinstance(candidates[0], dict):
                header = candidates[0].get("row_index")
            parts.append(
                self._metadata_summary(
                    "table",
                    name=table.get("name"),
                    used_range=table.get("used_range"),
                    row_count=table.get("row_count"),
                    column_count=table.get("column_count"),
                    header_row=header,
                    data_start_row=table.get("data_start_row"),
                )
            )
        return " | ".join(parts)

    def _summarize_list_files(self, output: BaseModel) -> str:
        values = output.model_dump(mode="json")
        files = values.get("files") or []
        previews: list[str] = []
        for file_info in files[:8]:
            if not isinstance(file_info, dict):
                continue
            previews.append(
                self._metadata_summary(
                    "file",
                    path=file_info.get("path"),
                    kind=file_info.get("file_kind"),
                    binary=file_info.get("is_binary"),
                    capabilities=self._list_preview(file_info.get("capabilities") or []),
                )
            )
        return " | ".join(
            [
                self._metadata_summary(
                    "list_files",
                    files=len(files),
                    truncated=values.get("truncated"),
                ),
                *previews,
            ]
        )

    def _summarize_read_file(self, output: BaseModel) -> str:
        values = output.model_dump(mode="json")
        return self._metadata_summary(
            "read_file",
            path=values.get("path"),
            size_bytes=values.get("size_bytes"),
            truncated=values.get("truncated"),
            is_binary=values.get("is_binary"),
            encoding=values.get("encoding"),
            content_chars=len(str(values.get("content") or "")),
        )

    def _summarize_asset_analyze(self, output: BaseModel) -> str:
        values = output.model_dump(mode="json")
        rows = values.get("rows") or []
        columns = values.get("columns") or []
        return self._metadata_summary(
            "asset_analyze",
            asset_id=values.get("asset_id"),
            operation=values.get("operation"),
            rows=len(rows),
            raw_row_count=values.get("raw_row_count"),
            truncated=values.get("truncated"),
            columns=self._list_preview(columns),
            query_chars=len(str(values.get("query") or "")),
        )

    def _summarize_unknown(self, output: BaseModel, *, tool_name: str) -> str:
        values = output.model_dump(mode="json")
        fields = list(values.keys()) if isinstance(values, dict) else []
        return self._metadata_summary(
            tool_name,
            original_output_model=_model_path(output),
            output_chars=len(output.model_dump_json()),
            fields=self._list_preview(fields),
        )

    def _eviction_summary(self, item: Any) -> str | None:
        if isinstance(item, ToolResult):
            return self.summarize_tool_result(item)
        for attr in ("summary", "text", "value_preview", "unit_type", "tool_name"):
            value = getattr(item, attr, None)
            if isinstance(value, str) and value.strip():
                return self._one_line(value)[: self._policy.max_memory_summary_chars]
        if isinstance(item, MemoryRef):
            return self._one_line(item.summary)[: self._policy.max_memory_summary_chars]
        return self._one_line(str(type(item).__name__))

    @staticmethod
    def _metadata_summary(label: str, **values: object) -> str:
        parts = [label]
        for key, value in values.items():
            if value in (None, "", []):
                continue
            parts.append(f"{key}={MemoryCompactor._one_line(str(value))}")
        return " ".join(parts)

    @staticmethod
    def _list_preview(values: object, *, limit: int = 6) -> str:
        if not isinstance(values, list):
            return ""
        shown = [MemoryCompactor._one_line(str(value)) for value in values[:limit]]
        remaining = len(values) - limit
        suffix = f", ...(+{remaining})" if remaining > 0 else ""
        return "[" + ", ".join(shown) + suffix + "]"

    def _truncate_summary(self, summary: str) -> str:
        if len(summary) <= self._policy.max_memory_summary_chars:
            return summary
        return summary[: self._policy.max_memory_summary_chars].rstrip() + " [truncated]"

    @staticmethod
    def _one_line(text: str) -> str:
        return " ".join(text.split())


@dataclass(frozen=True, slots=True)
class LoopCompactionResult:
    changed: bool
    channels: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class LoopContextCompactor:
    """Prepare bounded loop state before a model invocation."""

    _MESSAGE_CHANNELS = (
        "messages",
        "working_summary",
        "extracted_facts",
        "memory_refs",
        "memory_warnings",
    )

    def __init__(
        self,
        *,
        store: ToolOutputMemoryStore | None = None,
    ) -> None:
        self._store = store

    def prepare(self, state: LoopState) -> LoopCompactionResult:
        from rag.agent.loop.state import (
            LoopTransition,
            replace_latest_transition,
        )

        state_dict = cast(dict[str, Any], state)
        policy = state["run_config"].memory_policy
        initial_warnings = list(state["memory_warnings"])
        changed_channels: list[str] = []

        compacted_messages = MessageCompactor(
            policy=policy,
            store=self._store,
        ).compact_initial_state(dict(state_dict))
        message_update = {
            channel: compacted_messages.get(channel)
            for channel in self._MESSAGE_CHANNELS
            if compacted_messages.get(channel) != state_dict.get(channel)
        }
        if message_update:
            changed_channels.extend(message_update)
            self._apply_update(state_dict, message_update)

        memory_update = MemoryCompactor(
            policy=policy,
            store=self._store,
            loop_mode=True,
        ).compact_update(state_dict, {})
        meaningful_memory_update = {
            key: value
            for key, value in memory_update.items()
            if key != "memory_budget"
        }
        if meaningful_memory_update:
            changed_channels.extend(meaningful_memory_update)
        self._apply_update(state_dict, memory_update)

        changed = bool(changed_channels)
        warnings = tuple(
            warning
            for warning in state["memory_warnings"]
            if warning not in initial_warnings
        )
        channels = tuple(dict.fromkeys(changed_channels))
        if changed:
            replace_latest_transition(
                state,
                LoopTransition(
                    reason="compaction",
                    iteration=state["iteration"],
                    detail={
                        "channels": list(channels),
                        "warnings": list(warnings),
                    },
                ),
            )
        return LoopCompactionResult(
            changed=changed,
            channels=channels,
            warnings=warnings,
        )

    @staticmethod
    def _apply_update(
        state: dict[str, Any],
        update: dict[str, Any],
    ) -> None:
        for key, value in update.items():
            if (
                isinstance(value, list)
                and len(value) == 1
                and isinstance(value[0], StateChannelReplacement)
            ):
                state[key] = list(value[0].items)
                continue
            state[key] = value


def _bounded_recent(items: list[Any], *, limit: int) -> list[Any]:
    if len(items) <= limit:
        return list(items)
    required: list[Any] = []
    for item in items:
        if _must_preserve(item):
            required.append(item)
    selected: list[Any] = []
    for item in reversed(items):
        key = _item_key(item)
        if key in {_item_key(existing) for existing in selected}:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    for item in reversed(required):
        key = _item_key(item)
        if key in {_item_key(existing) for existing in selected}:
            continue
        if len(selected) >= limit:
            selected.pop(0)
        selected.append(item)
    return list(reversed(selected[-limit:]))


def _must_preserve(item: Any) -> bool:
    warnings = getattr(item, "warnings", None)
    if isinstance(warnings, list) and warnings:
        return True
    status = getattr(item, "status", None)
    return status == "error"


def _item_key(item: Any) -> str:
    key = getattr(item, "key", None)
    if isinstance(key, str) and key:
        return key
    for attr in ("tool_call_id", "source_tool_call_id", "unit_id", "evidence_id", "citation_id"):
        value = getattr(item, attr, None)
        if value:
            return f"{attr}:{value}"
    return repr(item)


def _source_tool_call_id(item: Any) -> str | None:
    for attr in ("tool_call_id", "source_tool_call_id", "content_ref"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _memory_ref_id(item: Any) -> str | None:
    if isinstance(item, MemoryRef):
        return item.ref_id
    output = getattr(item, "output", None)
    if isinstance(output, ExternalizedToolOutput):
        return output.ref.ref_id
    ref = getattr(item, "raw_memory_ref", None)
    if isinstance(ref, MemoryRef):
        return ref.ref_id
    error = getattr(item, "error", None)
    detail = getattr(error, "detail", None)
    if isinstance(detail, dict) and isinstance(detail.get("externalized_ref"), str):
        return str(detail["externalized_ref"])
    return None


def _active_plan_step(plan: Any) -> Any | None:
    active_step_id = getattr(plan, "active_step_id", None)
    steps = getattr(plan, "steps", []) or []
    if active_step_id is not None:
        for step in steps:
            if getattr(step, "step_id", None) == active_step_id:
                return step
    for step in steps:
        if getattr(step, "status", None) in {"in_progress", "pending"}:
            return step
    return None


def _model_path(model: BaseModel) -> str:
    return f"{model.__class__.__module__}.{model.__class__.__name__}"


__all__ = [
    "LoopCompactionResult",
    "LoopContextCompactor",
    "MemoryCompactor",
    "MessageCompactor",
    "ToolOutputMemoryStore",
    "WorkingMemoryCompactor",
]
