from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from rag.agent.core.definition import AgentDefinition
from rag.agent.memory.models import (
    ContextBudgetSnapshot,
    ContextSection,
    ContextSectionName,
    InjectedContext,
)
from rag.utils.text import text_unit_count

if TYPE_CHECKING:
    from langgraph.graph.message import BaseMessage

    from rag.agent.state import AgentState, ToolCallPlan
    from rag.agent.tools.spec import ToolResult
    from rag.schema.query import AnswerCitation, EvidenceItem


class ContextInjector:
    """Assemble bounded LLM context in the authority order defined by the spec."""

    def __init__(
        self,
        *,
        max_context_tokens: int,
        max_section_chars: int = 4000,
    ) -> None:
        if max_context_tokens < 0:
            raise ValueError("max_context_tokens must be non-negative")
        if max_section_chars <= 0:
            raise ValueError("max_section_chars must be positive")
        self._max_context_tokens = max_context_tokens
        self._max_section_chars = max_section_chars

    def assemble(
        self,
        *,
        definition: AgentDefinition,
        state: AgentState,
        policy_hints: Sequence[str] = (),
        recalled_memories: Sequence[str] = (),
    ) -> InjectedContext:
        candidates: list[ContextSection] = []

        self._add_section(candidates, "system", definition.system_prompt, required=True)
        self._add_section(candidates, "policy_hints", self._format_policy_hints(policy_hints))
        self._add_section(candidates, "task", self._format_task(state.get("task", "")), required=True)
        self._add_section(
            candidates,
            "evidence",
            self._format_evidence(
                state.get("evidence", []),
                state.get("citations", []),
            ),
            required=True,
        )
        self._add_section(
            candidates,
            "working_memory",
            self._format_working_memory(
                state.get("working_summary"),
                state.get("extracted_facts", []),
            ),
        )
        self._add_section(
            candidates,
            "historical_hints",
            self._format_historical_hints(recalled_memories),
        )
        self._add_section(
            candidates,
            "message_tail",
            self._format_message_tail(state.get("messages", [])),
        )
        self._add_section(
            candidates,
            "tool_results",
            self._format_tool_results(state.get("tool_results", [])),
            required=True,
        )
        self._add_section(
            candidates,
            "open_decisions",
            self._format_open_decisions(
                pending_tool_calls=state.get("pending_tool_calls", []),
                needs_user_input=state.get("needs_user_input"),
                user_decision=state.get("user_decision"),
            ),
            required=True,
        )

        sections = self._select_sections(candidates)
        return InjectedContext(
            sections=sections,
            context_budget=self._budget_snapshot(sections),
        )

    def _add_section(
        self,
        sections: list[ContextSection],
        name: ContextSectionName,
        content: str,
        *,
        required: bool = False,
    ) -> None:
        normalized = content.strip()
        if not normalized:
            return
        bounded = self._truncate(normalized)
        sections.append(
            ContextSection(
                name=name,
                content=bounded,
                token_count=text_unit_count(bounded),
                required=required,
            )
        )

    def _select_sections(self, candidates: list[ContextSection]) -> list[ContextSection]:
        if self._max_context_tokens == 0:
            return candidates

        required_total = sum(section.token_count for section in candidates if section.required)
        optional_total = 0
        selected: list[ContextSection] = []
        for section in candidates:
            if section.required:
                selected.append(section)
                continue
            projected_total = required_total + optional_total + section.token_count
            if projected_total <= self._max_context_tokens:
                selected.append(section)
                optional_total += section.token_count
        return selected

    def _budget_snapshot(self, sections: list[ContextSection]) -> ContextBudgetSnapshot:
        by_name = {section.name: section.token_count for section in sections}
        return ContextBudgetSnapshot(
            max_context_tokens=self._max_context_tokens,
            system_tokens=by_name.get("system", 0) + by_name.get("policy_hints", 0),
            evidence_tokens=by_name.get("evidence", 0),
            working_memory_tokens=by_name.get("working_memory", 0),
            recalled_memory_tokens=by_name.get("historical_hints", 0),
            message_tail_tokens=by_name.get("message_tail", 0),
            tool_result_tokens=by_name.get("tool_results", 0) + by_name.get("open_decisions", 0),
        )

    @staticmethod
    def _format_policy_hints(policy_hints: Sequence[str]) -> str:
        hints = [hint.strip() for hint in policy_hints if hint.strip()]
        if not hints:
            return ""
        lines = ["Instruction and policy hints:"]
        lines.extend(f"- {hint}" for hint in hints)
        return "\n".join(lines)

    @staticmethod
    def _format_task(task: str) -> str:
        task_text = task.strip()
        if not task_text:
            return ""
        return f"Current task:\n{task_text}"

    def _format_evidence(
        self,
        evidence_items: Sequence[EvidenceItem],
        citations: Sequence[AnswerCitation],
    ) -> str:
        if not evidence_items and not citations:
            return ""

        citations_by_evidence: dict[str, list[AnswerCitation]] = {}
        for citation in citations:
            citations_by_evidence.setdefault(citation.evidence_id, []).append(citation)

        lines = [
            "Retrieved evidence is the authoritative source for factual claims.",
            "If evidence conflicts with memory, trust this evidence.",
        ]
        for evidence in evidence_items:
            metadata = self._metadata_line(
                evidence_id=evidence.evidence_id,
                doc_id=evidence.doc_id,
                score=evidence.score,
                anchor=evidence.citation_anchor,
                record_type=evidence.record_type,
                file_name=evidence.file_name,
                source_id=evidence.source_id,
                source_type=evidence.source_type,
            )
            lines.append(f"- {metadata}")
            lines.append(f"  text: {self._one_line(evidence.text)}")
            evidence_citations = citations_by_evidence.get(evidence.evidence_id, [])
            if evidence_citations:
                citation_text = ", ".join(
                    self._format_citation(citation) for citation in evidence_citations
                )
                lines.append(f"  citations: {citation_text}")

        cited_evidence_ids = {citation.evidence_id for citation in citations}
        evidence_ids = {evidence.evidence_id for evidence in evidence_items}
        orphan_citations = [
            citation for citation in citations if citation.evidence_id not in evidence_ids
        ]
        if orphan_citations:
            lines.append("Citations without matching evidence items:")
            lines.extend(f"- {self._format_citation(citation)}" for citation in orphan_citations)
        if cited_evidence_ids - evidence_ids:
            lines.append(
                "Missing evidence ids referenced by citations: "
                + ", ".join(sorted(cited_evidence_ids - evidence_ids))
            )
        return "\n".join(lines)

    def _format_working_memory(self, working_summary: Any, facts: Sequence[Any]) -> str:
        if working_summary is None and not facts:
            return ""
        lines = [
            "Working memory is current-run context, not an authority above retrieved evidence.",
        ]
        if working_summary is not None:
            covered = ", ".join(working_summary.covered_message_ids)
            lines.append(f"summary: {self._one_line(working_summary.summary)}")
            if covered:
                lines.append(f"covered_message_ids: {covered}")
        if facts:
            lines.append("extracted_facts:")
            for fact in facts:
                evidence_ids = ", ".join(fact.evidence_ids)
                source_ids = ", ".join(fact.source_message_ids)
                stale = " stale=true" if fact.stale else ""
                suffix = f"{stale} confidence={fact.confidence:.3g}"
                lines.append(f"- fact_id={fact.fact_id}{suffix}")
                lines.append(f"  text: {self._one_line(fact.text)}")
                if evidence_ids:
                    lines.append(f"  evidence_ids: {evidence_ids}")
                if source_ids:
                    lines.append(f"  source_message_ids: {source_ids}")
        return "\n".join(lines)

    @staticmethod
    def _format_historical_hints(recalled_memories: Sequence[str]) -> str:
        memories = [memory.strip() for memory in recalled_memories if memory.strip()]
        if not memories:
            return ""
        lines = [
            "These memories are historical hints, not authoritative evidence.",
            "If they conflict with retrieved evidence or current tool results, trust retrieved evidence.",
        ]
        lines.extend(f"- {memory}" for memory in memories)
        return "\n".join(lines)

    def _format_message_tail(self, messages: Sequence[BaseMessage]) -> str:
        if not messages:
            return ""
        lines = ["Recent message tail:"]
        lines.extend(self._format_message(message) for message in messages)
        return "\n".join(lines)

    def _format_tool_results(self, tool_results: Sequence[ToolResult]) -> str:
        if not tool_results:
            return ""
        lines = ["Tool results:"]
        for result in tool_results:
            prefix = (
                f"- tool_call_id={result.tool_call_id} "
                f"tool_name={result.tool_name} status={result.status} "
                f"latency_ms={result.latency_ms:.3g}"
            )
            if result.status == "ok":
                output = self._stringify_output(result.output)
                lines.append(f"{prefix} output={self._one_line(output)}")
            else:
                error = result.error
                if error is None:
                    lines.append(f"{prefix} error=<missing error payload>")
                else:
                    lines.append(
                        f"{prefix} error_code={error.code} retryable={error.retryable} "
                        f"message={self._one_line(error.message)}"
                    )
        return "\n".join(lines)

    def _format_open_decisions(
        self,
        *,
        pending_tool_calls: Sequence[ToolCallPlan],
        needs_user_input: str | None,
        user_decision: str | None,
    ) -> str:
        lines: list[str] = []
        if needs_user_input:
            lines.append(f"needs_user_input: {self._one_line(needs_user_input)}")
        if user_decision:
            lines.append(f"user_decision: {self._one_line(user_decision)}")
        if pending_tool_calls:
            lines.append("pending_tool_calls:")
            for call in pending_tool_calls:
                lines.append(
                    f"- tool_call_id={call.tool_call_id} "
                    f"tool_name={call.tool_name} "
                    f"arguments={self._one_line(str(call.arguments))}"
                )
        return "\n".join(lines)

    @staticmethod
    def _metadata_line(**values: object) -> str:
        return " ".join(
            f"{key}={value}" for key, value in values.items() if value not in (None, "", [])
        )

    @staticmethod
    def _format_citation(citation: AnswerCitation) -> str:
        values = {
            "citation_id": citation.citation_id,
            "evidence_id": citation.evidence_id,
            "anchor": citation.citation_anchor,
            "record_type": citation.record_type,
            "doc_id": citation.doc_id,
            "file_name": citation.file_name,
        }
        return ContextInjector._metadata_line(**values)

    @staticmethod
    def _format_message(message: BaseMessage) -> str:
        message_id = message.id or "<no-id>"
        role = getattr(message, "type", message.__class__.__name__)
        content = ContextInjector._one_line(ContextInjector._message_text(message))
        return f"- message_id={message_id} role={role} content={content}"

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(part) for part in content)
        return str(content)

    @staticmethod
    def _stringify_output(output: BaseModel | None) -> str:
        if output is None:
            return "<missing output payload>"
        return str(output.model_dump(mode="json"))

    @staticmethod
    def _one_line(text: str) -> str:
        return " ".join(text.split())

    def _truncate(self, content: str) -> str:
        if len(content) <= self._max_section_chars:
            return content
        truncated = content[: self._max_section_chars].rstrip()
        return f"{truncated}\n[truncated]"
