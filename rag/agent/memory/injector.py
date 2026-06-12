from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from rag.agent.core.definition import AgentDefinition
from rag.agent.memory.models import (
    ContextBudgetSnapshot,
    ContextSection,
    ContextSectionName,
    ExternalizedToolOutput,
    InjectedContext,
    MemoryRef,
)
from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from rag.agent.loop.state import LoopState
    from rag.agent.state import AgentState, ToolCallPlan
    from rag.agent.tools.spec import ToolResult
    from rag.schema.query import AnswerCitation, EvidenceItem

    type ContextState = AgentState | LoopState


@dataclass
class _ContextSelection:
    sections: list[ContextSection]
    dropped_sections: list[ContextSectionName] = field(default_factory=list)
    summarized_sections: list[ContextSectionName] = field(default_factory=list)
    required_truncated: list[ContextSectionName] = field(default_factory=list)
    dropped_section_reasons: dict[str, str] = field(default_factory=dict)
    overflow: bool = False
    degraded: bool = False
    warnings: list[str] = field(default_factory=list)


class ContextTokenAccounting(Protocol):
    def count(self, text: str) -> int: ...

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str: ...


class ContextBuilder:
    """Assemble bounded LLM context in the authority order defined by the spec."""

    _MAX_LOCATORS_PER_OBSERVATION = 20
    _MAX_COLUMNS_PER_LOCATOR = 40
    _MAX_HEAD_ROWS_PER_LOCATOR = 8
    _MAX_ROW_CELLS = 12
    _SECTION_PRIORITY: dict[ContextSectionName, int] = {
        "instructions": 0,
        "system": 0,
        "policy_hints": 0,
        "task": 1,
        "open_decisions": 2,
        "plan": 3,
        "call_context": 4,
        "tool_results": 4,
        "evidence": 5,
        "memory": 6,
        "working_memory": 7,
        "historical_hints": 8,
        "message_tail": 9,
    }

    def __init__(
        self,
        *,
        max_context_tokens: int,
        max_section_chars: int = 4000,
        token_accounting: ContextTokenAccounting | None = None,
    ) -> None:
        if max_context_tokens < 0:
            raise ValueError("max_context_tokens must be non-negative")
        if max_section_chars <= 0:
            raise ValueError("max_section_chars must be positive")
        self._max_context_tokens = max_context_tokens
        self._max_section_chars = max_section_chars
        self._token_accounting = token_accounting or TokenAccountingService(
            TokenizerContract(
                embedding_model_name="agent-context",
                tokenizer_model_name="agent-context",
                chunking_tokenizer_model_name="agent-context",
                tokenizer_backend="simple",
                max_context_tokens=max(max_context_tokens, 1),
                prompt_reserved_tokens=0,
                local_files_only=True,
            )
        )

    def assemble(
        self,
        *,
        definition: AgentDefinition,
        state: AgentState,
        policy_hints: Sequence[str] = (),
        recalled_memories: Sequence[str] = (),
        included_sections: frozenset[ContextSectionName] | None = None,
        required_sections: frozenset[ContextSectionName] | None = None,
    ) -> InjectedContext:
        candidates: list[ContextSection] = []

        def add(
            name: ContextSectionName,
            content: str,
            *,
            required: bool = False,
        ) -> None:
            if included_sections is not None and name not in included_sections:
                return
            self._add_section(
                candidates,
                name,
                content,
                required=(
                    required
                    if required_sections is None
                    else name in required_sections
                ),
            )

        add("system", definition.system_prompt, required=True)
        add("policy_hints", self._format_policy_hints(policy_hints))
        add("task", self._format_task(state.get("task", "")), required=True)
        add(
            "open_decisions",
            self._format_open_decisions(
                pending_tool_calls=state.get("pending_tool_calls", []),
                needs_user_input=state.get("needs_user_input"),
                user_decision=state.get("user_decision"),
                goal_spec=state.get("goal_spec"),
                open_gaps=state.get("open_gaps", []),
                satisfied_requirements=state.get("satisfied_requirements", []),
                conflicts=state.get("conflicts", []),
            ),
            required=True,
        )
        add(
            "plan",
            self._format_plan(state.get("agent_plan")),
            required=True,
        )
        add(
            "evidence",
            self._format_evidence(
                state.get("evidence", []),
                state.get("citations", []),
            ),
            required=True,
        )
        add(
            "memory",
            self._format_memory_refs(
                state.get("memory_refs", []),
                state.get("memory_warnings", []),
            ),
        )
        add(
            "working_memory",
            self._format_working_memory(
                state.get("working_summary"),
                state.get("extracted_facts", []),
            ),
        )
        add(
            "historical_hints",
            self._format_historical_hints(recalled_memories),
        )
        add(
            "message_tail",
            self._format_message_tail(state.get("messages", [])),
        )
        add(
            "tool_results",
            self._format_tool_observations(state),
            required=True,
        )

        selection = self._select_sections(candidates)
        return InjectedContext(
            sections=selection.sections,
            context_budget=self._budget_snapshot(
                selection,
                state=state,
            ),
        )

    def assemble_loop(
        self,
        *,
        definition: AgentDefinition,
        state: LoopState,
        policy_hints: Sequence[str] = (),
        recalled_memories: Sequence[str] = (),
        included_sections: frozenset[ContextSectionName] | None = None,
        required_sections: frozenset[ContextSectionName] | None = None,
    ) -> InjectedContext:
        """Assemble loop context without importing legacy goal-controller state."""

        candidates: list[ContextSection] = []

        def add(
            name: ContextSectionName,
            content: str,
            *,
            required: bool = False,
        ) -> None:
            if included_sections is not None and name not in included_sections:
                return
            self._add_section(
                candidates,
                name,
                content,
                required=(
                    required
                    if required_sections is None
                    else name in required_sections
                ),
            )

        add("system", definition.system_prompt, required=True)
        add("policy_hints", self._format_policy_hints(policy_hints))
        add("task", self._format_task(state.get("task", "")), required=True)
        add(
            "open_decisions",
            self._format_loop_open_decisions(state),
            required=True,
        )
        add(
            "plan",
            self._format_plan(
                state.get("agent_plan"),
                include_goal_refs=False,
            ),
            required=True,
        )
        add(
            "evidence",
            self._format_evidence(
                state.get("evidence", []),
                state.get("citations", []),
            ),
            required=True,
        )
        add(
            "memory",
            self._format_memory_refs(
                state.get("memory_refs", []),
                state.get("memory_warnings", []),
            ),
        )
        add(
            "working_memory",
            self._format_working_memory(
                state.get("working_summary"),
                state.get("extracted_facts", []),
            ),
        )
        add(
            "historical_hints",
            self._format_historical_hints(recalled_memories),
        )
        add(
            "message_tail",
            self._format_message_tail(state.get("messages", [])),
        )
        add(
            "tool_results",
            self._format_tool_observations(state),
            required=True,
        )

        selection = self._select_sections(candidates)
        return InjectedContext(
            sections=selection.sections,
            context_budget=self._budget_snapshot(
                selection,
                state=state,
            ),
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
        bounded = normalized if required else self._truncate(normalized)
        sections.append(
            ContextSection(
                name=name,
                content=bounded,
                token_count=self._section_token_count(name, bounded),
                required=required,
            )
        )

    def _select_sections(
        self,
        candidates: list[ContextSection],
    ) -> _ContextSelection:
        if self._max_context_tokens == 0:
            return _ContextSelection(sections=candidates)

        selected_by_name: dict[ContextSectionName, ContextSection] = {}
        dropped: list[ContextSectionName] = []
        summarized: list[ContextSectionName] = []
        required_truncated: list[ContextSectionName] = []
        dropped_reasons: dict[str, str] = {}
        warnings: list[str] = []
        overflow = False
        degraded = False
        indexed = list(enumerate(candidates))
        ordered = sorted(
            indexed,
            key=lambda item: (
                0 if item[1].required else 1,
                self._SECTION_PRIORITY[item[1].name],
                item[0],
            ),
        )
        for _, section in ordered:
            candidate_selection = {
                **selected_by_name,
                section.name: section,
            }
            if self._selected_token_count(candidates, candidate_selection) <= (
                self._max_context_tokens
            ):
                selected_by_name[section.name] = section
                continue

            if section.required:
                overflow = True
                degraded = True
                warnings.append("context_overflow")
                required_truncated.append(section.name)
                dropped.append(section.name)
                dropped_reasons[section.name] = "required_section_overflow"
                continue

            clipped = self._clip_optional_section(
                candidates=candidates,
                selected_by_name=selected_by_name,
                section=section,
            )
            if clipped is not None:
                selected_by_name[section.name] = clipped
                summarized.append(section.name)
                degraded = True
                continue
            dropped.append(section.name)
            dropped_reasons[section.name] = "budget_priority"
            degraded = True
        return _ContextSelection(
            sections=[
                selected_by_name[section.name]
                for section in candidates
                if section.name in selected_by_name
            ],
            dropped_sections=dropped,
            summarized_sections=list(dict.fromkeys(summarized)),
            required_truncated=list(dict.fromkeys(required_truncated)),
            dropped_section_reasons=dropped_reasons,
            overflow=overflow,
            degraded=degraded,
            warnings=list(dict.fromkeys(warnings)),
        )

    def _budget_snapshot(
        self,
        selection: _ContextSelection,
        *,
        state: ContextState,
    ) -> ContextBudgetSnapshot:
        sections = selection.sections
        by_name = {section.name: section.token_count for section in sections}
        summarized = [
            section.name for section in sections if section.content.endswith("[truncated]")
        ]
        summarized = list(dict.fromkeys([*summarized, *selection.summarized_sections]))
        return ContextBudgetSnapshot(
            max_context_tokens=self._max_context_tokens,
            used_context_tokens=self._token_accounting.count(
                self._render_sections(sections)
            ),
            system_tokens=by_name.get("system", 0) + by_name.get("policy_hints", 0),
            planning_tokens=by_name.get("plan", 0),
            evidence_tokens=by_name.get("evidence", 0),
            memory_tokens=by_name.get("memory", 0),
            working_memory_tokens=by_name.get("working_memory", 0),
            recalled_memory_tokens=by_name.get("historical_hints", 0),
            message_tail_tokens=by_name.get("message_tail", 0),
            tool_result_tokens=by_name.get("tool_results", 0) + by_name.get("open_decisions", 0),
            dropped_sections=selection.dropped_sections,
            summarized_sections=summarized,
            overflow=selection.overflow,
            degraded=selection.degraded,
            required_truncated=selection.required_truncated,
            section_token_counts={str(key): value for key, value in by_name.items()},
            dropped_section_reasons=selection.dropped_section_reasons,
            memory_ref_count=len(state.get("memory_refs", [])),
            externalized_record_count=self._externalized_tool_output_count(
                state.get("tool_results", [])
            ),
            warnings=list(dict.fromkeys([*state.get("memory_warnings", []), *selection.warnings])),
        )

    def _clip_optional_section(
        self,
        *,
        candidates: list[ContextSection],
        selected_by_name: dict[ContextSectionName, ContextSection],
        section: ContextSection,
    ) -> ContextSection | None:
        if not section.content:
            return None
        low = 1
        high = self._token_accounting.count(section.content)
        best: ContextSection | None = None
        while low <= high:
            midpoint = (low + high) // 2
            content = self._token_accounting.clip(
                section.content,
                midpoint,
                add_ellipsis=True,
            ).strip()
            if not content:
                high = midpoint - 1
                continue
            clipped = ContextSection(
                name=section.name,
                content=content,
                token_count=self._section_token_count(section.name, content),
                required=False,
            )
            candidate_selection = {
                **selected_by_name,
                section.name: clipped,
            }
            if self._selected_token_count(candidates, candidate_selection) <= (
                self._max_context_tokens
            ):
                best = clipped
                low = midpoint + 1
            else:
                high = midpoint - 1
        return best

    def _selected_token_count(
        self,
        candidates: Sequence[ContextSection],
        selected_by_name: dict[ContextSectionName, ContextSection],
    ) -> int:
        selected = [
            selected_by_name[section.name]
            for section in candidates
            if section.name in selected_by_name
        ]
        return self._token_accounting.count(self._render_sections(selected))

    def _section_token_count(
        self,
        name: ContextSectionName,
        content: str,
    ) -> int:
        return self._token_accounting.count(f"[{name}]\n{content}")

    @staticmethod
    def _render_sections(sections: Sequence[ContextSection]) -> str:
        return "\n\n".join(
            f"[{section.name}]\n{section.content}" for section in sections
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

    def _format_plan(
        self,
        plan: Any,
        *,
        include_goal_refs: bool = True,
    ) -> str:
        if plan is None:
            return ""
        steps = getattr(plan, "steps", []) or []
        lines = ["Current autonomous plan:"]
        objective = getattr(plan, "objective", None)
        if isinstance(objective, str) and objective.strip():
            lines.append(f"objective: {self._one_line(objective)}")
        status = getattr(plan, "status", None)
        revision = getattr(plan, "revision", None)
        active_step_id = getattr(plan, "active_step_id", None)
        plan_bits: list[str] = []
        if status:
            plan_bits.append(f"status={status}")
        if revision is not None:
            plan_bits.append(f"revision={revision}")
        if active_step_id:
            plan_bits.append(f"active_step_id={self._format_identifier(active_step_id)}")
        if plan_bits:
            lines.append(" ".join(plan_bits))
        summary = getattr(plan, "summary", None)
        if isinstance(summary, str) and summary.strip():
            lines.append(f"summary: {self._one_line(summary)}")
        if steps:
            lines.append("steps:")
            for step in steps[:12]:
                title = self._one_line(str(getattr(step, "title", "")))
                step_line = (
                    f"- step_id={self._format_identifier(getattr(step, 'step_id', '<unknown>'))} "
                    f"status={getattr(step, 'status', '<unknown>')} "
                    f"title={title}"
                )
                lines.append(step_line)
                related_gap_ids = getattr(step, "related_gap_ids", None)
                if (
                    include_goal_refs
                    and isinstance(related_gap_ids, list)
                    and related_gap_ids
                ):
                    lines.append(
                        "  related_gap_ids: "
                        + self._format_list([str(item) for item in related_gap_ids])
                    )
                expected_tools = getattr(step, "expected_tool_names", None)
                if isinstance(expected_tools, list) and expected_tools:
                    lines.append(
                        "  expected_tool_names: "
                        + self._format_list([str(item) for item in expected_tools])
                    )
                tool_call_ids = getattr(step, "tool_call_ids", None)
                if isinstance(tool_call_ids, list) and tool_call_ids:
                    lines.append(
                        "  tool_call_ids: "
                        + self._format_list([str(item) for item in tool_call_ids])
                    )
                notes = getattr(step, "notes", None)
                if isinstance(notes, str) and notes.strip():
                    lines.append(f"  notes: {self._one_line(notes)}")
        return "\n".join(lines)

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

    def _format_memory_refs(
        self,
        memory_refs: Sequence[Any],
        memory_warnings: Sequence[str],
    ) -> str:
        refs = [ref for ref in memory_refs if isinstance(ref, MemoryRef)]
        warnings = [warning for warning in memory_warnings if warning]
        if not refs and not warnings:
            return ""
        lines = [
            "Run-local externalized memory refs. Use summaries for reasoning; "
            "raw payloads require internal resolution.",
        ]
        if refs:
            for ref in refs[:20]:
                lines.append(
                    "- "
                    + self._metadata_line(
                        ref_id=ref.ref_id,
                        status=ref.status,
                        source_tool=ref.source_tool_name,
                        tool_call_id=ref.source_tool_call_id,
                        size_bytes=ref.size_bytes,
                    )
                )
                lines.append(f"  summary: {self._one_line(ref.summary)}")
            remaining = len(refs) - 20
            if remaining > 0:
                lines.append(f"- ... {remaining} more memory refs")
        if warnings:
            lines.append("memory_warnings: " + ", ".join(dict.fromkeys(warnings)))
        return "\n".join(lines)

    def _format_message_tail(self, messages: Sequence[BaseMessage]) -> str:
        if not messages:
            return ""
        lines = ["Recent message tail:"]
        lines.extend(self._format_message(message) for message in messages)
        return "\n".join(lines)

    def _format_tool_observations(self, state: ContextState) -> str:
        structured = state.get("structured_observations", [])
        if structured:
            return self._format_structured_observations(structured)
        return self._format_tool_results(state.get("tool_results", []))

    def _format_structured_observations(self, observations: Sequence[Any]) -> str:
        lines = ["Structured tool observations:"]
        for observation in observations:
            prefix = (
                f"- tool_call_id={getattr(observation, 'tool_call_id', '<unknown>')} "
                f"tool_name={getattr(observation, 'tool_name', '<unknown>')} "
                f"status={getattr(observation, 'status', '<unknown>')}"
            )
            lines.append(prefix)
            answer = getattr(observation, "answer_candidate", None)
            if answer is not None:
                text = getattr(answer, "text", "")
                if isinstance(text, str) and text.strip():
                    lines.append(f"  answer_candidate: {self._one_line(text)}")
            evidence_refs = getattr(observation, "evidence_refs", []) or []
            if evidence_refs:
                refs = ", ".join(
                    self._one_line(str(getattr(ref, "key", ref)))
                    for ref in evidence_refs
                )
                lines.append(f"  evidence_refs: {refs}")
            context_units = getattr(observation, "context_units", []) or []
            if context_units:
                lines.append("  context_units:")
                for unit in context_units[: self._MAX_LOCATORS_PER_OBSERVATION]:
                    lines.append(
                        "    - "
                        f"unit_id={self._format_identifier(getattr(unit, 'unit_id', '<unknown>'))} "
                        f"unit_type={self._one_line(str(getattr(unit, 'unit_type', '<unknown>')))} "
                        f"{self._format_locator(getattr(unit, 'locator', {}))}"
                    )
                    capabilities = getattr(unit, "capabilities", None)
                    if isinstance(capabilities, list) and capabilities:
                        lines.append(
                            "      capabilities: "
                            + self._format_list([str(item) for item in capabilities])
                        )
                    preview = getattr(unit, "preview", None)
                    if preview:
                        lines.append(f"      preview: {self._one_line(str(preview))}")
            locators = [] if context_units else (getattr(observation, "locators", []) or [])
            if locators:
                lines.append("  locators:")
                for locator in locators[: self._MAX_LOCATORS_PER_OBSERVATION]:
                    lines.append(f"    - {self._format_locator(locator)}")
                remaining = len(locators) - self._MAX_LOCATORS_PER_OBSERVATION
                if remaining > 0:
                    lines.append(f"    - ... {remaining} more locators")
            if error := getattr(observation, "error", None):
                lines.append(f"  error: {self._one_line(str(error))}")
        return "\n".join(lines)

    def _format_locator(self, locator: Any) -> str:
        if not isinstance(locator, dict):
            return self._one_line(str(locator))

        fields = (
            "asset_id",
            "doc_id",
            "source_id",
            "section_id",
            "asset_type",
            "table_index",
            "table_name",
            "used_range",
            "sheet_name",
            "page_no",
            "element_ref",
            "citation_anchor",
            "evidence_id",
            "path",
            "name",
            "size_bytes",
            "is_dir",
            "mime_type",
            "file_kind",
            "truncated",
            "is_binary",
            "readable_as_text",
            "encoding",
            "source_tool",
            "generated",
            "generated_by",
            "ok",
            "exit_code",
            "duration_ms",
            "stdout_truncated",
            "stderr_truncated",
            "header_row_index",
            "header_confidence",
            "data_start_row",
            "row_count",
            "column_count",
        )
        parts = [
            f"{field}={self._format_locator_value(field, locator[field])}"
            for field in fields
            if locator.get(field) not in (None, "", [])
        ]

        capabilities = locator.get("analysis_capabilities")
        if isinstance(capabilities, list) and capabilities:
            parts.append("analysis_capabilities=" + self._format_list(capabilities))

        columns = locator.get("columns") or locator.get("column_names")
        if isinstance(columns, list) and columns:
            parts.append(
                "columns="
                + self._format_list(
                    columns,
                    limit=self._MAX_COLUMNS_PER_LOCATOR,
                )
            )

        head_rows = locator.get("head_rows")
        if isinstance(head_rows, list) and head_rows:
            rows = [
                self._format_row_preview(row)
                for row in head_rows[: self._MAX_HEAD_ROWS_PER_LOCATOR]
            ]
            parts.append("head_rows=" + self._format_list(rows, limit=len(rows)))

        return " ".join(parts) if parts else self._one_line(str(locator))

    def _format_row_preview(self, row: Any) -> str:
        if not isinstance(row, dict):
            return self._one_line(str(row))
        cells: list[str] = []
        for index, (key, value) in enumerate(row.items()):
            if index >= self._MAX_ROW_CELLS:
                cells.append("...")
                break
            cells.append(f"{key}={value}")
        return "{" + self._one_line(", ".join(cells)) + "}"

    def _format_list(self, values: Sequence[Any], *, limit: int | None = None) -> str:
        effective_limit = limit if limit is not None else len(values)
        shown = [self._one_line(str(value)) for value in values[:effective_limit]]
        remaining = len(values) - effective_limit
        suffix = f", ...(+{remaining})" if remaining > 0 else ""
        return "[" + ", ".join(shown) + suffix + "]"

    @staticmethod
    def _format_identifier(value: object) -> str:
        return ContextBuilder._preserve_spaces_one_line(str(value))

    @staticmethod
    def _format_locator_value(field: str, value: object) -> str:
        if field in {"path", "name", "sheet_name", "element_ref", "generated_by"}:
            return ContextBuilder._preserve_spaces_one_line(str(value))
        return ContextBuilder._one_line(str(value))

    @staticmethod
    def _preserve_spaces_one_line(text: str) -> str:
        return text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()

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
                output = result.output
                if isinstance(output, ExternalizedToolOutput):
                    lines.append(
                        f"{prefix} externalized_ref={output.ref.ref_id} "
                        f"status={output.status} summary={self._one_line(output.summary)}"
                    )
                else:
                    output_text = self._stringify_output(output)
                    lines.append(f"{prefix} output={self._one_line(output_text)}")
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
        goal_spec: Any | None = None,
        open_gaps: Sequence[Any] = (),
        satisfied_requirements: Sequence[str] = (),
        conflicts: Sequence[Any] = (),
    ) -> str:
        lines: list[str] = []
        if goal_spec is not None:
            original_query = getattr(goal_spec, "original_query", None)
            if isinstance(original_query, str) and original_query.strip():
                lines.append(f"goal: {self._one_line(original_query)}")
        if open_gaps:
            lines.append(
                "open_gaps: "
                + ", ".join(
                    str(getattr(gap, "gap_id", gap))
                    for gap in open_gaps
                )
            )
        if satisfied_requirements:
            lines.append("satisfied_requirements: " + ", ".join(satisfied_requirements))
        if conflicts:
            lines.append(
                "conflicts: "
                + ", ".join(
                    str(getattr(conflict, "description", conflict))
                    for conflict in conflicts
                )
            )
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

    def _format_loop_open_decisions(self, state: LoopState) -> str:
        lines: list[str] = []
        request = state.get("approval_request")
        if request is not None:
            lines.append(
                "approval_request: "
                f"kind={request.kind} request_id={request.request_id} "
                f"question={self._one_line(request.question)}"
            )
            for approval_call in request.tool_calls:
                lines.append(
                    f"- tool_call_id={approval_call.tool_call_id} "
                    f"tool_name={approval_call.tool_name} "
                    f"risk_level={approval_call.risk_level}"
                )
        response = state.get("approval_response")
        if response is not None:
            lines.append(
                "approval_response: "
                f"request_id={response.request_id} decision={response.decision}"
            )
            if response.user_message:
                lines.append(
                    f"user_message: {self._one_line(response.user_message)}"
                )
        pending_tool_calls = state.get("pending_tool_calls", [])
        if pending_tool_calls:
            lines.append("pending_tool_calls:")
            for pending_call in pending_tool_calls:
                lines.append(
                    f"- tool_call_id={pending_call.tool_call_id} "
                    f"tool_name={pending_call.tool_name} "
                    f"arguments={self._one_line(str(pending_call.arguments))}"
                )
        feedback = state.get("stop_hook_feedback", [])
        if feedback:
            lines.append("finish_feedback:")
            for item in feedback:
                lines.append(
                    f"- code={item.code} occurrences={item.occurrences} "
                    f"message={self._one_line(item.message)}"
                )
        warnings = state.get("stop_hook_warnings", [])
        if warnings:
            lines.append("finish_warnings:")
            for item in warnings:
                lines.append(
                    f"- code={item.code} occurrences={item.occurrences} "
                    f"message={self._one_line(item.message)}"
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
        return ContextBuilder._metadata_line(**values)

    @staticmethod
    def _format_message(message: BaseMessage) -> str:
        message_id = message.id or "<no-id>"
        role = getattr(message, "type", message.__class__.__name__)
        content = ContextBuilder._one_line(ContextBuilder._message_text(message))
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
    def _externalized_tool_output_count(tool_results: Sequence[ToolResult]) -> int:
        return sum(
            1
            for result in tool_results
            if isinstance(getattr(result, "output", None), ExternalizedToolOutput)
        )

    @staticmethod
    def _one_line(text: str) -> str:
        return " ".join(text.split())

    def _truncate(self, content: str) -> str:
        if len(content) <= self._max_section_chars:
            return content
        truncated = content[: self._max_section_chars].rstrip()
        return f"{truncated}\n[truncated]"

__all__ = ["ContextBuilder"]
