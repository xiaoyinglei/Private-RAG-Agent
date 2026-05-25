from __future__ import annotations

from dataclasses import replace

from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.context import derive_child_config
from rag.agent.core.delegation import AgentDelegationRequest
from rag.agent.core.registry import AgentRegistry
from rag.agent.service import AgentRunResult
from rag.agent.state import AgentState, ToolCallPlan

_MAX_SYNTHESIS_SECTIONS = 16
_MAX_SYNTHESIS_SECTION_CHARS = 1600


class BuiltinSubAgentRunner:
    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        service_factory: AgentServiceFactory,
    ) -> None:
        self._agent_registry = agent_registry
        self._service_factory = service_factory

    async def run_delegated_task(
        self,
        *,
        request: AgentDelegationRequest,
        parent_state: AgentState,
    ) -> AgentRunResult:
        child_definition = self._agent_registry.get(request.agent_type)
        child_config = derive_child_config(parent_state["run_config"], child_definition)
        if request.estimated_tokens is not None:
            child_config = replace(child_config, budget_total=request.estimated_tokens)

        child_service = self._service_factory.create(child_definition)
        return await child_service.run_with_config(
            task=request.prompt,
            run_config=child_config,
        )


class BuiltinSynthesisRunner:
    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        service_factory: AgentServiceFactory,
    ) -> None:
        self._agent_registry = agent_registry
        self._service_factory = service_factory

    async def run_synthesis(self, *, parent_state: AgentState) -> AgentRunResult:
        synthesis_definition = self._agent_registry.get("synthesize")
        child_config = derive_child_config(parent_state["run_config"], synthesis_definition)
        task = _synthesis_task(parent_state)
        child_service = self._service_factory.create(synthesis_definition)
        return await child_service.run_with_config(
            task=task,
            run_config=child_config,
            pending_tool_calls=[
                ToolCallPlan.create(
                    "llm_generate",
                    {
                        "prompt": task,
                        "context_sections": _synthesis_context_sections(parent_state),
                        "evidence_ids": _evidence_ids(parent_state),
                        "citation_ids": _citation_ids(parent_state),
                    },
                )
            ],
        )


def _synthesis_task(state: AgentState) -> str:
    return (
        "Synthesize a final grounded answer for the parent task. "
        "Use only the supplied support context and do not add unsupported claims. "
        f"Parent task: {state['task']}"
    )


def _synthesis_context_sections(state: AgentState) -> list[str]:
    sections: list[str] = []
    if evidence_section := _evidence_section(state):
        sections.append(evidence_section)
    if structured_section := _structured_observation_section(state):
        sections.append(structured_section)
    if tool_section := _tool_result_section(state):
        sections.append(tool_section)
    return [_clip_section(section) for section in sections[:_MAX_SYNTHESIS_SECTIONS]]


def _evidence_section(state: AgentState) -> str:
    lines = [
        f"- {evidence.evidence_id}: {evidence.text}"
        for evidence in state.get("evidence", [])
        if getattr(evidence, "text", "").strip()
    ]
    if not lines:
        return ""
    return "Evidence:\n" + "\n".join(lines)


def _tool_result_section(state: AgentState) -> str:
    if state.get("structured_observations"):
        return ""
    lines: list[str] = []
    for result in state.get("tool_results", []):
        if result.status != "ok" or result.output is None:
            continue
        text = getattr(result.output, "text", None)
        if isinstance(text, str) and text.strip():
            lines.append(f"- {result.tool_name}: {text.strip()}")
    if not lines:
        return ""
    return "Tool outputs:\n" + "\n".join(lines)


def _structured_observation_section(state: AgentState) -> str:
    lines: list[str] = []
    for candidate in state.get("answer_candidates", []):
        text = getattr(candidate, "text", "")
        if isinstance(text, str) and text.strip():
            lines.append(f"- answer_candidate: {text.strip()}")
    for result in state.get("computation_results", []):
        operation = getattr(result, "operation", None)
        preview = getattr(result, "value_preview", None)
        if operation or preview:
            lines.append(
                "- computation_result: "
                + " ".join(
                    part
                    for part in (
                        f"operation={operation}" if operation else "",
                        f"value={preview}" if preview else "",
                    )
                    if part
                )
            )
    for ref in state.get("evidence_refs", []):
        key = getattr(ref, "key", "")
        if isinstance(key, str) and key.strip():
            lines.append(f"- evidence_ref: {key.strip()}")
    context_units = state.get("context_units", [])
    if context_units:
        for unit in context_units:
            lines.append(
                "- context_unit: "
                f"unit_id={getattr(unit, 'unit_id', '<unknown>')} "
                f"unit_type={getattr(unit, 'unit_type', '<unknown>')} "
                f"locator={getattr(unit, 'locator', {})} "
                f"preview={getattr(unit, 'preview', None)}"
            )
    else:
        for locator in state.get("locators", []):
            lines.append(f"- locator: {locator}")
    if not lines:
        return ""
    return "Structured observations:\n" + "\n".join(lines)


def _evidence_ids(state: AgentState) -> list[str]:
    ids = [evidence.evidence_id for evidence in state.get("evidence", [])]
    for ref in state.get("evidence_refs", []):
        evidence_id = getattr(ref, "evidence_id", None)
        if isinstance(evidence_id, str) and evidence_id.strip():
            ids.append(evidence_id)
    return _dedupe(ids)


def _citation_ids(state: AgentState) -> list[str]:
    ids = [citation.citation_id for citation in state.get("citations", [])]
    for ref in state.get("evidence_refs", []):
        citation_id = getattr(ref, "citation_id", None)
        if isinstance(citation_id, str) and citation_id.strip():
            ids.append(citation_id)
    return _dedupe(ids)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _clip_section(section: str) -> str:
    if len(section) <= _MAX_SYNTHESIS_SECTION_CHARS:
        return section
    return section[: _MAX_SYNTHESIS_SECTION_CHARS - 3].rstrip() + "..."
