from __future__ import annotations

from dataclasses import replace

from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.context import derive_child_config
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.task import SubTaskNode, SubTaskResult, SubTaskStatus
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

    async def run_subtask(
        self,
        *,
        subtask: SubTaskNode,
        parent_state: AgentState,
    ) -> AgentRunResult:
        child_definition = self._agent_registry.get(subtask.agent_type)
        child_config = derive_child_config(parent_state["run_config"], child_definition)
        if subtask.estimated_tokens is not None:
            child_config = replace(child_config, budget_total=subtask.estimated_tokens)

        child_service = self._service_factory.create(child_definition)
        return await child_service.run_with_config(
            task=subtask.prompt,
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
        "Use only the supplied context sections, evidence ids, and citation ids. "
        f"Parent task: {state['task']}"
    )


def _synthesis_context_sections(state: AgentState) -> list[str]:
    sections: list[str] = []
    subtask_results = list(state.get("subtask_results", {}).values())
    if subtask_findings := _subtask_findings_section(subtask_results):
        sections.append(subtask_findings)
    if evidence_section := _evidence_section(state):
        sections.append(evidence_section)
    if tool_section := _tool_result_section(state):
        sections.append(tool_section)
    return [_clip_section(section) for section in sections[:_MAX_SYNTHESIS_SECTIONS]]


def _subtask_findings_section(subtask_results: list[SubTaskResult]) -> str:
    lines: list[str] = []
    for result in subtask_results:
        if result.status is not SubTaskStatus.COMPLETED:
            continue
        for finding in result.findings:
            if finding.strip():
                lines.append(f"- {result.subtask.subtask_id}: {finding.strip()}")
    if not lines:
        return ""
    return "Subtask findings:\n" + "\n".join(lines)


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


def _evidence_ids(state: AgentState) -> list[str]:
    ids = [evidence.evidence_id for evidence in state.get("evidence", [])]
    for result in state.get("subtask_results", {}).values():
        ids.extend(evidence.evidence_id for evidence in result.evidence)
    return _dedupe(ids)


def _citation_ids(state: AgentState) -> list[str]:
    ids = [citation.citation_id for citation in state.get("citations", [])]
    for result in state.get("subtask_results", {}).values():
        ids.extend(citation.citation_id for citation in result.citations)
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
