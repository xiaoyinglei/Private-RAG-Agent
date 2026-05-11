"""Planner that decomposes analysis work into bounded information gaps."""

from __future__ import annotations

from collections.abc import Sequence

from rag.agent.schema import AgentTaskRequest, SubTask, TaskUnderstanding


class AgentPlanner:
    """Break complex analysis tasks into bounded information gaps."""

    def __init__(self, *, enable_llm: bool = False) -> None:
        self._enable_llm = enable_llm

    def plan(self, *, request: AgentTaskRequest, understanding: TaskUnderstanding) -> list[SubTask]:
        del self._enable_llm
        subtasks = self._fallback_plan(request=request, understanding=understanding)
        capped = subtasks[: request.max_subtasks]
        return [
            subtask.model_copy(update={"priority": index})
            for index, subtask in enumerate(capped, start=1)
        ]

    def _fallback_plan(self, *, request: AgentTaskRequest, understanding: TaskUnderstanding) -> list[SubTask]:
        if understanding.task_type == "comparison" or understanding.needs_comparison:
            return self._comparison_plan(request=request, understanding=understanding)
        if understanding.needs_timeline:
            return self._timeline_research_plan(request=request, understanding=understanding)
        return self._research_plan(request=request, understanding=understanding)

    @staticmethod
    def _make_subtask(
        *,
        subtask_id: str,
        objective: str,
        instruction: str,
        expected_evidence: Sequence[str],
        retrieval_hint: str,
        allow_web: bool,
        stop_condition: str,
        priority: int,
    ) -> SubTask:
        return SubTask(
            subtask_id=subtask_id,
            objective=objective,
            instruction=instruction,
            expected_evidence=list(expected_evidence),
            retrieval_hint=retrieval_hint,
            allow_web=allow_web,
            stop_condition=stop_condition,
            priority=priority,
        )

    def _comparison_plan(self, *, request: AgentTaskRequest, understanding: TaskUnderstanding) -> list[SubTask]:
        allow_web = request.allow_web and understanding.needs_external_evidence
        return [
            self._make_subtask(
                subtask_id="s1",
                objective="Profile Alpha system evidence.",
                instruction=f"Collect evidence describing Alpha in the context of: {request.user_query}",
                expected_evidence=["Alpha capabilities", "Alpha constraints"],
                retrieval_hint="Prefer architecture, capability, and operational sections.",
                allow_web=allow_web,
                stop_condition="At least two supporting Alpha evidence units are grounded.",
                priority=1,
            ),
            self._make_subtask(
                subtask_id="s2",
                objective="Profile Beta system evidence.",
                instruction=f"Collect evidence describing Beta in the context of: {request.user_query}",
                expected_evidence=["Beta capabilities", "Beta constraints"],
                retrieval_hint="Prefer architecture, capability, and operational sections.",
                allow_web=allow_web,
                stop_condition="At least two supporting Beta evidence units are grounded.",
                priority=2,
            ),
            self._make_subtask(
                subtask_id="s3",
                objective="Compare the tradeoffs between Alpha and Beta.",
                instruction=f"Identify evidence-backed differences relevant to: {request.user_query}",
                expected_evidence=["Shared comparison dimensions", "Material differences", "Tradeoffs"],
                retrieval_hint="Prefer sections that discuss tradeoffs, pros/cons, or architecture differences.",
                allow_web=allow_web,
                stop_condition="Tradeoffs are grounded across both systems.",
                priority=3,
            ),
            self._make_subtask(
                subtask_id="s4",
                objective="Assess recommendation criteria for the final decision.",
                instruction=f"Find evidence that supports or weakens a recommendation for: {request.user_query}",
                expected_evidence=["Decision criteria", "Risks", "Recommendation support"],
                retrieval_hint="Prefer risk, deployment, limitation, and evaluation sections.",
                allow_web=allow_web,
                stop_condition="Recommendation criteria can be supported or explicitly limited.",
                priority=4,
            ),
        ]

    def _timeline_research_plan(self, *, request: AgentTaskRequest, understanding: TaskUnderstanding) -> list[SubTask]:
        allow_web = request.allow_web and understanding.needs_external_evidence
        return [
            self._make_subtask(
                subtask_id="s1",
                objective="Establish the core factual baseline for the task.",
                instruction=f"Collect the primary evidence needed to frame: {request.user_query}",
                expected_evidence=["Primary facts", "Named entities", "Scope anchors"],
                retrieval_hint="Prefer overview and directly relevant factual sections.",
                allow_web=allow_web,
                stop_condition="Primary facts are grounded by at least two evidence units.",
                priority=1,
            ),
            self._make_subtask(
                subtask_id="s2",
                objective="Build the task timeline or chronology.",
                instruction=f"Extract time-ordered evidence relevant to: {request.user_query}",
                expected_evidence=["Dates", "Sequence of events", "Chronology anchors"],
                retrieval_hint="Prefer timeline, history, incident, and release sections.",
                allow_web=allow_web,
                stop_condition="The chronology is grounded or the gap is explicit.",
                priority=2,
            ),
            self._make_subtask(
                subtask_id="s3",
                objective="Capture risks, unknowns, and open questions.",
                instruction=f"Find evidence for unresolved issues and limitations in: {request.user_query}",
                expected_evidence=["Risks", "Unknowns", "Open questions"],
                retrieval_hint="Prefer limitation, caveat, risk, and appendix sections.",
                allow_web=allow_web,
                stop_condition="Main unresolved areas are explicit and evidence-backed.",
                priority=3,
            ),
            self._make_subtask(
                subtask_id="s4",
                objective="Synthesize implications for the final report.",
                instruction=f"Retrieve evidence that helps interpret the significance of: {request.user_query}",
                expected_evidence=["Implications", "Recommendations", "Decision constraints"],
                retrieval_hint="Prefer conclusion, deployment, evaluation, and recommendation sections.",
                allow_web=allow_web,
                stop_condition="Implications are grounded or clearly bounded by evidence gaps.",
                priority=4,
            ),
        ]

    def _research_plan(self, *, request: AgentTaskRequest, understanding: TaskUnderstanding) -> list[SubTask]:
        allow_web = request.allow_web and understanding.needs_external_evidence
        return [
            self._make_subtask(
                subtask_id="s1",
                objective="Identify the core facts and scope of the task.",
                instruction=f"Collect the foundational evidence needed to answer: {request.user_query}",
                expected_evidence=["Core facts", "Scope anchors", "Definitions"],
                retrieval_hint="Prefer overview, definition, and architecture sections.",
                allow_web=allow_web,
                stop_condition="Foundational evidence is grounded by multiple units.",
                priority=1,
            ),
            self._make_subtask(
                subtask_id="s2",
                objective="Gather supporting mechanisms or architecture details.",
                instruction=f"Retrieve supporting details and mechanisms for: {request.user_query}",
                expected_evidence=["Mechanisms", "Architecture details", "Process descriptions"],
                retrieval_hint="Prefer detailed implementation or process sections.",
                allow_web=allow_web,
                stop_condition="Supporting mechanisms are grounded or gaps are explicit.",
                priority=2,
            ),
            self._make_subtask(
                subtask_id="s3",
                objective="Capture risks, unknowns, and evidence limits.",
                instruction=f"Collect caveats and unresolved questions for: {request.user_query}",
                expected_evidence=["Risks", "Unknowns", "Evidence limits"],
                retrieval_hint="Prefer limitation, evaluation, and appendix sections.",
                allow_web=allow_web,
                stop_condition="Report can distinguish confirmed findings from unknowns.",
                priority=3,
            ),
        ]


__all__ = ["AgentPlanner"]
