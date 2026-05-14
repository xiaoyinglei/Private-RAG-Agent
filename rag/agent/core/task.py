from __future__ import annotations

from collections import defaultdict, deque
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.schema.query import AnswerCitation, EvidenceItem

DEFAULT_SUBTASK_TOKEN_BUDGET = 10000


class SubTaskNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    subtask_id: str
    agent_type: str
    prompt: str
    priority: int = Field(ge=0)
    estimated_tokens: int | None = Field(default=DEFAULT_SUBTASK_TOKEN_BUDGET, gt=0)


class TaskEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_id: str
    to_id: str

    @model_validator(mode="after")
    def _reject_self_edge(self) -> TaskEdge:
        if self.from_id == self.to_id:
            raise ValueError("TaskEdge cannot point to itself")
        return self


class TaskDAG(BaseModel):
    model_config = ConfigDict(frozen=True)

    subtasks: list[SubTaskNode]
    edges: list[TaskEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_dag(self) -> TaskDAG:
        ids = [subtask.subtask_id for subtask in self.subtasks]
        unique_ids = set(ids)
        if len(unique_ids) != len(ids):
            raise ValueError("duplicate subtask_id in TaskDAG")

        for edge in self.edges:
            if edge.from_id not in unique_ids or edge.to_id not in unique_ids:
                raise ValueError("TaskDAG edge references unknown subtask")

        if self._has_cycle(unique_ids):
            raise ValueError("TaskDAG contains a cycle")
        return self

    def ready_subtasks(self, *, successful: set[str], terminal: set[str]) -> list[SubTaskNode]:
        prerequisites = self._prerequisites()
        ready = [
            subtask
            for subtask in self.subtasks
            if subtask.subtask_id not in terminal
            and prerequisites[subtask.subtask_id].issubset(successful)
        ]
        return sorted(ready, key=lambda subtask: (-subtask.priority, subtask.subtask_id))

    def _prerequisites(self) -> dict[str, set[str]]:
        prerequisites: dict[str, set[str]] = {
            subtask.subtask_id: set() for subtask in self.subtasks
        }
        for edge in self.edges:
            prerequisites[edge.to_id].add(edge.from_id)
        return prerequisites

    def _has_cycle(self, subtask_ids: set[str]) -> bool:
        outgoing: dict[str, list[str]] = defaultdict(list)
        indegree = {subtask_id: 0 for subtask_id in subtask_ids}
        for edge in self.edges:
            outgoing[edge.from_id].append(edge.to_id)
            indegree[edge.to_id] += 1

        queue = deque(subtask_id for subtask_id, count in indegree.items() if count == 0)
        visited_count = 0
        while queue:
            current = queue.popleft()
            visited_count += 1
            for neighbor in outgoing[current]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        return visited_count != len(subtask_ids)


class SubTaskStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class SubTaskResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    subtask: SubTaskNode
    status: SubTaskStatus = SubTaskStatus.PENDING
    findings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    error_message: str | None = None

    @model_validator(mode="after")
    def _validate_error_message(self) -> SubTaskResult:
        if self.status == SubTaskStatus.FAILED and not self.error_message:
            raise ValueError("error_message is required when status=FAILED")
        if self.status != SubTaskStatus.FAILED and self.error_message is not None:
            raise ValueError("error_message must be None unless status=FAILED")
        return self
