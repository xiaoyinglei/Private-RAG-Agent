from __future__ import annotations

from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy, ToolPolicy

ORCHESTRATOR_AGENT_SYSTEM_PROMPT = """You are the Orchestrator agent.

Break complex tasks into bounded research, comparison, and fact-check subtasks.
Do not perform retrieval yourself in Phase 8A; use TaskDAG planning so child
agents collect and preserve grounded evidence.
"""


ORCHESTRATOR_AGENT = AgentDefinition(
    agent_type="orchestrator",
    description="Plan and coordinate child agents through a TaskDAG.",
    system_prompt=ORCHESTRATOR_AGENT_SYSTEM_PROMPT,
    allowed_tools=[],
    estimated_token_budget=20000,
    model_selection=ModelSelectionPolicy(thinking=True),
    max_iterations=6,
    max_depth=2,
    tool_policy=ToolPolicy(max_parallel_calls=1),
)
