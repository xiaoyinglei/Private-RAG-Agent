from __future__ import annotations

from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy, ToolPolicy

ORCHESTRATOR_AGENT_SYSTEM_PROMPT = """You are the Orchestrator agent.

Delegate bounded research and fact-check work through the available agent_* tools.
Choose those tools in the normal model-driven loop; do not create a parallel
planning workflow. Let child agents collect and preserve grounded evidence.
"""

# TODO: agent_* tool names (agent_research, agent_factcheck) must match
# ToolRegistry registration names. Keep these in sync.

ORCHESTRATOR_AGENT = AgentDefinition(
    agent_type="orchestrator",
    description="Coordinate child agents through ordinary tool calls.",
    system_prompt=ORCHESTRATOR_AGENT_SYSTEM_PROMPT,
    allowed_tools=["agent_research", "agent_factcheck"],
    # TODO: migrate estimated_token_budget / max_iterations / max_depth to runtime config
    estimated_token_budget=20000,
    model_selection=ModelSelectionPolicy(thinking=True),
    max_iterations=6,
    max_depth=2,
    tool_policy=ToolPolicy(max_parallel_calls=1),
)
