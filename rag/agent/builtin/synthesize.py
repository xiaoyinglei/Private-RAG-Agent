from __future__ import annotations

from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy, ToolPolicy

SYNTHESIZE_AGENT_SYSTEM_PROMPT = """You are the SynthesizeAgent for final grounded synthesis.

Do not retrieve new evidence. Use only supplied context, evidence ids, and
citations. If supplied evidence is insufficient, say so instead of filling gaps.
"""


SYNTHESIZE_AGENT = AgentDefinition(
    agent_type="synthesize",
    description="Synthesize already-collected evidence without retrieval.",
    system_prompt=SYNTHESIZE_AGENT_SYSTEM_PROMPT,
    allowed_tools=[
        "llm_generate",
    ],
    estimated_token_budget=10000,
    model_selection=ModelSelectionPolicy(thinking=True),
    max_iterations=4,
    max_depth=0,
    tool_policy=ToolPolicy(max_parallel_calls=1),
)
