from __future__ import annotations

from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy, ToolPolicy

FACTCHECK_AGENT_SYSTEM_PROMPT = """You are the FactCheckAgent for verifying claims.

Use keyword_search for exact claim terms and grounding to inspect source text.
Preserve citations and evidence metadata. Use llm_generate only to state a
grounded verification result from the gathered evidence.
"""


FACTCHECK_AGENT = AgentDefinition(
    agent_type="factcheck",
    description="Verify factual claims against grounded source evidence.",
    system_prompt=FACTCHECK_AGENT_SYSTEM_PROMPT,
    allowed_tools=[
        "keyword_search",
        "grounding",
        "llm_generate",
    ],
    estimated_token_budget=6000,
    model_selection=ModelSelectionPolicy(thinking=True),
    max_iterations=8,
    max_depth=1,
    tool_policy=ToolPolicy(max_parallel_calls=3),
)
