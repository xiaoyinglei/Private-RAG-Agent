from __future__ import annotations

from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy, ToolPolicy

COMPARE_AGENT_SYSTEM_PROMPT = """You are the CompareAgent for grounded document comparison.

Collect comparable evidence before comparing. Preserve evidence ids, citations,
retrieval scores, citation anchors, and grounding metadata whenever available.
Use llm_compare only over provided evidence, and state insufficient evidence
when the available evidence cannot support a comparison.
"""


COMPARE_AGENT = AgentDefinition(
    agent_type="compare",
    description="Compare evidence across documents or entities with citations.",
    system_prompt=COMPARE_AGENT_SYSTEM_PROMPT,
    allowed_tools=[
        "vector_search",
        "grounding",
        "llm_compare",
    ],
    estimated_token_budget=10000,
    model_selection=ModelSelectionPolicy(thinking=True),
    max_iterations=10,
    max_depth=1,
    tool_policy=ToolPolicy(max_parallel_calls=3),
)
