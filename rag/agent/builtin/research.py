from __future__ import annotations

from collections.abc import Mapping

from rag.agent.core.definition import AgentDefinition, ModelPolicy, ToolPolicy
from rag.agent.graphs.nodes.evaluate import EvaluateDecisionProvider
from rag.agent.graphs.nodes.execute_subagent import SubAgentRunner
from rag.agent.graphs.nodes.plan import PlanProvider
from rag.agent.service import AgentService
from rag.agent.tools.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.registry import ToolRunner


RESEARCH_AGENT_SYSTEM_PROMPT = """You are the ResearchAgent for deep single-topic research.

Use retrieved evidence as the factual authority. Preserve evidence ids, citations,
retrieval scores, citation anchors, and grounding metadata whenever available.
Use memory only as historical or current-run context; if memory conflicts with
retrieved evidence, trust retrieved evidence.

Use vector_search and keyword_search to gather candidates, grounding to verify
source text, rerank when ordering matters, and llm_summarize only to synthesize
the provided evidence. Do not invent facts. When evidence is insufficient,
state insufficient evidence instead of filling gaps.
"""


RESEARCH_AGENT = AgentDefinition(
    agent_type="research",
    description="Deep single-topic research with grounded evidence and citations.",
    system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT,
    allowed_tools=[
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "llm_summarize",
    ],
    estimated_token_budget=8000,
    model_policy=ModelPolicy(model_alias="opus", fallback_model="sonnet", thinking=True),
    max_iterations=10,
    max_depth=2,
    tool_policy=ToolPolicy(max_parallel_calls=4),
)


def create_research_agent_service(
    *,
    runners: Mapping[str, ToolRunner] | None = None,
    query_understanding_service: object | None = None,
    evaluate_decision_provider: EvaluateDecisionProvider | None = None,
    plan_provider: PlanProvider | None = None,
    subagent_runner: SubAgentRunner | None = None,
) -> AgentService:
    return AgentService(
        definition=RESEARCH_AGENT,
        tool_registry=create_builtin_tool_registry(runners=runners),
        query_understanding_service=query_understanding_service,
        evaluate_decision_provider=evaluate_decision_provider,
        plan_provider=plan_provider,
        subagent_runner=subagent_runner,
    )
