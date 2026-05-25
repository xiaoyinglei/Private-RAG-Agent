from __future__ import annotations

from rag.agent.builtin import BUILTIN_AGENT_DEFINITIONS, create_builtin_agent_registry
from rag.agent.builtin.compare import COMPARE_AGENT
from rag.agent.builtin.orchestrator import ORCHESTRATOR_AGENT
from rag.agent.tools.builtin_registry import create_builtin_tool_registry


def test_create_builtin_agent_registry_registers_expected_agents() -> None:
    registry = create_builtin_agent_registry()

    assert {definition.agent_type for definition in registry.list_all()} == {
        "research",
        "orchestrator",
        "compare",
        "factcheck",
        "synthesize",
    }


def test_builtin_agent_definitions_are_keyed_by_agent_type() -> None:
    assert set(BUILTIN_AGENT_DEFINITIONS) == {
        definition.agent_type for definition in BUILTIN_AGENT_DEFINITIONS.values()
    }


def test_builtin_agent_allowed_tools_exist_in_builtin_tool_registry() -> None:
    tool_registry = create_builtin_tool_registry()
    tool_names = {tool.name for tool in tool_registry.list_all()}

    for definition in BUILTIN_AGENT_DEFINITIONS.values():
        assert set(definition.allowed_tools) <= tool_names


def test_compare_agent_uses_compare_tool_contract() -> None:
    assert "llm_compare" in COMPARE_AGENT.allowed_tools
    assert "llm_generate" not in COMPARE_AGENT.allowed_tools


def test_research_agent_allows_grounded_rag_answer_tool() -> None:
    assert "rag_search_answer" in BUILTIN_AGENT_DEFINITIONS["research"].allowed_tools


def test_builtin_agent_budget_defaults_match_orchestration_budget_policy() -> None:
    assert ORCHESTRATOR_AGENT.estimated_token_budget == 20000
    for agent_type in {"research", "compare", "factcheck", "synthesize"}:
        assert BUILTIN_AGENT_DEFINITIONS[agent_type].estimated_token_budget == 10000
