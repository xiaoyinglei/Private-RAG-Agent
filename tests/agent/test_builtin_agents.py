from __future__ import annotations

from rag.agent.builtin import BUILTIN_AGENT_DEFINITIONS, create_builtin_agent_registry
from rag.agent.builtin_registry import create_builtin_tool_registry


def test_create_builtin_agent_registry_registers_expected_agents() -> None:
    registry = create_builtin_agent_registry()

    assert {definition.agent_type for definition in registry.list_all()} == {
        "generic",
    }


def test_builtin_agent_definitions_are_keyed_by_agent_type() -> None:
    assert set(BUILTIN_AGENT_DEFINITIONS) == {
        definition.agent_type for definition in BUILTIN_AGENT_DEFINITIONS.values()
    }


def test_builtin_agent_allowed_tools_exist_in_builtin_tool_registry() -> None:
    tool_registry = create_builtin_tool_registry()
    tool_names = {tool.name for tool in tool_registry.list_all()}

    # These tools are registered dynamically by AgentService,
    # not in the static tool registry.
    dynamically_registered = {
        "tool_search", "activate_tools", "task",
        # Workspace tools — registered at runtime via create_workspace_tools()
        "list_files", "read_file", "write_file", "run_python",
        "search_text", "apply_patch", "run_command", "tool_repl",
        "structured_probe",
    }

    for definition in BUILTIN_AGENT_DEFINITIONS.values():
        assert set(definition.allowed_tools) <= tool_names | dynamically_registered


def test_generic_agent_includes_llm_tools() -> None:
    assert "llm_compare" in BUILTIN_AGENT_DEFINITIONS["generic"].allowed_tools
    assert "llm_generate" in BUILTIN_AGENT_DEFINITIONS["generic"].allowed_tools
    assert "llm_summarize" in BUILTIN_AGENT_DEFINITIONS["generic"].allowed_tools


def test_generic_agent_includes_semantic_rag_tools() -> None:
    """GENERIC_AGENT uses semantic tools (search_knowledge, search_assets)."""
    allowed = BUILTIN_AGENT_DEFINITIONS["generic"].allowed_tools
    assert "search_knowledge" in allowed
    assert "search_assets" in allowed


def test_generic_agent_budget_defaults() -> None:
    assert BUILTIN_AGENT_DEFINITIONS["generic"].estimated_token_budget == 96_000
