from __future__ import annotations


def test_agent_package_exports_new_contract_surface_only() -> None:
    import rag.agent as agent

    assert hasattr(agent, "AgentDefinition")
    assert hasattr(agent, "AgentRegistry")
    assert hasattr(agent, "AgentRunConfig")
    assert hasattr(agent, "AgentState")
    assert hasattr(agent, "AgentToolSpec")
    assert hasattr(agent, "ToolRegistry")
    assert hasattr(agent, "ToolSpec")
    assert not hasattr(agent, "AnalysisAgentService")
    assert not hasattr(agent, "AgentRunState")


def test_root_package_exports_new_agent_contract_surface() -> None:
    from rag import AgentDefinition, AgentRunConfig, AgentState, ToolRegistry, ToolSpec

    assert AgentDefinition is not None
    assert AgentRunConfig is not None
    assert AgentState is not None
    assert ToolRegistry is not None
    assert ToolSpec is not None
