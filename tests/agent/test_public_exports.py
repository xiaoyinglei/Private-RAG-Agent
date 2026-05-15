from __future__ import annotations


def test_agent_package_exports_new_contract_surface_only() -> None:
    import rag.agent as agent

    assert hasattr(agent, "AgentDefinition")
    assert hasattr(agent, "AgentRegistry")
    assert hasattr(agent, "AgentRunConfig")
    assert hasattr(agent, "AgentRunRequest")
    assert hasattr(agent, "AgentRunResult")
    assert hasattr(agent, "AgentService")
    assert hasattr(agent, "AgentState")
    assert hasattr(agent, "AgentAsToolRunner")
    assert hasattr(agent, "AgentToolSpec")
    assert hasattr(agent, "ToolRegistry")
    assert hasattr(agent, "ToolSpec")
    assert hasattr(agent, "derive_child_config")
    assert not hasattr(agent, "AnalysisAgentService")
    assert not hasattr(agent, "AgentRunState")


def test_root_package_exports_new_agent_contract_surface() -> None:
    from rag import AgentDefinition, AgentRunConfig, AgentRunRequest, AgentService, AgentState, ToolRegistry, ToolSpec

    assert AgentDefinition is not None
    assert AgentRunConfig is not None
    assert AgentRunRequest is not None
    assert AgentService is not None
    assert AgentState is not None
    assert ToolRegistry is not None
    assert ToolSpec is not None


def test_legacy_agent_service_module_no_longer_exports_old_service() -> None:
    import importlib

    service = importlib.import_module("rag.agent.service")
    assert not hasattr(service, "AnalysisAgentService")


def test_legacy_agent_modules_are_removed() -> None:
    import importlib.util

    legacy_modules = (
        "rag.agent.planner",
        "rag.agent.executor",
        "rag.agent.critic",
        "rag.agent.synthesizer",
        "rag.agent.understanding",
        "rag.agent.report",
        "rag.agent.schema",
    )

    for module_name in legacy_modules:
        assert importlib.util.find_spec(module_name) is None
