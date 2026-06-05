from __future__ import annotations

from pathlib import Path


def test_core_loop_short_names_are_available_with_legacy_aliases() -> None:
    from rag.agent.graphs.nodes import execute as execute_nodes
    from rag.agent.graphs.nodes import goal_runtime as goal_nodes
    from rag.agent.graphs.nodes import llm_decide as decide_nodes
    from rag.agent.graphs.nodes import synthesize as synth_nodes
    from rag.agent.memory import injector as context_nodes

    assert execute_nodes.run_tools_raw is execute_nodes.execute_node
    assert execute_nodes.run_tools_guarded is execute_nodes.execute_observe_compact_node
    assert goal_nodes.extract_obs_legacy is goal_nodes.reduce_observations_node
    assert decide_nodes.decide_next is decide_nodes.llm_decide_node
    assert synth_nodes.build_answer is synth_nodes.synthesize_node
    assert context_nodes.ContextBuilder is context_nodes.ContextInjector


def test_main_graph_uses_short_node_function_names() -> None:
    from rag.agent.graphs import base
    from rag.agent.graphs.nodes import execute as execute_nodes
    from rag.agent.graphs.nodes import llm_decide as decide_nodes
    from rag.agent.graphs.nodes import synthesize as synth_nodes

    assert base.run_tools_guarded is execute_nodes.run_tools_guarded
    assert base.decide_next is decide_nodes.decide_next
    assert base.build_answer is synth_nodes.build_answer


def test_agent_naming_guide_defines_core_terms() -> None:
    guide = Path("docs/agent_naming.md")

    content = guide.read_text()

    for term in (
        "raw_",
        "safe_",
        "Ref",
        "Payload",
        "Policy",
        "Guard",
        "Snapshot",
        "run_tools_guarded",
    ):
        assert term in content
