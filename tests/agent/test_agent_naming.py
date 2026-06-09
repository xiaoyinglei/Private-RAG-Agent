from __future__ import annotations

from pathlib import Path


def test_core_loop_uses_preferred_names_without_legacy_aliases() -> None:
    from rag.agent import goal_runtime, planning
    from rag.agent.core import compiler as compiler_nodes
    from rag.agent.core import context as context_core
    from rag.agent.graphs.nodes import execute as tool_nodes
    from rag.agent.graphs.nodes import goal_runtime as goal_nodes
    from rag.agent.graphs.nodes import llm_decide as decide_nodes
    from rag.agent.graphs.nodes import synthesize as synth_nodes
    from rag.agent.loop import controller as loop_nodes
    from rag.agent.memory import compactor as memory_nodes
    from rag.agent.memory import injector as context_nodes
    from rag.agent.memory import models as memory_models

    for module, preferred in (
        (compiler_nodes, "GraphCompiler"),
        (context_core, "RunRegistry"),
        (goal_runtime, "GoalBuilder"),
        (goal_runtime, "ObservationExtractor"),
        (loop_nodes, "TurnController"),
        (planning, "PlanTracker"),
        (memory_models, "WorkingMemoryDraft"),
        (tool_nodes, "run_tools_raw"),
        (tool_nodes, "run_tools_guarded"),
        (goal_nodes, "init_goal"),
        (goal_nodes, "control_turn"),
        (goal_nodes, "route_after_control"),
        (goal_nodes, "extract_obs_legacy"),
        (decide_nodes, "decide_next"),
        (synth_nodes, "build_answer"),
        (memory_nodes, "MessageCompactor"),
        (memory_nodes, "MemoryCompactor"),
        (memory_nodes, "WorkingMemoryCompactor"),
        (context_nodes, "ContextBuilder"),
    ):
        assert hasattr(module, preferred)

    for module, old_name in (
        (compiler_nodes, "AgentGraphCompiler"),
        (context_core, "RuntimeRegistry"),
        (goal_runtime, "GoalInitializer"),
        (goal_runtime, "StateReducer"),
        (loop_nodes, "AgentLoopController"),
        (planning, "PlanController"),
        (memory_models, "WorkingMemoryDehydration"),
        (tool_nodes, "execute_node"),
        (tool_nodes, "execute_observe_compact_node"),
        (goal_nodes, "initialize_goal_node"),
        (goal_nodes, "controller_node"),
        (goal_nodes, "route_after_controller"),
        (goal_nodes, "reduce_observations_node"),
        (decide_nodes, "llm_decide_node"),
        (synth_nodes, "synthesize_node"),
        (memory_nodes, "RunMessageCompactor"),
        (memory_nodes, "RunMemoryCompactor"),
        (memory_nodes, "WorkingMemoryDehydrator"),
        (context_nodes, "ContextInjector"),
    ):
        assert not hasattr(module, old_name)


def test_main_graph_uses_short_node_function_names() -> None:
    from rag.agent import goal_runtime
    from rag.agent.graphs import base
    from rag.agent.graphs.nodes import execute as tool_nodes
    from rag.agent.graphs.nodes import goal_runtime as goal_nodes
    from rag.agent.graphs.nodes import llm_decide as decide_nodes
    from rag.agent.graphs.nodes import synthesize as synth_nodes

    assert goal_nodes.GoalBuilder is goal_runtime.GoalBuilder
    assert goal_nodes.ObservationExtractor is goal_runtime.ObservationExtractor
    assert base.run_tools_guarded is tool_nodes.run_tools_guarded
    assert base.init_goal is base.graph_goal_nodes.init_goal
    assert base.control_turn is base.graph_goal_nodes.control_turn
    assert base.route_after_control is base.graph_goal_nodes.route_after_control
    assert hasattr(base, "route_after_tools")
    assert not hasattr(base, "route_after_execute")
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
        "control_turn",
        "GoalBuilder",
        "GraphCompiler",
        "MemoryCompactor",
        "ObservationExtractor",
        "PlanTracker",
        "RunRegistry",
        "run_tools_guarded",
        "WorkingMemoryCompactor",
        "WorkingMemoryDraft",
    ):
        assert term in content
