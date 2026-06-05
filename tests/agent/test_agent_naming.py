from __future__ import annotations

from pathlib import Path


def test_core_loop_short_names_are_available_with_legacy_aliases() -> None:
    from rag.agent import goal_runtime, planning
    from rag.agent.core import compiler as compiler_nodes
    from rag.agent.core import context as context_core
    from rag.agent.graphs.nodes import execute as execute_nodes
    from rag.agent.graphs.nodes import goal_runtime as goal_nodes
    from rag.agent.graphs.nodes import llm_decide as decide_nodes
    from rag.agent.graphs.nodes import synthesize as synth_nodes
    from rag.agent.loop import controller as loop_nodes
    from rag.agent.memory import compactor as memory_nodes
    from rag.agent.memory import injector as context_nodes
    from rag.agent.memory import models as memory_models

    assert compiler_nodes.GraphCompiler is compiler_nodes.AgentGraphCompiler
    assert context_core.RunRegistry is context_core.RuntimeRegistry
    assert goal_runtime.GoalBuilder is goal_runtime.GoalInitializer
    assert goal_runtime.ObservationExtractor is goal_runtime.StateReducer
    assert loop_nodes.TurnController is loop_nodes.AgentLoopController
    assert planning.PlanTracker is planning.PlanController
    assert memory_models.WorkingMemoryDraft is memory_models.WorkingMemoryDehydration
    assert execute_nodes.run_tools_raw is execute_nodes.execute_node
    assert execute_nodes.run_tools_guarded is execute_nodes.execute_observe_compact_node
    assert goal_nodes.init_goal is goal_nodes.initialize_goal_node
    assert goal_nodes.control_turn is goal_nodes.controller_node
    assert goal_nodes.route_after_control is goal_nodes.route_after_controller
    assert goal_nodes.extract_obs_legacy is goal_nodes.reduce_observations_node
    assert decide_nodes.decide_next is decide_nodes.llm_decide_node
    assert synth_nodes.build_answer is synth_nodes.synthesize_node
    assert memory_nodes.MessageCompactor is memory_nodes.RunMessageCompactor
    assert memory_nodes.MemoryCompactor is memory_nodes.RunMemoryCompactor
    assert memory_nodes.WorkingMemoryCompactor is memory_nodes.WorkingMemoryDehydrator
    assert context_nodes.ContextBuilder is context_nodes.ContextInjector


def test_main_graph_uses_short_node_function_names() -> None:
    from rag.agent import goal_runtime
    from rag.agent.graphs import base
    from rag.agent.graphs.nodes import execute as execute_nodes
    from rag.agent.graphs.nodes import goal_runtime as goal_nodes
    from rag.agent.graphs.nodes import llm_decide as decide_nodes
    from rag.agent.graphs.nodes import synthesize as synth_nodes

    assert goal_nodes.GoalBuilder is goal_runtime.GoalBuilder
    assert goal_nodes.ObservationExtractor is goal_runtime.ObservationExtractor
    assert base.run_tools_guarded is execute_nodes.run_tools_guarded
    assert base.init_goal is base.graph_goal_nodes.init_goal
    assert base.control_turn is base.graph_goal_nodes.control_turn
    assert base.route_after_control is base.graph_goal_nodes.route_after_control
    assert base.route_after_tools is base.route_after_execute
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
