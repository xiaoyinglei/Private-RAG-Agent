from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

import rag.agent as agent_api
from rag.agent.builtin.research import create_research_agent_service
from rag.agent.core.agent_as_tool import AgentAsToolRunner
from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.compiler import GraphCompiler
from rag.agent.core.finalization import FinishCandidateBuilder
from rag.agent.core.llm_providers import LoopModelDecision
from rag.agent.core.runtime_ports import RetrievalHintProvider
from rag.agent.loop.state import ModelTurnDraft
from rag.agent.service import AgentService


def test_agent_service_exposes_only_model_turn_provider() -> None:
    parameters = inspect.signature(AgentService).parameters

    assert "model_turn_provider" in parameters
    assert "tool_decision_provider" not in parameters
    assert "synthesis_runner" not in parameters


def test_agent_service_factory_has_no_legacy_provider_or_synthesis_binding() -> None:
    parameters = inspect.signature(AgentServiceFactory).parameters

    assert "model_turn_provider" in parameters
    assert "tool_decision_provider" not in parameters
    assert not hasattr(AgentServiceFactory, "bind_synthesis_runner")


@pytest.mark.parametrize(
    "entrypoint",
    [GraphCompiler, AgentAsToolRunner, create_research_agent_service],
)
def test_other_service_entrypoints_expose_only_model_turn_provider(
    entrypoint: object,
) -> None:
    parameters = inspect.signature(entrypoint).parameters

    assert "model_turn_provider" in parameters
    assert "tool_decision_provider" not in parameters
    assert "synthesis_runner" not in parameters


def test_public_api_does_not_export_legacy_decisions_or_automatic_synthesis() -> None:
    assert not hasattr(agent_api, "ThinkOutput")
    assert not hasattr(agent_api, "BuiltinSynthesisRunner")


def test_runtime_ports_only_keep_metadata_retrieval_hint_provider() -> None:
    from rag.agent.core import runtime_ports

    assert runtime_ports.RetrievalHintProvider is RetrievalHintProvider
    assert not hasattr(runtime_ports, "ToolDecisionProvider")


def test_finish_candidate_builder_has_no_synthesis_runner_dependency() -> None:
    assert set(inspect.signature(FinishCandidateBuilder).parameters) == set()


@pytest.mark.parametrize("model", [ModelTurnDraft, LoopModelDecision])
def test_model_turn_contract_rejects_legacy_synthesize_action(
    model: type[ModelTurnDraft] | type[LoopModelDecision],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate({"action": "synthesize"})
