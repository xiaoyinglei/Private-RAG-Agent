from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy
from rag.agent.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from rag.agent.core.llm_providers import (
    LLMEvaluateDecisionProvider,
    LLMPlanProvider,
    LLMRouteProvider,
    _generate_structured,
    create_default_providers,
)
from rag.agent.core.llm_registry import ModelRegistry, ResolvedModel
from rag.agent.core.task import TaskDAG
from rag.agent.memory.models import ContextBudgetSnapshot, ContextSection, InjectedContext
from rag.agent.state import ThinkOutput
from rag.schema.runtime import AccessPolicy

# ── Stub Generator ──


class _StubGenerator:
    """返回预置 JSON 的 Generator，不调用真实 LLM。"""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self._index = 0
        self.calls: list[tuple[str, type[BaseModel], dict[str, Any]]] = []

    def generate_structured(
        self, *, prompt: str, schema: type[BaseModel], **kwargs: Any
    ) -> BaseModel:
        self.calls.append((prompt, schema, kwargs))
        if self._index >= len(self._responses):
            raise RuntimeError("no more stub responses")
        raw = self._responses[self._index]
        self._index += 1
        return schema.model_validate(raw)

    def generate_text(self, *, prompt: str, **kwargs: Any) -> str:
        return "stub text"


class _FailingGenerator:
    """总是解析失败（返回不可解析的字符串）。"""

    def generate_structured(
        self, *, prompt: str, schema: type[BaseModel], **kwargs: Any
    ) -> BaseModel:
        raise ValueError("always fails")

    def generate_text(self, *, prompt: str, **kwargs: Any) -> str:
        return "garbage {{ not json"


class _FakeModelRegistry:
    def __init__(self, generator: object) -> None:
        self._generator = generator

    def resolve_for_node(
        self,
        *,
        node_model: str | None,
        node_name: str,
    ) -> ResolvedModel:
        del node_model, node_name
        return ResolvedModel(generator=self._generator, kwargs={})


def _make_config() -> AgentRunConfig:
    return AgentRunConfig(
        run_id="test",
        thread_id="test",
        budget_total=10000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )


def _make_state(**overrides: Any) -> dict:
    state: dict[str, Any] = {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": "test task",
        "run_config": _make_config(),
        "plan": None,
        "iteration": 0,
        "status": "running",
        "route_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "next_subtasks": None,
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "subtask_results": {},
        "terminal_subtasks": set(),
        "successful_subtasks": set(),
        "final_answer": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
    }
    state.update(overrides)
    return state


def _make_context() -> InjectedContext:
    return InjectedContext(
        sections=[
            ContextSection(name="task", content="test task", token_count=10, required=True),
        ],
        context_budget=ContextBudgetSnapshot(max_context_tokens=4096),
    )


# ── _generate_structured ──


class _TestSchema(BaseModel):
    value: str


class TestGenerateStructured:
    def test_parses_valid_json(self) -> None:
        gen = _StubGenerator([{"value": "hello"}])
        result = _generate_structured(gen, "prompt", _TestSchema)
        assert result is not None
        assert result.value == "hello"

    def test_returns_none_on_failure(self) -> None:
        gen = _FailingGenerator()
        result = _generate_structured(gen, "prompt", _TestSchema)
        assert result is None


# ── LLMRouteProvider ──


class TestLLMRouteProvider:
    def test_routes_fast_path(self) -> None:
        gen = _StubGenerator([{"route": "fast_path", "reason": "simple query"}])
        provider = LLMRouteProvider(gen)
        result = provider.route(_make_state(task="简单查询"))
        assert result["status"] == "fast_path"
        assert result["execution_mode"] == "fast_path"
        assert result["route_reason"] == "simple query"

    def test_routes_decompose(self) -> None:
        gen = _StubGenerator([{"route": "decompose", "reason": "multi-hop"}])
        provider = LLMRouteProvider(gen, decompose_enabled=True)
        result = provider.route(_make_state(task="对比 A 和 B"))
        assert result["status"] == "decompose"
        assert result["execution_mode"] == "decompose"

    def test_decompose_downgrades_when_disabled(self) -> None:
        """子 Agent 编排未启用时，decompose → direct，execution_mode 同步降级"""
        gen = _StubGenerator([{"route": "decompose", "reason": "multi-hop"}])
        provider = LLMRouteProvider(gen)
        result = provider.route(_make_state(task="对比 A 和 B"))
        assert result["status"] == "direct"
        assert result["execution_mode"] == "direct"
        assert result["decompose_disabled_single_agent_mode"] is True
        assert "decompose_disabled" in result["route_reason"]

    def test_routes_direct(self) -> None:
        gen = _StubGenerator([{"route": "direct", "reason": "needs tools"}])
        provider = LLMRouteProvider(gen)
        result = provider.route(_make_state())
        assert result["status"] == "direct"
        assert result["execution_mode"] == "direct"
        assert result["status"] == "direct"

    def test_unparseable_falls_back_to_direct(self) -> None:
        gen = _FailingGenerator()
        provider = LLMRouteProvider(gen)
        result = provider.route(_make_state())
        assert result["status"] == "direct"
        assert result["route_reason"] == "agent_research"

    def test_invalid_route_falls_back_to_direct(self) -> None:
        gen = _StubGenerator([{"route": "unknown_path", "reason": "bad"}])
        provider = LLMRouteProvider(gen)
        result = provider.route(_make_state())
        assert result["status"] == "direct"


# ── LLMEvaluateDecisionProvider ──


class TestLLMEvaluateDecisionProvider:
    def test_decide_execute(self) -> None:
        gen = _StubGenerator([
            {
                "action": "execute",
                "tool_calls": [
                    {
                        "tool_call_id": "tc_abc123def456",
                        "tool_name": "vector_search",
                        "arguments": {"query": "test", "top_k": 8},
                    }
                ],
                "thought": "need more evidence",
                "confidence": 0.8,
            }
        ])
        provider = LLMEvaluateDecisionProvider(gen)
        result = provider.decide(
            _make_state(),
            definition=AgentDefinition(
                agent_type="research",
                description="test",
                system_prompt="test",
                allowed_tools=["vector_search"],
            ),
            budget_remaining=5000,
            context=_make_context(),
        )
        assert isinstance(result, ThinkOutput)
        assert result.action == "execute"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "vector_search"

    def test_decide_synthesize(self) -> None:
        gen = _StubGenerator([
            {
                "action": "synthesize",
                "tool_calls": [],
                "thought": "evidence is sufficient",
                "confidence": 0.95,
                "stop_reason": "sufficient_evidence",
            }
        ])
        provider = LLMEvaluateDecisionProvider(gen)
        result = provider.decide(
            _make_state(),
            definition=AgentDefinition(
                agent_type="research",
                description="test",
                system_prompt="test",
                allowed_tools=[],
            ),
            budget_remaining=5000,
            context=_make_context(),
        )
        assert isinstance(result, ThinkOutput)
        assert result.action == "synthesize"

    def test_decide_pause(self) -> None:
        gen = _StubGenerator([
            {
                "action": "pause",
                "tool_calls": [],
                "thought": "need user input",
                "confidence": 0.5,
                "needs_user_input": "choose data source",
            }
        ])
        provider = LLMEvaluateDecisionProvider(gen)
        result = provider.decide(
            _make_state(),
            definition=AgentDefinition(
                agent_type="research",
                description="test",
                system_prompt="test",
                allowed_tools=[],
            ),
            budget_remaining=5000,
            context=_make_context(),
        )
        assert isinstance(result, ThinkOutput)
        assert result.action == "pause"
        assert result.needs_user_input == "choose data source"

    def test_unparseable_pauses(self) -> None:
        gen = _FailingGenerator()
        provider = LLMEvaluateDecisionProvider(gen)
        result = provider.decide(
            _make_state(),
            definition=AgentDefinition(
                agent_type="research",
                description="test",
                system_prompt="test",
                allowed_tools=[],
            ),
            budget_remaining=5000,
            context=_make_context(),
        )
        assert isinstance(result, ThinkOutput)
        assert result.action == "pause"
        assert result.confidence == 0.0


# ── LLMPlanProvider ──


class TestLLMPlanProvider:
    def test_creates_plan(self) -> None:
        gen = _StubGenerator([
            {
                "subtasks": [
                    {
                        "subtask_id": "s1",
                        "agent_type": "research",
                        "prompt": "search for A",
                        "priority": 5,
                    },
                    {
                        "subtask_id": "s2",
                        "agent_type": "research",
                        "prompt": "search for B",
                        "priority": 5,
                    },
                ],
                "edges": [],
            }
        ])
        provider = LLMPlanProvider(gen)
        plan = provider.create_plan(
            _make_state(task="对比 A 和 B"),
            definition=AgentDefinition(
                agent_type="orchestrator",
                description="test",
                system_prompt="test",
                allowed_tools=[],
            ),
        )
        assert isinstance(plan, TaskDAG)
        assert len(plan.subtasks) == 2

    def test_unparseable_raises(self) -> None:
        gen = _FailingGenerator()
        provider = LLMPlanProvider(gen)
        with pytest.raises(ValueError, match="valid TaskDAG"):
            provider.create_plan(
                _make_state(),
                definition=AgentDefinition(
                    agent_type="orchestrator",
                    description="test",
                    system_prompt="test",
                    allowed_tools=[],
                ),
            )

    def test_plan_prompt_uses_default_subtask_budget_10000(self) -> None:
        gen = _StubGenerator([
            {
                "subtasks": [
                    {
                        "subtask_id": "s1",
                        "agent_type": "research",
                        "prompt": "Research",
                        "priority": 1,
                        "estimated_tokens": 10000,
                    }
                ],
                "edges": [],
            }
        ])
        provider = LLMPlanProvider(gen)

        provider.create_plan(
            _make_state(),
            definition=AgentDefinition(
                agent_type="orchestrator",
                description="test",
                system_prompt="test",
                allowed_tools=[],
            ),
        )

        prompt = gen.calls[0][0]
        assert '"estimated_tokens": 10000' in prompt


# ── create_default_providers ──


class TestCreateDefaultProviders:
    def test_creates_three_providers_with_defaults(self) -> None:
        spec = ModelSpec(provider=ModelProvider.OLLAMA, model="test")
        config = AgentModelsConfig(
            models={"main": spec},
            default_model="main",
        )
        reg = ModelRegistry(config)
        selection = ModelSelectionPolicy()

        router, evaluator, planner = create_default_providers(reg, selection)
        assert isinstance(router, LLMRouteProvider)
        assert isinstance(evaluator, LLMEvaluateDecisionProvider)
        assert isinstance(planner, LLMPlanProvider)

    def test_creates_three_providers_with_per_node_models(self) -> None:
        main = ModelSpec(provider=ModelProvider.OLLAMA, model="main-model")
        fast = ModelSpec(provider=ModelProvider.OLLAMA, model="fast-model")
        config = AgentModelsConfig(
            models={"main": main, "fast": fast},
            default_model="main",
            fallback_model="fast",
        )
        reg = ModelRegistry(config)
        selection = ModelSelectionPolicy(route_model="fast")

        router, evaluator, planner = create_default_providers(reg, selection)
        assert isinstance(router, LLMRouteProvider)
        assert isinstance(evaluator, LLMEvaluateDecisionProvider)
        assert isinstance(planner, LLMPlanProvider)
        # router uses fast, others use main via default

    def test_decompose_enabled_is_forwarded_to_default_router(self) -> None:
        gen = _StubGenerator([{"route": "decompose", "reason": "multi-hop"}])
        reg = _FakeModelRegistry(gen)
        selection = ModelSelectionPolicy()

        router, _, _ = create_default_providers(
            reg,  # type: ignore[arg-type]
            selection,
            decompose_enabled=True,
        )
        result = router.route(_make_state(task="对比 A 和 B"))

        assert result["status"] == "decompose"
        assert result["execution_mode"] == "decompose"

    def test_per_node_max_tokens_override_model_default(self) -> None:
        gen = _StubGenerator([
            {"route": "fast_path", "reason": "single lookup"},
            {
                "action": "synthesize",
                "thought": "enough evidence",
                "confidence": 0.9,
            },
            {"subtasks": [], "edges": []},
        ])
        reg = _FakeModelRegistry(gen)
        selection = ModelSelectionPolicy(
            route_max_tokens=128,
            evaluate_max_tokens=256,
            plan_max_tokens=512,
        )

        router, evaluator, planner = create_default_providers(
            reg,  # type: ignore[arg-type]
            selection,
        )
        router.route(_make_state(task="查一个数"))
        evaluator.decide(
            _make_state(task="查一个数"),
            definition=AgentDefinition(
                agent_type="research",
                description="test",
                system_prompt="test",
                allowed_tools=[],
            ),
            budget_remaining=1000,
            context=InjectedContext(
                sections=[],
                context_budget=ContextBudgetSnapshot(max_context_tokens=1000),
            ),
        )
        planner.create_plan(
            _make_state(task="查一个数"),
            definition=AgentDefinition(
                agent_type="orchestrator",
                description="test",
                system_prompt="test",
                allowed_tools=[],
            ),
        )

        assert gen.calls[0][2]["max_tokens"] == 128
        assert gen.calls[1][2]["max_tokens"] == 256
        assert gen.calls[2][2]["max_tokens"] == 512
