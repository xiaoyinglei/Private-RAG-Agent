from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy
from rag.agent.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from rag.agent.core.llm_providers import (
    LLMRetrievalHintProvider,
    LLMToolDecisionProvider,
    _generate_structured,
    create_default_providers,
)
from rag.agent.core.llm_registry import ModelRegistry, ResolvedModel
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
        "iteration": 0,
        "status": "running",
        "decision_reason": None,
        "stop_reason": None,
        "needs_user_input": None,
        "pending_tool_calls": [],
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "user_decision": None,
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
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


# ── LLMRetrievalHintProvider ──


class TestLLMRetrievalHintProvider:
    def test_returns_retrieval_hint_without_execution_route(self) -> None:
        gen = _StubGenerator([{"reason": "simple query"}])
        provider = LLMRetrievalHintProvider(gen)
        result = provider.hint(_make_state(task="简单查询"))
        assert result["decision_reason"] == "simple query"
        assert "status" not in result
        assert "execution_mode" not in result

    def test_multi_hop_hint_does_not_create_decompose_branch(self) -> None:
        gen = _StubGenerator([
            {
                "reason": "multi-hop",
                "retrieval_signals": {"allow_graph_expansion": True},
            }
        ])
        provider = LLMRetrievalHintProvider(gen)
        result = provider.hint(_make_state(task="对比 A 和 B"))
        assert result["retrieval_signals"].allow_graph_expansion is True
        assert "status" not in result
        assert "execution_mode" not in result

    def test_unparseable_falls_back_to_generic_hint(self) -> None:
        gen = _FailingGenerator()
        provider = LLMRetrievalHintProvider(gen)
        result = provider.hint(_make_state())
        assert result["decision_reason"] == "agent_research"
        assert "status" not in result


# ── LLMToolDecisionProvider ──


class TestLLMToolDecisionProvider:
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
        provider = LLMToolDecisionProvider(gen)
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
        provider = LLMToolDecisionProvider(gen)
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
        provider = LLMToolDecisionProvider(gen)
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
        provider = LLMToolDecisionProvider(gen)
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


# ── create_default_providers ──


class TestCreateDefaultProviders:
    def test_creates_two_providers_with_defaults(self) -> None:
        spec = ModelSpec(provider=ModelProvider.OLLAMA, model="test")
        config = AgentModelsConfig(
            models={"main": spec},
            default_model="main",
        )
        reg = ModelRegistry(config)
        selection = ModelSelectionPolicy()

        hint_provider, decision_provider = create_default_providers(reg, selection)
        assert isinstance(hint_provider, LLMRetrievalHintProvider)
        assert isinstance(decision_provider, LLMToolDecisionProvider)

    def test_creates_two_providers_with_per_node_models(self) -> None:
        main = ModelSpec(provider=ModelProvider.OLLAMA, model="main-model")
        fast = ModelSpec(provider=ModelProvider.OLLAMA, model="fast-model")
        config = AgentModelsConfig(
            models={"main": main, "fast": fast},
            default_model="main",
            fallback_model="fast",
        )
        reg = ModelRegistry(config)
        selection = ModelSelectionPolicy(retrieval_hint_model="fast")

        hint_provider, decision_provider = create_default_providers(reg, selection)
        assert isinstance(hint_provider, LLMRetrievalHintProvider)
        assert isinstance(decision_provider, LLMToolDecisionProvider)
        # hints use fast, decisions use main via default

    def test_per_node_max_tokens_override_model_default(self) -> None:
        gen = _StubGenerator([
            {"reason": "single lookup"},
            {
                "action": "synthesize",
                "thought": "enough evidence",
                "confidence": 0.9,
            },
        ])
        reg = _FakeModelRegistry(gen)
        selection = ModelSelectionPolicy(
            retrieval_hint_max_tokens=128,
            tool_decision_max_tokens=256,
        )

        hint_provider, decision_provider = create_default_providers(
            reg,  # type: ignore[arg-type]
            selection,
        )
        hint_provider.hint(_make_state(task="查一个数"))
        decision_provider.decide(
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
        assert gen.calls[0][2]["max_tokens"] == 128
        assert gen.calls[1][2]["max_tokens"] == 256
