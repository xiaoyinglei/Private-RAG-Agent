from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy
from rag.agent.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from rag.agent.core.llm_context import (
    AgentLLMContextAssembler,
    AgentLLMContextOverflowError,
)
from rag.agent.core.llm_providers import (
    LLMGoalContractProvider,
    LLMRetrievalHintProvider,
    LLMToolDecisionProvider,
    RetrievalHintDecision,
    create_default_providers,
)
from rag.agent.core.llm_registry import ModelRegistry, ResolvedModel
from rag.agent.goal_runtime import GoalContractHint
from rag.agent.graphs.nodes.goal_runtime import _retrieval_hint
from rag.agent.memory.models import ContextBudgetSnapshot, ContextSection, InjectedContext
from rag.agent.state import ThinkOutput
from rag.providers.llm_gateway import LLMGateway, structured_accounted_prompt
from rag.schema.llm import LLMCallStage, LLMProviderResult, LLMStageBudget, LLMUsage
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


class _WordTokenAccounting:
    def count(self, text: str) -> int:
        return len(text.split())

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str:
        words = text.split()
        clipped = " ".join(words[: max(token_budget, 0)])
        if add_ellipsis and len(words) > token_budget and token_budget > 0:
            return f"{clipped} ..."
        return clipped


class _GatewayStructuredGenerator:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def generate_structured_with_usage(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> LLMProviderResult[BaseModel]:
        del prompt, kwargs
        self.calls += 1
        return LLMProviderResult(
            value=schema.model_validate(self.payload),
            usage=LLMUsage(
                input_tokens=10,
                output_tokens=4,
                source="provider",
            ),
        )


def _gateway(generator: object, stage: LLMCallStage) -> LLMGateway:
    return LLMGateway(
        generator=generator,
        token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
        model_context_tokens=2_000,
        stage_budgets={
            stage: LLMStageBudget(
                max_input_tokens=1_500,
                max_output_tokens=100,
                safety_margin_tokens=10,
            )
        },
    )


def _context_assembler(
    gateway: LLMGateway,
    stage: LLMCallStage,
) -> AgentLLMContextAssembler:
    return AgentLLMContextAssembler(
        token_accounting=gateway.token_accounting,
        stage_budgets={stage: gateway.effective_stage_budget(stage)},
    )


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


# ── LLMRetrievalHintProvider ──


@pytest.mark.anyio
async def test_goal_contract_provider_uses_shared_gateway_and_schema() -> None:
    generator = _GatewayStructuredGenerator(
        {
            "deliverable_kinds": ["answer", "evidence"],
            "reason": "The task requests sources.",
        }
    )
    gateway = _gateway(generator, LLMCallStage.GOAL_CONTRACT)
    provider = LLMGoalContractProvider(
        generator,
        gateway=gateway,
        context_assembler=_context_assembler(
            gateway,
            LLMCallStage.GOAL_CONTRACT,
        ),
    )

    hint = await provider.infer(_make_state(task="Explain with sources"))

    assert isinstance(hint, GoalContractHint)
    assert hint.deliverable_kinds == ["answer", "evidence"]
    assert generator.calls == 1


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

    def test_sync_fallback_context_overflow_does_not_call_model(self) -> None:
        generator = _StubGenerator([{"reason": "unused"}])
        provider = LLMRetrievalHintProvider(generator)

        with pytest.raises(AgentLLMContextOverflowError):
            provider.hint(_make_state(task="required task " * 5_000))

        assert generator.calls == []

    def test_rejects_different_gateway_and_assembler_token_accounting(self) -> None:
        generator = _GatewayStructuredGenerator({"reason": "unused"})
        gateway = _gateway(generator, LLMCallStage.RETRIEVAL_HINT)
        other_gateway = _gateway(generator, LLMCallStage.RETRIEVAL_HINT)

        with pytest.raises(ValueError, match="must share the same"):
            LLMRetrievalHintProvider(
                generator,
                gateway=gateway,
                context_assembler=_context_assembler(
                    other_gateway,
                    LLMCallStage.RETRIEVAL_HINT,
                ),
            )

    @pytest.mark.anyio
    async def test_gateway_call_commits_retrieval_hint_usage(self) -> None:
        state = _make_state(task="查一个数")
        RunRegistry.remove("test")
        handles = RunRegistry.get_or_create(state["run_config"])
        generator = _GatewayStructuredGenerator({"reason": "single lookup"})
        provider = LLMRetrievalHintProvider(
            generator,
            gateway=_gateway(generator, LLMCallStage.RETRIEVAL_HINT),
        )

        result = await provider.hint(state)

        assert result["decision_reason"] == "single lookup"
        assert await handles.budget_ledger.committed() == 14
        RunRegistry.remove("test")

    @pytest.mark.anyio
    async def test_required_context_overflow_does_not_call_hint_model(self) -> None:
        generator = _GatewayStructuredGenerator({"reason": "unused"})
        gateway = LLMGateway(
            generator=generator,
            token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
            model_context_tokens=100,
            stage_budgets={
                LLMCallStage.RETRIEVAL_HINT: LLMStageBudget(
                    max_input_tokens=5,
                    max_output_tokens=10,
                    safety_margin_tokens=0,
                )
            },
        )
        definition = AgentDefinition(
            agent_type="research",
            description="test",
            system_prompt="test",
            allowed_tools=[],
        )
        provider = LLMRetrievalHintProvider(
            generator,
            gateway=gateway,
            context_assembler=_context_assembler(
                gateway,
                LLMCallStage.RETRIEVAL_HINT,
            ),
            definition=definition,
        )

        with pytest.raises(AgentLLMContextOverflowError):
            await provider.hint(_make_state(task="very large required task " * 20))

        assert generator.calls == 0

    @pytest.mark.anyio
    async def test_structured_schema_overflow_is_rejected_by_assembler(self) -> None:
        state = _make_state(task="x")
        accounting = _WordTokenAccounting()
        plain_prompt = _context_assembler(
            LLMGateway(
                generator=_GatewayStructuredGenerator({"reason": "unused"}),
                token_accounting=accounting,  # type: ignore[arg-type]
                model_context_tokens=10_000,
                stage_budgets={
                    LLMCallStage.RETRIEVAL_HINT: LLMStageBudget(
                        max_input_tokens=9_000,
                        max_output_tokens=10,
                        safety_margin_tokens=0,
                    )
                },
            ),
            LLMCallStage.RETRIEVAL_HINT,
        ).assemble_retrieval_hint(
            definition=AgentDefinition(
                agent_type="research",
                description="test",
                system_prompt="test",
                allowed_tools=[],
            ),
            state=state,
        ).prompt
        plain_tokens = accounting.count(plain_prompt)
        assert accounting.count(
            structured_accounted_prompt(
                plain_prompt,
                RetrievalHintDecision,
            )
        ) > plain_tokens

        generator = _GatewayStructuredGenerator({"reason": "unused"})
        gateway = LLMGateway(
            generator=generator,
            token_accounting=accounting,  # type: ignore[arg-type]
            model_context_tokens=10_000,
            stage_budgets={
                LLMCallStage.RETRIEVAL_HINT: LLMStageBudget(
                    max_input_tokens=plain_tokens,
                    max_output_tokens=10,
                    safety_margin_tokens=0,
                )
            },
        )
        provider = LLMRetrievalHintProvider(generator, gateway=gateway)

        with pytest.raises(AgentLLMContextOverflowError):
            await provider.hint(state)

        assert generator.calls == 0

    @pytest.mark.anyio
    async def test_goal_initialization_pauses_on_hint_context_overflow(self) -> None:
        snapshot = ContextBudgetSnapshot(
            max_context_tokens=5,
            overflow=True,
            degraded=True,
            required_truncated=["instructions"],
            warnings=["context_overflow"],
        )

        class _OverflowProvider:
            def hint(self, state: object) -> dict[str, object]:
                del state
                raise AgentLLMContextOverflowError(
                    stage=LLMCallStage.RETRIEVAL_HINT,
                    context_budget=snapshot,
                )

        result = await _retrieval_hint(
            _make_state(),
            retrieval_hint_provider=_OverflowProvider(),  # type: ignore[arg-type]
        )

        assert result["status"] == "paused"
        assert result["decision_reason"] == "context_overflow"
        assert result["context_budget"] is snapshot


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

    @pytest.mark.anyio
    async def test_gateway_call_commits_tool_decision_usage(self) -> None:
        state = _make_state()
        RunRegistry.remove("test")
        handles = RunRegistry.get_or_create(state["run_config"])
        generator = _GatewayStructuredGenerator(
            {
                "action": "synthesize",
                "thought": "enough",
                "confidence": 0.9,
            }
        )
        provider = LLMToolDecisionProvider(
            generator,
            gateway=_gateway(generator, LLMCallStage.TOOL_DECISION),
        )

        result = await provider.decide(
            state,
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
        assert await handles.budget_ledger.committed() == 14
        RunRegistry.remove("test")

    @pytest.mark.anyio
    async def test_required_context_overflow_does_not_call_decision_model(self) -> None:
        generator = _GatewayStructuredGenerator(
            {
                "action": "synthesize",
                "thought": "unused",
                "confidence": 1.0,
            }
        )
        gateway = LLMGateway(
            generator=generator,
            token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
            model_context_tokens=100,
            stage_budgets={
                LLMCallStage.TOOL_DECISION: LLMStageBudget(
                    max_input_tokens=5,
                    max_output_tokens=10,
                    safety_margin_tokens=0,
                )
            },
        )
        provider = LLMToolDecisionProvider(
            generator,
            gateway=gateway,
            context_assembler=_context_assembler(
                gateway,
                LLMCallStage.TOOL_DECISION,
            ),
        )

        with pytest.raises(AgentLLMContextOverflowError):
            await provider.decide(
                _make_state(task="large decision task " * 20),
                definition=AgentDefinition(
                    agent_type="research",
                    description="test",
                    system_prompt="large system policy " * 20,
                    allowed_tools=[],
                ),
                budget_remaining=5000,
                context=_make_context(),
            )

        assert generator.calls == 0


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
