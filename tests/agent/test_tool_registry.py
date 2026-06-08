from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentDefinition
from rag.agent.tools.registry import (
    ToolInputValidationError,
    ToolOutputValidationError,
    ToolRegistry,
    ToolRunnerMissingError,
)
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec
from rag.schema.runtime import AccessPolicy


class DummyInput(BaseModel):
    text: str


class DummyOutput(BaseModel):
    result: str


_dummy_spec = ToolSpec(
    name="dummy",
    description="A dummy tool",
    input_model=DummyInput,
    output_model=DummyOutput,
    error_model=ToolError,
    permissions=ToolPermissions(),
    timeout_seconds=1.0,
)


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        registry.register(_dummy_spec)
        assert registry.get("dummy") is _dummy_spec

    def test_get_missing_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_all(self) -> None:
        registry = ToolRegistry()
        registry.register(_dummy_spec)
        another = ToolSpec(
            name="another",
            description="x",
            input_model=DummyInput,
            output_model=DummyOutput,
            error_model=ToolError,
            permissions=ToolPermissions(),
            timeout_seconds=2.0,
        )
        registry.register(another)
        names = [spec.name for spec in registry.list_all()]
        assert "dummy" in names
        assert "another" in names

    def test_register_duplicate_overwrites(self) -> None:
        registry = ToolRegistry()
        registry.register(_dummy_spec)
        updated = ToolSpec(
            name="dummy",
            description="updated",
            input_model=DummyInput,
            output_model=DummyOutput,
            error_model=ToolError,
            permissions=ToolPermissions(),
            timeout_seconds=3.0,
        )
        registry.register(updated)
        assert registry.get("dummy").timeout_seconds == 3.0

    @pytest.mark.anyio
    async def test_runner_executes_with_validated_input_and_output(self) -> None:
        registry = ToolRegistry()

        def runner(payload: DummyInput) -> dict[str, str]:
            return {"result": payload.text.upper()}

        registry.register(_dummy_spec, runner=runner)

        result = await registry.run("dummy", {"text": "hello"})

        assert result == DummyOutput(result="HELLO")

    @pytest.mark.anyio
    async def test_contextual_runner_receives_trusted_run_config(self) -> None:
        from rag.agent.tools.registry import ToolExecutionContext

        access_policy = AccessPolicy(allowed_runtimes=frozenset())
        run_config = AgentRunConfig(
            run_id="contextual-runner",
            thread_id="contextual-runner",
            budget_total=100,
            max_depth=1,
            access_policy=access_policy,
        )
        seen_contexts: list[ToolExecutionContext] = []

        def runner(
            payload: DummyInput,
            context: ToolExecutionContext,
        ) -> DummyOutput:
            seen_contexts.append(context)
            return DummyOutput(result=payload.text)

        registry = ToolRegistry()
        registry.register(_dummy_spec)
        registry.register_contextual_runner("dummy", runner)

        result = await registry.run(
            "dummy",
            {"text": "hello"},
            execution_context=ToolExecutionContext(run_config=run_config),
        )

        assert result == DummyOutput(result="hello")
        assert seen_contexts[0].run_config is run_config

    @pytest.mark.anyio
    async def test_contextual_runner_receives_trusted_state_and_definition(self) -> None:
        from rag.agent.tools.registry import ToolExecutionContext

        run_config = AgentRunConfig(
            run_id="trusted-context",
            thread_id="trusted-context",
            budget_total=100,
            max_depth=1,
            access_policy=AccessPolicy.default(),
        )
        state = {"task": "trusted task", "run_config": run_config}
        definition = AgentDefinition(
            agent_type="test",
            description="test",
            system_prompt="trusted policy",
            allowed_tools=["dummy"],
        )
        seen_contexts: list[ToolExecutionContext] = []

        def runner(
            payload: DummyInput,
            context: ToolExecutionContext,
        ) -> DummyOutput:
            seen_contexts.append(context)
            return DummyOutput(result=payload.text)

        registry = ToolRegistry()
        registry.register(_dummy_spec)
        registry.register_contextual_runner("dummy", runner)

        await registry.run(
            "dummy",
            {"text": "hello"},
            execution_context=ToolExecutionContext(
                run_config=run_config,
                state=state,  # type: ignore[arg-type]
                definition=definition,
            ),
        )

        assert seen_contexts[0].state is state
        assert seen_contexts[0].definition is definition

    @pytest.mark.anyio
    async def test_missing_runner_fails_closed(self) -> None:
        registry = ToolRegistry()
        registry.register(_dummy_spec)

        with pytest.raises(ToolRunnerMissingError, match="dummy has no registered callable runner"):
            await registry.run("dummy", {"text": "hello"})

    @pytest.mark.anyio
    async def test_invalid_runner_input_raises_typed_validation_error(self) -> None:
        registry = ToolRegistry()
        registry.register(_dummy_spec, runner=lambda payload: DummyOutput(result=payload.text))

        with pytest.raises(ToolInputValidationError):
            await registry.run("dummy", {"unexpected": "hello"})

    @pytest.mark.anyio
    async def test_invalid_runner_output_raises_typed_validation_error(self) -> None:
        registry = ToolRegistry()
        registry.register(_dummy_spec, runner=lambda payload: {"missing": payload.text})

        with pytest.raises(ToolOutputValidationError):
            await registry.run("dummy", {"text": "hello"})
