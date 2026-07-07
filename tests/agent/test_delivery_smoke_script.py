from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.tooling import ToolSurfaceRequest
from rag.schema.llm import LLMProviderResult


def _load_smoke_module():
    script_path = Path(__file__).parents[2] / "scripts" / "agent_delivery_smoke.py"
    spec = importlib.util.spec_from_file_location("agent_delivery_smoke", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_delivery_smoke_cases_cover_core_agent_matrix() -> None:
    module = _load_smoke_module()

    cases = {case.name: case for case in module.build_cases()}

    assert {
        "math_2_plus_2",
        "explain_recursion",
        "find_agent_service",
        "read_missing_file",
        "echo_hello",
    } <= set(cases)
    assert cases["math_2_plus_2"].expected_tools == ()
    assert cases["explain_recursion"].expected_tools == ()
    assert {"tool_search", "activate_tools"} <= set(cases["math_2_plus_2"].forbidden_tools)
    assert {"tool_search", "activate_tools"} <= set(cases["explain_recursion"].forbidden_tools)
    assert cases["find_agent_service"].expected_tools == ("search_text",)
    assert cases["read_missing_file"].expected_tools == ("read_file",)
    assert cases["echo_hello"].expected_tools == ("run_command",)
    assert cases["math_2_plus_2"].tool_surface_request is None
    assert cases["explain_recursion"].tool_surface_request is None
    assert cases["find_agent_service"].tool_surface_request == {
        "requested_tool_names": ["search_text", "list_files", "read_file"],
    }
    assert "class AgentService" in cases["find_agent_service"].task
    assert cases["find_agent_service"].max_turns == 4
    assert cases["read_missing_file"].tool_surface_request == {
        "requested_tool_names": ["read_file"],
    }
    assert cases["echo_hello"].tool_surface_request == {
        "requested_tool_names": ["run_command"],
        "allow_execute_tools": True,
    }
    assert cases["find_agent_service"].workspace_path is not None
    assert cases["echo_hello"].auto_approve is False


def test_delivery_smoke_answer_contains_validation_is_case_insensitive() -> None:
    module = _load_smoke_module()
    case = next(case for case in module.build_cases() if case.name == "explain_recursion")

    assert (
        module._validate_result(
            case,
            "done",
            "Recursion is a function calling itself on smaller inputs.",
            (),
            "/tmp/workspace",
        )
        == ""
    )


class _SmokeFakeModelRegistry:
    default_model = "fake"
    fallback_model = "fake"

    def __init__(self, generator: "_SmokeFakeGenerator") -> None:
        self.generator = generator

    def resolve_for_node(self, *, node_model: str | None, node_name: str) -> object:
        del node_model, node_name
        return SimpleNamespace(generator=self.generator, kwargs={})


class _SmokeFakeGenerator:
    def __init__(
        self,
        *,
        tool_name: str | None,
        arguments: dict[str, Any] | None,
        final_answer: str,
    ) -> None:
        self._tool_name = tool_name
        self._arguments = arguments or {}
        self._final_answer = final_answer
        self.calls: list[dict[str, Any]] = []

    def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: object,
    ) -> object:
        del kwargs
        self.calls.append({"messages": messages, "tools": tools})
        if len(self.calls) == 1 and self._tool_name is not None:
            message = SimpleNamespace(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        id=f"call_{self._tool_name}",
                        function=SimpleNamespace(
                            name=self._tool_name,
                            arguments=json.dumps(self._arguments),
                        ),
                    )
                ],
            )
            return LLMProviderResult(
                value=SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason="tool_calls", message=message)]
                )
            )
        return LLMProviderResult(
            value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content=self._final_answer,
                            tool_calls=[],
                        ),
                    )
                ]
            )
        )


def _fake_generator_for_case(case: object) -> _SmokeFakeGenerator:
    name = getattr(case, "name")
    if name == "math_2_plus_2":
        return _SmokeFakeGenerator(
            tool_name=None,
            arguments=None,
            final_answer="4",
        )
    if name == "explain_recursion":
        return _SmokeFakeGenerator(
            tool_name=None,
            arguments=None,
            final_answer="recursion is a function calling itself on smaller inputs.",
        )
    if name == "find_agent_service":
        return _SmokeFakeGenerator(
            tool_name="search_text",
            arguments={
                "pattern": "class AgentService",
                "path": "rag/agent/service.py",
                "max_results": 1,
            },
            final_answer="rag/agent/service.py",
        )
    if name == "read_missing_file":
        return _SmokeFakeGenerator(
            tool_name="read_file",
            arguments={"path": "does-not-exist-agent-smoke.txt"},
            final_answer="file_not_found",
        )
    if name == "echo_hello":
        return _SmokeFakeGenerator(
            tool_name="run_command",
            arguments={"command": "echo hello", "working_dir": ".", "timeout_seconds": 3},
            final_answer="hello",
        )
    raise AssertionError(f"unhandled smoke case: {name}")


@pytest.mark.anyio
async def test_delivery_smoke_cases_run_against_agent_service_with_fake_model() -> None:
    module = _load_smoke_module()

    for case in module.build_cases():
        generator = _fake_generator_for_case(case)
        service = AgentService(
            definition=AgentRuntimePolicy.test_factory(
                agent_type="generic",
                description="Generic",
                system_prompt="Use only visible tools.",
                allowed_tools=[],
                max_iterations=3,
            ),
            tool_registry=create_builtin_tool_registry(runners={}),
            model_registry=_SmokeFakeModelRegistry(generator),  # type: ignore[arg-type]
        )

        result = await service.run(
            AgentRunRequest(
                task=case.task,
                run_id=f"fake-smoke-{case.name}",
                thread_id=f"fake-smoke-{case.name}",
                workspace_path=case.workspace_path,
                max_turns=case.max_turns,
                tool_surface_request=(
                    ToolSurfaceRequest.model_validate(case.tool_surface_request)
                    if case.tool_surface_request is not None
                    else None
                ),
            )
        )

        tools = tuple(tool.tool_name for tool in result.tool_results)
        assert module._validate_result(
            case,
            result.status,
            result.final_answer,
            tools,
            result.workspace_path,
        ) == ""
        if case.expected_tools:
            assert case.expected_tools[0] in [
                tool["function"]["name"] for tool in generator.calls[0]["tools"]
            ]
        else:
            assert generator.calls[0]["tools"] == []
