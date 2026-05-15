from __future__ import annotations

import pytest

from rag.agent.builtin import create_builtin_agent_registry
from rag.agent.cli import (
    CLI_AGENT_CHOICES,
    _build_agent_service,
    _build_llm_tool_runners,
    _resolve_cli_agent_definition,
)
from rag.agent.tools.llm_tools import LLMCompareInput, LLMTextOutput


class _ChatBinding:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"response:{prompt}"


class _Runtime:
    retrieval_service = None

    def __init__(self) -> None:
        self.capability_bundle = type(
            "CapabilityBundle",
            (),
            {"chat_bindings": [_ChatBinding()]},
        )()


def test_cli_llm_runner_wiring_includes_compare_runner() -> None:
    chat = _ChatBinding()
    runners = _build_llm_tool_runners(chat)

    assert {"llm_generate", "llm_summarize", "llm_compare"} <= set(runners)
    result = runners["llm_compare"](
        LLMCompareInput(
            question="Compare A and B",
            left_context_sections=["A evidence"],
            right_context_sections=["B evidence"],
        )
    )

    assert result == LLMTextOutput(text="response:Compare A and B\n\n左:\nA evidence\n\n右:\nB evidence")


def test_cli_agent_choices_expose_top_level_agents_only() -> None:
    assert CLI_AGENT_CHOICES == ("research", "orchestrator", "compare", "factcheck")


def test_resolve_cli_agent_definition_rejects_internal_synthesize() -> None:
    registry = create_builtin_agent_registry()

    with pytest.raises(ValueError, match="not a supported CLI agent"):
        _resolve_cli_agent_definition(registry, "synthesize")


def test_build_agent_service_uses_requested_orchestrator_definition() -> None:
    service = _build_agent_service(_Runtime(), agent_type="orchestrator")

    assert service._definition.agent_type == "orchestrator"


def test_build_agent_service_rejects_unknown_agent() -> None:
    with pytest.raises(ValueError, match="not a supported CLI agent"):
        _build_agent_service(_Runtime(), agent_type="unknown")
