from __future__ import annotations

import pytest

from rag.agent.builtin import create_builtin_agent_registry
from rag.agent.cli import (
    CLI_AGENT_CHOICES,
    _build_agent_service,
    _build_llm_tool_runners,
    _resolve_cli_agent_definition,
)
from rag.agent.tools.llm_tools import LLMCompareInput, LLMGenerateInput, LLMTextOutput


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


class _RuntimeWithAssetStores(_Runtime):
    def __init__(self) -> None:
        super().__init__()
        self.stores = type(
            "Stores",
            (),
            {
                "metadata_repo": object(),
                "object_store": object(),
            },
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


def test_cli_generate_runner_preserves_supplied_grounding_ids() -> None:
    runners = _build_llm_tool_runners(_ChatBinding())

    result = runners["llm_generate"](
        LLMGenerateInput(
            prompt="Write grounded answer",
            evidence_ids=["ev1"],
            citation_ids=["cit1"],
        )
    )

    assert result == LLMTextOutput(
        text="response:Write grounded answer",
        evidence_ids=["ev1"],
        citation_ids=["cit1"],
    )


def test_cli_agent_choices_expose_top_level_agents_only() -> None:
    assert CLI_AGENT_CHOICES == ("research", "orchestrator", "compare", "factcheck")


def test_resolve_cli_agent_definition_rejects_internal_synthesize() -> None:
    registry = create_builtin_agent_registry()

    with pytest.raises(ValueError, match="not a supported CLI agent"):
        _resolve_cli_agent_definition(registry, "synthesize")


def test_build_agent_service_uses_requested_orchestrator_definition() -> None:
    service = _build_agent_service(_Runtime(), agent_type="orchestrator")

    assert service._definition.agent_type == "orchestrator"


def test_build_agent_service_registers_all_asset_tool_runners() -> None:
    service = _build_agent_service(_RuntimeWithAssetStores(), agent_type="research")

    assert service._base_tool_registry.has_runner("asset_list")
    assert service._base_tool_registry.has_runner("asset_inspect")
    assert service._base_tool_registry.has_runner("asset_read_slice")
    assert service._base_tool_registry.has_runner("asset_analyze")


def test_build_agent_service_honors_cli_model_alias_for_agent_decisions() -> None:
    service = _build_agent_service(
        _Runtime(),
        agent_type="research",
        model_alias="qwen3_8b_mlx_4bit",
    )

    assert service._model_registry is not None
    assert service._model_registry.default_model == "qwen3_8b_mlx_4bit"


def test_build_agent_service_rejects_unknown_agent() -> None:
    with pytest.raises(ValueError, match="not a supported CLI agent"):
        _build_agent_service(_Runtime(), agent_type="unknown")
