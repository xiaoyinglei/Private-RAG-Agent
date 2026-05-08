from __future__ import annotations

from rag.agent.tools.llm_tools import ALL_LLM_TOOLS, LLMCompareInput, LLMGenerateInput, LLMSummarizeInput


def test_all_llm_tools_contains_expected_specs() -> None:
    by_name = {tool.name: tool for tool in ALL_LLM_TOOLS}

    assert set(by_name) == {"llm_generate", "llm_summarize", "llm_compare"}


def test_llm_tool_permissions_are_generation_only() -> None:
    for tool in ALL_LLM_TOOLS:
        assert tool.permissions.generate is True
        assert tool.permissions.read_db is False
        assert tool.permissions.write_db is False
        assert tool.permissions.external_network is False
        assert tool.requires_confirmation is False


def test_llm_tool_input_models_are_specific_to_tool_intent() -> None:
    by_name = {tool.name: tool for tool in ALL_LLM_TOOLS}

    assert by_name["llm_generate"].input_model is LLMGenerateInput
    assert by_name["llm_summarize"].input_model is LLMSummarizeInput
    assert by_name["llm_compare"].input_model is LLMCompareInput
