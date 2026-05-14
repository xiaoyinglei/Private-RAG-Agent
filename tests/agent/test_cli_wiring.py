from __future__ import annotations

from rag.agent.cli import _build_llm_tool_runners
from rag.agent.tools.llm_tools import LLMCompareInput, LLMTextOutput


class _ChatBinding:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"response:{prompt}"


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
