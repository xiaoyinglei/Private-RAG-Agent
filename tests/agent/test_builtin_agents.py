from __future__ import annotations

from rag.agent.builtin import (
    BUILTIN_AGENT_DEFINITIONS,
    create_builtin_agent_registry,
)
from rag.agent.builtin.generic import GENERIC_AGENT, GENERIC_SYSTEM_PROMPT
from rag.agent.tools.builtins import RESIDENT_CODING_TOOL_NAMES


def test_builtin_registry_exposes_only_supported_generic_agent() -> None:
    registry = create_builtin_agent_registry()

    assert tuple(BUILTIN_AGENT_DEFINITIONS) == ("generic",)
    assert registry.get("generic") is GENERIC_AGENT


def test_generic_agent_uses_approved_resident_coding_order() -> None:
    assert GENERIC_AGENT.core_tool_names == RESIDENT_CODING_TOOL_NAMES
    assert GENERIC_AGENT.deferred_tool_names == ()


def test_generic_prompt_does_not_hard_code_runtime_tool_names() -> None:
    forbidden = (
        "tool_search",
        "activate_tools",
        "write_file",
        "run_python",
        "tool_repl",
    )

    assert all(name not in GENERIC_SYSTEM_PROMPT for name in forbidden)
    assert "Tool definitions are the authority" in GENERIC_SYSTEM_PROMPT
