"""Generic agent — one prompt, no role identity.

Capabilities are tools discovered at runtime, not baked into the prompt.
The model decides what tools to use based on the task, guided by tool_search
and the deferred tool mechanism.

Tool categories:
  CORE (always visible):  tool_search, activate_tools, task,
                           list_files, read_file, write_file,
                           run_python, search_text, apply_patch,
                           run_command, update_plan, tool_repl
  DEFERRED (activate on demand): search_knowledge, search_assets,
                                 llm_generate, llm_summarize, llm_compare,
                                 structured_probe
"""

from __future__ import annotations

from rag.agent.core.definition import AgentRuntimePolicy, ModelSelectionPolicy, ToolPolicy

GENERIC_SYSTEM_PROMPT = """\
You are a research assistant that uses tools to answer questions and
analyze data. Retrieved evidence is your factual authority — never
invent facts. When evidence is insufficient, say so. Be direct and
concise; when you have enough context, answer immediately.

Your tools carry their own instructions for when and how to use them.
If the visible tools cannot fulfill the task, call tool_search to
discover more, then activate_tools to load them. Preserve all citation
identifiers, evidence links, and artifact paths in your answer."""


GENERIC_AGENT = AgentRuntimePolicy(
    agent_type="generic",
    description="General-purpose research assistant with tool discovery.",
    system_instructions=GENERIC_SYSTEM_PROMPT,
    core_tool_names=(
        "tool_search",
        "activate_tools",
        "task",
        "list_files",
        "read_file",
        "write_file",
        "run_python",
        "search_text",
        "apply_patch",
        "run_command",
        "update_plan",
        "tool_repl",
    ),
    deferred_tool_names=(
        "search_knowledge",
        "search_assets",
        "llm_summarize",
        "llm_compare",
        "llm_generate",
        "structured_probe",
    ),
model_selection=ModelSelectionPolicy(
        thinking=True,
        retrieval_hint_max_tokens=256,
        tool_decision_max_tokens=2048,
    ),
    max_iterations=10,
    max_depth=2,
    tool_policy=ToolPolicy(max_parallel_calls=4),
)


__all__ = ["GENERIC_AGENT", "GENERIC_SYSTEM_PROMPT"]
