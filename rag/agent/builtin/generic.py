"""Generic agent — one prompt, no role identity.

Capabilities are tools discovered at runtime, not baked into the prompt.
The model decides what tools to use based on the task, guided by tool_search
and the deferred tool mechanism.

Tool categories:
  CORE (always visible):  tool_search, activate_tools,
                           list_files, read_file, write_file,
                           invoke_skill, materialize_skill_asset when skills exist
  DEFERRED (activate on demand): search_knowledge, search_assets,
                                 task,
                                 llm_generate, llm_summarize, llm_compare,
                                 run_python, search_text, apply_patch,
                                 run_command, update_plan, tool_repl,
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
identifiers, evidence links, and artifact paths in your answer.

## Skills

Skills are reusable workflows loaded from .agents/skills/ on disk.
When a skill listed in <available_skills> matches the user's request,
this is a BLOCKING REQUIREMENT: invoke the relevant skill with
invoke_skill BEFORE answering or following the workflow. Do NOT guess
skill ids — only use ids from the listing. Do NOT invoke a skill
that is already loaded in the current conversation."""


GENERIC_AGENT = AgentRuntimePolicy(
    agent_type="generic",
    description="General-purpose research assistant with tool discovery.",
    system_instructions=GENERIC_SYSTEM_PROMPT,
    core_tool_names=(
        "tool_search",
        "activate_tools",
        "list_files",
        "read_file",
        "write_file",
    ),
    deferred_tool_names=(
        "task",
        "search_knowledge",
        "search_assets",
        "llm_summarize",
        "llm_compare",
        "llm_generate",
        "run_python",
        "search_text",
        "apply_patch",
        "run_command",
        "update_plan",
        "tool_repl",
        "structured_probe",
    ),
    model_selection=ModelSelectionPolicy(
        thinking=True,
        retrieval_hint_max_tokens=256,
        tool_decision_max_tokens=768,
    ),
    max_iterations=10,
    max_depth=2,
    tool_policy=ToolPolicy(max_parallel_calls=4),
)


__all__ = ["GENERIC_AGENT", "GENERIC_SYSTEM_PROMPT"]
