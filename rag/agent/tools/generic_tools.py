"""Generic coding-agent tools — search, edit, execute, plan.

These are language- and domain-agnostic tools that every coding agent needs.
Each tool is packaged with a ToolCard (ACI companion) for discoverability.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag.agent.tools.card import ToolCard
from rag.agent.tools.spec import ExecutionCategory, ToolError, ToolPermissions, ToolSpec

# ═══════════════════════════════════════════════════════════════════════════════
# search_text — grep/rg for the workspace
# ═══════════════════════════════════════════════════════════════════════════════


class SearchTextInput(BaseModel):
    """Input for search_text — full-text / regex search in the workspace."""

    pattern: str = Field(
        min_length=1,
        max_length=2000,
        description="Search pattern (literal text or regex when regex=True).",
    )
    path: str = Field(
        default=".",
        description="Directory or file path to search within. Defaults to workspace root.",
    )
    file_types: str | None = Field(
        default=None,
        description="Optional comma-separated file extensions, e.g. '.py,.md,.toml'.",
    )
    max_results: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Maximum number of matching lines to return.",
    )
    regex: bool = Field(
        default=False,
        description="Interpret pattern as a regex. Default is literal substring.",
    )
    context_lines: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Number of surrounding context lines to show around each match.",
    )


class SearchTextMatch(BaseModel):
    """A single match from search_text."""

    file_path: str
    line_number: int
    line_content: str
    match_start: int = 0
    match_end: int = 0


class SearchTextOutput(BaseModel):
    """Output from search_text."""

    matches: list[SearchTextMatch] = Field(default_factory=list)
    total_matches: int = 0
    truncated: bool = False
    message: str = ""


search_text_spec = ToolSpec(
    name="search_text",
    description=(
        "Search the workspace for text patterns (literal or regex). "
        "Returns file_path, line_number, line_content for each match. "
        "Equivalent to grep/rg. Use before read_file to locate relevant code."
    ),
    input_model=SearchTextInput,
    output_model=SearchTextOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_fs=True),
    execution_category=ExecutionCategory.READ,
    timeout_seconds=15.0,
    max_retries=1,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=200,
    aci=ToolCard(
        when_to_use=(
            "Use when you need to find where a function, class, variable, or pattern "
            "is defined or used in the codebase. Call this BEFORE read_file to locate "
            "the right files."
        ),
        when_not_to_use=(
            "Do not use for semantic queries about code behavior — read the code. "
            "Do not use for searching document content — use vector_search instead."
        ),
        preconditions=("workspace must contain files",),
        required_context=("search pattern or identifier name",),
        input_examples=(
            {"pattern": "def test_search", "path": "tests/", "file_types": ".py"},
            {"pattern": "import.*pandas", "regex": True, "max_results": 20},
        ),
        output_examples=(
            "tests/test_search.py:42: def test_search_text_matches(pattern):",
        ),
        output_cap_policy="truncate",
        failure_codes=("timeout", "path_not_found"),
        retryable=True,
        user_recoverable=True,
        model_next_action="Try a more specific pattern, narrow the path, or use regex=False for literal search.",
        selection_tags=("search", "grep", "find", "locate"),
        file_types=(".py", ".md", ".toml", ".yaml", ".json", ".js", ".ts", ".rs", ".go"),
        domains=("code", "files"),
        activation_group="workspace",
    ),
)

# ═══════════════════════════════════════════════════════════════════════════════
# apply_patch — minimal precise string replacement
# ═══════════════════════════════════════════════════════════════════════════════


class ApplyPatchInput(BaseModel):
    """Input for apply_patch — exact string replacement in a file."""

    file_path: str = Field(
        min_length=1,
        description="Path to the file to edit, relative to workspace root.",
    )
    old_string: str = Field(
        min_length=1,
        description=(
            "The exact text to replace. Must match the file exactly, including "
            "indentation and blank lines. If not unique, the edit is rejected "
            "unless replace_all=True."
        ),
    )
    new_string: str = Field(
        description="The replacement text. Must be different from old_string.",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace ALL occurrences of old_string. Default is False (single replacement).",
    )


class ApplyPatchOutput(BaseModel):
    """Output from apply_patch."""

    file_path: str
    replaced: bool = False
    occurrences: int = 0
    message: str = ""


apply_patch_spec = ToolSpec(
    name="apply_patch",
    description=(
        "Apply a precise string replacement in a file. The old_string must match "
        "exactly once (or set replace_all=True). This is the preferred editing tool — "
        "use it instead of write_file for targeted edits. Never rewrite an entire file "
        "when a single-line change will do."
    ),
    input_model=ApplyPatchInput,
    output_model=ApplyPatchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(write_fs=True),
    execution_category=ExecutionCategory.WRITE,
    timeout_seconds=5.0,
    max_retries=0,
    work_budget_cost=100,
    aci=ToolCard(
        when_to_use=(
            "Use for targeted edits: rename a variable, fix a typo, add/remove a line, "
            "update a config value. ALWAYS prefer apply_patch over write_file for edits "
            "that touch less than ~30 lines."
        ),
        when_not_to_use=(
            "Do not use for creating new files — use write_file. "
            "Do not use when the edit spans more than ~30 lines — use write_file. "
            "Do not use when you're unsure of the exact text to replace — read_file first."
        ),
        preconditions=("file must exist and be readable",),
        required_context=("exact old_string from the file (copy-paste from read_file output)",),
        input_examples=(
            {"file_path": "src/app.py", "old_string": "debug = False", "new_string": "debug = True"},
            {"file_path": "README.md", "old_string": "## Usage",
             "new_string": "## Usage\n\n### Prerequisites", "replace_all": False},
        ),
        output_examples=("replaced=True occurrences=1", "replaced=False message='old_string not found in file_path'"),
        output_cap_policy="truncate",
        failure_codes=("file_not_found", "not_unique", "no_match", "unchanged"),
        retryable=True,
        user_recoverable=True,
        model_next_action=(
            "If not_unique: use replace_all=True or provide a more specific old_string with more surrounding context. "
            "If no_match: re-read the file to get the exact current text, including whitespace."
        ),
        selection_tags=("edit", "patch", "replace", "refactor"),
        file_types=(".py", ".md", ".toml", ".yaml", ".json", ".js", ".ts", ".html", ".css"),
        domains=("code", "files"),
        activation_group="workspace",
    ),
)

# ═══════════════════════════════════════════════════════════════════════════════
# run_command — shell command execution (sandboxed)
# ═══════════════════════════════════════════════════════════════════════════════


class RunCommandInput(BaseModel):
    """Input for run_command — execute a shell command in the workspace."""

    command: str = Field(
        min_length=1,
        max_length=4000,
        description=(
            "Shell command to execute. Examples: 'pytest tests/ -x', "
            "'git diff HEAD~1', 'uv add requests', 'ruff check .'"
        ),
    )
    working_dir: str = Field(
        default=".",
        description="Working directory for the command, relative to workspace root.",
    )
    timeout_seconds: int = Field(
        default=120,
        ge=1,
        le=600,
        description="Maximum execution time in seconds. Default 120s, max 600s.",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables to set for this command.",
    )


class RunCommandOutput(BaseModel):
    """Output from run_command."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    truncated: bool = False
    duration_ms: float = 0.0


run_command_spec = ToolSpec(
    name="run_command",
    description=(
        "Execute a shell command in the workspace. Use for running tests, linters, "
        "git commands, package installs, build tools, and scripts. The command runs "
        "in a sandboxed workspace directory. Output is captured and returned."
    ),
    input_model=RunCommandInput,
    output_model=RunCommandOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_fs=True, write_fs=True, execute_code=True),
    execution_category=ExecutionCategory.EXECUTE,
    timeout_seconds=600.0,
    max_retries=0,
    work_budget_cost=500,
    aci=ToolCard(
        when_to_use=(
            "Use for anything that is NOT a Python script: running tests (pytest, cargo test), "
            "linting (ruff, mypy), git operations (diff, log, status), package management "
            "(pip install, uv add), build tools (make, cargo build), or shell scripts."
        ),
        when_not_to_use=(
            "Do not use for ad-hoc Python code — use run_python_inline. "
            "Do not use for interactive commands or commands that need user input. "
            "Do not use for destructive operations outside the workspace."
        ),
        preconditions=("workspace directory must exist",),
        required_context=("exact command string", "understanding of expected output"),
        input_examples=(
            {"command": "pytest tests/agent/ -x -q", "timeout_seconds": 120},
            {"command": "git diff --stat HEAD~1"},
            {"command": "uv add requests"},
        ),
        output_examples=("exit_code=0 stdout='...' stderr=''", "exit_code=1 stderr='FAILED test_foo'"),
        output_cap_policy="truncate",
        failure_codes=("timeout", "non_zero_exit", "command_not_found", "permission_denied"),
        retryable=True,
        user_recoverable=True,
        model_next_action=(
            "Check stderr for error messages. Fix the issue and retry. "
            "If timeout: reduce scope or increase timeout_seconds."
        ),
        selection_tags=("shell", "execute", "test", "git", "build"),
        file_types=(),
        domains=("code", "devops"),
        activation_group="workspace",
    ),
)

# ═══════════════════════════════════════════════════════════════════════════════
# update_plan — explicit plan step tracking
# ═══════════════════════════════════════════════════════════════════════════════

from typing import Literal as LiteralType  # noqa: E402

PlanAction = LiteralType["add", "update", "complete", "reorder"]


class PlanStep(BaseModel):
    """A single plan step."""

    id: str = Field(default="", description="Unique step ID. Auto-generated if empty for 'add' action.")
    description: str = Field(default="", description="What this step does.")
    status: LiteralType["pending", "in_progress", "completed", "blocked"] = "pending"


class UpdatePlanInput(BaseModel):
    """Input for update_plan — mutate the agent's plan state."""

    action: PlanAction = Field(
        description="What to do: 'add' a step, 'update' an existing step, 'complete' a step, 'reorder' steps.",
    )
    steps: list[PlanStep] = Field(
        default_factory=list,
        description="Steps to add or update. For 'complete', only id + status is needed.",
    )
    step_ids: list[str] = Field(
        default_factory=list,
        description="Step IDs to complete (for action='complete').",
    )
    summary: str = Field(
        default="",
        max_length=500,
        description="Optional one-line summary of plan progress for the working memory.",
    )


class UpdatePlanOutput(BaseModel):
    """Output from update_plan — the current plan after mutation."""

    steps: list[PlanStep] = Field(default_factory=list)
    summary: str = ""
    message: str = ""


update_plan_spec = ToolSpec(
    name="update_plan",
    description=(
        "Explicitly track and update your plan. Use this to add steps, mark steps "
        "as in_progress or completed, reorder tasks, and write a progress summary. "
        "This is YOUR working memory — the runtime reads it back into context on the "
        "next turn so you know where you left off."
    ),
    input_model=UpdatePlanInput,
    output_model=UpdatePlanOutput,
    error_model=ToolError,
    permissions=ToolPermissions(),
    execution_category=ExecutionCategory.TRANSFORM,
    timeout_seconds=3.0,
    max_retries=0,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=50,
    aci=ToolCard(
        when_to_use=(
            "Use on EVERY turn to keep your plan visible. At minimum, mark the current "
            "step as in_progress when you start working on it, and completed when done. "
            "Break complex tasks into steps so progress is trackable."
        ),
        when_not_to_use=(
            "Do not skip plan updates — the runtime uses plan state for context, "
            "and without updates you may lose track of multi-step tasks."
        ),
        preconditions=(),
        required_context=("current task and progress",),
        input_examples=(
            {"action": "add", "steps": [{"description": "Fix auth bug", "status": "in_progress"}]},
            {"action": "complete", "step_ids": ["step-1"]},
            {"action": "update", "steps": [{"id": "step-2", "status": "in_progress", "description": "Run tests"}]},
        ),
        output_examples=("steps=[{id:step-1,status:completed},{id:step-2,status:in_progress}]",),
        output_cap_policy="truncate",
        failure_codes=("invalid_step_id",),
        retryable=True,
        user_recoverable=True,
        model_next_action="Check step IDs and retry.",
        selection_tags=("plan", "tracking", "progress"),
        file_types=(),
        domains=("agent_internal",),
        activation_group="resident",
    ),
)

# ═══════════════════════════════════════════════════════════════════════════════
# Module aggregates
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# tool_repl — programmatic batch tool calling (Claude REPL-like)
# ═══════════════════════════════════════════════════════════════════════════════

tool_repl_spec = ToolSpec(
    name="tool_repl",
    description=(
        "Enter batch tool-calling mode. Write Python code that declares tool "
        "calls using tools.declare(name, **args). All declared tools are executed "
        "in batch after your code finishes — you don't wait for each one. "
        "Use this when you need to call multiple tools, process their results, "
        "or iterate over data without returning to the LLM between each step."
    ),
    input_model=RunCommandInput,  # reuse: command is Python code
    output_model=RunCommandOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_fs=True, write_fs=True, execute_code=True),
    execution_category=ExecutionCategory.EXECUTE,
    timeout_seconds=120.0,
    max_retries=0,
    work_budget_cost=800,
    aci=ToolCard(
        when_to_use=(
            "Use when you need to call multiple tools in sequence or batch: "
            "search knowledge AND search assets, process results, iterate over "
            "findings. This is MORE efficient than calling tools one-by-one "
            "because results come back together on the next turn."
        ),
        when_not_to_use=(
            "Do not use for single tool calls — just call the tool directly. "
            "Do not use for simple file reads or edits — use read_file/search_text/apply_patch. "
            "Do not use for quick Python one-liners — use run_python with the code= parameter."
        ),
        preconditions=("activated tools are available via tools.list_available()",),
        required_context=("which tools to call and with what arguments",),
        input_examples=(
            {"command": (
                "tools.declare('search_knowledge', query='Q3 revenue', top_k=5)\n"
                "tools.declare('search_assets', query='financial tables', max_results=3)"
            )},
        ),
        output_examples=("stdout: 'declared: search_knowledge seq=1\\ndeclared: search_assets seq=2'",),
        output_cap_policy="truncate",
        failure_codes=("timeout", "syntax_error"),
        retryable=True,
        user_recoverable=True,
        model_next_action="Fix Python errors and retry.",
        selection_tags=("batch", "repl", "programmatic"),
        file_types=(".py",),
        domains=("code", "agent_internal"),
        activation_group="workspace",
    ),
)

ALL_GENERIC_TOOLS: list[ToolSpec] = [
    search_text_spec,
    apply_patch_spec,
    run_command_spec,
    update_plan_spec,
    tool_repl_spec,
]
