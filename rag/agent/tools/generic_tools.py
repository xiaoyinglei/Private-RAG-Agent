"""Generic coding-agent tools — I/O models and ToolSpec for tools without BaseTool.

Tools that have BaseTool equivalents (search_text, apply_patch, run_command, tool_repl)
are defined in workspace_tools.py.  This file keeps:

1. I/O model definitions (canonical source, imported by workspace_tools.py)
2. update_plan — ToolSpec + contextual runner (no BaseTool: needs LoopState)
"""

from __future__ import annotations

from typing import Literal as LiteralType

from pydantic import BaseModel, Field

from rag.agent.tools.card import ToolCard
from rag.agent.tools.spec import ExecutionCategory, ToolError, ToolPermissions, ToolSpec

# ═══════════════════════════════════════════════════════════════════════════════
# I/O models (canonical source — imported by workspace_tools.py)
# ═══════════════════════════════════════════════════════════════════════════════


class SearchTextInput(BaseModel):
    pattern: str = Field(min_length=1, max_length=2000, description="Search pattern (literal or regex).")
    path: str = Field(default=".", description="Directory or file path to search within.")
    file_types: str | None = Field(default=None, description="Comma-separated extensions, e.g. '.py,.md'.")
    max_results: int = Field(default=40, ge=1, le=200, description="Max matching lines to return.")
    regex: bool = Field(default=False, description="Interpret pattern as regex.")
    context_lines: int = Field(default=0, ge=0, le=5, description="Surrounding context lines.")


class SearchTextMatch(BaseModel):
    file_path: str = ""
    line_number: int = 0
    line_content: str = ""
    match_start: int = 0
    match_end: int = 0


class SearchTextOutput(BaseModel):
    matches: list[SearchTextMatch] = Field(default_factory=list)
    total_matches: int = 0
    truncated: bool = False
    message: str = ""


class ApplyPatchInput(BaseModel):
    file_path: str = Field(min_length=1, description="Path to file, relative to workspace root.")
    old_string: str = Field(min_length=1, description="Exact text to replace.")
    new_string: str = Field(description="Replacement text.")
    replace_all: bool = Field(default=False, description="Replace ALL occurrences.")


class ApplyPatchOutput(BaseModel):
    file_path: str = ""
    replaced: bool = False
    occurrences: int = 0
    message: str = ""


class RunCommandInput(BaseModel):
    command: str = Field(min_length=1, max_length=4000, description="Shell command to execute.")
    working_dir: str = Field(default=".", description="Working directory relative to workspace.")
    timeout_seconds: int = Field(default=120, ge=1, le=600, description="Max execution time in seconds.")
    env: dict[str, str] | None = Field(default=None, description="Optional environment variables.")


class RunCommandOutput(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    truncated: bool = False
    duration_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# update_plan — contextual tool (needs LoopState, no BaseTool equivalent)
# ═══════════════════════════════════════════════════════════════════════════════

PlanAction = LiteralType["add", "update", "complete", "reorder"]


class PlanStep(BaseModel):
    id: str = Field(default="", description="Unique step ID.")
    description: str = Field(default="", description="What this step does.")
    status: LiteralType["pending", "in_progress", "completed", "blocked"] = "pending"


class UpdatePlanInput(BaseModel):
    action: PlanAction = Field(description="add, update, complete, or reorder.")
    steps: list[PlanStep] = Field(default_factory=list, description="Steps to add or update.")
    step_ids: list[str] = Field(default_factory=list, description="Step IDs to complete.")
    summary: str = Field(default="", max_length=500, description="Progress summary.")


class UpdatePlanOutput(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    summary: str = ""
    message: str = ""


update_plan_spec = ToolSpec(
    name="update_plan",
    description=(
        "Explicitly track and update your plan. Use this to add steps, mark steps "
        "as in_progress or completed, reorder tasks, and write a progress summary."
    ),
    input_model=UpdatePlanInput,
    output_model=UpdatePlanOutput,
    error_model=ToolError,
    permissions=ToolPermissions(),
    execution_category=ExecutionCategory.TRANSFORM,
    timeout_seconds=3.0,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=50,
    aci=ToolCard(
        when_to_use="Use on EVERY turn to keep your plan visible.",
        when_not_to_use="Do not skip plan updates.",
        activation_group="resident",
        selection_tags=("plan", "tracking"),
        domains=("agent_internal",),
    ),
)

ALL_GENERIC_TOOLS: list[ToolSpec] = [update_plan_spec]
