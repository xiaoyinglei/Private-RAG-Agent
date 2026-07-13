"""Materialize files that belong to an active skill into workspace scratch."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.skills.models import SkillState
from rag.agent.tools.base import BaseTool
from rag.agent.tools.card import ToolCard
from rag.agent.tools.permissions import ToolExecutionContext
from rag.agent.tools.spec import ExecutionCategory, InterruptBehavior, RiskLevel, ToolPermissions
from rag.agent.workspace import WorkspaceRuntime


class MaterializeSkillAssetInput(BaseModel):
    """Input for materialize_skill_asset."""

    model_config = ConfigDict(frozen=True)

    skill_id: str = Field(
        min_length=1,
        description="Unique id of an already active skill, such as project:xlsx.",
    )
    relative_path: str = Field(
        min_length=1,
        description="Skill-local asset path under scripts/ or references/.",
    )


class MaterializeSkillAssetOutput(BaseModel):
    """Workspace-local path for a copied skill asset."""

    model_config = ConfigDict(frozen=True)

    workspace_path: str = Field(description="Path relative to the workspace root.")
    source_fingerprint: str = Field(description="SHA-256 hash of the source asset.")
    size_bytes: int = Field(description="Copied file size in bytes.")


class MaterializeSkillAssetTool(BaseTool):
    """Copy an active skill asset into scratch so tools can use it safely."""

    name = "materialize_skill_asset"
    description = (
        "Copy a scripts/ or references/ file from an already invoked skill into "
        "workspace scratch and return the workspace-relative path."
    )
    input_model = MaterializeSkillAssetInput
    output_model = MaterializeSkillAssetOutput
    permissions = ToolPermissions(read_fs=True, write_fs=True)
    execution_category = ExecutionCategory.WRITE
    risk_level = RiskLevel.MEDIUM
    interrupt_behavior = InterruptBehavior.BLOCK
    timeout_seconds = 5.0
    idempotent = True
    concurrency_safe = False
    work_budget_cost = 50
    max_result_size_chars = 2000
    aci = ToolCard(
        when_to_use=(
            "Use after invoke_skill when the loaded skill references helper files "
            "under scripts/ or references/ that another tool must read or execute."
        ),
        when_not_to_use=(
            "Do not use for arbitrary project files or before invoking the skill."
        ),
        activation_group="resident",
        selection_tags=("skill", "files"),
        domains=("agent_internal", "files"),
    )

    def __init__(self, workspace: WorkspaceRuntime) -> None:
        self._workspace = workspace

    async def execute(
        self,
        input_data: BaseModel,
        context: ToolExecutionContext | None = None,
    ) -> BaseModel:
        inp = MaterializeSkillAssetInput.model_validate(input_data)
        if context is None or context.state is None:
            raise ValueError("materialize_skill_asset requires execution context")

        skill_state = context.state.get("skill_state")
        if not isinstance(skill_state, SkillState):
            skill_state = SkillState.model_validate(skill_state or {})
            context.state["skill_state"] = skill_state

        ref = skill_state.active.get(inp.skill_id)
        if ref is None:
            raise ValueError(f"Skill '{inp.skill_id}' is not active; invoke it first")

        relative = _validate_asset_path(inp.relative_path)
        source_root = Path(ref.root_dir).resolve()
        source = (source_root / relative).resolve()
        _ensure_within(source_root, source)
        if not source.is_file():
            raise FileNotFoundError(f"Skill asset not found: {inp.relative_path}")

        dest = (
            self._workspace.scratch
            / "skills"
            / _safe_skill_id(inp.skill_id)
            / relative
        )
        self._workspace.ensure_within_scratch(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

        data = dest.read_bytes()
        return MaterializeSkillAssetOutput(
            workspace_path=self._workspace.relative_to_root(dest).as_posix(),
            source_fingerprint=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )


def _validate_asset_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or raw_path.strip() in {"", "."} or ".." in path.parts:
        raise ValueError("Skill asset path must be relative and must not contain '..'")
    if path.parts[0] not in {"scripts", "references"}:
        raise ValueError("Skill assets must live under scripts/ or references/")
    if len(path.parts) < 2:
        raise ValueError("Skill asset path must include a file under scripts/ or references/")
    return path


def _ensure_within(root: Path, child: Path) -> None:
    root_s = str(root)
    child_s = str(child)
    if child != root and not child_s.startswith(root_s + os.sep):
        raise ValueError(f"Skill asset path escapes skill root: {child}")


def _safe_skill_id(skill_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", skill_id).strip("._") or "skill"
