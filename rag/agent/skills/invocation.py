"""invoke_skill — resident core tool for skill invocation.

This module defines the ToolSpec and runner for invoke_skill.  The runner
is a factory: it takes a SkillCatalog (built at runtime assembly) and
returns a ContextualToolRunner that the ToolRegistry can call.

Phase 1: inline only.  The model reads the skill content from the tool
result and follows the instructions in the next turn.  No forked
sub-agent is created.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.skills.context import render_loaded_skill
from rag.agent.skills.loader import SkillLoadError
from rag.agent.skills.models import SkillInvocation, SkillState
from rag.agent.tools.card import ToolCard
from rag.agent.tools.spec import (
    ExecutionCategory,
    InterruptBehavior,
    RiskLevel,
    ToolError,
    ToolPermissions,
    ToolSpec,
)

if TYPE_CHECKING:
    from rag.agent.skills.catalog import SkillCatalog
    from rag.agent.skills.policy import SkillPolicy
    from rag.agent.tools.registry import ContextualToolRunner, ToolExecutionContext

logger = logging.getLogger(__name__)

# ── I/O schemas ──────────────────────────────────────────────────────


class InvokeSkillInput(BaseModel):
    """Input for the invoke_skill tool."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        min_length=1,
        description="The skill name. Must match a name in <available_skills>.",
    )
    args: str | None = Field(
        default=None,
        description="Optional arguments to pass to the skill ($ARGUMENTS).",
    )


class InvokeSkillOutput(BaseModel):
    """Output from invoke_skill — includes the loaded skill content.

    The ``loaded_content`` field carries the rendered skill body.  The
    model reads this directly from the tool result and follows the
    instructions in its next turn.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(description="Whether the skill was loaded successfully.")
    name: str = Field(description="The skill name that was invoked.")
    skill_id: str = Field(default="", description="The resolved unique skill id.")
    source: str = Field(default="", description="SkillSource value.")
    fingerprint: str = Field(
        default="",
        description="First 16 chars of the content fingerprint.",
    )
    error_code: str = Field(default="", description="Machine-readable error code.")
    loaded_content: str = Field(
        default="",
        description="Rendered skill body with base directory prefix. "
                    "The model reads this and follows the instructions.",
    )


# ── ToolSpec ─────────────────────────────────────────────────────────

INVOKE_SKILL_SPEC = ToolSpec(
    name="invoke_skill",
    description=(
        "Load a skill's full instructions into the conversation. "
        "Skills are reusable workflows that guide how to perform a task. "
        "Call this BEFORE following a matching skill's workflow."
    ),
    input_model=InvokeSkillInput,
    output_model=InvokeSkillOutput,
    error_model=ToolError,
    permissions=ToolPermissions(),
    execution_category=ExecutionCategory.READ,
    risk_level=RiskLevel.LOW,
    interrupt_behavior=InterruptBehavior.CANCEL,
    timeout_seconds=5.0,
    idempotent=True,
    concurrency_safe=True,
    work_budget_cost=50,
    max_result_size_chars=50000,
    aci=ToolCard(
        when_to_use=(
            "Call when a skill listed in <available_skills> matches the "
            "user's request. This is a BLOCKING REQUIREMENT — invoke the "
            "skill BEFORE answering or following the workflow."
        ),
        when_not_to_use=(
            "Do not call for skills that are already loaded. "
            "Do not guess skill names — only use names from the listing."
        ),
        activation_group="resident",
        selection_tags=("skill", "workflow"),
        domains=("agent_internal",),
    ),
)


# ── Runner factory ───────────────────────────────────────────────────


def make_invoke_skill_runner(
    catalog: SkillCatalog,
    policy: SkillPolicy | None = None,
) -> ContextualToolRunner:
    """Create a ContextualToolRunner that resolves skills from *catalog*.

    The returned runner has the signature::

        async def runner(input_data: BaseModel, context: ToolExecutionContext) -> BaseModel

    It validates the skill name, loads the full body from disk, renders
    it with $ARGUMENTS substitution, and returns an InvokeSkillOutput.
    """

    async def _run(input_data: BaseModel, context: ToolExecutionContext | None = None) -> BaseModel:
        inp: InvokeSkillInput = input_data  # type: ignore[assignment]
        name = inp.name.strip()
        skill = catalog.find(name)

        if skill is None:
            candidates = catalog.candidates_for(name)
            if candidates:
                available = ", ".join(m.skill_id for m in candidates)
                return InvokeSkillOutput(
                    success=False,
                    name=name,
                    error_code="ambiguous_skill_name",
                    loaded_content=(
                        f"Skill name '{name}' is ambiguous. Use one of: {available}"
                    ),
                )
            available = ", ".join(m.skill_id for m in catalog.list_all())
            return InvokeSkillOutput(
                success=False,
                name=name,
                error_code="skill_not_found",
                loaded_content=(
                    f"Skill '{name}' not found. Available skills: {available}"
                ),
            )

        if skill.disable_model_invocation:
            return InvokeSkillOutput(
                success=False,
                name=name,
                skill_id=skill.skill_id,
                source=skill.source.value,
                error_code="skill_disabled",
                loaded_content=(
                    f"Skill '{name}' has disable_model_invocation set and "
                    f"cannot be invoked autonomously by the model."
                ),
            )

        if policy is not None and not policy.is_skill_enabled(skill):
            return InvokeSkillOutput(
                success=False,
                name=name,
                skill_id=skill.skill_id,
                source=skill.source.value,
                error_code="skill_disabled",
                loaded_content=(
                    f"Skill '{name}' is disabled by the current policy."
                ),
            )

        try:
            loaded = catalog.load(skill.skill_id)
            if loaded is None:
                return InvokeSkillOutput(
                    success=False,
                    name=name,
                    skill_id=skill.skill_id,
                    source=skill.source.value,
                    error_code="skill_not_found",
                    loaded_content=f"Failed to load skill '{name}' from disk.",
                )
            rendered = render_loaded_skill(loaded, args=inp.args)
        except SkillLoadError as exc:
            logger.warning(
                "invoke_skill: invalid skill manifest for '%s'",
                skill.skill_id,
                exc_info=True,
            )
            return InvokeSkillOutput(
                success=False,
                name=name,
                skill_id=skill.skill_id,
                source=skill.source.value,
                error_code="invalid_skill_manifest",
                loaded_content=f"Failed to load skill '{name}': {exc}",
            )
        except Exception as exc:
            logger.exception("invoke_skill: failed to load '%s'", skill.skill_id)
            return InvokeSkillOutput(
                success=False,
                name=name,
                skill_id=skill.skill_id,
                source=skill.source.value,
                error_code="skill_load_failed",
                loaded_content=f"Failed to load skill '{name}': {exc}",
            )

        # ── Persist to LoopState so the skill survives across turns ──
        if context is not None and context.state is not None:
            skill_state = context.state.get("skill_state")
            if skill_state is not None:
                if not hasattr(skill_state, "active"):
                    skill_state = SkillState.model_validate(skill_state)
                    context.state["skill_state"] = skill_state
                skill_state.active[skill.skill_id] = loaded.to_ref(args=inp.args)
                invocation = SkillInvocation(
                    name=skill.name,
                    skill_id=skill.skill_id,
                    source=skill.source.value,
                    skill_file=str(skill.skill_file),
                    fingerprint=skill.content_fingerprint,
                    invoked_at_iteration=context.state.get("iteration", 0),
                    args=inp.args,
                )
                skill_state.invoked = skill_state.invoked + (invocation,)

        logger.info(
            "invoke_skill: loaded '%s' (%d chars, fingerprint=%s)",
            name, len(rendered), skill.content_fingerprint[:16],
        )

        return InvokeSkillOutput(
            success=True,
            name=skill.name,
            skill_id=skill.skill_id,
            source=skill.source.value,
            fingerprint=skill.content_fingerprint[:16],
            loaded_content=rendered,
        )

    return _run
