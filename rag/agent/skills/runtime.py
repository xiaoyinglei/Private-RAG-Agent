"""Runtime bridge for skills and model prompt assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.agent.skills.catalog import SkillCatalog
from rag.agent.skills.context import build_skills_prompt_section
from rag.agent.skills.models import SkillState

if TYPE_CHECKING:
    from rag.agent.loop.state import LoopState


class SkillRuntime:
    """Service-scoped runtime facade for skill prompt context."""

    def __init__(
        self,
        catalog: SkillCatalog,
        *,
        max_listing_chars: int = 2000,
    ) -> None:
        self.catalog = catalog
        self.max_listing_chars = max_listing_chars

    def render_prompt_context(self, state: LoopState) -> str:
        skill_state = state.get("skill_state")
        if not isinstance(skill_state, SkillState):
            skill_state = SkillState.model_validate(skill_state or {})
            state["skill_state"] = skill_state
        return build_skills_prompt_section(
            self.catalog,
            max_listing_chars=self.max_listing_chars,
            skill_state=skill_state,
        )


__all__ = ["SkillRuntime"]
