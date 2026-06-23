"""Memory selector: choose relevant persistent memories for the current task.

Two-phase selection for cost/quality balance:
- Phase 1 (rule-based): Always include 'user' type memories.
- Phase 2 (LLM-based): For project/feedback/reference types, use LLM to
  select from MEMORY.md index entries when count exceeds threshold.

Falls back to rule-only selection if LLM call fails.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag.agent.memory.persistent.models import IndexEntry, MemoryFile
from rag.agent.memory.persistent.prompts import MEMORY_SELECT_PROMPT
from rag.agent.memory.persistent.store import PersistentMemoryStore
from rag.schema.llm import LLMCallStage

if TYPE_CHECKING:
    from rag.providers.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)


class MemorySelector:
    """Select relevant persistent memories for the current task."""

    def __init__(
        self,
        *,
        llm_gateway: LLMGateway | None = None,
        max_selected: int = 5,
        max_tokens: int = 4000,
        llm_threshold: int = 8,
        llm_stage: LLMCallStage = LLMCallStage.MEMORY_SELECT,
    ) -> None:
        self._gateway = llm_gateway
        self._max_selected = max_selected
        self._max_tokens = max_tokens
        self._llm_threshold = llm_threshold
        self._llm_stage = llm_stage

    async def select(
        self,
        *,
        task: str,
        index_content: str,
        store: PersistentMemoryStore,
    ) -> list[MemoryFile]:
        """Select relevant memories for the given task.

        Returns up to max_selected MemoryFile objects, with total
        content truncated to max_tokens.
        """
        if not index_content.strip():
            return []

        # Parse index entries
        entries = self._parse_index(index_content)
        if not entries:
            return []

        # Phase 1: rule-based — always include user-type memories
        # Read all memories to check types
        all_memories = store.read_all_memories()
        memory_by_name = {m.name: m for m in all_memories}

        # Always include user-type memories
        selected: list[MemoryFile] = []
        for memory in all_memories:
            if memory.memory_type == "user":
                selected.append(memory)

        # If total count is small, include everything
        non_user_memories = [m for m in all_memories if m.memory_type != "user"]
        if len(all_memories) <= self._llm_threshold:
            selected.extend(non_user_memories)
            return self._truncate_to_budget(selected)

        remaining_slots = self._max_selected - len(selected)
        if remaining_slots <= 0:
            return self._truncate_to_budget(selected)

        # Phase 2: LLM-based selection for non-user memories
        if self._gateway is None or not non_user_memories:
            # No gateway or no non-user memories — fall back to most recent
            non_user_memories.sort(
                key=lambda m: str(m.metadata.get("updated", "")),
                reverse=True,
            )
            selected.extend(non_user_memories[:remaining_slots])
            return self._truncate_to_budget(selected)

        # Build index text for LLM
        non_user_entries = [
            e for e in entries if e.name in {m.name for m in non_user_memories}
        ]
        index_text = "\n".join(
            f"- [{e.name}]({e.file}) — {e.description}" for e in non_user_entries
        )

        try:
            llm_selected_names = await self._llm_select(
                task=task,
                index_text=index_text,
                max_memories=min(remaining_slots, len(non_user_memories)),
            )
            for name in llm_selected_names:
                if name in memory_by_name and name not in {m.name for m in selected}:
                    selected.append(memory_by_name[name])
        except Exception:
            logger.warning("LLM memory selection failed, using rule fallback", exc_info=True)
            # Fall back to most recently updated
            non_user_memories.sort(
                key=lambda m: str(m.metadata.get("updated", "")),
                reverse=True,
            )
            selected.extend(non_user_memories[:remaining_slots])

        return self._truncate_to_budget(selected)

    async def _llm_select(
        self,
        *,
        task: str,
        index_text: str,
        max_memories: int,
    ) -> list[str]:
        """Call LLM to select relevant memory names."""
        prompt = MEMORY_SELECT_PROMPT.format(
            max_memories=max_memories,
            task=task,
            index_entries=index_text,
        )

        result = await self._gateway.agenerate_text(  # type: ignore[union-attr]
            stage=self._llm_stage,
            prompt=prompt,
        )

        return self._parse_selection(result.value)

    @staticmethod
    def _parse_selection(response: str) -> list[str]:
        """Parse memory names from LLM response."""
        names: list[str] = []
        for line in response.strip().split("\n"):
            name = line.strip().lstrip("- ").strip()
            if name and name.upper() != "NONE":
                names.append(name)
        return names

    @staticmethod
    def _parse_index(content: str) -> list[IndexEntry]:
        """Parse MEMORY.md index into entries."""
        entries: list[IndexEntry] = []
        for line in content.split("\n"):
            entry = IndexEntry.parse_line(line)
            if entry is not None:
                entries.append(entry)
        return entries

    def _truncate_to_budget(self, memories: list[MemoryFile]) -> list[MemoryFile]:
        """Truncate memory list to fit within token budget.

        Always includes at least the first memory (user-type priority)
        even if it exceeds the budget, to avoid returning an empty list.
        """
        if not memories:
            return []

        selected: list[MemoryFile] = []
        total_chars = 0
        for i, memory in enumerate(memories):
            memory_chars = len(memory.content)
            # Always include the first memory (typically user-type, always relevant)
            if i == 0:
                selected.append(memory)
                total_chars += memory_chars
                continue
            if total_chars + memory_chars > self._max_tokens * 3:
                break
            selected.append(memory)
            total_chars += memory_chars
        return selected[: self._max_selected]


__all__ = ["MemorySelector"]
