"""Memory consolidator: merge and deduplicate persistent memories.

Triggered when memory count >= consolidation_threshold.
Uses LLM to decide: KEEP, MERGE, or DELETE for each memory.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rag.agent.memory.persistent.models import ConsolidationResult, MemoryFile
from rag.agent.memory.persistent.prompts import (
    MEMORY_CONSOLIDATE_PROMPT,
    MEMORY_MERGE_PROMPT,
)
from rag.agent.memory.persistent.store import PersistentMemoryStore
from rag.schema.llm import LLMCallStage

if TYPE_CHECKING:
    from rag.providers.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)

# Maximum memories to include in a single consolidation prompt
_MAX_MEMORIES_IN_PROMPT = 20


class MemoryConsolidator:
    """Merge and deduplicate persistent memories."""

    def __init__(
        self,
        *,
        llm_gateway: LLMGateway,
        consolidation_threshold: int = 10,
        llm_stage: LLMCallStage = LLMCallStage.MEMORY_CONSOLIDATE,
    ) -> None:
        self._gateway = llm_gateway
        self._threshold = consolidation_threshold
        self._llm_stage = llm_stage

    async def consolidate(
        self,
        store: PersistentMemoryStore,
    ) -> ConsolidationResult:
        """Merge/deduplicate memories if threshold is met.

        Returns ConsolidationResult with before/after counts.
        Skips if below threshold.
        """
        if not store.is_available:
            return ConsolidationResult(action="skipped")

        memories = store.read_all_memories()
        before_count = len(memories)

        if before_count < self._threshold:
            return ConsolidationResult(
                action="skipped",
                before_count=before_count,
                after_count=before_count,
            )

        # Cap memories in prompt to avoid context overflow
        prompt_memories = memories[:_MAX_MEMORIES_IN_PROMPT]
        memories_text = "\n\n---\n\n".join(
            f"## {m.name} (type={m.memory_type})\n{m.content}" for m in prompt_memories
        )

        prompt = MEMORY_CONSOLIDATE_PROMPT.format(
            count=before_count,
            memories_text=memories_text,
        )

        try:
            result = await self._gateway.agenerate_text(
                stage=self._llm_stage,
                prompt=prompt,
            )
        except Exception:
            logger.warning("Consolidation LLM call failed", exc_info=True)
            return ConsolidationResult(
                action="skipped",
                before_count=before_count,
                after_count=before_count,
            )

        # Parse decisions
        decisions = self._parse_decisions(result.value)
        if not decisions:
            return ConsolidationResult(
                action="skipped",
                before_count=before_count,
                after_count=before_count,
            )

        # Execute decisions
        memory_by_name = {m.name: m for m in memories}
        merged_names: list[str] = []
        deleted_names: list[str] = []
        kept_memories: list[MemoryFile] = []

        for decision in decisions:
            action = decision.get("action", "").upper()
            name = decision.get("name", "")

            if action == "KEEP":
                if name in memory_by_name:
                    kept_memories.append(memory_by_name[name])
            elif action == "MERGE":
                merge_with = decision.get("merge_with", "")
                if name in memory_by_name and merge_with in memory_by_name:
                    merged = await self._merge_memories(
                        memory_by_name[name],
                        memory_by_name[merge_with],
                    )
                    if merged:
                        merged.metadata["updated"] = datetime.now(UTC).isoformat()
                        kept_memories.append(merged)
                        merged_names.append(name)
                        merged_names.append(merge_with)
                    else:
                        kept_memories.append(memory_by_name[name])
                        kept_memories.append(memory_by_name[merge_with])
                elif name in memory_by_name:
                    kept_memories.append(memory_by_name[name])
            elif action == "DELETE":
                deleted_names.append(name)

        # Track all names accounted for in decisions (both merge source and target)
        mentioned: set[str] = set()
        for decision in decisions:
            mentioned.add(decision.get("name", ""))
            if decision.get("merge_with"):
                mentioned.add(decision["merge_with"])

        # Add any memories not mentioned in decisions (keep them)
        for memory in memories:
            if memory.name not in mentioned:
                kept_memories.append(memory)

        # Deduplicate kept memories by name
        seen_names: set[str] = set()
        final_memories: list[MemoryFile] = []
        for memory in kept_memories:
            if memory.name not in seen_names:
                seen_names.add(memory.name)
                final_memories.append(memory)

        # Safe write: write all final memories first, check each write succeeds,
        # only delete old memories after ALL writes confirm success.
        final_names = {m.name for m in final_memories}
        writes_ok = True
        for memory in final_memories:
            if not store.write_memory(memory):
                writes_ok = False
                logger.error("Failed to write memory during consolidation: %s", memory.name)

        if not writes_ok:
            # Some writes failed — do NOT delete anything. The old memories
            # are still on disk (writes are overwrites), so no data is lost.
            logger.error("Consolidation aborted: write failures detected, no memories deleted")
            return ConsolidationResult(
                action="consolidated",
                before_count=before_count,
                after_count=before_count,
                merged=[],
                deleted=[],
            )

        # All writes succeeded — now safe to delete merged/deleted memories
        for memory in memories:
            if memory.name not in final_names:
                store.delete_memory(memory.name)

        store.rebuild_index()

        return ConsolidationResult(
            action="consolidated",
            before_count=before_count,
            after_count=len(final_memories),
            merged=list(set(merged_names)),
            deleted=list(set(deleted_names)),
        )

    async def _merge_memories(
        self,
        memory1: MemoryFile,
        memory2: MemoryFile,
    ) -> MemoryFile | None:
        """Merge two memories using LLM."""
        try:
            prompt = MEMORY_MERGE_PROMPT.format(
                name1=memory1.name,
                content1=memory1.content,
                name2=memory2.name,
                content2=memory2.content,
            )
            result = await self._gateway.agenerate_text(
                stage=self._llm_stage,
                prompt=prompt,
            )
            return MemoryFile.from_markdown(result.value)
        except Exception:
            logger.warning(
                "Memory merge failed: %s + %s",
                memory1.name,
                memory2.name,
                exc_info=True,
            )
            return None

    @staticmethod
    def _parse_decisions(response: str) -> list[dict[str, str]]:
        """Parse consolidation decisions from LLM response."""
        decisions: list[dict[str, str]] = []
        blocks = re.split(r"---DECISION---", response)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            decision: dict[str, str] = {}
            for line in block.split("\n"):
                stripped = line.strip()
                if stripped.startswith("action:"):
                    decision["action"] = stripped[7:].strip()
                elif stripped.startswith("name:"):
                    decision["name"] = stripped[5:].strip()
                elif stripped.startswith("merge_with:"):
                    decision["merge_with"] = stripped[11:].strip()
                elif stripped.startswith("reason:"):
                    decision["reason"] = stripped[7:].strip()

            if decision.get("action") and decision.get("name"):
                decisions.append(decision)

        return decisions


__all__ = ["MemoryConsolidator"]
