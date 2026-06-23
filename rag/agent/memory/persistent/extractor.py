"""Memory extractor: extract durable facts from completed conversations.

Runs as a post-processing step after the agent loop completes.
Uses the LLM to identify facts worth remembering across sessions.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rag.agent.memory.persistent.models import MemoryFile
from rag.agent.memory.persistent.prompts import (
    MEMORY_EXTRACT_PROMPT,
    MEMORY_MERGE_PROMPT,
    MEMORY_SIMILARITY_PROMPT,
)
from rag.agent.memory.persistent.store import PersistentMemoryStore
from rag.schema.llm import LLMCallStage

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from rag.agent.loop.state import LoopState
    from rag.providers.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)

# Minimum conversation length to trigger extraction
_MIN_MESSAGES_FOR_EXTRACTION = 4
# Maximum total transcript characters to avoid context overflow
_MAX_TRANSCRIPT_CHARS = 12_000
# Maximum messages to include in transcript
_MAX_MESSAGES_IN_TRANSCRIPT = 40


class MemoryExtractor:
    """Extract persistent memories from a completed run."""

    def __init__(
        self,
        *,
        llm_gateway: LLMGateway,
        llm_stage: LLMCallStage = LLMCallStage.MEMORY_EXTRACT,
        max_extracted: int = 3,
    ) -> None:
        self._gateway = llm_gateway
        self._llm_stage = llm_stage
        self._max_extracted = max_extracted

    async def extract(
        self,
        *,
        state: LoopState,
        store: PersistentMemoryStore,
    ) -> list[str]:
        """Extract memories from the completed run.

        Returns list of written memory names.
        Skips extraction if conversation is too short or trivial.
        """
        if not store.is_available:
            return []

        messages = state.get("messages", [])
        if len(messages) < _MIN_MESSAGES_FOR_EXTRACTION:
            return []

        # Format conversation transcript
        transcript = self._format_transcript(messages)

        # Get existing index for dedup
        existing_index = store.read_index()

        # Call LLM to extract
        prompt = MEMORY_EXTRACT_PROMPT.format(
            max_memories=self._max_extracted,
            transcript=transcript,
            existing_index=existing_index or "(no existing memories)",
        )

        try:
            result = await self._gateway.agenerate_text(
                stage=self._llm_stage,
                prompt=prompt,
            )
        except Exception:
            logger.warning("Memory extraction LLM call failed", exc_info=True)
            return []

        response = result.value
        if "NO_MEMORIES" in response.upper():
            return []

        # Parse extracted memories
        extracted = self._parse_extractions(response)
        if not extracted:
            return []

        # Write memories (with dedup check)
        written: list[str] = []
        run_id = state.get("run_config", {})
        run_id_str = getattr(run_id, "run_id", "unknown") if run_id else "unknown"

        for memory in extracted:
            # Check for similar existing memories
            existing = store.read_all_memories()
            similar = await self._find_similar(memory, existing)

            if similar is not None:
                # Merge with existing
                merged = await self._merge(similar, memory)
                if merged is not None:
                    merged.metadata["updated"] = datetime.now(UTC).isoformat()
                    if store.write_memory(merged):
                        written.append(merged.name)
                    else:
                        logger.warning("Failed to write merged memory: %s", merged.name)
            else:
                # Write new memory
                memory.metadata["created"] = datetime.now(UTC).isoformat()
                memory.metadata["updated"] = datetime.now(UTC).isoformat()
                memory.metadata["source_run_id"] = run_id_str
                if store.write_memory(memory):
                    written.append(memory.name)
                else:
                    logger.warning("Failed to write extracted memory: %s", memory.name)

        return written

    async def _find_similar(
        self,
        new_memory: MemoryFile,
        existing: list[MemoryFile],
    ) -> MemoryFile | None:
        """Find an existing memory that covers the same fact."""
        if not existing or not self._gateway:
            return None

        for existing_memory in existing:
            try:
                prompt = MEMORY_SIMILARITY_PROMPT.format(
                    name1=existing_memory.name,
                    content1=existing_memory.content[:500],
                    name2=new_memory.name,
                    content2=new_memory.content[:500],
                )
                result = await self._gateway.agenerate_text(
                    stage=self._llm_stage,
                    prompt=prompt,
                )
                if "YES" in result.value.upper():
                    return existing_memory
            except Exception:
                logger.warning(
                    "Similarity check failed for %s vs %s",
                    existing_memory.name,
                    new_memory.name,
                    exc_info=True,
                )
                continue

        return None

    async def _merge(
        self,
        existing: MemoryFile,
        new_memory: MemoryFile,
    ) -> MemoryFile | None:
        """Merge new information into an existing memory."""
        if not self._gateway:
            return None

        try:
            prompt = MEMORY_MERGE_PROMPT.format(
                name1=existing.name,
                content1=existing.content,
                name2=new_memory.name,
                content2=new_memory.content,
            )
            result = await self._gateway.agenerate_text(
                stage=self._llm_stage,
                prompt=prompt,
            )
            return MemoryFile.from_markdown(result.value)
        except Exception:
            logger.warning("Memory merge failed", exc_info=True)
            return None

    @staticmethod
    def _format_transcript(messages: list[BaseMessage]) -> str:
        """Format conversation messages into a transcript string.

        Caps total length to _MAX_TRANSCRIPT_CHARS and message count
        to _MAX_MESSAGES_IN_TRANSCRIPT to avoid context overflow.
        """
        # Take the most recent messages if too many
        recent = messages[-_MAX_MESSAGES_IN_TRANSCRIPT:]

        lines: list[str] = []
        total_chars = 0
        for msg in recent:
            role = getattr(msg, "type", msg.__class__.__name__)
            content = msg.content
            if isinstance(content, list):
                content = " ".join(str(p) for p in content)
            elif not isinstance(content, str):
                content = str(content)
            # Truncate very long messages
            if len(content) > 1000:
                content = content[:1000] + "..."
            line = f"[{role}]: {content}"
            total_chars += len(line)
            if total_chars > _MAX_TRANSCRIPT_CHARS:
                lines.append(f"[truncated: transcript exceeded {_MAX_TRANSCRIPT_CHARS} chars]")
                break
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _parse_extractions(response: str) -> list[MemoryFile]:
        """Parse extracted memories from LLM response."""
        memories: list[MemoryFile] = []
        blocks = re.split(r"---MEMORY---", response)

        for block in blocks:
            block = block.strip()
            if not block or block.startswith("---END---"):
                continue

            # Remove trailing ---END---
            block = re.sub(r"---END---.*$", "", block, flags=re.DOTALL).strip()
            if not block:
                continue

            try:
                memory = _parse_memory_block(block)
                if memory:
                    memories.append(memory)
            except Exception:
                logger.warning("Failed to parse memory block: %s", block[:100], exc_info=True)
                continue

        return memories


def _parse_memory_block(block: str) -> MemoryFile | None:
    """Parse a single memory block from the extraction response.

    Validates name format, type enum, and content length.
    """
    name = ""
    description = ""
    memory_type = "reference"
    content_lines: list[str] = []
    in_content = False

    for line in block.split("\n"):
        stripped = line.strip()

        if stripped.startswith("name:"):
            name = stripped[5:].strip()
        elif stripped.startswith("description:"):
            description = stripped[12:].strip()
        elif stripped.startswith("type:"):
            memory_type = stripped[5:].strip()
        elif stripped.startswith("content:"):
            in_content = True
            rest = stripped[8:].strip()
            if rest:
                content_lines.append(rest)
        elif in_content:
            content_lines.append(line)

    if not name:
        return None

    # Validate name: must be kebab-case alphanumeric
    import re

    if not re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", name):
        logger.warning("Invalid memory name from LLM: %r, sanitizing", name)
        name = re.sub(r"[^a-z0-9_-]", "-", name.lower()).strip("-")[:64]
        if not name:
            return None

    # Validate type: must be one of the known types
    valid_types = {"user", "feedback", "project", "reference"}
    if memory_type not in valid_types:
        logger.warning("Invalid memory type from LLM: %r, defaulting to 'reference'", memory_type)
        memory_type = "reference"

    # Validate content length
    content = "\n".join(content_lines).strip()
    if not content:
        content = description  # fallback
    if len(content) > 10_000:
        logger.warning("Memory content too long (%d chars), truncating", len(content))
        content = content[:10_000] + "\n[truncated]"

    return MemoryFile(
        name=name,
        description=description or name,
        memory_type=memory_type,
        content=content,
    )


__all__ = ["MemoryExtractor"]
