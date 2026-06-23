"""File-backed persistent memory store.

Memories live under <workspace>/.agent_memory/persistent/ as Markdown files
with YAML frontmatter. An index file (MEMORY.md) provides a cheap always-loaded
summary for the selector.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from rag.agent.memory.persistent.models import (
    MemoryFile,
    MemoryFileMeta,
)
from rag.agent.workspace import WorkspaceRuntime

logger = logging.getLogger(__name__)

PERSISTENT_DIR = "persistent"
INDEX_FILE = "MEMORY.md"


class PersistentMemoryStore:
    """CRUD operations for persistent memory files on disk.

    No LLM calls — pure file I/O. Designed to be cheap and safe.
    """

    def __init__(self, workspace: WorkspaceRuntime) -> None:
        self._workspace = workspace
        self._root = workspace.root / ".agent_memory" / PERSISTENT_DIR

    @property
    def root(self) -> Path:
        return self._root

    @property
    def is_available(self) -> bool:
        """Persistent memory is only available for non-temporary workspaces."""
        return not self._workspace.is_temporary

    # ── Index operations ──

    def read_index(self) -> str:
        """Read MEMORY.md content. Returns '' if absent or unavailable."""
        if not self.is_available:
            return ""
        index_path = self._root / INDEX_FILE
        if not index_path.is_file():
            return ""
        try:
            return index_path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Failed to read memory index: %s", index_path)
            return ""

    def write_index(self, content: str) -> bool:
        """Write MEMORY.md index file. Returns True on success."""
        if not self.is_available:
            return False
        self._ensure_dir()
        index_path = self._root / INDEX_FILE
        try:
            index_path.write_text(content, encoding="utf-8")
            return True
        except OSError:
            logger.warning("Failed to write memory index: %s", index_path)
            return False

    def build_index(self, memories: list[MemoryFile]) -> str:
        """Build MEMORY.md content from a list of memory files."""
        lines = [m.index_line() for m in memories]
        return "\n".join(lines) + "\n" if lines else ""

    def rebuild_index(self) -> str:
        """Rebuild MEMORY.md from all existing memory files."""
        memories = self.read_all_memories()
        index_content = self.build_index(memories)
        self.write_index(index_content)
        return index_content

    # ── Memory CRUD ──

    def list_memories(self) -> list[MemoryFileMeta]:
        """List all memory files with metadata (no content loaded)."""
        if not self.is_available or not self._root.is_dir():
            return []

        result: list[MemoryFileMeta] = []
        for path in sorted(self._root.glob("*.md")):
            if path.name == INDEX_FILE:
                continue
            try:
                text = path.read_text(encoding="utf-8")
                memory = MemoryFile.from_markdown(text)
                result.append(
                    MemoryFileMeta(
                        name=memory.name,
                        description=memory.description,
                        memory_type=memory.memory_type,
                        created=str(memory.metadata.get("created", "")),
                        updated=str(memory.metadata.get("updated", "")),
                        path=path.name,
                    )
                )
            except Exception:
                logger.warning("Failed to read memory file: %s", path, exc_info=True)
                continue
        return result

    def read_memory(self, name: str) -> MemoryFile | None:
        """Read a single memory file by name. Returns None if absent."""
        if not self.is_available:
            return None
        path = self._memory_path(name)
        if not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8")
            return MemoryFile.from_markdown(text)
        except OSError:
            logger.warning("Failed to read memory: %s", path)
            return None

    def write_memory(self, memory: MemoryFile) -> bool:
        """Write/overwrite a memory file and update the index.

        Returns True on success, False on failure.
        Callers (e.g. consolidator) should check the return value
        before deleting other memories.
        """
        if not self.is_available:
            return False
        self._ensure_dir()
        path = self._memory_path(memory.name)
        try:
            path.write_text(memory.to_markdown(), encoding="utf-8")
        except OSError:
            logger.warning("Failed to write memory: %s", path)
            return False
        self.rebuild_index()
        return True

    def delete_memory(self, name: str) -> bool:
        """Delete a memory file. Returns True if it existed and was deleted."""
        if not self.is_available:
            return False
        path = self._memory_path(name)
        if not path.is_file():
            return False
        try:
            path.unlink()
        except OSError:
            logger.warning("Failed to delete memory: %s", path)
            return False
        self.rebuild_index()
        return True

    # ── Bulk operations ──

    def read_all_memories(self) -> list[MemoryFile]:
        """Read all memory files (excluding index)."""
        if not self.is_available or not self._root.is_dir():
            return []

        result: list[MemoryFile] = []
        for path in sorted(self._root.glob("*.md")):
            if path.name == INDEX_FILE:
                continue
            try:
                text = path.read_text(encoding="utf-8")
                result.append(MemoryFile.from_markdown(text))
            except Exception:
                logger.warning("Failed to read memory: %s", path, exc_info=True)
                continue
        return result

    def memory_count(self) -> int:
        """Count of memory files (excluding index)."""
        if not self.is_available or not self._root.is_dir():
            return 0
        return sum(1 for p in self._root.glob("*.md") if p.name != INDEX_FILE)

    # ── Internal helpers ──

    def _memory_path(self, name: str) -> Path:
        """Get the file path for a memory name."""
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        return self._root / f"{safe_name}.md"

    def _ensure_dir(self) -> None:
        """Ensure the persistent directory exists."""
        self._root.mkdir(parents=True, exist_ok=True)


__all__ = [
    "INDEX_FILE",
    "PERSISTENT_DIR",
    "PersistentMemoryStore",
]
