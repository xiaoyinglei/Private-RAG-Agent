"""Persistent cross-session memory system for the RAG Agent.

Four components:
- PersistentMemoryStore: file-backed CRUD for memory files
- MemorySelector: choose relevant memories for the current task
- MemoryExtractor: extract durable facts from completed conversations
- MemoryConsolidator: merge/deduplicate memories when count exceeds threshold
"""

from rag.agent.memory.persistent.consolidator import MemoryConsolidator
from rag.agent.memory.persistent.extractor import MemoryExtractor
from rag.agent.memory.persistent.models import (
    ConsolidationResult,
    IndexEntry,
    MemoryFile,
    MemoryFileMeta,
)
from rag.agent.memory.persistent.selector import MemorySelector
from rag.agent.memory.persistent.store import PersistentMemoryStore

__all__ = [
    "ConsolidationResult",
    "IndexEntry",
    "MemoryConsolidator",
    "MemoryExtractor",
    "MemoryFile",
    "MemoryFileMeta",
    "MemorySelector",
    "PersistentMemoryStore",
]
