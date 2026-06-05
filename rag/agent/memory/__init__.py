"""Working-memory compaction and context assembly for agent runs."""

from rag.agent.memory.compactor import RunMessageCompactor, WorkingMemoryDehydrator
from rag.agent.memory.injector import ContextBuilder, ContextInjector
from rag.agent.memory.models import (
    ContextBudgetSnapshot,
    ContextSection,
    EvictedStateItem,
    ExternalizedToolOutput,
    ExtractedFact,
    InjectedContext,
    MemoryBudgetSnapshot,
    MemoryPolicy,
    MemoryRecord,
    MemoryRef,
    MessageBatchPayload,
    StateChannelReplacement,
    ToolErrorDetailPayload,
    WorkingMemoryDehydration,
    WorkingSummary,
)
from rag.agent.memory.store import MemoryRefError, WorkspaceMemoryStore

__all__ = [
    "ContextBudgetSnapshot",
    "ContextBuilder",
    "ContextInjector",
    "ContextSection",
    "EvictedStateItem",
    "ExtractedFact",
    "ExternalizedToolOutput",
    "InjectedContext",
    "MessageBatchPayload",
    "MemoryBudgetSnapshot",
    "MemoryPolicy",
    "MemoryRecord",
    "MemoryRef",
    "MemoryRefError",
    "RunMessageCompactor",
    "StateChannelReplacement",
    "ToolErrorDetailPayload",
    "WorkspaceMemoryStore",
    "WorkingMemoryDehydration",
    "WorkingMemoryDehydrator",
    "WorkingSummary",
]
