"""Shared ContextVars for tool discovery.

These are set by AgentLoop / AgentService and read by tool runners,
avoiding circular imports between service.py and runtime.py.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.agent.capabilities.catalog import DeferredToolStore

# Per-run DeferredToolStore — set before each loop.run(), read by runners.
deferred_store_var: contextvars.ContextVar[DeferredToolStore] = contextvars.ContextVar(
    "deferred_store",
)

# Per-turn iteration — set by AgentLoop before each tool execution.
iteration_var: contextvars.ContextVar[int] = contextvars.ContextVar(
    "iteration",
    default=0,
)
