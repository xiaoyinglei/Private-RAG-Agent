"""tool_search and activate_tools — model-driven tool discovery.

tool_search: search the catalog for deferred tools matching a query.
             Returns candidates only — does NOT activate anything.
activate_tools: explicitly activate tools from the last search results.
                Only tools returned by the most recent tool_search can be activated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from rag.agent.capabilities.catalog import (
        DeferredToolStore,
        ToolCatalog,
    )


# ── tool_search I/O ──


class ToolSearchInput(BaseModel):
    """Input for the tool_search tool."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Natural language description of the capability you need. "
            "Examples: 'analyze spreadsheet', 'search documents', 'generate charts'"
        ),
    )
    max_results: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of candidate tools to return.",
    )


class ToolCandidate(BaseModel):
    """A candidate tool returned by tool_search."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    reason: str


class ToolSearchOutput(BaseModel):
    """Output from the tool_search tool."""

    model_config = ConfigDict(frozen=True)

    candidates: tuple[ToolCandidate, ...]
    query: str
    message: str = (
        "These are candidate tools. Call activate_tools with the "
        "names you need to make them available on the next turn."
    )


# ── activate_tools I/O ──


class ActivateToolsInput(BaseModel):
    """Input for the activate_tools tool."""

    model_config = ConfigDict(frozen=True)

    names: list[str] = Field(
        min_length=1,
        max_length=20,
        description=(
            "Tool names to activate. Must be from the most recent "
            "tool_search results. Example: ['excel_analyze', 'csv_reader']"
        ),
    )


class ActivateToolsOutput(BaseModel):
    """Output from the activate_tools tool."""

    model_config = ConfigDict(frozen=True)

    activated: tuple[str, ...] = ()
    already_active: tuple[str, ...] = ()
    not_in_candidates: tuple[str, ...] = ()
    denied: tuple[str, ...] = ()
    message: str = (
        "Activated tools will be available on the next model turn."
    )


# ── tool_search execution ──


def execute_tool_search(
    query: str,
    *,
    catalog: ToolCatalog,
    store: DeferredToolStore,
    max_results: int = 8,
) -> ToolSearchOutput:
    """Search the catalog for deferred tools.

    Stores candidates in the deferred store for subsequent activate_tools.
    Does NOT activate anything — the LLM decides.
    """
    candidates = catalog.search(query, max_results=max_results)
    store.set_pending_candidates(query, candidates)

    return ToolSearchOutput(
        candidates=tuple(
            ToolCandidate(name=c.name, description=c.description, reason=c.reason)
            for c in candidates
        ),
        query=query,
    )


# ── activate_tools execution ──


def execute_activate_tools(
    names: list[str],
    *,
    catalog: ToolCatalog,
    store: DeferredToolStore,
    allowed_tools: list[str],
    deny_tools: frozenset[str],
    iteration: int,
) -> ActivateToolsOutput:
    """Activate tools explicitly chosen by the model.

    Validation (per tool name):
    1. Must be in last_candidates (from most recent tool_search)
    2. Must be category == "deferred"
    3. Must be in allowed_tools
    4. Must not be in deny_tools
    5. Must exist in the catalog
    6. Activation count must not exceed max_active
    """
    activated: list[str] = []
    already_active: list[str] = []
    not_in_candidates: list[str] = []
    denied: list[str] = []

    for name in names:
        # Already active?
        if store.is_active(name):
            already_active.append(name)
            continue

        # In last search candidates?
        if not store.is_pending(name):
            not_in_candidates.append(name)
            continue

        # Category check
        category = catalog.classify(name)
        if category != "deferred":
            not_in_candidates.append(name)
            continue

        # Allowed tools check
        if name not in allowed_tools:
            denied.append(name)
            continue

        # Deny tools check
        if name in deny_tools:
            denied.append(name)
            continue

        # Max active check
        if len(store.active_names()) >= store.max_active:
            denied.append(name)
            continue

        store.activate(name, iteration=iteration)
        activated.append(name)

    return ActivateToolsOutput(
        activated=tuple(activated),
        already_active=tuple(already_active),
        not_in_candidates=tuple(not_in_candidates),
        denied=tuple(denied),
    )
