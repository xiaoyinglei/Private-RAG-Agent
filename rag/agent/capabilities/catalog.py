"""Provider-Agnostic Tool Discovery — catalog and deferred store.

ToolCatalog: searchable index of deferred tools (BM25).
DeferredToolStore: per-run activation state (active tools + last candidates).
VisibleToolResolver: determines which tools are visible each turn.
The LLM decides what to discover and activate; the Runtime never guesses.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

from rank_bm25 import BM25L  # type: ignore[import-untyped]

from rag.agent.tools.spec import ToolSpec

# ── Tool categories ──

ToolCategory = Literal["core", "deferred", "internal"]

CORE_TOOLS: frozenset[str] = frozenset({
    "tool_search",
    "activate_tools",
    "task",
    "list_files",
    "read_file",
    "write_file",
    "run_python",
    "search_text",
    "apply_patch",
    "run_command",
    "update_plan",
    "tool_repl",
    "invoke_skill",
    "materialize_skill_asset",
})

DEFERRED_TOOLS: frozenset[str] = frozenset({
    # Semantic RAG tools (agent-visible, on-demand activation)
    "search_knowledge",
    "search_assets",
    # LLM sub-agent tools
    "llm_generate",
    "llm_summarize",
    "llm_compare",
    # Primitive / workspace
    "structured_probe",
})

# Internal building blocks — used by semantic tool runners, NOT visible to LLM.
# These are the old fine-grained pipeline tools that search_knowledge/search_assets
# orchestrate internally.  Kept registered so runners can call them, but the agent
# should never see or activate them directly.
INTERNAL_TOOLS: frozenset[str] = frozenset({
    # RAG retrieval pipeline
    "vector_search",
    "keyword_search",
    "grounding",
    "rerank",
    "graph_expand",
    "rag_search_answer",
    # Asset pipeline
    "asset_list",
    "asset_inspect",
    "asset_read_slice",
    "asset_analyze",
})

# ── Activation groups (PR7: supplementary metadata, NOT a visibility switch) ──
# These describe which functional domain a tool belongs to.  They are used for:
#   1. Enriching tool_search results with grouping information
#   2. Supporting batch activation by group (e.g. activate all "rag" tools)
# They do NOT change resolve_visible_tools() — category remains the visibility
# master switch (core/deferred/internal).
#
# If a ToolCard on the ToolSpec specifies an activation_group, that takes
# precedence.  Otherwise this mapping is the fallback.

_DEFAULT_ACTIVATION_GROUPS: dict[str, str] = {
    # resident: always visible, no search/activation needed
    "tool_search": "resident",
    "activate_tools": "resident",
    "task": "resident",
    "list_files": "resident",
    "read_file": "resident",
    "update_plan": "resident",
    "invoke_skill": "resident",
    "materialize_skill_asset": "resident",

    # rag: semantic knowledge and asset retrieval (activated on demand)
    "search_knowledge": "rag",
    "search_assets": "rag",

    # code: LLM sub-agent tools
    "llm_generate": "code",
    "llm_summarize": "code",
    "llm_compare": "code",

    # workspace: file operations and code execution (activated on demand)
    "write_file": "workspace",
    "run_python": "workspace",
    "search_text": "workspace",
    "apply_patch": "workspace",
    "run_command": "workspace",
    "structured_probe": "workspace",
    "tool_repl": "workspace",

    # mcp: external tools from MCP servers (always deferred, never resident)
    # Individual MCP tools are not listed here — they get activation_group="mcp"
    # from their ToolCard.  This entry is for tools that don't have a ToolCard
    # but whose name starts with "mcp__".
}


# ── Tokenizer (ASCII + CJK single-char) ──

_TOKEN_RE = re.compile(
    r"[一-鿿぀-ゟ゠-ヿ가-힯]|[a-z0-9]+"
)


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenize.

    ASCII words are matched as whole tokens (splitting on non-alnum).
    CJK characters (Chinese, Japanese kana, Korean hangul) are matched
    individually (single-char tokens).
    """
    normalized = text.lower().replace("_", " ")
    return _TOKEN_RE.findall(normalized)


# ── Schema flattener ──


def flatten_schema(schema: dict[str, Any]) -> str:
    """Recursively flatten a JSON Schema into searchable plain text.

    Extracts: title, description, type, property names, property
    descriptions, enum values, required fields, additionalProperties.
    Does NOT include output_schema — only input parameters are indexed.

    Example:
        {"properties": {"file_path": {"type": "string", "description": "Path to xlsx"}}}
        → "file_path string Path to xlsx"
    """
    parts: list[str] = []

    def _walk(s: dict[str, Any], *, depth: int = 0) -> None:
        if depth > 8 or not isinstance(s, dict):
            return

        if "title" in s:
            parts.append(str(s["title"]))
        if "description" in s:
            parts.append(str(s["description"]))
        if "type" in s:
            parts.append(str(s["type"]))

        for val in s.get("enum", []):
            parts.append(str(val))

        properties = s.get("properties", {})
        required = set(s.get("required", []))
        for prop_name, prop_schema in properties.items():
            parts.append(prop_name)
            if prop_name in required:
                parts.append("required")
            if isinstance(prop_schema, dict):
                if "type" in prop_schema:
                    parts.append(str(prop_schema["type"]))
                if "description" in prop_schema:
                    parts.append(str(prop_schema["description"]))
                for val in prop_schema.get("enum", []):
                    parts.append(str(val))
                if prop_schema.get("type") == "object":
                    _walk(prop_schema, depth=depth + 1)
                if "items" in prop_schema:
                    _walk(prop_schema["items"], depth=depth + 1)

        # additionalProperties
        ap = s.get("additionalProperties")
        if isinstance(ap, dict):
            _walk(ap, depth=depth + 1)

        for key in ("anyOf", "oneOf", "allOf"):
            for variant in s.get(key, []):
                if isinstance(variant, dict):
                    _walk(variant, depth=depth + 1)

    _walk(schema)
    return " ".join(parts)


# ── Catalog entry ──


@dataclass(frozen=True)
class ToolCatalogEntry:
    """Metadata for a single tool in the catalog."""
    name: str
    description: str
    category: ToolCategory
    search_text: str  # pre-computed: name + description + flattened schema + examples + tags
    tags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    schema_text: str = ""
    source: str = "builtin"

    # ── ToolCard-derived fields (PR5: enrich search and candidate display) ──
    activation_group: str = ""
    when_to_use: str = ""
    when_not_to_use: str = ""
    domains: tuple[str, ...] = ()
    file_types: tuple[str, ...] = ()
    failure_codes: tuple[str, ...] = ()
    selection_tags: tuple[str, ...] = ()


# ── Search candidate ──


@dataclass(frozen=True)
class SearchCandidate:
    """A tool returned by catalog search.  Not yet activated."""
    name: str
    description: str
    reason: str

    # ── ToolCard summary fields (PR5) ──
    when_to_use: str = ""
    activation_group: str = ""
    tags: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()


# ── Tool Catalog with BM25 ──


class ToolCatalog:
    """Searchable index of deferred tools.

    Built once at service startup from ToolRegistry specs.
    Only deferred tools are indexed.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ToolCatalogEntry] = {}
        self._bm25: BM25L | None = None
        self._bm25_names: list[str] = []
        self._bm25_tokenized: list[list[str]] = []
        self._dirty = True

    def register(self, entry: ToolCatalogEntry) -> None:
        self._entries[entry.name] = entry
        self._dirty = True

    def get(self, name: str) -> ToolCatalogEntry | None:
        return self._entries.get(name)

    def get_spec(self, name: str) -> ToolSpec | None:
        """Backward compat — returns None for catalog entries (no ToolSpec stored)."""
        return None

    def list_all(self) -> list[ToolCatalogEntry]:
        """Backward compat — returns all entries."""
        return list(self._entries.values())

    def list_deferred(self) -> list[ToolCatalogEntry]:
        return [e for e in self._entries.values() if e.category == "deferred"]

    def classify(self, tool_name: str) -> ToolCategory:
        entry = self._entries.get(tool_name)
        if entry is not None:
            return entry.category
        if tool_name in CORE_TOOLS:
            return "core"
        if tool_name in DEFERRED_TOOLS:
            return "deferred"
        if tool_name in INTERNAL_TOOLS:
            return "internal"
        return "internal"

    def search(
        self,
        query: str,
        *,
        max_results: int = 8,
    ) -> list[SearchCandidate]:
        """Search deferred tools using BM25.  Returns candidates, does NOT activate."""
        self._rebuild_index_if_dirty()
        if self._bm25 is None:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        scored = [
            (scores[i], self._bm25_names[i])
            for i in range(len(self._bm25_names))
            if scores[i] > 0
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_results]

        candidates: list[SearchCandidate] = []
        for _score, name in top:
            entry = self._entries.get(name)
            if entry is None:
                continue
            reason = self._build_reason(query_tokens, entry)
            candidates.append(
                SearchCandidate(
                    name=name,
                    description=entry.description,
                    reason=reason,
                    when_to_use=entry.when_to_use,
                    activation_group=entry.activation_group,
                    tags=entry.selection_tags,
                    domains=entry.domains,
                )
            )
        return candidates

    def _rebuild_index_if_dirty(self) -> None:
        if not self._dirty:
            return
        deferred = self.list_deferred()
        self._bm25_names = [e.name for e in deferred]
        self._bm25_tokenized = [_tokenize(e.search_text) for e in deferred]
        if not self._bm25_tokenized:
            self._bm25 = None
        else:
            self._bm25 = BM25L(self._bm25_tokenized)
        self._dirty = False

    @staticmethod
    def _build_reason(query_tokens: list[str], entry: ToolCatalogEntry) -> str:
        search_text_tokens = set(_tokenize(entry.search_text))
        hits = [t for t in query_tokens if t in search_text_tokens]
        if not hits:
            return "matched query context"
        return f"matched: {', '.join(hits[:5])}"

    @staticmethod
    def build_search_text(
        name: str,
        description: str,
        schema_text: str = "",
        *,
        when_to_use: str = "",
        when_not_to_use: str = "",
        domains: tuple[str, ...] = (),
        file_types: tuple[str, ...] = (),
        selection_tags: tuple[str, ...] = (),
    ) -> str:
        """Build BM25 search text from tool identity and optional ToolCard fields.

        ToolCard fields are appended so they boost relevance without replacing
        the core identity text (name, description, schema).
        """
        parts: list[str] = [
            name,
            name.replace("_", " "),
            description,
            schema_text,
        ]
        # Append ToolCard fields (each is optional — no-op when empty)
        if when_to_use:
            parts.append(when_to_use)
        if when_not_to_use:
            parts.append(when_not_to_use)
        if domains:
            parts.extend(domains)
        if file_types:
            parts.extend(file_types)
        if selection_tags:
            parts.extend(selection_tags)
        return " ".join(filter(None, parts))


# ── Deferred Tool Store (per-run, backed by LoopState) ──


@dataclass(frozen=True)
class ActivatedToolRef:
    """Reference to an activated deferred tool."""
    tool_name: str
    activated_at_iteration: int
    source_query: str


class DeferredToolStore:
    """Per-run deferred tool activation state.

    Reads/writes LoopState discovery_* fields via ToolDiscoveryStateView.
    Only candidates from the last tool_search can be activated.
    """

    def __init__(self, max_active: int = 10) -> None:
        if max_active < 1:
            raise ValueError("max_active must be >= 1")
        self._max_active = max_active
        self._active: OrderedDict[str, ActivatedToolRef] = OrderedDict()
        self._pending_candidates: dict[str, SearchCandidate] = {}
        self._last_search_query: str = ""
        self._pinned: set[str] = set()

    @property
    def max_active(self) -> int:
        return self._max_active

    def set_pending_candidates(
        self,
        query: str,
        candidates: list[SearchCandidate],
    ) -> None:
        self._last_search_query = query
        self._pending_candidates = {c.name: c for c in candidates}

    def pending_names(self) -> list[str]:
        return list(self._pending_candidates.keys())

    def is_pending(self, name: str) -> bool:
        return name in self._pending_candidates

    def activate(
        self,
        tool_name: str,
        *,
        iteration: int,
        source_query: str | None = None,
    ) -> bool:
        """Activate a deferred tool.  Returns True if newly activated.

        Raises KeyError if tool_name is not in pending_candidates.
        """
        if tool_name in self._active:
            self._active.move_to_end(tool_name)
            return False
        if tool_name not in self._pending_candidates:
            raise KeyError(
                f"Cannot activate '{tool_name}': "
                f"not in pending candidates from last tool_search"
            )
        if not self._evict_if_needed():
            raise RuntimeError(
                f"Cannot activate '{tool_name}': "
                f"all {self._max_active} slots are pinned"
            )
        self._active[tool_name] = ActivatedToolRef(
            tool_name=tool_name,
            activated_at_iteration=iteration,
            source_query=source_query or self._last_search_query,
        )
        return True

    def is_active(self, tool_name: str) -> bool:
        return tool_name in self._active

    def active_names(self) -> list[str]:
        return list(self._active.keys())

    def active_refs(self) -> list[ActivatedToolRef]:
        return list(self._active.values())

    def pin(self, tool_name: str) -> None:
        self._pinned.add(tool_name)

    def unpin(self, tool_name: str) -> None:
        self._pinned.discard(tool_name)

    def _evict_if_needed(self) -> bool:
        while len(self._active) >= self._max_active:
            evicted = False
            for name in self._active:
                if name not in self._pinned:
                    del self._active[name]
                    evicted = True
                    break
            if not evicted:
                return False
        return True

    # ── LoopState sync ──

    def sync_to_state(self, state: dict[str, Any]) -> None:
        """Write current state to LoopState deferred_tool_state (single typed field)."""
        from rag.agent.loop.substate import DeferredToolState, DiscoveryCandidate, DiscoveryEvent

        # Append to search history if there's a new query
        history: list[DiscoveryEvent] = []
        existing = state.get("deferred_tool_state")
        if isinstance(existing, DeferredToolState):
            history = list(existing.search_history)
        if self._last_search_query and self._pending_candidates:
            history.append(DiscoveryEvent(
                query=self._last_search_query,
                candidates=list(self._pending_candidates.keys()),
                activated=list(self._active.keys()),
            ))

        state["deferred_tool_state"] = DeferredToolState(
            active_tools=list(self._active.keys()),
            active_tool_iterations={
                name: ref.activated_at_iteration
                for name, ref in self._active.items()
            },
            last_candidates=[
                DiscoveryCandidate(
                    name=c.name,
                    description=c.description,
                    reason=c.reason,
                )
                for c in self._pending_candidates.values()
            ],
            last_search_query=self._last_search_query,
            search_history=history[-50:],
            pinned_tools=list(self._pinned),
            capability_diagnostics=list(
                state.get("deferred_tool_state", DeferredToolState()).capability_diagnostics
            ),
        )

    def sync_from_state(self, state: dict[str, Any]) -> None:
        """Restore state from LoopState deferred_tool_state."""
        from rag.agent.loop.substate import DeferredToolState

        dts = state.get("deferred_tool_state")
        if not isinstance(dts, DeferredToolState):
            # Fallback for checkpoint migration: read old discovery_* flat fields
            active_names: list[str] = state.get("discovery_active_tools", [])
            iterations: dict[str, int] = state.get("discovery_active_tool_iterations", {})
            self._active = OrderedDict()
            for name in active_names:
                self._active[name] = ActivatedToolRef(
                    tool_name=name,
                    activated_at_iteration=iterations.get(name, 0),
                    source_query=state.get("discovery_last_search_query", ""),
                )
            candidates_raw = state.get("discovery_last_candidates", [])
            self._pending_candidates = {
                c["name"]: SearchCandidate(
                    name=c["name"],
                    description=c.get("description", ""),
                    reason=c.get("reason", ""),
                )
                for c in candidates_raw
            }
            self._last_search_query = state.get("discovery_last_search_query", "")
            self._pinned = set(state.get("discovery_pinned_tools", []))
            return

        self._active = OrderedDict()
        for name in dts.active_tools:
            self._active[name] = ActivatedToolRef(
                tool_name=name,
                activated_at_iteration=dts.active_tool_iterations.get(name, 0),
                source_query=dts.last_search_query,
            )
        self._pending_candidates = {
            c.name: SearchCandidate(
                name=c.name,
                description=c.description,
                reason=c.reason,
            )
            for c in dts.last_candidates
        }
        self._last_search_query = dts.last_search_query
        self._pinned = set(dts.pinned_tools)


# ── Visible Tool Resolver ──


def resolve_visible_tools(
    allowed_tools: list[str],
    *,
    catalog: ToolCatalog,
    store: DeferredToolStore,
) -> list[str]:
    """Return tool names currently visible to the model.

    Rules:
    1. Core tools always visible (if in allowed_tools).
    2. Deferred tools visible only when activated.
    3. Internal tools never visible.
    """
    visible: list[str] = []
    for name in allowed_tools:
        category = catalog.classify(name)
        if category == "internal":
            continue
        if category == "deferred" and not store.is_active(name):
            continue
        visible.append(name)
    return visible


# ── Tool Catalog Filter (backward compat, deprecated) ──


@dataclass(frozen=True)
class ToolCatalogFilter:
    """Per-definition overrides for tool categorization.

    .. deprecated:: Retained for backward compatibility.
    """
    promote_to_core: frozenset[str] = field(default_factory=frozenset)
    deny: frozenset[str] = field(default_factory=frozenset)
