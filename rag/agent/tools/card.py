"""ToolCard — ACI companion model for ToolSpec.

ToolCard is a lightweight metadata model that describes *how and when* to use a tool,
separate from the runtime contract in ToolSpec.  It is the "user manual" for LLMs
and the search index for tool discovery.

Design rules:
- ToolCard never enters LoopState (it lives on ToolSpec in the registry).
- ToolCard does not import from agent internals (no circular deps).
- All fields have defaults — a ToolCard can be incrementally populated.
- Phase 1: metadata only.  output_cap_policy / pagination / externalization
  are informational and do not change execution behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCard:
    """ACI companion model — the "user manual" for a tool.

    ToolSpec governs the runtime contract (permissions, timeouts, models).
    ToolCard governs discoverability and usage guidance for the LLM:
    when to use it, what it needs, what can go wrong, and how to find it.
    """

    # ── Usage guidance ──
    when_to_use: str = ""
    when_not_to_use: str = ""

    # ── Preconditions ──
    preconditions: tuple[str, ...] = ()
    required_context: tuple[str, ...] = ()

    # ── Examples ──
    input_examples: tuple[dict[str, object], ...] = ()
    output_examples: tuple[str, ...] = ()

    # ── Output contract (Phase 1: informational only, no execution changes) ──
    output_cap_policy: str = "truncate"  # "truncate" | "externalize" | "ref"
    pagination: str = ""                 # "page" | "cursor" | "offset" | ""
    externalization: str = "auto"        # "auto" | "always" | "never"

    # ── Failure model ──
    failure_codes: tuple[str, ...] = ()
    retryable: bool = False
    user_recoverable: bool = False
    model_next_action: str = ""

    # ── Grouping & tagging (for search and activation) ──
    selection_tags: tuple[str, ...] = ()
    file_types: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    activation_group: str = ""  # "" = unassigned; "rag" | "file" | "code" | "workspace" | "mcp" | "network"
