"""ToolCatalog assembly from a registered tool surface."""

from __future__ import annotations

from typing import Literal

from rag.agent.capabilities.catalog import (
    _DEFAULT_ACTIVATION_GROUPS,
    CORE_TOOLS,
    DEFERRED_TOOLS,
    INTERNAL_TOOLS,
    ToolCatalog,
    ToolCatalogEntry,
    flatten_schema,
)
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.tools.registry import ToolRegistry


def build_tool_catalog(
    registry: ToolRegistry,
    policy: AgentRuntimePolicy,
) -> ToolCatalog:
    """Build a searchable catalog from the currently registered tools."""
    filt = policy.tool_catalog_filter
    catalog = ToolCatalog()
    for spec in registry.list_all():
        if spec.name in filt.deny:
            continue
        category: Literal["core", "deferred", "internal"]
        if spec.name in filt.promote_to_core or spec.name in CORE_TOOLS:
            category = "core"
        elif spec.name in DEFERRED_TOOLS:
            category = "deferred"
        elif spec.name in INTERNAL_TOOLS:
            category = "internal"
        else:
            category = "internal"

        schema_text = ""
        if category == "deferred" and hasattr(
            spec.input_model,
            "model_json_schema",
        ):
            schema_text = flatten_schema(spec.input_model.model_json_schema())

        card = spec.aci
        search_text = ToolCatalog.build_search_text(
            spec.name,
            spec.description,
            schema_text,
            when_to_use=card.when_to_use if card else "",
            when_not_to_use=card.when_not_to_use if card else "",
            domains=card.domains if card else (),
            file_types=card.file_types if card else (),
            selection_tags=card.selection_tags if card else (),
        )
        catalog.register(
            ToolCatalogEntry(
                name=spec.name,
                description=spec.description,
                category=category,
                search_text=search_text,
                schema_text=schema_text,
                activation_group=(
                    card.activation_group
                    if card and card.activation_group
                    else _DEFAULT_ACTIVATION_GROUPS.get(spec.name, "")
                ),
                when_to_use=card.when_to_use if card else "",
                when_not_to_use=card.when_not_to_use if card else "",
                domains=card.domains if card else (),
                file_types=card.file_types if card else (),
                failure_codes=card.failure_codes if card else (),
                selection_tags=card.selection_tags if card else (),
            ),
        )
    return catalog


__all__ = ["build_tool_catalog"]
