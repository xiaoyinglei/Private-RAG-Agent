"""PR7: Activation group tests — batch activation and non-visibility change."""

from __future__ import annotations

from pydantic import BaseModel

from rag.agent.capabilities.catalog import (
    CORE_TOOLS,
    DEFERRED_TOOLS,
    DeferredToolStore,
    SearchCandidate,
    ToolCatalog,
    ToolCatalogEntry,
    flatten_schema,
    resolve_visible_tools,
)
from rag.agent.capabilities.tool_search import (
    execute_activate_tools,
)
from rag.agent.tools.card import ToolCard
from rag.agent.tools.spec import ToolPermissions, ToolSpec

# ── Helpers ──


class _DummyInput(BaseModel):
    query: str = ""


class _DummyOutput(BaseModel):
    results: list = []


def _make_spec(name: str, description: str, *, aci: ToolCard | None = None) -> ToolSpec:
    return ToolSpec(
        name=name, description=description,
        input_model=_DummyInput, output_model=_DummyOutput,
        error_model=_DummyOutput, permissions=ToolPermissions(),
        timeout_seconds=5.0, aci=aci,
    )


def _build_catalog(specs: list[ToolSpec]) -> ToolCatalog:
    """Build catalog with activation_group populated from ToolCard or _DEFAULT_ACTIVATION_GROUPS."""
    from rag.agent.capabilities.catalog import _DEFAULT_ACTIVATION_GROUPS

    catalog = ToolCatalog()
    for spec in specs:
        if spec.name in CORE_TOOLS or spec.name in DEFERRED_TOOLS:
            continue
        schema_text = flatten_schema(spec.input_model.model_json_schema())
        card = spec.aci
        activation_group = (
            card.activation_group
            if card and card.activation_group
            else _DEFAULT_ACTIVATION_GROUPS.get(spec.name, "")
        )
        search_text = ToolCatalog.build_search_text(
            spec.name, spec.description, schema_text,
            when_to_use=card.when_to_use if card else "",
            when_not_to_use=card.when_not_to_use if card else "",
            domains=card.domains if card else (),
            file_types=card.file_types if card else (),
            selection_tags=card.selection_tags if card else (),
        )
        catalog.register(ToolCatalogEntry(
            name=spec.name, description=spec.description,
            category="deferred", search_text=search_text,
            schema_text=schema_text,
            activation_group=activation_group,
        ))
    return catalog


class TestActivationGroups:

    def test_activate_by_group(self) -> None:
        """activate_tools with group='rag' activates all rag-group candidates."""
        specs = [
            _make_spec("search_tool", "RAG search", aci=ToolCard(activation_group="rag")),
            _make_spec("embed_tool", "RAG embed", aci=ToolCard(activation_group="rag")),
            _make_spec("code_tool", "Code runner", aci=ToolCard(activation_group="code")),
        ]
        catalog = _build_catalog(specs)
        store = DeferredToolStore(max_active=10)

        # Simulate tool_search
        candidates = catalog.search("rag", max_results=5)
        store.set_pending_candidates("rag", candidates)

        # Activate by group
        output = execute_activate_tools(
            names=[],
            catalog=catalog, store=store,
            allowed_tools=["search_tool", "embed_tool", "code_tool"],
            deny_tools=frozenset(),
            iteration=1,
            group="rag",
        )
        assert "search_tool" in output.activated
        assert "embed_tool" in output.activated
        assert "code_tool" not in output.activated  # different group

    def test_activate_by_group_ignores_non_matching(self) -> None:
        """Group activation only picks tools with matching activation_group."""
        specs = [
            _make_spec("code_tool", "Code", aci=ToolCard(activation_group="code")),
        ]
        catalog = _build_catalog(specs)
        store = DeferredToolStore(max_active=10)

        candidates = catalog.search("code", max_results=5)
        store.set_pending_candidates("code", candidates)

        output = execute_activate_tools(
            names=[], catalog=catalog, store=store,
            allowed_tools=["code_tool"],
            deny_tools=frozenset(),
            iteration=1,
            group="rag",
        )
        assert "code_tool" not in output.activated  # group mismatch

    def test_activation_group_does_not_change_visibility(self) -> None:
        """resolve_visible_tools result is unchanged: category still controls visibility."""
        catalog = _build_catalog([])
        store = DeferredToolStore(max_active=10)

        # Visibility should still be based on core/deferred, not activation_group
        allowed = ["tool_search", "vector_search", "run_python"]
        visible = resolve_visible_tools(allowed, catalog=catalog, store=store)

        # tool_search is in CORE_TOOLS → visible
        assert "tool_search" in visible
        # vector_search is in DEFERRED_TOOLS, not activated → NOT visible
        assert "vector_search" not in visible
        # run_python is a deferred workspace tool, so it is hidden until activated.
        assert "run_python" not in visible

        store.set_pending_candidates(
            "workspace",
            [
                SearchCandidate(
                    name="run_python",
                    description="Run Python",
                    reason="workspace python",
                )
            ],
        )
        store.activate("run_python", iteration=1, source_query="workspace")

        assert "run_python" in resolve_visible_tools(
            allowed,
            catalog=catalog,
            store=store,
        )

    def test_backward_compat_without_toolcard(self) -> None:
        """Tools without ToolCard get activation_group from _DEFAULT_ACTIVATION_GROUPS."""
        spec = _make_spec("search_tool", "Search")  # no ToolCard
        catalog = _build_catalog([spec])
        store = DeferredToolStore(max_active=10)

        candidates = catalog.search("search", max_results=5)
        store.set_pending_candidates("search", candidates)

        # Since no ToolCard and no _DEFAULT_ACTIVATION_GROUPS entry for "search_tool",
        # activation_group should be ""
        output = execute_activate_tools(
            names=[], catalog=catalog, store=store,
            allowed_tools=["search_tool"],
            deny_tools=frozenset(),
            iteration=1,
            group="rag",
        )
        assert "search_tool" not in output.activated  # no group match
