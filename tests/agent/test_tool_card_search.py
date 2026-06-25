"""PR5: ToolCard-driven search index — integration tests."""

from __future__ import annotations

from pydantic import BaseModel

from rag.agent.capabilities.catalog import (
    CORE_TOOLS,
    DEFERRED_TOOLS,
    SearchCandidate,
    ToolCatalog,
    ToolCatalogEntry,
    flatten_schema,
)
from rag.agent.tools.card import ToolCard
from rag.agent.tools.spec import ToolPermissions, ToolSpec

# ── Helpers ──


class _DummyInput(BaseModel):
    query: str = ""


class _DummyOutput(BaseModel):
    results: list = []


def _make_spec(
    name: str,
    description: str,
    *,
    aci: ToolCard | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_model=_DummyInput,
        output_model=_DummyOutput,
        error_model=_DummyOutput,
        permissions=ToolPermissions(),
        timeout_seconds=5.0,
        aci=aci,
    )


def _build_test_catalog(specs: list[ToolSpec]) -> ToolCatalog:
    catalog = ToolCatalog()
    for spec in specs:
        if spec.name in CORE_TOOLS or spec.name in DEFERRED_TOOLS:
            continue
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
        # Register as deferred so they are searchable
        catalog.register(
            ToolCatalogEntry(
                name=spec.name,
                description=spec.description,
                category="deferred",
                search_text=search_text,
                schema_text=schema_text,
                activation_group=card.activation_group if card else "",
                when_to_use=card.when_to_use if card else "",
                when_not_to_use=card.when_not_to_use if card else "",
                domains=card.domains if card else (),
                file_types=card.file_types if card else (),
                failure_codes=card.failure_codes if card else (),
                selection_tags=card.selection_tags if card else (),
            ),
        )
    return catalog


class TestToolCardSearch:
    """PR5: ToolCard fields enrich search index and results."""

    def test_search_text_includes_card_fields(self) -> None:
        """ToolCard when_to_use and domains appear in the indexed search_text."""
        card = ToolCard(
            when_to_use="Use when analyzing financial spreadsheets",
            when_not_to_use="Do not use for plain text",
            domains=("finance", "accounting"),
            selection_tags=("spreadsheet", "excel"),
        )
        spec = _make_spec("excel_analyze", "Analyze Excel spreadsheets", aci=card)
        catalog = _build_test_catalog([spec])

        # Search for a domain tag
        results = catalog.search("financial spreadsheets", max_results=5)
        assert len(results) > 0
        assert any(r.name == "excel_analyze" for r in results)

        # Search for a selection tag
        results2 = catalog.search("excel tool", max_results=5)
        assert any(r.name == "excel_analyze" for r in results2)

    def test_search_candidate_carries_card_summary(self) -> None:
        """SearchCandidate includes when_to_use, activation_group, tags, domains."""
        card = ToolCard(
            when_to_use="Use to generate text completions",
            activation_group="code",
            selection_tags=("llm", "generate"),
            domains=("nlp",),
        )
        spec = _make_spec("text_completions", "Generate text with an LLM", aci=card)
        catalog = _build_test_catalog([spec])

        results = catalog.search("generate text", max_results=5)
        assert len(results) == 1
        c = results[0]
        assert c.name == "text_completions"
        assert c.when_to_use == "Use to generate text completions"
        assert c.activation_group == "code"
        assert c.tags == ("llm", "generate")
        assert c.domains == ("nlp",)

    def test_search_without_card_fallback(self) -> None:
        """Tools without ToolCard remain searchable via name + description + schema."""
        spec = _make_spec("legacy_tool", "Legacy data processor")
        catalog = _build_test_catalog([spec])

        results = catalog.search("legacy data", max_results=5)
        assert len(results) == 1
        c = results[0]
        assert c.name == "legacy_tool"
        assert c.when_to_use == ""
        assert c.activation_group == ""
        assert c.tags == ()
        assert c.domains == ()

    def test_domain_tags_improve_ranking(self) -> None:
        """A tool whose card domains match the query ranks higher than one without."""
        card = ToolCard(
            when_to_use="Use for image processing",
            domains=("image", "vision"),
            selection_tags=("ocr",),
        )
        matching = _make_spec("image_tool", "Process images", aci=card)
        non_matching = _make_spec("text_tool", "Process text")

        catalog = _build_test_catalog([matching, non_matching])

        results = catalog.search("image vision ocr", max_results=5)
        assert len(results) >= 1
        # The tool with matching domain tags should appear first
        assert results[0].name == "image_tool"

    def test_build_search_text_empty_card(self) -> None:
        """build_search_text with no ToolCard fields returns name + desc + schema only."""
        text = ToolCatalog.build_search_text(
            "my_tool", "A test tool", '{"type":"object"}',
        )
        assert "my_tool" in text
        assert "A test tool" in text
        assert '{"type":"object"}' in text

    def test_build_search_text_full_card(self) -> None:
        """build_search_text appends all provided card fields."""
        text = ToolCatalog.build_search_text(
            "my_tool", "desc", '{"type":"object"}',
            when_to_use="Use for X",
            when_not_to_use="Not for Y",
            domains=("d1", "d2"),
            file_types=(".pdf", ".xlsx"),
            selection_tags=("tag1", "tag2"),
        )
        assert "Use for X" in text
        assert "Not for Y" in text
        assert "d1" in text
        assert "d2" in text
        assert ".pdf" in text
        assert "tag1" in text
