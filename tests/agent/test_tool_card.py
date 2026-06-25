"""PR4: ToolCard ACI companion model — unit tests."""

from __future__ import annotations

from pydantic import BaseModel

from rag.agent.tools.card import ToolCard
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec


class _DummyInput(BaseModel):
    text: str


class _DummyOutput(BaseModel):
    result: str


def _make_spec(name: str = "test_tool", *, aci: ToolCard | None = None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="A test tool",
        input_model=_DummyInput,
        output_model=_DummyOutput,
        error_model=ToolError,
        permissions=ToolPermissions(),
        timeout_seconds=1.0,
        aci=aci,
    )


class TestToolCard:
    def test_minimal_toolcard_defaults(self) -> None:
        """Empty ToolCard has sensible defaults."""
        card = ToolCard()
        assert card.when_to_use == ""
        assert card.when_not_to_use == ""
        assert card.preconditions == ()
        assert card.required_context == ()
        assert card.input_examples == ()
        assert card.output_examples == ()
        assert card.output_cap_policy == "truncate"
        assert card.pagination == ""
        assert card.externalization == "auto"
        assert card.failure_codes == ()
        assert card.retryable is False
        assert card.user_recoverable is False
        assert card.model_next_action == ""
        assert card.selection_tags == ()
        assert card.file_types == ()
        assert card.domains == ()
        assert card.activation_group == ""

    def test_fully_populated_toolcard(self) -> None:
        """All fields can be set and read back."""
        card = ToolCard(
            when_to_use="Use for searching documents by semantic similarity",
            when_not_to_use="Do not use for exact keyword matching",
            preconditions=("index_loaded",),
            required_context=("query_text",),
            input_examples=({"query": "machine learning", "top_k": 5},),
            output_examples=("[{evidence_id: ev-1, score: 0.95}]",),
            output_cap_policy="externalize",
            pagination="offset",
            externalization="auto",
            failure_codes=("timeout", "index_unavailable"),
            retryable=True,
            user_recoverable=True,
            model_next_action="Retry with a shorter query or smaller top_k",
            selection_tags=("search", "retrieval"),
            file_types=(),
            domains=("documents", "knowledge_base"),
            activation_group="rag",
        )
        assert card.when_to_use.startswith("Use for")
        assert card.when_not_to_use.startswith("Do not use")
        assert card.preconditions == ("index_loaded",)
        assert card.required_context == ("query_text",)
        assert len(card.input_examples) == 1
        assert len(card.output_examples) == 1
        assert card.output_cap_policy == "externalize"
        assert card.pagination == "offset"
        assert card.externalization == "auto"
        assert card.failure_codes == ("timeout", "index_unavailable")
        assert card.retryable is True
        assert card.user_recoverable is True
        assert card.model_next_action != ""
        assert card.selection_tags == ("search", "retrieval")
        assert card.domains == ("documents", "knowledge_base")
        assert card.activation_group == "rag"

    def test_toolspec_with_aci(self) -> None:
        """ToolSpec can carry a ToolCard via the aci field."""
        card = ToolCard(
            when_to_use="Use when you need to search",
            activation_group="rag",
        )
        spec = _make_spec("vector_search", aci=card)
        assert spec.aci is card
        assert spec.aci.when_to_use == "Use when you need to search"
        assert spec.aci.activation_group == "rag"

    def test_toolspec_without_aci(self) -> None:
        """Backward compat: ToolSpec without aci has aci=None."""
        spec = _make_spec("legacy_tool")
        assert spec.aci is None

    def test_toolcard_immutable(self) -> None:
        """ToolCard is frozen — cannot be mutated after creation."""
        from dataclasses import FrozenInstanceError

        import pytest

        card = ToolCard(when_to_use="test")
        with pytest.raises(FrozenInstanceError):
            card.when_to_use = "new"  # type: ignore[misc]
