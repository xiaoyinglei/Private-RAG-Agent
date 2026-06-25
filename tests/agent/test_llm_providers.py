from __future__ import annotations


class TestLLMLoopModelTurnProviderBasic:
    """Basic sanity checks for LLMLoopModelTurnProvider."""

    def test_module_exports_expected_symbols(self) -> None:
        """Verify the module exports the expected public API after cleanup."""
        import rag.agent.core.llm_providers as m

        assert hasattr(m, "LLMLoopModelTurnProvider")
        assert hasattr(m, "LoopModelDecision")
        assert hasattr(m, "parse_loop_model_turn")
        assert hasattr(m, "create_loop_model_turn_provider")

    def test_retrieval_hint_code_is_removed(self) -> None:
        """Verify LLMRetrievalHintProvider no longer exists in the module."""
        import rag.agent.core.llm_providers as m

        assert not hasattr(m, "LLMRetrievalHintProvider")
        assert not hasattr(m, "create_default_providers")
        assert not hasattr(m, "RetrievalHintDecision")
        assert not hasattr(m, "_extract_quoted_terms")
        assert not hasattr(m, "_validate_retrieval_signals")
