"""PR6: Formatter golden/snapshot tests.

Each formatter is tested with a known input to verify output contains
expected anchor strings.  Uses real Pydantic output models.
"""

from __future__ import annotations

from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.spec import ToolError, ToolResult


def _build_registry():
    return create_builtin_tool_registry()


def _format(tool_name: str, result: ToolResult) -> str | None:
    registry = _build_registry()
    formatter = registry.get_formatter(tool_name)
    if formatter is None:
        return None
    section = formatter.format_result(result)
    return section.content if section is not None else None


class TestFormatterSnapshots:

    def test_vector_search_renders_items(self) -> None:
        from rag.agent.tools.rag_tools import SearchOutput

        output = SearchOutput(
            items=[{"text": "Machine learning improves accuracy.", "evidence_id": "ev-1", "score": 0.95}],
        )
        result = ToolResult(
            tool_call_id="tc-1", tool_name="vector_search",
            status="ok", output=output, latency_ms=100.0,
        )
        content = _format("vector_search", result)
        assert content is not None
        assert "vector_search results" in content
        assert "Machine learning" in content

    def test_keyword_search_renders_items(self) -> None:
        from rag.agent.tools.rag_tools import SearchOutput

        output = SearchOutput(
            items=[{"text": "GDPR compliance requirements", "evidence_id": "ev-ks-1"}],
        )
        result = ToolResult(
            tool_call_id="tc-2", tool_name="keyword_search",
            status="ok", output=output, latency_ms=50.0,
        )
        content = _format("keyword_search", result)
        assert content is not None
        assert "keyword_search results" in content
        assert "GDPR" in content

    def test_asset_list_renders_assets(self) -> None:
        from rag.agent.tools.asset_tools import AssetDescriptor, AssetListOutput

        output = AssetListOutput(
            assets=[
                AssetDescriptor(
                    asset_id=1, doc_id=100, asset_type="table",
                    sheet_name="Sheet1",
                ),
            ],
            truncated=False,
        )
        result = ToolResult(
            tool_call_id="tc-3", tool_name="asset_list",
            status="ok", output=output, latency_ms=15.0,
        )
        content = _format("asset_list", result)
        assert content is not None
        assert "asset_list results" in content
        assert "asset_id=1" in content
        assert "doc_id=100" in content

    def test_llm_generate_renders_text(self) -> None:
        from rag.agent.tools.llm_tools import LLMTextOutput

        output = LLMTextOutput(
            text="The key finding: ML improves accuracy by 30%.",
            evidence_ids=["ev-1", "ev-2"],
            citation_ids=["c-1"],
        )
        result = ToolResult(
            tool_call_id="tc-4", tool_name="llm_generate",
            status="ok", output=output, latency_ms=200.0,
        )
        content = _format("llm_generate", result)
        assert content is not None
        assert "llm_generate" in content
        assert "ML improves" in content
        assert "ev-1" in content

    def test_rag_search_answer_renders_answer(self) -> None:
        from rag.agent.tools.rag_answer_tools import RAGSearchAnswerOutput

        output = RAGSearchAnswerOutput(
            text="Q3 revenue was $5.2B.",
            groundedness_flag=True,
        )
        result = ToolResult(
            tool_call_id="tc-5", tool_name="rag_search_answer",
            status="ok", output=output, latency_ms=300.0,
        )
        content = _format("rag_search_answer", result)
        assert content is not None
        assert "rag_search_answer results" in content
        assert "$5.2B" in content
        assert "groundedness=True" in content

    def test_formatter_returns_none_on_error(self) -> None:
        result = ToolResult(
            tool_call_id="tc-err", tool_name="vector_search",
            status="error",
            error=ToolError(code="timeout", message="timed out", retryable=True),
            latency_ms=5000.0,
        )
        content = _format("vector_search", result)
        assert content is None
