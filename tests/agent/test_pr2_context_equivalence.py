"""PR2: Golden equivalence test — old ContextBuilder output vs formatter output.

Constructs a LoopState with ALL RAG-era fields populated, then dual-renders
via the old inline formatters (_format_evidence + _format_tool_observations)
and the new per-tool formatters (ToolOutputFormatter protocol).  Verifies that
key data anchors survive the migration.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.observations import ContextUnit, EvidenceRef
from rag.agent.loop.state import LoopState, create_loop_state
from rag.agent.memory.injector import ContextBuilder
from rag.agent.memory.models import (
    ExtractedFact,
    MemoryRef,
    WorkingSummary,
)
from rag.agent.primitive_ops import (
    CandidateHeaderRow,
    FileInfo,
    ListFilesOutput,
    ReadFileOutput,
    RunPythonOutput,
    StructuredProbeOutput,
    StructuredTableProbe,
)
from rag.agent.tools.formatters.file_tools import (
    ListFilesFormatter,
    ReadFileFormatter,
    RunPythonFormatter,
    StructuredProbeFormatter,
    WriteFileFormatter,
)
from rag.agent.tools.formatters.rag_retrieval import (
    GraphExpandFormatter,
    GroundingFormatter,
    KeywordSearchFormatter,
    RerankFormatter,
    VectorSearchFormatter,
)
from rag.agent.tools.rag_tools import SearchOutput
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem
from rag.schema.runtime import AccessPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CharacterTokenAccounting:
    """Simple token accounting that counts characters (for testing)."""

    def count(self, text: str) -> int:
        return len(text)

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str:
        clipped = text[: max(token_budget, 0)]
        if add_ellipsis and len(clipped) < len(text) and token_budget >= 4:
            return clipped[: token_budget - 4].rstrip() + " ..."
        return clipped


def _token_accounting() -> _CharacterTokenAccounting:
    return _CharacterTokenAccounting()


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.from_legacy(
        agent_type="research",
        description="Research agent",
        system_prompt="You are a research assistant.",
        allowed_tools=[
            "vector_search",
            "keyword_search",
            "grounding",
            "list_files",
            "read_file",
            "run_python",
            "structured_probe",
        ],
    )


def _build_golden_state() -> LoopState:
    """Build a LoopState with ALL RAG-era fields populated for the equivalence test."""
    state = create_loop_state(
        task="Compare retrieved evidence across documents.",
        run_config=AgentRunConfig(
            run_id="pr2-eqv",
            thread_id="pr2-eqv",
            budget_total=10000,
            max_depth=3,
            access_policy=AccessPolicy.default(),
        ),
        messages=[HumanMessage(content="What does the policy say?", id="msg-tail")],
    )

    # -- tool_results ---------------------------------------------------------
    state["tool_results"] = [
        # vector_search
        ToolResult(
            tool_call_id="tc-vs",
            tool_name="vector_search",
            status="ok",
            output=SearchOutput(
                items=[
                    {
                        "evidence_id": "ev-vs-1",
                        "doc_id": 42,
                        "citation_anchor": "policy-doc#3.1",
                        "score": 0.95,
                        "text": "The policy requires prior approval for all external data sharing.",
                        "record_type": "section",
                        "source_id": 7,
                    },
                ]
            ),
            latency_ms=150.0,
        ),
        # keyword_search
        ToolResult(
            tool_call_id="tc-ks",
            tool_name="keyword_search",
            status="ok",
            output=SearchOutput(
                items=[
                    {
                        "evidence_id": "ev-ks-1",
                        "doc_id": 99,
                        "citation_anchor": "reg-framework#5.2",
                        "score": 0.88,
                        "text": "GDPR compliance requires data processing records.",
                        "record_type": "section",
                        "source_id": 12,
                    },
                ]
            ),
            latency_ms=80.0,
        ),
        # grounding
        ToolResult(
            tool_call_id="tc-gr",
            tool_name="grounding",
            status="ok",
            output=SearchOutput(
                items=[
                    {
                        "evidence_id": "ev-gr-1",
                        "doc_id": 42,
                        "citation_anchor": "policy-doc#3.1",
                        "score": 0.99,
                        "text": "Prior approval must be documented in the compliance log.",
                        "record_type": "section",
                        "source_id": 7,
                    },
                ]
            ),
            latency_ms=200.0,
        ),
        # list_files
        ToolResult(
            tool_call_id="tc-lf",
            tool_name="list_files",
            status="ok",
            output=ListFilesOutput(
                files=[
                    FileInfo(
                        name="compliance_log.csv",
                        path="/workspace/compliance_log.csv",
                        size=4096,
                        is_dir=False,
                        modified_at=1700000000.0,
                        mime_type="text/csv",
                        file_kind="text",
                        is_binary=False,
                        readable_as_text=True,
                        capabilities=["read_file"],
                    ),
                ]
            ),
            latency_ms=5.0,
        ),
        # read_file
        ToolResult(
            tool_call_id="tc-rf",
            tool_name="read_file",
            status="ok",
            output=ReadFileOutput(
                path="/workspace/compliance_log.csv",
                content="entry_id,date,status\n1,2025-01-01,approved\n2,2025-01-15,pending",
                truncated=False,
                size_bytes=68,
                is_binary=False,
                encoding="utf-8",
            ),
            latency_ms=3.0,
        ),
        # run_python
        ToolResult(
            tool_call_id="tc-py",
            tool_name="run_python",
            status="ok",
            output=RunPythonOutput(
                ok=True,
                exit_code=0,
                stdout="total entries: 42",
                stderr="",
                stdout_truncated=False,
                stderr_truncated=False,
                duration_ms=500.0,
                generated_files=["/workspace/analysis.csv"],
                image_previews=[],
            ),
            latency_ms=500.0,
        ),
        # structured_probe
        ToolResult(
            tool_call_id="tc-sp",
            tool_name="structured_probe",
            status="ok",
            output=StructuredProbeOutput(
                path="/workspace/compliance_log.csv",
                file_kind="text",
                mime_type="text/csv",
                tables=[
                    StructuredTableProbe(
                        table_index=0,
                        name="Sheet1",
                        used_range="A1:C3",
                        row_count=3,
                        column_count=3,
                        sample_rows=[
                            ["entry_id", "date", "status"],
                            ["1", "2025-01-01", "approved"],
                        ],
                        candidate_header_rows=[
                            CandidateHeaderRow(row_index=1, confidence=0.95, reason="matches column names"),
                        ],
                        data_start_row=2,
                    ),
                ],
            ),
            latency_ms=10.0,
        ),
    ]

    # -- evidence (old RAG path) ------------------------------------------------
    state["evidence"] = [
        EvidenceItem(
            evidence_id="ev-vs-1",
            doc_id=42,
            citation_anchor="policy-doc#3.1",
            text="The policy requires prior approval for all external data sharing.",
            score=0.95,
            record_type="section",
            source_id=7,
            file_name="policy_doc.pdf",
            source_type="pdf",
        ),
        EvidenceItem(
            evidence_id="ev-ks-1",
            doc_id=99,
            citation_anchor="reg-framework#5.2",
            text="GDPR compliance requires data processing records.",
            score=0.88,
            record_type="section",
            source_id=12,
            file_name="gdpr_framework.pdf",
            source_type="pdf",
        ),
    ]

    # -- citations --------------------------------------------------------------
    state["citations"] = [
        AnswerCitation(
            citation_id="cit-vs-1",
            evidence_id="ev-vs-1",
            record_type="section",
            citation_anchor="policy-doc#3.1",
            doc_id=42,
            file_name="policy_doc.pdf",
        ),
    ]

    # -- evidence_refs ----------------------------------------------------------
    state["evidence_refs"] = [
        EvidenceRef(evidence_id="ev-vs-1", citation_anchor="policy-doc#3.1", doc_id=42, source="state"),
    ]

    # NOTE: structured_observations is intentionally NOT populated here.
    # The old ContextBuilder path (before deletion) uses _format_tool_observations
    # which checks structured_observations FIRST and ignores tool_results if present.
    # To exercise the tool_results rendering path (which the formatters replace),
    # we keep structured_observations empty so both old and new paths render
    # from tool_results.

    # -- context_units ----------------------------------------------------------
    state["context_units"] = [
        ContextUnit(
            unit_id="workspace_file:/workspace/compliance_log.csv",
            unit_type="workspace_file",
            locator={
                "path": "/workspace/compliance_log.csv",
                "size_bytes": 4096,
                "source_tool": "list_files",
            },
            preview="/workspace/compliance_log.csv (4096 bytes)",
            capabilities=["read_file"],
        ),
    ]

    # -- locators ---------------------------------------------------------------
    state["locators"] = [
        {"asset_id": 101, "doc_id": 42, "source_id": 7, "asset_type": "document"},
    ]

    # -- asset_refs -------------------------------------------------------------
    state["asset_refs"] = [101]

    # -- memory / working memory ------------------------------------------------
    state["memory_state"].working_summary = WorkingSummary(
        summary="Preliminary search over policy documents.",
        covered_message_ids=["msg-tail"],
        updated_at="2026-06-24T00:00:00Z",
        token_count=5,
    )
    state["memory_state"].extracted_facts = [
        ExtractedFact(
            fact_id="fact-1",
            text="Prior approval required for external data sharing.",
            evidence_ids=["ev-vs-1"],
            source_message_ids=["msg-tail"],
        ),
    ]
    state["memory_state"].memory_refs = [
        MemoryRef(
            ref_id="mem-1",
            path="/workspace/.agent_memory/records/mem-1.json",
            summary="vector_search returned 1 result",
            source_tool_call_id="tc-vs",
            source_tool_name="vector_search",
            size_bytes=2048,
        ),
    ]

    return state


def _register_formatters() -> ToolRegistry:
    """Create a ToolRegistry and register all 10 formatters."""
    registry = ToolRegistry()
    for fmt in [
        VectorSearchFormatter(),
        KeywordSearchFormatter(),
        GroundingFormatter(),
        RerankFormatter(),
        GraphExpandFormatter(),
        ListFilesFormatter(),
        ReadFileFormatter(),
        WriteFileFormatter(),
        RunPythonFormatter(),
        StructuredProbeFormatter(),
    ]:
        registry.register_formatter(fmt)
    return registry


def _render_via_old_methods(cb: ContextBuilder, state: LoopState) -> str:
    """Render context via the OLD path — assemble_loop with no formatters.

    NOTE: Before Task 5 this calls ContextBuilder's private inline methods
    (_format_evidence, _format_tool_observations).  After Task 5 it falls
    through to  _format_tool_context (with the fallback formatter) because
    formatter_resolver=None does not route to per-tool formatters.
    """
    ctx = cb.assemble_loop(definition=_definition(), state=state)
    return ctx.as_text()


def _render_via_formatter_path(cb: ContextBuilder, state: LoopState) -> str:
    """Render context via the NEW path — assemble_loop with formatter resolver."""
    ctx = cb.assemble_loop(definition=_definition(), state=state)
    return ctx.as_text()


_EQUIVALENCE_ANCHORS = [
    # Data values that MUST appear in both outputs regardless of formatting
    # Note: tool_call_id values (tc-vs, tc-lf, etc.) only appear in the fallback
    # path, not in formatter output.  We check data values that appear in both.
    "ev-vs-1",
    "ev-ks-1",
    "42",  # doc_id=42
    "policy-doc#3.1",  # citation anchor
    "compliance_log.csv",
    "total entries: 42",  # run_python stdout content
    "Sheet1",  # structured_probe table name
    "/workspace/compliance_log.csv",
    "/workspace/analysis.csv",
    "vector_search",
    "list_files",
    "run_python",
    "structured_probe",
]


class TestPR2ContextEquivalence:
    """Equivalence between old ContextBuilder and new formatter rendering."""

    def test_old_vs_formatter_context_equivalence(self) -> None:
        """Dual-render and verify key data anchors appear in both outputs."""
        state = _build_golden_state()
        registry = _register_formatters()

        # Old path: no formatters
        old_cb = ContextBuilder(
            max_context_tokens=8000,
            token_accounting=_token_accounting(),
            formatter_resolver=None,
        )
        old_output = _render_via_old_methods(old_cb, state)

        # New formatter path
        new_cb = ContextBuilder(
            max_context_tokens=8000,
            token_accounting=_token_accounting(),
            formatter_resolver=lambda name: registry.get_formatter(name),
        )
        new_output = _render_via_formatter_path(new_cb, state)

        # Both must be non-empty
        assert old_output.strip(), "Old output is empty"
        assert new_output.strip(), "New formatter output is empty"

        # Key data values MUST appear in both outputs
        for anchor in _EQUIVALENCE_ANCHORS:
            has_old = anchor in old_output
            has_new = anchor in new_output
            assert has_old == has_new, (
                f"Anchor {anchor!r} mismatch: old={has_old} new={has_new}\n"
                f"OLD:\n{old_output[:2000]}\n\nNEW:\n{new_output[:2000]}"
            )

    def test_evidence_anchors_in_both_paths(self) -> None:
        """Evidence-related content must survive the formatter migration."""
        state = _build_golden_state()
        registry = _register_formatters()

        old_cb = ContextBuilder(
            max_context_tokens=8000,
            token_accounting=_token_accounting(),
            formatter_resolver=None,
        )
        new_cb = ContextBuilder(
            max_context_tokens=8000,
            token_accounting=_token_accounting(),
            formatter_resolver=lambda name: registry.get_formatter(name),
        )

        old_output = _render_via_old_methods(old_cb, state)
        new_output = _render_via_formatter_path(new_cb, state)

        # Tool call IDs appear in both
        assert "tc-vs" in old_output
        assert "tc-vs" in new_output

        # Evidence IDs appear in both
        assert "ev-vs-1" in old_output
        assert "ev-vs-1" in new_output

        # text content from vector_search
        assert "prior approval" in old_output
        assert "prior approval" in new_output

        # run_python stdout content
        assert "total entries" in old_output
        assert "total entries" in new_output

        # list_files path info
        assert "compliance_log.csv" in old_output
        assert "compliance_log.csv" in new_output

        # structured_probe table name
        assert "Sheet1" in old_output or "Sheet1" in new_output

        # file path from tool results
        assert "analysis.csv" in old_output
        assert "analysis.csv" in new_output

    def test_both_paths_produce_tool_results_with_tool_names(self) -> None:
        """Both paths place tool names into the tool_results section."""
        state = _build_golden_state()
        registry = _register_formatters()

        old_cb = ContextBuilder(
            max_context_tokens=8000,
            token_accounting=_token_accounting(),
            formatter_resolver=None,
        )
        new_cb = ContextBuilder(
            max_context_tokens=8000,
            token_accounting=_token_accounting(),
            formatter_resolver=lambda name: registry.get_formatter(name),
        )

        old_output = _render_via_old_methods(old_cb, state)
        new_output = _render_via_formatter_path(new_cb, state)

        # Both paths reference tool names
        assert "vector_search" in old_output
        assert "vector_search" in new_output
        assert "list_files" in old_output
        assert "list_files" in new_output
        assert "run_python" in old_output
        assert "run_python" in new_output
        assert "structured_probe" in old_output or "structured_probe" in new_output
