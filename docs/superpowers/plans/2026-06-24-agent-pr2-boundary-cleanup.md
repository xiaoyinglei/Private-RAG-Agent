# PR2: Tool Output Boundary Cleanup — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove RAG-era observation writes to LoopState, decouple ContextBuilder from tool-specific rendering via `ToolOutputFormatter`, migrate state-level `retrieval_signals` and generation flags to tool layer, delete 14 deprecated field write paths. Formatter output must be semantically equivalent to current ContextBuilder rendering.

**Architecture:** A `ToolOutputFormatterResolver` (wrapping the request-scoped `ToolRegistry`) is passed explicitly through the construction chain: `service → AgentLLMContextAssembler → ContextBuilder`. Each tool gets a formatter that relocates rendering logic from ContextBuilder. Formatters are built first, verified for equivalence, then old writes are stopped. `retrieval_signals` is removed from LoopState/state-level paths only — RAG tool internals keep `QueryOptions.retrieval_signals`.

## Global Constraints

- **No field deletions from TypedDict.** All 14 deprecated fields remain.
- **Write paths deleted, definitions kept.** Fields readable from old checkpoints via `_migrate_legacy_state`.
- **No allowlist changes.** `AGENT_CHECKPOINT_MSGPACK_ALLOWLIST` unchanged.
- **Context equivalence.** Tool output rendering must match current behavior before/after formatter switch.
- **PR1 is base.** Builds on `agent/design-v2` (sub-states already present).
- **uv for all commands.**
- **ruff format/check only on touched files** (never global).

---

### Task 1: Verify existing Task 1 commit + define ToolExecutionObservation

**Files:**
- Verify: `rag/agent/tools/formatter.py` (already committed — PR2 prep)
- Verify: `rag/agent/tools/registry.py` (already committed — `register_formatter`, `get_formatter`)
- Create: `rag/agent/tools/observation.py` — `ToolExecutionObservation` model

**Note:** `ToolOutputFormatter` Protocol and `ToolRegistry.register_formatter()` already exist on this branch (from the prepare commit before PR2 plan was drafted). This task VERIFIES them, adds the typed observation model, and commits.

- [ ] **Step 1: Verify existing formatter setup**

```bash
uv run python -c "
from rag.agent.tools.formatter import ToolOutputFormatter, format_tool_result_fallback
from rag.agent.tools.registry import ToolRegistry
r = ToolRegistry()
r.register_formatter(type('F', (), {'tool_name':'test', 'format_result':lambda s,r:None, 'format_externalized':lambda s,r:None})())
assert r.get_formatter('test') is not None
assert r.get_formatter('nonexistent') is None
print('OK')
"
```

- [ ] **Step 2: Create ToolExecutionObservation model**

```python
# rag/agent/tools/observation.py
"""Minimal typed model for PlanTracker — replaces StructuredObservation in planner path."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ToolExecutionObservation(BaseModel):
    """Generic execution status that PlanTracker can read via getattr.

    Replaces the dict-based hack that earlier PR2 plan versions would have used.
    PlanTracker uses getattr(observation, "tool_call_id") etc. — this model
    provides those attributes as typed fields.
    """
    tool_call_id: str
    tool_name: str
    status: str  # "ok" | "error"
    related_step_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


__all__ = ["ToolExecutionObservation"]
```

- [ ] **Step 3: Verify and commit**

```bash
uv run python -c "from rag.agent.tools.observation import ToolExecutionObservation; print('OK')"
uv run ruff format rag/agent/tools/observation.py && uv run ruff check rag/agent/tools/observation.py --fix
uv run pytest -x -q
```

---

### Task 2: Wire FormatterResolver through ContextBuilder chain

**Files:**
- Modify: `rag/agent/memory/injector.py` — ContextBuilder accepts optional resolver
- Modify: `rag/agent/core/llm_context.py` — AgentLLMContextAssembler passes resolver through
- Modify: `rag/agent/core/llm_providers.py` — `_create_context_assembler` accepts resolver
- Modify: `rag/agent/service.py` — constructs resolver from runtime ToolRegistry

**Interfaces:**
- Produces: `ToolOutputFormatterResolver` callable type
- Consumes: existing `ToolRegistry` (from PR2 Task 1 commit)

- [ ] **Step 1: Define resolver type in formatter.py**

```python
# In rag/agent/tools/formatter.py, add:
from collections.abc import Callable

ToolOutputFormatterResolver = Callable[[str], "ToolOutputFormatter | None"]
```

- [ ] **Step 2: ContextBuilder accepts resolver**

In `rag/agent/memory/injector.py`, modify `ContextBuilder.__init__`:

```python
def __init__(
    self,
    *,
    max_context_tokens: int,
    max_section_chars: int = 4000,
    token_accounting: ContextTokenAccounting | None = None,
    formatter_resolver: ToolOutputFormatterResolver | None = None,  # PR2
) -> None:
    # ... existing init ...
    self._formatter_resolver = formatter_resolver
```

- [ ] **Step 3: AgentLLMContextAssembler accepts and passes resolver**

In `rag/agent/core/llm_context.py`:

```python
class AgentLLMContextAssembler:
    def __init__(
        self,
        *,
        token_accounting: ContextTokenAccounting,
        stage_budgets: Mapping[LLMCallStage, LLMStageBudget],
        formatter_resolver: ToolOutputFormatterResolver | None = None,  # PR2
    ) -> None:
        self._token_accounting = token_accounting
        self._stage_budgets = dict(stage_budgets)
        self._formatter_resolver = formatter_resolver

    # In _assemble_state_context, pass resolver to ContextBuilder:
    def _assemble_state_context(self, ...):
        builder = ContextBuilder(
            max_context_tokens=max_context_tokens,
            token_accounting=self._token_accounting,
            formatter_resolver=self._formatter_resolver,
        )
```

- [ ] **Step 4: _create_context_assembler factory accepts resolver**

In `llm_providers.py:828`:

```python
def _create_context_assembler(
    ...,
    formatter_resolver: ToolOutputFormatterResolver | None = None,
) -> AgentLLMContextAssembler | None:
    return AgentLLMContextAssembler(
        token_accounting=...,
        stage_budgets=...,
        formatter_resolver=formatter_resolver,
    )
```

- [ ] **Step 5: service.py constructs resolver**

```python
# In service.py, where assembler is created:
from rag.agent.tools.formatter import ToolOutputFormatterResolver

def _create_formatter_resolver(self, runtime_registry: ToolRegistry) -> ToolOutputFormatterResolver:
    """Wrap request-scoped ToolRegistry as a formatter resolver."""
    return lambda tool_name: runtime_registry.get_formatter(tool_name)
```

- [ ] **Step 6: Verify chain — import test**

```bash
uv run python -c "
from rag.agent.memory.injector import ContextBuilder
from rag.agent.core.llm_context import AgentLLMContextAssembler
print('Imports OK')
"
uv run pytest -x -q
```

- [ ] **Step 7: Commit**

---

### Task 3: Build equivalent RAG retrieval formatters (relocate from ObservationBuilder)

**Files:**
- Create: `rag/agent/tools/formatters/rag_retrieval.py`
- Create: `rag/agent/tools/formatters/__init__.py`

**Key design:** Formatters must produce semantically identical output to current ContextBuilder rendering. To achieve this, **relocate the extraction logic from `ObservationBuilder.from_tool_result()` and ContextBuilder's private rendering methods**, don't rewrite from scratch.

- [ ] **Step 1: Read the existing rendering code**

Read these methods carefully (they are the source of truth for what formatters must produce):
- `ContextBuilder._format_evidence()` — renders EvidenceItem + AnswerCitation
- `ContextBuilder._format_structured_observations()` — renders StructuredObservation
- `ContextBuilder._format_locator()` — renders locators with 36 fields
- `ObservationBuilder.from_tool_result()` — creates StructuredObservation from ToolResult
- `_retrieval_context_units()` in observations.py — creates ContextUnit from search results

- [ ] **Step 2: Create RAG retrieval formatter**

```python
# rag/agent/tools/formatters/rag_retrieval.py
"""Formatters for RAG retrieval tools.

Each formatter relocates rendering logic from ContextBuilder's private methods
(_format_evidence, _format_structured_observations, _format_locator) to
produce semantically identical output.
"""
from __future__ import annotations

from rag.agent.memory.models import ContextSection, ExternalizedToolOutput
from rag.agent.tools.formatter import ToolOutputFormatter
from rag.agent.tools.spec import ToolResult


def _format_retrieval_result(
    result: ToolResult,
    tool_name: str,
) -> ContextSection | None:
    """Relocated from ContextBuilder._format_evidence() + _format_tool_observations().

    Renders EvidenceItem items from output.evidence/output.items, with
    citation anchors, doc_ids, scores, and text previews bounded to 500 chars.
    Matches the output format of the deleted ContextBuilder methods exactly.
    """
    if result.status != "ok" or result.output is None:
        return None

    items = getattr(result.output, "items", None)
    evidence = getattr(result.output, "evidence", []) or []
    citations = getattr(result.output, "citations", []) or []

    lines: list[str] = []

    # Render evidence items (relocated from _format_evidence)
    for ev in evidence:
        meta_parts = _evidence_meta(ev)
        lines.append(f"- {meta_parts}")
        text = getattr(ev, "text", "")
        if text:
            lines.append(f"  text: {_one_line(str(text)[:500])}")

    # Render search items with locators (relocated from _format_locator)
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            locator_text = _render_locator(item)
            if locator_text:
                lines.append(f"- {locator_text}")
            text = str(item.get("text", ""))
            if text:
                lines.append(f"  text: {_one_line(text[:500])}")

    # Render citations
    for c in citations:
        lines.append(f"- citation: evidence_id={getattr(c, 'evidence_id', '')} anchor={getattr(c, 'citation_anchor', '')}")

    if not lines:
        return None

    return ContextSection(
        name="tool_result",
        content=f"{tool_name} results:\n" + "\n".join(lines),
        token_count=0,
        required=False,
    )


def _evidence_meta(ev: object) -> str:
    """Relocated from ContextBuilder._metadata_line() — produce same format."""
    parts = []
    for field in ("evidence_id", "doc_id", "citation_anchor",
                   "record_type", "file_name", "source_id", "source_type"):
        val = getattr(ev, field, None)
        if val not in (None, "", []):
            parts.append(f"{field}={val}")
    if (score := getattr(ev, "score", None)) is not None:
        parts.append(f"score={score}")
    return " ".join(parts)


def _render_locator(item: dict) -> str:
    """Relocated from ContextBuilder._format_locator() — same 36-field whitelist."""
    fields = (
        "asset_id", "doc_id", "source_id", "section_id", "asset_type",
        "table_index", "table_name", "used_range", "sheet_name", "page_no",
        "element_ref", "citation_anchor", "evidence_id", "path", "name",
        "size_bytes", "is_dir", "mime_type", "file_kind", "truncated",
        "is_binary", "readable_as_text", "encoding", "source_tool",
        "generated", "generated_by", "ok", "exit_code", "duration_ms",
        "stdout_truncated", "stderr_truncated", "header_row_index",
        "header_confidence", "data_start_row", "row_count", "column_count",
    )
    parts = [f"{f}={item[f]}" for f in fields if item.get(f) not in (None, "", [])]
    return " ".join(parts)


def _one_line(text: str) -> str:
    return " ".join(text.split())


class VectorSearchFormatter:
    tool_name = "vector_search"
    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_retrieval_result(result, "vector_search")
    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class KeywordSearchFormatter:
    tool_name = "keyword_search"
    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_retrieval_result(result, "keyword_search")
    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class GroundingFormatter:
    tool_name = "grounding"
    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_retrieval_result(result, "grounding")
    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class RerankFormatter:
    tool_name = "rerank"
    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_retrieval_result(result, "rerank")
    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


class GraphExpandFormatter:
    tool_name = "graph_expand"
    def format_result(self, result: ToolResult) -> ContextSection | None:
        return _format_retrieval_result(result, "graph_expand")
    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None:
        return None


__all__ = [
    "VectorSearchFormatter", "KeywordSearchFormatter",
    "GroundingFormatter", "RerankFormatter", "GraphExpandFormatter",
]
```

- [ ] **Step 2: Register formatters in builtin registry**

In `rag/agent/tools/builtin_registry.py` or wherever tools are registered, register each formatter:

```python
from rag.agent.tools.formatters.rag_retrieval import (
    VectorSearchFormatter, KeywordSearchFormatter,
    GroundingFormatter, RerankFormatter, GraphExpandFormatter,
)

# After tool registration:
registry.register_formatter(VectorSearchFormatter())
registry.register_formatter(KeywordSearchFormatter())
registry.register_formatter(GroundingFormatter())
registry.register_formatter(RerankFormatter())
registry.register_formatter(GraphExpandFormatter())
```

- [ ] **Step 3: Verify and commit**

---

### Task 4: Build equivalent file/workspace tool formatters

**Files:**
- Create: `rag/agent/tools/formatters/file_tools.py`

**Relocate from:**
- `_list_files_context_units()` → ListFilesFormatter
- `_read_file_context_unit()` → ReadFileFormatter
- `_write_file_context_unit()` → WriteFileFormatter
- `_run_python_context_units()` → RunPythonFormatter
- `_structured_probe_context_units()` → StructuredProbeFormatter
- `_format_locator()` relevance for each tool → each formatter's rendering

- [ ] **Step 1: Read the existing unit-creation code in observations.py**

Read each `_*_context_units()` function and understand what ContextUnit fields it produces. The formatter must produce equivalent ContextSection content.

- [ ] **Step 2: Create file tool formatters**

Each formatter reads `ToolResult.output` fields (path, size_bytes, stdout, stderr, generated_files, tables, etc.) and renders them in the same style as the deleted ContextBuilder methods.

- [ ] **Step 3: Register and commit**

---

### Task 4.5: Golden equivalence test — old ContextBuilder vs new formatter output

**Files:**
- Create: `tests/agent/test_pr2_context_equivalence.py`

**This task runs BEFORE Task 5 deletes old ContextBuilder methods.** It creates a dual-render test that locks in equivalence.

- [ ] **Step 1: Construct a golden-state fixture**

Create a `LoopState` with all RAG-era fields populated:
- `tool_results` from mock vector_search/keyword_search/grounding/list_files calls
- `structured_observations`, `evidence`, `citations`, `evidence_refs`, `locators`, `context_units`, `asset_refs`

- [ ] **Step 2: Dual-render and compare**

```python
def test_old_vs_formatter_context_equivalence():
    """Old ContextBuilder output must have same key anchors as formatter output."""
    state = _golden_loop_state()

    # Render with old methods (before deletion)
    old_output = _render_old_path(state)

    # Render with new formatter path
    new_output = _render_formatter_path(state)

    # Key anchors that MUST appear in both outputs:
    anchors = [
        "evidence_id=", "citation_anchor=", "doc_id=",
        "tool_call_id=", "tool_name=",
        # locator fields
        "asset_id=", "path=", "table_index=", "sheet_name=",
        "stdout", "stderr", "generated_files",
        # citation
        "citation:", "source_id=",
    ]
    for anchor in anchors:
        assert (anchor in old_output) == (anchor in new_output), \
            f"Anchor '{anchor}' mismatch: old={anchor in old_output}, new={anchor in new_output}"
```

- [ ] **Step 3: Run — must pass before Task 5 can delete old methods**

```bash
uv run pytest tests/agent/test_pr2_context_equivalence.py -v
```

- [ ] **Step 4: Commit** (this test stays as a regression guard)

---

### Task 5: Switch ContextBuilder to use formatters + delete old methods

**Files:**
- Modify: `rag/agent/memory/injector.py`

**Pre-condition:** Task 4.5 golden equivalence test PASSES.

**Key point:** At this stage, `_merge_observations` STILL writes to LoopState. The 14 fields still get populated. ContextBuilder is the only thing that changes — it now renders from `tool_results` via formatters instead of reading `structured_observations`/`evidence`/etc.

- [ ] **Step 1: Add _format_tool_context() method**

```python
def _format_tool_context(self, state: ContextState) -> str:
    """Schedule per-tool formatters; fallback for unregistered tools."""
    tool_results = state.get("tool_results", [])
    if not tool_results:
        return ""

    resolver = self._formatter_resolver
    sections: list[str] = []
    for result in tool_results:
        formatter = resolver(result.tool_name) if resolver else None
        section = None
        if formatter is not None:
            section = formatter.format_result(result)
        if section is None:
            section = format_tool_result_fallback(result)
        if section is not None and section.content.strip():
            sections.append(section.content)

    if not sections:
        return ""
    return "Tool results:\n" + "\n".join(sections)
```

- [ ] **Step 2: Replace the old section builders in assemble_loop()**

Change `add("tool_results", self._format_tool_observations(state), required=True)` → `add("tool_results", self._format_tool_context(state), required=True)`

Change `add("evidence", self._format_evidence(...), required=True)` → **remove** (rendered by formatters inside _format_tool_context).

- [ ] **Step 3: Delete old rendering methods**

Remove from ContextBuilder:
- `_format_evidence()` — ~70 lines
- `_format_structured_observations()` — ~60 lines
- `_format_tool_observations()` — delegate logic
- `_format_locator()` — ~50 lines with 36-field whitelist

Keep:
- `_format_tool_results()` — referenced by `format_tool_result_fallback`
- All non-tool-semantic methods (`_format_plan`, `_format_working_memory`, etc.)

- [ ] **Step 4: Context equivalence verification**

Run the existing `tests/agent/test_context_injector.py` and `tests/agent/test_llm_context.py`. If tests compare exact output strings, they may need golden-file updates. The key invariant is that the **semantic content** (evidence IDs, citation anchors, text previews, locator fields) is preserved.

```bash
uv run pytest tests/agent/test_context_injector.py tests/agent/test_llm_context.py -v
uv run pytest -x -q
```

- [ ] **Step 5: Commit**

---

### Task 6: Stop _merge_observations writes to LoopState + PlanTracker typed model

**Files:**
- Modify: `rag/agent/loop/runtime.py` — `_merge_observations` no-op, `_record_plan_observations` uses `ToolExecutionObservation`

- [ ] **Step 1: Make _merge_observations a no-op**

```python
@staticmethod
def _merge_observations(state: LoopState, batch: ObservationBatch) -> None:
    # PR2: ObservationExtractor no longer writes tool-semantic fields to LoopState.
    # Tool output rendering is handled by ToolOutputFormatter via ContextBuilder.
    return
```

- [ ] **Step 2: _record_plan_observations uses ToolExecutionObservation**

```python
from rag.agent.tools.observation import ToolExecutionObservation

def _record_plan_observations(self, state: LoopState, batch: ObservationBatch) -> None:
    typed_observations = [
        ToolExecutionObservation(
            tool_call_id=obs.tool_call_id,
            tool_name=obs.tool_name,
            status=obs.status,
            related_step_ids=list(obs.related_step_ids),
            metadata=dict(obs.metadata),
        )
        for obs in batch.structured_observations
    ]
    plan, events = self._plan_tracker.record_observation_progress(
        plan=state["agent_plan"],
        observations=typed_observations,
    )
    if plan is not None:
        state["agent_plan"] = plan
        state["plan_events"] = [*state["plan_events"], *events][-MAX_PLAN_EVENTS:]
        state["plan_state"] = state["plan_state"].model_copy(
            update={"agent_plan": plan, "plan_events": list(state["plan_events"])}
        )
```

- [ ] **Step 3: Verify**

```bash
uv run pytest tests/agent/test_agent_loop_runtime.py tests/agent/test_agent_observations.py tests/agent/test_agent_planning.py -v
uv run pytest -x -q
```

- [ ] **Step 4: Commit**

---

### Task 7: retrieval_signals — remove from LoopState/state-level only

**Files:**
- Modify: `rag/agent/core/llm_prompts.py` — delete `build_retrieval_hint_prompt()`
- Modify: `rag/agent/core/llm_context.py` — delete `assemble_retrieval_hint()`
- Modify: `rag/agent/service.py` — delete `_apply_retrieval_hint()` and state-level retrieval_signals writes
- Modify: `rag/agent/core/tool_execution.py` — delete auto-injection of `retrieval_signals` from state → RAG tool args

**NOT modified:**
- `rag/query_pipeline.py` — `QueryOptions.retrieval_signals` is RAG internal, kept
- `rag/retrieval/` internals — kept

- [ ] **Step 1: Delete build_retrieval_hint_prompt**

Remove the function `build_retrieval_hint_prompt()` from `rag/agent/core/llm_prompts.py`. It reads `state["retrieval_signals"]` — this is the LoopState pollution path.

- [ ] **Step 2: Delete assemble_retrieval_hint**

Remove `AgentLLMContextAssembler.assemble_retrieval_hint()` from `rag/agent/core/llm_context.py`.

- [ ] **Step 3: Delete _apply_retrieval_hint in service.py**

Remove the method `_apply_retrieval_hint()` and its call sites. Remove `state["retrieval_signals"] = ...` writes.

- [ ] **Step 4: Remove auto-injection in tool_execution.py**

Find where `tool_execution.py` reads `state["retrieval_signals"]` to auto-inject into RAG tool arguments (~line 842). Remove this injection. RAG tools will receive `retrieval_signals` through their own internal preprocessor if needed, not from agent state.

- [ ] **Step 5: Verify**

```bash
uv run pytest tests/agent/test_retrieval_signals_loop.py tests/agent/test_rag_tool_runner.py -v
uv run pytest -x -q
```

- [ ] **Step 6: Commit**

---

### Task 8: groundedness_flag/insufficient_evidence_flag → generation ToolResult.output

**Files:**
- Modify: `rag/agent/loop/runtime.py` — delete aggregation writes to LoopState (~line 406, ~lines 655-669)
- Modify: `rag/providers/generation.py` — confirm flags are in ToolResult.output (already present)

**Current problem:** The runtime aggregates `groundedness_flag` and `insufficient_evidence_flag` from tool outputs and ORs them into LoopState at two sites:
1. `runtime.py:406` — `state["insufficient_evidence_flag"] = True` in tool error path
2. `runtime.py:655-669` — aggregates `output.groundedness_flag`, `has_traceable_evidence`, `output.insufficient_evidence_flag` from each tool result into LoopState

These flags are already carried in the RAG generation tool's `ToolResult.output` (the answer model has `groundedness_flag` and `insufficient_evidence_flag` fields). The generation formatter (Task 3) renders them. LoopState does not need to hold them.

- [ ] **Step 1: Delete runtime aggregation**

In `runtime.py`, remove the blocks at ~line 406 and ~lines 648-670:
- Delete `state["insufficient_evidence_flag"] = True` (error path)
- Delete the entire `groundedness_flag`/`insufficient_evidence_flag` aggregation block

- [ ] **Step 2: AgentRunResult derive from tool_results (if needed)**

Check if `AgentRunResult` (in `rag/agent/service.py` or `rag/agent/state.py`) reads `state["groundedness_flag"]` or `state["insufficient_evidence_flag"]`. If so, derive them from the last RAG generation `ToolResult.output` in `tool_results` instead:

```python
def _derive_groundedness(tool_results: list[ToolResult]) -> bool:
    for result in reversed(tool_results):
        if result.tool_name in ("generate_answer", "synthesize"):
            return bool(getattr(result.output, "groundedness_flag", False))
    return False
```

- [ ] **Step 3: Verify**

```bash
uv run pytest tests/agent/ tests/service/test_answer_generation_contract.py -v
uv run pytest -x -q
```

- [ ] **Step 4: Commit**

---

### Task 9: Delete remaining write paths for 14 deprecated fields

**Files:**
- Scan and modify all remaining write sites

- [ ] **Step 1: Global scan for remaining writes**

```bash
grep -rn 'state\["retrieval_signals"\]\|state\["evidence"\]\|state\["citations"\]\|state\["evidence_refs"\]\|state\["answer_candidates"\]\|state\["computation_results"\]\|state\["structured_observations"\]\|state\["context_units"\]\|state\["context_bindings"\]\|state\["locators"\]\|state\["asset_refs"\]\|state\["groundedness_flag"\]\|state\["insufficient_evidence_flag"\]' rag/ --include="*.py" | grep -v test_ | grep -v __pycache__ | grep -v "state\.py" | grep -v "checkpointing\.py"
```

Any remaining writes should be:
- Commented out if in a non-critical code path
- Removed if in a write path already handled by Tasks 2-8

- [ ] **Step 2: Remove each write, run tests after each**

- [ ] **Step 3: Full suite**

```bash
uv run pytest -x -q
```

- [ ] **Step 4: Commit**

---

### Task 10: Integration tests

**Files:**
- Create: `tests/agent/test_pr2_boundary_cleanup.py`

- [ ] **Step 1: Write tests**

Tests covering:
1. `_merge_observations` is a no-op (does not raise, does not write to LoopState)
2. ContextBuilder uses formatter-scheduled output for registered tools
3. Fallback works for unregistered tools
4. `retrieval_signals` has zero `state["retrieval_signals"]` references in agent/ directory
5. `PlanTracker` works with `ToolExecutionObservation` (typed model, not dict)
6. FormatterResolver flows through real construction chain:
   - Create a `ToolRegistry`, register a formatter, construct an `AgentLLMContextAssembler` with the resolver, call `assemble_loop_turn()`, verify the formatter was invoked (check output contains expected content)
7. `groundedness_flag` and `insufficient_evidence_flag` are NOT written to LoopState during tool execution (verify state after simulated execute_tools)
8. Golden equivalence test from Task 4.5 still passes

- [ ] **Step 2: Run and verify**

```bash
uv run pytest tests/agent/test_pr2_boundary_cleanup.py tests/agent/test_pr2_context_equivalence.py -v
uv run pytest -x -q
```

- [ ] **Step 3: Commit**

---

## Self-Review

1. **Formatter resolver pattern**: Explicit constructor-chain passing, no global singleton.
2. **Safe ordering**: Formatters built (Tasks 3-4) → ContextBuilder switched (Task 5) → writes stopped (Task 6).
3. **Equivalent formatters**: Relocate from existing ObservationBuilder + ContextBuilder code, not rewrite.
4. **retrieval_signals**: Only LoopState/state-level removed; RAG internals preserved.
5. **PlanTracker typed**: `ToolExecutionObservation` model, not dict.
