# PR3: Pending Single-Track + Final Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge pending dual-track into one loop-owned `PendingToolCall`, introduce `ToolCallLedger` for transcript rebuild, move `loop_messages` to provider-derived transcript, delete the remaining deprecated LoopState field definitions, clean up checkpoint allowlist only where types are no longer reachable, and set a compat-layer deprecation deadline.

**Architecture:** `PendingToolCall(plan, status, approval_request_id, operation_id, summary)` replaces both `list[ToolCallPlan]` pending state and the old PR0 `PendingToolCall`. `ToolCallLedger` stores `ToolCallLedgerEntry(plan, turn, sequence)` for every model-requested call - not pending state - to avoid state-machine confusion. Provider layer rebuilds native transcript from `messages + tool_call_ledger + tool_results` each turn.

**Hard boundary:** `PendingToolCall` is loop control state only. `ToolExecutionService`, `ToolBatchRequest`, and `ToolBatchResult` must continue to use `ToolCallPlan`. The execution boundary should not understand approval/pending lifecycle.

## Global Constraints

- **No runtime behavior regression.** Full test suite must pass.
- **Delete definitions, not just write paths.** Remaining deprecated fields are removed from TypedDict and factory only after all production readers are migrated.
- **Allowlist cleanup only after roundtrip test.** Confirm live `ToolResult.output` serialization still works before removing any allowlist entry.
- **uv for all commands.**
- **ruff format/check only on touched files.**
- **RAG remains a tool.** Keep `rag.tools.*Input.retrieval_signals` and RAG `QueryOptions.retrieval_signals`; remove only state-level retrieval signals.

---

### Task 1: Define PendingToolCall v2 + ToolCallLedger + ToolCallLedgerEntry

**Files:**
- Modify: `rag/agent/loop/state.py` — new PendingToolCall, ToolCallLedgerEntry, ToolCallLedger models
- Modify: `rag/agent/loop/state.py` — create_loop_state initializes tool_call_ledger

**Interfaces:**
- Produces: `PendingToolCall(plan: ToolCallPlan, status: str, approval_request_id: str|None, operation_id: str|None, summary: str|None)`
- Produces: `ToolCallLedgerEntry(plan: ToolCallPlan, turn: int, sequence: int)`
- Produces: `ToolCallLedger(entries: list[ToolCallLedgerEntry], max_entries: int)` with an append helper that dedupes by `tool_call_id`

- [ ] **Step 1: Define the models**

```python
# In rag/agent/loop/state.py, before LoopState TypedDict:
from collections.abc import Iterable

class PendingToolCall(BaseModel):
    """Single canonical pending tool call. Replaces ToolCallPlan-as-pending + old PendingToolCall."""
    plan: ToolCallPlan
    status: Literal["pending", "approved", "denied", "running", "completed", "failed"]
    approval_request_id: str | None = None
    operation_id: str | None = None
    summary: str | None = None

    @property
    def tool_call_id(self) -> str:
        return self.plan.tool_call_id

    @property
    def tool_name(self) -> str:
        return self.plan.tool_name


class ToolCallLedgerEntry(BaseModel):
    """Transcript source for one tool call — no runtime state, just plan + position."""
    plan: ToolCallPlan
    turn: int
    sequence: int


class ToolCallLedger(BaseModel):
    """Bounded ledger of all tool calls for native transcript rebuild.
    Only cleaned when entries are no longer needed for transcript reconstruction.
    """
    entries: list[ToolCallLedgerEntry] = Field(default_factory=list)
    max_entries: int = 128

    def append_plans(self, plans: Iterable[ToolCallPlan], *, turn: int) -> None:
        """Record model-requested calls idempotently; do not store pending state."""
        existing = {entry.plan.tool_call_id for entry in self.entries}
        for plan in plans:
            if plan.tool_call_id in existing:
                continue
            self.entries.append(
                ToolCallLedgerEntry(
                    plan=plan,
                    turn=turn,
                    sequence=len(self.entries),
                )
            )
            existing.add(plan.tool_call_id)
```

- [ ] **Step 2: Add tool_call_ledger to LoopState TypedDict and checkpoint allowlist**

```python
    # In LoopState:
    pending_tool_calls: list[PendingToolCall]       # single-track
    tool_call_ledger: ToolCallLedger                # bounded transcript source
```

Add to `AGENT_CHECKPOINT_MSGPACK_ALLOWLIST`:
```python
("rag.agent.loop.state", "PendingToolCall"),
("rag.agent.loop.state", "ToolCallLedger"),
("rag.agent.loop.state", "ToolCallLedgerEntry"),
```

- [ ] **Step 3: Update create_loop_state**

```python
    "pending_tool_calls": [
        PendingToolCall(plan=call, status="pending")
        for call in pending_tool_calls
    ],
    "tool_call_ledger": ToolCallLedger(),
```

Keep the public `create_loop_state(..., pending_tool_calls: Iterable[ToolCallPlan])` argument for compatibility; convert at the factory boundary.

- [ ] **Step 4: Verify and commit**

```bash
uv run python -c "from rag.agent.loop.state import PendingToolCall, ToolCallLedger, ToolCallLedgerEntry; print('OK')"
uv run pytest tests/agent/test_loop_state.py tests/agent/test_loop_checkpointing.py -q
```

---

### Task 2: Merge pending dual-track in runtime

**Files:**
- Modify: `rag/agent/loop/runtime.py` — all pending_tool_calls reads/writes use new PendingToolCall

**Key changes:**
1. Where `state["pending_tool_calls"]` is populated from `ModelTurn.tool_calls`, wrap each `ToolCallPlan` in `PendingToolCall(plan=tc, status="pending")`
2. Where `pending_loop_tool_calls` was read, remove — use only `pending_tool_calls`
3. Delete the old `PendingToolCall` type (PR0 version) from `rag/agent/core/messages.py`
4. Update `_execute_pending_tools` to iterate `PendingToolCall.plan` for execution
5. Keep `ToolBatchRequest.calls` and `ToolBatchResult.pending_tool_calls` as `ToolCallPlan`; wrap/unwrap only at the loop boundary

- [ ] **Step 1: Read all pending_tool_calls and pending_loop_tool_calls references in runtime.py**

```bash
grep -n "pending_tool_calls\|pending_loop_tool_calls" rag/agent/loop/runtime.py
```

- [ ] **Step 2: Migrate each site**

For each reference:
- `state["pending_tool_calls"]` → keep as-is but ensure items are `PendingToolCall` instances
- `state["pending_loop_tool_calls"]` → delete (merge into pending_tool_calls)
- When calling `ToolExecutionService`, pass `tuple(pending.plan for pending in state["pending_tool_calls"])`
- When applying `ToolBatchResult.pending_tool_calls`, wrap returned `ToolCallPlan` values back into `PendingToolCall(status="pending")`

Do **not** change `ToolBatchRequest` or `ToolBatchResult` to contain `PendingToolCall`. That would leak loop lifecycle state into the tool execution boundary.

- [ ] **Step 3: Update callers outside runtime**

```bash
grep -rn "pending_loop_tool_calls\|pending_tool_calls" rag/agent/ --include="*.py" | grep -v test_ | grep -v __pycache__
```

Update `service.py`, `llm_providers.py`, `compactor.py`, `checkpointing.py`, `llm_prompts.py`, and `memory/injector.py` to read `pending.plan.*` instead of direct `ToolCallPlan` attributes.

Add small local helpers instead of repeating attribute compatibility logic:
```python
def pending_plan(call: PendingToolCall | ToolCallPlan) -> ToolCallPlan:
    return call.plan if isinstance(call, PendingToolCall) else call

def pending_tool_call_id(call: PendingToolCall | ToolCallPlan) -> str:
    return pending_plan(call).tool_call_id
```
Use these only during migration/checkpoint compatibility; new writes should always produce `PendingToolCall`.

- [ ] **Step 4: Delete old PendingToolCall from messages.py**

Remove the PR0-era `PendingToolCall` class from `rag/agent/core/messages.py`. Update all imports.

Keep `ModelMessage` and `ToolCall` in `core/messages.py`; they are still provider-neutral transcript/message types even after `loop_messages` leaves LoopState.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest tests/agent/test_agent_loop_runtime.py tests/agent/test_loop_checkpointing.py -v
uv run pytest tests/agent/test_agent_service.py tests/agent/test_context_injector.py -q
```

---

### Task 3: Populate ToolCallLedger + move loop_messages to derived transcript

**Files:**
- Modify: `rag/agent/loop/runtime.py` — append to tool_call_ledger when the model schedules tool calls
- Modify: `rag/agent/core/llm_providers.py` — rebuild transcript from ledger + tool_results

**Key changes:**

1. In runtime, when `turn.action == "execute"` is accepted, append model-requested plans to `tool_call_ledger` **before** execution:
```python
if turn.action == "execute":
    state["tool_call_ledger"].append_plans(
        turn.tool_calls,
        turn=state["iteration"],
    )
    state["pending_tool_calls"] = [
        PendingToolCall(plan=call, status="pending")
        for call in turn.tool_calls
    ]
```

Reason: the ledger is the durable source for the assistant tool-call message. If it is populated only after execution, approval pauses, reconciliation, or excess pending calls can lose original tool arguments.

2. Bounded trim lives on `ToolCallLedger`, not inline in runtime:
```python
def trim(self, *, active_tool_call_ids: set[str]) -> None:
    while len(self.entries) > self.max_entries:
        for index, entry in enumerate(self.entries):
            if entry.plan.tool_call_id not in active_tool_call_ids:
                self.entries.pop(index)
                break
        else:
            break
```

Call trim after tool execution and after pending state changes.

3. In `llm_providers.py`, `_next_turn_with_tools()`: instead of reading `state["loop_messages"]`, rebuild provider-neutral transcript from:
```python
def _rebuild_tool_transcript(state: LoopState) -> list[ModelMessage]:
    """Rebuild native tool-call transcript from ledger + tool_results."""
    transcript: list[ModelMessage] = []
    results_by_id = {result.tool_call_id: result for result in state["tool_results"]}
    for entry in state["tool_call_ledger"].entries:
        result = results_by_id.get(entry.plan.tool_call_id)
        transcript.append(
            ModelMessage(
                role="assistant",
                content="",
                tool_calls=(
                    ToolCall(
                        id=entry.plan.tool_call_id,
                        name=entry.plan.tool_name,
                        input=dict(entry.plan.arguments),
                    ),
                ),
            )
        )
        if result is not None:
            transcript.append(
                ModelMessage(
                    role="tool",
                    tool_call_id=result.tool_call_id,
                    content=_tool_result_content(result),
                )
            )
    return transcript
```

`_tool_result_content()` must reuse the PR2 formatter/fallback path and handle `ExternalizedToolOutput` before `formatter.format_result()` so externalized refs remain visible.

4. Delete `state["loop_messages"]` writes. Delete `state["tool_result_store"]` writes.
5. Remove `loop_messages` and `tool_result_store` from LoopState TypedDict and create_loop_state.

- [ ] **Step 3: Update _migrate_legacy_state for loop_messages**

```python
# In _migrate_legacy_state:
if state.get("loop_messages"):
    # Old checkpoints may have loop_messages but may not have a ledger.
    # Do not keep loop_messages in LoopState; emit a diagnostic if transcript
    # cannot be rebuilt exactly from tool_call_ledger + tool_results.
    state.setdefault("runtime_diagnostics", []).append(
        RuntimeDiagnostic(
            code="legacy_loop_messages_dropped",
            component="checkpoint_migration",
            message="Old loop_messages were dropped; transcript is rebuilt from tool_call_ledger and tool_results.",
            severity="warning",
        )
    )
state.pop("loop_messages", None)
state.pop("tool_result_store", None)
```

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/agent/test_llm_providers.py tests/agent/test_agent_loop_runtime.py -v
uv run pytest tests/agent/test_pr2_boundary_cleanup.py tests/agent/test_pr2_context_equivalence.py -q
```

---

### Task 4: Delete remaining deprecated field definitions + compat layer deprecation

**Files:**
- Modify: `rag/agent/loop/state.py` — remove the remaining 12 deprecated fields from TypedDict and create_loop_state
- Modify: `rag/agent/state.py` — add DeprecationWarning

**Remaining fields to delete from TypedDict and create_loop_state:**
```text
retrieval_signals, retrieval_signals_debug,
evidence, citations, evidence_refs,
answer_candidates, computation_results,
structured_observations, context_units, context_bindings,
locators, asset_refs
```
`groundedness_flag` and `insufficient_evidence_flag` were already removed in PR2, so PR3 deletes the remaining 12 field definitions.

- [ ] **Step 0: Migrate all production readers before deleting fields**

Required replacements:
- `AgentRunResult.from_loop_result`: derive `evidence` and `citations` from `tool_results`, not LoopState fields.
- `GoalContractStopHook`: already derives evidence/computation/context bindings from `tool_results`; keep tests covering this path.
- `binding_providers.py`: either derive bindings from `tool_results` or remove the unused provider. Do not leave reads from `state["answer_candidates"]` or old context units.
- `llm_providers.py`: remove the fallback read from `state["answer_candidates"]`.
- `runtime.py`: stop using `state["structured_observations"]` as `ObservationExtractor` seen state; use only the new results for plan progress or a dedicated bounded observation-progress ledger.
- `memory/compactor.py` and `memory/models.py`: remove capped channels and policy fields that only apply to deleted semantic state (`structured_observations`, `answer_candidates`, `computation_results`, `evidence_refs`, `evidence`, `citations`, `locators`, `context_units`).
- `ToolBatchRequest`: remove state-level `retrieval_signals`. RAG retrieval signals must be supplied through explicit tool input schemas, not copied from LoopState.

- [ ] **Step 1: Remove from LoopState TypedDict**

Remove the 12 field entries from the TypedDict. Keep in `_migrate_legacy_state` which still reads them from old checkpoints.

- [ ] **Step 2: Remove from create_loop_state factory**

Remove the 12 field initializations from the factory dict.

- [ ] **Step 3: Add DeprecationWarning to rag/agent/state.py**

```python
import warnings

warnings.warn(
    "rag.agent.state is deprecated. Import LoopState and create_loop_state "
    "directly from rag.agent.loop.state instead. "
    "This compat module will be removed after 2026-08-24.",
    DeprecationWarning,
    stacklevel=2,
)
```

- [ ] **Step 4: Update _migrate_legacy_state to drop these fields**

Already done partially — the function reads old flat fields into sub-states. Now add explicit drop:

```python
_DEPRECATED_STATE_FIELDS = frozenset({
    "retrieval_signals", "retrieval_signals_debug",
    "evidence", "citations", "evidence_refs",
    "answer_candidates", "computation_results",
    "structured_observations", "context_units",
    "context_bindings", "locators", "asset_refs",
})

def _migrate_legacy_state(raw: dict) -> LoopState:
    state = dict(raw)
    # ... existing sub-state population ...
    for key in _DEPRECATED_STATE_FIELDS:
        state.pop(key, None)  # drop after sub-states are populated
    return cast(LoopState, state)
```

If old checkpoints contain `retrieval_signals`, do not migrate them into the loop state. Keep only diagnostics if needed; RAG internals still receive retrieval signals through tool inputs.

- [ ] **Step 5: Global grep to verify no remaining references**

```bash
grep -rn 'state\["retrieval_signals"\]\|state\["evidence"\]\|state\["citations"\]\|state\["evidence_refs"\]\|state\["answer_candidates"\]\|state\["computation_results"\]\|state\["structured_observations"\]\|state\["context_units"\]\|state\["context_bindings"\]\|state\["locators"\]\|state\["asset_refs"\]' rag/agent/ --include="*.py" | grep -v test_ | grep -v state.py | grep -v checkpointing.py
```
Expected: zero production references remaining.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -x -q
uv run ruff check <touched files>
```

---

### Task 5: Checkpoint allowlist cleanup

**Files:**
- Modify: `rag/agent/core/checkpointing.py` — AGENT_CHECKPOINT_MSGPACK_ALLOWLIST

- [ ] **Step 1: Write allowlist roundtrip test**

Create a test that serializes and deserializes a representative live checkpoint payload through the current serde. It must include:
- `LoopState` with `PendingToolCall`, `ToolCallLedger`, `ToolResult`, `MemoryRef`, and `ExternalizedToolOutput`
- `ToolResult.output` for a RAG answer output containing `EvidenceItem` and `AnswerCitation`
- `ToolCallPlan.arguments` with primitive JSON values only

This test decides which allowlist entries are still reachable through live state.

```python
def test_live_loop_state_serde_after_pr3_cleanup():
    """Live PR3 LoopState payload must serialize without deprecated state fields."""
    serde = agent_checkpoint_serde()

    state = create_loop_state(...)
    state["pending_tool_calls"] = [PendingToolCall(...)]
    state["tool_call_ledger"].append_plans([...], turn=1)
    state["tool_results"] = [rag_answer_tool_result_with_evidence_and_citations()]

    restored = serde.loads_typed(serde.dumps_typed(state))
    assert "evidence" not in restored
    assert restored["tool_results"][0].output.evidence
```

- [ ] **Step 2: Remove only unreachable deprecated types from allowlist**

Candidate removals from `AGENT_CHECKPOINT_MSGPACK_ALLOWLIST`:
```python
("rag.agent.core.observations", "AnswerCandidate"),
("rag.agent.core.observations", "ComputationResult"),
("rag.agent.core.observations", "ContextBinding"),
("rag.agent.core.observations", "ContextUnit"),
("rag.agent.core.observations", "EvidenceRef"),
("rag.agent.core.observations", "ObservationBatch"),
("rag.agent.core.observations", "ObservationError"),
("rag.agent.core.observations", "StructuredObservation"),
```

Do **not** remove `AnswerCitation`, `EvidenceItem`, or `RetrievalSignals` just because the state fields are deleted. Keep them if they are reachable via `ToolResult.output`, persisted tool arguments, RAG tool schemas, or compatibility metadata.

Also remove the old `("rag.agent.core.messages", "PendingToolCall")` allowlist entry after the old class is deleted, and add the new loop-state pending/ledger entries from Task 1.

- [ ] **Step 3: Verify and commit**

```bash
uv run pytest tests/agent/test_checkpointing.py tests/agent/test_loop_checkpointing.py -v
uv run pytest -x -q
```

---

### Task 6: Integration tests + final cleanup

**Files:**
- Create: `tests/agent/test_pr3_final_cleanup.py`

**Tests:**

1. `test_pending_single_track_roundtrip` — create PendingToolCall, serialize/deserialize through checkpoint serde
2. `test_tool_call_ledger_bounded_fifo` — add inactive entries past the cap, verify only 128 remain; separately verify active pending calls are not trimmed
3. `test_transcript_rebuild_from_ledger` — populate ledger + tool_results, rebuild native transcript, verify assistant tool-call args match original
4. `test_deprecated_fields_not_in_loopstate` — verify `create_loop_state()` dict has zero of the remaining 12 deprecated keys
5. `test_legacy_checkpoint_fields_dropped` — load old checkpoint with deprecated fields, verify _migrate_legacy_state drops them
6. `test_compat_module_deprecation_warning` — import rag.agent.state, verify DeprecationWarning
7. `test_live_loop_state_serde_after_pr3_cleanup` — from Task 5
8. `test_tool_execution_boundary_still_uses_tool_call_plan` — `ToolBatchRequest.calls` and `ToolBatchResult.pending_tool_calls` remain `ToolCallPlan`
9. `test_native_provider_transcript_does_not_use_loop_messages` — native provider rebuilds from ledger + tool_results and preserves original tool arguments

- [ ] **Run and verify:**

```bash
uv run pytest tests/agent/test_pr3_final_cleanup.py -v
uv run pytest -x -q
uv run ruff check <touched files>
```

---

## Self-Review

1. **Pending 单轨**: PendingToolCall v2 不含 attempt_count/result/error，canonical 来源清晰
2. **ToolCallLedger**: 用 ToolCallLedgerEntry 不复用 PendingToolCall，清理条件明确
3. **Transcript 重建**: 从 ledger + tool_results 每轮重建，不存 loop_messages
4. **12 字段删除**: 定义级删除仅在本 PR（写路径已在 PR2 删除），checkpoint 迁移仍可读取旧 checkpoint
5. **Allowlist**: 只删除不可达类型；`ToolResult.output` 仍需要的类型必须保留
6. **Compat 层**: 60 天日落（2026-08-24）
