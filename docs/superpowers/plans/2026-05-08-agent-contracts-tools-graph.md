# Agent Core Contracts, Tools, and Base Graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable Agent Graph (route→execute→observe→evaluate→synthesize) that can use RAG tools with enforceable contracts.

**Architecture:** Three layered contracts (ToolSpec → AgentRunConfig → AgentState) feed into a LangGraph StateGraph. Tools are defined with full input/output/error/permission specs and registered in a ToolRegistry. The graph uses conditional edges driven by LLM structured output.

**Tech Stack:** Python 3.12+, LangGraph >=1.1.8, Pydantic >=2.11, pytest

**Spec:** `docs/superpowers/specs/2026-05-07-agent-architecture-design.md` (Sections 4–7)

**This plan covers Phases 1–3 from the spec.** Subsequent plans cover: Memory (Phase 4), ResearchAgent (Phase 5), Orchestrator+Parallel (Phase 6), Interrupt/Resume (Phase 7), Remaining Agents (Phase 8), CLI (Phase 9), Long-term Memory (Phase 10).

## Implementation Authority and Legacy Replacement Policy

**Authoritative source:** The design spec is the source of truth. Do not preserve behavior from the current `rag/agent/` implementation unless the spec explicitly keeps it.

**No compatibility layer:** Do not add adapters, aliases, shims, or "new + old" fallback paths for the legacy agent service. The goal is to replace the old agent architecture with the LangGraph/tool-contract architecture described in the spec.

**Delete blocking legacy code:** If existing files such as `rag/agent/service.py`, `rag/agent/executor.py`, `rag/agent/planner.py`, `rag/agent/critic.py`, `rag/agent/report.py`, `rag/agent/synthesizer.py`, `rag/agent/understanding.py`, or legacy exports block the new implementation, delete or replace them in the same task. Do not contort the new design to keep old imports alive.

**Public API reset:** `rag.agent` should export the new contract/graph surface only. Remove legacy `AnalysisAgentService`, `AgentRunState`, `AgentFailureEvent`, and related exports when they conflict with the new `AgentState` contract.

**Command runner:** Use `uv run pytest ...` and `uv run python ...` for verification commands unless a task explicitly says otherwise.

---

## File Map

```
rag/                                  tests/agent/
├── __init__.py          (modify)     ├── test_contract_tool.py
└── agent/                            ├── test_contract_config.py
    ├── state.py         (replace)    ├── test_contract_state.py
    ├── __init__.py      (replace)    ├── test_tool_registry.py
    ├── core/                         ├── test_graph_base.py
    │   ├── __init__.py  (create)     └── conftest.py            (create)
    │   ├── context.py   (create)
    │   ├── definition.py(create)
    │   ├── registry.py  (create)
    │   └── agent_as_tool.py (create)
    ├── tools/
    │   ├── __init__.py  (create)
    │   ├── spec.py      (create)
    │   ├── registry.py  (create)
    │   └── rag_tools.py (create)
    └── graphs/
        ├── __init__.py  (create)
        ├── base.py      (create)
        └── nodes/
            ├── __init__.py  (create)
            ├── route.py     (create)
            ├── execute.py   (create)
            ├── observe.py   (create)
            ├── evaluate.py  (create)
            └── synthesize.py(create)
```

---

### Task 1: ToolSpec contract layer

**Files:**
- Create: `rag/agent/tools/__init__.py`
- Create: `rag/agent/tools/spec.py`
- Create: `tests/agent/__init__.py`
- Create: `tests/agent/conftest.py`
- Create: `tests/agent/test_contract_tool.py`

- [ ] **Step 1: Write tests for ToolPermissions and ToolSpec**

```python
# tests/agent/test_contract_tool.py
from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.tools.spec import ToolError, ToolPermissions, ToolResult, ToolSpec


class SearchInput(BaseModel):
    query: str
    limit: int = 10


class SearchOutput(BaseModel):
    items: list[str]


class TestToolPermissions:
    def test_default_all_false(self) -> None:
        p = ToolPermissions()
        assert p.read_db is False
        assert p.write_db is False
        assert p.kg_mutation is False
        assert p.external_network is False

    def test_kg_mutation_flags_write(self) -> None:
        p = ToolPermissions(write_db=True, kg_mutation=True)
        assert p.kg_mutation is True
        assert p.write_db is True


class TestToolSpec:
    def test_minimal_spec(self) -> None:
        spec = ToolSpec(
            name="test_search",
            description="Search for documents",
            input_model=SearchInput,
            output_model=SearchOutput,
            error_model=ToolError,
            permissions=ToolPermissions(read_db=True),
            timeout_seconds=5.0,
        )
        assert spec.name == "test_search"
        assert spec.timeout_seconds == 5.0
        assert spec.max_retries == 0
        assert spec.idempotent is False
        assert spec.requires_confirmation is False
        assert spec.audit_log is False

    def test_kg_tool_spec_enforces_confirmation(self) -> None:
        spec = ToolSpec(
            name="kg_write",
            description="Write to knowledge graph",
            input_model=SearchInput,
            output_model=SearchOutput,
            error_model=ToolError,
            permissions=ToolPermissions(kg_mutation=True, write_db=True),
            timeout_seconds=10.0,
            requires_confirmation=True,
            audit_log=True,
            idempotent=True,
            max_retries=2,
        )
        assert spec.requires_confirmation is True
        assert spec.audit_log is True
        assert spec.idempotent is True
        assert spec.permissions.kg_mutation is True


class TestToolResult:
    def test_ok_result(self) -> None:
        result = ToolResult(
            tool_call_id="tc_001",
            tool_name="search",
            status="ok",
            output=SearchOutput(items=["a", "b"]),
            latency_ms=100.0,
        )
        assert result.status == "ok"
        assert result.output is not None

    def test_error_result(self) -> None:
        result = ToolResult(
            tool_call_id="tc_002",
            tool_name="search",
            status="error",
            error=ToolError(code="timeout", message="timed out after 5s", retryable=True),
            latency_ms=5000.0,
        )
        assert result.status == "error"
        assert result.error is not None

    def test_ok_rejects_missing_output(self) -> None:
        with pytest.raises(ValueError, match="output is required"):
            ToolResult(tool_call_id="tc_003", tool_name="x", status="ok", output=None, latency_ms=0)

    def test_ok_rejects_error_present(self) -> None:
        with pytest.raises(ValueError, match="error must be None"):
            ToolResult(
                tool_call_id="tc_004", tool_name="x", status="ok",
                output=SearchOutput(items=[]),
                error=ToolError(code="internal", message="x", retryable=True),
                latency_ms=0,
            )

    def test_error_rejects_missing_error(self) -> None:
        with pytest.raises(ValueError, match="error is required"):
            ToolResult(tool_call_id="tc_005", tool_name="x", status="error", error=None, latency_ms=0)

    def test_error_rejects_output_present(self) -> None:
        with pytest.raises(ValueError, match="output must be None"):
            ToolResult(
                tool_call_id="tc_006", tool_name="x", status="error",
                output=SearchOutput(items=[]),
                error=ToolError(code="internal", message="x", retryable=True),
                latency_ms=0,
            )


class TestToolError:
    def test_timeout_error_is_retryable(self) -> None:
        e = ToolError(code="timeout", message="timed out", retryable=True)
        assert e.retryable is True
        assert e.code == "timeout"

    def test_tool_denied_is_not_retryable(self) -> None:
        e = ToolError(code="tool_denied", message="not allowed", retryable=False)
        assert e.retryable is False
        assert e.code == "tool_denied"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agent/test_contract_tool.py -v`
Expected: ImportError (module not created yet)

- [ ] **Step 3: Implement `rag/agent/tools/__init__.py`**

```python
"""Agent tool contracts."""
```

- [ ] **Step 4: Implement `rag/agent/tools/spec.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, model_validator


@dataclass(frozen=True)
class ToolPermissions:
    read_db: bool = False
    write_db: bool = False
    read_object_store: bool = False
    embed: bool = False
    generate: bool = False
    external_network: bool = False
    kg_mutation: bool = False
    user_data: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    error_model: type[BaseModel]
    permissions: ToolPermissions
    timeout_seconds: float
    max_retries: int = 0
    idempotent: bool = False
    token_budget_cost: int = 0
    requires_confirmation: bool = False
    audit_log: bool = False


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool
    detail: dict[str, object] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: str
    tool_name: str
    status: Literal["ok", "error"]
    output: BaseModel | None = None
    error: ToolError | None = None
    latency_ms: float
    token_used: int = 0
    retry_count: int = 0

    @model_validator(mode="after")
    def _check_exclusivity(self) -> ToolResult:
        if self.status == "ok":
            if self.output is None:
                raise ValueError("output is required when status='ok'")
            if self.error is not None:
                raise ValueError("error must be None when status='ok'")
        if self.status == "error":
            if self.error is None:
                raise ValueError("error is required when status='error'")
            if self.output is not None:
                raise ValueError("output must be None when status='error'")
        return self
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/agent/test_contract_tool.py -v`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add rag/agent/tools/__init__.py rag/agent/tools/spec.py tests/agent/
git commit -m "feat(agent): add ToolSpec, ToolPermissions, ToolResult, ToolError contracts"
```

---

### Task 2: AgentRunConfig + RuntimeRegistry + BudgetLedger

**Files:**
- Create: `rag/agent/core/__init__.py`
- Create: `rag/agent/core/context.py`
- Create: `tests/agent/test_contract_config.py`

- [ ] **Step 1: Write tests**

```python
# tests/agent/test_contract_config.py
from __future__ import annotations

import asyncio

import pytest

from rag.agent.core.context import (
    AgentRunConfig,
    AgentRuntimeHandles,
    BudgetLedger,
    RuntimeRegistry,
)
from rag.agent.core.definition import ToolPolicy
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference


class TestBudgetLedger:
    @pytest.mark.asyncio
    async def test_reserve_commit(self) -> None:
        ledger = BudgetLedger(total=1000)
        ok = await ledger.reserve("lease-1", 300)
        assert ok is True
        assert await ledger.remaining() == 700
        overrun = await ledger.commit("lease-1", 250)
        assert overrun == 0
        assert await ledger.remaining() == 750

    @pytest.mark.asyncio
    async def test_reserve_rejects_over_budget(self) -> None:
        ledger = BudgetLedger(total=500)
        ok = await ledger.reserve("lease-1", 600)
        assert ok is False

    @pytest.mark.asyncio
    async def test_refund_returns_tokens(self) -> None:
        ledger = BudgetLedger(total=1000)
        await ledger.reserve("lease-1", 300)
        refunded = await ledger.refund("lease-1")
        assert refunded == 300
        assert await ledger.remaining() == 1000

    @pytest.mark.asyncio
    async def test_commit_records_overrun(self) -> None:
        ledger = BudgetLedger(total=1000)
        await ledger.reserve("lease-1", 200)
        overrun = await ledger.commit("lease-1", 350)
        assert overrun == 150
        assert await ledger.remaining() == 650  # 1000 - 350

    @pytest.mark.asyncio
    async def test_concurrent_reserve(self) -> None:
        ledger = BudgetLedger(total=500)

        async def reserve_300() -> bool:
            return await ledger.reserve("a", 300)

        async def reserve_300_b() -> bool:
            return await ledger.reserve("b", 300)

        results = await asyncio.gather(reserve_300(), reserve_300_b())
        assert sum(1 for r in results if r) == 1  # only one succeeds


class TestAgentRunConfig:
    def test_minimal_config(self) -> None:
        cfg = AgentRunConfig(
            run_id="r1",
            thread_id="t1",
            budget_total=10000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        )
        assert cfg.run_id == "r1"
        assert cfg.max_depth == 2
        assert cfg.parent_run_id is None
        assert cfg.source_scope == ()

    def test_config_defaults(self) -> None:
        cfg = AgentRunConfig(
            run_id="r1",
            thread_id="t2",
            budget_total=5000,
            max_depth=1,
            access_policy=AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        )
        assert cfg.deadline_iso is None
        assert cfg.budget_committed == 0
        assert isinstance(cfg.tool_policy, ToolPolicy)


class TestRuntimeRegistry:
    def test_get_or_create_initializes_handles(self) -> None:
        cfg = AgentRunConfig(
            run_id="reg-test",
            thread_id="t",
            budget_total=8000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        )
        handles = RuntimeRegistry.get_or_create(cfg)
        assert isinstance(handles.budget_ledger, BudgetLedger)
        assert isinstance(handles.cancellation, asyncio.Event)

    def test_get_or_create_returns_same_handles(self) -> None:
        cfg = AgentRunConfig(
            run_id="reg-test-2",
            thread_id="t",
            budget_total=8000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        )
        h1 = RuntimeRegistry.get_or_create(cfg)
        h2 = RuntimeRegistry.get_or_create(cfg)
        assert h1 is h2

    def test_remove_cleans_up(self) -> None:
        cfg = AgentRunConfig(
            run_id="reg-test-3",
            thread_id="t",
            budget_total=8000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        )
        RuntimeRegistry.get_or_create(cfg)
        RuntimeRegistry.remove("reg-test-3")
        # After remove, get_or_create should make a new one
        h_new = RuntimeRegistry.get_or_create(cfg)
        assert h_new is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agent/test_contract_config.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `rag/agent/core/__init__.py`**

```python
"""Agent core contracts: config, registry, definition, compiler."""
```

- [ ] **Step 4: Implement `rag/agent/core/context.py`**

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference


@dataclass(frozen=True)
class AgentRunConfig:
    run_id: str
    thread_id: str
    budget_total: int
    max_depth: int
    access_policy: AccessPolicy
    execution_location_preference: ExecutionLocationPreference
    parent_run_id: str | None = None
    source_scope: tuple[str, ...] = ()
    deadline_iso: str | None = None
    trace_parent_id: str | None = None
    budget_committed: int = 0
    budget_reserved: dict[str, int] = field(default_factory=dict)
    tool_policy: object = field(default_factory=lambda: _default_tool_policy())


def _default_tool_policy() -> object:
    from rag.agent.core.definition import ToolPolicy
    return ToolPolicy()


class BudgetLedger:
    def __init__(self, total: int) -> None:
        self._total = total
        self._lock = asyncio.Lock()
        self._reserved: dict[str, int] = {}
        self._committed: int = 0

    async def remaining(self) -> int:
        async with self._lock:
            return max(0, self._total - self._committed - sum(self._reserved.values()))

    async def reserve(self, lease_id: str, amount: int) -> bool:
        async with self._lock:
            current = max(0, self._total - self._committed - sum(self._reserved.values()))
            if amount > current:
                return False
            self._reserved[lease_id] = amount
            return True

    async def commit(self, lease_id: str, actual: int) -> int:
        async with self._lock:
            reserved = self._reserved.pop(lease_id, 0)
            overrun = max(0, actual - reserved)
            self._committed += actual
            return overrun

    async def refund(self, lease_id: str) -> int:
        async with self._lock:
            return self._reserved.pop(lease_id, 0)


class RuntimeRegistry:
    _handles: dict[str, AgentRuntimeHandles] = {}

    @classmethod
    def get_or_create(cls, run_config: AgentRunConfig) -> AgentRuntimeHandles:
        if run_config.run_id not in cls._handles:
            cls._handles[run_config.run_id] = AgentRuntimeHandles(
                budget_ledger=BudgetLedger(total=run_config.budget_total),
                cancellation=asyncio.Event(),
            )
        return cls._handles[run_config.run_id]

    @classmethod
    def get(cls, run_id: str) -> AgentRuntimeHandles:
        return cls._handles[run_id]

    @classmethod
    def remove(cls, run_id: str) -> None:
        cls._handles.pop(run_id, None)


@dataclass
class AgentRuntimeHandles:
    budget_ledger: BudgetLedger
    cancellation: asyncio.Event
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/agent/test_contract_config.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add rag/agent/core/__init__.py rag/agent/core/context.py tests/agent/test_contract_config.py
git commit -m "feat(agent): add AgentRunConfig, BudgetLedger, RuntimeRegistry"
```

---

### Task 3: AgentDefinition + ModelPolicy + ToolPolicy

**Files:**
- Create: `rag/agent/core/definition.py`
- Modify: `rag/agent/core/context.py` (remove `_default_tool_policy` cycle)

- [ ] **Step 1: Write tests**

```python
# Append to tests/agent/test_contract_config.py

from rag.agent.core.definition import AgentDefinition, ModelPolicy, ToolPolicy
from rag.schema.runtime import AccessPolicy


class TestAgentDefinition:
    def test_minimal_definition(self) -> None:
        ad = AgentDefinition(
            agent_type="research",
            description="Deep research agent",
            system_prompt="You are a research agent.",
            allowed_tools=["vector_search", "grounding"],
        )
        assert ad.agent_type == "research"
        assert ad.allowed_tools == ["vector_search", "grounding"]
        assert ad.model_policy.model_alias == "opus"
        assert ad.max_iterations == 10
        assert ad.max_depth == 2
        assert ad.estimated_token_budget == 8000

    def test_definition_with_access_policy(self) -> None:
        policy = AccessPolicy.default()
        ad = AgentDefinition(
            agent_type="compare",
            description="Comparison agent",
            system_prompt="You compare documents.",
            allowed_tools=["vector_search", "llm_compare"],
            access_policy=policy,
            estimated_token_budget=12000,
        )
        assert ad.access_policy is policy
        assert ad.estimated_token_budget == 12000

    def test_tool_policy_defaults(self) -> None:
        tp = ToolPolicy()
        assert tp.max_parallel_calls == 4
        assert len(tp.require_confirmation_for) == 0
        assert len(tp.deny_tools) == 0

    def test_tool_policy_custom(self) -> None:
        tp = ToolPolicy(
            max_parallel_calls=2,
            require_confirmation_for=frozenset({"kg_upsert"}),
            deny_tools=frozenset({"web_search"}),
        )
        assert "kg_upsert" in tp.require_confirmation_for
        assert "web_search" in tp.deny_tools
        assert tp.max_parallel_calls == 2

    def test_model_policy_defaults(self) -> None:
        mp = ModelPolicy()
        assert mp.model_alias == "opus"
        assert mp.fallback_model == "sonnet"
        assert mp.thinking is True
        assert mp.temperature == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agent/test_contract_config.py::TestAgentDefinition -v`
Expected: ImportError

- [ ] **Step 3: Implement `rag/agent/core/definition.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from rag.schema.runtime import AccessPolicy


@dataclass(frozen=True)
class AgentDefinition:
    agent_type: str
    description: str
    system_prompt: str
    allowed_tools: list[str]
    access_policy: AccessPolicy | None = None
    estimated_token_budget: int = 8000
    model_policy: ModelPolicy = field(default_factory=lambda: ModelPolicy())
    output_model: type[BaseModel] | None = None
    max_iterations: int = 10
    max_depth: int = 2
    tool_policy: ToolPolicy = field(default_factory=lambda: ToolPolicy())


@dataclass(frozen=True)
class ModelPolicy:
    model_alias: str = "opus"
    fallback_model: str | None = "sonnet"
    thinking: bool = True
    temperature: float = 0.0


@dataclass(frozen=True)
class ToolPolicy:
    max_parallel_calls: int = 4
    require_confirmation_for: frozenset[str] = field(default_factory=frozenset)
    deny_tools: frozenset[str] = field(default_factory=frozenset)
```

- [ ] **Step 4: Update `rag/agent/core/context.py`**

Remove the `_default_tool_policy` import cycle. Change `tool_policy` field to:

```python
# In AgentRunConfig, change the tool_policy line to:
from rag.agent.core.definition import ToolPolicy as _ToolPolicy

# ... and the field:
tool_policy: _ToolPolicy = field(default_factory=_ToolPolicy)
```

Actually the full context.py should import directly:

```python
# rag/agent/core/context.py — top imports
from rag.agent.core.definition import ToolPolicy
```

And the `AgentRunConfig` `tool_policy` field becomes:

```python
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/agent/test_contract_config.py -v`
Expected: 13 passed

- [ ] **Step 6: Commit**

```bash
git add rag/agent/core/definition.py rag/agent/core/context.py tests/agent/test_contract_config.py
git commit -m "feat(agent): add AgentDefinition, ModelPolicy, ToolPolicy"
```

---

### Task 4: AgentState TypedDict + reducers

**Files:**
- Replace: `rag/agent/state.py` (new LangGraph `AgentState`; remove legacy `AgentRunState`/failure-state model)
- Create: `tests/agent/test_contract_state.py`

- [ ] **Step 1: Write tests**

```python
# tests/agent/test_contract_state.py
from __future__ import annotations

from pydantic import BaseModel

from rag.agent.state import (
    AgentState,
    _merge_evidence,
    _merge_citations,
    _merge_sets,
    _merge_subtask_results,
    _merge_tool_results,
)
from rag.schema.query import EvidenceItem, AnswerCitation


class TestMergeEvidence:
    def test_dedup_by_evidence_id(self) -> None:
        a = EvidenceItem(evidence_id="e1", doc_id=1, citation_anchor="a", text="A", score=0.8)
        b = EvidenceItem(evidence_id="e1", doc_id=1, citation_anchor="a", text="A better", score=0.9)
        merged = _merge_evidence([a], [b])
        assert len(merged) == 1
        assert merged[0].score == 0.9

    def test_conflict_preserves_both(self) -> None:
        a = EvidenceItem(evidence_id="e1", doc_id=1, citation_anchor="a", text="Alpha is good", score=0.8)
        b = EvidenceItem(evidence_id="e1", doc_id=1, citation_anchor="a", text="Alpha is not good", score=0.9)
        merged = _merge_evidence([a], [b])
        assert len(merged) >= 1
        assert any("conflict" in (item.retrieval_channels or []) for item in merged)


class TestMergeSets:
    def test_union(self) -> None:
        result = _merge_sets({"a", "b"}, {"b", "c"})
        assert result == {"a", "b", "c"}


class TestMergeSubtaskResults:
    def test_merge_disjoint(self) -> None:
        left = {"s1": "result1"}
        right = {"s2": "result2"}
        merged = _merge_subtask_results(left, right)
        assert merged == {"s1": "result1", "s2": "result2"}


class TestMergeToolResults:
    def test_dedup_by_tool_call_id(self) -> None:
        from rag.agent.tools.spec import ToolResult

        class DummyOutput(BaseModel):
            value: str

        r1 = ToolResult(tool_call_id="tc1", tool_name="search", status="ok",
                        output=DummyOutput(value="old"), latency_ms=10)
        r2 = ToolResult(tool_call_id="tc1", tool_name="search", status="ok",
                        output=DummyOutput(value="new"), latency_ms=20)
        merged = _merge_tool_results([r1], [r2])
        assert len(merged) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agent/test_contract_state.py -v`
Expected: ImportError (state.py not yet refactored)

- [ ] **Step 3: Replace `rag/agent/state.py`**

Replace the file with the new LangGraph state contract from the spec. Do not keep legacy `AgentRunState`, `AgentFailureEvent`, or `AgentFailureKind` in this module; old code that imports them must be updated, de-exported, or removed in later tasks.

```python
from __future__ import annotations

from typing import Annotated, Literal

from langgraph.graph import add_messages
from langgraph.graph.message import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from rag.agent.core.context import AgentRunConfig


# ── Structured decision output ──

class ThinkOutput(BaseModel):
    action: Literal["execute", "synthesize", "pause"]
    tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    thought: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    stop_reason: str | None = None
    needs_user_input: str | None = None


class ToolCallPlan(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object]

    @classmethod
    def create(cls, tool_name: str, arguments: dict) -> ToolCallPlan:
        from uuid import uuid4
        return cls(
            tool_call_id=f"tc_{uuid4().hex[:12]}",
            tool_name=tool_name,
            arguments=arguments,
        )


# ── Working memory models (Phase A) ──

class WorkingSummary(BaseModel):
    summary: str
    covered_message_ids: list[str]
    updated_at: str
    token_count: int


class ExtractedFact(BaseModel):
    fact_id: str
    text: str
    source_message_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    stale: bool = False


class ContextBudgetSnapshot(BaseModel):
    max_context_tokens: int
    system_tokens: int = 0
    evidence_tokens: int = 0
    working_memory_tokens: int = 0
    recalled_memory_tokens: int = 0
    message_tail_tokens: int = 0
    tool_result_tokens: int = 0


# ── AgentState ──

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    evidence: Annotated[list, _merge_evidence]
    citations: Annotated[list, _merge_citations]
    tool_results: Annotated[list, _merge_tool_results]
    task: str
    run_config: AgentRunConfig
    plan: object | None  # TaskDAG, typed loosely to avoid import cycle
    iteration: int
    status: str
    pending_tool_calls: list[ToolCallPlan]
    confirmed_tool_call_ids: set[str]
    user_decision: str | None
    next_subtasks: list[object] | None
    working_summary: WorkingSummary | None
    extracted_facts: list[ExtractedFact]
    context_budget: ContextBudgetSnapshot | None
    subtask_results: Annotated[dict, _merge_subtask_results]
    terminal_subtasks: Annotated[set[str], _merge_sets]
    successful_subtasks: Annotated[set[str], _merge_sets]
    final_answer: str | None
    groundedness_flag: bool
    insufficient_evidence_flag: bool


# ── Reducers ──

def _merge_evidence(left: list, right: list) -> list:
    from rag.schema.query import EvidenceItem
    from rag.utils.text import keyword_overlap, search_terms

    merged: dict[str, EvidenceItem] = {}
    for item in left + right:
        existing = merged.get(item.evidence_id)
        if existing is None:
            merged[item.evidence_id] = item
        elif _texts_contradict(existing.text, item.text):
            merged[item.evidence_id] = existing.model_copy(
                update={"retrieval_channels": [*existing.retrieval_channels, "conflict"]}
            )
            merged[f"{item.evidence_id}__conflict"] = item.model_copy(
                update={
                    "evidence_id": f"{item.evidence_id}__conflict",
                    "retrieval_channels": [*item.retrieval_channels, "conflict"],
                }
            )
        elif item.score > existing.score:
            merged[item.evidence_id] = item
    return sorted(merged.values(), key=lambda e: e.score, reverse=True)


def _texts_contradict(a: str, b: str) -> bool:
    NEGATION_MARKERS = (" not ", " no ", " does not ", " cannot ", " without ", "未", "不")
    a_lower = f" {a.lower().strip()} "
    b_lower = f" {b.lower().strip()} "
    a_neg = any(m in a_lower for m in NEGATION_MARKERS)
    b_neg = any(m in b_lower for m in NEGATION_MARKERS)
    return a_neg != b_neg


def _merge_citations(left: list, right: list) -> list:
    return list({c.citation_id: c for c in left + right}.values())


def _merge_tool_results(left: list, right: list) -> list:
    return list({r.tool_call_id: r for r in left + right}.values())


def _merge_subtask_results(left: dict, right: dict) -> dict:
    return {**left, **right}


def _merge_sets(left: set, right: set) -> set:
    return left | right
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/agent/test_contract_state.py -v`
Expected: 4-5 passed (merge helper tests)

- [ ] **Step 5: Commit**

```bash
git add rag/agent/state.py tests/agent/test_contract_state.py
git commit -m "feat(agent): add AgentState TypedDict with reducer functions"
```

---

### Task 5: ToolRegistry

**Files:**
- Create: `rag/agent/tools/registry.py`
- Create: `tests/agent/test_tool_registry.py`

- [ ] **Step 1: Write tests**

```python
# tests/agent/test_tool_registry.py
from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec


class DummyInput(BaseModel):
    text: str


class DummyOutput(BaseModel):
    result: str


_dummy_spec = ToolSpec(
    name="dummy", description="A dummy tool",
    input_model=DummyInput, output_model=DummyOutput,
    error_model=ToolError, permissions=ToolPermissions(),
    timeout_seconds=1.0,
)


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        registry.register(_dummy_spec)
        assert registry.get("dummy") is _dummy_spec

    def test_get_missing_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_all(self) -> None:
        registry = ToolRegistry()
        registry.register(_dummy_spec)
        another = ToolSpec(
            name="another", description="x",
            input_model=DummyInput, output_model=DummyOutput,
            error_model=ToolError, permissions=ToolPermissions(),
            timeout_seconds=2.0,
        )
        registry.register(another)
        names = [s.name for s in registry.list_all()]
        assert "dummy" in names
        assert "another" in names

    def test_register_duplicate_overwrites(self) -> None:
        registry = ToolRegistry()
        registry.register(_dummy_spec)
        updated = ToolSpec(
            name="dummy", description="updated",
            input_model=DummyInput, output_model=DummyOutput,
            error_model=ToolError, permissions=ToolPermissions(),
            timeout_seconds=3.0,
        )
        registry.register(updated)
        assert registry.get("dummy").timeout_seconds == 3.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agent/test_tool_registry.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `rag/agent/tools/registry.py`**

```python
from __future__ import annotations

from rag.agent.tools.spec import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def list_all(self) -> list[ToolSpec]:
        return list(self._tools.values())
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/agent/test_tool_registry.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add rag/agent/tools/registry.py tests/agent/test_tool_registry.py
git commit -m "feat(agent): add ToolRegistry"
```

---

### Task 6: AgentRegistry + AgentAsTool

**Files:**
- Create: `rag/agent/core/registry.py`
- Create: `rag/agent/core/agent_as_tool.py`
- Modify: `rag/agent/core/__init__.py` (update)

- [ ] **Step 1: Write tests**

```python
# Append to tests/agent/test_contract_config.py

from rag.agent.core.registry import AgentRegistry
from rag.agent.core.definition import AgentDefinition


class TestAgentRegistry:
    def test_register_and_get(self) -> None:
        ad = AgentDefinition(
            agent_type="test_research",
            description="Test agent",
            system_prompt="You are a test agent.",
            allowed_tools=["search"],
        )
        AgentRegistry.register(ad)
        retrieved = AgentRegistry.get("test_research")
        assert retrieved is ad

    def test_get_missing_raises(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            AgentRegistry.get("nonexistent_agent_type")

    def test_list_all(self) -> None:
        ad1 = AgentDefinition(
            agent_type="agent_a",
            description="A",
            system_prompt="A",
            allowed_tools=[],
        )
        AgentRegistry.register(ad1)
        all_agents = AgentRegistry.list_all()
        assert ad1 in all_agents
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agent/test_contract_config.py::TestAgentRegistry -v`
Expected: ImportError

- [ ] **Step 3: Implement `rag/agent/core/registry.py`**

```python
from __future__ import annotations

from rag.agent.core.definition import AgentDefinition


class AgentRegistry:
    _agents: dict[str, AgentDefinition] = {}

    @classmethod
    def register(cls, definition: AgentDefinition) -> None:
        cls._agents[definition.agent_type] = definition

    @classmethod
    def get(cls, agent_type: str) -> AgentDefinition:
        if agent_type not in cls._agents:
            raise KeyError(f"Agent type '{agent_type}' not found in registry")
        return cls._agents[agent_type]

    @classmethod
    def list_all(cls) -> list[AgentDefinition]:
        return list(cls._agents.values())
```

- [ ] **Step 4: Implement `rag/agent/core/agent_as_tool.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

from rag.agent.core.definition import AgentDefinition
from rag.agent.tools.spec import ToolSpec


@dataclass(frozen=True)
class AgentToolSpec:
    tool_spec: ToolSpec
    agent_definition: AgentDefinition
    inherits_context: bool = True
```

- [ ] **Step 5: Update `rag/agent/core/__init__.py`**

```python
"""Agent core contracts: config, registry, definition, compiler."""
from rag.agent.core.context import AgentRunConfig, BudgetLedger, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition, ModelPolicy, ToolPolicy
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.agent_as_tool import AgentToolSpec

__all__ = [
    "AgentDefinition",
    "AgentRegistry",
    "AgentRunConfig",
    "AgentToolSpec",
    "BudgetLedger",
    "ModelPolicy",
    "RuntimeRegistry",
    "ToolPolicy",
]
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/agent/test_contract_config.py -v`
Expected: 16 passed

- [ ] **Step 7: Commit**

```bash
git add rag/agent/core/registry.py rag/agent/core/agent_as_tool.py rag/agent/core/__init__.py tests/agent/test_contract_config.py
git commit -m "feat(agent): add AgentRegistry and AgentToolSpec"
```

---

### Task 7: Public exports reset to new Agent API

**Files:**
- Replace: `rag/agent/__init__.py`
- Modify: `rag/__init__.py`

- [ ] **Step 1: Replace `rag/agent/__init__.py` with new-design exports only**

Do not import or export legacy `AnalysisAgentService`, `AgentRunState`, `AgentFailureEvent`, or `AgentFailureKind`. If old modules depend on those exports, remove those old import paths in the task that touches them.

```python
"""Public exports for the agent orchestration package."""

from rag.agent.core.context import AgentRunConfig, BudgetLedger, RuntimeRegistry
from rag.agent.core.definition import AgentDefinition, ModelPolicy, ToolPolicy
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.agent_as_tool import AgentToolSpec
from rag.agent.state import (
    AgentState,
    ContextBudgetSnapshot,
    ExtractedFact,
    ThinkOutput,
    ToolCallPlan,
    WorkingSummary,
)
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolResult, ToolSpec
from rag.agent.tools.registry import ToolRegistry

__all__ = [
    "AgentDefinition",
    "AgentRegistry",
    "AgentRunConfig",
    "AgentState",
    "AgentToolSpec",
    "BudgetLedger",
    "ContextBudgetSnapshot",
    "ExtractedFact",
    "ModelPolicy",
    "RuntimeRegistry",
    "ThinkOutput",
    "ToolCallPlan",
    "ToolError",
    "ToolPermissions",
    "ToolPolicy",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
]
```

- [ ] **Step 2: Remove legacy agent lazy exports from `rag/__init__.py`**

Keep non-agent exports intact. Replace `AgentTaskRequest` / `AnalysisAgentService` with the new contract exports so root-level imports do not point at deleted legacy code.

```python
"""Core RAG library public exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AgentDefinition",
    "AgentRunConfig",
    "AgentState",
    "ToolRegistry",
    "ToolSpec",
    "AssemblyConfig",
    "AssemblyDiagnostics",
    "AssemblyOverrides",
    "AssemblyProfileSpec",
    "AssemblyRequest",
    "CapabilityAssemblyService",
    "CapabilityRequirements",
    "RAGRuntime",
    "StorageComponentConfig",
    "StorageConfig",
]

_EXPORTS = {
    "AgentDefinition": ("rag.agent", "AgentDefinition"),
    "AgentRunConfig": ("rag.agent", "AgentRunConfig"),
    "AgentState": ("rag.agent", "AgentState"),
    "ToolRegistry": ("rag.agent", "ToolRegistry"),
    "ToolSpec": ("rag.agent", "ToolSpec"),
    "AssemblyConfig": ("rag.assembly", "AssemblyConfig"),
    "AssemblyDiagnostics": ("rag.assembly", "AssemblyDiagnostics"),
    "AssemblyOverrides": ("rag.assembly", "AssemblyOverrides"),
    "AssemblyProfileSpec": ("rag.assembly", "AssemblyProfileSpec"),
    "AssemblyRequest": ("rag.assembly", "AssemblyRequest"),
    "CapabilityAssemblyService": ("rag.assembly", "CapabilityAssemblyService"),
    "CapabilityRequirements": ("rag.assembly", "CapabilityRequirements"),
    "RAGRuntime": ("rag.runtime", "RAGRuntime"),
    "StorageComponentConfig": ("rag.storage", "StorageComponentConfig"),
    "StorageConfig": ("rag.storage", "StorageConfig"),
}


def __getattr__(name: str) -> object:
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module 'rag' has no attribute {name!r}")
    module_name, attr_name = export
    module = import_module(module_name)
    return getattr(module, attr_name)
```

- [ ] **Step 3: Verify imports work**

Run: `uv run python -c "from rag.agent import ToolSpec, AgentRunConfig, AgentDefinition, AgentState, ToolRegistry; print('OK')"`
Expected: prints "OK"

Run: `uv run python -c "from rag import ToolSpec, AgentDefinition, AgentState; print('OK')"`
Expected: prints "OK"

- [ ] **Step 4: Commit**

```bash
git add rag/agent/__init__.py rag/__init__.py
git commit -m "feat(agent): reset public exports to new agent contracts"
```

---

### Task 8: Base Agent Graph (route → execute → observe → evaluate → synthesize)

**Files:**
- Create: `rag/agent/graphs/__init__.py`
- Create: `rag/agent/graphs/nodes/__init__.py`
- Create: `rag/agent/graphs/nodes/route.py`
- Create: `rag/agent/graphs/nodes/execute.py`
- Create: `rag/agent/graphs/nodes/observe.py`
- Create: `rag/agent/graphs/nodes/evaluate.py`
- Create: `rag/agent/graphs/nodes/synthesize.py`
- Create: `rag/agent/graphs/base.py`
- Create: `tests/agent/test_graph_base.py`

- [ ] **Step 1: Write tests for the base graph**

```python
# tests/agent/test_graph_base.py
from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentDefinition, ToolPolicy
from rag.agent.graphs.base import build_agent_graph
from rag.agent.state import AgentState
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolResult, ToolSpec
from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference


class EchoInput(BaseModel):
    message: str


class EchoOutput(BaseModel):
    message: str


_echo_spec = ToolSpec(
    name="echo",
    description="Echo back the message",
    input_model=EchoInput,
    output_model=EchoOutput,
    error_model=ToolError,
    permissions=ToolPermissions(),
    timeout_seconds=1.0,
)


def _make_test_config() -> AgentRunConfig:
    return AgentRunConfig(
        run_id="graph-test",
        thread_id="graph-test",
        budget_total=10000,
        max_depth=2,
        access_policy=AccessPolicy.default(),
        execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
    )


def _make_definition() -> AgentDefinition:
    return AgentDefinition(
        agent_type="echo_agent",
        description="Test echo agent",
        system_prompt="You have an echo tool.",
        allowed_tools=["echo"],
        max_iterations=3,
    )


class TestBaseGraph:
    def test_builds_without_errors(self) -> None:
        registry = ToolRegistry()
        registry.register(_echo_spec)
        definition = _make_definition()
        graph = build_agent_graph(definition=definition, tool_registry=registry)
        assert graph is not None

    def test_simple_route_to_fast_path(self) -> None:
        registry = ToolRegistry()
        registry.register(_echo_spec)
        definition = _make_definition()
        graph = build_agent_graph(definition=definition, tool_registry=registry)
        config = _make_test_config()

        initial_state: AgentState = {
            "messages": [],
            "evidence": [],
            "citations": [],
            "tool_results": [],
            "task": "hello",
            "run_config": config,
            "plan": None,
            "iteration": 0,
            "status": "running",
            "pending_tool_calls": [],
            "confirmed_tool_call_ids": set(),
            "user_decision": None,
            "next_subtasks": None,
            "working_summary": None,
            "extracted_facts": [],
            "context_budget": None,
            "subtask_results": {},
            "terminal_subtasks": set(),
            "successful_subtasks": set(),
            "final_answer": None,
            "groundedness_flag": False,
            "insufficient_evidence_flag": False,
        }
        result = graph.invoke(initial_state, config={"configurable": {"thread_id": "graph-test"}})
        assert result["status"] in ("done", "running")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agent/test_graph_base.py -v`
Expected: ImportError

- [ ] **Step 3: Implement `rag/agent/graphs/__init__.py`**

```python
"""Agent graph builders."""
```

- [ ] **Step 4: Implement `rag/agent/graphs/nodes/__init__.py`**

```python
"""Graph node implementations."""
```

- [ ] **Step 5: Implement `rag/agent/graphs/nodes/route.py`**

```python
from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, Field

from rag.agent.state import AgentState, ToolCallPlan
from rag.schema.query import RetrievalSignals


class AgentRouteDecision(BaseModel):
    route: Literal["fast_path", "decompose", "direct"]
    reason: str
    retrieval_signals: RetrievalSignals = Field(default_factory=RetrievalSignals)
    tool_calls: list[ToolCallPlan] = Field(default_factory=list)


def route_node(state: AgentState) -> dict:
    """
    Agent route 节点负责将用户任务翻译成：
    1. Agent 执行路径：fast_path / decompose / direct
    2. RAG 可执行检索信号：RetrievalSignals

    不调用 RAG Core 的 QueryUnderstandingService。
    不依赖 RAG 的 TaskType。
    """
    decision = agent_route_decider.decide(
        task=state["task"],
        run_config=state["run_config"],
        messages=state.get("messages", []),
    )

    return {
        "status": decision.route,
        "route_reason": decision.reason,
        "retrieval_signals": decision.retrieval_signals,
        "pending_tool_calls": decision.tool_calls,
    }


def route_after_route(state: AgentState) -> str:
    status = state.get("status", "running")
    if status == "fast_path":
        return "synthesize"  # skip agent loop, go straight to answer
    return "execute"
```

路由标准按执行需求定义，不按固定任务枚举：

- `fast_path`：单次 RAG 检索 + grounded generation 足够，不需要拆分子任务、并行子 Agent、用户确认或外部副作用。
- `decompose`：需要多个独立检索问题、多个证据维度、比较多个对象/版本/方案，或需要并行子 Agent / 任务 DAG。
- `direct`：需要普通 Agent loop 调工具，或需要先执行非 RAG 工具、多轮 evaluate、用户确认。

Agent route 可以使用 LLM structured output、确定性规则、历史上下文或 evaluator，但不得调用 RAG Core 的 `QueryUnderstandingService`；RAG 只接收 `RetrievalSignals`，不接收任务类型。
```

- [ ] **Step 6: Implement `rag/agent/graphs/nodes/execute.py`**

```python
from __future__ import annotations

import asyncio
from typing import Any

from langgraph.types import Command

from rag.agent.core.context import RuntimeRegistry
from rag.agent.state import AgentState, ToolCallPlan
from rag.agent.tools.spec import ToolError, ToolResult


async def execute_node(state: AgentState) -> dict | Command:
    pending = state.get("pending_tool_calls", [])
    if not pending:
        return {}

    tool_policy = state["run_config"].tool_policy
    results: list[ToolResult] = []

    # 1. Denied tools
    denied, rest = [], []
    for tc in pending:
        if tc.tool_name in tool_policy.deny_tools:
            denied.append(ToolResult(
                tool_call_id=tc.tool_call_id, tool_name=tc.tool_name,
                status="error",
                error=ToolError(code="tool_denied", message=f"{tc.tool_name} is denied", retryable=False),
                latency_ms=0,
            ))
        else:
            rest.append(tc)
    results.extend(denied)

    # 2. Confirmation check
    confirmed = state.get("confirmed_tool_call_ids", set())
    needs_confirmation = [
        tc for tc in rest
        if tc.tool_name in tool_policy.require_confirmation_for
        and tc.tool_call_id not in confirmed
    ]
    if needs_confirmation:
        return Command(
            goto="pause",
            update={
                "status": "paused",
                "needs_user_input": f"Confirm tool execution: {[tc.tool_name for tc in needs_confirmation]}",
                "pending_tool_calls": needs_confirmation,
            },
        )

    # 3. Execute within max_parallel_calls
    executables = rest[:tool_policy.max_parallel_calls]
    excess = rest[tool_policy.max_parallel_calls:]

    gathered = await asyncio.gather(
        *[_execute_one_tool(tc, state) for tc in executables],
        return_exceptions=True,
    )
    for i, result_or_exc in enumerate(gathered):
        if isinstance(result_or_exc, Exception):
            results.append(ToolResult(
                tool_call_id=executables[i].tool_call_id,
                tool_name=executables[i].tool_name,
                status="error",
                error=ToolError(code="internal", message=str(result_or_exc), retryable=True),
                latency_ms=0,
            ))
        else:
            results.append(result_or_exc)

    return {"tool_results": results, "pending_tool_calls": excess}


async def _execute_one_tool(tc: ToolCallPlan, state: AgentState) -> ToolResult:
    """Fail closed until a registered callable runner is wired in a later task."""
    import time
    t0 = time.perf_counter()
    try:
        # Until real tool execution is wired, fail closed instead of fabricating success.
        return ToolResult(
            tool_call_id=tc.tool_call_id,
            tool_name=tc.tool_name,
            status="error",
            error=ToolError(code="tool_not_implemented", message=f"{tc.tool_name} has no runner", retryable=False),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as exc:
        return ToolResult(
            tool_call_id=tc.tool_call_id,
            tool_name=tc.tool_name,
            status="error",
            error=ToolError(code="internal", message=str(exc), retryable=True),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
```

- [ ] **Step 7: Implement remaining nodes**

```python
# rag/agent/graphs/nodes/observe.py
from __future__ import annotations

from rag.agent.state import AgentState


def observe_node(state: AgentState) -> dict:
    """Process tool results. In the full implementation this formats results for the LLM."""
    results = state.get("tool_results", [])
    if not results:
        return {}
    # Build observation messages from tool results
    observations: list[str] = []
    for r in results:
        if r.status == "ok":
            observations.append(f"[{r.tool_name}] succeeded in {r.latency_ms:.0f}ms")
        else:
            observations.append(f"[{r.tool_name}] FAILED ({r.error.code}): {r.error.message}")
    return {}
```

```python
# rag/agent/graphs/nodes/evaluate.py
from __future__ import annotations

from rag.agent.state import AgentState


async def evaluate_node(state: AgentState) -> dict:
    """Evaluate evidence and decide next action."""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("run_config", type("x", (), {"max_depth": 2})()).max_depth * 5

    # Early stop: check budget
    handles = None
    try:
        from rag.agent.core.context import RuntimeRegistry
        handles = RuntimeRegistry.get(state["run_config"].run_id)
        remaining = await handles.budget_ledger.remaining()
        if remaining <= 0:
            return {"status": "done", "stop_reason": "budget_exhausted"}
    except Exception:
        pass

    # Simple heuristic for MVP: stop after max_iterations
    if iteration >= max_iterations:
        return {"status": "done", "stop_reason": "max_iterations"}
    if not state.get("pending_tool_calls"):
        return {"status": "done", "stop_reason": "no_pending_tools"}
    return {"status": "running"}


def route_after_evaluate(state: AgentState) -> str:
    if state.get("status") == "done":
        return "synthesize"
    return "execute"
```

```python
# rag/agent/graphs/nodes/synthesize.py
from __future__ import annotations

from rag.agent.state import AgentState


def synthesize_node(state: AgentState) -> dict:
    """Format final output from evidence and tool results."""
    tool_results = state.get("tool_results", [])
    ok_count = sum(1 for r in tool_results if r.status == "ok")
    error_count = sum(1 for r in tool_results if r.status == "error")
    return {
        "status": "done",
        "final_answer": f"Agent run complete. {ok_count} tools succeeded, {error_count} failed.",
        "groundedness_flag": ok_count > 0,
    }
```

- [ ] **Step 8: Implement `rag/agent/graphs/base.py`**

```python
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rag.agent.core.definition import AgentDefinition
from rag.agent.graphs.nodes.execute import execute_node
from rag.agent.graphs.nodes.evaluate import evaluate_node, route_after_evaluate
from rag.agent.graphs.nodes.observe import observe_node
from rag.agent.graphs.nodes.route import route_after_route, route_node
from rag.agent.graphs.nodes.synthesize import synthesize_node
from rag.agent.state import AgentState
from rag.agent.tools.registry import ToolRegistry


def build_agent_graph(
    *,
    definition: AgentDefinition,
    tool_registry: ToolRegistry,
) -> StateGraph:
    """Build a base Agent Graph: route → execute → observe → evaluate → synthesize."""

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("route", route_node)
    graph.add_node("execute", execute_node)
    graph.add_node("observe", observe_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("synthesize", synthesize_node)

    # Edges
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", route_after_route, {
        "execute": "execute",
        "synthesize": "synthesize",
    })
    graph.add_edge("execute", "observe")
    graph.add_edge("observe", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluate, {
        "execute": "execute",
        "synthesize": "synthesize",
    })
    graph.add_edge("synthesize", END)

    return graph.compile()
```

- [ ] **Step 9: Run tests**

Run: `uv run pytest tests/agent/test_graph_base.py -v`
Expected: 2 passed

- [ ] **Step 10: Commit**

```bash
git add rag/agent/graphs/ tests/agent/test_graph_base.py
git commit -m "feat(agent): add base Agent Graph with route→execute→observe→evaluate→synthesize"
```

---

### Task 9: Wire RAG tools into the ToolRegistry

**Files:**
- Create: `rag/agent/tools/rag_tools.py`

- [ ] **Step 1: Implement `rag/agent/tools/rag_tools.py`**

```python
from __future__ import annotations

from pydantic import BaseModel

from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec


class SearchInput(BaseModel):
    query: str
    top_k: int = 8


class SearchOutput(BaseModel):
    items: list[dict[str, object]]


vector_search = ToolSpec(
    name="vector_search",
    description="Semantic vector search across document summaries. Use for natural language queries.",
    input_model=SearchInput,
    output_model=SearchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True, embed=True),
    timeout_seconds=10.0,
    max_retries=1,
    token_budget_cost=500,
)

keyword_search = ToolSpec(
    name="keyword_search",
    description="Lexical/keyword search. Use for exact terms, document IDs, codes, dates.",
    input_model=SearchInput,
    output_model=SearchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True),
    timeout_seconds=5.0,
    max_retries=1,
    token_budget_cost=200,
)

grounding = ToolSpec(
    name="grounding",
    description="Read original document text at a precise location. Use to verify retrieved evidence.",
    input_model=SearchInput,
    output_model=SearchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True, read_object_store=True),
    timeout_seconds=15.0,
    max_retries=2,
    token_budget_cost=1000,
)

rerank = ToolSpec(
    name="rerank",
    description="Re-rank candidate evidence by relevance to the query. Use when evidence ordering matters.",
    input_model=SearchInput,
    output_model=SearchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True, embed=True, generate=True),
    timeout_seconds=10.0,
    max_retries=1,
    token_budget_cost=800,
)

graph_expand = ToolSpec(
    name="graph_expand",
    description="Expand retrieval via knowledge graph neighbors. Use for multi-hop or relational queries.",
    input_model=SearchInput,
    output_model=SearchOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_db=True),
    timeout_seconds=5.0,
    max_retries=1,
    token_budget_cost=300,
)

ALL_RAG_TOOLS = [vector_search, keyword_search, grounding, rerank, graph_expand]
```

- [ ] **Step 2: Verify imports**

Run: `uv run python -c "from rag.agent.tools.rag_tools import ALL_RAG_TOOLS; print(len(ALL_RAG_TOOLS))"`
Expected: prints "5"

- [ ] **Step 3: Commit**

```bash
git add rag/agent/tools/rag_tools.py
git commit -m "feat(agent): add RAG tool specs (vector_search, keyword_search, grounding, rerank, graph_expand)"
```

---

## After This Plan

After completing all 9 tasks, you will have:
- `ToolSpec`/`ToolPermissions`/`ToolResult`/`ToolError` contracts with tests
- `AgentRunConfig`/`BudgetLedger`/`RuntimeRegistry` with tests
- `AgentDefinition`/`ModelPolicy`/`ToolPolicy` with tests
- `AgentState` TypedDict with 6 custom reducers
- `AgentRegistry` and `AgentToolSpec`
- A working base `StateGraph` (route→execute→observe→evaluate→synthesize)
- 5 RAG tool specs

Next plan: **Phase 4–5** (Working Memory compaction + ResearchAgent end-to-end).

---

> **Agentic implementation note:** This plan contains 9 tasks. Each task uses checkbox syntax for step tracking. Use superpowers:subagent-driven-development for implementation.
