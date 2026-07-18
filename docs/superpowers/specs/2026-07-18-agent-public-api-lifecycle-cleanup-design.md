# Agent Public API and Session Lifecycle Cleanup Design

## Status

Approved design boundary as of 2026-07-18. This document replaces the earlier
proposal to make one public SDK method the unique execution entry. The public
SDK keeps distinct one-shot, conversation, and resume operations while the
implementation converges on one canonical runtime chain.

## Goal

Reduce the number of public concepts, duplicate capability entry points,
historical compatibility names, public `Any` values, ineffective
configuration, and unused alternate implementations without changing the
verified runtime behavior delivered by PRs #26-#28.

The target is semantic consolidation, not line-count reduction.

## Current-state evidence

The current SDK already routes both `Agent.run/arun` and `Agent.chat/achat`
through `AgentService.chat`, which enters `_run_request`, `AgentLoop`, and
`ToolExecutor`. The cleanup therefore does not need a new executor, graph,
checkpoint system, or event source.

The current gaps are:

- one-shot runs and conversations create indistinguishable Session rows;
- ordinary Session listing exposes every internal one-shot Session;
- `stream` mixes one-shot and conversation semantics;
- `AgentResult` exposes `raw`, `thread_id`, a `run_id` compatibility property,
  and internal values through `Any`;
- `knowledge: tuple[str, ...]` only acts as a truthiness switch and never
  selects a named source;
- RAG parameters are duplicated across the high-level facade, CLI, runtime
  binding, builder, and provider;
- legacy compiler, graph, factory, delegation, registry, and PrimitiveOps
  implementations remain importable despite not being on the product runtime
  path.

## Protected behavior

The implementation must preserve:

- the CLI behavior validated by PRs #26-#28;
- strict `max_turns` enforcement;
- failure circuit breaking;
- `ToolPolicy` and execution-time permission enforcement;
- `AgentLoop` and `ToolExecutor` behavior;
- deterministic approval and multi-call resume behavior;
- checkpoint persistence and legacy checkpoint decoding;
- streaming ordering, cancellation, and the canonical event source;
- the existing CLI/SDK -> AgentService -> AgentLoop -> ToolExecutor runtime
  chain.

No second Registry, Executor, event system, graph runtime, or knowledge source
catalog is introduced.

## Public lifecycle semantics

### Session kinds

Session kind is an internal persistence concept:

```python
class SessionKind(StrEnum):
    ONE_SHOT = "one_shot"
    CONVERSATION = "conversation"
```

- `run/arun` create an internal `ONE_SHOT` Session.
- `chat/achat(session_id=None)` create a `CONVERSATION` Session.
- `chat/achat(session_id=...)` continue only an existing `CONVERSATION`.
- Passing a one-shot Session ID to `chat/achat` fails before starting a Turn.
- `resume/aresume` restore a paused or interrupted Turn in either Session kind.
- A crashed Turn whose RUNNING lease has expired is atomically normalized to
  INTERRUPTED before it becomes resumable. A live RUNNING lease is never
  resumable.
- Session kind is immutable after Session creation.

One-shot Sessions remain durable so checkpointing, approval, audit, and Turn
resume continue to work. They are hidden from ordinary product Session views;
they are not automatically deleted after completion.

### Result identifiers

Every result returns `turn_id`.

- A one-shot result returns `session_id=None`, including a resumed one-shot
  Turn.
- A conversation result returns the durable Conversation Session ID, including
  a resumed conversation Turn.
- The internal one-shot Session ID is never promoted through result or stream
  DTOs.

### Public method contract

The high-level constructor becomes:

```python
Agent(
    *,
    model: str | None = None,
    checkpoint_db: Path | None = None,
    workspace_path: Path | str | None = None,
    model_session_path: Path | None = None,
    knowledge: RAGKnowledgeConfig | None = None,
)
```

`agent_type` and every flat RAG/vector/embedding/reranker parameter are
removed. `checkpoint_db`, `workspace_path`, and `model_session_path` remain
because they currently affect behavior.

The non-streaming methods are:

```python
def run(
    self,
    task: str,
    *,
    files: Sequence[str] | None = None,
    max_turns: int | None = None,
    max_tokens_total: int | None = None,
    allow_write_tools: bool = False,
    allow_execute_tools: bool = False,
) -> AgentResult: ...

async def arun(...) -> AgentResult: ...

def chat(
    self,
    message: str,
    *,
    session_id: str | None = None,
    files: Sequence[str] | None = None,
    max_turns: int | None = None,
    max_tokens_total: int | None = None,
    allow_write_tools: bool = False,
    allow_execute_tools: bool = False,
) -> AgentResult: ...

async def achat(...) -> AgentResult: ...

def resume(
    self,
    turn_id: str,
    action: ResumeAction,
    *,
    user_input: str | None = None,
) -> AgentResult: ...

async def aresume(...) -> AgentResult: ...
```

The historical public parameters `run_id`, `tools`, `disabled_tools`, and
`allow_discovery_tools` are removed rather than retained as deprecated aliases.
The active permission parameters remain because they feed the verified
ToolPolicy behavior.

The current `models`, `current_model`, and `switch_model` methods remain.

### Streaming contract

The mixed `stream` method is removed and replaced by:

```python
async def astream(
    self,
    task: str,
    *,
    files: Sequence[str] | None = None,
    max_turns: int | None = None,
    max_tokens_total: int | None = None,
    allow_write_tools: bool = False,
    allow_execute_tools: bool = False,
) -> AsyncIterator[StreamEvent]: ...

async def astream_chat(
    self,
    message: str,
    *,
    session_id: str | None = None,
    files: Sequence[str] | None = None,
    max_turns: int | None = None,
    max_tokens_total: int | None = None,
    allow_write_tools: bool = False,
    allow_execute_tools: bool = False,
) -> AsyncIterator[StreamEvent]: ...
```

`astream` is one-shot. `astream_chat` creates or continues a Conversation. The
new Conversation Session ID is present on the first TURN_START event when
`astream_chat` is called without a Session ID. An `astream` event always
projects `session_id=None` even though the underlying one-shot Session is
durable.

Streaming resume is not added in this change. A paused stream exposes its Turn
ID and the caller can use the existing `aresume` method.

## One private execution facade

Public and CLI methods use two private facade helpers:

```text
run/arun -----------+
                    +--> Agent._execute_turn --> AgentService.chat
chat/achat ---------+                           --> _run_request
                                                 --> AgentLoop
                                                 --> ToolExecutor

astream ------------+
                    +--> Agent._stream_turn --> AgentService.chat_streaming
astream_chat -------+                          --> existing event source

resume/aresume --------> AgentService.resume_turn --> existing resume path
```

`_execute_turn` selects Session kind and projects the internal result. It does
not own runtime state. `_stream_turn` selects Session kind and projects the
canonical events. It does not create an event bus, sink, queue, or alternate
loop.

The CLI calls these package-private facade helpers when it needs an event sink
or its interactive approval driver. It no longer constructs `AgentRunRequest`,
opens `AgentService` directly, or reads `AgentRunResult`/`AgentResult.raw`.

The internal checkpoint adapter may continue to emit the third-party
`"thread_id"` configurable key required by LangGraph, and legacy loop/checkpoint
structures may continue to carry internal run-oriented field names where a
rename would change checkpoint compatibility. These names are not part of the
SDK, CLI, result, or stream public contract.

## Stable public result types

`AgentResult` becomes a frozen, stable projection:

```python
AgentResult(
    answer: str | None,
    status: Literal["done", "paused", "failed"],
    files: tuple[str, ...],
    tool_calls: tuple[AgentToolCall, ...],
    evidence: tuple[AgentEvidence, ...],
    citations: tuple[AgentCitation, ...],
    usage: AgentUsage,
    diagnostics: tuple[AgentDiagnostic, ...],
    session_id: str | None,
    turn_id: str,
    stop_reason: str | None,
    pause: AgentPause | None,
    workspace_path: str | None,
    groundedness: bool,
    insufficient_evidence: bool,
    plan: AgentPlan | None,
    plan_events: tuple[PlanEvent, ...],
)
```

The public result removes:

- `raw`;
- `thread_id`;
- the `run_id` compatibility property;
- `Any`-typed citations, diagnostics, plan, and plan events;
- direct internal `AgentRunResult`, `ToolResult`, or runtime diagnostic values.

`AgentToolCall` contains a stable call ID, name, JSON arguments and structured
output, error fields, retry/truncation flags, and latency where available.
`AgentPause` contains the request ID, pause kind, question, bounded tool
summaries, options, and JSON context. Result-side dynamic payloads use a
recursive `JsonValue` type, never `Any`.

`AgentUsage` retains the current token fields and adds the production fields
currently obtained through `raw`, including model-call count,
`tool_schema_bytes`, and the existing bounded latency/CLI metrics needed by the
quality gate and verbose CLI.

There is one canonical Plan representation. `AgentPlan` and `PlanEvent` are
promoted or moved to stable public ownership and consumed by the runtime;
another public Plan implementation is not created. Legacy checkpoint module
paths are decoder compatibility details, not public aliases.

The package-root exports become:

```text
Agent
AgentResult
AgentUsage
RAGKnowledgeConfig
StreamEvent
EventType
ModelSpec
```

Additional result component types remain importable from
`agent_runtime.result` without all being promoted to package-root concepts.
Model assembly types such as `ModelCatalog`, `ModelControlPlane`, `ModelPolicy`,
`ModelRuntimeSpec`, and `ModelSessionState` are removed from the package root.

## Stream event contract

The existing canonical event type is adjusted rather than supplemented:

- `run_id` becomes `turn_id`;
- `session_id` becomes `str | None`;
- the loop counter `turn` becomes `iteration`;
- `data` becomes `dict[str, JsonValue]`;
- ordering, sequence assignment, event kinds, cancellation, and sink behavior
  remain unchanged.

One-shot event projection suppresses the internal Session ID. Conversation
events retain the formal Session ID. Event producers continue to publish into
the same canonical source.

## Knowledge configuration

The sole high-level knowledge configuration is:

```python
class RAGKnowledgeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    storage_root: Path = Path(".rag")
    embedding_model: str | None = None
    reranker_model: str | None = None
    vector_backend: Literal["milvus", "sqlite"] = "milvus"
    vector_namespace: str | None = None
    vector_collection_prefix: str | None = None
```

The model is JSON/YAML serializable. Paths serialize as strings. It contains no
source name, source registry key, catalog entry, resolver, or connection
secret.

- `knowledge=None` disables knowledge-tool assembly.
- A `RAGKnowledgeConfig` explicitly enables one RAG provider.
- Storage root, namespace, and collection prefix retain the existing ability to
  select an indexed knowledge store without inventing named-source semantics.
- A configured provider that cannot be opened reports a configuration
  error/diagnostic. It cannot silently disappear from the tool surface.

The only high-level secret environment name is `AGENT_VECTOR_DSN`. The SDK and
Agent CLI do not accept the DSN as a constructor or command-line argument. A
lower-level provider factory may accept a secret value through internal
dependency/configuration injection for tests and deployments. The DSN is never
stored in RuntimeBinding, checkpoints, result DTOs, or logs. Broad historical
fallback names such as `VECTOR_DSN` are not retained in the high-level path.

`KnowledgeSearchInput.constraints` is removed because it currently has no
mapping to `QueryOptions` or a lower-level metadata filter. Because the schema
forbids extras, callers that continue to send `constraints` receive validation
failure rather than silently ignored behavior.

No Source Registry, Knowledge Catalog, or named resolver is introduced.

## RuntimeBinding v2

The persisted binding becomes:

```python
class RuntimeBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = 2
    model_alias: str | None = None
    workspace_path: str | None = None
    knowledge: RAGKnowledgeConfig | None = None
```

### v1 mapping

An object without `schema_version` is v1.

| v1 value | v2 value |
| --- | --- |
| `agent_type` | dropped |
| empty `knowledge` tuple | `knowledge=None` |
| non-empty `knowledge` tuple | names dropped; sibling effective RAG values become one config |
| `rag_storage_root` | `knowledge.storage_root` |
| `embedding_model_alias` | `knowledge.embedding_model` |
| `reranker_model_alias` | `knowledge.reranker_model` |
| `vector_backend` | `knowledge.vector_backend` |
| `vector_namespace` | `knowledge.vector_namespace` |
| `vector_collection_prefix` | `knowledge.vector_collection_prefix` |
| `vector_dsn` | never persisted; resolved at runtime |

If the old knowledge-name tuple is empty, sibling RAG values do not enable
knowledge during migration. This preserves current behavior. If it is
non-empty, the names are discarded because they never selected a different
source; migration must not retroactively give them catalog semantics.

Both `agent_sessions.runtime_json` and `agent_turns.runtime_json` are migrated.
Store opening performs one atomic eager migration:

1. read and validate every legacy payload;
2. normalize every payload to canonical v2 JSON;
3. update both tables only after all rows validate;
4. roll back the complete migration on any error and identify the affected
   Session or Turn;
5. reject unknown v2 fields rather than ignoring configuration typos.

## SQLite migration

`agent_sessions` gains:

```sql
kind TEXT NOT NULL DEFAULT 'conversation'
    CHECK (kind IN ('one_shot', 'conversation'))
```

All rows in an existing database become `conversation`. The database default
exists for migration safety only. New application calls to `create_session`
must provide `SessionKind` explicitly.

`SessionRecord` gains `kind`. Kind cannot be updated by runtime initialization,
archiving, unarchiving, or Turn creation. A listing-oriented index includes
kind, archive state, and update order. Turn, event-history, and checkpoint table
schemas do not change.

Query behavior is:

- ordinary `list_sessions` returns Conversation Sessions only;
- `session list --all` explicitly includes both kinds, all workspaces, and
  archived rows, and displays kind;
- `latest_session` returns a Conversation only;
- `latest_resumable_turn` searches paused/interrupted Turns in both kinds;
- expired RUNNING Turns are normalized to INTERRUPTED before selection;
- direct `session show/archive/unarchive/delete` may address a one-shot Session
  as a debugging/maintenance operation;
- `session show` never suggests `chat --session-id` for a one-shot Session.

Both the facade and `AgentService.chat` enforce the Conversation-only
continuation rule so an internal caller cannot bypass it.

## CLI contract

### `agent run`

Remove:

- `--agent`;
- `--turn-id` and `--run-id`;
- repeatable `--knowledge TEXT`;
- `--input-file` alias;
- `--tool`, `--disable-tool`, and discovery overrides;
- flat storage/embedding/reranker/vector options;
- `--budget` compatibility alias.

Retain or add:

- `--model/-m`;
- `--file/-f`;
- one `--knowledge-config PATH` JSON/YAML file;
- `--checkpoint-db`;
- `--max-turns`;
- canonical `--max-tokens-total`;
- write/execute pre-authorization;
- current interactive approval, verbose, and non-interactive behavior.

### `agent chat`

Remove `--agent`, `--budget`, and every flat RAG/vector option. Retain
`--session-id`, `--last`, model and Turn limits, and add
`--knowledge-config PATH` for creation of a new Conversation. Passing a
knowledge config while continuing an existing Session is rejected; the
persisted RuntimeBinding is authoritative.

### `agent resume`

Retain Turn ID, `--last`, `--all`, `--action`, and `--input`. Remove the
`--decision` compatibility alias and hidden `--vector-dsn`; the sensitive DSN
comes from the explicit runtime environment.

### Session commands

Ordinary listing and interactive `/sessions` show Conversation Sessions only.
`session list --all` is the debugging entry for both kinds. `session show`
prints kind and non-sensitive knowledge binding data.

The separate general-purpose RAG ingest, retrieval, and benchmark CLI retains
its low-level vector configuration. This design removes those parameters only
from the high-level Agent constructor and Agent commands.

## Deletion inventory

### Whole production files confirmed outside the product runtime path

- `rag/agent/core/agent_service_factory.py`
- `rag/agent/core/compiler.py`
- `rag/agent/core/subagent_runner.py`
- `rag/agent/core/agent_as_tool.py`
- `rag/agent/core/agent_tool_contract.py`
- `rag/agent/core/delegation.py`
- `rag/agent/core/runtime_ports.py`
- `rag/agent/graphs/base.py`
- `rag/agent/graphs/__init__.py`
- `rag/agent/graphs/nodes/__init__.py`

Their remaining references are tests, lazy public exports, or other members of
the same legacy closure.

### Whole files deleted after their current wrapper reference is removed

- `rag/agent/core/registry.py`, after the runtime builder directly uses the
  single `GENERIC_AGENT` definition;
- `rag/agent/runner/python_runner.py`, after PrimitiveOps Python execution is
  removed;
- `rag/agent/runner/__init__.py`.

`core/registry.py` is not currently reference-free; the implementation must
first replace its sole production use rather than classify it as unused.

### Whole tests removed after equivalent canonical coverage exists

- `tests/agent/test_agent_graph_compiler.py`
- `tests/agent/test_builtin_subagent_runner.py`
- `tests/agent/test_python_runner.py`
- `tests/agent/test_primitive_ops.py`

Still-valid structured-file preview coverage moves to
`tests/agent/test_file_manifest.py` before the PrimitiveOps tests are removed.

### Partial cleanup

- `agent_runtime/agent.py`: remove the old constructor and execution
  parameters, mixed stream method, compatibility helpers, and private `Any`
  annotations; add the shared private facades.
- `agent_runtime/result.py`: replace internal/raw projections with stable DTOs.
- `agent_runtime/__init__.py`: reduce package-root exports.
- `agent_runtime/runtime/builder.py`: remove AgentRegistry/agent-type selection
  and duplicate auto-RAG parameter resolution.
- `agent_runtime/knowledge_providers/rag.py`: consume one knowledge config and
  internal secret injection.
- `rag/agent/sessions.py`: add SessionKind, v2 binding migration, query rules,
  and kind validation.
- `rag/agent/service.py`: remove the `policy` alias, ignored constructor
  parameters, `initial_state_from_config`, `run_with_config`, legacy `resume`,
  and `AgentRunResult.from_state`; retain `_run_request`, `chat`, streaming, and
  `resume_turn`.
- `rag/agent/core/definition.py`: remove `agent_type`, `description`, unused MCP
  declaration fields, unused retrieval-hint fields, `thinking`, and the
  backward-compatible `allowed_tools` property. Keep active tool-decision and
  ToolPolicy behavior.
- `rag/agent/core/context.py`: remove delegation-only helpers and fields proven
  behaviorless, with legacy checkpoint normalization before structural
  deletion. Keep active token budget, cancellation, memory, max-turn, and
  ToolPolicy state.
- `rag/agent/primitive_ops.py`: remove the second list/read/write/Python
  execution implementation while preserving `FileKind`, `CellValue`,
  `CandidateHeaderRow`, `StructuredTableProbe`, and `StructuredProbeOutput` at
  checkpoint-stable module paths.
- `rag/agent/file_manifest.py`: own pure structured-preview helpers and stop
  advertising removed tools.
- `rag/agent/core/observations.py`: remove obsolete write-file, run-Python, and
  structured-probe special cases while retaining the production read-file
  behavior.
- `rag/agent/tools/integrations/knowledge.py`: delete `constraints`.
- `rag/agent/streaming/events.py` and event producers: adopt the public Turn
  naming and JSON type without changing event behavior.
- `rag/agent/cli.py`: remove duplicate RAG resolution, AgentRegistry selection,
  legacy CLI options, direct Service construction, and raw-result access.
- `rag/agent/core/checkpointing.py`: normalize legacy removed fields and retain
  explicit fixtures for old PrimitiveOps/RunConfig data.
- `rag/agent/__init__.py`, `rag/agent/core/__init__.py`, and
  `rag/agent/builtin/__init__.py`: remove legacy lazy exports and registry
  factories.
- `scripts/agent_model_quality_gate.py`, `scripts/agent_session_smoke.py`, and
  Agent CLI smoke scripts: consume stable result fields instead of `raw` or
  facade-private stores.
- README, runbook, and naming documentation: replace the historical
  run/thread/stream/tool-option contracts. Historical design documents may be
  marked superseded rather than rewritten as if they had never existed.

The production `rag/agent/tools/integrations/subagent.py` is explicitly kept.
It is the current canonical bounded subagent tool and is distinct from the
legacy delegation/factory/graph closure. Its child execution still enters the
same `_run_request`, AgentLoop, and ToolExecutor path.

## Failure behavior

- Continuing a one-shot Session fails before Turn allocation.
- Continuing an archived or busy Conversation preserves the existing state
  checks.
- Resuming a completed or failed Turn remains invalid.
- Live RUNNING leases remain invalid; expired RUNNING leases first become
  INTERRUPTED.
- Unknown RuntimeBinding or RAG config fields fail validation.
- Legacy binding migration is all-or-nothing.
- Missing or invalid configured knowledge storage produces an explicit
  provider/tool diagnostic rather than silent disablement.
- Public callers cannot supply their own Turn ID after the compatibility
  parameter is removed.

## TDD implementation slices

The implementation is one design on one branch, delivered in four gated TDD
slices. A sweeping rewrite would make runtime regressions hard to localize;
multiple PRs would require temporary compatibility layers that contradict the
cleanup goal.

### Slice 1: Session and persistence

Start with failing tests for:

- legacy SQLite Session rows becoming Conversation;
- explicit new Session kind creation;
- one-shot hiding and `--all` visibility;
- Conversation-only `latest_session`;
- cross-kind resumable-Turn selection;
- one-shot chat rejection at facade and service boundaries;
- v1 RuntimeBinding migration in both Session and Turn rows;
- secret non-persistence and rollback on an invalid row.

### Slice 2: facade, result, and streaming

Start with public signature tests and failing behavior tests for:

- one-shot versus Conversation result identifiers;
- exact `run/arun/chat/achat/resume/aresume` semantics;
- `astream` and `astream_chat` lifecycle behavior;
- event Turn naming and one-shot Session-ID suppression;
- cancellation and canonical event ordering;
- absence of `raw`, `run_id`, `thread_id`, and public `Any`;
- stable pause, tool-call, diagnostic, usage, evidence, and citation DTOs.

### Slice 3: knowledge and CLI

Start with failing tests for:

- serializable and extra-forbidden `RAGKnowledgeConfig`;
- config-preserving Session continuation;
- explicit environment secret injection and secret non-persistence;
- absent `KnowledgeSearchInput.constraints`;
- exact CLI option removals/additions;
- CLI use of the shared facade and stable result fields;
- unchanged interactive approval, failure exit codes, max-turn behavior, and
  streaming display.

### Slice 4: legacy deletion

Before deletion, add import/reference guards and checkpoint fixtures. Then
remove the listed modules and migrate still-valid tests. Verification must show
no production references to the deleted compiler/factory/runner/graph/
delegation closure and no second PrimitiveOps executor.

## Final verification

At minimum:

- focused Session, facade, result, stream, knowledge, and CLI suites;
- unchanged PR #26-#28 contract suites for max turns, ToolPolicy,
  approval/resume, circuit breaking, ToolExecutor, checkpointing, and
  streaming;
- legacy SQLite and checkpoint fixture tests;
- touched-file Ruff and mypy;
- `uv run lint-imports`;
- `git diff --check`;
- full pytest, with any pre-existing timing flake separately reproduced and
  reported rather than hidden;
- CLI delivery smoke and cross-process Session smoke;
- a proportionate real runnable-path smoke before a delivery claim.

## Non-goals

- Removing `chat/achat` in favor of a single public method.
- Adding named knowledge source resolution, a Source Registry, a Knowledge
  Catalog, or provider discovery product.
- Adding streaming resume.
- Replacing AgentLoop, ToolExecutor, checkpointing, ToolPolicy, or the canonical
  event source.
- Changing general RAG ingest/benchmark configuration surfaces.
- Renaming third-party checkpoint wire keys at the cost of legacy checkpoint
  compatibility.
- Automatically deleting completed one-shot Sessions or adding retention/GC
  policy.
