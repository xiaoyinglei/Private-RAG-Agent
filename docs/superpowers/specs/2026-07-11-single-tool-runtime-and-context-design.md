# Single Tool Runtime and Canonical Agent Context Design

**Status:** Approved in design discussion on 2026-07-11
**Supersedes:** `2026-07-06-claude-like-tooling-main-path-design.md`

## 1. Decision

Replace the current dual tool runtime with one destructive cutover.

The final runtime has one implementation of each control responsibility:

```text
Tool
ToolRegistry
select_tools(...)
can_use_tool(...)
ToolExecutor
ToolResult
```

Supporting value types such as `ToolCall`, `ContentBlock`, resolved targets,
usage records, and traces are data contracts, not additional runtime layers.

The refactor may break and delete internal APIs. The only compatibility
boundaries are:

- the `agent` CLI;
- `from agent_runtime import Agent` and its documented behavior;
- explicitly committed checkpoint and persistence formats.

For the CLI and `agent_runtime.Agent`, compatibility preserves command names,
method signatures, accepted parameter types, and documented result shapes. The
default execution behavior intentionally changes from the current accidental
tool-less surface to the approved coding-agent resident baseline. Section 9.4
defines the exact behavior of every existing public tool option.

An old import may temporarily re-export a final type. It must not contain
runtime logic, state, conversion, routing, or a second registry, executor, or
visibility path.

## 2. Problem Being Solved

The repository currently contains two partially overlapping tool systems:

- `rag/agent/tools/*` contains the older, richer contracts and execution path;
- `rag/agent/tooling/*` contains a newer but thinner main path;
- the service assembles both registries;
- initial execution and resume can reach different executors;
- visibility is represented by catalog, deferred-store, surface-policy, and
  legacy compatibility concepts;
- the public default Agent can become tool-less;
- model requests and tool schemas change in ways that prevent reliable prompt
  caching;
- model-call usage is available at the gateway but is discarded before it
  reaches the loop and public result.

The goal is not to finish the transition layer. The goal is to delete the
transition and leave one understandable coding-agent runtime.

## 3. Goals

1. Make the default product a CLI-first coding and file agent with Python SDK
   parity.
2. Keep a small, accurate resident tool surface instead of loading every
   installed tool.
3. Add knowledge, MCP, skills, and subagents as explicit or discoverable
   extensions without a second runtime.
4. Make tool validation, permission, cancellation, execution, normalization,
   and tracing pass through one choke point.
5. Make the model-facing context deterministic, cache-aware, checkpoint-safe,
   and measurable.
6. Preserve old durable records through decoding or migration at the
   persistence boundary, not through legacy runtime paths.

## 4. Non-goals

- Do not rename the entire `rag.agent` package in this refactor.
- Do not rewrite vector retrieval, ingestion, MCP client lifecycles, skill
  loading, or child-agent loops.
- Do not convert source-code search into RAG.
- Do not unify every auxiliary LLM stage such as RAG answer generation,
  summarization, or comparison. This design converges the public agent-loop
  model turn used by the CLI and `agent_runtime.Agent`.
- Do not add an Anthropic provider as part of the Tool cutover. The current
  runtime supports MLX, Ollama, and OpenAI-compatible providers. A future
  Anthropic adapter may use native deferred tools without changing core tool
  semantics.
- Do not introduce a generic `invoke_tool`, `tool_repl`, or arbitrary code
  dispatch mechanism to keep the tool schema artificially stable.

## 5. Final Package Ownership

The canonical internal package is `rag/agent/tools/`:

```text
rag/agent/tools/
  __init__.py
  tool.py
  registry.py
  selection.py
  permissions.py
  executor.py

  builtins/
    filesystem.py
    search.py
    shell.py
    planning.py

  integrations/
    knowledge.py
    mcp.py
    subagent.py
```

`rag/agent/tooling/` is deleted after cutover. `rag/agent/capabilities/` no
longer owns a tool catalog or activation store. Skill loading remains under
`rag/agent/skills/`; a skill is not automatically a tool.

The canonical model request belongs to the model/provider boundary, not to the
tools package.

## 6. Tool Contract

`Tool` is the installed executable contract. It replaces `BaseTool`, both
`ToolSpec` types, `ToolCard`, separate runner registration, and model-facing
formatter registration.

Conceptual shape:

```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: Mapping[str, JsonValue]
    validate_input: ValidateInput
    run: ToolRunner
    normalize_output: NormalizeOutput
    output_schema: Mapping[str, JsonValue] | None
    static_effects: frozenset[ToolEffect]
    resolve_use: ResolveToolUse
    timeout_seconds: float
    max_model_output_bytes: int
    execution_revision: str
    idempotent: bool
    concurrency_safe: bool
    cancellation_mode: CancellationMode
    interrupt_behavior: InterruptBehavior
```

### 6.1 Input schemas

- Builtins may generate JSON Schema and validation from Pydantic models.
- MCP tools preserve their original JSON Schema, including `$ref`, `$defs`,
  `oneOf`, `anyOf`, and other supported constraints.
- External references must not trigger implicit network resolution during
  validation. MCP schemas may resolve local references contained in the same
  schema document.
- `validate_input` returns canonical validated arguments or a structured
  validation error. It must not silently discard extra arguments.
- Builtin schemas use `additionalProperties: false` unless the tool explicitly
  accepts an open object.

### 6.2 Effects and targets

Effects describe facts, not approval decisions. Examples include:

```text
READ_WORKSPACE
WRITE_WORKSPACE
EXECUTE_PROCESS
NETWORK
DESTRUCTIVE
```

The effective effects are conservative:

```text
effective_effects = static_effects union resolved_effects
```

Dynamic analysis may add restrictions; it must not erase a static safety
floor. Unknown or unparseable shell behavior is treated conservatively.
Workspace escape is a non-bypassable guard, not an approvable higher risk.
MCP annotations are hints and cannot weaken local policy.

### 6.3 Execution semantics

Idempotency, concurrency, cancellation, and interrupt behavior are executable
contract facts:

- `execution_revision` is a stable source-supplied revision for runner and
  output-normalization semantics; changing either requires a new revision;
- `idempotent` controls whether an ambiguous operation may reuse the same
  operation identity or requires reconciliation;
- `concurrency_safe` permits consideration for parallel execution but does not
  override target-conflict checks;
- `cancellation_mode` is one of cooperative cancellation, managed child
  process, remote best-effort, or non-cancellable;
- `interrupt_behavior` says whether a user interrupt cancels immediately or
  waits for the current atomic operation to finish.

A locally side-effecting tool cannot register as non-cancellable. Parallel
execution requires every tool to be concurrency-safe and their resolved
targets/effects to be non-conflicting. The executor, not the model, makes that
decision. Sandbox selection remains derived from resolved effects and runtime
environment; it is not a self-declared `sandboxed=True` escape hatch.

### 6.4 Output normalization

`normalize_output` converts a runner-specific value into canonical content.
It is not a CLI, Rich, JSON, or UI presenter.

The output order is:

```text
runner raw output
-> normalize_output
-> validate canonical output
-> bound model-visible content or externalize attachments
-> ToolResult
```

## 7. ToolResult

`ToolResult` is the single result used by the executor, AgentLoop, checkpoint,
and model transcript.

It contains at least:

```text
tool_call_id
tool_name
content: ordered content blocks
structured_content: JSON-compatible value or null
is_error
error_code
truncated
metadata
attachments: stable artifact references
```

Rules:

- `content` supports text and future image/resource blocks; it is not limited
  to one string.
- metadata is runtime-only unless explicitly projected into model content.
- attachments are stable references, not embedded arbitrary Python objects.
- large structured output is externalized rather than truncated into invalid
  JSON.
- the model-visible content is fixed when the result first enters the
  canonical transcript. Resume and later turns do not rerun a formatter.

## 8. ToolRegistry and Assembly

There is one `ToolRegistry` implementation, but it is not a global singleton.
Each runtime or session may own an independent instance.

The registry only:

- registers a `Tool`;
- rejects duplicate names;
- gets a tool by name;
- lists installed tools;
- freezes into an immutable snapshot.

It does not own current visibility, activation, approval, provider state, or
execution state.

`build_tool_registry(...)` is a plain assembly function. It installs builtins
and adapts explicitly configured knowledge, MCP, and subagent capabilities into
the current registry. Assembly order is deterministic. The registry freezes
before the first model request. An in-flight run never observes MCP reconnects
or later registration mutations.

## 9. Resident and Discoverable Tools

The first coding-agent baseline sends these resident tools:

```text
list_files
search_text
read_file
apply_patch
run_command
update_plan
```

When hidden extension tools exist and the public run enables discovery,
`find_tools` is also resident. Discovery is disabled by default because the
stable public `allow_discovery_tools` option defaults to false.

This set is an evaluated baseline, not a permanent law. Changes require ACI
evaluation rather than ad-hoc additions.

### 9.1 Workspace search

`search_text` is structured Grep, not vector retrieval. It supports bounded
pattern or literal search, path, glob, context lines, and result limits.
Source repositories are not chunked, embedded, or indexed by default.

RAG is installed only through explicit knowledge configuration such as
`Agent(knowledge=...)`, which may make `search_knowledge` resident for that
runtime.

### 9.2 Visibility state

Visibility uses names, not another registry:

```text
installed: present in the frozen registry snapshot
resident: product assembly sends it every turn
active: discovered during this run and sent on later turns
```

The only visibility function is:

```python
select_tools(
    registry_snapshot,
    resident_names=...,
    active_names=...,
    schema_budget=...,
) -> tuple[Tool, ...]
```

The returned order is deterministic:

1. resident tools in the frozen product order;
2. explicitly configured resident extensions in frozen order;
3. discovered tools in activation order.

Provider adapters must preserve this order. They must not sort the complete
visible set again.

### 9.3 find_tools

`find_tools(query, limit)` searches metadata from the same frozen Registry.
There is no `ToolCatalog`, `DeferredToolStore`, or activation service.

The first implementation uses an explainable inverted index or BM25 over:

- name and source namespace;
- description;
- input property names and descriptions;
- lightweight multilingual search aliases.

It does not use embeddings, a vector database, or another routing model.

One call returns and proposes activation of at most five matches. Active tools
are monotonic within a run. No first-version LRU silently removes a tool the
model previously saw. If count or schema budget would be exceeded,
`find_tools` returns an explicit recoverable error.

The runner returns matched and proposed names in `ToolResult.metadata`.
AgentLoop applies the ToolResult and active-name delta together before one
checkpoint save. The runner and executor do not write checkpoints.

### 9.4 Public CLI and Agent option mapping

The existing CLI flags and `Agent.run/arun/stream` parameters remain accepted.
Their final semantics are:

| Public option | Final meaning |
|---|---|
| `tools=None` or an empty sequence | use the approved default resident coding set plus product-configured resident extensions |
| non-empty `tools=[...]` | replace the default resident-name set with exactly these installed names, preserving caller order |
| `disabled_tools=[...]` | subtract these names from resident, active, discovery results, and execution eligibility |
| `allow_write_tools=True` | pre-authorize eligible write effects subject to all hard guards; false leaves the decision at ask/deny rather than hiding the schema |
| `allow_execute_tools=True` | pre-authorize eligible process execution subject to sandbox and hard guards; false leaves the decision at ask/deny rather than hiding the schema |
| `allow_discovery_tools=True` | expose `find_tools` when hidden discoverable tools exist; false prevents client-side discovery and active-set growth |

Unknown explicit `tools` or `disabled_tools` names fail before the first model
call with a clear configuration error. A disabled name wins over every other
source. Explicit knowledge configuration contributes `search_knowledge` to
the default resident extensions, but a non-empty `tools=[...]` remains an exact
caller override. These semantics apply identically in CLI, sync Agent, async
Agent, streaming, and resume.

The precedence for discovery combinations is authoritative:

1. `disabled_tools` always wins, including for `find_tools`.
2. With `tools=None` or an empty sequence, `allow_discovery_tools=True`
   automatically adds `find_tools` only when hidden discoverable tools exist.
3. With a non-empty `tools=[...]`, the list remains exact. Enabling discovery
   permits `find_tools` but does not append it; the caller must include
   `find_tools` explicitly.
4. Explicitly listing `find_tools` while `allow_discovery_tools=False` is a
   configuration error before the first model call, not a silent suppression.
5. Without `find_tools`, active names cannot grow through client-side
   discovery even when the permission gate is enabled.

## 10. Permission and Execution

Every model tool call follows one executor path:

```text
lookup installed tool
-> verify schema was exposed for this turn
-> validate input
-> resolve conservative effects and canonical targets
-> enforce non-bypassable guards
-> determine the execution boundary or sandbox
-> can_use_tool(...)
-> obtain approval when the decision is ask
-> invoke inside the selected boundary
-> normalize raw output
-> validate canonical output
-> externalize or bound model-visible content
-> return ToolResult
```

`can_use_tool(...)` returns only `allow`, `ask`, or `deny` plus a reason. Human
interaction resolves `ask`; it does not live inside the permission function.
Approval cannot bypass workspace, cwd, sandbox, target, output, or command
guards.

Trace recording covers every exit, including unknown tools, schema-not-sent,
validation failure, denial, timeout, cancellation, and runner failure.

The schema-exposure check uses the originating model request recorded on the
ToolCall, not the current turn's visible set. Approval pauses, discovery,
compaction, and resume must not change the evidence used for this check.

### 10.1 Cancellation

Timeout means the runtime attempts real termination, not merely returning
early while a thread continues:

- shell and Python execute in isolated child processes or process groups;
- timeout sends graceful termination, escalates to forced termination, and
  waits for process reaping;
- cooperative async runners must acknowledge cancellation;
- a non-cancellable local side-effect runner is rejected at registration or
  forced behind a cancellable process boundary;
- remote cancellation cannot promise rollback of an already accepted side
  effect.

Timeout results distinguish at least `timeout_cancelled` from
`timeout_outcome_unknown`.

## 11. Canonical Agent Model Request

The public agent-loop path builds one provider-neutral request. The current
OpenAI-shaped `rag.agent.tooling.ModelRequest` is not retained.

Conceptual shape:

```python
class ModelRequest:
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolDefinition, ...]
    tool_choice: ToolChoice
    settings: ModelSettings
    request_id: str
    exposed_tool_names: tuple[str, ...]
    prompt_revision: str
    toolset_revision: str
```

`ToolDefinition` is an immutable model-facing projection of `Tool`: name,
description, input schema, and provider-supported model annotations. It never
contains a runner, permission callback, or mutable runtime state.

Every accepted `ToolCall` carries a checkpoint-safe origin record containing:

```text
request_id
toolset_revision
exposed_tool_names
```

The AgentLoop persists this origin with the call before scheduling execution.
`ToolExecutor` checks `schema_not_exposed` against that record. A single mutable
`LoopState.sent_schema_names` field is not an authoritative execution input and
is removed with the old path.

Core semantics are ordered from stable to dynamic:

```text
selected tools
stable instructions
frozen run context
initial user task
canonical context transcript
current dynamic tail
```

This is a semantic order, not a universal wire layout. Provider adapters own
wire rendering.

The main path must not maintain both `AgentMessageAssembler` and a separate
structured-output loop prompt. Providers without native tool syntax may render
the same canonical request into their supported input format; they do not get
a second loop or visibility policy.

Auxiliary RAG, summarize, compare, and generation stages may keep their scoped
context assembly during this refactor.

## 12. Context Revisions, Compaction, and Caching

Prompt caching is a consequence of deterministic context organization. It is
not a second Tool runtime or a provider-neutral cache service.

### 12.1 Context revision

Within one context revision:

- selected tools, schemas, descriptions, stable instructions, and frozen run
  context do not change;
- the canonical transcript is append-only;
- tool results retain their original model-visible content;
- tool choice, thinking settings, and parallel-tool settings remain stable.

The following explicitly create a new revision:

- client-side tool activation that changes the rendered tool set;
- context compaction;
- prompt or schema revision;
- model or cache-relevant setting changes.

Compaction is allowed and required. It closes the previous revision and creates
a new frozen summary plus bounded tail. The provider-visible context is not
promised to grow without limit.

Skill activation appends a canonical context event. It must not rewrite the
old system prompt. Initial memory is frozen for the revision; new memory enters
through an append or a new compaction revision.

### 12.2 Stable prompt layout

The system prompt excludes iteration counters, tool success/error counts,
timestamps, run IDs, and a repeated list of visible tool names. The provider
already receives tool schemas.

Tool schemas use deterministic name, property, and array ordering. Dynamic
class names, set iteration, memory addresses, and unstable JSON serialization
must not enter the canonical request.

### 12.3 Cache diagnostics

Revisions and hashes diagnose and reproduce requests; real provider usage is
the source of truth for cache behavior.

Each model call records:

```text
logical_input_tokens
uncached_input_tokens
cache_read_input_tokens
cache_write_input_tokens
output_tokens
usage_source
bounded raw_provider_usage
prompt_revision
toolset_revision
provider_wire_hash
```

Unsupported or unreported cache fields are `None`, not zero. Tokenizer
estimates never masquerade as cache hits.

The model-call result carries usage through `LLMGateway`, the model-turn
provider, `ModelTurnEnvelope`, AgentLoop state, checkpoint diagnostics, and
`AgentResult`. Streaming adapters must consume the provider's final usage event
when available.

Model pricing configuration may add optional cache-read and cache-write rates.
These additions are backward-compatible. Provider-specific billing remains at
the provider/pricing boundary.

## 13. Provider Boundary

One canonical request is serialized by provider adapters:

- OpenAI-compatible adapters render messages, tools, tool choice, cache key,
  and supported cache options;
- local adapters render the same request into their supported native or flat
  form;
- a future Anthropic adapter may render cache-control blocks and native
  deferred tools.

Capabilities are declared per resolved model or adapter, not guessed from
natural-language task text.

Provider-native deferred discovery may emit a canonical discovery event. The
AgentLoop applies the event to the same active-name state and checkpoint. The
adapter never owns a registry, permission path, executor, or persistent
activation state.

There is no Anthropic implementation requirement in this cutover. When one is
added, native deferred loading is an early cost optimization because it can
preserve the provider's stable tool prefix; it must still obey the canonical
runtime state described here.

## 14. Persistence and Resume

New checkpoints persist canonical model transcript content rather than
rebuilding it from ToolResult formatters on every turn.

They also persist a versioned model-facing tool manifest for the run:

```text
resident and explicit tool order
active tool activation order
tool name
canonical description hash
canonical input-schema hash
static-effects hash
execution-contract hash, including execution revision, idempotency,
concurrency, cancellation, interrupt behavior, and output-schema revision
toolset revision
provider serializer revision
```

Resume guarantees:

- the same canonical transcript semantics;
- the same tool and prompt revisions;
- the same provider-visible wire hash when the provider serializer revision is
  unchanged.

It does not promise byte-identical wire output across arbitrary serializer
upgrades.

Resume compares the rebuilt frozen Registry projection with the persisted
manifest before any pending tool executes:

- if the manifest and serializer revision match, the run retains its toolset
  revision and deterministic wire-hash guarantee;
- if a pending or paused ToolCall's tool is missing or its executable contract
  changed, resume pauses with `tool_definition_changed` and requires explicit
  reconciliation; it never executes against the new definition silently;
- if no pending call depends on the drifted definition, resume creates an
  explicit new context/toolset revision, removes unavailable active names,
  and records the drift diagnostic. The old cache-hit guarantee ends at that
  revision boundary.

The persisted manifest is evidence and model-facing replay data, not an
executable registry. Execution always uses the newly assembled registry after
compatibility checks.

Legacy checkpoints remain readable through migration:

```text
old checkpoint
-> decode old ToolResult and ledger data
-> rebuild canonical transcript once
-> write the new checkpoint representation on the next save
```

Migration lives in checkpoint decoding. It does not preserve legacy registry,
visibility, executor, formatter, or resume paths.

Active tool names are persisted, not Tool objects. If a missing active tool has
no dependent pending or paused call, the runtime removes the name while
creating the new revision and emits a bounded diagnostic. If a dependent call
exists, the name and origin evidence remain intact until explicit
`tool_definition_changed` reconciliation resolves the call.

## 15. Source Boundaries

- MCP adapters register concrete MCP tools. MCP client/server lifecycle and
  reconnect state stay outside ToolRegistry.
- `search_knowledge` may register as a Tool. Vector stores and ingestion stay
  outside ToolRegistry.
- Skills remain instructions, resources, and optional scripts. A skill may
  contribute explicit executable Tools, but is not automatically wrapped as a
  `SkillTool`.
- Subagent assembly may create a Tool such as `delegate_task`; the child
  AgentLoop and task lifecycle remain outside ToolRegistry.

## 16. Destructive Cutover

The cutover deletes or folds these concepts into the final runtime:

| Existing concept | Final treatment |
|---|---|
| two `ToolSpec` types | replace with `Tool` |
| `BaseTool` | delete |
| two runtime registries | replace with one implementation |
| `MCPToolRegistry` | adapter produces `Tool` values |
| `ToolCard` | delete; search uses Tool metadata |
| `ToolCatalog` | delete as source of truth |
| `DeferredToolStore` | delete; active names live in LoopState |
| `ToolSurfaceRequest/Decision` | delete |
| `ToolSurfacePolicy/DiscoveryPolicy` | delete |
| `RuntimeToolRegistryBuilder` | replace with plain assembly function |
| `legacy_tool_visibility` | delete |
| `ToolExecutionService` | delete after AgentLoop calls final executor |
| `ToolExecutorLoopAdapter` | delete |
| `ModelRequestBuilder` class | delete |
| runtime result formatters | replace with `normalize_output`; UI presentation stays outside |
| provider task-text regex gates | delete |
| `tool_search + activate_tools` | replace with `find_tools` |
| `tool_repl` | delete |

The public CLI and `agent_runtime.Agent` adapt directly to the final runtime.
No public request option may select a legacy path.

## 17. Testing and ACI Evaluation

### 17.1 Contract tests

- builtin Pydantic validation and complete JSON Schema constraints;
- MCP raw schema preservation, local `$ref`, `oneOf`, and `anyOf` validation;
- duplicate registry names fail loudly;
- registry freeze prevents mutation;
- deterministic assembly and selection order;
- unknown and schema-not-exposed calls return structured recoverable errors;
- every scheduled ToolCall persists its originating request ID, exposed names,
  and toolset revision;
- dynamic effects can add but not remove the static floor;
- workspace and cwd escape cannot be approved;
- allow/ask/deny ordering is exact;
- output normalization, validation, externalization, and truncation;
- every executor exit emits a trace;
- local process timeout actually terminates and reaps the child;
- remote ambiguous timeout reports `timeout_outcome_unknown`.
- non-idempotent ambiguous operations enter reconciliation rather than silent
  replay;
- parallel batches require concurrency-safe tools and non-conflicting targets.

### 17.2 Visibility and discovery tests

- the default resident set is exactly the evaluated coding baseline;
- installed does not imply visible;
- explicit knowledge configuration makes `search_knowledge` resident;
- `find_tools` searches multilingual metadata and returns bounded matches;
- hidden extensions expose `find_tools` only when discovery is enabled;
- disabled discovery never grows the active set;
- default tools plus enabled discovery auto-add `find_tools` only when hidden
  discoverable tools exist;
- an exact non-empty tools list adds `find_tools` only when explicitly named;
- explicit `find_tools` with disabled discovery fails before the model call;
- `disabled_tools` overrides every discovery combination;
- activation is monotonic and ordered;
- activation and ToolResult are checkpointed together;
- budget overflow is explicit and never silently evicts a tool;
- resume restores active names or diagnoses missing tools.
- public `tools`, `disabled_tools`, and `allow_*` options have identical tested
  semantics across CLI, sync, async, streaming, and resume.

### 17.3 Context and cache tests

- the same snapshot produces identical canonical tool JSON and prompt hashes;
- provider adapters preserve canonical tool order;
- iteration, timestamps, and counters do not change the stable prefix;
- ToolResult model content is not reformatted on later turns or resume;
- ten no-discovery turns retain one toolset and prompt revision while appending
  transcript content;
- one client-side discovery batch creates exactly one new revision;
- compaction creates an explicit new revision and bounded transcript;
- old checkpoints migrate once into canonical transcript storage;
- manifest drift with a pending call pauses before execution;
- a missing active tool with a dependent paused call is retained for
  reconciliation;
- safe manifest drift without pending calls creates one explicit new revision;
- normalized usage preserves provider cache reads and writes;
- missing cache usage is `None` and estimated usage is labeled;
- streaming and non-streaming usage reach the Agent result.

### 17.4 Model-level ACI evaluation

The evaluation set covers direct answers, file navigation, Grep, reading,
patching, commands, explicit knowledge, hidden MCP tools, subagents, hidden-tool
hallucination, similar-tool confusion, and Chinese tool discovery.

Report at least:

```text
surface recall
surface precision
tool choice accuracy
argument validity
unnecessary call rate
discovery recall@5
recovery rate
schema bytes and tokens
provider cache read/write tokens
```

Thresholds are set after measuring the current supported model baselines; they
are not invented in the architecture document.

### 17.5 Main-path acceptance

1. CLI and `agent_runtime.Agent` use the same registry, selection function,
   executor, canonical model request, and resume path.
2. No runtime import from the deleted tooling or legacy visibility packages is
   reachable from the public path.
3. The default coding agent can list, Grep, read, patch, run tests, and update a
   plan without discovery.
4. Knowledge, MCP, and subagent tools are explicit or discoverable without
   loading the full installed schema set.
5. Invalid, hidden, denied, timed-out, and failed calls return one ToolResult
   contract and remain model-recoverable where appropriate.
6. A timed-out local side-effect process is not still running after the result
   returns.
7. Cache metrics use real provider usage when available and survive through the
   public result and checkpoint diagnostics.
8. Full agent tests, public CLI/SDK smoke tests, checkpoint migration tests,
   scoped static checks, compile checks, and `git diff --check` pass.
