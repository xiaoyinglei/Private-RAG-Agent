# Single Tool Runtime and Canonical Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dual tool runtime with one CLI-first coding-agent tool path that has deterministic selection, checkpoint-safe execution, canonical model context, and real cache usage telemetry.

**Architecture:** Build the final contracts under `rag/agent/tools/`, project selected tools into one provider-neutral `ModelRequest`, and make `AgentLoop` own activation and checkpoint state while one `ToolExecutor` owns validation through normalized results. Cut the CLI and `agent_runtime.Agent` over once the final path is complete, then delete the old registry, catalog, surface, adapter, executor, formatter, and discovery code in the same branch.

**Tech Stack:** Python 3.12, Pydantic v2, `jsonschema`, asyncio, subprocess process groups, LangGraph checkpoint storage, OpenAI-compatible chat APIs, pytest, Ruff, mypy, uv.

---

## Execution Rules

- Execute in a new worktree created from the plan commit, not in the current dirty workspace.
- Preserve `/Users/leixiaoying/LLM/RAG学习` exactly as it is; its uncommitted WIP is not the implementation base.
- Do not copy `rag/agent/tooling/*` from the dirty workspace into the implementation worktree. Rebuild from the approved spec and committed code.
- Tasks 1-9 must not add another active path to the committed dual-runtime
  baseline. Task 10 removes both legacy public branches and leaves exactly one
  active path; every later commit preserves that invariant.
- Tasks 1-9 create final contracts, pure functions, provider serializers, and backward-compatible codecs, but must not wire them into `AgentLoop`, `AgentService`, `agent_runtime.Agent`, or CLI execution.
- Temporary new contracts may exist before cutover, but no adapter may route public execution between old and new registries or executors. All active runtime wiring changes are staged together and committed only in Task 10.
- Use TDD for every behavior change and commit after each task.
- Do not stage unrelated files. Every commit command below names its files explicitly.

## Final File Map

### Canonical tool runtime

| File | Responsibility |
|---|---|
| `rag/agent/tools/tool.py` | `Tool`, `ToolDefinition`, `ToolCall`, `ToolCallOrigin`, `ToolResult`, content/effect/execution value types, Pydantic and JSON-Schema validator factories |
| `rag/agent/tools/registry.py` | mutable assembly registry, immutable snapshot, duplicate rejection, deterministic manifest projection |
| `rag/agent/tools/selection.py` | one `select_tools(...)` function, public option precedence, bounded metadata search, `find_tools` Tool factory |
| `rag/agent/tools/permissions.py` | pure `can_use_tool(...)` allow/ask/deny decision and hard preflight helpers |
| `rag/agent/tools/executor.py` | only validation/permission/approval/sandbox/runner/normalization/trace execution choke point |
| `rag/agent/tools/builtins/filesystem.py` | `list_files`, `read_file`, `apply_patch` |
| `rag/agent/tools/builtins/search.py` | structured Grep `search_text` |
| `rag/agent/tools/builtins/shell.py` | cancellable `run_command` process-group runner |
| `rag/agent/tools/builtins/planning.py` | `update_plan` |
| `rag/agent/tools/integrations/knowledge.py` | knowledge runners projected into `Tool` values |
| `rag/agent/tools/integrations/mcp.py` | MCP schema/name/output adapter; no MCP lifecycle ownership |
| `rag/agent/tools/integrations/skills.py` | skill invocation and asset runners projected into Tools; no catalog/loader ownership |
| `rag/agent/tools/integrations/subagent.py` | child-agent runner projected into a Tool |

### Canonical model boundary

| File | Responsibility |
|---|---|
| `rag/agent/core/model_request.py` | typed canonical `ModelRequest`, model transcript, context/tool revisions, tool manifest, stable hashing |
| `rag/agent/core/messages.py` | provider-neutral ordered messages and tool-call origin records |
| `rag/providers/openai_wire.py` | OpenAI-compatible serialization/parsing and cache wire fields |
| `rag/providers/local_agent_wire.py` | deterministic MLX/Ollama canonical-request rendering and response parsing when native tools are unavailable |
| `rag/agent/core/llm_providers.py` | assemble one canonical agent-loop request and return draft plus usage |
| `rag/schema/llm.py` | normalized provider usage including cache reads/writes and bounded raw usage |

### Runtime integration

| File | Responsibility |
|---|---|
| `rag/agent/loop/state.py` | canonical transcript, active tool names, per-call origin, manifest/revision state |
| `rag/agent/loop/runtime.py` | model-turn scheduling, activation reducer, one executor call, checkpoint transition ownership |
| `rag/agent/core/checkpointing.py` | legacy decode, canonical transcript migration, manifest drift and reconciliation |
| `rag/agent/service.py` | one registry assembly and one loop wiring path |
| `agent_runtime/runtime/builder.py` | product composition of builtin and configured extension Tools |
| `agent_runtime/agent.py` | stable public parameters mapped to final selection and permission inputs |
| `rag/agent/cli.py` | stable CLI flags mapped identically to the SDK |

### Deleted after cutover

```text
rag/agent/tooling/
rag/agent/capabilities/catalog.py
rag/agent/capabilities/context.py
rag/agent/capabilities/tool_search.py
rag/agent/builtin_registry.py
rag/agent/tools/asset_tools.py
rag/agent/tools/base.py
rag/agent/tools/builtin_registry.py
rag/agent/tools/card.py
rag/agent/tools/catalog_assembly.py
rag/agent/tools/formatter.py
rag/agent/tools/formatters/
rag/agent/tools/generic_tools.py
rag/agent/tools/llm_tools.py
rag/agent/tools/mcp_adapter.py
rag/agent/tools/observation.py
rag/agent/tools/rag_answer_tools.py
rag/agent/tools/rag_semantic_tools.py
rag/agent/tools/rag_tool_runner.py
rag/agent/tools/rag_tools.py
rag/agent/tools/runtime_registry_builder.py
rag/agent/tools/spec.py
rag/agent/tools/task_tool.py
rag/agent/tools/tool_sdk.py
rag/agent/tools/workspace_tools.py
rag/agent/core/tool_execution.py
rag/agent/core/approval_policy.py
rag/agent/core/tool_batch_reader.py
rag/agent/core/tool_schema.py
rag/agent/core/llm_tool_runners.py
rag/agent/skills/invocation.py
rag/agent/skills/assets.py
```

Tasks 4-5 port any still-required runner, asset, knowledge, MCP, and subagent
input/output behavior into the final builtin/integration modules before this
deletion. No superseded module remains as a compatibility runtime.

## Task 0: Create the Isolated Implementation Worktree

**Files:** None

- [ ] **Step 1: Verify the original workspace remains dirty and record its state**

Run:

```bash
cd /Users/leixiaoying/LLM/RAG学习
git status --short
git rev-parse HEAD
```

Expected: the existing WIP is visible and `HEAD` is the commit containing this plan.

- [ ] **Step 2: Create a clean implementation worktree**

Run:

```bash
mkdir -p /Users/leixiaoying/LLM/RAG学习-worktrees
git worktree add /Users/leixiaoying/LLM/RAG学习-worktrees/single-tool-runtime -b codex/single-tool-runtime HEAD
cd /Users/leixiaoying/LLM/RAG学习-worktrees/single-tool-runtime
git status --short
```

Expected: the new worktree is on `codex/single-tool-runtime` and has no uncommitted files.

- [ ] **Step 3: Run the committed baseline**

Run:

```bash
uv run pytest tests/agent tests/provider tests/ui -q
uv run python -m compileall -q rag/agent agent_runtime rag/providers
git diff --check
```

Expected: all committed baseline tests pass. If they do not, record the exact pre-existing failures and stop before implementation.

## Task 1: Add the Dormant Canonical Tool Contract

**Files:**
- Create: `rag/agent/tools/tool.py`
- Create: `tests/agent/test_single_tool_contract.py`

- [ ] **Step 1: Write failing contract tests**

Cover:

```python
def test_tool_projects_definition_without_runner() -> None: ...
def test_tool_rejects_local_side_effect_with_non_cancellable_mode() -> None: ...
def test_tool_result_metadata_is_not_model_content() -> None: ...
```

The fixture Tool must include name, description, schema, validator, runner,
normalizer, effects, execution revision, idempotency, concurrency, cancellation,
interrupt behavior, timeout, and output limit.

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
uv run pytest tests/agent/test_single_tool_contract.py -q
```

Expected: collection or assertion failures because the final types do not exist.

- [ ] **Step 3: Implement the immutable data contracts**

Implement these shapes in `tool.py`:

```python
class ToolEffect(StrEnum):
    READ_WORKSPACE = "read_workspace"
    WRITE_WORKSPACE = "write_workspace"
    EXECUTE_PROCESS = "execute_process"
    NETWORK = "network"
    DESTRUCTIVE = "destructive"

class CancellationMode(StrEnum):
    COOPERATIVE = "cooperative"
    MANAGED_PROCESS = "managed_process"
    REMOTE_BEST_EFFORT = "remote_best_effort"
    NOT_CANCELLABLE = "not_cancellable"

class InterruptBehavior(StrEnum):
    CANCEL = "cancel"
    FINISH_CURRENT = "finish_current"

@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, JsonValue]

@dataclass(frozen=True, slots=True)
class Tool:
    definition: ToolDefinition
    validate_input: ValidateInput
    run: ToolRunner
    normalize_output: NormalizeOutput
    output_schema: Mapping[str, JsonValue] | None
    static_effects: frozenset[ToolEffect]
    resolve_use: ResolveToolUse
    execution_revision: str
    idempotent: bool
    concurrency_safe: bool
    cancellation_mode: CancellationMode
    interrupt_behavior: InterruptBehavior
    timeout_seconds: float
    max_model_output_bytes: int
```

Add `ToolCallOrigin`, `ToolCall`, ordered content blocks, artifact references,
structured error fields, and `ToolResult`. Validate contract contradictions in
`Tool.__post_init__`.

- [ ] **Step 4: Keep the contract dormant**

Do not export the new types from `rag.agent.tools.__init__` and do not replace
the currently active `rag.agent.tools.registry` in this task. New tests import
`rag.agent.tools.tool` directly. The only final Registry is implemented at the
atomic Task 10 cutover, after its tests are written against this Tool contract.

- [ ] **Step 5: Run contract tests**

Run:

```bash
uv run pytest tests/agent/test_single_tool_contract.py -q
uv run ruff check rag/agent/tools/tool.py tests/agent/test_single_tool_contract.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rag/agent/tools/tool.py tests/agent/test_single_tool_contract.py
git commit -m "refactor(agent): add canonical tool contract"
```

## Task 2: Implement Complete Input and Output Validation

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `rag/agent/tools/tool.py`
- Create: `tests/agent/test_tool_schema_validation.py`

- [ ] **Step 1: Declare `jsonschema` as a direct dependency**

Run:

```bash
uv add 'jsonschema>=4.26.0'
```

Expected: `pyproject.toml` and `uv.lock` explicitly include `jsonschema`.

- [ ] **Step 2: Write failing validation tests**

Test Pydantic-backed builtins and raw MCP schemas for:

```text
required
additionalProperties=false
minimum/maximum
enum
oneOf/anyOf
local $defs/$ref
no remote $ref retrieval
no silent argument dropping
output-schema failure
```

Include the previous regression: a schema with `minimum=5` and `enum=[7]`
must reject `1` before the runner executes.

- [ ] **Step 3: Verify the tests fail**

Run:

```bash
uv run pytest tests/agent/test_tool_schema_validation.py -q
```

Expected: FAIL because validator factories are incomplete.

- [ ] **Step 4: Implement validator factories**

Expose:

```python
def pydantic_input(model: type[BaseModel]) -> tuple[dict[str, JsonValue], ValidateInput]: ...
def json_schema_input(schema: Mapping[str, JsonValue]) -> ValidateInput: ...
def json_schema_output(schema: Mapping[str, JsonValue] | None, value: JsonValue) -> JsonValue: ...
```

Use the declared schema dialect's validator. Resolve only references inside the
provided schema document. Convert failures into bounded field paths and
messages; never return raw exception reprs to the model.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run pytest tests/agent/test_tool_schema_validation.py tests/agent/test_single_tool_contract.py -q
uv run ruff check rag/agent/tools/tool.py tests/agent/test_tool_schema_validation.py
git diff --check
```

Expected: PASS.

```bash
git add pyproject.toml uv.lock rag/agent/tools/tool.py tests/agent/test_tool_schema_validation.py
git commit -m "feat(agent): validate complete tool schemas"
```

## Task 3: Build the Only Permission and Execution Choke Point

**Files:**
- Create: `rag/agent/tools/permissions.py`
- Create: `rag/agent/tools/executor.py`
- Create: `tests/agent/test_single_tool_executor.py`

- [ ] **Step 1: Write failing executor-order tests**

Use spies to assert this exact order:

```text
unknown
schema_not_exposed
input validation
resolve effects/targets
hard guards
execution boundary
can_use_tool
approval
runner
normalize
output validation
externalize/bound
trace
```

Add tests proving unknown and schema-not-exposed calls never reach permission,
permission never bypasses guards, denial never reaches the runner, and every
exit creates one trace.

Also test the mandatory safety cases:

```text
managed local timeout -> timeout_cancelled and child process gone
remote best-effort timeout -> timeout_outcome_unknown and execution record unknown
non-idempotent unknown outcome -> reconciliation required before retry
parallel batch -> all tools concurrency_safe and resolved targets non-conflicting
conflicting target or one unsafe tool -> deterministic serial execution
```

- [ ] **Step 2: Verify failure**

Run:

```bash
uv run pytest tests/agent/test_single_tool_executor.py -q
```

Expected: FAIL because the final executor does not exist.

- [ ] **Step 3: Implement the pure permission decision**

Implement:

```python
class UseToolDecision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"

@dataclass(frozen=True, slots=True)
class CanUseToolResult:
    decision: UseToolDecision
    reason: str

def can_use_tool(tool: Tool, args: ValidatedArgs, resolved: ResolvedToolUse,
                 context: ToolExecutionContext) -> CanUseToolResult: ...
```

Read effects default to allow. Writes and process execution use the public
pre-authorization flags or ask. Network and destructive actions are
conservative. Human UI remains outside this function.

- [ ] **Step 4: Implement `ToolExecutor`**

The executor accepts a call carrying its originating exposed names and an
immutable `Mapping[str, Tool]` lookup supplied by the future frozen Registry.
It owns
single-call and conflict-aware batch execution. It normalizes all errors into
one ToolResult and records traces through one completion helper.

For `REMOTE_BEST_EFFORT`, cancellation stops local waiting but records the
operation outcome as unknown. A non-idempotent unknown outcome returns an
execution record that requires `tool_reconciliation`; the executor must not
retry it automatically. Batch selection checks both `concurrency_safe` and
resolved effect/target conflicts before starting any parallel runner.

Do not import `ToolExecutionService`, old `ApprovalPolicy`, old Registry, or
`rag.agent.tooling`.

- [ ] **Step 5: Run executor tests**

Run:

```bash
uv run pytest tests/agent/test_single_tool_executor.py tests/agent/test_human_input.py -q
uv run ruff check rag/agent/tools/permissions.py rag/agent/tools/executor.py tests/agent/test_single_tool_executor.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rag/agent/tools/permissions.py rag/agent/tools/executor.py tests/agent/test_single_tool_executor.py
git commit -m "feat(agent): add single tool executor"
```

## Task 4: Port the Resident Coding Tools with Real Cancellation

**Files:**
- Create: `rag/agent/tools/builtins/__init__.py`
- Create: `rag/agent/tools/builtins/filesystem.py`
- Create: `rag/agent/tools/builtins/search.py`
- Create: `rag/agent/tools/builtins/shell.py`
- Create: `rag/agent/tools/builtins/planning.py`
- Create: `tests/agent/test_builtin_coding_tools.py`
- Create: `tests/agent/test_tool_process_cancellation.py`

- [ ] **Step 1: Write failing resident-tool behavior tests**

Cover the exact baseline and non-overlap:

```python
assert resident_names == (
    "list_files", "search_text", "read_file",
    "apply_patch", "run_command", "update_plan",
)
```

Test `search_text` literal/regex, path, glob, context lines, limits, symlinks,
and immediate visibility of file changes. Assert no embedding or retrieval
dependency is imported.

- [ ] **Step 2: Write the cancellation regression**

Run a command that spawns a child and writes a sentinel after the timeout. The
test must wait beyond the sentinel delay and prove:

```python
assert result.error_code == "timeout_cancelled"
assert not sentinel.exists()
assert process_group_is_gone(pid)
```

This replaces the invalid `asyncio.to_thread + wait_for` behavior.

- [ ] **Step 3: Verify failure**

Run:

```bash
uv run pytest tests/agent/test_builtin_coding_tools.py tests/agent/test_tool_process_cancellation.py -q
```

Expected: FAIL because the final builtin factories are absent.

- [ ] **Step 4: Implement builtin Tool factories**

Each module exports explicit factory functions returning `Tool`; no subclass
hierarchy and no builtin registry singleton. Import stable read-only workspace
helpers where their current behavior already satisfies the final contract;
otherwise implement the bounded operation inside the new dormant builtin.
Do not modify `workspace.py` or `primitive_ops.py` before Task 10 because the
legacy public runtime still calls them.

`run_command` uses `asyncio.create_subprocess_exec` or a controlled shell
process group, terminates the group, escalates to kill, and awaits reaping.
`apply_patch` is the model-facing edit primitive; do not register `write_file`
as a resident duplicate.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run pytest tests/agent/test_builtin_coding_tools.py tests/agent/test_tool_process_cancellation.py -q
uv run ruff check rag/agent/tools/builtins tests/agent/test_builtin_coding_tools.py tests/agent/test_tool_process_cancellation.py
```

Expected: PASS.

```bash
git add rag/agent/tools/builtins tests/agent/test_builtin_coding_tools.py tests/agent/test_tool_process_cancellation.py
git commit -m "feat(agent): port resident coding tools"
```

## Task 5: Adapt Knowledge, MCP, Skills, and Subagents into the One Registry

**Files:**
- Create: `rag/agent/tools/integrations/__init__.py`
- Create: `rag/agent/tools/integrations/knowledge.py`
- Create: `rag/agent/tools/integrations/mcp.py`
- Create: `rag/agent/tools/integrations/skills.py`
- Create: `rag/agent/tools/integrations/subagent.py`
- Replace tests in: `tests/agent/test_mcp_adapter.py`
- Modify: `tests/agent/test_rag_answer_tool.py`
- Modify: `tests/agent/test_skills.py`
- Modify: `tests/agent/test_task_tool.py`

- [ ] **Step 1: Write failing source-boundary tests**

Assert:

```text
MCP concrete tools enter the one snapshot
MCP client/session lifecycle does not
raw MCP inputSchema is unchanged
knowledge installs search_knowledge only when configured
vector stores and ingestion do not enter ToolRegistry
skill factories return Tools without owning the catalog or loader
skill ToolResult contains a bounded canonical activation-event payload
subagent factory returns a Tool without owning the child loop
duplicate canonical MCP names fail loudly
```

- [ ] **Step 2: Verify failure**

Run:

```bash
uv run pytest tests/agent/test_mcp_adapter.py tests/agent/test_rag_answer_tool.py tests/agent/test_skills.py tests/agent/test_task_tool.py -q
```

Expected: FAIL against old ToolSpec/registry contracts.

- [ ] **Step 3: Implement integration factories and deterministic assembly**

Each integration factory returns an ordinary final `Tool` value. The runtime
builder will own provider/client lifecycles and pass closures into these Tool
factories at the Task 10 cutover. `build_tool_registry(*tool_sources)` and the
only final Registry are deliberately deferred to Task 10, where deterministic
source ordering and freeze-once semantics are tested together. This task must
not modify or wire the active registry, runtime builder, knowledge provider,
subagent runner, AgentLoop, service, SDK, or CLI.

Move old formatter behavior into each Tool's `normalize_output`; do not carry
UI presentation classes into the registry. Port `invoke_skill` and
`materialize_skill_asset` Tool-facing runners into `integrations/skills.py`,
but keep `SkillCatalog`, `SkillLoader`, and skill lifecycle/state in
`rag/agent/skills/`. Port the subagent Tool adapter into
`integrations/subagent.py`, but keep the child loop and task state outside it.
The skill Tool normalizer emits an activation event value only; Task 7 defines
how that event enters canonical context and Task 10 applies it to LoopState.

- [ ] **Step 4: Run integration tests and commit**

Run:

```bash
uv run pytest tests/agent/test_mcp_adapter.py tests/agent/test_mcp_e2e.py tests/agent/test_rag_answer_tool.py tests/agent/test_rag_tool_specs.py tests/agent/test_skills.py tests/agent/test_task_tool.py tests/agent/test_agent_as_tool_runner.py -q
uv run ruff check rag/agent/tools/integrations
```

Expected: PASS.

```bash
git add rag/agent/tools/integrations tests/agent/test_mcp_adapter.py tests/agent/test_rag_answer_tool.py tests/agent/test_skills.py tests/agent/test_task_tool.py
git commit -m "refactor(agent): adapt extension tools"
```

## Task 6: Implement the Only Selection Path and Atomic Discovery

**Files:**
- Create: `rag/agent/tools/selection.py`
- Create: `tests/agent/test_tool_selection.py`
- Create: `tests/agent/test_find_tools.py`

- [ ] **Step 1: Write the public precedence matrix as failing parametrized tests**

Cover every combination locked in the spec:

```text
default tools / discovery off
default tools / discovery on / no hidden tools
default tools / discovery on / hidden tools
exact tools / discovery off
exact tools / discovery on without find_tools
exact tools / discovery on with find_tools
find_tools explicitly named while discovery off -> configuration error
disabled_tools always wins
unknown explicit name -> configuration error
```

Test the final pure option resolver directly. Public SDK and CLI parity is
wired and tested only in the atomic cutover in Task 10.

- [ ] **Step 2: Write failing selection and search tests**

Assert installed/resident/active are distinct, order is resident then explicit
then activation order, active names only grow, schema budget errors are
explicit, Chinese aliases recall the expected tools, and disabled tools never
appear in search results.

- [ ] **Step 3: Verify failure**

Run:

```bash
uv run pytest tests/agent/test_tool_selection.py tests/agent/test_find_tools.py -q
```

Expected: FAIL because final selection and public mapping are absent.

- [ ] **Step 4: Implement `select_tools` and `find_tools`**

Use one pure selector over the frozen snapshot and name tuples. Implement a
bounded inverted-index/BM25-style scorer over canonical metadata; do not add
embeddings or another LLM.

Until Task 10, tests supply an immutable ordered `Mapping[str, Tool]` directly.
Task 10 replaces the active Registry and passes the mapping returned by its
`freeze()` method; no temporary Registry class or compatibility wrapper is
introduced.

`find_tools` returns matched and proposed activation names. It does not mutate
the registry or checkpoint.

- [ ] **Step 5: Implement the pure activation reducer without wiring it**

Implement a pure helper that validates a proposed activation against the
frozen snapshot and returns the new monotonic ordered name tuple plus trace
metadata. It must not import or mutate AgentLoop, LoopState, checkpointing,
SDK, or CLI. Task 10 applies this helper atomically with ToolResult persistence.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
uv run pytest tests/agent/test_tool_selection.py tests/agent/test_find_tools.py -q
uv run ruff check rag/agent/tools/selection.py tests/agent/test_tool_selection.py tests/agent/test_find_tools.py
```

Expected: PASS.

```bash
git add rag/agent/tools/selection.py tests/agent/test_tool_selection.py tests/agent/test_find_tools.py
git commit -m "feat(agent): add deterministic tool selection"
```

## Task 7: Build the Canonical Model Request and Stable Context Revision

**Files:**
- Create: `rag/agent/core/model_request.py`
- Modify: `rag/agent/core/messages.py`
- Create: `rag/providers/openai_wire.py`
- Create: `rag/providers/local_agent_wire.py`
- Create: `tests/agent/test_canonical_model_request.py`
- Create: `tests/provider/test_openai_wire.py`
- Create: `tests/provider/test_local_agent_wire.py`

- [ ] **Step 1: Write failing deterministic-request tests**

Assert two builds from the same snapshot produce identical canonical JSON,
prompt revision, toolset revision, exposed names, and provider wire hash.
Assert changes to iteration, success/error counters, timestamps, and run IDs do
not change the stable prefix.

Assert active tools preserve activation order and are not globally resorted.

- [ ] **Step 2: Write failing transcript tests**

Assert ToolResult model content is stored once, later formatter changes cannot
change it, skill activation appends an event, and compaction creates a new
context revision rather than pretending the transcript remained append-only.

- [ ] **Step 3: Verify failure**

Run:

```bash
uv run pytest tests/agent/test_canonical_model_request.py tests/provider/test_openai_wire.py tests/provider/test_local_agent_wire.py -q
```

Expected: FAIL against the OpenAI-shaped tooling request and dynamic system
prompt.

- [ ] **Step 4: Implement canonical request and stable hashing**

Implement:

```python
@dataclass(frozen=True, slots=True)
class ModelRequest:
    request_id: str
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolDefinition, ...]
    exposed_tool_names: tuple[str, ...]
    tool_choice: ToolChoice
    settings: ModelSettings
    prompt_revision: str
    toolset_revision: str
```

Canonical serialization must normalize mappings but preserve semantically
ordered arrays such as tool and message order.

- [ ] **Step 5: Move OpenAI wire logic to the provider boundary**

`rag/providers/openai_wire.py` converts canonical messages/tools to wire dicts,
applies supported cache parameters, parses model text/tool calls, and computes
the final serialized hash. It never selects tools.

`rag/providers/local_agent_wire.py` renders the same canonical request into a
deterministic flat prompt for MLX and Ollama when their configured generator
does not expose native tool calling. The prompt includes the selected schemas
and requires one validated response envelope:

```json
{"text":"", "tool_calls":[{"id":"call_1", "name":"read_file", "arguments":{"path":"README.md"}}]}
```

The parser returns the same provider-neutral model turn as OpenAI parsing. Add
separate MLX and Ollama capability tests proving both resolve this adapter and
that selected tools are not omitted. This is serialization, not a second
selector, executor, or loop.

- [ ] **Step 6: Implement the dormant stable context builder**

Build the final stable instructions and frozen run-context blocks without
visible-tool name lists, iteration, tool success/error counts, timestamps, or
run IDs. Freeze initial memory for the revision. Represent skill activation
and later memory as appended canonical events. Do not replace the active
`AgentMessageAssembler` or `LLMLoopModelTurnProvider` until Task 10.
Any additions to `messages.py` must be additive and must leave existing
message construction behavior unchanged until that cutover.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
uv run pytest tests/agent/test_canonical_model_request.py tests/provider/test_openai_wire.py tests/provider/test_local_agent_wire.py -q
uv run ruff check rag/agent/core/model_request.py rag/agent/core/messages.py rag/providers/openai_wire.py rag/providers/local_agent_wire.py
```

Expected: PASS.

```bash
git add rag/agent/core/model_request.py rag/agent/core/messages.py rag/providers/openai_wire.py rag/providers/local_agent_wire.py tests/agent/test_canonical_model_request.py tests/provider/test_openai_wire.py tests/provider/test_local_agent_wire.py
git commit -m "refactor(agent): add canonical model request"
```

## Task 8: Define Cache Usage and Model-Call Diagnostic Records

**Files:**
- Modify: `rag/schema/llm.py`
- Modify: `rag/agent/core/model_request.py`
- Modify: `rag/agent/core/llm_config.py`
- Modify: `rag/providers/openai_wire.py`
- Modify: `rag/providers/local_agent_wire.py`
- Create: `tests/agent/test_model_usage_contract.py`
- Modify: `tests/provider/test_openai_wire.py`
- Modify: `tests/provider/test_local_agent_wire.py`

- [ ] **Step 1: Write failing normalized-usage tests**

Cover OpenAI-compatible usage where cached input is included in total input,
provider responses with cache-write details, absent cache fields, tokenizer
estimates, and a simulated future provider whose raw accounting separates
cache reads and writes.

Unknown values must be `None`, never fabricated zeroes.

- [ ] **Step 2: Write failing model-call diagnostic tests**

Define one checkpoint-safe record and assert it always binds usage to the exact
request evidence:

```text
request_id
prompt_revision
toolset_revision
provider_wire_hash
normalized usage
bounded raw provider usage
```

The public propagation path remains untouched until Task 10. A streaming wire
fixture may parse final provider usage here, but it is not wired into the
active gateway.

- [ ] **Step 3: Verify failure**

Run:

```bash
uv run pytest tests/agent/test_model_usage_contract.py tests/provider/test_openai_wire.py tests/provider/test_local_agent_wire.py -q
```

Expected: FAIL because normalized cache fields and bound call evidence are
absent.

- [ ] **Step 4: Implement normalized usage and bound call records without wiring them**

Add optional fields with defaults so public result construction remains
backward-compatible:

```text
logical_input_tokens
uncached_input_tokens
cache_read_input_tokens
cache_write_input_tokens
output_tokens
usage_source
raw_provider_usage
```

Bound and JSON-normalize raw usage. Add `ModelCallRecord` containing the three
revision/hash fields plus usage. Add optional cache-read/cache-write price
fields to `ModelSpec`; do not invent costs when absent. Task 10 carries this
record through gateway, envelope, loop, checkpoint, and AgentResult in one
atomic cutover.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run pytest tests/agent/test_model_usage_contract.py tests/provider/test_openai_wire.py tests/provider/test_local_agent_wire.py -q
uv run ruff check rag/schema/llm.py rag/agent/core/model_request.py rag/agent/core/llm_config.py rag/providers/openai_wire.py rag/providers/local_agent_wire.py tests/agent/test_model_usage_contract.py
```

Expected: PASS.

```bash
git add rag/schema/llm.py rag/agent/core/model_request.py rag/agent/core/llm_config.py rag/providers/openai_wire.py rag/providers/local_agent_wire.py tests/agent/test_model_usage_contract.py tests/provider/test_openai_wire.py tests/provider/test_local_agent_wire.py
git commit -m "feat(agent): define model cache diagnostics"
```

## Task 9: Implement Tool Checkpoint Codecs and Drift Decisions

**Files:**
- Modify: `rag/agent/core/turn_contracts.py`
- Modify: `rag/agent/core/model_request.py`
- Modify: `rag/agent/core/checkpointing.py`
- Create: `tests/agent/fixtures/checkpoints/legacy_tool_state_v1.json`
- Create: `tests/agent/test_tool_manifest_resume.py`

- [ ] **Step 1: Add a committed legacy fixture before changing migration**

Create a minimal JSON-safe fixture representing the oldest committed tool
result/ledger/discovery fields still promised readable. Test a new pure legacy
decode helper directly; do not attach it to the active checkpoint load path.

- [ ] **Step 2: Write failing originating-request tests**

Construct codec inputs for a call under request A and a later request B, then
assert the decoded call retains A's:

```text
request_id
toolset_revision
exposed_tool_names
```

The test must fail if a mutable current-turn sent-name list is used.

- [ ] **Step 3: Write failing drift-policy tests**

Cover:

```text
matching manifest -> retain revision
missing/changed tool with pending call -> pause tool_definition_changed
missing active tool with paused call -> retain evidence for reconciliation
drift without dependent call -> remove unavailable active name and create revision
serializer revision change -> no old wire-hash guarantee
```

- [ ] **Step 4: Verify failure**

Run:

```bash
uv run pytest tests/agent/test_tool_manifest_resume.py -q
```

Expected: FAIL because canonical transcript/origin/manifest codecs are absent.

- [ ] **Step 5: Implement versioned checkpoint codecs without wiring them**

Implement pure encode/decode helpers for canonical transcript content, manifest
entries, resident/active order, prompt/toolset/serializer revisions,
ModelCallRecord usage evidence, and ToolCall origin. The legacy helper rebuilds
canonical transcript exactly once and returns the new checkpoint value.

Do not retain legacy registry, formatter, visibility, or executor objects in
the encoded value. Do not change `LangGraphCheckpointStore.save_snapshot`,
`load_latest`, LoopState fields, or the active resume path until Task 10.

- [ ] **Step 6: Implement drift reconciliation**

Implement a pure comparison function returning `match`,
`reconciliation_required`, or `new_revision_required`. It compares the rebuilt
snapshot projection with the persisted execution-contract hash and retains
origin evidence for dependent pending/paused calls. Task 10 translates this
decision into LoopState and HumanInput transitions.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
uv run pytest tests/agent/test_tool_manifest_resume.py -q
uv run ruff check rag/agent/core/turn_contracts.py rag/agent/core/checkpointing.py tests/agent/test_tool_manifest_resume.py
```

Expected: PASS.

```bash
git add rag/agent/core/turn_contracts.py rag/agent/core/model_request.py rag/agent/core/checkpointing.py tests/agent/fixtures/checkpoints/legacy_tool_state_v1.json tests/agent/test_tool_manifest_resume.py
git commit -m "feat(agent): add tool checkpoint codecs"
```

## Task 10: Atomically Cut Every Public Path Over to the Final Runtime

**Files:**
- Replace: `rag/agent/tools/registry.py`
- Modify: `rag/agent/tools/__init__.py`
- Modify: `rag/agent/workspace.py`
- Modify: `rag/agent/loop/runtime.py`
- Modify: `rag/agent/loop/state.py`
- Modify: `rag/agent/loop/substate.py`
- Modify: `rag/agent/builtin/generic.py`
- Modify: `rag/agent/core/definition.py`
- Modify: `rag/agent/core/compiler.py`
- Modify: `rag/agent/core/agent_as_tool.py`
- Modify: `rag/agent/core/agent_tool_contract.py`
- Modify: `rag/agent/core/delegation.py`
- Modify: `rag/agent/core/llm_context.py`
- Modify: `rag/agent/core/observations.py`
- Modify: `rag/agent/core/model_provider_runtime.py`
- Modify: `rag/agent/core/llm_registry.py`
- Modify: `rag/agent/core/llm_providers.py`
- Modify: `rag/agent/core/checkpointing.py`
- Modify: `rag/agent/core/human_input.py`
- Modify: `rag/agent/core/runtime_diagnostics.py`
- Modify: `rag/providers/llm_gateway.py`
- Modify: `rag/assembly/support.py`
- Modify: `rag/agent/service.py`
- Modify: `agent_runtime/runtime/builder.py`
- Modify: `agent_runtime/knowledge_providers/rag.py`
- Modify: `rag/agent/core/subagent_runner.py`
- Modify: `rag/agent/loop/stop_hooks.py`
- Modify: `rag/agent/memory/compactor.py`
- Modify: `rag/agent/memory/injector.py`
- Modify: `rag/agent/primitive_ops.py`
- Modify: `rag/agent/skills/runtime.py`
- Modify: `agent_runtime/agent.py`
- Modify: `agent_runtime/result.py`
- Modify: `rag/agent/cli.py`
- Modify: `tests/agent/test_agent_loop_runtime.py`
- Modify: `tests/agent/test_agent_loop_parity.py`
- Modify: `tests/agent/parity/fixtures.py`
- Modify: `tests/agent/parity/loop_scenarios.py`
- Modify: `tests/agent/parity/normalize.py`
- Replace tests in: `tests/agent/test_tool_registry.py`
- Modify: `tests/agent/test_agent_service_loop_boundary.py`
- Modify: `tests/agent/test_agent_service.py`
- Modify: `tests/agent/test_agent_graph_compiler.py`
- Modify: `tests/agent/test_builtin_agents.py`
- Modify: `tests/agent/test_contract_config.py`
- Modify: `tests/agent/test_agent_as_tool_runner.py`
- Modify: `tests/agent/test_agent_observations.py`
- Modify: `tests/agent/test_llm_context.py`
- Modify: `tests/agent/test_context_injector.py`
- Modify: `tests/agent/test_cli_wiring.py`
- Modify: `tests/agent/test_stop_hooks.py`
- Modify: `tests/agent/test_working_memory_compactor.py`
- Modify: `tests/agent/test_model_provider_runtime.py`
- Modify: `tests/agent/test_llm_providers.py`
- Modify: `tests/agent/test_loop_model_context.py`
- Modify: `tests/provider/test_llm_gateway.py`
- Modify: `tests/agent/test_mcp_e2e.py`
- Modify: `tests/agent/test_primitive_ops.py`
- Create: `tests/agent/test_model_usage_propagation.py`
- Modify: `tests/agent/test_checkpointing.py`
- Modify: `tests/agent/test_loop_checkpointing.py`
- Modify: `tests/agent/test_agent_service_resume.py`
- Modify: `tests/agent/test_tool_manifest_resume.py`
- Modify: `tests/agent/test_agent_runtime_imports.py`
- Modify: `tests/agent/test_agent_runtime_facade.py`
- Modify: `tests/agent/test_agent_cli_resume.py`
- Modify: `tests/ui/test_cli.py`

This is deliberately one larger commit. Tasks 1-9 leave the old path active;
this task switches loop, provider, persistence, service, SDK, and CLI together
so no commit exposes a mixed public runtime.

- [ ] **Step 1: Write failing single-path and public-option tests**

Construct the real `AgentService` through both `agent_runtime.Agent` and CLI
assembly. Assert identity across every initial, streaming, paused, and resumed
path:

```text
one frozen Registry snapshot
one select_tools implementation
one ToolExecutor instance
one canonical ModelRequest builder
one checkpoint/resume codec path
```

Patch old registry/executor/surface entry points to raise and prove the public
path never calls them. Preserve public method signatures and CLI flags. For
sync, async, stream, and resume, assert the complete Task 6 precedence matrix,
including the default six coding tools and the `find_tools` configuration
error when discovery is false.

Before service construction, test the final Registry directly: duplicate
names fail, freeze preserves insertion order, the frozen mapping is immutable,
later registration is rejected, and its manifest changes when any execution
contract field changes.

- [ ] **Step 2: Write failing provider-parity tests**

For OpenAI-compatible, MLX, and Ollama configurations, capture the actual
request passed to the configured generator and assert it comes from the same
canonical `ModelRequest`:

```text
same ordered selected tool definitions
same request_id, prompt_revision, and toolset_revision
OpenAI -> native canonical wire adapter
MLX/Ollama -> deterministic local-agent prompt and JSON response envelope
no provider performs selection, permission, or tool execution
```

Cover non-streaming and streaming finalization. The local adapter test must
prove tool calls return to the same provider-neutral model-turn type used by
the OpenAI adapter.

- [ ] **Step 3: Write failing usage-propagation tests**

Inject real-looking stream and non-stream provider usage and assert one
`ModelCallRecord` reaches diagnostics and checkpoint state with:

```text
request_id
prompt_revision
toolset_revision
provider_wire_hash
normalized cache read/write usage
bounded raw provider usage
```

Assert the same normalized usage is exposed by `AgentResult`; missing cache
fields remain `None`, and tokenizer estimates are labelled rather than mixed
with provider-reported usage.

- [ ] **Step 4: Write failing checkpoint, drift, and origin tests**

Exercise the real save/load/resume path. Assert canonical transcript content,
tool manifest, resident/active order, serializer revision, revisions, model
call records, and each `ToolCallOrigin` survive exactly. A call created under
request A must still use A's exposed names and toolset revision after request B
and resume.

Assert activation applies the `find_tools` result and monotonic active-name
tuple in one LoopState transition before checkpointing. Assert manifest drift
becomes either a new revision or `tool_definition_changed` reconciliation; a
remote non-idempotent timeout with unknown outcome cannot be replayed without
that reconciliation.

- [ ] **Step 5: Verify all cutover tests fail against the old path**

Run:

```bash
uv run pytest tests/agent/test_agent_service_loop_boundary.py tests/agent/test_model_provider_runtime.py tests/agent/test_llm_providers.py tests/agent/test_model_usage_propagation.py tests/agent/test_tool_manifest_resume.py tests/agent/test_agent_runtime_facade.py tests/agent/test_agent_cli_resume.py tests/provider/test_llm_gateway.py tests/ui/test_cli.py -q
```

Expected: FAIL because the active loop still uses the legacy surface,
OpenAI-shaped request construction, incomplete usage propagation, and old
checkpoint state.

- [ ] **Step 6: Wire the final runtime in one working-tree change**

Make the following changes together before committing:

1. `AgentLoop` calls the final executor batch API directly; it supplies the
   originating call record, atomically applies activation plus ToolResult, and
   checkpoints the resulting state. Remove `LoopToolRunner`,
   `ToolExecutionService`, and mutable-current-turn visibility from the active
   path.
2. `LLMLoopModelTurnProvider` builds only canonical `ModelRequest`. The gateway
   chooses `openai_wire` or `local_agent_wire` from provider capability; it
   does not build a second loop prompt or choose tools.
3. The stream and non-stream envelopes carry the final real provider usage.
   Create the bound `ModelCallRecord` only after the provider wire hash is
   known; append it to diagnostics/checkpoint state and project it into
   `AgentResult` without changing provider accounting.
4. LoopState stores canonical transcript content, manifest, resident/active
   ordered names, prompt/toolset/serializer revisions, call origins, and model
   call records. Compaction creates a prompt revision; it never rewrites the
   prior revision in place.
5. Checkpoint load uses the Task 9 codec, then manifest drift reconciliation,
   before permitting pending execution. Human-input resume uses the same
   originating-call evidence and executor.
6. Replace the active `tools/registry.py` with the only final
   `ToolRegistry`; export it and the Task 1 contracts from `tools/__init__.py`.
   `build_tool_registry(...)` assembles ordinary `Tool` outputs from builtin,
   configured knowledge, MCP, skill, and subagent integration factories once,
   freezes once, and passes closures for
   provider/client/child-loop lifecycles without owning those lifecycles in the
   Registry. The generic definition and stable prompt stop naming
   `tool_search`, `activate_tools`, `write_file`, `run_python`, or `tool_repl`;
   compiler and subagent assembly consume the same final snapshot/options.
7. `Agent`, CLI, service, and resume map the stable public parameters through
   the same pure option resolver. They do not import or construct a legacy
   surface/policy type.
8. Observation, memory injection/compaction, stop hooks, delegation, and
   primitive helpers consume the final `ToolResult` and its already-normalized
   canonical content. None can invoke an output formatter or re-render a prior
   transcript entry.
9. Rewrite loop parity fixtures to construct final `Tool` values and compare
   user-visible/invariant outcomes rather than old ToolResult object layouts.
   Preserve the committed legacy checkpoint JSON separately as Task 9 decoder
   evidence; do not make final runtime tests import deleted contracts.

Old modules may still exist until Task 12, but after this step they are
unreachable from every public run path.

- [ ] **Step 7: Run focused cutover verification**

Run:

```bash
uv run pytest tests/agent/test_tool_registry.py tests/agent/test_agent_loop_runtime.py tests/agent/test_agent_loop_parity.py tests/agent/test_agent_service_loop_boundary.py tests/agent/test_agent_service.py tests/agent/test_agent_graph_compiler.py tests/agent/test_builtin_agents.py tests/agent/test_contract_config.py tests/agent/test_agent_as_tool_runner.py tests/agent/test_agent_observations.py tests/agent/test_llm_context.py tests/agent/test_context_injector.py tests/agent/test_cli_wiring.py tests/agent/test_stop_hooks.py tests/agent/test_working_memory_compactor.py tests/agent/test_model_provider_runtime.py tests/agent/test_llm_providers.py tests/agent/test_loop_model_context.py tests/agent/test_model_usage_propagation.py tests/agent/test_checkpointing.py tests/agent/test_loop_checkpointing.py tests/agent/test_agent_service_resume.py tests/agent/test_tool_manifest_resume.py tests/agent/test_mcp_e2e.py tests/agent/test_primitive_ops.py tests/agent/test_agent_runtime_imports.py tests/agent/test_agent_runtime_facade.py tests/agent/test_agent_cli_resume.py tests/provider/test_llm_gateway.py tests/ui/test_cli.py -q
uv run ruff check rag/agent/tools/registry.py rag/agent/tools/__init__.py rag/agent/workspace.py rag/agent/loop rag/agent/builtin/generic.py rag/agent/core/definition.py rag/agent/core/compiler.py rag/agent/core/agent_as_tool.py rag/agent/core/agent_tool_contract.py rag/agent/core/delegation.py rag/agent/core/llm_context.py rag/agent/core/observations.py rag/agent/core/model_provider_runtime.py rag/agent/core/llm_registry.py rag/agent/core/llm_providers.py rag/agent/core/checkpointing.py rag/agent/core/human_input.py rag/agent/core/runtime_diagnostics.py rag/agent/core/subagent_runner.py rag/agent/memory/compactor.py rag/agent/memory/injector.py rag/agent/primitive_ops.py rag/agent/skills/runtime.py rag/providers/llm_gateway.py rag/assembly/support.py rag/agent/service.py agent_runtime
```

Expected: PASS with no public call into the old runtime and provider parity for
OpenAI-compatible, MLX, and Ollama.

- [ ] **Step 8: Commit the atomic cutover**

Review the whole staged set because this is the one intentional large commit:

```bash
git add rag/agent/tools/registry.py rag/agent/tools/__init__.py rag/agent/workspace.py rag/agent/loop/runtime.py rag/agent/loop/state.py rag/agent/loop/substate.py rag/agent/loop/stop_hooks.py rag/agent/builtin/generic.py rag/agent/core/definition.py rag/agent/core/compiler.py rag/agent/core/agent_as_tool.py rag/agent/core/agent_tool_contract.py rag/agent/core/delegation.py rag/agent/core/llm_context.py rag/agent/core/observations.py rag/agent/core/model_provider_runtime.py rag/agent/core/llm_registry.py rag/agent/core/llm_providers.py rag/agent/core/checkpointing.py rag/agent/core/human_input.py rag/agent/core/runtime_diagnostics.py rag/providers/llm_gateway.py rag/assembly/support.py rag/agent/service.py agent_runtime/runtime/builder.py agent_runtime/knowledge_providers/rag.py rag/agent/core/subagent_runner.py rag/agent/memory/compactor.py rag/agent/memory/injector.py rag/agent/primitive_ops.py rag/agent/skills/runtime.py agent_runtime/agent.py agent_runtime/result.py rag/agent/cli.py tests/agent/test_tool_registry.py tests/agent/test_agent_loop_runtime.py tests/agent/test_agent_loop_parity.py tests/agent/parity/fixtures.py tests/agent/parity/loop_scenarios.py tests/agent/parity/normalize.py tests/agent/test_agent_service_loop_boundary.py tests/agent/test_agent_service.py tests/agent/test_agent_graph_compiler.py tests/agent/test_builtin_agents.py tests/agent/test_contract_config.py tests/agent/test_agent_as_tool_runner.py tests/agent/test_agent_observations.py tests/agent/test_llm_context.py tests/agent/test_context_injector.py tests/agent/test_cli_wiring.py tests/agent/test_stop_hooks.py tests/agent/test_working_memory_compactor.py tests/agent/test_model_provider_runtime.py tests/agent/test_llm_providers.py tests/agent/test_loop_model_context.py tests/agent/test_model_usage_propagation.py tests/agent/test_checkpointing.py tests/agent/test_loop_checkpointing.py tests/agent/test_agent_service_resume.py tests/agent/test_tool_manifest_resume.py tests/agent/test_mcp_e2e.py tests/agent/test_primitive_ops.py tests/agent/test_agent_runtime_imports.py tests/agent/test_agent_runtime_facade.py tests/agent/test_agent_cli_resume.py tests/provider/test_llm_gateway.py tests/ui/test_cli.py
git diff --cached --check
git commit -m "refactor(agent): atomically cut over single tool runtime"
```

## Task 11: Prove the Stable CLI and SDK Behavior End to End

**Files:**
- Modify: `scripts/agent_delivery_smoke.py`
- Modify: `tests/agent/test_delivery_smoke_script.py`

- [ ] **Step 1: Add failing fake-model delivery cases**

Required cases:

```text
direct answer -> no unnecessary call, resident schemas stable
find AgentService -> search_text/read_file
patch a fixture -> apply_patch
echo hello -> run_command
missing file -> recoverable ToolResult
hidden knowledge/MCP -> find_tools only when discovery enabled
resume pending approval -> originating toolset revision retained
cache usage fake -> revisions, wire hash, and source visible in diagnostics
MLX/Ollama local envelope -> same final ToolResult path
```

- [ ] **Step 2: Verify failure, implement, and run public smoke**

Run:

```bash
uv run pytest tests/agent/test_delivery_smoke_script.py -q
uv run python scripts/agent_delivery_smoke.py --fake-model --verbose
```

Expected after implementation: PASS; smoke reports tools, revisions, schema
bytes, tool errors, provider wire kind, and real-or-explicitly-estimated cache
usage source.

- [ ] **Step 3: Re-run the public contract tests**

Run:

```bash
uv run pytest tests/agent/test_agent_runtime_imports.py tests/agent/test_agent_runtime_facade.py tests/agent/test_agent_cli_resume.py tests/agent/test_delivery_smoke_script.py tests/ui/test_cli.py -q
```

Expected: PASS without further production wiring changes.

- [ ] **Step 4: Commit the smoke proof**

```bash
git add scripts/agent_delivery_smoke.py tests/agent/test_delivery_smoke_script.py
git commit -m "test(agent): prove public tool runtime delivery"
```

## Task 12: Delete Every Legacy Tool Runtime Path

**Files:**
- Delete: `rag/agent/tooling/`
- Delete: `rag/agent/capabilities/catalog.py`
- Delete: `rag/agent/capabilities/context.py`
- Delete: `rag/agent/capabilities/tool_search.py`
- Modify: `rag/agent/capabilities/__init__.py`
- Delete: `rag/agent/builtin_registry.py`
- Delete: `rag/agent/tools/asset_tools.py`
- Delete: `rag/agent/tools/base.py`
- Delete: `rag/agent/tools/builtin_registry.py`
- Delete: `rag/agent/tools/card.py`
- Delete: `rag/agent/tools/catalog_assembly.py`
- Delete: `rag/agent/tools/formatter.py`
- Delete: `rag/agent/tools/formatters/`
- Delete: `rag/agent/tools/generic_tools.py`
- Delete: `rag/agent/tools/llm_tools.py`
- Delete: `rag/agent/tools/mcp_adapter.py`
- Delete: `rag/agent/tools/observation.py`
- Delete: `rag/agent/tools/rag_answer_tools.py`
- Delete: `rag/agent/tools/rag_semantic_tools.py`
- Delete: `rag/agent/tools/rag_tool_runner.py`
- Delete: `rag/agent/tools/rag_tools.py`
- Delete: `rag/agent/tools/runtime_registry_builder.py`
- Delete: `rag/agent/tools/spec.py`
- Delete: `rag/agent/tools/task_tool.py`
- Delete: `rag/agent/tools/tool_sdk.py`
- Delete: `rag/agent/tools/workspace_tools.py`
- Delete: `rag/agent/core/tool_execution.py`
- Delete: `rag/agent/core/approval_policy.py`
- Delete: `rag/agent/core/tool_batch_reader.py`
- Delete: `rag/agent/core/tool_schema.py`
- Delete: `rag/agent/core/llm_tool_runners.py`
- Delete: `rag/agent/skills/invocation.py`
- Delete: `rag/agent/skills/assets.py`
- Modify: `rag/agent/tools/__init__.py`
- Modify: `rag/agent/core/__init__.py`
- Modify: `rag/agent/__init__.py`
- Modify: `rag/__init__.py`
- Modify: `rag/agent/skills/__init__.py`
- Modify: `rag/agent/primitive_ops.py`
- Create: `tests/agent/test_single_tool_runtime_imports.py`
- Delete: `tests/agent/test_activation_groups.py`
- Delete: `tests/agent/test_approval_policy.py`
- Delete: `tests/agent/test_asset_tools.py`
- Delete: `tests/agent/test_builtin_tool_registry.py`
- Delete: `tests/agent/test_code_as_tool_integration.py`
- Delete: `tests/agent/test_contract_tool.py`
- Delete: `tests/agent/test_formatter_maturity.py`
- Delete: `tests/agent/test_formatter_snapshots.py`
- Delete: `tests/agent/test_generic_tools.py`
- Delete: `tests/agent/test_legacy_adapter_removal.py`
- Delete: `tests/agent/test_llm_tool_runners.py`
- Delete: `tests/agent/test_llm_tool_specs.py`
- Delete: `tests/agent/test_rag_tool_runner.py`
- Delete: `tests/agent/test_pr2_context_equivalence.py`
- Delete: `tests/agent/test_runtime_tool_registry_builder.py`
- Delete: `tests/agent/test_tool_batch.py`
- Delete: `tests/agent/test_tool_card.py`
- Delete: `tests/agent/test_tool_card_search.py`
- Delete: `tests/agent/test_tool_execution_service.py`
- Delete: `tests/agent/test_tooling_main_path.py`
- Delete: `tests/agent/test_tooling_workspace_tools.py`
- Modify: `tests/agent/test_skills.py`
- Modify: `tests/agent/test_aci_conventions.py`
- Modify: `tests/agent/test_public_exports.py`
- Delete: `tests/test_tool_discovery.py`

- [ ] **Step 1: Write the deletion/import guard first**

The test scans production Python files and fails on executable imports,
definitions, or references for:

```text
rag.agent.tooling
ToolSpec
BaseTool
ToolCard
MCPToolRegistry
ToolExecutionService
ToolExecutorLoopAdapter
ToolSurfaceRequest
ToolSurfaceDecision
ToolSurfacePolicy
DiscoveryPolicy
ModelRequestBuilder
ToolCatalog
DeferredToolStore
RuntimeToolRegistryBuilder
resolve_visible_tools
tool_search
activate_tools
tool_repl
ToolOutputFormatter
ToolOutputFormatterResolver
register_formatter
get_formatter
format_tool_result_fallback
```

Allow a legacy checkpoint decoder to recognize old serialized module/type
names as explicitly enumerated string literals. Do not allow executable
imports, callables, local state, wildcard exemptions, or whole-file ignores.

Also assert exactly one production definition exists for each final concept:
`Tool`, `ToolRegistry`, `select_tools`, `can_use_tool`, `ToolExecutor`, and
`ToolResult`.

- [ ] **Step 2: Verify the guard fails**

Run:

```bash
uv run pytest tests/agent/test_single_tool_runtime_imports.py -q
```

Expected: FAIL listing all remaining legacy references.

- [ ] **Step 3: Delete the enumerated files and port the last imports**

Delete every file and test listed above. Tasks 4-5 already moved required
asset/knowledge inputs and output normalization into
`tools/integrations/knowledge.py`, MCP adaptation into
`tools/integrations/mcp.py`, subagent adaptation into
`tools/integrations/subagent.py`, and coding behavior into `tools/builtins/`.
Consequently no old tool module is reduced to a half-compatible shell.

Port remaining production imports directly to the final modules. Replace old
formatter assertions with ToolResult content-block and normalization tests
from Tasks 1-5. Replace old visibility/discovery assertions with Task 6 tests,
old execution assertions with Task 3 tests, and code-as-tool behavior with the
explicit `run_command`/subagent contracts.

If an old import is proven part of the documented public surface, provide only
a direct re-export from the final module and add a test that it has no local
state or callable wrapper logic.

- [ ] **Step 4: Run import and package tests**

Run:

```bash
uv run pytest tests/agent/test_single_tool_runtime_imports.py tests/agent/test_public_exports.py tests/agent/test_agent_runtime_imports.py -q
uv run python -m compileall -q rag/agent agent_runtime rag/providers
rg -n 'rag\.agent\.tooling|ToolSpec|BaseTool|ToolCard|MCPToolRegistry|ToolExecutionService|ToolExecutorLoopAdapter|ToolSurface(Request|Decision|Policy)|DiscoveryPolicy|ModelRequestBuilder|ToolCatalog|DeferredToolStore|RuntimeToolRegistryBuilder|resolve_visible_tools|tool_search|activate_tools|tool_repl|ToolOutputFormatter|ToolOutputFormatterResolver|register_formatter|get_formatter|format_tool_result_fallback' rag agent_runtime scripts
```

Expected: tests and compile pass; `rg` returns only approved checkpoint string
literals or no matches.

- [ ] **Step 5: Run the full agent suite before committing deletion**

Run:

```bash
uv run pytest -q
git diff --check
```

Expected: PASS.

- [ ] **Step 6: Commit**

Use `git status --short` to enumerate the exact migrated/deleted files, then
stage only those paths and commit:

```bash
git add -u rag/agent agent_runtime scripts tests
git add rag/agent/tools rag/agent/core/model_request.py rag/providers/openai_wire.py rag/providers/local_agent_wire.py tests/agent/test_single_tool_runtime_imports.py
git commit -m "refactor(agent): remove legacy tool runtimes"
```

Before committing, verify `git diff --cached --name-only` contains no files
from the original dirty workspace because execution is in the isolated
worktree.

## Task 13: Add ACI Evaluation and Complete Verification

**Files:**
- Create: `scripts/agent_tool_aci_eval.py`
- Create: `tests/agent/fixtures/tool_aci_cases.json`
- Create: `tests/agent/test_tool_aci_eval.py`
- Modify: `scripts/agent_delivery_smoke.py`
- Modify: `README.md`

- [ ] **Step 1: Write a failing deterministic eval-harness test**

The fixture set covers direct answer, navigation, Grep, read, patch, command,
knowledge, hidden MCP, subagent, hidden hallucination, similar-tool confusion,
and Chinese discovery. The harness reports:

```text
surface recall
surface precision
tool choice accuracy
argument validity
unnecessary call rate
discovery recall@5
recovery rate
schema bytes/tokens
cache read/write tokens and usage source
```

Offline fake-model cases must be deterministic. Live-provider cases are opt-in
and never required for unit CI.

- [ ] **Step 2: Verify failure, implement, and run the harness**

Run:

```bash
uv run pytest tests/agent/test_tool_aci_eval.py -q
uv run python scripts/agent_tool_aci_eval.py --fake-model --json
```

Expected after implementation: PASS and valid JSON metrics with all required
keys. Do not invent quality thresholds until a real-model baseline is recorded.

- [ ] **Step 3: Document the final public behavior**

Document the six resident tools, Grep-not-RAG workspace rule, explicit
knowledge behavior, discovery option precedence, approval flags, cache metric
meaning, and resume drift behavior. Do not document deleted internal types.

- [ ] **Step 4: Run focused static verification**

Run:

```bash
uv run ruff check rag/agent/tools rag/agent/core/model_request.py rag/agent/core/messages.py rag/agent/core/llm_providers.py rag/agent/loop rag/agent/service.py rag/providers/openai_wire.py rag/providers/local_agent_wire.py agent_runtime scripts/agent_delivery_smoke.py scripts/agent_tool_aci_eval.py
uv run mypy rag/agent/tools rag/agent/core/model_request.py rag/agent/core/messages.py rag/providers/openai_wire.py rag/providers/local_agent_wire.py agent_runtime
uv run python -m compileall -q rag/agent agent_runtime rag/providers scripts
git diff --check
```

Expected: PASS with no new scoped lint/type/compile errors.

- [ ] **Step 5: Run full verification**

Run:

```bash
uv run pytest -q
uv run python scripts/agent_delivery_smoke.py --fake-model --verbose
uv run python scripts/agent_tool_aci_eval.py --fake-model --json
git status --short
```

Expected: all tests and both smoke commands pass. Only intentional files are
modified; no legacy runtime file exists.

- [ ] **Step 6: Commit final evaluation and docs**

```bash
git add scripts/agent_tool_aci_eval.py scripts/agent_delivery_smoke.py tests/agent/fixtures/tool_aci_cases.json tests/agent/test_tool_aci_eval.py README.md
git commit -m "test(agent): add tool ACI evaluation"
```

## Final Review Checklist

- [ ] `agent` CLI commands and flags remain present.
- [ ] `from agent_runtime import Agent` and sync/async/stream/resume signatures remain present.
- [ ] The default Agent receives the six resident coding tools.
- [ ] `search_text` is Grep and imports no retrieval or embedding runtime.
- [ ] One Registry implementation exists and each runtime may own its own instance.
- [ ] One selection function determines model-visible schemas.
- [ ] One executor handles initial, streaming, approval, resume, MCP, knowledge, and subagent calls.
- [ ] Every ToolCall carries its originating exposed names and toolset revision.
- [ ] Local timeout tests prove the process group is gone after return.
- [ ] Canonical transcript content is checkpointed and not re-rendered.
- [ ] Compaction and client activation create explicit revisions.
- [ ] Real provider cache usage reaches AgentResult when reported.
- [ ] Missing provider cache usage remains unknown, not fabricated zero.
- [ ] OpenAI-compatible, MLX, and Ollama serialize one canonical request and return one provider-neutral model turn.
- [ ] Legacy checkpoints load through decoder migration only.
- [ ] No runtime import reaches `rag.agent.tooling`, old catalog/store, old executor, or old visibility code.
- [ ] Full tests, smoke, ACI eval, Ruff, scoped mypy, compileall, and diff checks pass.
