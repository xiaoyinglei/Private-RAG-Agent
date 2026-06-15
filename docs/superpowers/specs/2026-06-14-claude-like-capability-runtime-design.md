# Claude-Like Capability Runtime Design

## Context

The Claude-like single-agent while-loop kernel is already the default control
path. The remaining built-in architecture still models ordinary capabilities
as named role agents:

- `ResearchAgent`;
- `OrchestratorAgent`;
- `CompareAgent`;
- `FactCheckAgent`;
- `SynthesizeAgent`.

That split is no longer aligned with the runtime. Retrieval, comparison,
fact-checking, synthesis, file access, structured analysis, Skills, and MCP are
capabilities selected by the model inside one loop. They are not separate
control-flow owners.

The current `AgentDefinition.allowed_tools` also makes a static list of tool
schemas visible on every model turn. This works while the tool pool is small,
but it does not scale to RAG, asset tools, local operations, Skills, MCP
servers, and future integrations. Loading all schemas up front wastes context
and makes tool selection less reliable.

This design applies the mature Claude-style harness pattern:

- one model-driven loop;
- a small always-visible core tool set;
- deferred tool discovery through Tool Search;
- progressive Skill loading;
- MCP tools added to the same searchable catalog;
- a generic bounded `task` tool for context isolation;
- no role-specific child agents in the default runtime.

## Decision

Replace role-agent capability routing with a provider-neutral capability
runtime around the existing loop.

```text
user task
   |
   v
prepare context
   |-- runtime policy
   |-- active tool schemas
   |-- Skill catalog summaries
   |-- connected MCP summaries
   v
model turn
   | finish ----------------------------------------------> stop hooks
   | pause ------------------------------------------------> checkpoint
   | core/deferred tool call
   v
approval -> execute -> structured observation -> next turn
   |
   +-- tool_search -> activate bounded matching tools -----+
   +-- load_skill  -> return one Skill body ---------------+
   +-- task        -> isolated child loop, result only ----+
   +-- MCP tool    -> namespaced external execution -------+
```

The loop remains the only agent control mechanism. Tool Search, Skills, MCP,
and subagents are harness services around it.

## Design Principles

1. The model chooses actions; the harness owns safety, visibility, budgets, and
   execution.
2. Capabilities are tools or instructions, not named agents.
3. Only a small stable set of tools is visible on every turn.
4. Deferred tools become visible through structured Tool Search, not keyword
   intent routing.
5. Tool discovery never grants permissions. Approval and access policy remain
   authoritative at execution time.
6. Skills provide on-demand instructions and workflows. They do not execute
   code or bypass tool policy.
7. MCP extends the same typed tool catalog and uses the same approval,
   checkpoint, retry, and audit paths.
8. Subagents exist only for bounded context or execution isolation.
9. RAG evidence, citations, scores, provenance, and asset references must
   survive discovery, execution, delegation, and finalization.
10. Dynamic state remains bounded and checkpointable.

## Non-Goals

- Do not add a `MainAgent` or another root role.
- Do not retain `OrchestratorAgent` as a hidden router.
- Do not infer capability selection with natural-language keyword rules.
- Do not load every Skill body or MCP schema into every model call.
- Do not allow arbitrary MCP server installation or connection without an
  explicit configured allowlist and approval policy.
- Do not replace the existing typed tool execution, approval, checkpoint,
  retry, fallback, compaction, or citation systems.
- Do not make LangGraph the inner single-agent loop again.

## Runtime Policy Replaces Role Definition

`AgentDefinition` currently combines two unrelated concepts:

- role identity (`agent_type`, role description, role prompt);
- operational policy (budgets, model choice, tools, limits, output schema).

The target root-loop contract is `AgentRuntimePolicy`:

```python
@dataclass(frozen=True)
class AgentRuntimePolicy:
    system_instructions: str
    core_tool_names: tuple[str, ...]
    deferred_tool_filter: ToolCatalogFilter
    access_policy_ceiling: AccessPolicy | None
    token_budget: int
    work_budget: int
    max_iterations: int
    max_depth: int
    max_active_deferred_tools: int
    model_selection: ModelSelectionPolicy
    tool_policy: ToolPolicy
    output_model: type[BaseModel] | None = None
```

This is runtime configuration, not an Agent persona. The default policy carries
general grounding and safety instructions but no research, comparison,
fact-check, or synthesis identity.

`ToolCatalogFilter` is an explicit allow/deny boundary:

```python
class ToolCatalogFilter(BaseModel):
    allowed_sources: frozenset[str]
    allowed_names: frozenset[str] = frozenset()
    denied_names: frozenset[str] = frozenset()
    allowed_mcp_servers: frozenset[str] = frozenset()
```

An empty `allowed_names` means names are selected by source and policy, not
that every process tool is automatically permitted.

During migration, an adapter may convert an `AgentDefinition` into an
`AgentRuntimePolicy` for external callers. The default CLI and service path
must stop selecting a role by `agent_type`. The adapter is removed after
downstream imports and persisted callers migrate.

`AgentRunConfig` keeps run identity, parent identity, budgets, depth, effective
access policy, and memory policy. The effective policy must be no broader than
the runtime policy ceiling. Its legacy `agent_type` field becomes optional
checkpoint compatibility metadata and does not select behavior.

## Module Boundaries

The new responsibilities fit the existing package without introducing another
framework:

```text
rag/agent/
  capabilities/
    catalog.py       # ToolCatalogEntry, ToolCatalog, ActiveToolSet
    tool_search.py   # typed search contract and local search runner
    skills.py        # SkillManifest, scanning, bounded loading
    mcp.py           # configured MCP lifecycle and ToolSpec adaptation
  core/
    definition.py    # AgentRuntimePolicy and temporary compatibility adapter
    delegation.py    # generic TaskInput/TaskOutput contracts
    subagent_runner.py
  tools/
    registry.py      # typed execution registry remains authoritative
```

Provider-native Tool Search adaptation stays with the existing model/provider
integration rather than entering the catalog. CLI and API assembly create the
catalog and effective runtime policy, then pass them into `AgentService`.

## Tool Catalog, Visibility, and Execution

The current `ToolRegistry` stores both schemas and runners. The new runtime
separates three responsibilities:

### `ToolCatalog`

`ToolCatalog` is the complete searchable inventory of tools that the current
process is allowed to know about. It contains built-in, RAG, asset, local, and
connected MCP entries.

```python
class ToolCatalogEntry(BaseModel):
    name: str
    description: str
    exposure: Literal["core", "deferred", "internal"]
    source: Literal["builtin", "rag", "mcp", "local"]
    tags: tuple[str, ...] = ()
    server_name: str | None = None
    schema_fingerprint: str
```

The catalog maps each entry to an existing `ToolSpec`; it does not duplicate
input or output schemas in searchable metadata. `schema_fingerprint` is a
stable hash of the input/output schemas and execution-relevant policy metadata.
Searchable metadata is bounded and excludes full Skill bodies, large examples,
secrets, and runner state.

Exposure semantics:

- `core`: schema is included in every model turn;
- `deferred`: searchable but omitted until selected;
- `internal`: available to harness code only and never exposed directly.

### `ActiveToolSet`

`ActiveToolSet` is request-scoped and checkpointed:

```python
class ActivatedToolRef(BaseModel):
    name: str
    schema_fingerprint: str
    activated_at_iteration: int
    last_used_iteration: int


class ActiveToolSet(BaseModel):
    core_names: tuple[str, ...]
    deferred: tuple[ActivatedToolRef, ...] = ()
```

It is bounded by `max_active_deferred_tools`. Activating a tool adds its schema
to subsequent model turns. Activation does not run the tool and does not
change its permission requirements.

When the bound is reached, the runtime evicts the least recently used deferred
tool that is not referenced by a pending call, approval request, or current
iteration result. This is deterministic runtime bookkeeping, not semantic
intent routing. The catalog remains available for rediscovery.

### `ToolRegistry`

`ToolRegistry` remains the execution authority:

- validates typed inputs and outputs;
- resolves request-scoped runners;
- enforces runner availability;
- supplies `ToolSpec.permissions` to approval;
- preserves idempotency, retry, concurrency, timeout, and audit metadata.

The catalog never executes a tool. The active set never bypasses the registry.

## Tool Search

`tool_search` is a core tool. It searches only catalog metadata visible under
the current runtime policy.

```python
class ToolSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=10)
    source: Literal["builtin", "rag", "mcp", "local", "any"] = "any"


class ToolSearchMatch(BaseModel):
    name: str
    description: str
    source: str
    tags: tuple[str, ...]


class ToolSearchOutput(BaseModel):
    matches: tuple[ToolSearchMatch, ...]
    activated_tool_names: tuple[str, ...]
```

Search uses structured catalog metadata. It must not use scattered
task-keyword checks. The initial implementation may use deterministic lexical
ranking over names, descriptions, and tags; semantic ranking may be added
behind the same contract when evaluation shows a need.

The runtime handles a successful result in two steps:

1. append the bounded search result as a normal tool observation;
2. add bounded matched references and their schema fingerprints to
   `ActiveToolSet.deferred` for the next model turn.

Unknown, denied, internal, disconnected, or policy-excluded tools are not
returned. Search and activation are checkpointed so resume reproduces the same
visible tool set.

Provider integration has one normalized contract:

- providers with native deferred Tool Search may receive deferred tool
  metadata through their native API;
- providers without it receive the local `tool_search` core tool;
- provider responses normalize into the same active-set transition.

Provider-native support is an optimization, not a second semantic path, and is
not required before role-agent removal.

## Default Tool Exposure

The root loop keeps a deliberately small conditional core set:

- `tool_search`;
- `load_skill` when at least one Skill is configured;
- `task` when delegation is enabled and remaining depth is positive;
- existing loop-control or memory tools that are proven necessary on every
  run.

Everything else is deferred unless a product-specific caller explicitly
installs a narrower policy:

- RAG retrieval and grounded-answer tools;
- asset inspection and structured analysis;
- file and primitive operations;
- LLM helper tools;
- connected MCP tools.

This does not mean deferred tools are initialized or executed eagerly. Catalog
metadata may be loaded cheaply, while expensive runners, clients, embeddings,
or external connections remain lazy and request-scoped.

## RAG as a General Capability

RAG is represented by typed deferred tools, not a `ResearchAgent`:

- `vector_search`;
- `keyword_search`;
- `grounding`;
- `rerank`;
- `graph_expand`;
- `rag_search_answer`;
- asset listing, inspection, slicing, and analysis tools.

Their catalog metadata identifies retrieval, grounding, citation, document,
table, and asset capabilities. Tool Search makes the relevant schemas visible;
the ordinary model loop decides which tools to call and in what order.

All existing RAG output contracts remain authoritative. Tool Search results
contain only tool metadata. Evidence enters state only through actual RAG tool
results, preserving:

- evidence IDs;
- citation IDs and anchors;
- document and asset IDs;
- retrieval and rerank scores;
- expression and computation provenance;
- grounding and evaluation metadata.

No hidden retrieval occurs inside Tool Search.

## Skill Loading

Skills are local, versioned instruction packages with a `SKILL.md` entrypoint.
They are not tools and not agents.

```python
class SkillManifest(BaseModel):
    name: str
    description: str
    path: Path
    required_tool_tags: tuple[str, ...] = ()
```

At run preparation:

- scan configured Skill roots;
- validate unique names and safe paths;
- place only name and one-line description in the system context;
- keep full content outside model context.

`load_skill(name)` is a core typed tool. It returns one bounded Skill document
or a visible error. Loading a Skill does not automatically activate tools,
grant permissions, or execute instructions. The model may subsequently use
Tool Search to discover tools recommended by the Skill.

Loaded Skill content is treated as configured harness instruction, not as
retrieved factual evidence. Its name, version/hash, and load event are
checkpointed; the full body may be referenced by content hash rather than
duplicated in long-lived state. A hash mismatch during resume is visible and
requires an explicit restart or acceptance of the changed Skill; the runtime
must not silently substitute new instructions into an old run.

## MCP Integration

MCP is an external tool source, not a separate agent layer.

`MCPConnectionManager` owns configured server lifecycle:

- server configuration and allowlisting;
- transport creation;
- connection health;
- tool discovery;
- namespacing;
- runner dispatch;
- disconnect and failure diagnostics.

Discovered MCP tools become ordinary `ToolSpec` plus `ToolCatalogEntry`
instances. Names are normalized as:

```text
mcp__<server>__<tool>
```

MCP annotations map into local policy:

- read-only hints inform concurrency and approval defaults;
- destructive or mutating hints require confirmation;
- external network and user-data access are explicit permissions;
- unknown annotations fail conservatively and remain visible in diagnostics.

Configured servers may connect during runtime assembly, but their tool schemas
remain deferred. A future `connect_mcp` model tool, if provided, may connect
only preconfigured allowlisted servers and must pass approval. Arbitrary server
commands, URLs, credentials, or installation requests are outside this design.

MCP connection failure does not remove built-in capabilities. It produces a
visible diagnostic, and Tool Search must not return unavailable MCP tools.

## Generic `task` Subagent

The default runtime exposes one generic `task` tool. It replaces
`agent_research`, `agent_compare`, `agent_factcheck`, and `agent_synthesize`.

```python
class TaskInput(BaseModel):
    task: str = Field(min_length=1)
    context_summary: str | None = None
    required_outputs: tuple[str, ...] = ()
    tool_query: str | None = None
    token_budget: int | None = Field(default=None, gt=0)


class TaskOutput(BaseModel):
    conclusion: str
    key_facts: tuple[str, ...]
    evidence_refs: tuple[DelegatedEvidenceRef, ...]
    citations: tuple[AnswerCitation, ...]
    status: Literal["done", "failed", "paused"]
    child_run_id: str
    stop_reason: str | None = None
```

Execution semantics:

1. derive a child `AgentRunConfig` from the parent;
2. allocate a bounded child token and work budget from the parent ledger;
3. start a fresh message history containing only the task and bounded context
   summary;
4. derive neutral child instructions that require completion, concise return,
   and no further delegation;
5. apply the same runtime policy, restricted to the parent's access policy and
   tool catalog filter;
6. disable `task` by default in the child to prevent recursive delegation;
7. let the child use Tool Search over its restricted catalog;
8. return only the bounded typed result, citations, and evidence references;
9. discard or externalize the child transcript according to audit policy.

`tool_query` is a hint, not a permission grant. The child may discover only
tools allowed by the derived policy. Mutating tools still require approval, and
child approval pauses must propagate explicitly rather than being auto-approved.

If the child pauses, the parent `task` operation remains `started` with its
`child_run_id`; the parent loop also pauses and checkpoints the delegation
reference. Resume applies the human response to the child, completes or
re-pauses the child, then records exactly one final `task` result in the parent.
The parent must not restart the child from its original prompt.

The subagent has no role name. "Research this", "compare these", "verify this",
and "synthesize this evidence" are task prompts handled by the same isolated
loop.

## State and Checkpointing

Add bounded capability state to `LoopState`:

- active deferred tool references and schema fingerprints;
- loaded Skill references;
- connected MCP server summaries;
- child delegation references;
- capability diagnostics.

Do not persist:

- full catalog copies;
- full Skill bodies;
- MCP clients or transport objects;
- complete child transcripts;
- full tool schemas on every checkpoint.

Resume reconstructs the catalog from configured sources, validates saved names
and schema fingerprints, restores the active set, then resumes the loop. A
missing or changed schema invalidates that active reference visibly. If a
previously active MCP tool is no longer available, resume records a visible
diagnostic and does not replay it.

Tool execution records remain authoritative. An activated tool is not an
executed tool. Confirmed completed calls are not replayed; non-idempotent
`started/unknown` calls still require human confirmation.

## LangGraph Boundary

LangGraph remains an outer orchestrator for explicit application workflows:

- parallel branches and joins;
- long-lived human workflows;
- multi-session teams;
- application-specific DAGs.

Tool Search, Skill loading, MCP discovery, and generic `task` delegation belong
to the single-loop harness and do not require graph nodes.

## Failure Semantics

- Tool Search returns a typed empty result when no permitted tools match.
- Invalid search output or activation fails visibly and does not mutate the
  active set.
- Active-set limits produce a typed bounded result instead of silently loading
  more schemas.
- Skill lookup, parse, size, or hash failures are visible tool errors.
- Skill instructions cannot elevate permissions.
- MCP discovery and execution failures include server and tool identity.
- Disconnected MCP tools are not searchable or executable.
- Child budget, depth, timeout, or approval exhaustion returns a typed task
  failure to the parent.
- Child failures do not become successful conclusions.
- Deferred tool discovery never suppresses approval, retry, provenance, or
  checkpoint failures.

## Migration

### Increment 0: Freeze Capability Baseline

Before changing registration or prompts, freeze scenarios and metadata for:

- direct RAG question answering with citations;
- document comparison;
- fact verification;
- synthesis from supplied evidence;
- asset analysis;
- file operations and approval;
- delegated child execution;
- checkpoint resume.

The baseline must capture final status, tool calls, citations, evidence
metadata, diagnostics, and approval behavior.

### Increment 1: Catalog and Active Tool Set

- Introduce `ToolCatalogEntry`, `ToolCatalog`, and `ActiveToolSet`.
- Introduce `AgentRuntimePolicy` plus the temporary `AgentDefinition` adapter.
- Classify existing tools as core, deferred, or internal.
- Make model context consume the active set instead of
  `AgentDefinition.allowed_tools`.
- Keep existing role definitions temporarily as compatibility inputs.

### Increment 2: Tool Search

- Add typed local `tool_search`.
- Checkpoint activation transitions.

### Increment 3: Generic Task Delegation

- Replace agent-type delegation with `task`.
- Derive child policy and catalog scope from the parent.
- Preserve bounded evidence and citation return contracts.
- Prevent recursive delegation by default.
- Implement child pause/resume without replay.

### Increment 4: Runtime Policy and Role-Agent Removal

- Switch the default service and CLI to `AgentRuntimePolicy`.
- Remove role selection from the default service path.

Delete:

```text
rag/agent/builtin/research.py
rag/agent/builtin/orchestrator.py
rag/agent/builtin/compare.py
rag/agent/builtin/factcheck.py
rag/agent/builtin/synthesize.py
```

Remove:

- `BUILTIN_AGENT_DEFINITIONS`;
- built-in role selection from CLI;
- `AgentRegistry` from the default path;
- `agent_research`, `agent_compare`, `agent_factcheck`, and
  `agent_synthesize`;
- role-specific prompts, factories, tests, and documentation.

Keep:

- the generic child-loop runner;
- typed delegation results;
- budget and depth propagation;
- approval propagation;
- evidence and citation dehydration;
- LangGraph outer orchestration adapters.

### Increment 5: Skills

- Add Skill manifest scanning and conditional `load_skill`.
- Checkpoint Skill hashes and handle resume drift visibly.

### Increment 6: MCP

- Add configured MCP lifecycle and namespaced deferred tools.
- Route MCP execution through the existing ToolRegistry and approval policy.
- Keep disconnected or unhealthy tools out of search results.

### Increment 7: Provider Optimization and Compatibility Cleanup

- Add provider-native deferred Tool Search adapters where supported.
- Verify parity through the normalized active-set contract.
- Remove the `AgentDefinition` compatibility adapter after downstream callers
  migrate.
- Retain legacy checkpoint fields only as long as existing persisted runs
  require them.

The implementation must be planned as independently reviewable deliveries:

1. catalog, active set, local Tool Search, and runtime policy foundation;
2. generic `task` and role-agent removal;
3. Skill loading;
4. MCP integration;
5. optional provider-native Tool Search and compatibility cleanup.

## Testing

### Unit Tests

- enabled core tools are always visible;
- deferred tools are absent before search;
- Tool Search returns only permitted catalog entries;
- activation is bounded, deterministic, and checkpointed;
- active schemas appear on the next model turn;
- stale tools can be evicted without removing pending calls;
- RAG search metadata does not fabricate evidence;
- Skill catalog includes summaries but not bodies;
- `load_skill` validates names, paths, size, and hashes;
- Skill loading does not grant tools or permissions;
- MCP namespacing and annotation mapping are deterministic;
- unavailable MCP tools are not searchable;
- `task` creates a fresh context and cannot recurse by default;
- child policy cannot exceed parent access or budget;
- child evidence and citations remain traceable.

### Integration Tests

- a general question finishes without loading RAG tools;
- a grounded question discovers RAG tools, retrieves, and returns citations;
- a table question discovers asset tools and preserves expression provenance;
- a comparison task uses ordinary tools without `CompareAgent`;
- a fact-check task runs directly or through generic `task`;
- a supplied-evidence synthesis finishes without `SynthesizeAgent`;
- a Skill is summarized in context and loaded only when selected;
- an MCP tool is discovered, approved when necessary, executed, and audited;
- checkpoint resume restores active tools and loaded Skill references;
- delegated child approval pauses propagate to the parent;
- provider-native and local Tool Search produce equivalent active tool sets.

### Removal Gate

Role agents may be deleted only after:

- all capability parity scenarios pass;
- no default CLI or service path selects an agent role;
- no active tool schema depends on an agent-role registry;
- generic `task` covers bounded child isolation;
- RAG citations and metadata remain unchanged or improve;
- checkpoint and approval tests pass.

## References

- Local learning implementation:
  `learn-claude-code/s06_subagent/code.py`,
  `learn-claude-code/s07_skill_loading/code.py`,
  `learn-claude-code/s19_mcp_plugin/code.py`, and
  `learn-claude-code/s20_comprehensive/code.py`.
- Claude Tool Search:
  <https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool>
- Claude Code Skills:
  <https://code.claude.com/docs/en/skills>
- Claude Code subagents:
  <https://code.claude.com/docs/en/sub-agents>
- Claude Code MCP:
  <https://code.claude.com/docs/en/mcp>
