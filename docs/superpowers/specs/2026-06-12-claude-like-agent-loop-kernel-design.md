# Claude-Like Agent Loop Kernel Design

## Problem

The current single-agent runtime is graph-driven and goal-contract-driven.
`GoalSpec`, `open_gaps`, context binding, and `SatisfactionChecker` participate
in every turn and can reject a model's request to finish. This makes a
heuristic semantic contract more authoritative than the model even when the
task is exploratory and the correct analysis path cannot be known in advance.

The same runtime also combines several independent concerns:

- model/tool loop control;
- semantic completion policy;
- observation extraction;
- approval and tool scheduling;
- retry and degraded execution;
- memory compaction;
- checkpoint and resume;
- LangGraph routing.

The result is harder to read and extend than a production agent loop needs to
be. In particular, adding a tool or discovering a new analysis path can require
changes to centralized goal and observation logic.

## Decision

Build a Claude-like asynchronous loop kernel for one agent:

```text
prepare context
      |
      v
call model
  | tool calls ----------------> approve -> execute -> observe --+
  | pause ---------------------> paused                          |
  | finish -> stop hooks                                           |
                 | accept/warn -> finished                         |
                 | block -> append feedback -----------------------+
                 | halt -> halted
```

The model owns semantic progress and proposes when the task is finished. The
runtime owns hard operational boundaries. Optional stop hooks inspect a proposed
finish but do not control ordinary tool selection.

This is a control-flow change, not a capability reduction. RAG, citations,
approval, retries, fallback, compaction, checkpoints, streaming, and subagents
remain separate services used by the loop.

LangGraph remains available for workflows that are genuinely graph-shaped:
multi-agent coordination, explicit parallel branches and joins, long-lived
human workflows, and application-specific DAGs. It is no longer required to
express the ordinary model -> tool -> result cycle of one agent.

## Design Principles

1. Keep the single-agent loop sequentially readable.
2. Separate immutable run configuration from bounded mutable loop state.
3. Give the model semantic control; keep safety and resource limits
   deterministic.
4. Preserve typed tool contracts, evidence provenance, and visible failures.
5. Install strict completion requirements explicitly rather than inferring a
   universal goal contract from every natural-language task.
6. Use LangGraph where graph semantics add value, not as a mandatory inner
   loop.
7. Migrate through behavior parity tests instead of maintaining two permanent
   runtimes.

## Core Components

### `AgentLoop`

`rag.agent.loop.runtime.AgentLoop` coordinates one agent run. Its main method is
an asynchronous while loop with explicit transition and terminal values.

It depends on narrow ports rather than concrete UI, CLI, or LangGraph code:

- model turn provider;
- context manager;
- tool runner;
- checkpoint store;
- stop-hook runner;
- event sink;
- optional child-agent runner.

The loop does not contain retrieval, table analysis, CLI output, or
tool-specific result parsing.

The initial module layout stays intentionally small:

```text
rag/agent/loop/
  runtime.py       # AgentLoop and sequential control flow
  state.py         # LoopState, transitions, terminal results
  stop_hooks.py    # hook protocol, verdicts, bounded runner
```

The existing `controller.py` becomes compatibility code and is removed when the
old single-agent graph is retired. New files are added only when one of these
modules develops a second independent responsibility.

### Run Configuration

Run configuration is snapshotted at run entry and treated as immutable for the
life of the run. Existing `AgentDefinition` and `AgentRunConfig` remain the
public configuration boundary.

The snapshot includes:

- model and tool availability;
- iteration, token, and output budgets;
- approval policy;
- retry and fallback policy references;
- compaction policy;
- installed stop hooks;
- checkpoint identity;
- event and tracing configuration.

Configuration mutation during a run is not allowed. Runtime recovery counters
belong to loop state. Mutable budget accounting remains in `BudgetLedger`;
legacy `budget_committed` and `budget_reserved` values are not treated as
mutable configuration.

### Loop State

The kernel owns a focused `LoopState` `TypedDict`. It does not directly depend
on the current broad graph `AgentState`.

- current messages and bounded observation references;
- pending tool calls and approval requests;
- iteration and recovery counters;
- latest `LoopTransition`;
- bounded runtime diagnostics;
- pause or terminal information;
- final model output.

During migration, a boundary adapter converts between legacy `AgentState` and
`LoopState` for existing graph, service, and checkpoint callers. The adapter is
temporary and is not used inside the loop. The existing checkpoint service
persists the new state; no second persistence backend or permanent parallel
state model is introduced.

`LoopTransition` records the latest transition, not an unbounded trace. Event
sinks may externalize the complete history.

Initial transition reasons include:

- `next_turn`;
- `tool_execution`;
- `approval_required`;
- `stop_hook_blocked`;
- `retry`;
- `fallback`;
- `compaction`;
- `paused`;
- `finished`;
- `max_iterations`.

### Model Turn

The model receives the task, messages, available tool schemas, compacted
context, and recent structured observations. It returns one of three semantic
outcomes:

- tool calls;
- finish with a candidate answer;
- pause with a reason that requires external input.

The runtime continues because actual tool calls are present, not merely because
a provider supplied a nominal stop reason.

The current `synthesize` action is accepted as a compatibility alias for
`finish` during migration. New internal contracts use `finish`.

### Tool Runner

The existing typed tool registry, approval policy, idempotent retry rules, and
concurrency-safe scheduling remain authoritative.

The runner:

1. validates tool calls;
2. evaluates approval before side effects;
3. executes safe read batches concurrently and mutations serially;
4. returns typed `ToolResult` values;
5. exposes failures as observations rather than swallowing them.

Tool results enter the next model turn as canonical structured observations.
Evidence locators, retrieval scores, rerank scores, citations, expression
provenance, and artifact references are preserved.

Large payloads remain externalized or summarized. Long-lived state stores
references and bounded summaries.

### Stop Hooks

Stop hooks run only after the model proposes `finish`.

```python
class StopVerdict(BaseModel):
    action: Literal["accept", "warn", "block", "halt"]
    code: str
    message: str | None = None
```

A hook may:

- `accept`: allow finalization;
- `warn`: allow finalization and attach a visible diagnostic;
- `block`: append concise feedback and let the model continue;
- `halt`: produce a typed paused or failed terminal result.

Blocking is bounded by a configured maximum. Repeated equivalent feedback does
not create an infinite loop.

Hook execution failures are visible. Advisory hook failures degrade to a
warning and allow completion. Hooks explicitly marked critical fail closed and
produce a typed halt.

Default hard hooks cover only real invariants, such as required output schema
validation. Task-specific hooks may enforce explicit user requirements:

- citations are required;
- a named source must be used;
- output must match a JSON schema;
- calculations must include reproducible expressions;

Evidence completeness, plan completeness, or inferred answer quality are
advisory by default. They do not globally revoke the model's ability to finish.

Tool approval never belongs to a stop hook. It remains a hard pre-execution
gate in the tool runner.

### Goal Contract Migration

`GoalSpec` is removed from the default control path.

- Natural-language tasks are not automatically converted into authoritative
  answer/evidence/computation gaps.
- `open_gaps` no longer determines which tool the model may call.
- `premature_synthesis` no longer causes a global pause or forced
  `llm_summarize`.
- `SatisfactionChecker` is decomposed into optional stop hooks where its checks
  represent explicit requirements.
- Planning remains an advisory model aid and may change after new evidence.

Existing callers that explicitly provide a strict goal contract receive a
compatibility adapter that installs corresponding stop hooks. The adapter is a
migration boundary, not a new universal runtime abstraction.

`goal_runtime.py` remains only while compatibility imports and tests are moved.
It is deleted after its contracts have clear owners.

## Checkpoint and Resume

A while loop does not prevent durable execution.

The kernel checkpoints after each externally meaningful transition:

- model output accepted;
- tool results recorded;
- approval requested;
- stop hook blocked completion;
- compaction completed;
- terminal result produced.

Approval returns a typed resumable pause. Resume loads the checkpointed
`LoopState`, applies the external response, and re-enters `AgentLoop.run()`.
Tool calls already recorded as completed are not replayed.

The loop depends on a small `CheckpointStore` protocol. An adapter over the
existing LangGraph `BaseCheckpointSaver` supplies the initial implementation,
so the project keeps the same memory and SQLite backends and serialized data
contracts. It does not introduce a second persistence database.

## Retry, Fallback, and Compaction

These mechanisms remain orthogonal to semantic completion:

- model retries and provider fallback live in the model gateway;
- tool retries remain limited to idempotent tools;
- compaction runs before model calls when context policy requires it;
- recovery counters and degraded execution are recorded in state diagnostics;
- all retry, fallback, and compaction transitions are observable.

None of these mechanisms create or close semantic goal gaps.

## LangGraph Integration

LangGraph becomes an outer orchestration option.

Two supported integration shapes are:

1. A graph node invokes one `AgentLoop` run and receives a typed terminal
   result.
2. A parent graph invokes multiple bounded agents as tools or subgraphs and
   handles explicit branch, join, and human workflow semantics.

The kernel must not import graph nodes. Graph adapters may import and invoke the
kernel.

Existing public services continue to expose the same run and resume behavior.
The default internal implementation switches only after parity tests cover
tool use, approval, checkpoint resume, RAG citations, subagents, retry,
fallback, compaction, and terminal results.

## Failure Semantics

- Invalid model output becomes a visible structured failure or bounded retry.
- A missing model provider produces the existing degraded diagnostic and a
  typed pause or failure; it does not activate hidden deterministic semantics.
- Tool validation and execution failures become typed observations.
- Non-idempotent tool calls are never replayed automatically.
- Approval denials are visible model observations and may lead to replanning.
- Context overflow triggers compaction or a typed terminal error.
- Maximum iterations always terminates the loop.
- Stop-hook failures follow their declared advisory or critical policy.
- Checkpoint write failure is surfaced and prevents reporting a durable pause
  or durable completion.

## Migration Sequence

### Increment 1: Contracts and Kernel

- Add loop transition, terminal, and stop-hook contracts.
- Implement the loop against existing model, tool, context, and checkpoint
  ports.
- Keep existing graph runtime active.

### Increment 2: Behavior Parity

- Run the same scenario fixtures through graph and loop implementations.
- Cover tool execution, approval resume, RAG provenance, subagents, retry,
  fallback, compaction, and finalization.
- Resolve behavioral differences explicitly rather than adding broad
  compatibility flags.

### Increment 3: Default Switch

- Switch `AgentService` to the loop kernel.
- Retain graph adapters for explicit complex workflows.
- Convert strict `GoalSpec` inputs into stop hooks.
- Remove goal-driven routing from prompts and the default runtime.

### Increment 4: Cleanup

- Remove obsolete graph nodes used only by the old single-agent loop.
- Delete compatibility-only goal runtime code after downstream imports move.
- Preserve specialized LangGraph workflows and their tests.

## Testing

Focused unit tests cover:

- model tool calls continue the loop;
- model finish invokes stop hooks;
- blocked finish feeds bounded feedback into the next model turn;
- accepted and warned finishes return typed results;
- critical hook failure halts visibly;
- maximum iterations terminates;
- checkpoint resume does not replay completed tools;
- approval denial is observable and allows replanning;
- compaction, retry, and fallback record explicit transitions;
- strict goal compatibility installs hooks without restoring `open_gaps`
  routing.

Integration tests cover:

- RAG retrieval through final answer with citations and retrieval metadata;
- table analysis with expression provenance;
- mutating tool approval and resume;
- parent-agent and child-agent budget propagation;
- degraded provider initialization;
- existing CLI run and resume behavior;
- LangGraph outer workflow invoking the loop kernel.

Behavior evaluation uses the same realistic task corpus before and after the
default switch. It measures:

- task success and answer quality;
- citation and provenance correctness;
- tool selection and tool error rates;
- total model turns and tool calls;
- token use and latency;
- retry, fallback, compaction, and stop-hook frequency.

Evaluations verify outcomes without requiring one exact tool sequence. Multiple
valid analysis paths remain acceptable. The corpus includes multi-step RAG,
mixed retrieval and calculation, table analysis, approval denial, noisy or
insufficient evidence, and tasks that should finish without tools.

Verification requires:

- focused Agent tests;
- the complete test suite;
- Ruff;
- mypy;
- import-linter.

## Non-Goals

- Rewriting every existing tool or provider.
- Removing LangGraph from the dependency set.
- Building a new persistence database.
- Inferring perfect answer correctness with another global evaluator.
- Keeping graph and loop implementations as permanent selectable modes.
- Renaming unrelated public APIs.

## Rationale and References

Anthropic describes effective agents as models using tools in a feedback loop
and recommends starting with simple, composable patterns rather than adding
framework complexity prematurely:

- https://www.anthropic.com/research/building-effective-agents
- https://www.anthropic.com/engineering/writing-tools-for-agents

LangGraph remains valuable for durable graph execution, interrupts, and
subgraph persistence:

- https://docs.langchain.com/oss/python/langgraph/persistence
- https://docs.langchain.com/oss/python/langgraph/interrupts
- https://docs.langchain.com/oss/python/langgraph/use-subgraphs

The local Claude Code reference demonstrates the same separation:
`src/query.ts` owns the readable loop, while stop hooks, retry, and tool
orchestration live in focused services.
