# Agent Skill Layer Design

## Purpose

Build a mature, usable skill layer for the local agent runtime.

A skill is reusable workflow knowledge. It tells the model how to perform a
class of tasks. A tool performs typed actions. A skill can reference tools and
local helper assets, but it does not execute by itself and it must not weaken
the normal tool approval path.

The target shape is:

```text
user task
  -> agent loop
  -> model sees compact available-skill listing
  -> model calls invoke_skill when a listed skill matches
  -> runtime loads full SKILL.md into required dynamic context
  -> model uses normal tools to act
  -> compaction and resume keep active loaded skills visible
```

The first production version should be conservative: inline local skills,
checkpoint-safe state, no remote loading, no forked skill execution, and no
implicit script execution.

## Answer: Will Adapting The Xlsx Skill Hurt It?

It should not hurt the xlsx skill if the adaptation preserves the core workflow
rules and only changes runtime-specific execution details.

What to preserve from Claude's xlsx skill:

- trigger semantics for `.xlsx`, `.xlsm`, `.csv`, and `.tsv`;
- spreadsheet quality requirements;
- formula-first guidance for created workbooks;
- verification and recalculation expectations;
- pandas/openpyxl workflow guidance;
- formatting and existing-template preservation rules.

What must change for this agent:

- script references must use `${SKILL_DIR}` or a materialized workspace path,
  not `python scripts/recalc.py` from the current working directory;
- input spreadsheets should start with this agent's `structured_probe` and
  workspace file manifest when available;
- helper scripts must enter the workspace explicitly before `run_python` uses
  them, or run through a normal approved command with an absolute path;
- the skill should cite workspace-relative input/output paths and generated
  artifacts.

Do not edit an upstream skill blindly. Keep an adapted project-local skill that
states which upstream rules are preserved and which runtime rules are local.
This keeps behavior strong while making it executable in this agent.

## Non-Goals For Phase One

- No forked skill subagents.
- No plugin marketplace.
- No remote or MCP-provided skills.
- No automatic execution of scripts referenced by `SKILL.md`.
- No hidden prompt-only skill injection without audit.
- No skill-specified permission escalation.

## Recommended Approach

Use a first-class inline skill runtime.

Alternatives:

- Prompt-only skills are too weak. They lack invocation audit, checkpoint state,
  and reliable compaction behavior.
- Skill-as-subagent from day one is too heavy. It adds child lifecycle,
  checkpointing, and state merge complexity before inline skills are proven.
- First-class inline skills give the mature core: progressive disclosure,
  explicit invocation, tool-bound execution, and checkpointable active context.

## Components

```text
rag/agent/skills/
  models.py          # manifests, ids, loaded refs, invocation records, state
  loader.py          # scan roots, parse SKILL.md, fingerprint content
  catalog.py         # identity resolution, listing, search, duplicate policy
  runtime.py         # invoke/load/materialize/restore active skills
  context.py         # available and loaded skill prompt sections
  invocation.py      # invoke_skill ToolSpec and runner
  assets.py          # safe skill asset materialization helpers
  policy.py          # source trust and per-skill visibility policy
```

`ToolRegistry` should only contain the resident tools used to interact with the
skill layer, such as `invoke_skill` and optionally `materialize_skill_asset`.
The skills themselves do not become tools.

## Skill Identity

Skill lookup must be deterministic. The model cannot see one skill and load
another.

Use a stable `skill_id`:

```text
<source>:<name>
```

Examples:

```text
project:xlsx
user:xlsx
bundled:spreadsheets
plugin.github:gh-fix-ci
```

Rules:

- `SkillManifest.name` is the author-facing name from frontmatter.
- `SkillManifest.skill_id` is the runtime-facing unique id.
- `<available_skills>` lists `skill_id`, not only bare `name`.
- `invoke_skill.name` accepts `skill_id`.
- Bare names are allowed only when exactly one visible skill has that name.
- Ambiguous bare names return `ambiguous_skill_name` with candidate ids.
- Duplicate physical files are deduped by resolved `SKILL.md` path.
- Duplicate names from different sources are not last-write-wins; they remain
  distinct ids and receive a diagnostic if both are visible.

Source priority is used for sorting and optional default selection, not for
silently overwriting another skill.

## Skill Format

Directory layout:

```text
.agents/skills/<skill-name>/SKILL.md
.agents/skills/<skill-name>/references/...
.agents/skills/<skill-name>/scripts/...
```

Minimum `SKILL.md`:

```md
---
name: xlsx
description: Use for spreadsheet input or spreadsheet output tasks.
---

Workflow instructions here.
```

Supported phase-one frontmatter:

```yaml
name: string
description: string
when_to_use: string | null
version: string | null
allowed_tools: string[] | null
paths: string[] | null
disable_model_invocation: bool
```

Unknown fields are preserved in `extra` for compatibility. Unknown fields that
look like execution policy, such as `hooks`, `model`, `agent`, or `context`,
must produce a warning in phase one because they are not active yet.

Reserved future fields:

```yaml
context: inline | fork
agent: string
model: string
effort: string
hooks: object
```

## Data Contracts

`SkillManifest`:

```python
@dataclass(frozen=True)
class SkillManifest:
    skill_id: str
    name: str
    description: str
    source: SkillSource
    skill_file: Path
    root_dir: Path
    when_to_use: str | None = None
    version: str | None = None
    allowed_tools: tuple[str, ...] = ()
    path_patterns: tuple[str, ...] = ()
    disable_model_invocation: bool = False
    content_fingerprint: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
```

Checkpoint-friendly active skill reference:

```python
class LoadedSkillRef(BaseModel):
    skill_id: str
    name: str
    source: str
    skill_file: str
    root_dir: str
    fingerprint: str
    loaded_at_iteration: int
    args: str | None = None
```

`LoadedSkillRef` is the state object. The full body can be reloaded from disk
when building prompt context. During one live process, `SkillRuntime` may cache
the expanded body, but checkpoint state should not depend on private object
identity.

`SkillState`:

```python
class SkillState(BaseModel):
    visible_skill_ids: tuple[str, ...] = ()
    invoked: tuple[SkillInvocation, ...] = ()
    active: dict[str, LoadedSkillRef] = Field(default_factory=dict)
```

Use `active`, not `loaded_skills`, to make the lifecycle explicit. Historical
invocations stay in `invoked`; only active skills are re-injected.

## Prompt Contract

The model should receive two separate skill sections.

Available listing:

```text
<available_skills>
- project:xlsx: Use for spreadsheet input or spreadsheet output tasks.
- project:code-review: Review code for bugs, regressions, and missing tests.
</available_skills>
```

Loaded skills:

```text
<loaded_skills>
<loaded_skill id="project:xlsx" name="xlsx" source="project" fingerprint="...">
Base directory for this skill: /abs/path/.agents/skills/xlsx

...expanded SKILL.md body...
</loaded_skill>
</loaded_skills>
```

The listing can be optional under budget pressure. Active loaded skills are
required context while active. If active loaded skills do not fit the model
budget, the runtime should pause with a context overflow diagnostic rather than
silently dropping them.

Prompt guidance:

```text
Available skills are reusable workflows. When a skill listed in
<available_skills> matches the user's request, this is a BLOCKING REQUIREMENT:
invoke the relevant skill BEFORE generating any other response about the task.
Only use skill ids from the listing. Do not invoke a skill already present in
<loaded_skills>.
```

Loaded skill content can override generic workflow guidance, but never user
instructions, safety policy, tool permissions, or approval rules.

## Context Assembly

Do not mutate `AgentRuntimePolicy.system_instructions` to smuggle the skill
listing into every service instance. Add skill context as explicit prompt
sections during assembly.

Native tool-calling path:

- `AgentMessageAssembler.build_system_message(...)` receives a `SkillRuntime`
  or pre-rendered `SkillPromptContext`.
- stable section: identity and general skill guidance;
- dynamic section: available listing, active loaded skills, runtime state.

Legacy structured-output path:

- `AgentLLMContextAssembler` gets the same rendered sections;
- add `loaded_skills` to `ContextSectionName`;
- make loaded skills required for tool-decision turns.

This keeps prompt construction testable and avoids accumulating duplicated skill
text inside `system_instructions`.

## Listing Budget

`listing_for_prompt(max_chars=2000)` is required in phase one.

Rules:

- sort by source priority and name;
- include only skills enabled by `SkillPolicy`;
- exclude active skills from the available listing or mark them loaded;
- bundled/project critical skills keep full descriptions first;
- non-critical descriptions are truncated proportionally;
- if still over budget, fall back to name-only entries;
- if still over budget, drop lowest-priority entries and append an omitted
  count.

The catalog must return the same visible set used by `invoke_skill`. Prompt
listing and invocation cannot have different filters.

## Invocation Flow

`invoke_skill` input:

```python
class InvokeSkillInput(BaseModel):
    name: str  # skill_id preferred, bare name only if unambiguous
    args: str | None = None
```

Runtime flow:

1. Resolve input to exactly one visible `SkillManifest`.
2. Apply `SkillPolicy`.
3. Reject `disable_model_invocation` unless explicitly user-forced.
4. Load `SKILL.md` body from disk.
5. Expand `${SKILL_DIR}`, `$SKILL_DIR`, and `$ARGUMENTS`.
6. Record `SkillInvocation`.
7. Store `LoadedSkillRef` in `SkillState.active`.
8. Return a compact tool result with id, source, fingerprint, and summary.
9. The next model prompt re-injects the full loaded skill from `SkillState`.

The tool result may include the loaded block for immediate model visibility, but
the reliable path is state-backed re-injection on every subsequent turn.

Error codes:

- `skill_not_found`
- `ambiguous_skill_name`
- `skill_disabled`
- `skill_source_untrusted`
- `invalid_skill_manifest`
- `skill_content_changed`
- `skill_asset_not_found`

## Skill Assets And Scripts

Skill scripts are useful, but they must not run implicitly.

Problem: this agent's `run_python(script_path=...)` executes workspace-relative
scripts. A skill script under `.agents/skills/.../scripts` is outside the
workspace created for an agent run. Therefore a mature skill layer needs an
explicit bridge.

Add a resident read/copy tool:

```python
class MaterializeSkillAssetInput(BaseModel):
    skill_id: str
    relative_path: str  # e.g. scripts/recalc.py

class MaterializeSkillAssetOutput(BaseModel):
    workspace_path: str
    source_fingerprint: str
    size_bytes: int
```

Rules:

- skill must already be active;
- path is relative to that skill root;
- no `..`, symlink escape, or absolute input paths;
- only `references/` and `scripts/` are materializable;
- size limit applies;
- copied file lands under `scratch/skills/<safe-skill-id>/...`;
- copying is read-only and low risk;
- executing the copied script still goes through `run_python` or `run_command`
  and normal approval.

For xlsx formula recalculation, the adapted skill should say:

```text
If formula recalculation is required, materialize scripts/recalc.py from this
skill, then run it through run_python or run_command against the workspace file.
Do not call python scripts/recalc.py from the repository root.
```

This preserves the skill's helper-script effect without breaking the workspace
security model.

## Permission And Trust

Skill invocation and tool execution are separate gates.

Skill invocation gate:

- `SkillPolicy` controls source visibility and per-skill enable/disable;
- repo/project skills can auto-load when trusted;
- user/global/external skills require explicit trust configuration;
- disabled skills are not listed and cannot be model-invoked;
- explicit user invocation can override only if policy allows it.

Tool execution gate:

- `allowed_tools` in a skill is an intent declaration and optional narrowing
  overlay;
- it cannot grant unavailable tools;
- it cannot change `ToolSpec.permissions`;
- it cannot bypass approval;
- unknown names in `allowed_tools` become diagnostics, not permissions.

## Checkpoint And Resume

The checkpoint serializer must allow skill model types:

- `SkillState`
- `SkillInvocation`
- `LoadedSkillRef`
- `SkillSource` if enum values are serialized directly

`_migrate_legacy_state()` must normalize:

- missing `skill_state` -> empty `SkillState`;
- dict `skill_state` -> `SkillState.model_validate(...)`;
- old `loaded_skills` field -> new `active` refs if possible.

Resume flow:

1. Load checkpoint.
2. Normalize `SkillState`.
3. For each active `LoadedSkillRef`, reload `SKILL.md`.
4. Compare fingerprint.
5. If unchanged, re-inject normally.
6. If changed, add diagnostic and either:
   - reload current content in normal mode; or
   - pause in strict replay mode.

Compaction must not compact active loaded skills through `tool_results`.
Active loaded skills are rendered from `SkillState.active`.

## Xlsx Project Skill

Use a project-local adapted `project:xlsx` skill.

It should preserve Claude's spreadsheet quality rules, but its operational
workflow should be local-agent specific:

1. Use file manifest and `structured_probe` for input spreadsheets.
2. Use `run_python` with pandas/openpyxl for analysis and workbook editing.
3. Preserve existing workbook formatting when editing.
4. Use formulas rather than hardcoded calculated values when creating models.
5. If recalculation is needed, materialize `scripts/recalc.py` before executing.
6. Verify produced files with `structured_probe` or a targeted Python check.
7. Report artifact paths, sheet names, columns used, row counts, and method.

The project skill can include a concise body plus a `references/upstream.md`
containing the original Claude wording. The body should inline the core
requirements that must always apply; optional long references should be read
only when needed.

## Path-Conditional Visibility

Phase one can load all repo skills. Phase two should support `paths` filters:

```yaml
paths:
  - "**/*.xlsx"
  - "**/*.csv"
```

When a task includes a matching input file or the file manifest contains a
matching kind, the skill is visible. Otherwise it may be omitted from the
listing to save context.

Path matching should be deterministic:

- repo/workspace-relative paths;
- gitignore-style matching;
- tests for positive and negative matches.

## Observability And ACI

The skill layer should expose enough detail to debug model behavior:

- list loaded manifests in CLI/debug output;
- show visible skill ids in prompt-debug mode;
- `invoke_skill` result includes id, source, fingerprint, root dir, and active
  status;
- materialized assets report source and workspace path;
- diagnostics include duplicate names, policy filtering, invalid manifests, and
  fingerprint drift.

The user should be able to answer:

- which skills were visible;
- which skill was invoked;
- which content fingerprint was loaded;
- whether a helper script was copied;
- which normal tool executed any script.

## Phase Plan

### Phase 1: Mature Inline Local Skills

Implement:

- deterministic `skill_id`;
- duplicate-name diagnostics;
- `SkillPolicy` wired into listing and invocation;
- `SkillRuntime`;
- resident `invoke_skill`;
- `SkillState.active`;
- explicit available and loaded skill prompt sections;
- checkpoint allowlist and migration;
- `$ARGUMENTS`, `${SKILL_DIR}`, and `$SKILL_DIR` expansion;
- `materialize_skill_asset`;
- adapted `project:xlsx` skill.

Tests:

- valid and invalid manifests;
- unknown compatibility fields preserved with warnings;
- duplicate same-name skills do not collide;
- bare-name invocation rejects ambiguity;
- listing budget truncation;
- invoke writes `SkillState.active`;
- next model prompt contains `<loaded_skill>`;
- compaction keeps active loaded skill visible;
- checkpoint dump/load restores `SkillState`;
- fingerprint drift creates diagnostic;
- materialize blocks path traversal and copies allowed assets;
- xlsx script path uses materialized workspace file or absolute skill dir, not
  `scripts/recalc.py` from repo root.

### Phase 2: Conditional And Cached Skills

Implement:

- path-filtered visibility;
- skill file change cache invalidation;
- listing cache keyed by visible ids and fingerprints;
- user/global skill source with explicit trust.

### Phase 3: Plugin And MCP Skills

Implement:

- plugin skill source ids;
- plugin-provided skill roots;
- MCP prompt-as-skill only if it satisfies the same manifest contract;
- source-level enable/disable policy.

### Phase 4: Forked Skills

Implement only after inline skills are stable:

- `context: fork`;
- child loop with cloned state;
- bounded child tools;
- returned summary to parent;
- child transcript/audit link;
- max-depth and concurrency limits.

## Acceptance Criteria

The skill layer is not mature until these are true:

- a matching skill is a blocking model requirement and the tool is visible;
- invoking a skill makes it visible in the next prompt from state, not only from
  an old tool result;
- compaction and resume do not remove active skill instructions;
- same-name skills cannot resolve to the wrong source;
- skill helper scripts can run through normal approved tools without wrong-path
  failures;
- xlsx spreadsheet tasks trigger `project:xlsx`, use structured file tools, and
  produce verifiable artifacts.

## Final Recommendation

Implement Phase 1 as the next bounded unit. This keeps the runtime simple while
closing the real maturity gaps: deterministic identity, prompt re-injection,
checkpoint-safe active state, and executable skill assets.
