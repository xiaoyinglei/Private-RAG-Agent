# Packaged Models and Qwen3.5 File Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the built wheel resolve its bundled model catalog outside the repository and make Qwen3.5 file tasks serialize with one leading system message.

**Architecture:** Keep `configs/models.yaml` as the single authored catalog, force-include it as a `rag.agent` wheel resource, and let `ModelRegistry` use the repository-relative source file or packaged resource through its existing YAML loader. Normalize only the OpenAI-compatible wire message sequence: fold leading system/context into one system, preserve later context as user events, and reject later canonical system messages.

**Tech Stack:** Python 3.12, Hatchling, importlib.resources, Pydantic, pytest, MLX-LM, OpenAI-compatible chat wire.

---

### Task 1: Prove the wheel distribution failure

**Files:**
- Create: `tests/agent/test_package_distribution.py`
- Modify: `tests/agent/test_llm_registry.py`

- [ ] **Step 1: Write the failing wheel test**

Build a wheel into `tmp_path`, run `sys.executable` from an unrelated directory
with `PYTHONPATH` replaced by the wheel path, clear both model-config environment
variables, and assert that this code exits successfully:

```python
import agent_runtime
import rag.agent.core.llm_registry as registry_module
from agent_runtime import Agent

assert ".whl" in agent_runtime.__file__
assert ".whl" in registry_module.__file__
spec = Agent(model="qwen3_5_9b_mlx_4bit").current_model()
assert spec.provider_model == "mlx-community/Qwen3.5-9B-4bit"
```

Build with this exact subprocess:

```python
subprocess.run(
    ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
)
wheel, = dist_dir.glob("*.whl")
```

Set `env["PYTHONPATH"] = str(wheel)` rather than retaining or appending the
inherited value. The dual module-origin assertion prevents the editable
checkout from satisfying the registry import accidentally.

- [ ] **Step 2: Add a catalog test for the Qwen3.5 declaration**

Load the repository catalog and assert provider, model ID, local location,
262144-token context, health URL, and expected-model marker.

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
uv run pytest tests/agent/test_package_distribution.py tests/agent/test_llm_registry.py -q
```

Expected: the wheel subprocess fails because the wheel has no bundled model
catalog, and the catalog assertion fails because the Qwen3.5 alias is absent.

### Task 2: Package and resolve the single model catalog

**Files:**
- Modify: `pyproject.toml`
- Modify: `configs/models.yaml`
- Modify: `rag/agent/core/llm_registry.py`
- Test: `tests/agent/test_package_distribution.py`
- Test: `tests/agent/test_llm_registry.py`

- [ ] **Step 1: Force-include the catalog in the wheel**

Add:

```toml
[tool.hatch.build.targets.wheel.force-include]
"configs/models.yaml" = "rag/agent/_data/models.yaml"
```

- [ ] **Step 2: Add the local Qwen3.5 alias**

Add `qwen3_5_9b_mlx_4bit` with model
`mlx-community/Qwen3.5-9B-4bit`, context window `262144`, the existing local
MLX provider, and `expected_model_contains: Qwen3.5-9B-4bit`.

- [ ] **Step 3: Resolve source or packaged config without CWD dependence**

Set the source path from `Path(__file__).resolve().parents[3]`, then use:

```python
resource = files("rag.agent").joinpath("_data").joinpath("models.yaml")
with as_file(resource) as resource_path:
    return cls._load_yaml_file(resource_path)
```

Keep environment path and JSON overrides ahead of both bundled paths and retain
fail-loud behavior when no source or packaged resource exists.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/agent/test_package_distribution.py tests/agent/test_llm_registry.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit the packaging slice**

```bash
git add pyproject.toml configs/models.yaml rag/agent/core/llm_registry.py \
  tests/agent/test_package_distribution.py tests/agent/test_llm_registry.py
git commit -m "fix(agent): package Qwen3.5 model catalog"
```

### Task 3: Prove the Qwen-compatible message invariant

**Files:**
- Modify: `tests/provider/test_openai_wire.py`
- Modify: `tests/provider/test_llm_gateway.py`

- [ ] **Step 1: Change the existing frozen-context expectation**

Assert `_request()` serializes roles as `system, user, assistant`, that only
one system exists at index zero, and that its content contains both the system
instruction and frozen context event.

- [ ] **Step 2: Add later-context and invalid-system tests**

Append an assistant message and a canonical context event, then assert the event
serializes as a user message in its original position with no later system.
Append a later canonical system message in a separate request and assert
serialization raises `ValueError` mentioning a non-leading system message.

- [ ] **Step 3: Expect the new serializer revision**

Update the live OpenAI-compatible gateway assertion from
`openai-compatible-chat-v1` to `openai-compatible-chat-v2`.

- [ ] **Step 4: Run the tests and verify RED**

Run:

```bash
uv run pytest tests/provider/test_openai_wire.py tests/provider/test_llm_gateway.py -q
```

Expected: role/invariant tests fail because context still becomes a second
system and the revision is still v1.

### Task 4: Normalize OpenAI-compatible messages once

**Files:**
- Modify: `rag/providers/openai_wire.py`
- Test: `tests/provider/test_openai_wire.py`
- Test: `tests/provider/test_llm_gateway.py`

- [ ] **Step 1: Add one sequence serializer**

Replace the per-message comprehension with a helper that collects contiguous
leading system/context content, emits one system payload, delegates normal
user/assistant/tool messages to the existing payload logic, maps later context
to user, and rejects later system.

- [ ] **Step 2: Advance the serializer revision**

Set `OPENAI_WIRE_REVISION = "openai-compatible-chat-v2"`.

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/provider/test_openai_wire.py tests/provider/test_llm_gateway.py -q
```

Expected: all pass.

- [ ] **Step 4: Commit the wire slice**

```bash
git add rag/providers/openai_wire.py tests/provider/test_openai_wire.py \
  tests/provider/test_llm_gateway.py
git commit -m "fix(agent): serialize one leading system message"
```

### Task 5: Verify distribution, regressions, and live Qwen3.5

**Files:**
- No additional production files expected.

- [ ] **Step 1: Run focused Agent/provider suites**

```bash
uv run pytest tests/agent/test_package_distribution.py \
  tests/agent/test_llm_registry.py tests/agent/test_canonical_model_request.py \
  tests/agent/test_loop_model_context.py tests/provider/test_openai_wire.py \
  tests/provider/test_llm_gateway.py -q
```

- [ ] **Step 2: Run static and architecture checks**

```bash
uv run ruff check agent_runtime rag/agent rag/providers/openai_wire.py \
  tests/agent/test_package_distribution.py tests/agent/test_llm_registry.py \
  tests/provider/test_openai_wire.py tests/provider/test_llm_gateway.py
uv run lint-imports
git diff --check
```

- [ ] **Step 3: Run the complete test suite**

```bash
uv run pytest -q
```

- [ ] **Step 4: Run the installed-wheel probe manually**

Build to a temporary directory, import the wheel from an unrelated CWD with
both model-config environment variables unset, and print the Qwen3.5 public
model spec.

- [ ] **Step 5: Run the live Qwen3.5 file smoke**

Start the local MLX-LM server with
`mlx-community/Qwen3.5-9B-4bit` using an owned background PID, wait for the
models endpoint, run the smoke, and always stop the server:

```bash
log_file="$(mktemp)"
uv run mlx_lm.server \
  --model mlx-community/Qwen3.5-9B-4bit \
  --host 127.0.0.1 --port 8080 \
  --chat-template-args '{"enable_thinking": false}' \
  >"$log_file" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; rm -f "$log_file"' EXIT

ready=0
for _attempt in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:8080/v1/models \
      | rg -q 'Qwen3.5-9B-4bit'; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    sed -n '1,200p' "$log_file"
    exit 1
  fi
  sleep 1
done
test "$ready" -eq 1

uv run python scripts/agent_delivery_smoke.py \
  --model qwen3_5_9b_mlx_4bit \
  --case find_agent_service --verbose
```

Expected: `PASS`, tools exactly `search_text,read_file`, answer exactly
`input_files/service.py`. Stop the temporary server afterward.

- [ ] **Step 6: Review the final diff and repository state**

```bash
git status --short --branch
git diff main...HEAD --check
git diff main...HEAD --stat
```
