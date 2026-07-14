# Packaged Models and Qwen3.5 File Context Design

## Goal

Make the installed `agent-runtime` wheel load its bundled model catalog from any
working directory, and make OpenAI-compatible file tasks work with
`mlx-community/Qwen3.5-9B-4bit` without emitting a second `system` message.

## Scope

- Keep `configs/models.yaml` as the only authored model catalog.
- Add `qwen3_5_9b_mlx_4bit` using the provider model ID
  `mlx-community/Qwen3.5-9B-4bit` and its 262144-token context declaration.
- Package that catalog inside the wheel without creating a top-level `configs`
  Python package or a second maintained YAML file.
- Preserve environment overrides in their current order:
  `RAG_AGENT_MODELS_PATH`, then `RAG_AGENT_MODELS`, then bundled configuration.
- Preserve provider-neutral `ModelMessage(role="context")` values in canonical
  requests and checkpoints.
- Change only OpenAI-compatible wire serialization so it emits at most one
  `system` message, at index zero.

## Design

### Packaged model catalog

Hatch force-includes the repository catalog at
`rag/agent/_data/models.yaml` inside the wheel. `ModelRegistry` first checks the
source path derived from `llm_registry.py`'s location, never the process CWD. If
it is absent, as it is in a built wheel, it opens the packaged resource with
`importlib.resources.as_file()` and reuses the existing YAML loader. This keeps
one authored source and one parsing path and prevents an unrelated working
directory from shadowing the bundled catalog.

### OpenAI-compatible context serialization

The serializer processes the complete canonical message sequence rather than
mapping each message independently:

1. Contiguous leading `system` and `context` messages are joined into one
   `system` payload.
2. The first non-leading message starts the conversation.
3. Any later `context` message is serialized as a `user` event at its original
   position.
4. Any later canonical `system` message fails loudly because its position is an
   invalid provider-neutral contract rather than silently weakening its role.
5. Assistant/tool serialization and canonical request/checkpoint data remain
   unchanged.

This satisfies Qwen3.5/MLX's system-message constraint without moving dynamic
events into the stable prefix or adding provider rules to the loop. Because the
wire semantics change, the OpenAI-compatible serializer revision advances from
`openai-compatible-chat-v1` to `openai-compatible-chat-v2`.

## Failure behavior

- Missing environment overrides, source catalog, and packaged catalog still
  fail loudly with `FileNotFoundError`.
- Invalid YAML still follows the existing validation path.
- Non-context OpenAI-compatible message roles keep their existing validation.

## Verification

- A wheel built from the repository is imported directly from an unrelated
  directory and resolves `qwen3_5_9b_mlx_4bit` without model-config environment
  variables.
- OpenAI wire regression tests assert exactly one leading `system` message for
  both frozen file context and later context events, and rejection of a later
  canonical `system` message.
- Existing canonical-request, provider, registry, delivery-smoke, import-boundary,
  and full test suites remain green.
- A live Qwen3.5 delivery smoke performs `search_text`, then `read_file`, then
  returns the expected workspace-relative path.

## Non-goals

- Changing the default model away from Groq.
- Redesigning model routing, checkpoint schemas, or local runtime launch policy.
- Fixing the separate real-model repeated-tool-error recovery behavior.
