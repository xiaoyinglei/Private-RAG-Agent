from __future__ import annotations

import asyncio
from typing import Any

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.tool_execution import ToolBatchRequest
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.tooling import (
    DiscoveryPolicy,
    ToolExecutorLoopAdapter,
    ModelRequestBuilder,
    ProviderCapability,
    ToolCall,
    ToolDomain,
    ToolExecutor,
    ToolDiscoveryState,
    ToolExposure,
    ToolRegistry,
    ToolRisk,
    ToolSpec,
    ToolSurfacePolicy,
    ToolSurfaceRequest,
)
from rag.schema.runtime import AccessPolicy


def _read_spec(name: str = "read_file") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Read a file from the workspace.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        domain=ToolDomain.WORKSPACE,
        risk=ToolRisk.READ,
        timeout_seconds=3.0,
    )


def _write_spec(name: str = "write_file") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Write a file inside the workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        domain=ToolDomain.WORKSPACE,
        risk=ToolRisk.WRITE,
        timeout_seconds=3.0,
    )


def _execute_spec(name: str = "run_command") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Run an allowlisted shell command in the workspace.",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        domain=ToolDomain.EXECUTION,
        risk=ToolRisk.EXECUTE,
        timeout_seconds=3.0,
    )


def _network_spec(name: str = "fetch_url") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Fetch a URL.",
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
        domain=ToolDomain.KNOWLEDGE,
        risk=ToolRisk.NETWORK,
        timeout_seconds=3.0,
    )


def _destructive_spec(name: str = "delete_tree") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Delete a workspace tree.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        domain=ToolDomain.WORKSPACE,
        risk=ToolRisk.DESTRUCTIVE,
        timeout_seconds=3.0,
    )


def _deferred_spec(name: str = "semantic_search") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Search a specialized workspace index.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        domain=ToolDomain.WORKSPACE,
        risk=ToolRisk.READ,
        exposure=ToolExposure.DEFERRED,
        timeout_seconds=3.0,
    )


def _runner(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": f"read {args['path']}", "path": args["path"]}


def _run_config() -> AgentRunConfig:
    return AgentRunConfig(
        run_id="run_tooling_test",
        thread_id="thread_tooling_test",
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )


def test_registry_installation_does_not_make_tools_visible() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)

    decision = ToolSurfacePolicy().decide(registry, ToolSurfaceRequest())

    assert [spec.name for spec in registry.list_specs()] == ["read_file"]
    assert decision.visible_tools == []
    assert decision.sent_schema_names == []
    assert decision.hidden_tools == ["read_file"]


def test_surface_policy_consumes_explicit_request_and_risk_flags() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)
    registry.register(_write_spec(), lambda args: args)
    registry.register(_execute_spec(), lambda args: args)

    decision = ToolSurfacePolicy().decide(
        registry,
        ToolSurfaceRequest(
            requested_tool_names=["read_file", "write_file", "run_command"],
            disabled_tool_names=["read_file"],
        ),
    )

    assert decision.visible_tools == []
    assert decision.sent_schema_names == []
    assert decision.hidden_tools == ["read_file", "run_command", "write_file"]

    allowed = ToolSurfacePolicy().decide(
        registry,
        ToolSurfaceRequest(
            requested_tool_names=["read_file", "write_file", "run_command"],
            allow_write_tools=True,
            allow_execute_tools=True,
        ),
    )

    assert [spec.name for spec in allowed.visible_tools] == [
        "read_file",
        "write_file",
        "run_command",
    ]
    assert allowed.sent_schema_names == ["read_file", "write_file", "run_command"]


def test_surface_policy_uses_provider_capability_to_disable_tool_schemas() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)

    decision = ToolSurfacePolicy().decide(
        registry,
        ToolSurfaceRequest(requested_tool_names=["read_file"]),
        provider_capability=ProviderCapability(supports_tools=False),
    )

    assert decision.visible_tools == []
    assert decision.sent_schema_names == []
    assert decision.hidden_tools == ["read_file"]
    assert decision.tool_choice == "none"


def test_discovery_policy_adds_only_structured_discovered_tools() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)
    registry.register(_deferred_spec(), lambda args: args)

    enriched = DiscoveryPolicy().apply(
        registry,
        ToolSurfaceRequest(
            allow_discovery_tools=True,
            disabled_tool_names=["read_file"],
        ),
        ToolDiscoveryState(
            discovered_tool_names=["semantic_search", "missing_tool", "read_file"],
        ),
    )
    decision = ToolSurfacePolicy().decide(registry, enriched)

    assert enriched.requested_tool_names == ["semantic_search"]
    assert decision.sent_schema_names == ["semantic_search"]


def test_discovery_policy_ignores_discovered_tools_without_discovery_gate() -> None:
    registry = ToolRegistry()
    registry.register(_deferred_spec(), lambda args: args)

    enriched = DiscoveryPolicy().apply(
        registry,
        ToolSurfaceRequest(),
        ToolDiscoveryState(discovered_tool_names=["semantic_search"]),
    )
    decision = ToolSurfacePolicy().decide(registry, enriched)

    assert enriched.requested_tool_names == []
    assert decision.sent_schema_names == []


def test_request_builder_emits_empty_schema_without_task_classification() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)
    decision = ToolSurfacePolicy().decide(
        registry,
        ToolSurfaceRequest(force_empty=True),
    )

    requests = [
        ModelRequestBuilder(provider="openai", model="gpt-test").build(
            messages=[{"role": "user", "content": prompt}],
            surface=decision,
        )
        for prompt in ("2+2等于几", "解释递归")
    ]

    for request in requests:
        assert request.payload["tools"] == []
        assert request.payload["tool_choice"] == "none"
        assert request.trace.visible_tools == []
        assert request.trace.hidden_tools == ["read_file"]
        assert request.trace.schema_bytes == 0
        assert request.sent_schema_names == []


def test_request_builder_emits_openai_compatible_tool_schema() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)
    decision = ToolSurfacePolicy().decide(
        registry,
        ToolSurfaceRequest(requested_tool_names=["read_file"]),
    )

    request = ModelRequestBuilder(provider="groq", model="llama-test").build(
        messages=[{"role": "user", "content": "找到 README"}],
        surface=decision,
    )

    assert request.payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the workspace.",
                "parameters": _read_spec().input_schema,
            },
        }
    ]
    assert request.payload["tool_choice"] == "auto"
    assert request.trace.visible_tools == ["read_file"]
    assert request.trace.schema_bytes > 0
    assert request.sent_schema_names == ["read_file"]


def test_request_builder_omits_tool_choice_when_provider_does_not_support_it() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)
    decision = ToolSurfacePolicy().decide(
        registry,
        ToolSurfaceRequest(requested_tool_names=["read_file"]),
        provider_capability=ProviderCapability(supports_tools=True, supports_tool_choice=False),
    )

    request = ModelRequestBuilder(
        provider="openai-compatible",
        model="custom-model",
        provider_capability=ProviderCapability(
            supports_tools=True,
            supports_tool_choice=False,
        ),
    ).build(
        messages=[{"role": "user", "content": "读取 README"}],
        surface=decision,
    )

    assert "tool_choice" not in request.payload
    assert request.trace.tool_choice is None
    assert request.sent_schema_names == ["read_file"]


def test_executor_returns_recoverable_unknown_and_schema_not_sent_results() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)
    executor = ToolExecutor(registry)

    unknown = asyncio.run(
        executor.execute(
            ToolCall(id="call_1", name="missing_tool", arguments={}),
            sent_schema_names=[],
        )
    )
    hidden = asyncio.run(
        executor.execute(
            ToolCall(id="call_2", name="read_file", arguments={"path": "README.md"}),
            sent_schema_names=[],
        )
    )

    assert unknown.ok is False
    assert unknown.recoverable is True
    assert unknown.error_code == "unknown_tool"
    assert hidden.ok is False
    assert hidden.recoverable is True
    assert hidden.error_code == "schema_not_sent"


def test_executor_checks_unknown_and_schema_not_sent_before_can_use_tool() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)

    def fail_can_use_tool(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("unknown/schema_not_sent must not call canUseTool")

    executor = ToolExecutor(registry, can_use_tool=fail_can_use_tool)

    unknown = asyncio.run(
        executor.execute(
            ToolCall(id="call_1", name="missing_tool", arguments={}),
            sent_schema_names=[],
        )
    )
    hidden = asyncio.run(
        executor.execute(
            ToolCall(id="call_2", name="read_file", arguments={"path": "README.md"}),
            sent_schema_names=[],
        )
    )

    assert unknown.error_code == "unknown_tool"
    assert hidden.error_code == "schema_not_sent"
    assert executor.traces[0].can_use_tool_decision is None
    assert executor.traces[1].can_use_tool_decision is None


def test_executor_validates_args_before_runner_call() -> None:
    registry = ToolRegistry()
    calls: list[dict[str, Any]] = []

    def runner(args: dict[str, Any]) -> dict[str, Any]:
        calls.append(args)
        return args

    registry.register(_read_spec(), runner)
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            ToolCall(id="call_1", name="read_file", arguments={}),
            sent_schema_names=["read_file"],
        )
    )

    assert result.ok is False
    assert result.recoverable is True
    assert result.error_code == "invalid_arguments"
    assert calls == []


def test_executor_runs_visible_tool_and_records_trace() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"}),
            sent_schema_names=["read_file"],
        )
    )

    assert result.ok is True
    assert result.content == "read README.md"
    assert result.data["path"] == "README.md"
    assert result.data["_meta"]["truncated"] is False
    assert result.data["_meta"]["size_bytes"] > 0
    assert executor.traces[-1].tool_name == "read_file"
    assert executor.traces[-1].status == "ok"
    assert executor.traces[-1].can_use_tool_decision == "allow"
    assert executor.traces[-1].truncated is False
    assert executor.traces[-1].output_size_bytes > 0


def test_executor_truncates_large_runner_output_and_records_size() -> None:
    registry = ToolRegistry()
    spec = _read_spec()
    spec = spec.model_copy(update={"output_limit_chars": 20})
    registry.register(spec, lambda args: {"content": "x" * 100, "payload": "y" * 100})
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"}),
            sent_schema_names=["read_file"],
        )
    )

    assert result.ok is True
    assert result.content.endswith("[truncated]")
    assert result.data["_meta"]["truncated"] is True
    assert result.data["_meta"]["size_bytes"] > 20
    assert len(result.data["payload"]) <= len("y" * 20 + "\n[truncated]")
    assert executor.traces[-1].truncated is True
    assert executor.traces[-1].output_size_bytes == result.data["_meta"]["size_bytes"]


def test_executor_runner_exception_is_runner_error_without_traceback() -> None:
    registry = ToolRegistry()

    def broken_runner(args: dict[str, Any]) -> dict[str, Any]:
        del args
        raise RuntimeError("boom")

    registry.register(_read_spec(), broken_runner)
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"}),
            sent_schema_names=["read_file"],
        )
    )

    assert result.ok is False
    assert result.error_code == "runner_error"
    assert result.content == "boom"
    assert "Traceback" not in result.content
    assert result.data["_meta"]["truncated"] is False
    assert executor.traces[-1].status == "error"
    assert executor.traces[-1].error_code == "runner_error"


def test_executor_invalid_arguments_do_not_expose_traceback() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            ToolCall(id="call_1", name="read_file", arguments={}),
            sent_schema_names=["read_file"],
        )
    )

    assert result.ok is False
    assert result.error_code == "invalid_arguments"
    assert "Traceback" not in result.content
    assert result.data["_meta"]["size_bytes"] == len(result.content.encode("utf-8"))
    assert executor.traces[-1].error_code == "invalid_arguments"


def test_executor_asks_for_write_without_entry_allow_flag() -> None:
    registry = ToolRegistry()
    calls: list[dict[str, Any]] = []
    registry.register(_write_spec(), lambda args: calls.append(args) or args)
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="call_1",
                name="write_file",
                arguments={"path": "README.md", "content": "changed"},
            ),
            sent_schema_names=["write_file"],
        )
    )

    assert result.ok is False
    assert result.error_code == "permission_required"
    assert result.data["can_use_tool"]["decision"] == "ask"
    assert calls == []
    assert executor.traces[-1].can_use_tool_decision == "ask"


def test_executor_allows_write_when_entry_allows_write_tools() -> None:
    registry = ToolRegistry()
    calls: list[dict[str, Any]] = []
    registry.register(_write_spec(), lambda args: calls.append(args) or args)
    executor = ToolExecutor(registry, allow_write_tools=True)

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="call_1",
                name="write_file",
                arguments={"path": "README.md", "content": "changed"},
            ),
            sent_schema_names=["write_file"],
        )
    )

    assert result.ok is True
    assert calls == [{"path": "README.md", "content": "changed"}]
    assert executor.traces[-1].can_use_tool_decision == "allow"


def test_executor_denies_network_and_destructive_tools_by_default() -> None:
    registry = ToolRegistry()
    calls: list[dict[str, Any]] = []
    registry.register(_network_spec(), lambda args: calls.append(args) or args)
    registry.register(_destructive_spec(), lambda args: calls.append(args) or args)
    executor = ToolExecutor(registry)

    network = asyncio.run(
        executor.execute(
            ToolCall(id="call_1", name="fetch_url", arguments={"url": "https://example.com"}),
            sent_schema_names=["fetch_url"],
        )
    )
    destructive = asyncio.run(
        executor.execute(
            ToolCall(id="call_2", name="delete_tree", arguments={"path": "."}),
            sent_schema_names=["delete_tree"],
        )
    )

    assert network.error_code == "permission_denied"
    assert network.data["can_use_tool"]["decision"] == "deny"
    assert destructive.error_code == "permission_denied"
    assert destructive.data["can_use_tool"]["decision"] == "deny"
    assert calls == []
    assert [trace.can_use_tool_decision for trace in executor.traces[-2:]] == [
        "deny",
        "deny",
    ]


def test_loop_adapter_executes_through_new_executor_and_legacy_result_shape() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)
    adapter = ToolExecutorLoopAdapter(ToolExecutor(registry))
    call = ToolCallPlan(
        tool_call_id="call_1",
        tool_name="read_file",
        arguments={"path": "README.md"},
    )

    result = asyncio.run(
        adapter.execute_batch(
            ToolBatchRequest(
                calls=(call,),
                run_config=_run_config(),
                allowed_tools=frozenset(),
            ),
            state={"tooling_sent_schema_names": ["read_file"]},
            definition=None,
        )
    )

    assert result.status == "completed"
    legacy = result.tool_results[0]
    assert legacy.status == "ok"
    assert legacy.output.content == "read README.md"
    assert legacy.output.data["path"] == "README.md"


def test_loop_adapter_maps_schema_not_sent_to_recoverable_legacy_error() -> None:
    registry = ToolRegistry()
    registry.register(_read_spec(), _runner)
    adapter = ToolExecutorLoopAdapter(ToolExecutor(registry))
    call = ToolCallPlan(
        tool_call_id="call_1",
        tool_name="read_file",
        arguments={"path": "README.md"},
    )

    result = asyncio.run(
        adapter.execute_batch(
            ToolBatchRequest(
                calls=(call,),
                run_config=_run_config(),
                allowed_tools=frozenset(),
            ),
            state={"tooling_sent_schema_names": []},
            definition=None,
        )
    )

    legacy = result.tool_results[0]
    assert legacy.status == "error"
    assert legacy.error.code == "schema_not_sent"
    assert legacy.error.retryable is True
