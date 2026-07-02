from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pytest

from rag.agent.builtin import create_builtin_agent_registry
from rag.agent.capabilities.tool_search import ToolSearchInput, ToolSearchOutput
from rag.agent.cli import (
    CLI_AGENT_CHOICES,
    _build_agent_service,
    _build_llm_tool_runners,
    _build_optional_rag_runtime,
    _display_result,
    _looks_like_rag_storage,
    _resolve_auto_rag_config,
    _resolve_cli_agent_definition,
)
from rag.agent.core.context import AgentRunConfig, RunRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.loop.state import LoopState as AgentState
from rag.agent.service import AgentRunRequest, AgentRunResult
from rag.agent.tools.llm_tools import LLMCompareInput, LLMGenerateInput
from rag.agent.tools.registry import ToolExecutionContext
from rag.schema.runtime import AccessPolicy, RuntimeMode


class _ChatBinding:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, prompt: str, **_: object) -> str:
        self.prompts.append(prompt)
        return f"response:{prompt}"


class _Runtime:
    retrieval_service = None
    chat_context_window_tokens = 32_768
    llm_stage_budgets = None
    token_accounting = type(
        "WordTokens",
        (),
        {"count": lambda self, text: len(text.split())},
    )()

    def __init__(self) -> None:
        self.capability_bundle = type(
            "CapabilityBundle",
            (),
            {"chat_bindings": [_ChatBinding()]},
        )()


class _RuntimeWithAssetStores(_Runtime):
    def __init__(self) -> None:
        super().__init__()
        self.stores = type(
            "Stores",
            (),
            {
                "metadata_repo": object(),
                "object_store": object(),
            },
        )()


class _RetrievalService:
    def __init__(self) -> None:
        self.access_policies: list[AccessPolicy] = []

    async def aretrieve_payload(
        self,
        query: str,
        *,
        access_policy: AccessPolicy,
        query_options: object,
    ) -> object:
        del query, query_options
        self.access_policies.append(access_policy)
        evidence = type(
            "Evidence",
            (),
            {"internal": [], "external": [], "graph": []},
        )()
        return type("Payload", (), {"evidence": evidence})()


class _RuntimeWithRetrieval(_Runtime):
    def __init__(self) -> None:
        super().__init__()
        self.retrieval_service = _RetrievalService()


def _trusted_llm_execution_context(
    config: AgentRunConfig,
    *,
    tool_name: str,
) -> ToolExecutionContext:
    state = cast(
        AgentState,
        {
            "task": "CLI model tool test",
            "run_config": config,
        },
    )
    definition = AgentRuntimePolicy.test_factory(
        agent_type="test",
        description="CLI model tool test",
        system_prompt="Use only trusted supplied context.",
        allowed_tools=[tool_name],
    )
    return ToolExecutionContext(
        run_config=config,
        state=state,
        definition=definition,
    )


@pytest.mark.anyio
async def test_cli_llm_runner_wiring_includes_compare_runner() -> None:
    chat = _ChatBinding()
    runners = _build_llm_tool_runners(chat)
    config = AgentRunConfig(
        run_id="cli-compare",
        thread_id="cli-compare",
        llm_budget_total=10_000,
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )
    RunRegistry.remove(config.run_id)
    RunRegistry.get_or_create(config)

    assert {"llm_generate", "llm_summarize", "llm_compare"} <= set(runners)
    result = await runners["llm_compare"](
        LLMCompareInput(
            question="Compare A and B",
            left_context_sections=["A evidence"],
            right_context_sections=["B evidence"],
        ),
        _trusted_llm_execution_context(config, tool_name="llm_compare"),
    )

    assert result.text.startswith("response:")
    assert "[system]\nUse only trusted supplied context." in result.text
    assert "[task]\nCompare A and B" in result.text
    assert "Left context:\nA evidence" in result.text
    assert "Right context:\nB evidence" in result.text
    RunRegistry.remove(config.run_id)


@pytest.mark.anyio
async def test_cli_generate_runner_preserves_supplied_grounding_ids() -> None:
    runners = _build_llm_tool_runners(_ChatBinding())
    config = AgentRunConfig(
        run_id="cli-generate",
        thread_id="cli-generate",
        llm_budget_total=10_000,
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )
    RunRegistry.remove(config.run_id)
    RunRegistry.get_or_create(config)

    result = await runners["llm_generate"](
        LLMGenerateInput(
            prompt="Write grounded answer",
            evidence_ids=["ev1"],
            citation_ids=["cit1"],
        ),
        _trusted_llm_execution_context(config, tool_name="llm_generate"),
    )

    assert result.text.startswith("response:")
    assert "[system]\nUse only trusted supplied context." in result.text
    assert "[task]\nWrite grounded answer" in result.text
    assert result.evidence_ids == ["ev1"]
    assert result.citation_ids == ["cit1"]
    RunRegistry.remove(config.run_id)


def test_cli_agent_choices_expose_top_level_agents_only() -> None:
    assert CLI_AGENT_CHOICES == ("generic",)


def test_resolve_cli_agent_definition_rejects_internal_synthesize() -> None:
    registry = create_builtin_agent_registry()

    with pytest.raises(ValueError, match="not a supported CLI agent"):
        _resolve_cli_agent_definition(registry, "synthesize")


def test_build_agent_service_registers_all_asset_tool_runners() -> None:
    service = _build_agent_service(_RuntimeWithAssetStores(), agent_type="generic")

    assert service._base_tool_registry.has_runner("asset_list")
    assert service._base_tool_registry.has_runner("asset_inspect")
    assert service._base_tool_registry.has_runner("asset_read_slice")
    assert service._base_tool_registry.has_runner("asset_analyze")


def test_build_agent_service_without_rag_runtime_hides_rag_tools() -> None:
    service = _build_agent_service(None, agent_type="generic")
    run_config = AgentRunConfig(
        run_id="pure-agent-cli",
        thread_id="pure-agent-cli",
        max_depth=1,
        access_policy=AccessPolicy.default(),
    )
    runtime_registry = service._runtime_tool_registry(run_config)

    assert "search_knowledge" not in service._policy.allowed_tools
    assert "search_assets" not in service._policy.allowed_tools
    assert service._catalog.get("search_knowledge") is None
    assert service._catalog.get("search_assets") is None
    assert "structured_probe" in service._policy.allowed_tools
    assert "llm_generate" in service._policy.allowed_tools
    assert runtime_registry.has_runner("llm_generate")


def test_build_agent_service_with_retrieval_runtime_exposes_rag_tool() -> None:
    service = _build_agent_service(_RuntimeWithRetrieval(), agent_type="generic")

    assert "search_knowledge" in service._policy.allowed_tools
    assert service._catalog.get("search_knowledge") is not None
    assert service._base_tool_registry.has_runner("search_knowledge")


def test_resolve_auto_rag_config_uses_existing_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STORAGE_ROOT", "/tmp/agent-rag-store")
    monkeypatch.setenv("VECTOR_DSN", "http://127.0.0.1:19530")
    monkeypatch.setenv("VECTOR_PREFIX", "private_docs_v1")

    config = _resolve_auto_rag_config(
        storage_root=Path(".rag"),
        vector_backend="milvus",
        vector_dsn=None,
        vector_namespace=None,
        vector_collection_prefix=None,
    )

    assert config.storage_root == Path("/tmp/agent-rag-store")
    assert config.vector_dsn == "http://127.0.0.1:19530"
    assert config.vector_collection_prefix == "private_docs_v1"
    assert config.explicit is True


def test_default_checkpoint_dir_does_not_auto_attach_rag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "AGENT_RAG_STORAGE_ROOT",
        "RAG_STORAGE_ROOT",
        "STORAGE_ROOT",
        "AGENT_VECTOR_BACKEND",
        "VECTOR_BACKEND",
        "AGENT_VECTOR_DSN",
        "VECTOR_DSN",
        "AGENT_VECTOR_NAMESPACE",
        "VECTOR_NAMESPACE",
        "AGENT_VECTOR_PREFIX",
        "VECTOR_PREFIX",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    storage_root = tmp_path / ".rag"
    storage_root.mkdir()
    (storage_root / "agent_checkpoints.sqlite").touch()

    runtime, diagnostics = _build_optional_rag_runtime(
        storage_root=Path(".rag"),
        model_alias=None,
        embedding_model_alias=None,
        reranker_model_alias=None,
        vector_backend="milvus",
        vector_dsn=None,
        vector_namespace=None,
        vector_collection_prefix=None,
    )

    assert not _looks_like_rag_storage(storage_root)
    assert runtime is None
    assert diagnostics == ()


def test_build_agent_service_honors_cli_model_alias_for_agent_decisions() -> None:
    service = _build_agent_service(
        _Runtime(),
        agent_type="generic",
        model_alias="qwen3_8b_mlx_4bit",
    )

    assert service._model_registry is not None
    assert service._model_registry.default_model == "qwen3_8b_mlx_4bit"


def test_build_agent_service_rejects_explicit_model_registry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_from_env(*args: object, **kwargs: object) -> ModelRegistry:
        del args, kwargs
        raise KeyError("unknown explicit alias")

    monkeypatch.setattr(ModelRegistry, "from_env", fail_from_env)

    with pytest.raises(KeyError, match="unknown explicit alias"):
        _build_agent_service(
            _Runtime(),
            agent_type="generic",
            model_alias="missing",
        )


def test_build_agent_service_records_automatic_model_registry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_from_env(*args: object, **kwargs: object) -> ModelRegistry:
        del args, kwargs
        raise FileNotFoundError("models config missing")

    monkeypatch.setattr(ModelRegistry, "from_env", fail_from_env)

    service = _build_agent_service(_Runtime(), agent_type="generic")
    state = service.initial_state(
        AgentRunRequest(
            task="Explain policy",
            run_id="cli-registry-failure",
            thread_id="cli-registry-failure",
        )
    )

    assert state["runtime_diagnostics"][0].code == "model_registry_initialization_failed"


def test_build_agent_service_records_skill_catalog_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from rag.agent.skills import loader

    def fail_scan(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise PermissionError("skills directory unreadable")

    monkeypatch.setattr(loader, "scan_and_load_skills", fail_scan)

    with caplog.at_level(logging.WARNING, logger="rag.agent.cli"):
        service = _build_agent_service(_Runtime(), agent_type="generic")

    state = service.initial_state(
        AgentRunRequest(
            task="Explain policy",
            run_id="cli-skill-catalog-failure",
            thread_id="cli-skill-catalog-failure",
        )
    )

    assert "Skill catalog loading failed" in caplog.text
    assert any(
        diagnostic.code == "skill_catalog_load_failed"
        for diagnostic in state["runtime_diagnostics"]
    )


@pytest.mark.parametrize("verbose", [False, True])
def test_display_result_surfaces_runtime_degradation(
    capsys: pytest.CaptureFixture[str],
    *,
    verbose: bool,
) -> None:
    result = AgentRunResult(
        run_id="display-diagnostics",
        thread_id="display-diagnostics",
        status="paused",
        runtime_diagnostics=[
            RuntimeDiagnostic(
                code="default_providers_initialization_failed",
                component="model_providers",
                message="decision model unavailable",
                error_type="RuntimeError",
            )
        ],
    )

    _display_result(result, verbose=verbose)

    output = capsys.readouterr().out
    assert "降级模式" in output
    if verbose:
        assert "model_providers" in output
        assert "decision model unavailable" in output


def test_display_result_surfaces_model_failure_without_verbose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = AgentRunResult(
        run_id="display-model-failure",
        thread_id="display-model-failure",
        status="failed",
        stop_reason="model_provider_failed",
        runtime_diagnostics=[
            RuntimeDiagnostic(
                code="model_provider_failed",
                component="agent_loop",
                message="Error code: 502, InternalServerError",
                severity="error",
            )
        ],
    )

    _display_result(result, verbose=False)

    output = capsys.readouterr().out
    assert "模型调用失败" in output
    assert "model_provider_failed" in output
    assert "Error code: 502" in output
    assert "降级模式" not in output


@pytest.mark.anyio
async def test_build_agent_service_registers_rag_runner_with_execution_context() -> None:
    runtime = _RuntimeWithRetrieval()
    service = _build_agent_service(runtime, agent_type="generic")
    access_policy = AccessPolicy(
        allowed_runtimes=frozenset({RuntimeMode.FAST})
    )
    run_config = AgentRunConfig(
        run_id="cli-rag-context",
        thread_id="cli-rag-context",
        llm_budget_total=100,
        max_depth=1,
        access_policy=access_policy,
    )

    await service._base_tool_registry.run(
        "vector_search",
        {"query": "test"},
        execution_context=ToolExecutionContext(run_config=run_config),
    )

    assert runtime.retrieval_service.access_policies == [access_policy]


def test_build_agent_service_rejects_unknown_agent() -> None:
    with pytest.raises(ValueError, match="not a supported CLI agent"):
        _build_agent_service(_Runtime(), agent_type="unknown")


def test_validate_workspace_core_runners_detects_missing_runner() -> None:
    """search_text/apply_patch/run_command must have runners after workspace setup."""
    from rag.agent.service import AgentService
    from rag.agent.tools.registry import ToolRegistry
    from rag.agent.tools.spec import ExecutionCategory, ToolPermissions

    # Registry without any runners — should fail
    registry = ToolRegistry()
    # Register a bare spec without runner (simulating a bug where workspace tool creation skips search_text)
    from rag.agent.tools.spec import ToolSpec
    registry.register(ToolSpec(
        name="search_text",
        description="Search files",
        input_model=ToolSearchInput,
        output_model=ToolSearchOutput,
        error_model=ToolSearchOutput,
        permissions=ToolPermissions(),
        execution_category=ExecutionCategory.READ,
        timeout_seconds=5,
    ))

    with pytest.raises(RuntimeError, match="search_text"):
        AgentService._validate_workspace_core_runners(registry)


def test_validate_workspace_core_runners_passes_when_runners_present() -> None:
    """No error when all workspace core tools have runners."""
    from rag.agent.service import AgentService
    from rag.agent.tools.registry import ToolRegistry
    from rag.agent.tools.spec import ExecutionCategory, ToolPermissions, ToolSpec

    registry = ToolRegistry()
    for name in ("search_text", "apply_patch", "run_command", "list_files", "read_file", "write_file", "run_python"):
        registry.register(ToolSpec(
            name=name,
            description=f"Tool: {name}",
            input_model=ToolSearchInput,
            output_model=ToolSearchOutput,
            error_model=ToolSearchOutput,
            permissions=ToolPermissions(),
            execution_category=ExecutionCategory.READ,
            timeout_seconds=5,
        ))
        registry.register_runner(name, lambda **kw: None)

    # Should not raise
    AgentService._validate_workspace_core_runners(registry)
