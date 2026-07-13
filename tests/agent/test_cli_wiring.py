from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.runtime.builder import (
    CLI_AGENT_CHOICES,
    build_agent_service,
    build_optional_rag_runtime,
    resolve_auto_rag_config,
    resolve_cli_agent_definition,
)
from rag.agent.builtin import create_builtin_agent_registry
from rag.agent.cli import _build_agent_service
from rag.agent.service import AgentRunRequest
from rag.agent.tools.builtins import RESIDENT_CODING_TOOL_NAMES
from rag.agent.tools.integrations.knowledge import KnowledgeSearchOutput
from rag.agent.tools.integrations.mcp import (
    MCPToolDescriptor,
    create_mcp_tools,
)


class _ModelRegistry:
    default_model = "fake"

    def resolve_for_node(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError("model resolution is not needed for assembly")


def test_cli_supports_only_the_product_generic_agent() -> None:
    registry = create_builtin_agent_registry()

    assert CLI_AGENT_CHOICES == ("generic",)
    assert resolve_cli_agent_definition(registry, "generic").agent_type == (
        "generic"
    )
    with pytest.raises(ValueError, match="supported CLI agent"):
        resolve_cli_agent_definition(registry, "research")


def test_builder_assembles_default_six_tools_in_product_order() -> None:
    service = build_agent_service(
        None,
        model_control_plane=_ModelRegistry(),  # type: ignore[arg-type]
    )

    assert tuple(service._tool_snapshot) == RESIDENT_CODING_TOOL_NAMES
    assert service._tool_executor._tools is service._tool_snapshot
    state = service.initial_state(AgentRunRequest(task="Inspect repository."))
    assert tuple(state["resident_tool_names"]) == RESIDENT_CODING_TOOL_NAMES


@pytest.mark.anyio
async def test_configured_knowledge_is_a_resident_extension() -> None:
    async def search(_payload: object, **_kwargs: object) -> object:
        return KnowledgeSearchOutput(
            answer_text="configured knowledge",
            total_found=0,
        )

    service = build_agent_service(
        None,
        model_control_plane=_ModelRegistry(),  # type: ignore[arg-type]
        knowledge_runner=search,  # type: ignore[arg-type]
    )
    state = service.initial_state(AgentRunRequest(task="Search docs."))

    assert tuple(service._tool_snapshot) == (
        *RESIDENT_CODING_TOOL_NAMES,
        "search_knowledge",
    )
    assert state["resident_tool_names"] == [
        *RESIDENT_CODING_TOOL_NAMES,
        "search_knowledge",
    ]


def test_cli_wrapper_uses_the_same_builder_snapshot() -> None:
    service = _build_agent_service(
        None,
        model_control_plane=_ModelRegistry(),  # type: ignore[arg-type]
    )

    assert tuple(service._tool_snapshot) == RESIDENT_CODING_TOOL_NAMES


def test_builder_installs_hidden_factory_outputs_with_find_tools() -> None:
    mcp_tools = create_mcp_tools(
        (
            MCPToolDescriptor(
                server_name="docs",
                tool_name="search",
                description="Search external documentation.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                read_only_hint=True,
            ),
        ),
        lambda _server, _tool, _arguments: {"text": "found"},
    )

    service = build_agent_service(
        None,
        model_control_plane=_ModelRegistry(),  # type: ignore[arg-type]
        mcp_tools=mcp_tools,
    )

    assert tuple(service._tool_snapshot) == (
        *RESIDENT_CODING_TOOL_NAMES,
        "mcp__docs__search",
        "find_tools",
    )
    default_state = service.initial_state(AgentRunRequest(task="Inspect docs."))
    discovery_state = service.initial_state(
        AgentRunRequest(
            task="Inspect docs.",
            allow_discovery_tools=True,
        )
    )
    assert tuple(default_state["resident_tool_names"]) == (
        RESIDENT_CODING_TOOL_NAMES
    )
    assert tuple(discovery_state["resident_tool_names"]) == (
        *RESIDENT_CODING_TOOL_NAMES,
        "find_tools",
    )
    assert discovery_state["active_tool_names"] == []


def test_auto_rag_config_prefers_explicit_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_VECTOR_BACKEND", "sqlite")
    resolved = resolve_auto_rag_config(
        storage_root=Path("custom-rag"),
        vector_backend="milvus",
        vector_dsn="explicit-dsn",
        vector_namespace="explicit-ns",
        vector_collection_prefix="explicit-prefix",
    )

    assert resolved.storage_root == Path("custom-rag")
    assert resolved.vector_backend == "sqlite"
    assert resolved.vector_dsn == "explicit-dsn"
    assert resolved.explicit is True


def test_optional_rag_runtime_is_lazy_when_not_explicit() -> None:
    runtime, diagnostics = build_optional_rag_runtime(
        storage_root=Path(".rag"),
        model_alias=None,
        embedding_model_alias=None,
        reranker_model_alias=None,
        vector_backend="milvus",
        vector_dsn=None,
        vector_namespace=None,
        vector_collection_prefix=None,
        explicit=False,
    )

    assert runtime is None
    assert diagnostics == ()
