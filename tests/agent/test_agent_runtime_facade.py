from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_runtime import Agent, AgentResult, AgentUsage
from agent_runtime.knowledge_providers.rag import LazyRAGKnowledgeProvider
from rag.agent import cli as agent_cli
from rag.agent.cli import agent_app
from rag.agent.service import AgentRunResult
from rag.agent.tools.rag_semantic_tools import AssetSearchInput
from rag.schema.core import AssetRecord


def test_agent_runtime_exports_sdk_facade() -> None:
    assert Agent is not None
    assert AgentResult is not None
    assert AgentUsage is not None


def test_agent_facade_run_maps_public_request_to_internal_service(
    monkeypatch,
) -> None:
    built: list[dict[str, Any]] = []
    requests: list[Any] = []

    def fail_rag_runtime(**_: object) -> object:
        raise AssertionError("Agent() without knowledge must not initialize RAG")

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            requests.append(request)
            return AgentRunResult(
                run_id=request.run_id,
                thread_id=request.thread_id,
                status="done",
                final_answer="facade answer",
            )

    def build_service(runtime: object, **kwargs: object) -> _Service:
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(agent_cli, "_build_optional_rag_runtime", fail_rag_runtime)
    monkeypatch.setattr(agent_cli, "_build_agent_service", build_service)

    result = Agent(model="qwen3_14b_4bit").run(
        "summarize",
        files=["README.md"],
        run_id="sdk-run",
        max_tokens_total=1234,
    )

    assert result.answer == "facade answer"
    assert result.status == "done"
    assert result.files == ("README.md",)
    assert built == [
        {
            "runtime": None,
            "checkpoint_db": None,
            "agent_type": "generic",
            "model_alias": "qwen3_14b_4bit",
            "runtime_diagnostics": (),
            "knowledge_runner": None,
            "knowledge_asset_runner": None,
        }
    ]
    assert len(requests) == 1
    request = requests[0]
    assert request.task == "summarize"
    assert request.run_id == "sdk-run"
    assert request.thread_id == "sdk-run"
    assert request.llm_budget_total == 1234
    assert request.input_files == ["README.md"]


def test_agent_facade_registers_knowledge_runner_lazily(monkeypatch) -> None:
    built: list[dict[str, Any]] = []

    def fail_rag_runtime(**_: object) -> object:
        raise AssertionError("Knowledge provider must initialize RAG only when the tool is called")

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            return AgentRunResult(
                run_id=request.run_id,
                thread_id=request.thread_id,
                status="done",
                final_answer="knowledge runner registered",
            )

    def build_service(runtime: object, **kwargs: object) -> _Service:
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(agent_cli, "_build_optional_rag_runtime", fail_rag_runtime)
    monkeypatch.setattr(agent_cli, "_build_agent_service", build_service)

    result = Agent(model="qwen3_14b_4bit", knowledge=["company_docs"]).run(
        "lookup policy",
        run_id="knowledge-run",
    )

    assert result.answer == "knowledge runner registered"
    assert built[0]["runtime"] is None
    assert built[0]["knowledge_runner"] is not None
    assert built[0]["knowledge_asset_runner"] is not None


@pytest.mark.anyio
async def test_lazy_knowledge_provider_search_assets_uses_typed_asset_inputs(monkeypatch) -> None:
    class _MetadataRepo:
        def list_assets(
            self,
            *,
            doc_id: int | None = None,
            source_id: int | None = None,
            section_id: int | None = None,
        ) -> list[AssetRecord]:
            del doc_id, source_id, section_id
            return [
                AssetRecord(
                    asset_id=7,
                    doc_id=11,
                    source_id=13,
                    asset_type="chart",
                    page_no=1,
                    caption="Revenue chart",
                    content_hash="hash",
                    storage_key="chart.png",
                )
            ]

        def get_asset(self, asset_id: int) -> AssetRecord | None:
            assert asset_id == 7
            return self.list_assets()[0]

    metadata_repo = _MetadataRepo()
    runtime = type(
        "Runtime",
        (),
        {
            "stores": type(
                "Stores",
                (),
                {
                    "metadata_repo": metadata_repo,
                    "object_store": object(),
                },
            )()
        },
    )()

    def build_runtime(**_: object) -> tuple[object, tuple[object, ...]]:
        return runtime, ()

    monkeypatch.setattr(agent_cli, "_build_optional_rag_runtime", build_runtime)

    provider = LazyRAGKnowledgeProvider()
    result = await provider.search_assets(
        AssetSearchInput(query="revenue chart", asset_type="chart", max_results=1),
        execution_context=object(),
    )

    assert result.total_found == 1
    assert result.assets[0].asset_id == 7
    assert result.assets[0].caption == "Revenue chart"


def test_agent_run_cli_delegates_to_agent_facade(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _Facade:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("init", kwargs))

        def run(self, task: str, **kwargs: object) -> AgentResult:
            calls.append(("run", {"task": task, **kwargs}))
            return AgentResult(
                answer="cli facade answer",
                status="done",
                files=tuple(kwargs.get("files") or ()),
                tool_calls=(),
                citations=(),
                usage=AgentUsage(),
                diagnostics=(),
                run_id=str(kwargs["run_id"]),
                thread_id=str(kwargs["run_id"]),
                raw=None,
            )

    def fail_rag_runtime(**_: object) -> object:
        raise AssertionError("CLI run without --knowledge must not initialize RAG")

    monkeypatch.setattr(agent_cli, "Agent", _Facade)
    monkeypatch.setattr(agent_cli, "_build_optional_rag_runtime", fail_rag_runtime)

    result = CliRunner().invoke(
        agent_app,
        [
            "run",
            "hello",
            "--model",
            "qwen3_14b_4bit",
            "--file",
            str(Path("README.md")),
            "--run-id",
            "cli-run",
        ],
        env={"COLUMNS": "240"},
    )

    assert result.exit_code == 0, result.output
    assert "cli facade answer" in result.output
    assert calls == [
        (
            "init",
            {
                "model": "qwen3_14b_4bit",
                "agent_type": "generic",
                "checkpoint_db": None,
                "knowledge": (),
                "rag_storage_root": Path(".rag"),
                "embedding_model": None,
                "reranker_model": None,
                "vector_backend": "milvus",
                "vector_dsn": None,
                "vector_namespace": None,
                "vector_collection_prefix": None,
            },
        ),
        (
            "run",
            {
                "task": "hello",
                "files": ["README.md"],
                "run_id": "cli-run",
                "max_tokens_total": None,
            },
        ),
    ]


def test_agent_run_help_matches_public_api_surface() -> None:
    result = CliRunner().invoke(agent_app, ["run", "--help"], env={"COLUMNS": "240"})

    assert result.exit_code == 0
    output = result.output
    assert "--model" in output
    assert "--file" in output
    assert "--knowledge" in output
    assert "--input-file" in output
    assert "--budget" not in output
    assert "--embedding-model" not in output
    assert "--reranker-model" not in output
    assert "--storage-root" not in output
