import json
from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

import rag.cli as cli
from rag import StorageConfig
from rag.cli import app
from rag.retrieval.models import BuiltContext, PublicQueryResult
from rag.schema.query import GroundedAnswer
from rag.schema.runtime import RetrievalDiagnostics
from tests.support import make_runtime

runner = CliRunner()


def _use_isolated_cli_runtime(monkeypatch: MonkeyPatch) -> None:
    def _runtime(
        storage_root: Path,
        *,
        require_chat: bool = False,
        model: str | None = None,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
    ):
        del model, embedding_model, reranker_model
        return make_runtime(storage=StorageConfig(root=storage_root), require_chat=require_chat)

    monkeypatch.setattr(cli, "_runtime", _runtime)


def test_cli_ingest_query_round_trip(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _use_isolated_cli_runtime(monkeypatch)
    storage_root = tmp_path / ".rag"

    ingest = runner.invoke(
        app,
        [
            "ingest",
            "--storage-root",
            str(storage_root),
            "--source-type",
            "plain_text",
            "--location",
            "memory://note-1",
            "--content",
            "Alpha Engine handles ingestion. Beta Service depends on Alpha Engine.",
        ],
    )

    query = runner.invoke(
        app,
        [
            "query",
            "--storage-root",
            str(storage_root),
            "--query",
            "What does Alpha Engine handle?",
            "--json",
        ],
    )
    payload = json.loads(query.stdout)

    assert ingest.exit_code == 0
    assert query.exit_code == 0
    assert payload["answer"]["answer_text"]
    assert payload["context"]["evidence"]


def test_cli_delete_and_rebuild_use_new_runtime_contract(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _use_isolated_cli_runtime(monkeypatch)
    storage_root = tmp_path / ".rag"

    ingest = runner.invoke(
        app,
        [
            "ingest",
            "--storage-root",
            str(storage_root),
            "--source-type",
            "plain_text",
            "--location",
            "memory://note-1",
            "--content",
            "Alpha Engine handles ingestion.",
        ],
    )
    delete = runner.invoke(
        app,
        [
            "delete",
            "--storage-root",
            str(storage_root),
            "--location",
            "memory://note-1",
        ],
    )
    rebuild = runner.invoke(
        app,
        [
            "rebuild",
            "--storage-root",
            str(storage_root),
            "--location",
            "memory://note-1",
        ],
    )

    assert ingest.exit_code == 0
    assert delete.exit_code == 0
    delete_payload = json.loads(delete.stdout)
    assert delete_payload["deleted_doc_ids"]
    assert delete_payload["deleted_source_ids"]
    assert delete_payload["deleted_vector_count"] >= 2
    assert rebuild.exit_code == 0
    rebuild_payload = json.loads(rebuild.stdout)
    assert rebuild_payload["rebuilt_doc_ids"] == delete_payload["deleted_doc_ids"]
    assert rebuild_payload["results"][0]["indexed_object_count"] >= 2


def test_cli_rejects_missing_source_payload_for_ingest(tmp_path: Path) -> None:
    storage_root = tmp_path / ".rag"

    result = runner.invoke(
        app,
        [
            "ingest",
            "--storage-root",
            str(storage_root),
            "--source-type",
            "plain_text",
            "--location",
            "memory://note-1",
        ],
    )

    assert result.exit_code != 0
    assert "content" in result.stdout.lower() or "content" in result.stderr.lower()


def test_cli_main_delegates_to_typer_app(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeApp:
        def __call__(self) -> None:
            calls.append("called")

    monkeypatch.setattr(cli, "app", FakeApp())

    cli.main()

    assert calls == ["called"]


def test_assembly_profile_cli_surface_is_removed() -> None:
    help_env = {"COLUMNS": "240"}
    root_help = runner.invoke(app, ["--help"], env=help_env)
    query_help = runner.invoke(app, ["query", "--help"], env=help_env)
    agent_run_help = runner.invoke(app, ["agent", "run", "--help"], env=help_env)
    agent_resume_help = runner.invoke(app, ["agent", "resume", "--help"], env=help_env)

    assert root_help.exit_code == 0
    assert query_help.exit_code == 0
    assert agent_run_help.exit_code == 0
    assert agent_resume_help.exit_code == 0

    assert "profiles" not in root_help.output
    assert "--profile" not in query_help.output
    assert "--profile" not in agent_run_help.output
    assert "--profile" not in agent_resume_help.output

    for output in (agent_run_help.output, agent_resume_help.output):
        assert "--model" in output
        assert "--embedding-model" in output
        assert "--reranker-model" in output


def test_cli_analyze_task_is_disabled_on_new_runtime(tmp_path: Path) -> None:
    storage_root = tmp_path / ".rag"

    analyze = runner.invoke(
        app,
        [
            "analyze-task",
            "--storage-root",
            str(storage_root),
            "--query",
            "Summarize Alpha Engine responsibilities.",
            "--json",
        ],
    )

    assert analyze.exit_code == 1
    assert "disabled on the new runtime CLI" in analyze.output


def test_cli_query_uses_public_query_contract(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    class _FakeRuntime:
        def __enter__(self) -> "_FakeRuntime":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def query(self, *_args, **_kwargs):
            raise AssertionError("cli query should not use runtime.query")

        def query_public(self, query_text: str, *, options=None) -> PublicQueryResult:
            calls.append((query_text, getattr(options, "retrieval_profile", None)))
            return PublicQueryResult(
                query=query_text,
                retrieval_profile="auto",
                answer=GroundedAnswer(
                    answer_text="Alpha answer",
                    groundedness_flag=True,
                    insufficient_evidence_flag=False,
                ),
                context=BuiltContext(
                    evidence=[],
                    token_budget=1200,
                    token_count=12,
                    grounded_candidate="alpha",
                    prompt="prompt",
                ),
                routing_decision={},
                retrieval_diagnostics=RetrievalDiagnostics(),
            )

    monkeypatch.setattr(cli, "_runtime", lambda *args, **kwargs: _FakeRuntime())

    result = runner.invoke(
        app,
        [
            "query",
            "--storage-root",
            ".rag",
            "--query",
            "What does Alpha Engine do?",
            "--retrieval-profile",
            "auto",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["answer"]["answer_text"] == "Alpha answer"
    assert "retrieval" not in payload
    assert calls == [("What does Alpha Engine do?", "auto")]


def test_cli_query_help_uses_new_retrieval_profile_option() -> None:
    result = runner.invoke(app, ["query", "--help"], env={"COLUMNS": "240"})

    assert result.exit_code == 0
    assert "--retrieval-profile" in result.output
    assert "--model" in result.output


def test_agent_run_help_exposes_explicit_agent_selector() -> None:
    result = runner.invoke(app, ["agent", "run", "--help"], env={"COLUMNS": "240"})

    assert result.exit_code == 0
    assert "--agent" in result.output
    assert "research" in result.output
    assert "orchestrator" in result.output


def test_agent_resume_help_exposes_agent_selector_for_checkpoint_restore() -> None:
    result = runner.invoke(app, ["agent", "resume", "--help"], env={"COLUMNS": "240"})

    assert result.exit_code == 0
    assert "--agent" in result.output


def test_cli_benchmark_help_defaults_to_new_milvus_backend() -> None:
    help_env = {"COLUMNS": "240"}
    ingest_help = runner.invoke(app, ["benchmark-ingest", "--help"], env=help_env)
    evaluate_help = runner.invoke(app, ["benchmark-evaluate", "--help"], env=help_env)

    assert ingest_help.exit_code == 0
    assert evaluate_help.exit_code == 0
    assert "--retrieval-profile" in evaluate_help.output
    assert "--mode" not in evaluate_help.output
    assert "[default: milvus]" in ingest_help.output
    assert "[default: milvus]" in evaluate_help.output
