from __future__ import annotations

from pathlib import Path

import pytest

from rag.agent.memory.models import (
    ExternalizedToolOutput,
    MemoryRecord,
    MemoryRef,
)
from rag.agent.memory.store import MemoryRefError, WorkspaceMemoryStore
from rag.agent.primitive_ops import FileInfo, ListFilesOutput
from rag.agent.workspace import WorkspaceRuntime


def _workspace(tmp_path: Path) -> WorkspaceRuntime:
    workspace = WorkspaceRuntime(root=tmp_path / "workspace", is_temporary=True)
    workspace.initialize()
    return workspace


def test_memory_models_serialize_schema_versions() -> None:
    ref = MemoryRef(
        ref_id="mem_schema",
        path=".agent_memory/records/mem_schema.json",
        summary="listed files",
    )
    externalized = ExternalizedToolOutput(
        original_output_model="rag.agent.primitive_ops.ListFilesOutput",
        summary="listed files",
        ref=ref,
    )

    dumped = externalized.model_dump(mode="json")

    assert dumped["schema_version"] == 1
    assert dumped["ref"]["schema_version"] == 1
    assert dumped["original_output_model"] == "rag.agent.primitive_ops.ListFilesOutput"


def test_store_writes_reads_and_tombstones_records(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = WorkspaceMemoryStore(workspace=workspace)
    output = ListFilesOutput(
        files=[
            FileInfo(
                name="sales.csv",
                path="input_files/sales.csv",
                size=123,
                is_dir=False,
                modified_at=1.0,
                capabilities=["read_file", "structured_probe"],
            )
        ]
    )

    ref = store.write_tool_output(
        output,
        summary="list_files files=1 first=input_files/sales.csv",
        source_tool_call_id="tc-list",
        source_tool_name="list_files",
    )
    record = store.resolve(ref)

    assert isinstance(record, MemoryRecord)
    assert record.schema_version == 1
    assert record.status == "available"
    assert record.original_output_model == "rag.agent.primitive_ops.ListFilesOutput"
    assert record.payload == output
    assert ref.path.startswith(".agent_memory/records/")

    tombstone = store.tombstone(ref, reason="retention_limit")
    resolved = store.resolve(ref)

    assert tombstone.status == "deleted"
    assert resolved.status == "deleted"
    assert resolved.ref_deleted is True
    assert resolved.payload is None
    assert resolved.summary == "list_files files=1 first=input_files/sales.csv"
    assert resolved.reason == "retention_limit"


def test_store_rejects_invalid_or_escaping_refs(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store = WorkspaceMemoryStore(workspace=workspace)

    with pytest.raises(MemoryRefError):
        store.resolve(
            MemoryRef(
                ref_id="../bad",
                path=".agent_memory/records/../bad.json",
                summary="bad",
            )
        )

    with pytest.raises(MemoryRefError):
        store.resolve(
            MemoryRef(
                ref_id="mem_abs",
                path=str((tmp_path / "outside.json").resolve()),
                summary="bad",
            )
        )
