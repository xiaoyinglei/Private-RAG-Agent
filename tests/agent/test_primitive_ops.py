"""Tests for rag.agent.primitive_ops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from rag.agent.primitive_ops import (
    FileInfo,
    ListFilesInput,
    PrimitiveOps,
    ReadFileInput,
    RunPythonInput,
    RunPythonOutput,
    WriteFileInput,
)
from rag.agent.runner.python_runner import PythonRunResult
from rag.agent.workspace import WorkspacePathError, WorkspaceRuntime

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ws(tmp_path: Path) -> WorkspaceRuntime:
    root = tmp_path / "workspace"
    root.mkdir()
    runtime = WorkspaceRuntime(root=root, is_temporary=True)
    runtime.initialize()
    return runtime


@pytest.fixture()
def ops(ws: WorkspaceRuntime) -> PrimitiveOps:
    return PrimitiveOps(workspace=ws)


# ---------------------------------------------------------------------------
# Fake runner (for deterministic tests)
# ---------------------------------------------------------------------------


@dataclass
class FakePythonRunner:
    """Records the call and returns a canned result."""

    result: PythonRunResult
    last_call: dict | None = None

    def run(
        self,
        script_path: Path,
        *,
        args: list[str],
        cwd: Path,
        timeout: float,
    ) -> PythonRunResult:
        self.last_call = {
            "script_path": script_path,
            "args": args,
            "cwd": cwd,
            "timeout": timeout,
        }
        return self.result


# ===================================================================
# I/O model defaults
# ===================================================================


class TestIOModels:
    def test_list_files_input_defaults(self) -> None:
        inp = ListFilesInput()
        assert inp.path == ""
        assert inp.pattern is None

    def test_read_file_input_defaults(self) -> None:
        inp = ReadFileInput(path="foo.txt")
        assert inp.encoding == "utf-8"
        assert inp.max_bytes == 1_000_000

    def test_write_file_input_defaults(self) -> None:
        inp = WriteFileInput(path="x", content="hi")
        assert inp.encoding == "utf-8"
        assert inp.overwrite is False

    def test_run_python_input_defaults(self) -> None:
        inp = RunPythonInput(script_path="s.py")
        assert inp.args == []
        assert inp.timeout_seconds == 30.0

    def test_run_python_output_fields(self) -> None:
        out = RunPythonOutput(
            ok=True,
            exit_code=0,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_ms=1.0,
            generated_files=[],
        )
        assert out.ok is True

    def test_file_info_model(self) -> None:
        fi = FileInfo(name="a", path="a", size=10, is_dir=False, modified_at=0.0)
        assert fi.is_dir is False


# ===================================================================
# list_files
# ===================================================================


class TestListFiles:
    def test_root_directory(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        (ws.scratch / "hello.txt").write_text("hi")
        out = ops.list_files(ListFilesInput())
        names = [f.name for f in out.files]
        assert "scratch" in names

    def test_subdirectory(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        (ws.scratch / "a.txt").write_text("a")
        (ws.scratch / "b.txt").write_text("b")
        out = ops.list_files(ListFilesInput(path="scratch"))
        names = [f.name for f in out.files]
        assert sorted(names) == ["a.txt", "b.txt"]

    def test_empty_directory(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        out = ops.list_files(ListFilesInput(path="logs"))
        assert out.files == []

    def test_nonexistent_directory(self, ops: PrimitiveOps) -> None:
        out = ops.list_files(ListFilesInput(path="no_such_dir"))
        assert out.files == []

    def test_pattern_filter(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        (ws.scratch / "a.py").write_text("x")
        (ws.scratch / "b.txt").write_text("y")
        out = ops.list_files(ListFilesInput(path="scratch", pattern="*.py"))
        assert len(out.files) == 1
        assert out.files[0].name == "a.txt"  or out.files[0].name == "a.py"

    def test_escapes_workspace_raises(self, ops: PrimitiveOps) -> None:
        with pytest.raises(WorkspacePathError):
            ops.list_files(ListFilesInput(path="../.."))


# ===================================================================
# read_file
# ===================================================================


class TestReadFile:
    def test_text_file(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        (ws.scratch / "note.txt").write_text("hello world")
        out = ops.read_file(ReadFileInput(path="scratch/note.txt"))
        assert out.content == "hello world"
        assert out.truncated is False
        assert out.size_bytes == len("hello world")

    def test_input_files(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        (ws.input_files / "data.csv").write_text("col1,col2\n1,2")
        out = ops.read_file(ReadFileInput(path="input_files/data.csv"))
        assert "col1" in out.content

    def test_nonexistent_raises(self, ops: PrimitiveOps) -> None:
        with pytest.raises(FileNotFoundError):
            ops.read_file(ReadFileInput(path="scratch/nope.txt"))

    def test_truncation(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        big = "x" * 200
        (ws.scratch / "big.txt").write_text(big)
        out = ops.read_file(
            ReadFileInput(path="scratch/big.txt", max_bytes=100)
        )
        assert out.truncated is True
        assert len(out.content) == 100

    def test_escapes_workspace_raises(self, ops: PrimitiveOps) -> None:
        with pytest.raises(WorkspacePathError):
            ops.read_file(ReadFileInput(path="../../etc/passwd"))


# ===================================================================
# write_file
# ===================================================================


class TestWriteFile:
    def test_write_to_scratch(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        out = ops.write_file(
            WriteFileInput(path="scratch/out.txt", content="result")
        )
        assert out.size_bytes == len("result")
        assert (ws.scratch / "out.txt").read_text() == "result"

    def test_write_to_artifacts(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        out = ops.write_file(
            WriteFileInput(path="artifacts/res.json", content="{}")
        )
        assert out.path == "artifacts/res.json"
        assert (ws.artifacts / "res.json").read_text() == "{}"

    def test_write_to_reports(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        ops.write_file(
            WriteFileInput(path="reports/summary.md", content="# Report")
        )
        assert (ws.reports / "summary.md").read_text() == "# Report"

    def test_write_to_logs(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        ops.write_file(
            WriteFileInput(path="logs/run.log", content="log line")
        )
        assert (ws.logs / "run.log").read_text() == "log line"

    def test_write_to_input_files_denied(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        with pytest.raises(PermissionError, match="input_files"):
            ops.write_file(
                WriteFileInput(path="input_files/bad.txt", content="nope")
            )

    def test_write_to_workspace_root_denied(self, ops: PrimitiveOps) -> None:
        with pytest.raises(PermissionError, match="scratch/.*artifacts/.*reports/.*logs/"):
            ops.write_file(
                WriteFileInput(path="root_file.txt", content="nope")
            )

    def test_no_overwrite_raises(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        (ws.scratch / "exists.txt").write_text("old")
        with pytest.raises(FileExistsError, match="overwrite=False"):
            ops.write_file(
                WriteFileInput(path="scratch/exists.txt", content="new")
            )

    def test_overwrite_works(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        (ws.scratch / "exists.txt").write_text("old")
        ops.write_file(
            WriteFileInput(path="scratch/exists.txt", content="new", overwrite=True)
        )
        assert (ws.scratch / "exists.txt").read_text() == "new"

    def test_creates_parent_dirs(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        ops.write_file(
            WriteFileInput(path="scratch/sub/deep/file.txt", content="deep")
        )
        assert (ws.scratch / "sub/deep/file.txt").read_text() == "deep"

    def test_escapes_workspace_raises(self, ops: PrimitiveOps) -> None:
        with pytest.raises(WorkspacePathError):
            ops.write_file(
                WriteFileInput(path="../escape.txt", content="bad")
            )


# ===================================================================
# run_python
# ===================================================================


class TestRunPython:
    def test_successful_script(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        script = ws.scratch / "hello.py"
        script.write_text('print("hello")')
        out = ops.run_python(RunPythonInput(script_path="scratch/hello.py"))
        assert out.ok is True
        assert out.exit_code == 0
        assert "hello" in out.stdout

    def test_failing_script(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        script = ws.scratch / "fail.py"
        script.write_text("import sys; sys.exit(1)")
        out = ops.run_python(RunPythonInput(script_path="scratch/fail.py"))
        assert out.ok is False
        assert out.exit_code == 1

    def test_script_generating_files(
        self, ws: WorkspaceRuntime, ops: PrimitiveOps
    ) -> None:
        script = ws.scratch / "gen.py"
        script.write_text(
            'from pathlib import Path\nPath("scratch/output.txt").write_text("created")'
        )
        out = ops.run_python(RunPythonInput(script_path="scratch/gen.py"))
        assert "scratch/output.txt" in out.generated_files
        assert (ws.scratch / "output.txt").read_text() == "created"

    def test_script_outside_scratch_denied(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        script = ws.artifacts / "evil.py"
        script.write_text("print('hi')")
        with pytest.raises(WorkspacePathError):
            ops.run_python(RunPythonInput(script_path="artifacts/evil.py"))

    def test_non_python_file_denied(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        script = ws.scratch / "run.sh"
        script.write_text("#!/bin/bash\necho hi")
        with pytest.raises(ValueError, match="Only .py"):
            ops.run_python(RunPythonInput(script_path="scratch/run.sh"))

    def test_nonexistent_script_raises(self, ops: PrimitiveOps) -> None:
        with pytest.raises(FileNotFoundError, match="Script not found"):
            ops.run_python(RunPythonInput(script_path="scratch/missing.py"))

    def test_uses_fake_runner(self, ws: WorkspaceRuntime) -> None:
        script = ws.scratch / "fake.py"
        script.write_text("")
        fake = FakePythonRunner(
            result=PythonRunResult(
                exit_code=0, stdout="from fake", stderr="", duration_ms=42.0
            )
        )
        ops = PrimitiveOps(workspace=ws, python_runner=fake)
        out = ops.run_python(RunPythonInput(script_path="scratch/fake.py"))
        assert out.stdout == "from fake"
        assert out.duration_ms == 42.0
        assert fake.last_call is not None
        assert fake.last_call["cwd"] == ws.root

    def test_escapes_workspace_raises(self, ops: PrimitiveOps) -> None:
        with pytest.raises(WorkspacePathError):
            ops.run_python(RunPythonInput(script_path="../../evil.py"))


# ===================================================================
# runners()
# ===================================================================


class TestRunners:
    def test_returns_all_four(self, ops: PrimitiveOps) -> None:
        r = ops.runners()
        assert set(r.keys()) == {"list_files", "read_file", "write_file", "run_python"}

    def test_runners_are_callable(self, ops: PrimitiveOps) -> None:
        r = ops.runners()
        assert callable(r["list_files"])
        assert callable(r["read_file"])
        assert callable(r["write_file"])
        assert callable(r["run_python"])
