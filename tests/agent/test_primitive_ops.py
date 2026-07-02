"""Tests for rag.agent.primitive_ops."""

from __future__ import annotations

import shutil
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
    StructuredProbeInput,
    WriteFileInput,
)
from rag.agent.runner.python_runner import LocalSubprocessPythonRunner, PythonRunResult
from rag.agent.workspace import WorkspacePathError, WorkspaceRuntime

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


requires_seatbelt = pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)


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


@pytest.fixture()
def subprocess_ops(ws: WorkspaceRuntime) -> PrimitiveOps:
    return PrimitiveOps(workspace=ws, python_runner=LocalSubprocessPythonRunner())


# ---------------------------------------------------------------------------
# Fake runner (for deterministic tests)
# ---------------------------------------------------------------------------


@dataclass
class FakePythonRunner:
    """Records the call and returns a canned result."""

    result: PythonRunResult
    last_call: dict[str, object] | None = None

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

    def test_structured_probe_input_defaults(self) -> None:
        inp = StructuredProbeInput(path="input_files/data.csv")
        assert inp.max_rows == 20
        assert inp.max_columns == 50
        assert inp.max_tables == 5

    def test_file_info_model(self) -> None:
        fi = FileInfo(name="a", path="a", size=10, is_dir=False, modified_at=0.0)
        assert fi.is_dir is False
        assert fi.file_kind == "unknown"
        assert fi.capabilities == []


# ===================================================================
# list_files
# ===================================================================


class TestListFiles:
    def test_root_directory(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        (ws.scratch / "hello.txt").write_text("hi")
        out = ops.list_files(ListFilesInput())
        names = [f.name for f in out.files]
        assert "scratch" in names
        assert ".agent_memory" not in names

    def test_agent_memory_directory_is_hidden_from_root_listing(
        self,
        ws: WorkspaceRuntime,
        ops: PrimitiveOps,
    ) -> None:
        (ws.root / ".agent_memory" / "records").mkdir(parents=True)
        (ws.root / ".agent_memory" / "records" / "mem_1.json").write_text("{}")

        out = ops.list_files(ListFilesInput())

        assert ".agent_memory" not in {item.name for item in out.files}

    def test_agent_memory_directory_cannot_be_listed_directly(
        self,
        ops: PrimitiveOps,
    ) -> None:
        with pytest.raises(PermissionError, match="agent memory"):
            ops.list_files(ListFilesInput(path=".agent_memory"))

    def test_subdirectory(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        (ws.scratch / "a.txt").write_text("a")
        (ws.scratch / "b.txt").write_text("b")
        out = ops.list_files(ListFilesInput(path="scratch"))
        names = [f.name for f in out.files]
        assert sorted(names) == ["a.txt", "b.txt"]

    def test_file_metadata_advertises_text_and_binary_capabilities(
        self,
        ws: WorkspaceRuntime,
        ops: PrimitiveOps,
    ) -> None:
        (ws.input_files / "note.txt").write_text("hello")
        (ws.input_files / "report.xlsx").write_bytes(
            b"PK\x03\x04\x00\x00\x00\x00\x00\x00binary\x00payload"
        )

        out = ops.list_files(ListFilesInput(path="input_files"))
        by_name = {item.name: item for item in out.files}

        text_file = by_name["note.txt"]
        assert text_file.mime_type == "text/plain"
        assert text_file.file_kind == "text"
        assert text_file.is_binary is False
        assert text_file.readable_as_text is True
        assert text_file.capabilities == ["read_file", "structured_probe", "run_python"]

        binary_file = by_name["report.xlsx"]
        assert binary_file.mime_type == (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert binary_file.file_kind == "binary"
        assert binary_file.is_binary is True
        assert binary_file.readable_as_text is False
        assert binary_file.capabilities == ["structured_probe", "run_python"]

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

    def test_binary_excel_file_is_not_decoded_as_text(
        self,
        ws: WorkspaceRuntime,
        ops: PrimitiveOps,
    ) -> None:
        raw = b"PK\x03\x04\x00\x00\x00\x00\x00\x00binary\x00payload"
        (ws.input_files / "report.xlsx").write_bytes(raw)

        out = ops.read_file(ReadFileInput(path="input_files/report.xlsx"))

        assert out.is_binary is True
        assert out.content == ""
        assert out.truncated is False
        assert out.size_bytes == len(raw)

    def test_escapes_workspace_raises(self, ops: PrimitiveOps) -> None:
        with pytest.raises(WorkspacePathError):
            ops.read_file(ReadFileInput(path="../../etc/passwd"))

    def test_agent_memory_file_read_is_denied(
        self,
        ws: WorkspaceRuntime,
        ops: PrimitiveOps,
    ) -> None:
        record = ws.agent_memory / "records" / "mem_1.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("{}")

        with pytest.raises(PermissionError, match="agent memory"):
            ops.read_file(ReadFileInput(path=".agent_memory/records/mem_1.json"))


# ===================================================================
# structured_probe
# ===================================================================


class TestStructuredProbe:
    def test_csv_detects_header_after_title_and_note(
        self,
        ws: WorkspaceRuntime,
        ops: PrimitiveOps,
    ) -> None:
        (ws.input_files / "sales.csv").write_text(
            "2026 sales report,,,\n"
            "source: finance team,,,\n"
            "region,city,amount,price\n"
            "north,beijing,10,2.5\n"
            "east,shanghai,12,3.0\n",
            encoding="utf-8",
        )

        out = ops.structured_probe(StructuredProbeInput(path="input_files/sales.csv"))

        assert out.path == "input_files/sales.csv"
        assert out.mime_type == "text/csv"
        assert len(out.tables) == 1
        [table] = out.tables
        assert table.name == "sales.csv"
        assert table.row_count == 5
        assert table.column_count == 4
        assert table.sample_rows[0][0] == "2026 sales report"
        assert table.candidate_header_rows
        assert table.candidate_header_rows[0].row_index == 3
        assert table.data_start_row == 4
        assert table.used_range == "A1:D5"

    def test_excel_detects_header_after_title_and_note(
        self,
        ws: WorkspaceRuntime,
        ops: PrimitiveOps,
    ) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Sales"
        sheet.append(["2026 sales report", None, None, None])
        sheet.append(["source: finance team", None, None, None])
        sheet.append(["region", "city", "amount", "price"])
        sheet.append(["north", "beijing", 10, 2.5])
        sheet.append(["east", "shanghai", 12, 3.0])
        workbook.save(ws.input_files / "sales.xlsx")

        out = ops.structured_probe(StructuredProbeInput(path="input_files/sales.xlsx"))

        assert out.path == "input_files/sales.xlsx"
        assert out.file_kind == "binary"
        assert len(out.tables) == 1
        [table] = out.tables
        assert table.name == "Sales"
        assert table.row_count == 5
        assert table.column_count == 4
        assert table.sample_rows[2] == ["region", "city", "amount", "price"]
        assert table.candidate_header_rows
        assert table.candidate_header_rows[0].row_index == 3
        assert table.data_start_row == 4
        assert table.used_range == "A1:D5"


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
    def test_successful_script(self, ws: WorkspaceRuntime, subprocess_ops: PrimitiveOps) -> None:
        script = ws.scratch / "hello.py"
        script.write_text('print("hello")')
        out = subprocess_ops.run_python(RunPythonInput(script_path="scratch/hello.py"))
        assert out.ok is True
        assert out.exit_code == 0
        assert "hello" in out.stdout

    def test_failing_script(self, ws: WorkspaceRuntime, subprocess_ops: PrimitiveOps) -> None:
        script = ws.scratch / "fail.py"
        script.write_text("import sys; sys.exit(1)")
        out = subprocess_ops.run_python(RunPythonInput(script_path="scratch/fail.py"))
        assert out.ok is False
        assert out.exit_code == 1

    def test_script_generating_files(
        self, ws: WorkspaceRuntime, subprocess_ops: PrimitiveOps
    ) -> None:
        script = ws.scratch / "gen.py"
        script.write_text(
            'from pathlib import Path\nPath("scratch/output.txt").write_text("created")'
        )
        out = subprocess_ops.run_python(RunPythonInput(script_path="scratch/gen.py"))
        assert "scratch/output.txt" in out.generated_files
        assert (ws.scratch / "output.txt").read_text() == "created"

    @requires_seatbelt
    def test_script_tempfile_uses_scratch(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        script = ws.scratch / "tmpfile.py"
        script.write_text(
            "import tempfile\n"
            "from pathlib import Path\n"
            "with tempfile.NamedTemporaryFile('w', delete=False) as f:\n"
            "    f.write('tmp')\n"
            "    print(Path(f.name).relative_to(Path.cwd()))\n"
        )

        out = ops.run_python(RunPythonInput(script_path="scratch/tmpfile.py"))

        assert out.ok is True
        assert "scratch/" in out.stdout

    @requires_seatbelt
    def test_script_cannot_write_outside_workspace(
        self, tmp_path: Path, ws: WorkspaceRuntime, ops: PrimitiveOps
    ) -> None:
        script = ws.scratch / "escape.py"
        script.write_text(
            "from pathlib import Path\n"
            "Path('../outside.txt').write_text('bad')\n"
        )

        out = ops.run_python(RunPythonInput(script_path="scratch/escape.py"))

        assert out.ok is False
        assert out.exit_code != 0
        assert not (tmp_path / "outside.txt").exists()
        assert "Operation not permitted" in out.stderr

    @requires_seatbelt
    def test_script_cannot_write_input_files(
        self, ws: WorkspaceRuntime, ops: PrimitiveOps
    ) -> None:
        script = ws.scratch / "write_input.py"
        script.write_text(
            "from pathlib import Path\n"
            "Path('input_files/source.csv').write_text('mutated')\n"
        )

        out = ops.run_python(RunPythonInput(script_path="scratch/write_input.py"))

        assert out.ok is False
        assert out.exit_code != 0
        assert not (ws.input_files / "source.csv").exists()
        assert "Operation not permitted" in out.stderr

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


class TestWorkspaceTools:
    """All workspace tools are self-contained BaseTool classes."""

    def test_all_tool_classes_registered(self) -> None:
        from rag.agent.tools.workspace_tools import WORKSPACE_TOOL_CLASSES

        names = {cls.name for cls in WORKSPACE_TOOL_CLASSES}
        assert "list_files" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "run_python" in names
        assert "search_text" in names
        assert "apply_patch" in names
        assert "run_command" in names
        assert "tool_repl" in names
        assert "structured_probe" in names

    def test_create_workspace_tools(self, ops: PrimitiveOps) -> None:
        from rag.agent.tools.workspace_tools import create_workspace_tools

        tools = create_workspace_tools(ops._workspace)
        assert {tool.name for tool in tools} >= {
            "materialize_skill_asset",
        }
        assert len(tools) == 10
        for tool in tools:
            spec = tool.to_spec()
            assert spec.name
            assert spec.description
            assert tool.aci is not None  # every tool has a ToolCard
