"""Tests for file manifest, auto-probe, chart capture, and file-first processing.

Covers:
- FileManifest model and build_file_manifest
- Pandas preview for CSV and XLSX
- Structured probe with merged cells and formulas
- Chart/image capture in run_python_inline
- Context block generation
- LoopState file_manifest field
- Auto-activation of structured_probe
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.agent.file_manifest import (
    FileManifest,
    FileManifestEntry,
    SheetPreview,
    build_file_manifest,
)
from rag.agent.primitive_ops import (
    ImagePreview,
    PrimitiveOps,
    RunPythonInlineInput,
    StructuredProbeInput,
)
from rag.agent.runner.python_runner import LocalSubprocessPythonRunner
from rag.agent.workspace import WorkspaceRuntime

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


@pytest.fixture()
def subprocess_ops(ws: WorkspaceRuntime) -> PrimitiveOps:
    return PrimitiveOps(workspace=ws, python_runner=LocalSubprocessPythonRunner())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv(ws: WorkspaceRuntime, name: str, content: str) -> Path:
    p = ws.input_files / name
    p.write_text(content, encoding="utf-8")
    return p


def _make_xlsx(ws: WorkspaceRuntime, name: str, sheets: dict[str, list[list]]):
    """Create an XLSX with multiple sheets."""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    for i, (sheet_name, rows) in enumerate(sheets.items()):
        if i == 0:
            ws_obj = wb.active
            ws_obj.title = sheet_name
        else:
            ws_obj = wb.create_sheet(sheet_name)
        for row in rows:
            ws_obj.append(row)
    path = ws.input_files / name
    wb.save(path)
    return path


# ===================================================================
# FileManifest model
# ===================================================================


class TestFileManifestModel:
    def test_empty_manifest(self) -> None:
        m = FileManifest(
            files=[], total_size_bytes=0,
            has_structured_files=False, has_probeable_files=False,
        )
        assert m.to_context_block() == ""

    def test_single_csv_entry(self) -> None:
        entry = FileManifestEntry(
            path="input_files/data.csv",
            filename="data.csv",
            size_bytes=1024,
            mime_type="text/csv",
            file_kind="csv",
            hash="abc123",
            structured=True,
            probeable=True,
            sheets=[SheetPreview(
                sheet_name="data.csv",
                total_rows=100,
                total_columns=5,
                columns=[],
                head=[],
                dtypes={},
            )],
        )
        m = FileManifest(
            files=[entry], total_size_bytes=1024,
            has_structured_files=True, has_probeable_files=True,
        )
        block = m.to_context_block()
        assert "data.csv" in block
        assert "100 rows" in block
        assert "5 columns" in block
        assert "Available Tools" in block
        assert "run_python" in block

    def test_context_block_shows_warnings(self) -> None:
        entry = FileManifestEntry(
            path="input_files/report.xlsx",
            filename="report.xlsx",
            size_bytes=2048,
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            file_kind="xlsx",
            hash="def456",
            structured=True,
            probeable=True,
            sheets=[SheetPreview(
                sheet_name="Sheet1",
                total_rows=50,
                total_columns=3,
                columns=[],
                head=[],
                dtypes={},
                merged_cells=True,
                formula_columns=["C"],
            )],
        )
        m = FileManifest(
            files=[entry], total_size_bytes=2048,
            has_structured_files=True, has_probeable_files=True,
        )
        block = m.to_context_block()
        assert "merged cells" in block
        assert "formulas" in block


# ===================================================================
# build_file_manifest with CSV
# ===================================================================


class TestBuildManifestCSV:
    def test_simple_csv(self, ws: WorkspaceRuntime) -> None:
        _make_csv(ws, "sales.csv", "name,amount\nAlice,100\nBob,200\n")
        manifest = build_file_manifest(ws)

        assert len(manifest.files) == 1
        entry = manifest.files[0]
        assert entry.file_kind == "csv"
        assert entry.structured is True
        assert entry.probeable is True
        assert entry.error is None

        # Should have pandas preview
        assert len(entry.sheets) == 1
        sheet = entry.sheets[0]
        assert sheet.total_rows == 3  # header + 2 data rows
        assert sheet.total_columns == 2
        assert len(sheet.columns) == 2
        assert sheet.columns[0].name == "name"
        assert sheet.columns[1].name == "amount"
        assert len(sheet.head) == 2

    def test_csv_with_chinese_filename(self, ws: WorkspaceRuntime) -> None:
        _make_csv(ws, "销售数据.csv", "产品,数量\nA,10\nB,20\n")
        manifest = build_file_manifest(ws)

        assert len(manifest.files) == 1
        assert manifest.files[0].filename == "销售数据.csv"
        assert manifest.files[0].structured is True

    def test_empty_input_dir(self, ws: WorkspaceRuntime) -> None:
        manifest = build_file_manifest(ws)
        assert manifest.files == []
        assert manifest.has_structured_files is False


# ===================================================================
# build_file_manifest with XLSX
# ===================================================================


class TestBuildManifestXLSX:
    def test_single_sheet_xlsx(self, ws: WorkspaceRuntime) -> None:
        _make_xlsx(ws, "data.xlsx", {
            "Sales": [
                ["product", "qty", "price"],
                ["Widget", 10, 25.0],
                ["Gadget", 20, 30.0],
            ],
        })
        manifest = build_file_manifest(ws)

        assert len(manifest.files) == 1
        entry = manifest.files[0]
        assert entry.file_kind == "xlsx"
        assert entry.structured is True
        assert entry.probeable is True

        assert len(entry.sheets) == 1
        sheet = entry.sheets[0]
        assert sheet.sheet_name == "Sales"
        assert sheet.total_rows == 3
        assert sheet.total_columns == 3
        assert sheet.columns[0].name == "product"
        assert sheet.head[0]["product"] == "Widget"

    def test_multi_sheet_xlsx(self, ws: WorkspaceRuntime) -> None:
        _make_xlsx(ws, "report.xlsx", {
            "Summary": [["total", "count"], [1000, 50]],
            "Detail": [["id", "value"], [1, 100], [2, 200]],
        })
        manifest = build_file_manifest(ws)

        assert len(manifest.files) == 1
        assert len(manifest.files[0].sheets) == 2
        names = [s.sheet_name for s in manifest.files[0].sheets]
        assert "Summary" in names
        assert "Detail" in names

    def test_xlsx_with_merged_cells(self, ws: WorkspaceRuntime) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = "Merged"
        sheet.append(["Header1", "Header2"])
        sheet.merge_cells("A1:B1")
        sheet.append([1, 2])
        wb.save(ws.input_files / "merged.xlsx")

        manifest = build_file_manifest(ws)
        entry = manifest.files[0]
        assert entry.sheets[0].merged_cells is True

    def test_xlsx_with_formulas(self, ws: WorkspaceRuntime) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = "Calc"
        sheet.append(["A", "B", "Sum"])
        sheet.append([10, 20, None])
        # Write formula in C2
        sheet["C2"] = "=A2+B2"
        wb.save(ws.input_files / "formula.xlsx")

        manifest = build_file_manifest(ws)
        entry = manifest.files[0]
        assert "C" in entry.sheets[0].formula_columns


# ===================================================================
# Structured probe enhancements
# ===================================================================


class TestProbeEnhancements:
    def test_probe_xlsx_merged_cells(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = "Test"
        sheet.append(["A", "B"])
        sheet.merge_cells("A1:B1")
        sheet.append([1, 2])
        wb.save(ws.input_files / "test.xlsx")

        out = ops.structured_probe(StructuredProbeInput(path="input_files/test.xlsx"))
        assert len(out.tables) == 1
        assert out.tables[0].merged_cells is True

    def test_probe_xlsx_formulas(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = "Test"
        sheet.append(["X", "Y", "Z"])
        sheet.append([1, 2, None])
        sheet["C2"] = "=A2+B2"
        wb.save(ws.input_files / "formula.xlsx")

        out = ops.structured_probe(StructuredProbeInput(path="input_files/formula.xlsx"))
        assert len(out.tables) == 1
        assert "C" in out.tables[0].formula_columns

    def test_probe_csv_no_merged_or_formulas(self, ws: WorkspaceRuntime, ops: PrimitiveOps) -> None:
        _make_csv(ws, "plain.csv", "a,b\n1,2\n")
        out = ops.structured_probe(StructuredProbeInput(path="input_files/plain.csv"))
        assert len(out.tables) == 1
        assert out.tables[0].merged_cells is False
        assert out.tables[0].formula_columns == []


# ===================================================================
# Chart/image capture
# ===================================================================


class TestChartCapture:
    def test_matplotlib_savefig_captured(
        self, ws: WorkspaceRuntime, subprocess_ops: PrimitiveOps,
    ) -> None:
        """Matplotlib savefig should be captured as image_preview."""
        code = """
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig("scratch/test_chart.png")
plt.close()
"""
        out = subprocess_ops.run_python_inline(RunPythonInlineInput(code=code))
        assert out.ok is True
        assert any("test_chart.png" in f for f in out.generated_files)

    def test_plt_show_auto_saves(
        self, ws: WorkspaceRuntime, subprocess_ops: PrimitiveOps,
    ) -> None:
        """plt.show() should auto-save figures via the preamble."""
        code = """
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [10, 20, 30])
plt.title("Test Chart")
plt.show()
"""
        out = subprocess_ops.run_python_inline(RunPythonInlineInput(code=code))
        assert out.ok is True
        # The preamble should have saved the chart
        chart_files = [f for f in out.generated_files if "chart_" in f and f.endswith(".png")]
        assert len(chart_files) >= 1

    def test_no_chart_when_no_matplotlib(
        self, ws: WorkspaceRuntime, subprocess_ops: PrimitiveOps,
    ) -> None:
        """Code without matplotlib should not produce charts."""
        code = "print(42)"
        out = subprocess_ops.run_python_inline(RunPythonInlineInput(code=code))
        assert out.ok is True
        assert out.image_previews == []

    def test_image_preview_model(self) -> None:
        """ImagePreview model works correctly."""
        preview = ImagePreview(
            path="scratch/chart_001.png",
            base64_data="iVBORw0KGgo=",
            mime_type="image/png",
            width=800,
            height=600,
        )
        assert preview.path == "scratch/chart_001.png"
        assert preview.width == 800


# ===================================================================
# LoopState integration
# ===================================================================


class TestLoopStateIntegration:
    def _make_config(self):
        from rag.agent.core.context import AgentRunConfig
        from rag.schema.runtime import AccessPolicy
        return AgentRunConfig(
            run_id="test",
            thread_id="test",
            budget_total=1000,
            max_depth=2,
            access_policy=AccessPolicy.default(),
            work_budget_total=1000,
            agent_type="generic",
        )

    def test_create_loop_state_with_manifest(self) -> None:
        from rag.agent.file_manifest import FileManifest
        from rag.agent.loop.state import create_loop_state

        manifest = FileManifest(
            files=[], total_size_bytes=0,
            has_structured_files=False, has_probeable_files=False,
        )
        config = self._make_config()
        state = create_loop_state(task="test", run_config=config, file_manifest=manifest)
        assert state["file_manifest"] is not None
        assert state["file_manifest"].files == []

    def test_create_loop_state_without_manifest(self) -> None:
        from rag.agent.loop.state import create_loop_state

        config = self._make_config()
        state = create_loop_state(task="test", run_config=config)
        assert state["file_manifest"] is None


# ===================================================================
# Context block rendering
# ===================================================================


class TestContextBlock:
    def test_manifest_with_no_files_renders_empty(self) -> None:
        m = FileManifest(
            files=[], total_size_bytes=0,
            has_structured_files=False, has_probeable_files=False,
        )
        assert m.to_context_block() == ""

    def test_manifest_shows_available_packages(self) -> None:
        entry = FileManifestEntry(
            path="input_files/data.csv",
            filename="data.csv",
            size_bytes=100,
            mime_type="text/csv",
            file_kind="csv",
            hash="abc",
            structured=True,
            probeable=True,
            sheets=[SheetPreview(
                sheet_name="data.csv",
                total_rows=5,
                total_columns=2,
                columns=[],
                head=[{"a": 1, "b": 2}],
                dtypes={},
            )],
        )
        m = FileManifest(
            files=[entry], total_size_bytes=100,
            has_structured_files=True, has_probeable_files=True,
        )
        block = m.to_context_block()
        assert "Available Python Packages" in block
        # pandas should be listed since it's installed
        assert "pandas" in block

    def test_manifest_error_entry(self) -> None:
        entry = FileManifestEntry(
            path="input_files/bad.xlsx",
            filename="bad.xlsx",
            size_bytes=100,
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            file_kind="xlsx",
            hash="abc",
            structured=True,
            probeable=True,
            error="probe_failed: corrupt file",
        )
        m = FileManifest(
            files=[entry], total_size_bytes=100,
            has_structured_files=True, has_probeable_files=True,
        )
        block = m.to_context_block()
        assert "probe_failed" in block


# ===================================================================
# End-to-end: file-first processing path
# ===================================================================


class TestFileFirstPath:
    def test_csv_manifest_enables_first_turn_analysis(
        self, ws: WorkspaceRuntime,
    ) -> None:
        """Full path: CSV → manifest → model has everything it needs."""
        _make_csv(ws, "orders.csv",
                  "order_id,product,amount\n"
                  "1,Widget,100\n2,Gadget,200\n3,Widget,150\n")
        manifest = build_file_manifest(ws)

        # Manifest should have complete info
        assert manifest.has_structured_files is True
        assert manifest.has_probeable_files is True
        entry = manifest.files[0]
        assert entry.structured is True
        assert len(entry.sheets) == 1

        sheet = entry.sheets[0]
        assert sheet.total_rows == 4
        assert sheet.total_columns == 3
        col_names = [c.name for c in sheet.columns]
        assert "order_id" in col_names
        assert "product" in col_names
        assert "amount" in col_names

        # Context block should be usable
        block = manifest.to_context_block()
        assert "orders.csv" in block
        assert "run_python" in block

    def test_xlsx_manifest_shows_all_sheets(
        self, ws: WorkspaceRuntime,
    ) -> None:
        """Multi-sheet XLSX → manifest lists all sheets with previews."""
        _make_xlsx(ws, "report.xlsx", {
            "Sales": [["date", "amount"], ["2024-01", 1000], ["2024-02", 1500]],
            "Inventory": [["item", "stock"], ["Widget", 50], ["Gadget", 30]],
        })
        manifest = build_file_manifest(ws)

        entry = manifest.files[0]
        assert len(entry.sheets) == 2
        sheet_names = [s.sheet_name for s in entry.sheets]
        assert "Sales" in sheet_names
        assert "Inventory" in sheet_names

        # Each sheet should have columns
        for sheet in entry.sheets:
            assert len(sheet.columns) == 2
            assert len(sheet.head) >= 1
