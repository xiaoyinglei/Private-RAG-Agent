from __future__ import annotations

from rag.agent.tools.primitive_tools import ALL_PRIMITIVE_TOOLS
from rag.agent.tools.spec import ToolError


class TestPrimitiveToolSpecs:
    def test_all_four_specs_present(self) -> None:
        names = {spec.name for spec in ALL_PRIMITIVE_TOOLS}
        assert names == {"list_files", "read_file", "write_file", "run_python"}

    def test_list_files_permissions(self) -> None:
        spec = next(s for s in ALL_PRIMITIVE_TOOLS if s.name == "list_files")
        assert spec.permissions.read_fs is True
        assert spec.permissions.write_fs is False
        assert spec.permissions.execute_code is False

    def test_read_file_permissions(self) -> None:
        spec = next(s for s in ALL_PRIMITIVE_TOOLS if s.name == "read_file")
        assert spec.permissions.read_fs is True
        assert spec.permissions.write_fs is False

    def test_write_file_permissions(self) -> None:
        spec = next(s for s in ALL_PRIMITIVE_TOOLS if s.name == "write_file")
        assert spec.permissions.write_fs is True
        assert spec.permissions.read_fs is False

    def test_run_python_permissions(self) -> None:
        spec = next(s for s in ALL_PRIMITIVE_TOOLS if s.name == "run_python")
        assert spec.permissions.execute_code is True
        assert spec.permissions.read_fs is True
        assert spec.permissions.write_fs is True

    def test_all_specs_have_timeout(self) -> None:
        for spec in ALL_PRIMITIVE_TOOLS:
            assert spec.timeout_seconds > 0

    def test_all_specs_have_error_model(self) -> None:
        for spec in ALL_PRIMITIVE_TOOLS:
            assert spec.error_model is ToolError
