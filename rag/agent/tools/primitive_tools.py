from __future__ import annotations

from rag.agent.primitive_ops import (
    ListFilesInput,
    ListFilesOutput,
    ReadFileInput,
    ReadFileOutput,
    RunPythonInput,
    RunPythonOutput,
    WriteFileInput,
    WriteFileOutput,
)
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec

list_files_spec = ToolSpec(
    name="list_files",
    description="List files and directories in the workspace. Returns name, path, size, type, and modification time.",
    input_model=ListFilesInput,
    output_model=ListFilesOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_fs=True),
    timeout_seconds=5.0,
    max_retries=1,
    token_budget_cost=200,
)

read_file_spec = ToolSpec(
    name="read_file",
    description="Read a text file from the workspace. Returns content with truncation protection.",
    input_model=ReadFileInput,
    output_model=ReadFileOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_fs=True),
    timeout_seconds=10.0,
    max_retries=1,
    token_budget_cost=500,
)

write_file_spec = ToolSpec(
    name="write_file",
    description="Write content to a file in the workspace (scratch/, artifacts/, reports/, logs/ only).",
    input_model=WriteFileInput,
    output_model=WriteFileOutput,
    error_model=ToolError,
    permissions=ToolPermissions(write_fs=True),
    timeout_seconds=5.0,
    max_retries=0,
    token_budget_cost=200,
)

run_python_spec = ToolSpec(
    name="run_python",
    description="Execute a Python script from scratch/. Returns exit code, stdout, stderr, and generated files.",
    input_model=RunPythonInput,
    output_model=RunPythonOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_fs=True, write_fs=True, execute_code=True),
    timeout_seconds=60.0,
    max_retries=0,
    token_budget_cost=1000,
)

ALL_PRIMITIVE_TOOLS = [list_files_spec, read_file_spec, write_file_spec, run_python_spec]


__all__ = ["ALL_PRIMITIVE_TOOLS"]
