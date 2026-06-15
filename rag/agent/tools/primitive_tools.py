from __future__ import annotations

from rag.agent.primitive_ops import (
    ListFilesInput,
    ListFilesOutput,
    ReadFileInput,
    ReadFileOutput,
    RunPythonInput,
    RunPythonOutput,
    StructuredProbeInput,
    StructuredProbeOutput,
    WriteFileInput,
    WriteFileOutput,
)
from rag.agent.tools.spec import ToolError, ToolPermissions, ToolSpec

list_files_spec = ToolSpec(
    name="list_files",
    description=(
        "List files and directories in the workspace. Returns path, size, MIME/type "
        "metadata, text/binary flags, and advertised capabilities."
    ),
    input_model=ListFilesInput,
    output_model=ListFilesOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_fs=True),
    timeout_seconds=5.0,
    max_retries=1,
    idempotent=True,
    concurrency_safe=True,
    is_read_only=True,
    work_budget_cost=200,
)

read_file_spec = ToolSpec(
    name="read_file",
    description=(
        "Read a bounded text file from the workspace. Binary or non-text files return "
        "is_binary=True without body content. Supports offset/limit for reading "
        "specific byte ranges of large files."
    ),
    input_model=ReadFileInput,
    output_model=ReadFileOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_fs=True),
    timeout_seconds=10.0,
    max_retries=1,
    idempotent=True,
    concurrency_safe=True,
    is_read_only=True,
    work_budget_cost=500,
)

structured_probe_spec = ToolSpec(
    name="structured_probe",
    description=(
        "Inspect a workspace file for bounded structured data samples. Returns "
        "candidate tables, sample rows, candidate header rows, data start rows, and "
        "confidence without performing business analysis."
    ),
    input_model=StructuredProbeInput,
    output_model=StructuredProbeOutput,
    error_model=ToolError,
    permissions=ToolPermissions(read_fs=True),
    timeout_seconds=20.0,
    max_retries=0,
    idempotent=True,
    concurrency_safe=True,
    is_read_only=True,
    work_budget_cost=700,
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
    work_budget_cost=200,
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
    work_budget_cost=1000,
)

ALL_PRIMITIVE_TOOLS = [
    list_files_spec,
    read_file_spec,
    structured_probe_spec,
    write_file_spec,
    run_python_spec,
]


__all__ = ["ALL_PRIMITIVE_TOOLS"]
