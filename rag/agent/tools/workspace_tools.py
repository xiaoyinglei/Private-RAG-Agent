"""Workspace tools — self-contained Tool classes (Claude-style).

Each tool = spec + execute in one class.  Workspace is injected via __init__.
Registration: registry.register_tool(tool_instance) — one call, no forgetting.

I/O models are imported from primitive_ops for compatibility with PrimitiveOps
execution logic. Generic tools (search_text, apply_patch, run_command, tool_repl)
define their own models since they have no PrimitiveOps equivalent.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

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
from rag.agent.tools.base import BaseTool
from rag.agent.tools.card import ToolCard
from rag.agent.tools.spec import ExecutionCategory, ToolPermissions
from rag.agent.workspace import WorkspaceRuntime


# ═══════════════════════════════════════════════════════════════════════════════
# list_files / read_file / write_file / run_python / structured_probe
# ═══════════════════════════════════════════════════════════════════════════════

class ListFilesTool(BaseTool):
    name = "list_files"
    description = (
        "List files and directories in the workspace. Returns path, size, MIME/type "
        "metadata, text/binary flags, and advertised capabilities."
    )
    input_model = ListFilesInput
    output_model = ListFilesOutput
    permissions = ToolPermissions(read_fs=True)
    execution_category = ExecutionCategory.READ
    timeout_seconds = 5.0
    idempotent = True
    concurrency_safe = True
    work_budget_cost = 200
    aci = ToolCard(when_to_use="Use to explore directory structure.", activation_group="resident", selection_tags=("files",), domains=("files",))

    def __init__(self, workspace: WorkspaceRuntime): self._workspace = workspace
    async def execute(self, i: BaseModel, c: Any = None) -> BaseModel:
        from rag.agent.primitive_ops import PrimitiveOps
        return PrimitiveOps(workspace=self._workspace).list_files(i)


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a bounded text file from the workspace."
    input_model = ReadFileInput
    output_model = ReadFileOutput
    permissions = ToolPermissions(read_fs=True)
    execution_category = ExecutionCategory.READ
    timeout_seconds = 10.0
    idempotent = True
    concurrency_safe = True
    work_budget_cost = 500
    aci = ToolCard(when_to_use="Use to read file contents.", activation_group="resident", selection_tags=("files",), domains=("files",))

    def __init__(self, workspace: WorkspaceRuntime): self._workspace = workspace
    async def execute(self, i: BaseModel, c: Any = None) -> BaseModel:
        from rag.agent.primitive_ops import PrimitiveOps
        return PrimitiveOps(workspace=self._workspace).read_file(i)


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file in the workspace."
    input_model = WriteFileInput
    output_model = WriteFileOutput
    permissions = ToolPermissions(write_fs=True)
    execution_category = ExecutionCategory.WRITE
    timeout_seconds = 5.0
    work_budget_cost = 200
    aci = ToolCard(when_to_use="Create new files or rewrite entire files.", when_not_to_use="For targeted edits prefer apply_patch.", activation_group="workspace", selection_tags=("files",), domains=("files",))

    def __init__(self, workspace: WorkspaceRuntime): self._workspace = workspace
    async def execute(self, i: BaseModel, c: Any = None) -> BaseModel:
        from rag.agent.primitive_ops import PrimitiveOps
        return PrimitiveOps(workspace=self._workspace).write_file(i)


class RunPythonTool(BaseTool):
    name = "run_python"
    description = "Execute Python code. Provide script_path or code parameter."
    input_model = RunPythonInput
    output_model = RunPythonOutput
    permissions = ToolPermissions(read_fs=True, write_fs=True, execute_code=True)
    execution_category = ExecutionCategory.EXECUTE
    timeout_seconds = 120.0
    work_budget_cost = 1000
    aci = ToolCard(when_to_use="Python data processing, analysis, chart generation.", activation_group="workspace", selection_tags=("code",), domains=("code",))

    def __init__(self, workspace: WorkspaceRuntime): self._workspace = workspace
    async def execute(self, i: BaseModel, c: Any = None) -> BaseModel:
        from rag.agent.primitive_ops import PrimitiveOps
        return PrimitiveOps(workspace=self._workspace).run_python(i)


class StructuredProbeTool(BaseTool):
    name = "structured_probe"
    description = "Inspect workspace file for structured data samples."
    input_model = StructuredProbeInput
    output_model = StructuredProbeOutput
    permissions = ToolPermissions(read_fs=True)
    execution_category = ExecutionCategory.READ
    timeout_seconds = 20.0
    work_budget_cost = 700
    aci = ToolCard(when_to_use="Inspect CSV/XLSX/JSON before loading with run_python.", activation_group="workspace", selection_tags=("probe",), domains=("files",))

    def __init__(self, workspace: WorkspaceRuntime): self._workspace = workspace
    async def execute(self, i: BaseModel, c: Any = None) -> BaseModel:
        from rag.agent.primitive_ops import PrimitiveOps
        return PrimitiveOps(workspace=self._workspace).structured_probe(i)


# ═══════════════════════════════════════════════════════════════════════════════
# generic tools (own I/O models, implemented in PrimitiveOps methods)
# ═══════════════════════════════════════════════════════════════════════════════

class SearchTextInput(BaseModel):
    pattern: str
    path: str = "."
    file_types: str | None = None
    max_results: int = 40
    regex: bool = False
    context_lines: int = 0

class SearchTextMatch(BaseModel):
    file_path: str = ""
    line_number: int = 0
    line_content: str = ""

class SearchTextOutput(BaseModel):
    matches: list[SearchTextMatch] = []
    total_matches: int = 0
    truncated: bool = False


class SearchTextTool(BaseTool):
    name = "search_text"
    description = "Search workspace for text patterns (grep/rg equivalent)."
    input_model = SearchTextInput
    output_model = SearchTextOutput
    permissions = ToolPermissions(read_fs=True)
    execution_category = ExecutionCategory.READ
    timeout_seconds = 15.0
    idempotent = True
    concurrency_safe = True
    work_budget_cost = 200
    aci = ToolCard(when_to_use="Find where functions/classes/patterns are defined.", activation_group="workspace", selection_tags=("search",), domains=("code",))

    def __init__(self, workspace: WorkspaceRuntime): self._workspace = workspace
    async def execute(self, i: BaseModel, c: Any = None) -> BaseModel:
        from rag.agent.primitive_ops import PrimitiveOps
        return PrimitiveOps(workspace=self._workspace).search_text(i)


class ApplyPatchInput(BaseModel):
    file_path: str
    old_string: str
    new_string: str
    replace_all: bool = False

class ApplyPatchOutput(BaseModel):
    file_path: str = ""
    replaced: bool = False
    occurrences: int = 0
    message: str = ""


class ApplyPatchTool(BaseTool):
    name = "apply_patch"
    description = "Apply precise string replacement in a file."
    input_model = ApplyPatchInput
    output_model = ApplyPatchOutput
    permissions = ToolPermissions(write_fs=True)
    execution_category = ExecutionCategory.WRITE
    timeout_seconds = 5.0
    work_budget_cost = 100
    aci = ToolCard(when_to_use="Targeted edits under ~30 lines.", when_not_to_use="New files → write_file.", activation_group="workspace", selection_tags=("edit",), domains=("code",))

    def __init__(self, workspace: WorkspaceRuntime): self._workspace = workspace
    async def execute(self, i: BaseModel, c: Any = None) -> BaseModel:
        from rag.agent.primitive_ops import PrimitiveOps
        return PrimitiveOps(workspace=self._workspace).apply_patch(i)


class RunCommandInput(BaseModel):
    command: str
    working_dir: str = "."
    timeout_seconds: int = 120

class RunCommandOutput(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    truncated: bool = False
    duration_ms: float = 0.0


class RunCommandTool(BaseTool):
    name = "run_command"
    description = "Execute shell command (pytest, ruff, git, pip, npm, make)."
    input_model = RunCommandInput
    output_model = RunCommandOutput
    permissions = ToolPermissions(read_fs=True, write_fs=True, execute_code=True)
    execution_category = ExecutionCategory.EXECUTE
    timeout_seconds = 600.0
    work_budget_cost = 500
    aci = ToolCard(when_to_use="Tests, linters, git, package management.", activation_group="workspace", selection_tags=("shell",), domains=("code",))

    def __init__(self, workspace: WorkspaceRuntime): self._workspace = workspace
    async def execute(self, i: BaseModel, c: Any = None) -> BaseModel:
        from rag.agent.primitive_ops import PrimitiveOps
        return PrimitiveOps(workspace=self._workspace).run_command(i)


class ToolReplTool(BaseTool):
    name = "tool_repl"
    description = "Batch tool-calling via Python code using tools.declare()."
    input_model = RunCommandInput   # reuse: command field = Python code
    output_model = RunPythonOutput
    permissions = ToolPermissions(read_fs=True, write_fs=True, execute_code=True)
    execution_category = ExecutionCategory.EXECUTE
    timeout_seconds = 120.0
    work_budget_cost = 800
    aci = ToolCard(when_to_use="Call multiple tools in batch mode.", activation_group="workspace", selection_tags=("batch",), domains=("code",))

    def __init__(self, workspace: WorkspaceRuntime): self._workspace = workspace
    async def execute(self, i: BaseModel, c: Any = None) -> BaseModel:
        from rag.agent.primitive_ops import PrimitiveOps
        return PrimitiveOps(workspace=self._workspace).tool_repl(i)


# ═══════════════════════════════════════════════════════════════════════════════
# Module aggregate
# ═══════════════════════════════════════════════════════════════════════════════

WORKSPACE_TOOL_CLASSES: list[type[BaseTool]] = [
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
    RunPythonTool,
    StructuredProbeTool,
    SearchTextTool,
    ApplyPatchTool,
    RunCommandTool,
    ToolReplTool,
]


def create_workspace_tools(workspace: WorkspaceRuntime) -> list[BaseTool]:
    """Create tool instances with workspace injected (for registry registration)."""
    return [cls(workspace) for cls in WORKSPACE_TOOL_CLASSES]
