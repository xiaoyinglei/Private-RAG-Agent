"""Installed tool registry and phase-1 workspace tool registration."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag.agent.primitive_ops import (
    ListFilesInput,
    PrimitiveOps,
    ReadFileInput,
    RunPythonInput,
    WriteFileInput,
)
from rag.agent.tooling.spec import ToolDomain, ToolRisk, ToolSpec
from rag.agent.workspace import WorkspaceRuntime

ToolRunner = Callable[[dict[str, Any]], Any | Awaitable[Any]]


class SearchTextInput(BaseModel):
    pattern: str = Field(min_length=1, max_length=2000)
    path: str = "."
    file_types: str | None = None
    max_results: int = Field(default=40, ge=1, le=200)
    regex: bool = False
    context_lines: int = Field(default=0, ge=0, le=5)


class SearchTextMatch(BaseModel):
    file_path: str = ""
    line_number: int = 0
    line_content: str = ""
    match_start: int = 0
    match_end: int = 0


class SearchTextOutput(BaseModel):
    matches: list[SearchTextMatch] = Field(default_factory=list)
    total_matches: int = 0
    truncated: bool = False
    message: str = ""


class RunCommandInput(BaseModel):
    command: str = Field(min_length=1, max_length=4000)
    working_dir: str = "."
    timeout_seconds: int = Field(default=120, ge=1, le=600)
    env: dict[str, str] | None = None


class RunCommandOutput(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    truncated: bool = False
    duration_ms: float = 0.0


class ToolRegistry:
    """Installed tools only: spec plus runner lookup."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._runners: dict[str, ToolRunner] = {}

    def register(self, spec: ToolSpec, runner: ToolRunner) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        self._runners[spec.name] = runner

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def get_runner(self, name: str) -> ToolRunner | None:
        return self._runners.get(name)

    def list_specs(self) -> list[ToolSpec]:
        return list(self._specs.values())


class _RecoverableRunnerError(Exception):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def install_minimal_workspace_tools(
    registry: ToolRegistry,
    workspace: WorkspaceRuntime,
    *,
    allowed_commands: Iterable[str] | None = None,
    trusted_commands: bool = False,
) -> None:
    """Install the phase-1 workspace and execution tools into the registry."""

    ops = PrimitiveOps(workspace)
    allowed_command_set = frozenset(allowed_commands or ())

    registry.register(
        _tool_spec(
            "search_text",
            "Search text in workspace files.",
            SearchTextInput,
            ToolDomain.WORKSPACE,
            ToolRisk.READ,
            timeout_seconds=15.0,
        ),
        lambda args: _search_text(workspace, args),
    )
    registry.register(
        _tool_spec(
            "list_files",
            "List files under a workspace path.",
            ListFilesInput,
            ToolDomain.WORKSPACE,
            ToolRisk.READ,
            timeout_seconds=5.0,
        ),
        lambda args: ops.list_files(ListFilesInput(**args)),
    )
    registry.register(
        _tool_spec(
            "read_file",
            "Read a workspace file with bounded output.",
            ReadFileInput,
            ToolDomain.WORKSPACE,
            ToolRisk.READ,
            timeout_seconds=5.0,
        ),
        lambda args: ops.read_file(ReadFileInput(**args)),
    )
    registry.register(
        _tool_spec(
            "write_file",
            "Write a file to an approved workspace output directory.",
            WriteFileInput,
            ToolDomain.WORKSPACE,
            ToolRisk.WRITE,
            timeout_seconds=5.0,
        ),
        lambda args: ops.write_file(WriteFileInput(**args)),
    )
    registry.register(
        _tool_spec(
            "run_python",
            "Run Python code or a scratch Python script inside the workspace.",
            RunPythonInput,
            ToolDomain.EXECUTION,
            ToolRisk.EXECUTE,
            timeout_seconds=120.0,
            output_limit_chars=100_000,
        ),
        lambda args: ops.run_python(RunPythonInput(**args)),
    )
    registry.register(
        _tool_spec(
            "run_command",
            "Run an allowlisted command in the workspace.",
            RunCommandInput,
            ToolDomain.EXECUTION,
            ToolRisk.EXECUTE,
            timeout_seconds=120.0,
        ),
        lambda args: _run_command(
            workspace,
            args,
            allowed_commands=allowed_command_set,
            trusted_commands=trusted_commands,
        ),
    )


def _tool_spec(
    name: str,
    description: str,
    input_model: type[BaseModel],
    domain: ToolDomain,
    risk: ToolRisk,
    *,
    timeout_seconds: float,
    output_limit_chars: int = 50_000,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema=input_model.model_json_schema(),
        domain=domain,
        risk=risk,
        timeout_seconds=timeout_seconds,
        output_limit_chars=output_limit_chars,
    )


def _search_text(workspace: WorkspaceRuntime, args: dict[str, Any]) -> SearchTextOutput:
    inp = SearchTextInput(**args)
    target = workspace.resolve_path(inp.path)
    workspace.ensure_within_workspace(target)
    files = _searchable_files(target, inp.file_types)
    pattern = re.compile(inp.pattern) if inp.regex else None

    matches: list[SearchTextMatch] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern:
                found = pattern.search(line)
                if not found:
                    continue
                start, end = found.span()
            else:
                start = line.find(inp.pattern)
                if start < 0:
                    continue
                end = start + len(inp.pattern)
            matches.append(
                SearchTextMatch(
                    file_path=str(workspace.relative_to_root(path)),
                    line_number=line_no,
                    line_content=line,
                    match_start=start,
                    match_end=end,
                )
            )
            if len(matches) >= inp.max_results:
                return SearchTextOutput(
                    matches=matches,
                    total_matches=len(matches),
                    truncated=True,
                )
    return SearchTextOutput(matches=matches, total_matches=len(matches), truncated=False)


def _searchable_files(target: Path, file_types: str | None) -> list[Path]:
    allowed_suffixes = _suffixes(file_types)
    if target.is_file():
        return [target] if _suffix_allowed(target, allowed_suffixes) else []
    if not target.is_dir():
        return []
    return [
        path
        for path in sorted(target.rglob("*"))
        if path.is_file() and _suffix_allowed(path, allowed_suffixes)
    ]


def _suffixes(file_types: str | None) -> set[str]:
    if not file_types:
        return set()
    return {
        suffix if suffix.startswith(".") else f".{suffix}"
        for suffix in (part.strip() for part in file_types.split(","))
        if suffix
    }


def _suffix_allowed(path: Path, allowed_suffixes: set[str]) -> bool:
    return not allowed_suffixes or path.suffix in allowed_suffixes


def _run_command(
    workspace: WorkspaceRuntime,
    args: dict[str, Any],
    *,
    allowed_commands: frozenset[str],
    trusted_commands: bool,
) -> RunCommandOutput:
    inp = RunCommandInput(**args)
    cwd = workspace.resolve_path(inp.working_dir)
    workspace.ensure_within_workspace(cwd)
    if not cwd.is_dir():
        raise NotADirectoryError(f"Working directory not found: {inp.working_dir}")

    argv = shlex.split(inp.command)
    if not argv:
        raise ValueError("command must not be empty")
    command_name = Path(argv[0]).name
    if not trusted_commands and command_name not in allowed_commands:
        raise _RecoverableRunnerError(
            "command_not_allowed",
            f"Command is not allowlisted: {command_name}",
        )

    start = time.monotonic()
    env = os.environ.copy()
    if inp.env:
        env.update(inp.env)
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=inp.timeout_seconds,
            env=env,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        exit_code = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or "command timed out"
        exit_code = -1
        timed_out = True

    stdout_limited = stdout[:50_000]
    stderr_limited = stderr[:50_000]
    return RunCommandOutput(
        stdout=stdout_limited,
        stderr=stderr_limited,
        exit_code=exit_code,
        timed_out=timed_out,
        truncated=len(stdout) > len(stdout_limited) or len(stderr) > len(stderr_limited),
        duration_ms=(time.monotonic() - start) * 1000,
    )
