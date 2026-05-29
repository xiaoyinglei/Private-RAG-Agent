"""Primitive operations for agent workspace interactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag.agent.runner.python_runner import (
    LocalSubprocessPythonRunner,
    PythonRunner,
)
from rag.agent.workspace import WorkspaceRuntime

# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------

class ListFilesInput(BaseModel):
    path: str = ""
    pattern: str | None = None


class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    is_dir: bool
    modified_at: float


class ListFilesOutput(BaseModel):
    files: list[FileInfo]


class ReadFileInput(BaseModel):
    path: str
    encoding: str = "utf-8"
    max_bytes: int = 1_000_000


class ReadFileOutput(BaseModel):
    path: str
    content: str
    truncated: bool
    size_bytes: int
    is_binary: bool = False
    encoding: str = "utf-8"


class WriteFileInput(BaseModel):
    path: str
    content: str
    encoding: str = "utf-8"
    overwrite: bool = False


class WriteFileOutput(BaseModel):
    path: str
    size_bytes: int


class RunPythonInput(BaseModel):
    script_path: str
    args: list[str] = Field(default_factory=list)
    timeout_seconds: float = 30.0


class RunPythonOutput(BaseModel):
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    duration_ms: float
    generated_files: list[str]


# ---------------------------------------------------------------------------
# PrimitiveOps
# ---------------------------------------------------------------------------

class PrimitiveOps:
    _WRITABLE_DIRS = ("scratch", "artifacts", "reports", "logs")

    def __init__(
        self,
        workspace: WorkspaceRuntime,
        python_runner: PythonRunner | None = None,
    ) -> None:
        self._workspace = workspace
        self._python_runner = python_runner or LocalSubprocessPythonRunner()

    # -- list_files --------------------------------------------------------

    def list_files(self, payload: ListFilesInput) -> ListFilesOutput:
        base = (
            self._workspace.resolve_path(payload.path)
            if payload.path
            else self._workspace.root
        )
        self._workspace.ensure_within_workspace(base)

        if not base.is_dir():
            return ListFilesOutput(files=[])

        entries: list[FileInfo] = []
        for entry in sorted(base.iterdir()):
            if payload.pattern and not entry.match(payload.pattern):
                continue
            stat = entry.stat()
            rel = self._workspace.relative_to_root(entry)
            entries.append(
                FileInfo(
                    name=entry.name,
                    path=str(rel),
                    size=stat.st_size,
                    is_dir=entry.is_dir(),
                    modified_at=stat.st_mtime,
                )
            )
        return ListFilesOutput(files=entries)

    # -- read_file ---------------------------------------------------------

    def read_file(self, payload: ReadFileInput) -> ReadFileOutput:
        target = self._workspace.resolve_path(payload.path)
        self._workspace.ensure_within_workspace(target)

        if not target.is_file():
            raise FileNotFoundError(f"File not found: {payload.path}")

        size = target.stat().st_size
        raw = target.read_bytes()[: payload.max_bytes + 1]
        truncated = len(raw) > payload.max_bytes
        if truncated:
            raw = raw[: payload.max_bytes]
        content = raw.decode(payload.encoding, errors="replace")

        return ReadFileOutput(
            path=payload.path,
            content=content,
            truncated=truncated,
            size_bytes=size,
            encoding=payload.encoding,
        )

    # -- write_file --------------------------------------------------------

    def write_file(self, payload: WriteFileInput) -> WriteFileOutput:
        target = self._workspace.resolve_path(payload.path)
        self._workspace.ensure_within_workspace(target)

        rel = self._workspace.relative_to_root(target)
        top_dir = rel.parts[0] if rel.parts else ""
        if top_dir not in self._WRITABLE_DIRS:
            raise PermissionError(
                f"Cannot write to {top_dir or 'workspace root'}/: only "
                f"{', '.join(d + '/' for d in self._WRITABLE_DIRS)} are allowed"
            )

        if target.exists() and not payload.overwrite:
            raise FileExistsError(
                f"File already exists and overwrite=False: {payload.path}"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload.content, encoding=payload.encoding)

        return WriteFileOutput(
            path=payload.path,
            size_bytes=len(payload.content.encode(payload.encoding)),
        )

    # -- run_python --------------------------------------------------------

    def run_python(self, payload: RunPythonInput) -> RunPythonOutput:
        script_abs = self._workspace.resolve_path(payload.script_path)
        self._workspace.ensure_within_scratch(script_abs)

        if not script_abs.suffix == ".py":
            raise ValueError(f"Only .py files allowed: {payload.script_path}")

        if not script_abs.is_file():
            raise FileNotFoundError(f"Script not found: {payload.script_path}")

        before = _snapshot_files(self._workspace.root)

        result = self._python_runner.run(
            script_abs,
            args=payload.args,
            cwd=self._workspace.root,
            timeout=payload.timeout_seconds,
        )

        after = _snapshot_files(self._workspace.root)
        generated = sorted(after - before)

        return RunPythonOutput(
            ok=result.exit_code == 0,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            stdout_truncated=len(result.stdout) >= 100_000,
            stderr_truncated=len(result.stderr) >= 50_000,
            duration_ms=result.duration_ms,
            generated_files=[
                str(self._workspace.relative_to_root(Path(f))) for f in generated
            ],
        )

    # -- runners registry --------------------------------------------------

    def runners(self) -> dict[str, Any]:
        return {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "run_python": self.run_python,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file():
            files.add(str(path))
    return files


__all__ = [
    "FileInfo",
    "ListFilesInput",
    "ListFilesOutput",
    "PrimitiveOps",
    "ReadFileInput",
    "ReadFileOutput",
    "RunPythonInput",
    "RunPythonOutput",
    "WriteFileInput",
    "WriteFileOutput",
]
