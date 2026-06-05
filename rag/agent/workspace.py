"""Workspace runtime for agent file isolation and sandboxing."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


class WorkspacePathError(ValueError):
    """Path escapes workspace boundary."""


@dataclass
class WorkspaceRuntime:
    """Manages a workspace directory tree with isolated scratch/artifacts."""

    root: Path
    is_temporary: bool

    @property
    def input_files(self) -> Path:
        return self.root / "input_files"

    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def agent_memory(self) -> Path:
        return self.root / ".agent_memory"

    def initialize(self) -> None:
        """Create the standard workspace subdirectories."""
        for subdir in (
            self.input_files,
            self.scratch,
            self.artifacts,
            self.reports,
            self.logs,
            self.agent_memory,
        ):
            subdir.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, relative: str | Path) -> Path:
        """Resolve a relative path against the workspace root (returns absolute)."""
        resolved = (self.root / relative).resolve()
        return resolved

    def ensure_within_workspace(self, path: Path) -> Path:
        """Ensure a path is within the workspace root; raise if it escapes."""
        resolved = path.resolve()
        workspace_root = self.root.resolve()
        if not (resolved == workspace_root or str(resolved).startswith(str(workspace_root) + os.sep)):
            raise WorkspacePathError(f"Path {path} escapes workspace boundary {self.root}")
        return resolved

    def ensure_within_scratch(self, path: Path) -> Path:
        """Ensure a path is within scratch/; raise otherwise."""
        resolved = self.ensure_within_workspace(path)
        scratch_root = self.scratch.resolve()
        if not str(resolved).startswith(str(scratch_root) + os.sep):
            raise WorkspacePathError(f"Path {path} is not within scratch/ directory")
        return resolved

    def relative_to_root(self, path: Path) -> Path:
        """Return the path relative to workspace root (after validation)."""
        resolved = self.ensure_within_workspace(path)
        return resolved.relative_to(self.root.resolve())


def create_temp_workspace(prefix: str = "agent_run_") -> WorkspaceRuntime:
    """Create a temporary workspace directory and initialize it."""
    root = Path(tempfile.mkdtemp(prefix=prefix))
    ws = WorkspaceRuntime(root=root, is_temporary=True)
    ws.initialize()
    return ws


def open_workspace(path: str | Path, *, create: bool = False) -> WorkspaceRuntime:
    """Open an existing workspace directory, optionally creating it."""
    root = Path(path)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.exists():
        raise FileNotFoundError(f"Workspace path does not exist: {root}")
    ws = WorkspaceRuntime(root=root.resolve(), is_temporary=False)
    ws.initialize()
    return ws


def import_files(workspace: WorkspaceRuntime, sources: list[str | Path]) -> list[Path]:
    """Copy source files into the workspace input_files directory.

    Returns the list of destination paths in input_files.
    """
    imported: list[Path] = []
    for src in sources:
        src_path = Path(src)
        if src_path.is_dir():
            raise ValueError(f"Directory import not supported: {src_path}")
        if not src_path.is_file():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        dest = _unique_dest(workspace.input_files, src_path.name)
        shutil.copy2(src_path, dest)
        imported.append(dest)
    return imported


def _unique_dest(directory: Path, filename: str) -> Path:
    """Generate a unique destination path, appending __N on collision."""
    dest = directory / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}__{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


__all__ = [
    "WorkspacePathError",
    "WorkspaceRuntime",
    "create_temp_workspace",
    "import_files",
    "open_workspace",
]
