from __future__ import annotations

import os
import shutil
import site
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class PythonRunResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float


class PythonRunner(Protocol):
    def run(
        self,
        script_path: Path,
        *,
        args: list[str],
        cwd: Path,
        timeout: float,
    ) -> PythonRunResult: ...


STDOUT_MAX_BYTES = 100_000
STDERR_MAX_BYTES = 50_000
DEFAULT_WRITABLE_DIRS = ("scratch", "artifacts", "reports", "logs")


class LocalSubprocessPythonRunner:
    """Trusted-only PythonRunner using subprocess.run without OS sandboxing."""

    def run(
        self,
        script_path: Path,
        *,
        args: list[str],
        cwd: Path,
        timeout: float,
    ) -> PythonRunResult:
        cmd = ["python", str(script_path), *args]
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                timeout=timeout,
            )
            elapsed = (time.monotonic() - start) * 1000
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            return PythonRunResult(
                exit_code=proc.returncode,
                stdout=stdout[:STDOUT_MAX_BYTES],
                stderr=stderr[:STDERR_MAX_BYTES],
                duration_ms=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - start) * 1000
            return PythonRunResult(
                exit_code=-1,
                stdout="",
                stderr=f"Script timed out after {timeout}s",
                duration_ms=elapsed,
            )


class SeatbeltPythonRunner:
    """PythonRunner constrained by macOS Seatbelt via sandbox-exec."""

    def __init__(
        self,
        *,
        sandbox_exec_path: str | None = None,
        python_executable: str | None = None,
        writable_dirs: tuple[str, ...] = DEFAULT_WRITABLE_DIRS,
        deny_read_roots: tuple[Path, ...] | None = None,
    ) -> None:
        self._sandbox_exec_path = sandbox_exec_path or shutil.which("sandbox-exec")
        self._python_executable = python_executable or sys.executable
        self._writable_dirs = writable_dirs
        self._deny_read_roots = (
            tuple(path.resolve() for path in deny_read_roots)
            if deny_read_roots is not None
            else (Path.home().resolve(),)
        )

    def run(
        self,
        script_path: Path,
        *,
        args: list[str],
        cwd: Path,
        timeout: float,
    ) -> PythonRunResult:
        start = time.monotonic()
        if self._sandbox_exec_path is None:
            elapsed = (time.monotonic() - start) * 1000
            return PythonRunResult(
                exit_code=-1,
                stdout="",
                stderr="Seatbelt sandbox-exec is not available; refusing unsandboxed Python execution",
                duration_ms=elapsed,
            )

        workspace_root = cwd.resolve()
        writable_paths = tuple((workspace_root / name).resolve() for name in self._writable_dirs)
        readable_paths = _python_readable_paths(
            workspace_root=workspace_root,
            python_executable=Path(self._python_executable),
        )
        profile = _build_seatbelt_profile(
            writable_paths=writable_paths,
            deny_read_roots=self._deny_read_roots,
            readable_paths=readable_paths,
        )
        cmd = [
            self._sandbox_exec_path,
            "-p",
            profile,
            self._python_executable,
            str(script_path),
            *args,
        ]
        scratch = workspace_root / "scratch"
        env = {
            **os.environ,
            "HOME": str(scratch),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TEMP": str(scratch),
            "TMP": str(scratch),
            "TMPDIR": str(scratch),
        }

        try:
            proc = subprocess.run(
                cmd,
                cwd=workspace_root,
                env=env,
                capture_output=True,
                timeout=timeout,
            )
            elapsed = (time.monotonic() - start) * 1000
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            return PythonRunResult(
                exit_code=proc.returncode,
                stdout=stdout[:STDOUT_MAX_BYTES],
                stderr=stderr[:STDERR_MAX_BYTES],
                duration_ms=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - start) * 1000
            return PythonRunResult(
                exit_code=-1,
                stdout="",
                stderr=f"Script timed out after {timeout}s",
                duration_ms=elapsed,
            )


def _build_seatbelt_profile(
    *,
    writable_paths: tuple[Path, ...],
    deny_read_roots: tuple[Path, ...],
    readable_paths: tuple[Path, ...],
) -> str:
    deny_read_rules = "\n".join(
        f'  (subpath "{_escape_seatbelt_string(str(path))}")'
        for path in deny_read_roots
    )
    read_rules = "\n".join(_seatbelt_path_filter(path) for path in readable_paths)
    write_rules = "\n".join(
        f'  (subpath "{_escape_seatbelt_string(str(path))}")'
        for path in writable_paths
    )
    return (
        "(version 1)\n"
        "(allow default)\n"
        "(deny network*)\n"
        "(deny file-read*\n"
        f"{deny_read_rules})\n"
        "(allow file-read*\n"
        f"{read_rules})\n"
        "(deny file-write*)\n"
        "(allow file-write*\n"
        f"{write_rules})"
    )


def _python_readable_paths(*, workspace_root: Path, python_executable: Path) -> tuple[Path, ...]:
    paths = {
        workspace_root.resolve(),
        python_executable.resolve(),
    }
    for raw in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        paths.add(Path(raw).resolve())
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        value = sysconfig.get_path(key)
        if value:
            paths.add(Path(value).resolve())
    for raw in site.getsitepackages():
        paths.add(Path(raw).resolve())
    return tuple(sorted(paths))


def _seatbelt_path_filter(path: Path) -> str:
    kind = "literal" if path.is_file() else "subpath"
    return f'  ({kind} "{_escape_seatbelt_string(str(path))}")'


def _escape_seatbelt_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


__all__ = [
    "DEFAULT_WRITABLE_DIRS",
    "LocalSubprocessPythonRunner",
    "PythonRunResult",
    "PythonRunner",
    "SeatbeltPythonRunner",
]
