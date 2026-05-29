from __future__ import annotations

import subprocess
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


class LocalSubprocessPythonRunner:
    """Default PythonRunner using subprocess.run."""

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


__all__ = [
    "LocalSubprocessPythonRunner",
    "PythonRunResult",
    "PythonRunner",
]
