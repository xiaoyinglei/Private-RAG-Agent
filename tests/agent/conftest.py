"""Shared fixtures for agent contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def fake_sandbox_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Run process-lifecycle tests without depending on macOS Seatbelt.

    The wrapper only emulates sandbox-exec's argv contract. Tests that assert
    actual filesystem or network isolation must continue to use real Seatbelt.
    """

    from rag.agent.tools.builtins import shell as shell_module

    executable = tmp_path / "fake-sandbox-exec"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" != "-p" ] || [ "$#" -lt 3 ]; then\n'
        "  exit 64\n"
        "fi\n"
        "shift 2\n"
        'exec "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(
        shell_module,
        "_SANDBOX_EXEC_PATH",
        str(executable),
    )
    return executable
