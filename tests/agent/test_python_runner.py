from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag.agent.runner.python_runner import (
    LocalSubprocessPythonRunner,
    PythonRunResult,
    SeatbeltPythonRunner,
)


class TestPythonRunResult:
    def test_fields(self) -> None:
        result = PythonRunResult(exit_code=0, stdout="ok", stderr="", duration_ms=42.0)
        assert result.exit_code == 0
        assert result.stdout == "ok"
        assert result.stderr == ""
        assert result.duration_ms == 42.0


class TestLocalSubprocessPythonRunner:
    def test_runs_simple_script(self, tmp_path: Path) -> None:
        script = tmp_path / "hello.py"
        script.write_text("print('hello world')")

        runner = LocalSubprocessPythonRunner()
        result = runner.run(script, args=[], cwd=tmp_path, timeout=10.0)

        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.duration_ms > 0

    def test_captures_stderr(self, tmp_path: Path) -> None:
        script = tmp_path / "fail.py"
        script.write_text("import sys; print('oops', file=sys.stderr); exit(1)")

        runner = LocalSubprocessPythonRunner()
        result = runner.run(script, args=[], cwd=tmp_path, timeout=10.0)

        assert result.exit_code == 1
        assert "oops" in result.stderr

    def test_passes_args(self, tmp_path: Path) -> None:
        script = tmp_path / "args.py"
        script.write_text("import sys; print(' '.join(sys.argv[1:]))")

        runner = LocalSubprocessPythonRunner()
        result = runner.run(script, args=["hello", "world"], cwd=tmp_path, timeout=10.0)

        assert result.exit_code == 0
        assert "hello world" in result.stdout

    def test_timeout_returns_error_result(self, tmp_path: Path) -> None:
        script = tmp_path / "slow.py"
        script.write_text("import time; time.sleep(60)")

        runner = LocalSubprocessPythonRunner()
        result = runner.run(script, args=[], cwd=tmp_path, timeout=0.1)

        assert result.exit_code == -1
        assert "timed out" in result.stderr

    def test_stdout_truncation(self, tmp_path: Path) -> None:
        script = tmp_path / "noisy.py"
        script.write_text("print('x' * 200_000)")

        runner = LocalSubprocessPythonRunner()
        result = runner.run(script, args=[], cwd=tmp_path, timeout=10.0)

        assert len(result.stdout) <= 100_000


class TestSeatbeltPythonRunner:
    @pytest.mark.skipif(
        shutil.which("sandbox-exec") is None,
        reason="Seatbelt sandbox-exec is not available on this platform",
    )
    def test_blocks_network_access(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        for name in ("scratch", "artifacts", "reports", "logs", "input_files"):
            (workspace / name).mkdir(parents=True)
        script = workspace / "scratch" / "net.py"
        script.write_text(
            "import socket\n"
            "sock = socket.socket()\n"
            "sock.settimeout(0.2)\n"
            "sock.connect(('127.0.0.1', 9))\n"
        )

        runner = SeatbeltPythonRunner()
        result = runner.run(script, args=[], cwd=workspace, timeout=10.0)

        assert result.exit_code != 0
        assert "Operation not permitted" in result.stderr

    @pytest.mark.skipif(
        shutil.which("sandbox-exec") is None,
        reason="Seatbelt sandbox-exec is not available on this platform",
    )
    def test_blocks_reads_from_denied_root_except_workspace(self, tmp_path: Path) -> None:
        denied_root = tmp_path / "denied"
        workspace = denied_root / "workspace"
        for name in ("scratch", "artifacts", "reports", "logs", "input_files"):
            (workspace / name).mkdir(parents=True)
        (denied_root / "secret.txt").write_text("secret")
        (workspace / "input_files" / "data.csv").write_text("value\n1")
        script = workspace / "scratch" / "read_scope.py"
        script.write_text(
            "from pathlib import Path\n"
            "print(Path('input_files/data.csv').read_text())\n"
            "print(Path('../secret.txt').read_text())\n"
        )

        runner = SeatbeltPythonRunner(deny_read_roots=(denied_root,))
        result = runner.run(script, args=[], cwd=workspace, timeout=10.0)

        assert result.exit_code != 0
        assert "value" in result.stdout
        assert "Operation not permitted" in result.stderr
