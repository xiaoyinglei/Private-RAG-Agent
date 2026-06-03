from __future__ import annotations

from pathlib import Path

from rag.agent.runner.python_runner import (
    LocalSubprocessPythonRunner,
    PythonRunResult,
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
