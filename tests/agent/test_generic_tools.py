"""Tests for the 4 generic coding-agent tools — spec, card, formatter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.generic_tools import (
    ALL_GENERIC_TOOLS,
    ApplyPatchInput,
    RunCommandInput,
    SearchTextInput,
    SearchTextOutput,
    UpdatePlanInput,
)
from rag.agent.tools.spec import ExecutionCategory, ToolResult


def _reg():
    return create_builtin_tool_registry()


_GENERIC_TOOL_NAMES = [s.name for s in ALL_GENERIC_TOOLS]


# ── Minimal output models for testing ──


class _ApplyPatchOutput(BaseModel):
    file_path: str = ""
    replaced: bool = False
    occurrences: int = 0
    message: str = ""


class _RunCommandOutput(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    truncated: bool = False
    duration_ms: float = 0.0


class _PlanStep(BaseModel):
    id: str = ""
    description: str = ""
    status: str = "pending"


class _UpdatePlanOutput(BaseModel):
    steps: list = []
    summary: str = ""
    message: str = ""


class TestGenericToolsRegistered:
    """All 4 tools are registered with spec, card, and formatter."""

    @pytest.mark.parametrize("tool_name", _GENERIC_TOOL_NAMES)
    def test_spec_registered(self, tool_name: str) -> None:
        registry = _reg()
        spec = registry.get(tool_name)
        assert spec is not None

    @pytest.mark.parametrize("tool_name", _GENERIC_TOOL_NAMES)
    def test_toolcard_populated(self, tool_name: str) -> None:
        registry = _reg()
        spec = registry.get(tool_name)
        assert spec.aci is not None, f"{tool_name} missing ToolCard"
        assert spec.aci.when_to_use, f"{tool_name} ToolCard.when_to_use is empty"
        assert spec.aci.activation_group, f"{tool_name} ToolCard.activation_group is empty"

    @pytest.mark.parametrize("tool_name", _GENERIC_TOOL_NAMES)
    def test_formatter_registered(self, tool_name: str) -> None:
        registry = _reg()
        formatter = registry.get_formatter(tool_name)
        assert formatter is not None, f"{tool_name} missing formatter"


class TestSearchText:
    """search_text tool spec and IO."""

    def test_valid_input(self) -> None:
        inp = SearchTextInput.model_validate({"pattern": "def test"})
        assert inp.pattern == "def test"
        assert inp.regex is False
        assert inp.max_results == 40

    def test_regex_input(self) -> None:
        inp = SearchTextInput.model_validate(
            {"pattern": r"import\s+\w+", "regex": True, "file_types": ".py", "max_results": 10}
        )
        assert inp.regex is True
        assert inp.file_types == ".py"

    def test_empty_pattern_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchTextInput.model_validate({"pattern": ""})

    def test_spec_category(self) -> None:
        spec = _reg().get("search_text")
        assert spec.execution_category == ExecutionCategory.READ
        assert spec.is_read_only is True

    def test_formatter_renders_matches(self) -> None:
        registry = _reg()
        fmt = registry.get_formatter("search_text")
        output = SearchTextOutput(
            matches=[{"file_path": "a.py", "line_number": 1, "line_content": "def foo():"}],
            total_matches=1,
        )
        result = ToolResult(
            tool_call_id="tc", tool_name="search_text",
            status="ok", output=output, latency_ms=10.0,
        )
        section = fmt.format_result(result)
        assert section is not None
        assert "a.py:1" in section.content
        assert "def foo()" in section.content


class TestApplyPatch:
    """apply_patch tool spec and IO."""

    def test_valid_input(self) -> None:
        inp = ApplyPatchInput.model_validate({
            "file_path": "src/app.py",
            "old_string": "debug = False",
            "new_string": "debug = True",
        })
        assert inp.file_path == "src/app.py"
        assert inp.replace_all is False

    def test_replace_all(self) -> None:
        inp = ApplyPatchInput.model_validate({
            "file_path": "config.toml",
            "old_string": "version",
            "new_string": "ver",
            "replace_all": True,
        })
        assert inp.replace_all is True

    def test_empty_old_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApplyPatchInput.model_validate({
                "file_path": "f.py", "old_string": "", "new_string": "x",
            })

    def test_spec_category(self) -> None:
        spec = _reg().get("apply_patch")
        assert spec.execution_category == ExecutionCategory.WRITE

    def test_toolcard_failure_codes(self) -> None:
        spec = _reg().get("apply_patch")
        assert "not_unique" in spec.aci.failure_codes
        assert "no_match" in spec.aci.failure_codes

    def test_formatter_renders_result(self) -> None:
        registry = _reg()
        fmt = registry.get_formatter("apply_patch")
        output = _ApplyPatchOutput(
            file_path="a.py", replaced=True, occurrences=1,
        )
        result = ToolResult(
            tool_call_id="tc", tool_name="apply_patch",
            status="ok", output=output, latency_ms=5.0,
        )
        section = fmt.format_result(result)
        assert section is not None
        assert "a.py" in section.content
        assert "replaced=True" in section.content


class TestRunCommand:
    """run_command tool spec and IO."""

    def test_valid_input(self) -> None:
        inp = RunCommandInput.model_validate({"command": "pytest tests/ -x -q"})
        assert inp.command == "pytest tests/ -x -q"
        assert inp.timeout_seconds == 120
        assert inp.working_dir == "."

    def test_empty_command_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunCommandInput.model_validate({"command": ""})

    def test_spec_category(self) -> None:
        spec = _reg().get("run_command")
        assert spec.execution_category == ExecutionCategory.EXECUTE
        assert spec.permissions.execute_code is True

    def test_toolcard_activation_group(self) -> None:
        spec = _reg().get("run_command")
        assert spec.aci.activation_group == "workspace"

    def test_formatter_renders_output(self) -> None:
        registry = _reg()
        fmt = registry.get_formatter("run_command")
        output = _RunCommandOutput(
            stdout="3 passed", stderr="", exit_code=0, duration_ms=1500.0,
        )
        result = ToolResult(
            tool_call_id="tc", tool_name="run_command",
            status="ok", output=output, latency_ms=1500.0,
        )
        section = fmt.format_result(result)
        assert section is not None
        assert "exit_code=0" in section.content
        assert "3 passed" in section.content


class TestUpdatePlan:
    """update_plan tool spec and IO."""

    def test_add_steps(self) -> None:
        inp = UpdatePlanInput.model_validate({
            "action": "add",
            "steps": [
                {"description": "Fix auth bug", "status": "in_progress"},
                {"description": "Run tests", "status": "pending"},
            ],
        })
        assert inp.action == "add"
        assert len(inp.steps) == 2
        assert inp.steps[0].status == "in_progress"

    def test_complete_steps(self) -> None:
        inp = UpdatePlanInput.model_validate({
            "action": "complete",
            "step_ids": ["step-1", "step-2"],
            "summary": "Auth bug fixed, running tests",
        })
        assert inp.action == "complete"
        assert inp.summary == "Auth bug fixed, running tests"

    def test_invalid_action_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UpdatePlanInput.model_validate({"action": "delete"})

    def test_spec_category(self) -> None:
        spec = _reg().get("update_plan")
        assert spec.execution_category == ExecutionCategory.TRANSFORM
        assert spec.is_read_only is True

    def test_toolcard_resident(self) -> None:
        spec = _reg().get("update_plan")
        assert spec.aci.activation_group == "resident"

    def test_formatter_renders_plan(self) -> None:
        registry = _reg()
        fmt = registry.get_formatter("update_plan")
        output = _UpdatePlanOutput(
            steps=[
                _PlanStep(id="step-1", description="Fix auth", status="completed"),
                _PlanStep(id="step-2", description="Run tests", status="in_progress"),
            ],
            summary="Working on tests",
        )
        result = ToolResult(
            tool_call_id="tc", tool_name="update_plan",
            status="ok", output=output, latency_ms=1.0,
        )
        section = fmt.format_result(result)
        assert section is not None
        assert "Fix auth" in section.content
        assert "completed" in section.content
        assert "in_progress" in section.content
