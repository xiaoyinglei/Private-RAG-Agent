"""Tests for generic tool I/O models and the update_plan spec."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.generic_tools import (
    SearchTextInput,
    SearchTextOutput,
    ApplyPatchInput,
    RunCommandInput,
    UpdatePlanInput,
)


def _reg():
    return create_builtin_tool_registry()


# ═══════════════════════════════════════════════════════════════════════════════
# I/O model validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchText:
    def test_valid_input(self) -> None:
        inp = SearchTextInput.model_validate({"pattern": "def test"})
        assert inp.pattern == "def test"
        assert inp.regex is False

    def test_regex_input(self) -> None:
        inp = SearchTextInput.model_validate(
            {"pattern": r"import\s+\w+", "regex": True, "file_types": ".py", "max_results": 10}
        )
        assert inp.regex is True

    def test_empty_pattern_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchTextInput.model_validate({"pattern": ""})


class TestApplyPatch:
    def test_valid_input(self) -> None:
        inp = ApplyPatchInput.model_validate({
            "file_path": "src/app.py",
            "old_string": "debug = False",
            "new_string": "debug = True",
        })
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


class TestRunCommand:
    def test_valid_input(self) -> None:
        inp = RunCommandInput.model_validate({"command": "pytest tests/ -x -q"})
        assert inp.command == "pytest tests/ -x -q"
        assert inp.timeout_seconds == 120

    def test_empty_command_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunCommandInput.model_validate({"command": ""})


class TestUpdatePlan:
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

    def test_complete_steps(self) -> None:
        inp = UpdatePlanInput.model_validate({
            "action": "complete",
            "step_ids": ["step-1", "step-2"],
            "summary": "Auth bug fixed",
        })
        assert inp.action == "complete"

    def test_invalid_action_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UpdatePlanInput.model_validate({"action": "delete"})

    def test_update_plan_registered(self) -> None:
        spec = _reg().get("update_plan")
        assert spec is not None
        assert spec.aci.activation_group == "resident"
