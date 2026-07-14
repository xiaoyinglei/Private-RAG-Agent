from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tool_aci_cases.json"
SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "agent_tool_aci_eval.py"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "agent_tool_aci_eval",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_cases() -> list[dict[str, object]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    return payload["cases"]


def test_fixture_covers_the_locked_aci_case_matrix() -> None:
    cases = _fixture_cases()

    assert len(cases) == 12
    assert {case["category"] for case in cases} == {
        "direct_answer",
        "navigation",
        "grep",
        "read",
        "patch",
        "command",
        "knowledge",
        "hidden_mcp",
        "subagent",
        "hidden_hallucination",
        "similar_tool_confusion",
        "chinese_discovery",
    }
    assert len({case["id"] for case in cases}) == len(cases)
    hallucination = next(
        case for case in cases if case["id"] == "hidden_tool_hallucination"
    )
    assert hallucination["fake_recovery"] == {"tool": None}


@pytest.mark.anyio
async def test_fake_model_eval_reports_locked_metrics_without_thresholds() -> None:
    module = _load_eval_module()

    report = await module.run_evaluation(
        fixture_path=FIXTURE_PATH,
        fake_model=True,
    )

    assert report["mode"] == "fake-model"
    assert report["case_count"] == 12
    assert "thresholds" not in report
    assert set(report["metrics"]) == {
        "surface_recall",
        "surface_precision",
        "tool_choice_accuracy",
        "argument_validity",
        "unnecessary_call_rate",
        "discovery_recall_at_5",
        "recovery_rate",
        "schema_bytes",
        "schema_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cache_usage_source",
    }
    for metric_name in (
        "surface_recall",
        "surface_precision",
        "tool_choice_accuracy",
        "argument_validity",
        "unnecessary_call_rate",
        "discovery_recall_at_5",
        "recovery_rate",
    ):
        assert 0.0 <= report["metrics"][metric_name] <= 1.0

    assert report["metrics"]["surface_recall"] == 1.0
    assert report["metrics"]["surface_precision"] == 1.0
    assert report["metrics"]["tool_choice_accuracy"] == 11 / 12
    assert report["metrics"]["argument_validity"] == 10 / 11
    assert report["metrics"]["unnecessary_call_rate"] == 0.5
    assert report["metrics"]["discovery_recall_at_5"] == 1.0
    assert report["metrics"]["recovery_rate"] == 1.0
    assert report["metrics"]["schema_bytes"] > 0
    assert report["metrics"]["schema_tokens"] > 0
    assert report["metrics"]["cache_read_tokens"] == 7
    assert report["metrics"]["cache_write_tokens"] == 3
    assert report["metrics"]["cache_usage_source"] == "provider"
    assert len(report["cases"]) == 12


def test_fixture_rejects_quality_thresholds() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert "thresholds" not in payload
    assert all("threshold" not in case for case in payload["cases"])
