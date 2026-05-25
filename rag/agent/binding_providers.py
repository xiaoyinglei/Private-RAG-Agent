from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from rag.agent.goal_runtime import ContextBinding, ContextUnit, GoalConstraint


class AssetContextBindingProvider:
    """Check whether an answer is bound to the explicitly requested asset scope."""

    def assess_bindings(
        self,
        *,
        state: dict[str, Any],
        constraints: Sequence[GoalConstraint],
        context_units: Sequence[ContextUnit],
    ) -> list[ContextBinding]:
        selected_asset_ids = _latest_answer_asset_ids(state)
        bindings: list[ContextBinding] = []
        for constraint in constraints:
            if not constraint.required or constraint.constraint_type != "context_title":
                continue
            asset_units = [
                unit
                for unit in context_units
                if unit.unit_type in {"table_asset", "image_asset", "document_asset"}
                and isinstance(unit.locator.get("sheet_name"), str)
            ]
            selected_units = [
                unit
                for unit in asset_units
                if unit.locator.get("asset_id") in selected_asset_ids
            ]
            if selected_units:
                selected = selected_units[-1]
                matched = _matches_context_title(selected, constraint)
                bindings.append(
                    ContextBinding(
                        binding_id=constraint.constraint_id,
                        constraint_id=constraint.constraint_id,
                        unit_id=selected.unit_id,
                        status="satisfied" if matched else "violated",
                        evidence_refs=list(selected.evidence_refs),
                        rationale=(
                            "Computed result is bound to the requested context title."
                            if matched
                            else (
                                "Computed result came from an asset whose sheet name "
                                "does not match the requested source."
                            )
                        ),
                    )
                )
                continue
            matching_units = [
                unit for unit in asset_units if _matches_context_title(unit, constraint)
            ]
            if len(matching_units) == 1:
                unit = matching_units[0]
                bindings.append(
                    ContextBinding(
                        binding_id=constraint.constraint_id,
                        constraint_id=constraint.constraint_id,
                        unit_id=unit.unit_id,
                        status="satisfied",
                        evidence_refs=list(unit.evidence_refs),
                        rationale="Asset sheet name matches the requested context title.",
                    )
                )
            elif len(matching_units) > 1:
                bindings.append(
                    ContextBinding(
                        binding_id=constraint.constraint_id,
                        constraint_id=constraint.constraint_id,
                        status="ambiguous",
                        rationale="Multiple assets match the requested context title.",
                    )
                )
        return bindings


def _matches_context_title(unit: ContextUnit, constraint: GoalConstraint) -> bool:
    expected = constraint.expected_value
    actual = unit.locator.get("sheet_name")
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    return _normalize_text(expected) == _normalize_text(actual)


def _latest_answer_asset_ids(state: dict[str, Any]) -> set[int]:
    for candidate in reversed(state.get("answer_candidates", [])):
        asset_ids = {
            int(evidence_id.split(":", maxsplit=1)[1])
            for ref in getattr(candidate, "evidence_refs", []) or []
            if isinstance((evidence_id := getattr(ref, "evidence_id", None)), str)
            and evidence_id.startswith("asset:")
            and evidence_id.split(":", maxsplit=1)[1].isdigit()
        }
        if asset_ids:
            return asset_ids
    return set()


def _normalize_text(value: str) -> str:
    return re.sub(r"[\s_：:（）()]+", "", value)


__all__ = ["AssetContextBindingProvider"]
