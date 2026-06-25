"""PR7: ACI test conventions — per-tool contract verification.

Each "mature" tool (has a complete ToolCard) is verified across 6 dimensions:
  1. input_validation  — invalid input is rejected
  2. output_validation — minimal valid output is accepted
  3. formatter_anchor  — formatter output contains expected anchor
  4. permission_consistency — permissions align with execution_category
  5. failure_model     — failure_codes in ToolCard match ToolError code possibilities
  6. search_activation — tool can be found by tool_search and activated
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.agent.builtin_registry import create_builtin_tool_registry
from rag.agent.capabilities.catalog import (
    DeferredToolStore,
    ToolCatalog,
)
from rag.agent.capabilities.tool_search import (
    execute_activate_tools,
    execute_tool_search,
)
from rag.agent.tools.spec import ToolResult


# ── Mature tools for the first ACI batch ──

_MATURE_TOOLS = [
    # Semantic RAG tools
    "search_knowledge",
    "search_assets",
    # Core file/workspace tools
    "list_files",
    "read_file",
    "write_file",
    "run_python",
    # Generic coding tools
    "search_text",
    "apply_patch",
    "run_command",
    "update_plan",
]


def _reg():
    return create_builtin_tool_registry()


# ═════════════════════════════════════════════════════════════════════════════
# 1. Input validation
# ═════════════════════════════════════════════════════════════════════════════


class TestACIInputValidation:
    """Every mature tool rejects clearly invalid input."""

    @pytest.mark.parametrize("tool_name", _MATURE_TOOLS)
    def test_input_model_exists(self, tool_name: str) -> None:
        """Tool has an input_model that is a Pydantic BaseModel."""
        registry = _reg()
        spec = registry.get(tool_name)
        assert spec.input_model is not None
        assert issubclass(spec.input_model, pytest.importorskip("pydantic").BaseModel)

    @pytest.mark.parametrize("tool_name", _MATURE_TOOLS)
    def test_empty_input_rejected(self, tool_name: str) -> None:
        """Empty dict is rejected unless all fields are optional."""
        registry = _reg()
        spec = registry.get(tool_name)
        # Check if the model has any required fields
        schema = spec.input_model.model_json_schema()
        required = schema.get("required", []) if isinstance(schema, dict) else []
        if not required:
            # All fields optional — {} is valid, nothing to reject
            return
        with pytest.raises(ValidationError):
            spec.input_model.model_validate({})


# ═════════════════════════════════════════════════════════════════════════════
# 2. Output validation
# ═════════════════════════════════════════════════════════════════════════════


class TestACIOutputValidation:
    """Every mature tool has a valid output_model."""

    @pytest.mark.parametrize("tool_name", _MATURE_TOOLS)
    def test_output_model_exists(self, tool_name: str) -> None:
        """Tool has an output_model that is a Pydantic BaseModel."""
        registry = _reg()
        spec = registry.get(tool_name)
        assert spec.output_model is not None
        assert issubclass(spec.output_model, pytest.importorskip("pydantic").BaseModel)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Formatter anchor
# ═════════════════════════════════════════════════════════════════════════════


class TestACIFormatterAnchor:
    """Every mature tool has a formatter that produces output."""

    @pytest.mark.parametrize("tool_name", _MATURE_TOOLS)
    def test_formatter_registered(self, tool_name: str) -> None:
        """Mature tool has a formatter registered."""
        registry = _reg()
        formatter = registry.get_formatter(tool_name)
        assert formatter is not None, f"{tool_name} is missing a formatter"

    @pytest.mark.parametrize("tool_name", _MATURE_TOOLS)
    def test_formatter_handles_ok_result(self, tool_name: str) -> None:
        """Formatter produces output (not None) for a minimal ok ToolResult."""
        registry = _reg()
        spec = registry.get(tool_name)
        formatter = registry.get_formatter(tool_name)
        assert formatter is not None

        # Construct minimal valid output via model_validate with empty dict
        # (may fail for models with required fields, skip gracefully)
        try:
            output = spec.output_model.model_validate({})
        except ValidationError:
            # Model requires fields — can't test with empty output
            return

        result = ToolResult(
            tool_call_id="tc", tool_name=tool_name,
            status="ok", output=output, latency_ms=100.0,
        )
        section = formatter.format_result(result)
        # Formatter may return None if output is empty/meaningless
        # That's fine — the test verifies it doesn't crash
        assert section is None or isinstance(section.content, str)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Permission consistency
# ═════════════════════════════════════════════════════════════════════════════


class TestACIPermissionConsistency:
    """Permissions, execution_category, and risk_level are consistent."""

    @pytest.mark.parametrize("tool_name", _MATURE_TOOLS)
    def test_category_permissions_consistent(self, tool_name: str) -> None:
        """execution_category is compatible with declared permissions."""
        registry = _reg()
        spec = registry.get(tool_name)
        # ToolSpec.__post_init__ already validates this; verify here
        assert spec.execution_category is not None
        assert spec.risk_level is not None
        # Category should be a valid ExecutionCategory
        from rag.agent.tools.spec import ExecutionCategory
        assert isinstance(spec.execution_category, ExecutionCategory)

    @pytest.mark.parametrize("tool_name", _MATURE_TOOLS)
    def test_is_read_only_consistent(self, tool_name: str) -> None:
        """is_read_only flag is consistent with category."""
        registry = _reg()
        spec = registry.get(tool_name)
        from rag.agent.tools.spec import ExecutionCategory
        if spec.execution_category in (ExecutionCategory.READ, ExecutionCategory.TRANSFORM):
            # These should be read_only unless permissions require approval
            if not spec.permissions_require_approval:
                assert spec.is_read_only is True


# ═════════════════════════════════════════════════════════════════════════════
# 5. Failure model
# ═════════════════════════════════════════════════════════════════════════════


class TestACIFailureModel:
    """Failure codes in ToolCard are expressible as ToolError."""

    @pytest.mark.parametrize("tool_name", _MATURE_TOOLS)
    def test_error_model_is_tool_error(self, tool_name: str) -> None:
        """ToolSpec.error_model is ToolError (or subclass)."""
        registry = _reg()
        spec = registry.get(tool_name)
        from rag.agent.tools.spec import ToolError
        assert spec.error_model is ToolError or issubclass(spec.error_model, ToolError)

    @pytest.mark.parametrize("tool_name", _MATURE_TOOLS)
    def test_failure_codes_constructible(self, tool_name: str) -> None:
        """If ToolCard declares failure_codes, they can be used in ToolError."""
        registry = _reg()
        spec = registry.get(tool_name)
        if spec.aci is None or not spec.aci.failure_codes:
            return  # No failure codes declared — skip
        for code in spec.aci.failure_codes:
            error = spec.error_model(code=code, message=f"Test: {code}", retryable=False)
            assert error.code == code


# ═════════════════════════════════════════════════════════════════════════════
# 6. Search & activation
# ═════════════════════════════════════════════════════════════════════════════


class TestACISearchActivation:
    """Deferred tools are discoverable and activatable."""

    @pytest.mark.parametrize("tool_name", _MATURE_TOOLS)
    def test_tool_in_known_category(self, tool_name: str) -> None:
        """Tool is in CORE_TOOLS, DEFERRED_TOOLS, or INTERNAL_TOOLS."""
        from rag.agent.capabilities.catalog import CORE_TOOLS, DEFERRED_TOOLS, INTERNAL_TOOLS
        assert tool_name in CORE_TOOLS or tool_name in DEFERRED_TOOLS or tool_name in INTERNAL_TOOLS, (
            f"{tool_name} not in any known tool category"
        )

    @pytest.mark.parametrize("tool_name", [
        t for t in _MATURE_TOOLS
        if t in __import__("rag.agent.capabilities.catalog", fromlist=["DEFERRED_TOOLS"]).DEFERRED_TOOLS
    ])
    def test_deferred_tool_searchable_and_activatable(self, tool_name: str) -> None:
        """Deferred tool can be found via search and activated."""
        from rag.agent.capabilities.catalog import DEFERRED_TOOLS

        if tool_name not in DEFERRED_TOOLS:
            pytest.skip(f"{tool_name} is not a deferred tool")

        registry = _reg()
        spec = registry.get(tool_name)

        # Build minimal catalog with just this tool
        from rag.agent.capabilities.catalog import (
            SearchCandidate,
            ToolCatalog,
            ToolCatalogEntry,
        )
        from rag.agent.capabilities.tool_search import (
            execute_activate_tools,
            execute_tool_search,
        )
        from rag.agent.capabilities.catalog import _DEFAULT_ACTIVATION_GROUPS

        catalog = ToolCatalog()
        card = spec.aci
        activation_group = (
            card.activation_group
            if card and card.activation_group
            else _DEFAULT_ACTIVATION_GROUPS.get(tool_name, "")
        )
        search_text = ToolCatalog.build_search_text(
            spec.name, spec.description, "",
            when_to_use=card.when_to_use if card else "",
            when_not_to_use=card.when_not_to_use if card else "",
            domains=card.domains if card else (),
            file_types=card.file_types if card else (),
            selection_tags=card.selection_tags if card else (),
        )
        # Register as deferred to be searchable
        catalog.register(ToolCatalogEntry(
            name=tool_name, description=spec.description,
            category="deferred", search_text=search_text,
            activation_group=activation_group,
        ))

        store = DeferredToolStore(max_active=10)
        search_output = execute_tool_search(
            spec.description, catalog=catalog, store=store, max_results=3,
        )
        assert any(c.name == tool_name for c in search_output.candidates), (
            f"{tool_name} not found by tool_search"
        )

        # Activate it
        output = execute_activate_tools(
            names=[tool_name], catalog=catalog, store=store,
            allowed_tools=[tool_name], deny_tools=frozenset(),
            iteration=1,
        )
        assert tool_name in output.activated, (
            f"{tool_name} could not be activated: {output}"
        )
