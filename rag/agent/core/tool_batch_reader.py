"""Tool batch reader — validates and converts JSONL declarations to ToolCallPlan.

The main loop reads scratch/tool_calls.jsonl after run_python finishes.
Each line is validated:
1. Tool is activated (in CORE_TOOLS or DeferredToolStore)
2. Tool is in allowed_tools
3. Arguments pass input_model.model_validate()
4. Batch size is within limits
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag.agent.capabilities.catalog import CORE_TOOLS, DeferredToolStore, ToolCatalog
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 10


@dataclass
class ToolBatchDeclaration:
    """A single tool call declaration from the JSONL batch file."""

    tool_name: str
    arguments: dict[str, Any]
    line_number: int = 0


@dataclass
class ToolBatchResult:
    """Result of processing a tool batch file."""

    plans: list[ToolCallPlan] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    tool_results: list[tuple[str, str]] = field(default_factory=list)
    # (tool_name, error_message) for declarations that failed validation


def read_tool_batch(
    scratch_dir: Path | str,
    *,
    max_batch: int = MAX_BATCH_SIZE,
) -> list[ToolBatchDeclaration]:
    """Read tool_calls.jsonl from scratch directory.

    Returns list of declarations, empty list if no file exists.
    """
    batch_file = Path(scratch_dir) / "tool_calls.jsonl"
    if not batch_file.exists():
        return []

    declarations: list[ToolBatchDeclaration] = []
    try:
        with open(batch_file) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in tool_calls.jsonl line {lineno}: {e}")
                    continue
                if not isinstance(obj, dict):
                    continue
                tool_name = obj.get("tool_name", "")
                arguments = obj.get("arguments", {})
                if isinstance(tool_name, str) and isinstance(arguments, dict):
                    declarations.append(
                        ToolBatchDeclaration(
                            tool_name=str(tool_name),
                            arguments=arguments,
                            line_number=lineno,
                        )
                    )
    except Exception as e:
        logger.error(f"Error reading tool_calls.jsonl: {e}")

    return declarations[:max_batch]


def validate_declaration(
    decl: ToolBatchDeclaration,
    *,
    catalog: ToolCatalog,
    store: DeferredToolStore,
    allowed_tools: list[str] | frozenset[str],
    registry: ToolRegistry,
) -> ToolCallPlan | str:
    """Validate one tool call declaration. Returns ToolCallPlan or error message.

    Checks:
    1. Tool is activated: either in CORE_TOOLS or active in DeferredToolStore
    2. Tool is in allowed_tools
    3. Arguments pass input_model.model_validate()
    """
    name = decl.tool_name

    # Check allowed_tools
    allowed = set(allowed_tools) if isinstance(allowed_tools, list) else allowed_tools
    if name not in allowed:
        return f"tool '{name}' not in allowed_tools"

    # Check activation
    if name not in CORE_TOOLS and not store.is_active(name):
        return f"tool '{name}' is not activated — use tool_search + activate_tools first"

    # Get spec and validate input
    try:
        spec = registry.get(name)
    except KeyError:
        return f"tool '{name}' is not registered"

    try:
        validated = spec.input_model(**decl.arguments)
        validated_args = validated.model_dump(exclude_none=True)
    except ValidationError as e:
        return f"invalid arguments for '{name}': {e}"
    except Exception as e:
        return f"error validating arguments for '{name}': {e}"

    return ToolCallPlan(
        tool_call_id=f"batch__{name}__{decl.line_number}",
        tool_name=name,
        arguments=validated_args,
    )


def clean_batch_file(scratch_dir: Path | str) -> None:
    """Delete tool_calls.jsonl after processing."""
    batch_file = Path(scratch_dir) / "tool_calls.jsonl"
    try:
        batch_file.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to clean up tool_calls.jsonl", exc_info=True)
