"""Agent Tool SDK — prepended before user code in run_python.

This module provides the `tools` object available in agent-written Python code.
It is write-only: tools.declare() appends declarations to a JSONL file but does
NOT execute tools. The main loop reads the file after execution and processes
tool calls through ToolExecutionService.

Usage (in agent-written Python):
    # Declare tool calls (returns immediately, no execution)
    tools.declare("search_knowledge", query="Q3 revenue", top_k=5)
    tools.declare("search_text", pattern="TODO", path=".")

    # Code continues — do data processing with the results on the next turn
    import pandas as pd
    df = pd.read_csv("scratch/data.csv")
    print(df.describe())
"""

import json
import os
from pathlib import Path

# ── Configuration (set by PrimitiveOps before subprocess execution) ──

_SCRATCH_DIR: str | None = os.environ.get("AGENT_SCRATCH_DIR", None)
_BATCH_FILE: str | None = os.environ.get("AGENT_BATCH_FILE", None)
_MAX_BATCH_SIZE: int = int(os.environ.get("AGENT_MAX_BATCH_SIZE", "10"))


def _get_batch_path() -> str:
    """Get the path to the tool batch file."""
    if _BATCH_FILE:
        return _BATCH_FILE
    scratch = _SCRATCH_DIR or "."
    return os.path.join(scratch, "tool_calls.jsonl")


# ── Public API (injected into exec namespace as `tools`) ──


class _ToolDeclarer:
    """Minimal SDK injected as `tools` in agent-written Python code.

    Writes tool call declarations to a JSONL file. Returns immediately.
    The main agent loop reads the file after execution and processes
    all declarations through ToolExecutionService.
    """

    def __init__(self) -> None:
        self._count = 0

    def declare(self, name: str, **args: object) -> dict[str, object]:
        """Declare a tool call. Tool is NOT executed here.

        The declaration is appended to the batch file. The main loop
        reads it after this script finishes, validates against
        activated tools, and executes through ToolExecutionService.

        Returns a dict with the declared tool name and sequence number.
        """
        if self._count >= _MAX_BATCH_SIZE:
            return {
                "declared": False,
                "error": f"Batch size limit ({_MAX_BATCH_SIZE}) exceeded",
            }

        call: dict[str, object] = {"tool_name": name, "arguments": args}
        batch_path = _get_batch_path()

        try:
            # Ensure scratch directory exists
            Path(batch_path).parent.mkdir(parents=True, exist_ok=True)

            with open(batch_path, "a") as f:
                f.write(json.dumps(call, default=str) + "\n")

            self._count += 1
            return {"declared": name, "seq": self._count}
        except Exception as e:
            return {"declared": False, "error": str(e)}

    def list_available(self) -> list[str]:
        """List tools available for declaration.

        This does NOT know which tools are activated — it returns a
        static list from the environment. The main loop validates
        activation status at execution time.
        """
        raw = os.environ.get("AGENT_AVAILABLE_TOOLS", "")
        if raw:
            return [t.strip() for t in raw.split(",") if t.strip()]
        return []


# Singleton instance
tools = _ToolDeclarer()
