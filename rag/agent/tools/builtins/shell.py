from __future__ import annotations

import asyncio
import math
import os
import signal
import time
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    json_schema_output,
    pydantic_input,
)
from rag.agent.workspace import WorkspaceRuntime

_MAX_STREAM_BYTES = 50_000


class RunCommandInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(
        min_length=1,
        max_length=8000,
        description="Shell command to execute in an isolated process group.",
    )
    working_dir: str = Field(
        default=".",
        max_length=4096,
        description="Workspace-relative working directory.",
    )
    timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=600.0,
        description="Per-command timeout before the whole process group is terminated.",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables merged into the current environment.",
    )


class RunCommandOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    truncated: bool
    duration_ms: float = Field(ge=0)


_COMMAND_INPUT_SCHEMA, _validate_command_input = pydantic_input(RunCommandInput)
_COMMAND_OUTPUT_SCHEMA, _unused_command_output_validator = pydantic_input(
    RunCommandOutput
)


def create_run_command_tool(
    workspace: WorkspaceRuntime,
    *,
    hard_timeout_seconds: float = 605.0,
    termination_grace_seconds: float = 0.5,
) -> Tool:
    if (
        isinstance(termination_grace_seconds, bool)
        or not isinstance(termination_grace_seconds, (int, float))
        or not math.isfinite(termination_grace_seconds)
        or termination_grace_seconds <= 0
    ):
        raise ValueError("termination_grace_seconds must be positive and finite")

    async def run(arguments: Mapping[str, JsonValue]) -> RunCommandOutput:
        return await _run_command(
            workspace,
            RunCommandInput.model_validate(arguments),
            termination_grace_seconds=float(termination_grace_seconds),
        )

    return Tool(
        definition=ToolDefinition(
            name="run_command",
            description=(
                "Run one shell command from a workspace-relative directory. The command "
                "may read, write, or execute and therefore always passes permission and "
                "workspace guards. Timeout or cancellation terminates the complete process "
                "group, escalates to SIGKILL, and waits for reaping before returning."
            ),
            input_schema=_COMMAND_INPUT_SCHEMA,
        ),
        validate_input=_validate_command_input,
        run=run,
        normalize_output=_normalize_command_output,
        output_schema=_COMMAND_OUTPUT_SCHEMA,
        static_effects=frozenset(
            {
                ToolEffect.READ_WORKSPACE,
                ToolEffect.WRITE_WORKSPACE,
                ToolEffect.EXECUTE_PROCESS,
            }
        ),
        resolve_use=lambda arguments: _resolve_command_use(workspace, arguments),
        execution_revision="builtin-run-command-v1",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=hard_timeout_seconds,
        max_model_output_bytes=150_000,
    )


async def _run_command(
    workspace: WorkspaceRuntime,
    request: RunCommandInput,
    *,
    termination_grace_seconds: float,
) -> RunCommandOutput:
    cwd = workspace.ensure_within_workspace(
        workspace.resolve_path(request.working_dir or "."),
    )
    if not cwd.is_dir():
        raise NotADirectoryError(
            f"workspace working directory not found: {request.working_dir}"
        )

    environment = os.environ.copy()
    if request.env:
        environment.update(request.env)
    started_at = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        request.command,
        cwd=str(cwd),
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    communication = asyncio.create_task(process.communicate())
    timed_out = False
    try:
        done, _pending = await asyncio.wait(
            {communication},
            timeout=request.timeout_seconds,
        )
        if communication not in done:
            timed_out = True
            await _terminate_process_group(
                process,
                grace_seconds=termination_grace_seconds,
            )
        stdout_bytes, stderr_bytes = await communication
    except asyncio.CancelledError:
        await _terminate_process_group(
            process,
            grace_seconds=termination_grace_seconds,
        )
        try:
            await communication
        except Exception:
            pass
        raise

    stdout, stdout_truncated = _bounded_stream(stdout_bytes)
    stderr, stderr_truncated = _bounded_stream(stderr_bytes)
    if timed_out and not stderr:
        stderr = "command timed out and its process group was terminated"
    return RunCommandOutput(
        stdout=stdout,
        stderr=stderr,
        exit_code=process.returncode if process.returncode is not None else -1,
        timed_out=timed_out,
        truncated=stdout_truncated or stderr_truncated,
        duration_ms=(time.monotonic() - started_at) * 1000,
    )


async def _terminate_process_group(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float,
) -> None:
    process_group_id = process.pid
    _signal_process_group(process_group_id, signal.SIGTERM)
    if not await _wait_for_process_group_exit(
        process_group_id,
        timeout=grace_seconds,
    ):
        _signal_process_group(process_group_id, signal.SIGKILL)
    if process.returncode is None:
        await process.wait()
    if not await _wait_for_process_group_exit(
        process_group_id,
        timeout=grace_seconds,
    ):
        _signal_process_group(process_group_id, signal.SIGKILL)
        if not await _wait_for_process_group_exit(
            process_group_id,
            timeout=grace_seconds,
        ):
            raise RuntimeError("command process group did not terminate")


def _signal_process_group(process_group_id: int, value: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, value)
    except ProcessLookupError:
        pass


async def _wait_for_process_group_exit(
    process_group_id: int,
    *,
    timeout: float,
) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while _process_group_exists(process_group_id):
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.01, remaining))
    return True


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _bounded_stream(value: bytes) -> tuple[str, bool]:
    truncated = len(value) > _MAX_STREAM_BYTES
    bounded = value[:_MAX_STREAM_BYTES]
    return bounded.decode("utf-8", errors="replace"), truncated


def _resolve_command_use(
    workspace: WorkspaceRuntime,
    arguments: Mapping[str, JsonValue],
) -> ResolvedToolUse:
    cwd = workspace.resolve_path(str(arguments["working_dir"]) or ".").resolve()
    effects = frozenset(
        {
            ToolEffect.READ_WORKSPACE,
            ToolEffect.WRITE_WORKSPACE,
            ToolEffect.EXECUTE_PROCESS,
        }
    )
    return ResolvedToolUse(
        effects=effects,
        targets=(
            ToolTarget(kind="workspace_path", value=str(cwd)),
            ToolTarget(kind="cwd_path", value=str(cwd)),
        ),
    )


def _normalize_command_output(raw: object) -> NormalizedToolOutput:
    validated = RunCommandOutput.model_validate(raw)
    structured = json_schema_output(
        _COMMAND_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    if validated.timed_out:
        return NormalizedToolOutput(
            structured_content=structured,
            is_error=True,
            error_code="timeout_cancelled",
            error_message="command timed out and its process group was terminated",
            retryable=False,
        )
    return NormalizedToolOutput(structured_content=structured)


__all__ = [
    "RunCommandInput",
    "RunCommandOutput",
    "create_run_command_tool",
]
