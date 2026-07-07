#!/usr/bin/env python
"""Run a live delivery smoke matrix for the product agent path."""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "groq_gpt_oss_120b"


@dataclass(frozen=True)
class SmokeCase:
    name: str
    task: str
    expected_answer_contains: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    input_files: dict[str, str] = field(default_factory=dict)
    workspace_path: str | None = None
    workspace_assertions: dict[str, str] = field(default_factory=dict)
    tool_surface_request: dict[str, object] | None = None
    auto_approve: bool = False
    max_turns: int = 12


@dataclass(frozen=True)
class SmokeResult:
    name: str
    passed: bool
    status: str
    answer: str | None
    tools: tuple[str, ...]
    workspace_path: str | None
    error: str = ""
    stop_reason: str | None = None
    diagnostics: tuple[str, ...] = ()
    schema_bytes: int | None = None
    tool_errors: tuple[str, ...] = ()


def build_cases() -> tuple[SmokeCase, ...]:
    repo_root = str(Path(__file__).parents[1])
    return (
        SmokeCase(
            name="math_2_plus_2",
            task="What is 2+2? Answer with exactly the number.",
            expected_answer_contains=("4",),
            forbidden_tools=("tool_search", "activate_tools"),
            max_turns=10,
        ),
        SmokeCase(
            name="explain_recursion",
            task="Explain recursion in one concise English sentence and include the word recursion.",
            expected_answer_contains=("recursion",),
            forbidden_tools=("tool_search", "activate_tools"),
            max_turns=10,
        ),
        SmokeCase(
            name="find_agent_service",
            task=(
                "Use the search_text tool once to search for the exact text "
                "`class AgentService` under path `rag/agent/service.py`. "
                "Do not call read_file. Answer with exactly the file path."
            ),
            expected_answer_contains=("rag/agent/service.py",),
            expected_tools=("search_text",),
            tool_surface_request={
                "requested_tool_names": ["search_text", "list_files", "read_file"],
            },
            workspace_path=repo_root,
            max_turns=4,
        ),
        SmokeCase(
            name="read_missing_file",
            task=(
                "Read does-not-exist-agent-smoke.txt. If the file is missing, "
                "answer exactly file_not_found."
            ),
            expected_answer_contains=("file_not_found",),
            expected_tools=("read_file",),
            tool_surface_request={
                "requested_tool_names": ["read_file"],
            },
            max_turns=8,
        ),
        SmokeCase(
            name="echo_hello",
            task="Run echo hello and answer with exactly the stdout value.",
            expected_answer_contains=("hello",),
            expected_tools=("run_command",),
            tool_surface_request={
                "requested_tool_names": ["run_command"],
                "allow_execute_tools": True,
            },
            max_turns=8,
        ),
    )


async def run_case(case: SmokeCase, *, model: str) -> SmokeResult:
    from agent_runtime.models import ModelControlPlane
    from agent_runtime.runtime.builder import build_agent_service
    from rag.agent.core.human_input import HumanInputResponse
    from rag.agent.service import AgentRunRequest
    from rag.agent.tooling import ToolSurfaceRequest

    run_id = f"delivery_smoke_{case.name}"
    service = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        service = build_agent_service(
            None,
            agent_type="generic",
            model_alias=model,
            model_control_plane=ModelControlPlane.from_env(initial_model_id=model),
        )
        temp_dir = tempfile.TemporaryDirectory(prefix=f"{run_id}_")
        input_paths: list[str] = []
        for filename, content in case.input_files.items():
            path = Path(temp_dir.name) / filename
            path.write_text(content, encoding="utf-8")
            input_paths.append(str(path))

        result = await service.run(
            AgentRunRequest(
                task=case.task,
                run_id=run_id,
                thread_id=run_id,
                input_files=input_paths,
                workspace_path=case.workspace_path,
                max_turns=case.max_turns,
                tool_surface_request=(
                    ToolSurfaceRequest.model_validate(case.tool_surface_request)
                    if case.tool_surface_request is not None
                    else None
                ),
            )
        )
        approvals = 0
        while result.status == "paused" and case.auto_approve and approvals < 5:
            request = service.pending_human_input_request(run_id=run_id)
            if request.kind != "tool_approval":
                break
            result = await service.resume(
                run_id=run_id,
                workspace_path=result.workspace_path,
                response=HumanInputResponse(
                    request_id=request.request_id,
                    decision="allow_once",
                    approved_tool_call_ids=[tc.tool_call_id for tc in request.tool_calls],
                ),
            )
            approvals += 1

        tools = tuple(tool.tool_name for tool in result.tool_results)
        error = _validate_result(case, result.status, result.final_answer, tools, result.workspace_path)
        return SmokeResult(
            name=case.name,
            passed=not error,
            status=result.status,
            answer=result.final_answer,
            tools=tools,
            workspace_path=result.workspace_path,
            error=error,
            stop_reason=result.stop_reason,
            diagnostics=_diagnostic_lines(result.runtime_diagnostics),
            schema_bytes=(
                result.latency_profile.tool_schema_bytes
                if result.latency_profile is not None
                else None
            ),
            tool_errors=_tool_error_lines(result.tool_results),
        )
    except Exception as exc:
        return SmokeResult(
            name=case.name,
            passed=False,
            status="error",
            answer=None,
            tools=(),
            workspace_path=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        if service is not None:
            close_method = getattr(service, "aclose", None)
            if callable(close_method):
                await close_method()


def _validate_result(
    case: SmokeCase,
    status: str,
    answer: str | None,
    tools: tuple[str, ...],
    workspace_path: str | None,
) -> str:
    if status != "done":
        return f"expected status done, got {status}"
    answer_text = answer or ""
    normalized_answer = answer_text.casefold()
    for expected in case.expected_answer_contains:
        if expected.casefold() not in normalized_answer:
            return f"answer missing {expected!r}: {answer_text!r}"
    for tool_name in case.expected_tools:
        if tool_name not in tools:
            return f"expected tool {tool_name!r}, got {tools!r}"
    for tool_name in case.forbidden_tools:
        if tool_name in tools:
            return f"forbidden tool {tool_name!r} was called: {tools!r}"
    if case.workspace_assertions:
        if workspace_path is None:
            return "workspace assertions require workspace_path"
        root = Path(workspace_path)
        for rel_path, expected_content in case.workspace_assertions.items():
            path = root / rel_path
            if not path.exists():
                return f"missing workspace file {rel_path!r}"
            content = path.read_text(encoding="utf-8")
            if content != expected_content:
                return f"{rel_path!r} content mismatch: {content!r}"
    return ""


def _diagnostic_lines(diagnostics: object) -> tuple[str, ...]:
    lines: list[str] = []
    for diagnostic in diagnostics or ():
        code = str(getattr(diagnostic, "code", "diagnostic"))
        message = str(getattr(diagnostic, "message", ""))
        lines.append(f"{code}: {message}" if message else code)
    return tuple(lines)


def _tool_error_lines(tool_results: object) -> tuple[str, ...]:
    lines: list[str] = []
    for result in tool_results or ():
        error = getattr(result, "error", None)
        if error is None:
            continue
        tool_name = str(getattr(result, "tool_name", "tool"))
        code = str(getattr(error, "code", "tool_error"))
        message = str(getattr(error, "message", ""))
        lines.append(f"{tool_name}:{code}: {message}" if message else f"{tool_name}:{code}")
    return tuple(lines)


def _format_result(result: SmokeResult, *, verbose: bool) -> list[str]:
    marker = "PASS" if result.passed else "FAIL"
    tools = ",".join(result.tools) or "-"
    lines = [f"{marker} {result.name} status={result.status} tools={tools}"]
    if result.error:
        lines.append(f"  error: {result.error}")
    if result.answer:
        lines.append(f"  answer: {result.answer}")
    if result.workspace_path:
        lines.append(f"  workspace: {result.workspace_path}")

    show_diagnostics = verbose or not result.passed
    if show_diagnostics and result.stop_reason:
        lines.append(f"  stop_reason: {result.stop_reason}")
    if show_diagnostics and result.schema_bytes is not None:
        lines.append(f"  schema_bytes: {result.schema_bytes}")
    if show_diagnostics:
        for diagnostic in result.diagnostics:
            lines.append(f"  diagnostic: {diagnostic}")
        for tool_error in result.tool_errors:
            lines.append(f"  tool_error: {tool_error}")
    return lines


async def run_matrix(*, model: str, only: set[str] | None = None) -> list[SmokeResult]:
    cases = [case for case in build_cases() if only is None or case.name in only]
    return [await run_case(case, model=model) for case in cases]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only this case. Can be provided multiple times.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stop reasons, request schema size, diagnostics, and tool errors.",
    )
    args = parser.parse_args()

    results = asyncio.run(run_matrix(model=args.model, only=set(args.cases) if args.cases else None))
    for result in results:
        for line in _format_result(result, verbose=args.verbose):
            print(line)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
