from __future__ import annotations

import asyncio
import logging
import os
import shlex
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast
from uuid import uuid4

import typer

from agent_runtime.models import (
    ModelControlPlane,
    ModelPolicyError,
    format_model_rows,
)
from rag.agent.core.llm_registry import UnknownModelAliasError
from rag.agent.streaming.events import EventType, StreamEvent

if TYPE_CHECKING:
    from agent_runtime import AgentResult
    from rag.agent.core.definition import AgentRuntimePolicy
    from rag.agent.core.human_input import HumanInputResponse
    from rag.agent.core.registry import AgentRegistry
    from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
    from rag.agent.service import AgentRunResult, AgentService

agent_app = typer.Typer(add_completion=False, no_args_is_help=True)
model_app = typer.Typer(add_completion=False, no_args_is_help=True)
agent_app.add_typer(model_app, name="model", help="查看和切换当前模型会话。")
logger = logging.getLogger(__name__)

CLI_AGENT_CHOICES = ("generic",)
DEFAULT_MODEL_SESSION_PATH = Path(".rag/agent_model_session.json")
DEFAULT_CHECKPOINT_PATH = Path(".rag/agent_checkpoints.sqlite")
DEFAULT_VECTOR_BACKEND = "milvus"


class _CLIToolEventDisplay:
    """Render canonical tool-start events without deriving runtime state."""

    def __init__(self) -> None:
        self._displayed_tool_ids: set[str] = set()

    async def emit(self, event: StreamEvent) -> None:
        if event.type is not EventType.TOOL_USE_START:
            return
        tool_id = event.data.get("tool_id")
        if isinstance(tool_id, str) and tool_id:
            if tool_id in self._displayed_tool_ids:
                return
            self._displayed_tool_ids.add(tool_id)
        tool_name = event.data.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            return
        preview = event.data.get("input_preview")
        suffix = f": {preview}" if isinstance(preview, str) and preview else ""
        print(f"\n→ {tool_name}{suffix}")


@dataclass(frozen=True)
class AutoRAGConfig:
    storage_root: Path
    vector_backend: str
    vector_dsn: str | None
    vector_namespace: str | None
    vector_collection_prefix: str | None
    explicit: bool


def _resolve_auto_rag_config(
    *,
    storage_root: Path,
    vector_backend: str,
    vector_dsn: str | None,
    vector_namespace: str | None,
    vector_collection_prefix: str | None,
) -> AutoRAGConfig:
    effective_storage_root = storage_root
    env_storage_root = (
        os.environ.get("AGENT_RAG_STORAGE_ROOT")
        or os.environ.get("RAG_STORAGE_ROOT")
        or os.environ.get("STORAGE_ROOT")
    )
    if storage_root == Path(".rag"):
        if env_storage_root:
            effective_storage_root = Path(env_storage_root)
    env_vector_backend = os.environ.get("AGENT_VECTOR_BACKEND") or os.environ.get("VECTOR_BACKEND")
    env_vector_dsn = os.environ.get("AGENT_VECTOR_DSN") or os.environ.get("VECTOR_DSN")
    env_vector_namespace = os.environ.get("AGENT_VECTOR_NAMESPACE") or os.environ.get("VECTOR_NAMESPACE")
    env_vector_prefix = os.environ.get("AGENT_VECTOR_PREFIX") or os.environ.get("VECTOR_PREFIX")
    return AutoRAGConfig(
        storage_root=effective_storage_root,
        vector_backend=env_vector_backend or vector_backend,
        vector_dsn=vector_dsn or env_vector_dsn,
        vector_namespace=vector_namespace or env_vector_namespace,
        vector_collection_prefix=vector_collection_prefix or env_vector_prefix,
        explicit=(
            storage_root != Path(".rag")
            or bool(env_storage_root)
            or vector_backend != DEFAULT_VECTOR_BACKEND
            or bool(env_vector_backend)
            or bool(vector_dsn)
            or bool(env_vector_dsn)
            or bool(vector_namespace)
            or bool(env_vector_namespace)
            or bool(vector_collection_prefix)
            or bool(env_vector_prefix)
        ),
    )


def _looks_like_rag_storage(storage_root: Path) -> bool:
    return any(
        (storage_root / marker).exists()
        for marker in ("metadata.sqlite3", "vectors.sqlite3", "index.sqlite")
    )


def _resolve_cli_agent_definition(
    agent_registry: AgentRegistry,
    agent_type: str,
) -> AgentRuntimePolicy:
    if agent_type not in CLI_AGENT_CHOICES:
        allowed = ", ".join(CLI_AGENT_CHOICES)
        raise ValueError(f"{agent_type!r} is not a supported CLI agent. Allowed: {allowed}")
    return agent_registry.get(agent_type)


def _service_model_alias(service: AgentService, requested_model: str | None) -> str:
    registry = getattr(service, "_model_registry", None)
    if registry is not None:
        return str(registry.default_model)
    return requested_model or "unavailable"


def _build_model_control_plane(
    *,
    model_alias: str | None = None,
    session_path: Path | None = None,
) -> ModelControlPlane:
    return ModelControlPlane.from_env(
        initial_model_id=model_alias,
        session_path=session_path,
    )


def _service_model_control_plane(service: AgentService) -> ModelControlPlane | None:
    registry = getattr(service, "_model_registry", None)
    return registry if isinstance(registry, ModelControlPlane) else None


def _build_agent_service(
    runtime: Any | None,
    *,
    checkpoint_db: Path | None = None,
    agent_type: str = "generic",
    model_alias: str | None = None,
    model_control_plane: ModelControlPlane | None = None,
    runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
    knowledge_runner: Callable[..., object] | None = None,
    strict_model_provider: bool = True,
    startup_ms: float = 0.0,
) -> AgentService:
    from agent_runtime.runtime.builder import build_agent_service

    return build_agent_service(
        runtime,
        checkpoint_db=checkpoint_db,
        agent_type=agent_type,
        model_alias=model_alias,
        model_control_plane=model_control_plane,
        runtime_diagnostics=runtime_diagnostics,
        knowledge_runner=knowledge_runner,
        strict_model_provider=strict_model_provider,
        startup_ms=startup_ms,
    )


def _build_optional_rag_runtime(
    *,
    storage_root: Path,
    model_alias: str | None,
    embedding_model_alias: str | None,
    reranker_model_alias: str | None,
    vector_backend: str,
    vector_dsn: str | None,
    vector_namespace: str | None,
    vector_collection_prefix: str | None,
    explicit: bool = False,
) -> tuple[Any | None, tuple[RuntimeDiagnostic, ...]]:
    from agent_runtime.runtime.builder import build_optional_rag_runtime

    return build_optional_rag_runtime(
        storage_root=storage_root,
        model_alias=model_alias,
        embedding_model_alias=embedding_model_alias,
        reranker_model_alias=reranker_model_alias,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
        explicit=explicit,
    )


def _format_tool_summary(result: AgentRunResult) -> str:
    if not result.tool_results:
        return ""
    lines = ["", "─" * 40, "工具执行:"]
    for tr in result.tool_results:
        status_icon = "✗" if tr.is_error else "✓"
        tool_info = f"  {status_icon} {tr.tool_name}"
        if tr.is_error:
            tool_info += (
                f" ({tr.error_code or 'tool_error'}: "
                f"{(tr.error_message or 'unknown tool error')[:60]})"
            )
        lines.append(tool_info)
    return "\n".join(lines)


def _failure_title(stop_reason: str | None) -> str:
    if stop_reason == "model_provider_failed":
        return "错误: 模型调用失败 (model_provider_failed)"
    if stop_reason:
        return f"错误: Agent 运行失败 ({stop_reason})"
    return "错误: Agent 运行失败"


def _failure_diagnostics(
    diagnostics: Sequence[object],
    *,
    stop_reason: str | None,
) -> list[object]:
    all_diagnostics = list(diagnostics)
    if not all_diagnostics:
        return []
    if stop_reason:
        exact = [
            diagnostic for diagnostic in all_diagnostics
            if getattr(diagnostic, "code", None) == stop_reason
        ]
        if exact:
            return exact
    errors = [
        diagnostic for diagnostic in all_diagnostics
        if getattr(diagnostic, "severity", None) == "error"
    ]
    return errors or all_diagnostics[:1]


def _print_diagnostic(diagnostic: object) -> None:
    component = getattr(diagnostic, "component", "diagnostic")
    code = getattr(diagnostic, "code", "unknown")
    message = getattr(diagnostic, "message", "")
    error_type = getattr(diagnostic, "error_type", None)
    suffix = f", {error_type}" if error_type is not None else ""
    print(f"  [{component}] {code}: {message}{suffix}")


def _display_failure(
    *,
    stop_reason: str | None,
    diagnostics: Sequence[object],
    verbose: bool,
) -> None:
    print(f"\n{_failure_title(stop_reason)}")
    selected = _failure_diagnostics(diagnostics, stop_reason=stop_reason)
    if selected:
        for diagnostic in selected:
            _print_diagnostic(diagnostic)
    elif verbose and stop_reason:
        print(f"  stop_reason: {stop_reason}")
    if verbose:
        selected_ids = {id(diagnostic) for diagnostic in selected}
        for diagnostic in diagnostics:
            if id(diagnostic) not in selected_ids:
                _print_diagnostic(diagnostic)


def _display_result(result: AgentRunResult, *, verbose: bool) -> None:
    """干净输出 AgentRunResult。"""
    if result.status == "failed":
        _display_failure(
            stop_reason=result.stop_reason,
            diagnostics=result.runtime_diagnostics,
            verbose=verbose,
        )
    elif result.runtime_diagnostics:
        degraded = sum(
            1 for diagnostic in result.runtime_diagnostics if diagnostic.degraded
        )
        if degraded:
            print(f"\n警告: Agent 以降级模式运行（{degraded} 项诊断）")
        if verbose:
            for diagnostic in result.runtime_diagnostics:
                _print_diagnostic(diagnostic)

    if result.final_answer:
        print(f"\n{result.final_answer}")

    if result.tool_results:
        print(_format_tool_summary(result))

    if verbose and result.tool_call_metrics:
        from rag.agent.core.runtime_diagnostics import ToolCallMetrics

        m = result.tool_call_metrics
        if isinstance(m, ToolCallMetrics):
            print(f"\n调用统计: native={m.native_calls}({m.native_errors}err/{m.native_latency_ms_total:.0f}ms) "
                  f"deferred={m.deferred_calls} mcp={m.mcp_calls}({m.mcp_errors}err/{m.mcp_latency_ms_total:.0f}ms)")

    if verbose and result.latency_profile is not None:
        p = result.latency_profile
        print(
            "\n耗时: "
            f"total={p.total_ms:.0f}ms "
            f"startup={p.startup_ms:.0f}ms "
            f"build={p.build_service_ms:.0f}ms "
            f"model_ready={p.model_ready_ms:.0f}ms "
            f"model={p.model_latency_ms:.0f}ms "
            f"tool={p.tool_latency_ms:.0f}ms "
            f"finalize={p.finalize_latency_ms:.0f}ms "
            f"prompt_bytes={p.prompt_bytes} "
            f"tool_schema_bytes={p.tool_schema_bytes}"
        )

    if verbose and result.evidence:
        print(f"证据: {len(result.evidence)} 条")

    if result.stop_reason and verbose:
        print(f"停止原因: {result.stop_reason}")


def _display_agent_result(result: AgentResult, *, verbose: bool) -> None:
    from rag.agent.service import AgentRunResult

    if isinstance(result.raw, AgentRunResult):
        _display_result(result.raw, verbose=verbose)
    else:
        if result.status == "failed":
            _display_failure(
                stop_reason=None,
                diagnostics=result.diagnostics,
                verbose=verbose,
            )
        elif result.diagnostics:
            degraded = sum(
                1
                for diagnostic in result.diagnostics
                if getattr(diagnostic, "degraded", False)
            )
            if degraded:
                print(
                    f"\n警告: Agent 以降级模式运行（{degraded} 项诊断）"
                )
            if verbose:
                for diagnostic in result.diagnostics:
                    _print_diagnostic(diagnostic)

        if result.answer:
            print(f"\n{result.answer}")

        if verbose and result.tool_calls:
            print(_format_public_tool_summary(result.tool_calls))

    print(f"\nSession: {result.session_id}")
    print(f"Turn: {result.turn_id}")

    if verbose:
        print(f"状态: {result.status}")


def _format_public_tool_summary(tool_calls: Sequence[str]) -> str:
    lines = ["", "─" * 40, "工具执行:"]
    for tool_name in tool_calls:
        lines.append(f"  ✓ {tool_name}")
    return "\n".join(lines)


def _handle_pause(
    result: AgentRunResult,
    turn_id: str,
) -> HumanInputResponse | None:
    from rag.agent.core.human_input import HumanInputRequest, HumanInputResponse

    """展示暂停信息，获取用户决策。返回 None 表示退出。"""
    del turn_id
    req = result.human_input_request
    if req is None:
        return None
    req = cast(HumanInputRequest, req)

    print(f"\n⏸  需要确认: {result.needs_user_input or req.question}")

    if req.tool_calls:
        for tc in req.tool_calls:
            risk_mark = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(getattr(tc, "risk_level", "low"), "")
            print(f"  {risk_mark} {tc.tool_name}: {getattr(tc, 'args_preview', '')}")
            if getattr(tc, "reason", ""):
                print(f"     原因: {tc.reason}")

    options = getattr(req, "options", []) or ["allow_once", "deny", "continue", "abort"]
    print(f"  选项: {', '.join(options)}")

    while True:
        choice = input("> ").strip()
        if choice in options:
            break
        if choice in {"a", "y", "yes"}:
            choice = "allow_once"
            break
        if choice in {"n", "no", "d"}:
            choice = "deny"
            break
        if choice in {"c"}:
            choice = "continue"
            break
        if choice in {"q", "exit", "/exit"}:
            return None
        print(f"  请输入: {', '.join(options)} (或 a=允许, n=拒绝, q=退出)")

    approved = (
        [tc.tool_call_id for tc in req.tool_calls]
        if choice == "allow_once" else []
    )
    denied = (
        [tc.tool_call_id for tc in req.tool_calls]
        if choice == "deny" else []
    )

    return HumanInputResponse(
        request_id=req.request_id,
        decision=cast(Any, choice),
        approved_tool_call_ids=approved,
        denied_tool_call_ids=denied,
    )


def _build_resume_response(request: object, decision: str) -> HumanInputResponse:
    from rag.agent.core.human_input import HumanInputRequest, HumanInputResponse

    r = cast(HumanInputRequest, request)
    tool_call_ids = [
        tool_call.tool_call_id
        for tool_call in r.tool_calls
    ]
    allowed = set(r.options)
    if not allowed and r.kind == "tool_approval":
        allowed = {"allow_once", "deny", "abort"}
    if decision not in allowed:
        raise ValueError(
            f"unsupported decision {decision!r} for {r.kind} request"
        )
    return HumanInputResponse(
        request_id=r.request_id,
        decision=cast(Any, decision),
        approved_tool_call_ids=tool_call_ids if decision == "allow_once" else [],
        denied_tool_call_ids=tool_call_ids if decision == "deny" else [],
    )


def _close_agent_service(service: object) -> None:
    close_method = getattr(service, "aclose", None)
    if callable(close_method):
        asyncio.run(close_method())


def _create_agent_facade(
    *,
    model: str | None = None,
    agent_type: str = "generic",
    checkpoint_db: Path | None = None,
    workspace_path: Path | str | None = None,
    model_session_path: Path | None = None,
    knowledge: tuple[str, ...] | list[str] | None = None,
    rag_storage_root: Path = Path(".rag"),
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    vector_backend: str = DEFAULT_VECTOR_BACKEND,
    vector_dsn: str | None = None,
    vector_namespace: str | None = None,
    vector_collection_prefix: str | None = None,
) -> Any:
    from agent_runtime import Agent

    return Agent(
        model=model,
        agent_type=agent_type,
        checkpoint_db=checkpoint_db,
        workspace_path=workspace_path,
        model_session_path=model_session_path,
        knowledge=knowledge,
        rag_storage_root=rag_storage_root,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )


def _is_interactive_terminal() -> bool:
    if any(
        _environment_flag(name)
        for name in ("CI", "GITHUB_ACTIONS")
    ):
        return False
    return _is_tty(sys.stdin) and _is_tty(sys.stdout)


def _environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _is_tty(stream: object) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


async def _run_facade_command(
    facade: Any,
    *,
    task: str,
    files: Sequence[str],
    turn_id: str,
    max_tokens_total: int | None,
    interactive_approval: bool,
    tools: Sequence[str] | None = None,
    disabled_tools: Sequence[str] | None = None,
    allow_write_tools: bool = False,
    allow_execute_tools: bool = False,
    allow_discovery_tools: bool | None = None,
) -> AgentResult:
    """Run and optionally approve on one service, event loop, and ToolCall."""

    from agent_runtime.agent import _effective_discovery_option
    from agent_runtime.result import AgentResult
    from rag.agent.service import AgentRunRequest

    effective_discovery = _effective_discovery_option(
        tools=None if tools is None else list(tools),
        disabled_tools=(
            None if disabled_tools is None else list(disabled_tools)
        ),
        allow_discovery_tools=allow_discovery_tools,
    )
    async with facade._open_product_runtime(
        stream_sink=_CLIToolEventDisplay(),
    ) as service:
        raw = await service.chat(
            AgentRunRequest(
                task=task,
                session_id=None,
                run_id=turn_id,
                thread_id=turn_id,
                llm_budget_total=max_tokens_total,
                input_files=list(files),
                workspace_path=(
                    None
                    if facade.workspace_path is None
                    else str(facade.workspace_path)
                ),
                tools=None if tools is None else tuple(tools),
                disabled_tools=tuple(disabled_tools or ()),
                allow_write_tools=allow_write_tools,
                allow_execute_tools=allow_execute_tools,
                allow_discovery_tools=effective_discovery,
            )
        )
        while raw.status == "paused" and interactive_approval:
            response = _handle_pause(raw, raw.run_id)
            if response is None:
                break
            raw = await service.resume_turn(
                turn_id=raw.run_id,
                action=response.decision,
                user_input=response.user_message,
            )
        return AgentResult.from_internal(raw, files=tuple(files))


async def _resume_facade_command(
    facade: Any,
    *,
    turn_id: str,
    action: str,
    user_input: str | None = None,
) -> AgentResult:
    return cast(
        "AgentResult",
        await facade.aresume(
            turn_id,
            action,
            user_input=user_input,
        ),
    )


def _print_startup_banner(model_alias: str, *, agent_type: str) -> None:
    print(f"Agent 就绪 (agent: {agent_type}, 模型: {model_alias})")
    print(
        "输入查询，或 /exit 退出，/verbose 切换详细输出，"
        "/model 查看模型（首条消息前可切换）"
    )
    print()


async def _chat_facade_session(
    facade: Any,
    *,
    agent_type: str,
    requested_model: str | None,
    budget: int | None,
    session_id: str | None = None,
) -> None:
    from rag.agent.service import AgentRunRequest

    runtime_facade = (
        facade
        if session_id is None
        else facade._agent_for_session(session_id)
    )
    async with runtime_facade._open_product_runtime(
        stream_sink=_CLIToolEventDisplay(),
    ) as service:
        current_session_id = session_id
        verbose = False
        _print_startup_banner(
            _service_model_alias(service, requested_model),
            agent_type=agent_type,
        )
        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                return
            if not query:
                continue
            if query == "/exit":
                return
            if query == "/verbose":
                verbose = not verbose
                print(f"详细输出: {'开' if verbose else '关'}")
                continue
            if query == "/model" or query.startswith("/model "):
                _handle_model_slash_command(
                    query,
                    control_plane=_service_model_control_plane(service),
                    allow_switch=current_session_id is None,
                )
                continue

            result = await service.chat(
                AgentRunRequest(
                    task=query,
                    session_id=current_session_id,
                    run_id=str(uuid4()),
                    llm_budget_total=budget,
                    workspace_path=(
                        None
                        if runtime_facade.workspace_path is None
                        else str(runtime_facade.workspace_path)
                    ),
                )
            )
            current_session_id = result.session_id
            while result.status == "paused":
                response = _handle_pause(result, result.run_id)
                if response is None:
                    print("已取消。")
                    break
                result = await service.resume_turn(
                    turn_id=result.run_id,
                    action=response.decision,
                    user_input=response.user_message,
                )
            _display_result(result, verbose=verbose)


def _print_current_model(control_plane: ModelControlPlane) -> None:
    spec = control_plane.current_model()
    print(f"{spec.id}")
    print(f"provider: {spec.provider}")
    print(f"provider_model: {spec.provider_model}")
    print(f"context_window: {spec.context_window}")
    print(f"location: {spec.location}")


def _handle_model_slash_command(
    query: str,
    *,
    control_plane: ModelControlPlane | None,
    allow_switch: bool = True,
) -> None:
    if control_plane is None:
        print("模型控制平面不可用。")
        return
    parts = query.split()
    action = parts[1] if len(parts) > 1 else "current"
    switch_requested = action in {"switch", "use"} or (
        len(parts) == 2 and action not in {"current", "list"}
    )
    if switch_requested and not allow_switch:
        print(
            "Session 已创建，模型绑定已冻结；"
            "请新建 Session 并在首条消息前切换模型。"
        )
        return
    try:
        if action == "list":
            for line in format_model_rows(
                control_plane.list_models(),
                current_model_id=control_plane.current_model().id,
            ):
                print(line)
            return
        if action == "current":
            _print_current_model(control_plane)
            return
        if action in {"switch", "use"}:
            if len(parts) < 3:
                print("用法: /model switch <model_id>")
                return
            spec = control_plane.switch_model(parts[2], requested_by="user")
            print(f"已切换模型: {spec.id}")
            return
        if len(parts) == 2:
            spec = control_plane.switch_model(action, requested_by="user")
            print(f"已切换模型: {spec.id}")
            return
    except (ModelPolicyError, UnknownModelAliasError) as exc:
        print(f"模型切换失败: {exc}")
        return
    print("用法: /model [current|list|switch <model_id>]")


# ── CLI Commands ──


@model_app.command(name="list")
def model_list(
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """列出可用模型，并标记当前会话模型。"""
    control_plane = _build_model_control_plane(session_path=session_path)
    for line in format_model_rows(
        control_plane.list_models(),
        current_model_id=control_plane.current_model().id,
    ):
        print(line)


@model_app.command(name="current")
def model_current(
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """显示当前会话模型。"""
    _print_current_model(_build_model_control_plane(session_path=session_path))


@model_app.command(name="switch")
def model_switch(
    model_id: Annotated[str, typer.Argument(help="要切换到的模型 id")],
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """切换当前模型 session state，不修改 models.yaml。"""
    control_plane = _build_model_control_plane(session_path=session_path)
    try:
        spec = control_plane.switch_model(model_id, requested_by="user")
    except (ModelPolicyError, UnknownModelAliasError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    print(f"已切换模型: {spec.id}")


@agent_app.command(name="chat")
def agent_chat(
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="继续一个持久化 Session；省略则创建新 Session。",
        ),
    ] = None,
    storage_root: Annotated[
        Path, typer.Option("--storage-root", help="RAG 存储根目录", hidden=True)
    ] = Path(".rag"),
    agent: Annotated[
        str,
        typer.Option(
            "--agent",
            help="根 Agent 类型。默认 generic。",
        ),
    ] = "generic",
    model: Annotated[
        str | None,
        typer.Option("--model", help="主生成模型别名，对应 configs/models.yaml 中 capability=chat 的条目"),
    ] = None,
    budget: Annotated[
        int | None,
        typer.Option("--budget", min=1, help="本次 Agent 运行的 LLM/工具预算上限"),
    ] = None,
    embedding_model: Annotated[
        str | None,
        typer.Option(
            "--embedding-model",
            help="Embedding 模型别名，对应 configs/models.yaml 中 capability=embedding 的条目",
            hidden=True,
        ),
    ] = None,
    reranker_model: Annotated[
        str | None,
        typer.Option(
            "--reranker-model",
            help="Reranker 模型别名，对应 configs/models.yaml 中 capability=reranker 的条目",
            hidden=True,
        ),
    ] = None,
    vector_backend: Annotated[
        str,
        typer.Option("--vector-backend", help="Vector backend: milvus or sqlite.", hidden=True),
    ] = DEFAULT_VECTOR_BACKEND,
    vector_dsn: Annotated[
        str | None,
        typer.Option("--vector-dsn", help="Vector backend DSN.", hidden=True),
    ] = None,
    vector_namespace: Annotated[
        str | None,
        typer.Option("--vector-namespace", help="Vector namespace/database.", hidden=True),
    ] = None,
    vector_collection_prefix: Annotated[
        str | None,
        typer.Option("--vector-collection-prefix", help="Milvus collection prefix used at ingest time.", hidden=True),
    ] = None,
) -> None:
    """交互式 Agent 对话。暂停时支持工具审批。"""
    facade = _create_agent_facade(
        model=model,
        agent_type=agent,
        checkpoint_db=DEFAULT_CHECKPOINT_PATH,
        workspace_path=Path.cwd(),
        model_session_path=DEFAULT_MODEL_SESSION_PATH,
        rag_storage_root=storage_root,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )
    asyncio.run(
        _chat_facade_session(
            facade,
            agent_type=agent,
            requested_model=model,
            budget=budget,
            session_id=session_id,
        )
    )


@agent_app.command(name="run")
def agent_run(
    task: Annotated[str, typer.Argument(help="查询任务")],
    storage_root: Annotated[
        Path, typer.Option("--storage-root", help="RAG 存储根目录", hidden=True)
    ] = Path(".rag"),
    agent: Annotated[
        str,
        typer.Option(
            "--agent",
            help="根 Agent 类型。默认 generic。",
        ),
    ] = "generic",
    non_interactive: Annotated[
        bool, typer.Option("--non-interactive", help="非交互模式")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="详细输出")
    ] = False,
    turn_id: Annotated[
        str | None,
        typer.Option(
            "--turn-id",
            "--run-id",
            help="指定 UUID Turn ID；--run-id 仅作兼容别名。",
        ),
    ] = None,
    checkpoint_db: Annotated[
        Path,
        typer.Option("--checkpoint-db", help="SQLite checkpoint 文件；启用后可跨进程 resume"),
    ] = DEFAULT_CHECKPOINT_PATH,
    budget: Annotated[
        int | None,
        typer.Option("--max-tokens-total", "--budget", min=1, help="本次 Agent 运行的 LLM/工具预算上限", hidden=True),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="主生成模型别名，对应 configs/models.yaml 中 capability=chat 的条目"),
    ] = None,
    embedding_model: Annotated[
        str | None,
        typer.Option(
            "--embedding-model",
            help="Embedding 模型别名，对应 configs/models.yaml 中 capability=embedding 的条目",
            hidden=True,
        ),
    ] = None,
    reranker_model: Annotated[
        str | None,
        typer.Option(
            "--reranker-model",
            help="Reranker 模型别名，对应 configs/models.yaml 中 capability=reranker 的条目",
            hidden=True,
        ),
    ] = None,
    vector_backend: Annotated[
        str,
        typer.Option("--vector-backend", help="Vector backend: milvus or sqlite.", hidden=True),
    ] = DEFAULT_VECTOR_BACKEND,
    vector_dsn: Annotated[
        str | None,
        typer.Option("--vector-dsn", help="Vector backend DSN.", hidden=True),
    ] = None,
    vector_namespace: Annotated[
        str | None,
        typer.Option("--vector-namespace", help="Vector namespace/database.", hidden=True),
    ] = None,
    vector_collection_prefix: Annotated[
        str | None,
        typer.Option("--vector-collection-prefix", help="Milvus collection prefix used at ingest time.", hidden=True),
    ] = None,
    knowledge: Annotated[
        list[str] | None,
        typer.Option("--knowledge", help="启用显式知识库，可多次指定"),
    ] = None,
    input_files: Annotated[
        list[str] | None,
        typer.Option("--file", "-f", "--input-file", help="导入 workspace 的输入文件，可多次指定"),
    ] = None,
    tool_names: Annotated[
        list[str] | None,
        typer.Option(
            "--tool",
            help="兼容用：显式发送给模型的工具 schema",
            hidden=True,
        ),
    ] = None,
    disabled_tool_names: Annotated[
        list[str] | None,
        typer.Option(
            "--disable-tool",
            help="兼容用：从本次工具面中禁用工具",
            hidden=True,
        ),
    ] = None,
    allow_write_tools: Annotated[
        bool,
        typer.Option("--allow-write-tools", help="预授权本次 workspace 写入类调用"),
    ] = False,
    allow_execute_tools: Annotated[
        bool,
        typer.Option("--allow-execute-tools", help="预授权本次进程执行类调用"),
    ] = False,
    allow_discovery_tools: Annotated[
        bool | None,
        typer.Option(
            "--allow-discovery-tools/--no-discovery-tools",
            help="兼容用：覆盖自动 discovery 选择",
            hidden=True,
        ),
    ] = None,
) -> None:
    """单次 Agent 运行；交互终端内联审批，其他环境持久化后返回。"""
    from rag.agent.service import AgentRunResult

    facade = _create_agent_facade(
        model=model,
        agent_type=agent,
        checkpoint_db=checkpoint_db,
        workspace_path=Path.cwd(),
        model_session_path=DEFAULT_MODEL_SESSION_PATH,
        knowledge=tuple(knowledge or ()),
        rag_storage_root=storage_root,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )
    effective_turn_id = turn_id or str(uuid4())
    interactive_approval = (
        not non_interactive and _is_interactive_terminal()
    )
    result = asyncio.run(
        _run_facade_command(
            facade,
            task=task,
            files=input_files or [],
            turn_id=effective_turn_id,
            max_tokens_total=budget,
            interactive_approval=interactive_approval,
            tools=tool_names,
            disabled_tools=disabled_tool_names,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
            allow_discovery_tools=allow_discovery_tools,
        )
    )
    _display_agent_result(result, verbose=verbose)

    raw_result = result.raw if isinstance(result.raw, AgentRunResult) else None
    if result.status == "paused":
        print()
        print("⏸  已保存 checkpoint，可跨进程恢复:")
        resume_args = [
            "agent",
            "resume",
            result.turn_id,
            "--checkpoint-db",
            str(checkpoint_db),
        ]
        print(f"   {shlex.join(resume_args)}")

        if raw_result is not None and raw_result.needs_user_input:
            print(f"\n   待处理: {raw_result.needs_user_input}")

        raise typer.Exit(code=2)

    if result.status == "failed":
        raise typer.Exit(code=1)


@agent_app.command(name="resume")
def agent_resume(
    turn_id: Annotated[str, typer.Argument(help="要恢复的 UUID Turn ID")],
    checkpoint_db: Annotated[
        Path,
        typer.Option("--checkpoint-db", help="SQLite checkpoint 文件"),
    ] = DEFAULT_CHECKPOINT_PATH,
    action: Annotated[
        str,
        typer.Option(
            "--action",
            "--decision",
            help=(
                "allow_once | deny | continue | mark_completed | "
                "mark_failed | abort"
            ),
        ),
    ] = "continue",
    user_input: Annotated[
        str | None,
        typer.Option("--input", help="clarification/choice 恢复时的用户输入"),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="详细输出")
    ] = False,
    vector_dsn: Annotated[
        str | None,
        typer.Option(
            "--vector-dsn",
            help="恢复知识库时从当前进程注入的敏感 DSN。",
            hidden=True,
        ),
    ] = None,
) -> None:
    """先读取持久化 Turn 元数据，再恢复未完成的 Turn。"""
    facade = _create_agent_facade(
        checkpoint_db=checkpoint_db,
        vector_dsn=vector_dsn,
    )
    result = asyncio.run(
        _resume_facade_command(
            facade,
            turn_id=turn_id,
            action=action,
            user_input=user_input,
        )
    )
    _display_agent_result(result, verbose=verbose)
    if result.status == "paused":
        raise typer.Exit(code=2)
    if result.status == "failed":
        raise typer.Exit(code=1)
