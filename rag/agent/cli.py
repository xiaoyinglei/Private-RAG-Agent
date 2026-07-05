from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import typer

from agent_runtime.models import (
    ModelControlPlane,
    ModelPolicyError,
    format_model_rows,
)
from rag.agent.core.llm_registry import UnknownModelAliasError

if TYPE_CHECKING:
    from agent_runtime import AgentResult
    from rag.agent.core.definition import AgentRuntimePolicy
    from rag.agent.core.human_input import HumanInputResponse
    from rag.agent.core.registry import AgentRegistry
    from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
    from rag.agent.service import AgentRunResult, AgentService
    from rag.agent.tools.registry import ContextualToolRunner

agent_app = typer.Typer(add_completion=False, no_args_is_help=True)
model_app = typer.Typer(add_completion=False, no_args_is_help=True)
agent_app.add_typer(model_app, name="model", help="查看和切换当前模型会话。")
logger = logging.getLogger(__name__)

CLI_AGENT_CHOICES = ("generic",)
_SEMANTIC_RAG_TOOLS = frozenset({"search_knowledge", "search_assets"})
DEFAULT_MODEL_SESSION_PATH = Path(".rag/agent_model_session.json")
DEFAULT_VECTOR_BACKEND = "milvus"


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


def _build_llm_tool_runners(
    primary_chat: Any,
    *,
    token_accounting: object | None = None,
    model_context_tokens: int = 32_768,
    stage_budgets: object | None = None,
) -> dict[str, ContextualToolRunner]:
    from rag.agent.core.llm_registry import ResolvedModel
    from rag.agent.core.llm_tool_runners import create_model_llm_tool_runners
    from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract
    from rag.providers.llm_gateway import LLMGateway
    from rag.schema.llm import DEFAULT_LLM_STAGE_BUDGETS

    if primary_chat is None:
        return {}

    accounting = token_accounting or TokenAccountingService(
        TokenizerContract(
            embedding_model_name="cli-chat",
            tokenizer_model_name="cli-chat",
            chunking_tokenizer_model_name="cli-chat",
            tokenizer_backend="simple",
            max_context_tokens=model_context_tokens,
            prompt_reserved_tokens=512,
            local_files_only=True,
        )
    )
    gateway = LLMGateway(
        generator=primary_chat,
        token_accounting=cast(Any, accounting),
        model_context_tokens=model_context_tokens,
        stage_budgets=cast(
            Any,
            stage_budgets or DEFAULT_LLM_STAGE_BUDGETS,
        ),
    )

    class _Registry:
        def resolve_for_node(
            self,
            *,
            node_model: str | None,
            node_name: str,
        ) -> ResolvedModel:
            del node_model, node_name
            return ResolvedModel(
                generator=primary_chat,
                kwargs={},
                context_window_tokens=model_context_tokens,
                gateway=gateway,
                token_accounting=cast(Any, accounting),
            )

    return create_model_llm_tool_runners(cast(Any, _Registry()))


def _resolve_cli_agent_definition(
    agent_registry: AgentRegistry,
    agent_type: str,
) -> AgentRuntimePolicy:
    if agent_type not in CLI_AGENT_CHOICES:
        allowed = ", ".join(CLI_AGENT_CHOICES)
        raise ValueError(f"{agent_type!r} is not a supported CLI agent. Allowed: {allowed}")
    return agent_registry.get(agent_type)


def _without_unavailable_deferred_tools(
    definition: AgentRuntimePolicy,
    unavailable_tools: set[str],
) -> AgentRuntimePolicy:
    if not unavailable_tools:
        return definition
    filt = definition.tool_catalog_filter
    return replace(
        definition,
        deferred_tool_names=tuple(
            name for name in definition.deferred_tool_names
            if name not in unavailable_tools
        ),
        tool_catalog_filter=replace(filt, deny=filt.deny | frozenset(unavailable_tools)),
    )


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
    knowledge_runner: ContextualToolRunner | None = None,
    knowledge_asset_runner: ContextualToolRunner | None = None,
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
        knowledge_asset_runner=knowledge_asset_runner,
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
        status_icon = "✓" if tr.status == "ok" else "✗"
        tool_info = f"  {status_icon} {tr.tool_name}"
        if tr.status == "error" and tr.error:
            tool_info += f" ({tr.error.code}: {tr.error.message[:60]})"
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
        print(f"\n警告: Agent 以降级模式运行（{degraded} 项诊断）")
        if verbose:
            for diagnostic in result.runtime_diagnostics:
                _print_diagnostic(diagnostic)

    if result.final_answer:
        print(f"\n{result.final_answer}")

    if result.tool_results:
        if verbose:
            print(_format_tool_summary(result))
        else:
            ok = sum(1 for tr in result.tool_results if tr.status == "ok")
            err = sum(1 for tr in result.tool_results if tr.status == "error")
            summary_parts = [f"{ok} 成功"] if ok else []
            if err:
                summary_parts.append(f"{err} 失败")
            if summary_parts:
                print(f"\n工具: {', '.join(summary_parts)}")

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
        return

    if result.status == "failed":
        _display_failure(
            stop_reason=None,
            diagnostics=result.diagnostics,
            verbose=verbose,
        )
    elif result.diagnostics:
        degraded = sum(1 for diagnostic in result.diagnostics if getattr(diagnostic, "degraded", False))
        if degraded:
            print(f"\n警告: Agent 以降级模式运行（{degraded} 项诊断）")
        if verbose:
            for diagnostic in result.diagnostics:
                _print_diagnostic(diagnostic)

    if result.answer:
        print(f"\n{result.answer}")

    if verbose and result.tool_calls:
        print(_format_public_tool_summary(result.tool_calls))

    if verbose:
        print(f"状态: {result.status}")


def _format_public_tool_summary(tool_calls: Sequence[str]) -> str:
    lines = ["", "─" * 40, "工具执行:"]
    for tool_name in tool_calls:
        lines.append(f"  ✓ {tool_name}")
    return "\n".join(lines)


def _handle_pause(result: AgentRunResult, run_id: str) -> HumanInputResponse | None:
    from rag.agent.core.human_input import HumanInputRequest, HumanInputResponse

    """展示暂停信息，获取用户决策。返回 None 表示退出。"""
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
        if choice in {"allow_once", "deny", "continue", "abort"}:
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
        decision=cast(Literal["allow_once", "deny", "continue", "abort"], choice),
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
    return HumanInputResponse(
        request_id=r.request_id,
        decision=cast(Literal["allow_once", "deny", "continue", "abort"], decision),
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


def _print_startup_banner(model_alias: str, *, agent_type: str) -> None:
    print(f"Agent 就绪 (agent: {agent_type}, 模型: {model_alias})")
    print("输入查询，或 /exit 退出，/verbose 切换详细输出，/model 查看/切换模型")
    print()


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
) -> None:
    if control_plane is None:
        print("模型控制平面不可用。")
        return
    parts = query.split()
    action = parts[1] if len(parts) > 1 else "current"
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
    from rag.agent.service import AgentRunRequest
    from rag.utils.text import load_env_file

    startup_started_at = time.perf_counter()
    load_env_file()
    runtime, diagnostics = _build_optional_rag_runtime(
        storage_root=storage_root,
        model_alias=model,
        embedding_model_alias=embedding_model,
        reranker_model_alias=reranker_model,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
        explicit=False,
    )

    with runtime if runtime is not None else nullcontext():
        try:
            model_control_plane = _build_model_control_plane(
                model_alias=model,
                session_path=DEFAULT_MODEL_SESSION_PATH,
            )
        except Exception:
            if model is not None:
                raise
            model_control_plane = None
        service = _build_agent_service(
            runtime,
            agent_type=agent,
            model_alias=model,
            model_control_plane=model_control_plane,
            runtime_diagnostics=diagnostics,
            startup_ms=(time.perf_counter() - startup_started_at) * 1000,
        )
        try:
            run_id = f"chat_{id(service):x}"
            verbose = False

            _print_startup_banner(_service_model_alias(service, model), agent_type=agent)

            while True:
                try:
                    query = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n再见。")
                    break

                if not query:
                    continue
                if query == "/exit":
                    break
                if query == "/verbose":
                    verbose = not verbose
                    print(f"详细输出: {'开' if verbose else '关'}")
                    continue
                if query == "/model" or query.startswith("/model "):
                    _handle_model_slash_command(
                        query,
                        control_plane=_service_model_control_plane(service),
                    )
                    continue

                result = asyncio.run(
                    service.run(
                        AgentRunRequest(
                            task=query,
                            run_id=run_id,
                            thread_id=run_id,
                            llm_budget_total=budget,
                        )
                    )
                )

                while result.status == "paused":
                    _display_result(result, verbose=verbose)
                    response = _handle_pause(result, run_id)
                    if response is None:
                        print("已取消。")
                        break
                    result = asyncio.run(
                        service.resume(run_id=run_id, response=response)
                    )

                if result.status in ("done", "failed"):
                    _display_result(result, verbose=verbose)

                if result.status == "failed" and result.stop_reason:
                    if verbose:
                        print(f"失败: {result.stop_reason}")
        finally:
            _close_agent_service(service)


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
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="指定 run_id，便于后续 resume")
    ] = None,
    checkpoint_db: Annotated[
        Path | None,
        typer.Option("--checkpoint-db", help="SQLite checkpoint 文件；启用后可跨进程 resume"),
    ] = None,
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
) -> None:
    """单次 Agent 运行。传入 --checkpoint-db 后支持跨进程恢复。"""
    from rag.agent.service import AgentRunResult

    facade = _create_agent_facade(
        model=model,
        agent_type=agent,
        checkpoint_db=checkpoint_db,
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
    effective_run_id = run_id or f"run_{id(facade):x}"
    result = facade.run(
        task,
        files=input_files or [],
        run_id=effective_run_id,
        max_tokens_total=budget,
    )
    _display_agent_result(result, verbose=verbose)

    raw_result = result.raw if isinstance(result.raw, AgentRunResult) else None
    if result.status == "paused":
        print()
        if checkpoint_db is None:
            print("⚠  当前命令使用 MemorySaver，进程结束后暂停状态无法恢复。")
            print("   请使用 --checkpoint-db 启用 SQLite checkpoint 后重试。")
        else:
            print("⏸  已保存 checkpoint，可跨进程恢复:")
            resume_cmd = (
                f"   agent resume {effective_run_id} "
                f"--agent {agent} "
                f"--checkpoint-db {checkpoint_db}"
            )
            if raw_result is not None and raw_result.workspace_path:
                resume_cmd += f" --workspace-path {raw_result.workspace_path}"
            print(resume_cmd)

        if raw_result is not None and raw_result.needs_user_input:
            print(f"\n   待处理: {raw_result.needs_user_input}")

        if non_interactive:
            raise typer.Exit(code=2)

    if result.status == "failed":
        raise typer.Exit(code=1)


@agent_app.command(name="resume")
def agent_resume(
    run_id: Annotated[str, typer.Argument(help="要恢复的 run_id/thread_id")],
    storage_root: Annotated[
        Path, typer.Option("--storage-root", help="RAG 存储根目录", hidden=True)
    ] = Path(".rag"),
    agent: Annotated[
        str,
        typer.Option(
            "--agent",
            help="恢复时使用的根 Agent，必须与原 run 一致。",
        ),
    ] = "generic",
    checkpoint_db: Annotated[
        Path,
        typer.Option("--checkpoint-db", help="SQLite checkpoint 文件"),
    ] = Path(".rag/agent_checkpoints.sqlite"),
    decision: Annotated[
        str,
        typer.Option("--decision", help="allow_once | deny | continue | abort"),
    ] = "allow_once",
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="详细输出")
    ] = False,
    model: Annotated[
        str | None,
        typer.Option("--model", help="主生成模型别名，对应 configs/models.yaml 中 capability=chat 的条目"),
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
    workspace_path: Annotated[
        str | None,
        typer.Option("--workspace-path", help="Workspace 路径，恢复 PrimitiveOps runner 所需"),
    ] = None,
) -> None:
    """从 SQLite checkpoint 恢复暂停的 Agent 运行。"""
    from rag.utils.text import load_env_file

    startup_started_at = time.perf_counter()
    load_env_file()
    runtime, diagnostics = _build_optional_rag_runtime(
        storage_root=storage_root,
        model_alias=model,
        embedding_model_alias=embedding_model,
        reranker_model_alias=reranker_model,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )

    with runtime if runtime is not None else nullcontext():
        try:
            model_control_plane = _build_model_control_plane(
                model_alias=model,
                session_path=DEFAULT_MODEL_SESSION_PATH,
            )
        except Exception:
            if model is not None:
                raise
            model_control_plane = None
        service = _build_agent_service(
            runtime,
            checkpoint_db=checkpoint_db,
            agent_type=agent,
            model_alias=model,
            model_control_plane=model_control_plane,
            runtime_diagnostics=diagnostics,
            startup_ms=(time.perf_counter() - startup_started_at) * 1000,
        )
        try:
            request = asyncio.run(service.apending_human_input_request(run_id=run_id))
            response = _build_resume_response(request, decision)
            result = asyncio.run(service.resume(run_id=run_id, response=response, workspace_path=workspace_path))
            _display_result(result, verbose=verbose)
            if result.status == "failed":
                raise typer.Exit(code=1)
        finally:
            _close_agent_service(service)
