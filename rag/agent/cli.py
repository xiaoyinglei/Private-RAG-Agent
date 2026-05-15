from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer

from rag.agent.builtin import create_builtin_agent_registry
from rag.agent.core.agent_service_factory import AgentServiceFactory
from rag.agent.core.checkpointing import create_agent_checkpointer
from rag.agent.core.definition import AgentDefinition
from rag.agent.core.human_input import HumanInputRequest, HumanInputResponse
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.core.registry import AgentRegistry
from rag.agent.core.subagent_runner import BuiltinSubAgentRunner, BuiltinSynthesisRunner
from rag.agent.service import AgentRunRequest, AgentRunResult, AgentService
from rag.agent.tools.builtin_registry import create_builtin_tool_registry
from rag.agent.tools.fast_path_tools import RAGSearchAnswerRunner
from rag.agent.tools.llm_tools import (
    LLMCompareInput,
    LLMGenerateInput,
    LLMSummarizeInput,
    LLMTextOutput,
)
from rag.agent.tools.registry import ToolRunner

agent_app = typer.Typer(add_completion=False, no_args_is_help=True)

CLI_AGENT_CHOICES = ("research", "orchestrator", "compare", "factcheck")


def _build_llm_tool_runners(primary_chat: Any) -> dict[str, ToolRunner]:
    if primary_chat is None:
        return {}

    def _llm_generate(payload: LLMGenerateInput) -> LLMTextOutput:
        text = primary_chat.chat(payload.prompt)
        return LLMTextOutput(text=text)

    def _llm_summarize(payload: LLMSummarizeInput) -> LLMTextOutput:
        prompt = payload.task
        if payload.context_sections:
            prompt = payload.task + "\n\n" + "\n".join(payload.context_sections)
        text = primary_chat.chat(prompt)
        return LLMTextOutput(
            text=text,
            evidence_ids=payload.evidence_ids,
            citation_ids=payload.citation_ids,
        )

    def _llm_compare(payload: LLMCompareInput) -> LLMTextOutput:
        prompt = payload.question
        if payload.left_context_sections or payload.right_context_sections:
            prompt += "\n\n左:\n" + "\n".join(payload.left_context_sections)
            prompt += "\n\n右:\n" + "\n".join(payload.right_context_sections)
        text = primary_chat.chat(prompt)
        return LLMTextOutput(text=text)

    return {
        "llm_generate": cast(ToolRunner, _llm_generate),
        "llm_summarize": cast(ToolRunner, _llm_summarize),
        "llm_compare": cast(ToolRunner, _llm_compare),
    }


def _resolve_cli_agent_definition(
    agent_registry: AgentRegistry,
    agent_type: str,
) -> AgentDefinition:
    if agent_type not in CLI_AGENT_CHOICES:
        allowed = ", ".join(CLI_AGENT_CHOICES)
        raise ValueError(f"{agent_type!r} is not a supported CLI agent. Allowed: {allowed}")
    return agent_registry.get(agent_type)


def _build_agent_service(
    runtime: Any,
    *,
    checkpoint_db: Path | None = None,
    agent_type: str = "research",
) -> AgentService:
    """从 RAGRuntime 构造 AgentService，注册真实 RAG tool runners。

    只在可以构造 RAGRuntime / retrieval_service 时成功。
    无法构造时报错，不静默使用 stub。
    """
    chat_bindings = list(runtime.capability_bundle.chat_bindings)
    primary_chat = chat_bindings[0] if chat_bindings else None

    runners: dict[str, ToolRunner] = {}

    # RAG tools — AsyncRAGToolRunner（aretrieve_payload → to_thread fallback）
    from rag.agent.tools.rag_tool_runner import AsyncRAGToolRunner

    rag_runner = AsyncRAGToolRunner(
        runtime=runtime,
        retrieval_service=runtime.retrieval_service,
        max_context_tokens=4096,
    )
    for name in ("vector_search", "keyword_search", "grounding", "rerank", "graph_expand"):
        runners[name] = cast(ToolRunner, rag_runner.retrieve_evidence)
    fast_path_runner = RAGSearchAnswerRunner(runtime=runtime)
    runners["rag_search_answer"] = cast(ToolRunner, fast_path_runner.answer)

    runners.update(_build_llm_tool_runners(primary_chat))

    tool_registry = create_builtin_tool_registry(runners=runners)
    try:
        model_registry = ModelRegistry.from_env()
    except Exception:
        model_registry = None

    agent_registry = create_builtin_agent_registry()
    definition = _resolve_cli_agent_definition(agent_registry, agent_type)
    service_factory = AgentServiceFactory(
        tool_registry=tool_registry,
        model_registry=model_registry,
        checkpointer=create_agent_checkpointer(checkpoint_db),
    )
    subagent_runner = BuiltinSubAgentRunner(
        agent_registry=agent_registry,
        service_factory=service_factory,
    )
    synthesis_runner = BuiltinSynthesisRunner(
        agent_registry=agent_registry,
        service_factory=service_factory,
    )
    service_factory.bind_subagent_runner(subagent_runner)
    service_factory.bind_synthesis_runner(synthesis_runner)
    return service_factory.create(definition)


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


def _display_result(result: AgentRunResult, *, verbose: bool) -> None:
    """干净输出 AgentRunResult。"""
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

    if verbose and result.evidence:
        print(f"证据: {len(result.evidence)} 条")

    if result.stop_reason and verbose:
        print(f"停止原因: {result.stop_reason}")


def _handle_pause(result: AgentRunResult, run_id: str) -> HumanInputResponse | None:
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


def _print_startup_banner(model_alias: str, *, agent_type: str) -> None:
    print(f"Agent 就绪 (agent: {agent_type}, 模型: {model_alias})")
    print("输入查询，或 /exit 退出，/verbose 切换详细输出")
    print()


# ── CLI Commands ──


@agent_app.command(name="chat")
def agent_chat(
    storage_root: Annotated[
        Path, typer.Option("--storage-root", help="RAG 存储根目录")
    ] = Path(".rag"),
    agent: Annotated[
        str,
        typer.Option(
            "--agent",
            help="根 Agent：research, orchestrator, compare, factcheck。默认 research，不做自动意图判断。",
        ),
    ] = "research",
    model: Annotated[
        str | None,
        typer.Option("--model", help="主生成模型别名，对应 configs/models.yaml 中 capability=chat 的条目"),
    ] = None,
    embedding_model: Annotated[
        str | None,
        typer.Option(
            "--embedding-model",
            help="Embedding 模型别名，对应 configs/models.yaml 中 capability=embedding 的条目",
        ),
    ] = None,
    reranker_model: Annotated[
        str | None,
        typer.Option(
            "--reranker-model",
            help="Reranker 模型别名，对应 configs/models.yaml 中 capability=reranker 的条目",
        ),
    ] = None,
) -> None:
    """交互式 Agent 对话。暂停时支持工具审批。"""
    from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageComponentConfig, StorageConfig
    from rag.models.assembly_adapter import to_assembly_overrides
    from rag.models.runtime import RuntimeOverrides, resolve_runtime_config
    from rag.retrieval import QueryOptions

    runtime_config = resolve_runtime_config(
        RuntimeOverrides(
            model_alias=model,
            embedding_model_alias=embedding_model,
            reranker_model_alias=reranker_model,
        )
    )
    assembly_overrides = to_assembly_overrides(runtime_config)

    storage = StorageConfig(
        root=storage_root,
        vectors=StorageComponentConfig(backend="milvus", dsn="http://127.0.0.1:19530"),
    )
    requirements = CapabilityRequirements(
        require_chat=True,
        default_context_tokens=QueryOptions().max_context_tokens,
    )
    runtime = RAGRuntime.from_request(
        storage=storage,
        request=AssemblyRequest(
            requirements=requirements,
            overrides=assembly_overrides,
        ),
    )

    with runtime:
        service = _build_agent_service(runtime, agent_type=agent)
        run_id = f"chat_{id(service):x}"
        verbose = False

        _print_startup_banner(runtime_config.primary_model.alias, agent_type=agent)

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

            result = asyncio.run(
                service.run(AgentRunRequest(task=query, run_id=run_id, thread_id=run_id))
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


@agent_app.command(name="run")
def agent_run(
    task: Annotated[str, typer.Argument(help="查询任务")],
    storage_root: Annotated[
        Path, typer.Option("--storage-root", help="RAG 存储根目录")
    ] = Path(".rag"),
    agent: Annotated[
        str,
        typer.Option(
            "--agent",
            help="根 Agent：research, orchestrator, compare, factcheck。默认 research，不做自动意图判断。",
        ),
    ] = "research",
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
    model: Annotated[
        str | None,
        typer.Option("--model", help="主生成模型别名，对应 configs/models.yaml 中 capability=chat 的条目"),
    ] = None,
    embedding_model: Annotated[
        str | None,
        typer.Option(
            "--embedding-model",
            help="Embedding 模型别名，对应 configs/models.yaml 中 capability=embedding 的条目",
        ),
    ] = None,
    reranker_model: Annotated[
        str | None,
        typer.Option(
            "--reranker-model",
            help="Reranker 模型别名，对应 configs/models.yaml 中 capability=reranker 的条目",
        ),
    ] = None,
) -> None:
    """单次 Agent 运行。传入 --checkpoint-db 后支持跨进程恢复。"""
    from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageComponentConfig, StorageConfig
    from rag.models.assembly_adapter import to_assembly_overrides
    from rag.models.runtime import RuntimeOverrides, resolve_runtime_config
    from rag.retrieval import QueryOptions

    runtime_config = resolve_runtime_config(
        RuntimeOverrides(
            model_alias=model,
            embedding_model_alias=embedding_model,
            reranker_model_alias=reranker_model,
        )
    )
    assembly_overrides = to_assembly_overrides(runtime_config)

    storage = StorageConfig(
        root=storage_root,
        vectors=StorageComponentConfig(backend="milvus", dsn="http://127.0.0.1:19530"),
    )
    requirements = CapabilityRequirements(
        require_chat=True,
        default_context_tokens=QueryOptions().max_context_tokens,
    )
    runtime = RAGRuntime.from_request(
        storage=storage,
        request=AssemblyRequest(requirements=requirements, overrides=assembly_overrides),
    )

    with runtime:
        service = _build_agent_service(runtime, checkpoint_db=checkpoint_db, agent_type=agent)
        effective_run_id = run_id or f"run_{id(service):x}"
        result = asyncio.run(
            service.run(
                AgentRunRequest(
                    task=task,
                    run_id=effective_run_id,
                    thread_id=effective_run_id,
                )
            )
        )

        _display_result(result, verbose=verbose)

        if result.status == "paused":
            print()
            if checkpoint_db is None:
                print("⚠  当前命令使用 MemorySaver，进程结束后暂停状态无法恢复。")
                print("   请使用 --checkpoint-db 启用 SQLite checkpoint 后重试。")
            else:
                print("⏸  已保存 checkpoint，可跨进程恢复:")
                print(
                    f"   rag agent resume {effective_run_id} "
                    f"--agent {agent} "
                    f"--checkpoint-db {checkpoint_db}"
                )

            if result.needs_user_input:
                print(f"\n   待处理: {result.needs_user_input}")

            if non_interactive:
                raise typer.Exit(code=2)

        if result.status == "failed":
            raise typer.Exit(code=1)


@agent_app.command(name="resume")
def agent_resume(
    run_id: Annotated[str, typer.Argument(help="要恢复的 run_id/thread_id")],
    storage_root: Annotated[
        Path, typer.Option("--storage-root", help="RAG 存储根目录")
    ] = Path(".rag"),
    agent: Annotated[
        str,
        typer.Option(
            "--agent",
            help="恢复时使用的根 Agent，必须与原 run 一致：research, orchestrator, compare, factcheck。",
        ),
    ] = "research",
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
        ),
    ] = None,
    reranker_model: Annotated[
        str | None,
        typer.Option(
            "--reranker-model",
            help="Reranker 模型别名，对应 configs/models.yaml 中 capability=reranker 的条目",
        ),
    ] = None,
) -> None:
    """从 SQLite checkpoint 恢复暂停的 Agent 运行。"""
    from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageComponentConfig, StorageConfig
    from rag.models.assembly_adapter import to_assembly_overrides
    from rag.models.runtime import RuntimeOverrides, resolve_runtime_config
    from rag.retrieval import QueryOptions

    runtime_config = resolve_runtime_config(
        RuntimeOverrides(
            model_alias=model,
            embedding_model_alias=embedding_model,
            reranker_model_alias=reranker_model,
        )
    )
    assembly_overrides = to_assembly_overrides(runtime_config)

    storage = StorageConfig(
        root=storage_root,
        vectors=StorageComponentConfig(backend="milvus", dsn="http://127.0.0.1:19530"),
    )
    requirements = CapabilityRequirements(
        require_chat=True,
        default_context_tokens=QueryOptions().max_context_tokens,
    )
    runtime = RAGRuntime.from_request(
        storage=storage,
        request=AssemblyRequest(requirements=requirements, overrides=assembly_overrides),
    )

    with runtime:
        service = _build_agent_service(runtime, checkpoint_db=checkpoint_db, agent_type=agent)
        request = asyncio.run(service.apending_human_input_request(run_id=run_id))
        response = _build_resume_response(request, decision)
        result = asyncio.run(service.resume(run_id=run_id, response=response))
        _display_result(result, verbose=verbose)
        if result.status == "failed":
            raise typer.Exit(code=1)
