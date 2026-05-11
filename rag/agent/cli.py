from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from rag.agent.builtin.research import create_research_agent_service
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.service import AgentRunRequest, AgentRunResult, AgentService
from rag.agent.tools.llm_tools import (
    LLMCompareInput,
    LLMGenerateInput,
    LLMSummarizeInput,
    LLMTextOutput,
)
from rag.agent.tools.rag_tools import SearchInput, SearchOutput
from rag.agent.tools.registry import ToolRunner

agent_app = typer.Typer(add_completion=False, no_args_is_help=True)


def _build_agent_service(runtime) -> AgentService:
    """从 RAGRuntime 构造 AgentService，注册真实 RAG tool runners。

    只在可以构造 RAGRuntime / retrieval_service 时成功。
    无法构造时报错，不静默使用 stub。
    """
    chat_bindings = list(runtime.capability_bundle.chat_bindings)
    primary_chat = chat_bindings[0] if chat_bindings else None

    runners: dict[str, ToolRunner] = {}

    # RAG tools — 统一通过 runtime.query() 执行检索
    def _run_rag(payload: SearchInput) -> SearchOutput:
        from rag.retrieval import QueryOptions
        result = runtime.query(payload.query, options=QueryOptions(max_context_tokens=4096))
        items: list[dict[str, object]] = []
        if hasattr(result, "evidence") and result.evidence:
            for item in result.evidence:
                items.append({"text": getattr(item, "text", ""), "score": getattr(item, "score", 0.0)})
        if not items and hasattr(result, "answer"):
            for section in getattr(result.answer, "answer_sections", []):
                text = getattr(section, "text", "")
                if text:
                    items.append({"text": text})
        return SearchOutput(items=items)

    for name in ("vector_search", "keyword_search", "grounding", "rerank", "graph_expand"):
        runners[name] = _run_rag

    # LLM tools
    if primary_chat is not None:

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

        runners["llm_generate"] = _llm_generate
        runners["llm_summarize"] = _llm_summarize
        runners["llm_compare"] = _llm_compare

    return create_research_agent_service(runners=runners)


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
        decision=choice,
        approved_tool_call_ids=approved,
        denied_tool_call_ids=denied,
    )


def _print_startup_banner(model_alias: str) -> None:
    print(f"Agent 就绪 (模型: {model_alias})")
    print("输入查询，或 /exit 退出，/verbose 切换详细输出")
    print()


# ── CLI Commands ──


@agent_app.command(name="chat")
def agent_chat(
    storage_root: Annotated[
        Path, typer.Option("--storage-root", help="RAG 存储根目录")
    ] = Path(".rag"),
    profile: Annotated[
        str | None, typer.Option("--profile", help="Assembly profile")
    ] = None,
) -> None:
    """交互式 Agent 对话。暂停时支持工具审批。"""
    from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageComponentConfig, StorageConfig
    from rag.retrieval import QueryOptions

    # 构建 RAGRuntime
    storage = StorageConfig(
        root=storage_root,
        vector=StorageComponentConfig(backend="milvus", dsn="http://127.0.0.1:19530"),
    )
    requirements = CapabilityRequirements(
        require_chat=True,
        default_context_tokens=QueryOptions().max_context_tokens,
    )
    if profile:
        runtime = RAGRuntime.from_profile(storage=storage, profile_id=profile, requirements=requirements)
    else:
        runtime = RAGRuntime.from_request(
            storage=storage,
            request=AssemblyRequest(requirements=requirements),
        )

    with runtime:
        service = _build_agent_service(runtime)
        run_id = f"chat_{id(service):x}"
        verbose = False

        _print_startup_banner("local_main")

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
    profile: Annotated[
        str | None, typer.Option("--profile", help="Assembly profile")
    ] = None,
    non_interactive: Annotated[
        bool, typer.Option("--non-interactive", help="非交互模式")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="详细输出")
    ] = False,
) -> None:
    """单次 Agent 运行。不支持跨进程恢复。"""
    from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageComponentConfig, StorageConfig
    from rag.retrieval import QueryOptions

    storage = StorageConfig(
        root=storage_root,
        vector=StorageComponentConfig(backend="milvus", dsn="http://127.0.0.1:19530"),
    )
    requirements = CapabilityRequirements(
        require_chat=True,
        default_context_tokens=QueryOptions().max_context_tokens,
    )
    if profile:
        runtime = RAGRuntime.from_profile(storage=storage, profile_id=profile, requirements=requirements)
    else:
        runtime = RAGRuntime.from_request(
            storage=storage,
            request=AssemblyRequest(requirements=requirements),
        )

    with runtime:
        service = _build_agent_service(runtime)
        run_id = f"run_{id(service):x}"
        result = asyncio.run(
            service.run(AgentRunRequest(task=task, run_id=run_id, thread_id=run_id))
        )

        _display_result(result, verbose=verbose)

        if result.status == "paused":
            print()
            print("⚠  当前命令使用 MemorySaver，进程结束后暂停状态无法恢复。")
            print("   请使用 `rag agent chat` 重新发起同一任务并在同一会话内完成审批。")
            print("   如需跨进程恢复，请先启用 SqliteSaver/PostgresSaver。")

            if result.needs_user_input:
                print(f"\n   待处理: {result.needs_user_input}")

            if non_interactive:
                raise typer.Exit(code=2)

        if result.status == "failed":
            raise typer.Exit(code=1)


@agent_app.command(name="resume", hidden=True)
def agent_resume() -> None:
    """（暂未开放）MemorySaver 不支持跨进程 resume。"""
    print("错误: MemorySaver 不支持跨进程 resume。")
    print("请使用 `rag agent chat` 重新发起查询并在同一会话内完成交互。")
    raise typer.Exit(code=1)
