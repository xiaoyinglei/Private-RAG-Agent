#!/usr/bin/env python3
"""体验 LLM 驱动的 route / evaluate / plan 决策。

用法:
  uv run python scripts/demo_llm_agent.py Qwen/Qwen3-8B-MLX-4bit
  uv run python scripts/demo_llm_agent.py mlx-community/Qwen3-14B-4bit
"""
from __future__ import annotations

import sys
from typing import Any

from rag.agent.core.definition import AgentDefinition, ModelSelectionPolicy
from rag.agent.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from rag.agent.core.llm_providers import (
    LLMEvaluateDecisionProvider,
    LLMRouteProvider,
)
from rag.agent.core.llm_prompts import build_evaluate_prompt, build_route_prompt
from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.memory.models import ContextBudgetSnapshot, ContextSection, InjectedContext
from rag.agent.state import ThinkOutput


def _make_config(use_model: str) -> AgentModelsConfig:
    spec = ModelSpec(
        provider=ModelProvider.MLX,
        model=use_model,
        max_tokens=2048,
        defaults={"temperature": 0.0, "top_p": 0.9},
    )
    return AgentModelsConfig(
        models={"main": spec, "fast": spec},
        default_model="main",
        fallback_model="fast",
    )


def _make_state(task: str, **overrides: Any) -> dict[str, Any]:
    from rag.agent.core.context import AgentRunConfig
    from rag.schema.runtime import AccessPolicy, ExecutionLocationPreference

    state: dict[str, Any] = {
        "messages": [],
        "evidence": [],
        "citations": [],
        "tool_results": [],
        "task": task,
        "retrieval_signals": None,
        "run_config": AgentRunConfig(
            run_id="demo", thread_id="demo", budget_total=10000, max_depth=2,
            access_policy=AccessPolicy.default(),
            execution_location_preference=ExecutionLocationPreference.LOCAL_FIRST,
        ),
        "plan": None, "iteration": 0, "status": "running",
        "route_reason": None, "stop_reason": None, "needs_user_input": None,
        "pending_tool_calls": [], "confirmed_tool_call_ids": set(),
        "user_decision": None, "next_subtasks": None,
        "working_summary": None, "extracted_facts": [], "context_budget": None,
        "subtask_results": {}, "terminal_subtasks": set(), "successful_subtasks": set(),
        "final_answer": None, "groundedness_flag": False, "insufficient_evidence_flag": False,
    }
    state.update(overrides)
    return state


def _make_context() -> InjectedContext:
    return InjectedContext(
        sections=[
            ContextSection(name="system", content="你是知识库问答 Agent。", token_count=10, required=True),
            ContextSection(name="task", content="test task", token_count=10, required=True),
            ContextSection(
                name="evidence",
                content="- E1 ref=Doc-1 score=0.9 text: 公积金缴纳比例为 12%\n- E2 ref=Doc-2 score=0.7 text: 公积金缴纳比例为 5%",
                token_count=50, required=True,
            ),
        ],
        context_budget=ContextBudgetSnapshot(max_context_tokens=4096),
    )


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B-MLX-4bit"
    print(f"使用模型: {model_name}\n加载中...")

    config = _make_config(model_name)
    registry = ModelRegistry(config)
    resolved = registry.resolve("main")
    gen = resolved.generator
    kwargs = resolved.kwargs

    route_provider = LLMRouteProvider(gen, kwargs=kwargs)
    eval_provider = LLMEvaluateDecisionProvider(gen, kwargs=kwargs)

    print("模型就绪。\n" + "=" * 60)

    # ── 测试路由 ──
    test_queries = [
        "公积金缴纳比例是多少",
        "对比 A 制度和 B 制度在公积金提取条件上的差异",
        "今年的员工福利政策有哪些变化，需要分别查询住房补贴和交通补贴",
    ]

    for query in test_queries:
        print(f"\n📝 查询: {query}")
        raw = gen.generate_text(
            prompt=build_route_prompt(_make_state(query)),
            max_tokens=256, temperature=0.0,
        )
        print(f"   LLM 原始输出: {raw.strip()[:200]}")
        route_result = route_provider.route(_make_state(query))
        print(f"   路由决策: {route_result['status']} ({route_result.get('route_reason', '')})")

    # ── 测试评估 ──
    eval_context = _make_context()
    eval_state = _make_state("公积金缴纳比例是多少")
    eval_prompt = build_evaluate_prompt(
        eval_state, budget_remaining=5000, context_text=eval_context.as_text(),
    )

    print("\n" + "=" * 60)
    print("\n🧠 评估测试: 有两条冲突证据时")
    raw_eval = gen.generate_text(prompt=eval_prompt, max_tokens=512, temperature=0.0)
    print(f"   LLM 原始输出: {raw_eval.strip()[:300]}")

    eval_result = eval_provider.decide(
        eval_state,
        definition=AgentDefinition(
            agent_type="research", description="test", system_prompt="test",
            allowed_tools=["vector_search", "grounding"],
        ),
        budget_remaining=5000,
        context=eval_context,
    )
    if isinstance(eval_result, ThinkOutput):
        print(f"\n   解析结果:")
        print(f"   action: {eval_result.action}")
        print(f"   thought: {eval_result.thought[:100]}...")
        print(f"   confidence: {eval_result.confidence}")
        if eval_result.tool_calls:
            for tc in eval_result.tool_calls:
                print(f"   tool: {tc.tool_name}({tc.arguments})")

    print("\n" + "=" * 60)
    print("✅ 完成。")


if __name__ == "__main__":
    main()
