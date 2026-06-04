from __future__ import annotations

from typing import cast

from rag.agent.core.llm_registry import ModelRegistry
from rag.agent.tools.llm_tools import (
    LLMCompareInput,
    LLMGenerateInput,
    LLMSummarizeInput,
    LLMTextOutput,
)
from rag.agent.tools.registry import ToolRunner


def create_model_llm_tool_runners(registry: ModelRegistry) -> dict[str, ToolRunner]:
    """Create model-backed runners for llm_* tools.

    These runners are request-scoped glue for the existing LLM tool specs. They
    preserve explicit grounding ids supplied by the caller and fail visibly if
    the configured model cannot generate text.
    """

    def _llm_generate(payload: LLMGenerateInput) -> LLMTextOutput:
        prompt = payload.prompt
        if payload.context_sections:
            prompt += "\n\n上下文:\n" + "\n".join(payload.context_sections)
        return LLMTextOutput(
            text=_generate_text(registry, node_name="llm_generate", prompt=prompt),
            evidence_ids=payload.evidence_ids,
            citation_ids=payload.citation_ids,
        )

    def _llm_summarize(payload: LLMSummarizeInput) -> LLMTextOutput:
        prompt = (
            "请只基于给定上下文回答任务。不要编造上下文中没有的信息；"
            "如果上下文显示工具已经生成报告或计算结果，请直接给出结果和相关路径。\n\n"
            f"任务:\n{payload.task}"
        )
        if payload.context_sections:
            prompt += "\n\n上下文:\n" + "\n".join(payload.context_sections)
        return LLMTextOutput(
            text=_generate_text(registry, node_name="llm_summarize", prompt=prompt),
            evidence_ids=payload.evidence_ids,
            citation_ids=payload.citation_ids,
        )

    def _llm_compare(payload: LLMCompareInput) -> LLMTextOutput:
        prompt = payload.question
        if payload.left_context_sections or payload.right_context_sections:
            prompt += "\n\n左:\n" + "\n".join(payload.left_context_sections)
            prompt += "\n\n右:\n" + "\n".join(payload.right_context_sections)
        return LLMTextOutput(
            text=_generate_text(registry, node_name="llm_compare", prompt=prompt),
            evidence_ids=payload.evidence_ids,
            citation_ids=payload.citation_ids,
        )

    return {
        "llm_generate": cast(ToolRunner, _llm_generate),
        "llm_summarize": cast(ToolRunner, _llm_summarize),
        "llm_compare": cast(ToolRunner, _llm_compare),
    }


def _generate_text(registry: ModelRegistry, *, node_name: str, prompt: str) -> str:
    resolved = registry.resolve_for_node(node_model=None, node_name=node_name)
    kwargs = dict(resolved.kwargs)
    generator = resolved.generator
    generate_text = getattr(generator, "generate_text", None)
    if callable(generate_text):
        return str(generate_text(prompt=prompt, **kwargs))
    chat = getattr(generator, "chat", None)
    if callable(chat):
        return str(chat(prompt, **kwargs))
    raise RuntimeError(f"Configured model for {node_name} cannot generate text")


__all__ = ["create_model_llm_tool_runners"]
