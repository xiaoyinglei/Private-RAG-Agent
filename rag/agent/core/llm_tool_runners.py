from __future__ import annotations

from typing import cast
from uuid import uuid4

from rag.agent.core.context import RunRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.llm_context import AgentLLMContextAssembler
from rag.agent.core.llm_registry import ModelRegistry, ResolvedModel
from rag.agent.loop.state import LoopState
from rag.agent.tools.llm_tools import (
    LLMCompareInput,
    LLMGenerateInput,
    LLMSummarizeInput,
    LLMTextOutput,
)
from rag.agent.tools.registry import ContextualToolRunner, ToolExecutionContext
from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract
from rag.providers.llm_gateway import LLMGateway
from rag.schema.llm import DEFAULT_LLM_STAGE_BUDGETS, LLMCallStage


def create_model_llm_tool_runners(
    registry: ModelRegistry,
) -> dict[str, ContextualToolRunner]:
    """Create model-backed runners for llm_* tools.

    These runners are request-scoped glue for the existing LLM tool specs. They
    preserve explicit grounding ids supplied by the caller and fail visibly if
    the configured model cannot generate text.
    """

    async def _llm_generate(
        payload: LLMGenerateInput,
        execution_context: ToolExecutionContext,
    ) -> LLMTextOutput:
        state, definition = _trusted_agent_context(execution_context)
        effective_stage = (
            LLMCallStage.FINAL_SYNTHESIS
            if execution_context.run_config.agent_type == "synthesize"
            else LLMCallStage.LLM_GENERATE
        )
        gateway, resolved = _model_boundary(
            registry,
            node_name="llm_generate",
            stage=effective_stage,
        )
        assembled = AgentLLMContextAssembler(
            token_accounting=gateway.token_accounting,
            stage_budgets={
                effective_stage: gateway.effective_stage_budget(effective_stage),
            },
            formatter_resolver=None,
        ).assemble_generate(
            definition=definition,
            state=state,
            prompt=payload.prompt,
            context_sections=payload.context_sections,
            stage=effective_stage,
        )
        return LLMTextOutput(
            text=await _generate_text(
                gateway=gateway,
                resolved=resolved,
                node_name="llm_generate",
                stage=effective_stage,
                prompt=assembled.prompt,
                execution_context=execution_context,
            ),
            evidence_ids=payload.evidence_ids,
            citation_ids=payload.citation_ids,
        )

    async def _llm_summarize(
        payload: LLMSummarizeInput,
        execution_context: ToolExecutionContext,
    ) -> LLMTextOutput:
        state, definition = _trusted_agent_context(execution_context)
        gateway, resolved = _model_boundary(
            registry,
            node_name="llm_summarize",
            stage=LLMCallStage.LLM_SUMMARIZE,
        )
        assembled = AgentLLMContextAssembler(
            token_accounting=gateway.token_accounting,
            stage_budgets={
                LLMCallStage.LLM_SUMMARIZE: gateway.effective_stage_budget(LLMCallStage.LLM_SUMMARIZE),
            },
            formatter_resolver=None,
        ).assemble_summarize(
            definition=definition,
            state=state,
            task=payload.task,
            context_sections=payload.context_sections,
        )
        return LLMTextOutput(
            text=await _generate_text(
                gateway=gateway,
                resolved=resolved,
                node_name="llm_summarize",
                stage=LLMCallStage.LLM_SUMMARIZE,
                prompt=assembled.prompt,
                execution_context=execution_context,
            ),
            evidence_ids=payload.evidence_ids,
            citation_ids=payload.citation_ids,
        )

    async def _llm_compare(
        payload: LLMCompareInput,
        execution_context: ToolExecutionContext,
    ) -> LLMTextOutput:
        state, definition = _trusted_agent_context(execution_context)
        gateway, resolved = _model_boundary(
            registry,
            node_name="llm_compare",
            stage=LLMCallStage.LLM_COMPARE,
        )
        assembled = AgentLLMContextAssembler(
            token_accounting=gateway.token_accounting,
            stage_budgets={
                LLMCallStage.LLM_COMPARE: gateway.effective_stage_budget(LLMCallStage.LLM_COMPARE),
            },
            formatter_resolver=None,
        ).assemble_compare(
            definition=definition,
            state=state,
            question=payload.question,
            left_context_sections=payload.left_context_sections,
            right_context_sections=payload.right_context_sections,
        )
        return LLMTextOutput(
            text=await _generate_text(
                gateway=gateway,
                resolved=resolved,
                node_name="llm_compare",
                stage=LLMCallStage.LLM_COMPARE,
                prompt=assembled.prompt,
                execution_context=execution_context,
            ),
            evidence_ids=payload.evidence_ids,
            citation_ids=payload.citation_ids,
        )

    return {
        "llm_generate": cast(ContextualToolRunner, _llm_generate),
        "llm_summarize": cast(ContextualToolRunner, _llm_summarize),
        "llm_compare": cast(ContextualToolRunner, _llm_compare),
    }


async def _generate_text(
    *,
    gateway: LLMGateway,
    resolved: ResolvedModel,
    node_name: str,
    stage: LLMCallStage,
    prompt: str,
    execution_context: ToolExecutionContext,
) -> str:
    run_config = execution_context.run_config
    try:
        ledger = RunRegistry.get(run_config.run_id).budget_ledger
    except KeyError as exc:
        raise RuntimeError(f"Runtime handles missing for run_id={run_config.run_id}") from exc
    result = await gateway.agenerate_text(
        stage=stage,
        prompt=prompt,
        ledger=ledger,
        lease_id=(execution_context.tool_call_id or f"{run_config.run_id}:{node_name}:{uuid4().hex}"),
        kwargs={key: value for key, value in getattr(resolved, "kwargs", {}).items() if key != "max_tokens"},
    )
    return result.value


def _model_boundary(
    registry: ModelRegistry,
    *,
    node_name: str,
    stage: LLMCallStage,
) -> tuple[LLMGateway, ResolvedModel]:
    resolved = registry.resolve_for_node(node_model=None, node_name=node_name)
    gateway = getattr(resolved, "gateway", None)
    if gateway is not None:
        return gateway, resolved
    model_context_tokens = getattr(resolved, "context_window_tokens", 32_768)
    token_accounting = getattr(resolved, "token_accounting", None)
    if token_accounting is None:
        token_accounting = TokenAccountingService(
            TokenizerContract(
                embedding_model_name="agent-fallback",
                tokenizer_model_name="agent-fallback",
                chunking_tokenizer_model_name="agent-fallback",
                tokenizer_backend="simple",
                max_context_tokens=model_context_tokens,
                prompt_reserved_tokens=512,
                local_files_only=True,
            )
        )
    return (
        LLMGateway(
            generator=resolved.generator,
            token_accounting=token_accounting,
            model_context_tokens=model_context_tokens,
            stage_budgets={
                stage: DEFAULT_LLM_STAGE_BUDGETS[stage],
            },
        ),
        resolved,
    )


def _trusted_agent_context(
    execution_context: ToolExecutionContext,
) -> tuple[LoopState, AgentRuntimePolicy]:
    if execution_context.state is None or execution_context.definition is None:
        raise RuntimeError("Agent LLM tools require trusted LoopState and AgentRuntimePolicy")
    return execution_context.state, execution_context.definition


__all__ = ["create_model_llm_tool_runners"]
