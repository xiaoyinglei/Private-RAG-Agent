from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from rag.ingest.table_executor import ComputeResult, TableExecutor
from rag.providers.citation_formatter import CitationFormatter
from rag.providers.generation import AnswerGenerationResult, AnswerGenerator
from rag.retrieval.authorization_service import AuthorizationService
from rag.retrieval.context import (
    ContextPromptBuilder,
    ContextPromptBuildResult,
    ContextTruncationResult,
    EvidenceTruncator,
)
from rag.retrieval.evidence import ContextEvidenceMerger
from rag.retrieval.grounding_service import GroundingService
from rag.retrieval.models import (
    BuiltContext,
    PublicQueryResult,
    QueryOptions,
    RAGQueryResult,
    RetrievalProfile,
    RetrievalResult,
)
from rag.retrieval.orchestrator import RetrievalService
from rag.retrieval.runtime_coordinator import (
    CoreRetrievalPayload,
    RuntimeCoordinator,
    build_retrieval_diagnostics,
    to_retrieval_result,
)
from rag.retrieval.synthesis_service import SynthesisService
from rag.schema.query import EvidenceItem
from rag.schema.runtime import AccessPolicy
from rag.utils.guard import RateLimitExceeded


@dataclass(slots=True, frozen=True)
class _LLMFallbackResult:
    answer: object = field(
        default_factory=lambda: type(
            "_FallbackAnswer",
            (),
            {"answer_text": "抱歉，回答生成服务暂时不可用，请稍后重试。"},
        )()
    )
    provider: str = "fallback"
    model: str = "none"
    attempts: int = 1


@dataclass(slots=True, frozen=True)
class _QueryExecutionResult:
    retrieval_payload: CoreRetrievalPayload | None
    retrieval: RetrievalResult
    generated: AnswerGenerationResult
    context: BuiltContext


@dataclass(slots=True)
class _QueryPipeline:
    retrieval: RetrievalService
    context_merger: ContextEvidenceMerger
    grounding_service: GroundingService | object
    truncator: EvidenceTruncator
    prompt_builder: ContextPromptBuilder
    answer_generator: AnswerGenerator
    synthesis_service: SynthesisService | object | None = None
    authorization_service: AuthorizationService | object | None = None
    table_executor: TableExecutor | object | None = None
    rate_limiter: object | None = None
    llm_circuit_breaker: object | None = None
    _compute_executed: bool = field(default=False, init=False, repr=False)

    _citation_formatter: CitationFormatter = field(default_factory=CitationFormatter, init=False, repr=False)

    @staticmethod
    def _run_async(awaitable: Any) -> Any:
        return RuntimeCoordinator().run_sync(awaitable)

    def _render_answer(self, generated: AnswerGenerationResult) -> AnswerGenerationResult:
        answer = getattr(generated, "answer", None)
        if answer is None:
            return generated
        formatted = self._citation_formatter.format(answer)
        rendered = answer.model_copy(update={"answer_text": formatted.answer_text})
        return replace(generated, answer=rendered)

    def _generate_with_breaker(self, awaitable: Any) -> Any:
        breaker = self.llm_circuit_breaker
        if breaker is not None and not cast(Any, breaker).allow():
            _logger = logging.getLogger("rag.runtime")
            _logger.warning("LLM circuit breaker open, returning fallback")
            return _LLMFallbackResult()
        try:
            result = self._run_async(awaitable)
        except Exception:
            if breaker is not None:
                cast(Any, breaker).on_failure()
            raise
        else:
            if breaker is not None:
                cast(Any, breaker).on_success()
            return result

    def run(
        self,
        query: str,
        *,
        options: QueryOptions,
    ) -> RAGQueryResult:
        result = self._execute_query(query=query, options=options)
        rendered = self._render_answer(result.generated)
        return RAGQueryResult(
            query=query,
            retrieval_profile=options.resolved_retrieval_profile.value,
            answer=rendered.answer,
            retrieval=result.retrieval,
            context=result.context,
            generation_provider=result.generated.provider,
            generation_model=result.generated.model,
            generation_attempts=result.generated.attempts,
        )

    def run_public(
        self,
        query: str,
        *,
        options: QueryOptions,
    ) -> PublicQueryResult:
        result = self._execute_query(query=query, options=options)
        rendered = self._render_answer(result.generated)
        return PublicQueryResult(
            query=query,
            retrieval_profile=options.resolved_retrieval_profile.value,
            answer=rendered.answer,
            context=result.context,
            routing_decision=result.retrieval.decision.model_dump(mode="json"),
            retrieval_diagnostics=(
                build_retrieval_diagnostics(result.retrieval_payload)
                if result.retrieval_payload is not None
                else result.retrieval.diagnostics
            ),
            retrieval_self_check=result.retrieval.self_check.model_dump(mode="json"),
            generation_provider=result.generated.provider,
            generation_model=result.generated.model,
            generation_attempts=result.generated.attempts,
        )

    def _execute_query(self, *, query: str, options: QueryOptions) -> _QueryExecutionResult:
        self._enforce_rate_limit(options)
        access_policy, source_scope = self._resolve_query_scope(options)
        retrieval_payload = self._retrieve_payload(
            query=query,
            access_policy=access_policy,
            source_scope=source_scope,
            options=options,
        )
        retrieval = (
            to_retrieval_result(retrieval_payload)
            if retrieval_payload is not None
            else self.retrieval.retrieve(
                query,
                access_policy=access_policy,
                source_scope=source_scope,
                query_options=options,
            )
        )
        if options.resolved_retrieval_profile is RetrievalProfile.BYPASS:
            return self._execute_direct_query(
                query=query,
                options=options,
                access_policy=access_policy,
                retrieval_payload=retrieval_payload,
                retrieval=retrieval,
            )
        return self._execute_grounded_query(
            query=query,
            options=options,
            access_policy=access_policy,
            retrieval_payload=retrieval_payload,
            retrieval=retrieval,
        )

    def _enforce_rate_limit(self, options: QueryOptions) -> None:
        if self.rate_limiter is None:
            return
        user_id = options.user_id or "anonymous"
        if not cast(Any, self.rate_limiter).allow(user_id=user_id):
            raise RateLimitExceeded(f"rate limit exceeded for user '{user_id}'")

    def _execute_direct_query(
        self,
        *,
        query: str,
        options: QueryOptions,
        access_policy: AccessPolicy,
        retrieval_payload: CoreRetrievalPayload | None,
        retrieval: RetrievalResult,
    ) -> _QueryExecutionResult:
        prompt = self.prompt_builder.answer_generation_service.build_direct_prompt(
            query=query,
            response_type=options.response_type,
            user_prompt=options.user_prompt,
            conversation_history=options.conversation_history,
        )
        generated = self._generate_with_breaker(
            self.answer_generator.generate_direct(
                query=query,
                prompt=prompt,
                access_policy=access_policy,
            )
        )
        return _QueryExecutionResult(
            retrieval_payload=retrieval_payload,
            retrieval=retrieval,
            generated=generated,
            context=BuiltContext(
                evidence=[],
                token_budget=options.max_context_tokens,
                token_count=self.prompt_builder.token_accounting.count(prompt),
                truncated_count=0,
                grounded_candidate="Bypass mode does not use retrieved evidence.",
                prompt=prompt,
            ),
        )

    def _execute_grounded_query(
        self,
        *,
        query: str,
        options: QueryOptions,
        access_policy: AccessPolicy,
        retrieval_payload: CoreRetrievalPayload | None,
        retrieval: RetrievalResult,
    ) -> _QueryExecutionResult:
        merged_evidence = self.context_merger.merge(retrieval_payload or retrieval)
        grounding_service = getattr(self, "grounding_service", None)
        if grounding_service is not None and callable(getattr(grounding_service, "ground", None)):
            merged_evidence = list(grounding_service.ground(query=query, evidence=merged_evidence))
        merged_evidence = self._section_diversity_filter(merged_evidence)
        synthesis_service = getattr(self, "synthesis_service", None)
        if synthesis_service is not None and callable(getattr(synthesis_service, "filter_evidence", None)):
            merged_evidence = list(
                synthesis_service.filter_evidence(
                    evidence=merged_evidence,
                    access_policy=access_policy,
                    user_id=options.user_id,
                )
            )
        total_budget = max(options.max_context_tokens, 1)
        evidence_budget = self.truncator.token_accounting.prompt_budget(total_budget)
        truncated, prompt_build = self._build_bounded_context(
            query=query,
            options=options,
            retrieval=retrieval,
            merged_evidence=merged_evidence,
            total_budget=total_budget,
            evidence_budget=evidence_budget,
        )
        context_evidence_items = [item.as_evidence_item() for item in truncated.evidence]
        generated = self._generate_with_breaker(
            self.answer_generator.generate(
                query=query,
                prompt=prompt_build.prompt,
                evidence_pack=context_evidence_items,
                grounded_candidate=prompt_build.grounded_candidate,
                runtime_mode=retrieval.decision.runtime_mode,
                access_policy=access_policy,
            )
        )
        generated, merged_evidence, truncated, prompt_build = self._maybe_execute_compute_loop(
            generated=generated,
            merged_evidence=merged_evidence,
            prompt_evidence=truncated.evidence,
            grounded_candidate=prompt_build.grounded_candidate,
            query=query,
            options=options,
            retrieval=retrieval,
            total_budget=total_budget,
            evidence_budget=evidence_budget,
            access_policy=access_policy,
        )
        generated = cast(AnswerGenerationResult, generated)
        return _QueryExecutionResult(
            retrieval_payload=retrieval_payload,
            retrieval=retrieval,
            generated=generated,
            context=BuiltContext(
                evidence=truncated.evidence,
                token_budget=total_budget,
                token_count=prompt_build.token_count,
                truncated_count=truncated.truncated_count,
                grounded_candidate=prompt_build.grounded_candidate,
                prompt=prompt_build.prompt,
            ),
        )

    def _resolve_query_scope(self, options: QueryOptions) -> tuple[AccessPolicy, tuple[str, ...]]:
        access_policy = options.access_policy
        source_scope = options.source_scope
        authorization_service = getattr(self, "authorization_service", None)
        if authorization_service is not None and callable(getattr(authorization_service, "resolve_query", None)):
            auth_context = authorization_service.resolve_query(
                user_id=options.user_id,
                access_policy=options.access_policy,
                source_scope=options.source_scope,
            )
            access_policy = auth_context.access_policy
            source_scope = auth_context.source_scope
        return access_policy, source_scope

    _COMPUTE_REQUEST_PATTERN: re.Pattern[str] | None = field(default=None, init=False, repr=False)

    def _maybe_execute_compute_loop(
        self,
        *,
        generated: object,
        merged_evidence: list[EvidenceItem],
        prompt_evidence: Sequence[EvidenceItem],
        grounded_candidate: str,
        query: str,
        options: QueryOptions,
        retrieval: RetrievalResult,
        total_budget: int,
        evidence_budget: int,
        access_policy: AccessPolicy,
    ) -> tuple[object, list[EvidenceItem], ContextTruncationResult, ContextPromptBuildResult]:
        def _recontext(
            ev: list[EvidenceItem],
        ) -> tuple[list[EvidenceItem], ContextTruncationResult, ContextPromptBuildResult]:
            t, pb = self._build_bounded_context(
                query=query, options=options, retrieval=retrieval,
                merged_evidence=ev, total_budget=total_budget,
                evidence_budget=evidence_budget,
            )
            return ev, t, pb

        def _passthrough() -> tuple[object, list[EvidenceItem], ContextTruncationResult, ContextPromptBuildResult]:
            ev, t, pb = _recontext(merged_evidence)
            return generated, ev, t, pb

        if self._compute_executed:
            return _passthrough()

        executor = self.table_executor
        if executor is None or not hasattr(executor, "execute"):
            return _passthrough()

        answer_text = self._compute_request_search_text(generated)
        if not answer_text:
            return _passthrough()

        compute_request = self._parse_compute_request(answer_text, list(prompt_evidence))
        if compute_request is None:
            if not self._has_pending_compute_only_table(prompt_evidence):
                return _passthrough()
            retry_prompt = (
                self.prompt_builder.answer_generation_service.build_compute_request_prompt(
                    query=query,
                    evidence_pack=[item.as_evidence_item() for item in prompt_evidence],
                    grounded_candidate=grounded_candidate,
                    runtime_mode=retrieval.decision.runtime_mode,
                )
            )
            generated = self._generate_with_breaker(
                self.answer_generator.generate(
                    query=query,
                    prompt=retry_prompt,
                    evidence_pack=[item.as_evidence_item() for item in prompt_evidence],
                    grounded_candidate=grounded_candidate,
                    runtime_mode=retrieval.decision.runtime_mode,
                    access_policy=access_policy,
                )
            )
            answer_text = self._compute_request_search_text(generated)
            compute_request = self._parse_compute_request(answer_text, list(prompt_evidence))
            if compute_request is None:
                return _passthrough()
        asset_id, sql = compute_request

        if not sql or asset_id <= 0:
            return _passthrough()

        compute_result = executor.execute(asset_id=asset_id, sql=sql)
        self._compute_executed = True

        if compute_result is None:
            updated_evidence = self._strip_system_instructions(merged_evidence)
            ev, t, pb = _recontext(updated_evidence)
            return generated, ev, t, pb

        updated_evidence = self._inject_compute_result(
            merged_evidence, asset_id=asset_id, result=compute_result,
        )
        stripped_evidence = self._strip_system_instructions(updated_evidence)
        focused_evidence = self._focus_compute_result_context(
            stripped_evidence,
            asset_id=asset_id,
        )
        ev, t, pb = _recontext(focused_evidence)
        context_evidence_items = [item.as_evidence_item() for item in t.evidence]
        regenerated = self._generate_with_breaker(
            self.answer_generator.generate(
                query=query,
                prompt=pb.prompt,
                evidence_pack=context_evidence_items,
                grounded_candidate=pb.grounded_candidate,
                runtime_mode=retrieval.decision.runtime_mode,
                access_policy=access_policy,
            )
        )
        return regenerated, ev, t, pb

    def _compute_request_re(self) -> re.Pattern[str]:
        if self._COMPUTE_REQUEST_PATTERN is None:
            self._COMPUTE_REQUEST_PATTERN = re.compile(
                r"<compute_request>\s*(.*?)\s*</compute_request>", re.DOTALL
            )
        return self._COMPUTE_REQUEST_PATTERN

    @staticmethod
    def _has_pending_compute_only_table(evidence: Sequence[EvidenceItem]) -> bool:
        has_compute_only = any("[TABLE_COMPUTE_ONLY:" in item.text for item in evidence)
        has_compute_result = any("[TABLE_COMPUTE_RESULT:" in item.text for item in evidence)
        return has_compute_only and not has_compute_result

    def _parse_compute_request(
        self,
        answer_text: str,
        evidence: list[EvidenceItem],
    ) -> tuple[int, str] | None:
        match = self._compute_request_re().search(answer_text)
        if match is None:
            return self._parse_bare_select_compute_request(answer_text, evidence)
        body = self._strip_compute_request_body(match.group(1))
        if not body:
            return None

        asset_id = 0
        sql = ""
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            sql = body
        else:
            if isinstance(payload, dict):
                try:
                    asset_id = int(payload.get("asset_id", 0) or 0)
                except (TypeError, ValueError):
                    asset_id = 0
                sql = str(payload.get("sql", "") or "").strip()
            elif isinstance(payload, str):
                sql = payload.strip()
            else:
                return None

        if asset_id <= 0:
            inferred_asset_id = self._single_compute_asset_id(evidence)
            if inferred_asset_id is None:
                return None
            asset_id = inferred_asset_id
        sql = self._prepare_compute_sql(sql, evidence)
        if not sql:
            return None
        return asset_id, sql

    @staticmethod
    def _compute_request_search_text(generated: object) -> str:
        answer = getattr(generated, "answer", None)
        if answer is None:
            return ""
        parts: list[str] = []
        answer_text = getattr(answer, "answer_text", None)
        if isinstance(answer_text, str) and answer_text.strip():
            parts.append(answer_text)
        sections = getattr(answer, "answer_sections", None)
        if isinstance(sections, Sequence) and not isinstance(sections, (str, bytes)):
            for section in sections:
                section_text = getattr(section, "text", None)
                if isinstance(section_text, str) and section_text.strip():
                    parts.append(section_text)
        return "\n\n".join(dict.fromkeys(parts))

    def _parse_bare_select_compute_request(
        self,
        answer_text: str,
        evidence: list[EvidenceItem],
    ) -> tuple[int, str] | None:
        asset_id = self._single_compute_asset_id(evidence)
        if asset_id is None:
            return None
        sql = self._extract_single_select_statement(answer_text)
        if sql is None:
            return None
        sql = self._prepare_compute_sql(sql, evidence)
        if not sql:
            return None
        return asset_id, sql

    @staticmethod
    def _extract_single_select_statement(text: str) -> str | None:
        def _find_selects(candidate_text: str) -> list[str]:
            pattern = re.compile(
                r"\bSELECT\b.+?\bFROM\s+sheet\b.*?(?:;|(?=\n\s*\n)|$)",
                re.IGNORECASE | re.DOTALL,
            )
            results: list[str] = []
            for match in pattern.finditer(candidate_text):
                candidate = _QueryPipeline._clean_sql_fragment(match.group(0))
                if candidate:
                    results.append(candidate)
            return results

        fenced_results: list[str] = []
        for fence_match in re.finditer(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL):
            fenced_results.extend(_find_selects(fence_match.group(1)))
        results = fenced_results or _find_selects(text)
        unique_results = list(dict.fromkeys(results))
        if len(unique_results) != 1:
            return None
        return unique_results[0]

    @staticmethod
    def _clean_sql_fragment(text: str) -> str:
        sql = _QueryPipeline._strip_compute_request_body(text)
        sql = re.sub(r"\s*\[(?:\d+:\d+|\d+)(?:,\s*(?:\d+:\d+|\d+))*\]", "", sql)
        return sql.strip()

    @staticmethod
    def _prepare_compute_sql(sql: str, evidence: list[EvidenceItem]) -> str:
        cleaned = _QueryPipeline._clean_sql_fragment(sql)
        if not cleaned:
            return ""
        quoted = _QueryPipeline._quote_known_table_columns(cleaned, evidence)
        return quoted

    @staticmethod
    def _quote_known_table_columns(sql: str, evidence: list[EvidenceItem]) -> str:
        columns = _QueryPipeline._known_table_columns(evidence)
        if not columns:
            return sql
        quoted_sql = sql
        for column in sorted(columns, key=len, reverse=True):
            pattern = re.compile(rf"(?<![\"'\w]){re.escape(column)}(?![\"'\w])")
            quoted_sql = pattern.sub(f'"{column}"', quoted_sql)
        return quoted_sql

    @staticmethod
    def _known_table_columns(evidence: list[EvidenceItem]) -> tuple[str, ...]:
        columns: list[str] = []

        def _add(name: str) -> None:
            column = name.strip().strip('`"')
            if column and column not in columns:
                columns.append(column)

        for item in evidence:
            for line in item.text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("Table columns:"):
                    _, value = stripped.split(":", 1)
                    for part in value.split("|"):
                        _add(part)
                    continue
                if stripped.startswith("Schema:"):
                    _, value = stripped.split(":", 1)
                    for part in re.split(r"[|,]", value):
                        _add(part)
                    continue
                if "->" in stripped:
                    name, _type_hint = stripped.split("->", 1)
                    _add(name)
        return tuple(columns)

    @staticmethod
    def _strip_compute_request_body(text: str) -> str:
        body = text.strip()
        if not body.startswith("```"):
            return body
        lines = body.splitlines()
        if not lines:
            return body
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _single_compute_asset_id(evidence: list[EvidenceItem]) -> int | None:
        asset_ids = {
            int(match.group(1))
            for item in evidence
            for match in re.finditer(r"\[TABLE_COMPUTE_ONLY:asset_id=(\d+)\]", item.text)
        }
        if len(asset_ids) != 1:
            return None
        return next(iter(asset_ids))

    @staticmethod
    def _inject_compute_result(
        evidence: list[EvidenceItem],
        *,
        asset_id: int,
        result: ComputeResult,
    ) -> list[EvidenceItem]:
        anchor_pattern = re.compile(rf"\[TABLE_COMPUTE_ONLY:asset_id={asset_id}\]")
        updated: list[EvidenceItem] = []
        replaced = False
        for item in evidence:
            if not replaced and anchor_pattern.search(item.text):
                new_text = anchor_pattern.sub(result.markdown, item.text, count=1)
                new_text = _QueryPipeline._strip_compute_only_sample_rows(new_text)
                updated.append(
                    item.model_copy(
                        update={
                            "text": new_text,
                            "score": 1.0,
                            "retrieval_channels": [*item.retrieval_channels, "compute_result"],
                        }
                    )
                )
                replaced = True
            else:
                updated.append(item)
        if not replaced:
            if evidence:
                result_item = evidence[0].model_copy(
                    update={
                        "evidence_id": f"compute_result:{asset_id}",
                        "text": result.markdown,
                        "score": 1.0,
                        "retrieval_channels": ["compute_result"],
                    }
                )
            else:
                result_item = EvidenceItem(
                    evidence_id=f"compute_result:{asset_id}",
                    doc_id=0,
                    citation_anchor=f"table@{asset_id}",
                    text=result.markdown,
                    score=1.0,
                    retrieval_channels=["compute_result"],
                )
            updated.append(result_item)
        return updated

    @staticmethod
    def _strip_compute_only_sample_rows(text: str) -> str:
        lines = text.splitlines()
        cleaned: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.startswith("Sample rows ("):
                cleaned.append(line)
                index += 1
                continue

            index += 1
            while index < len(lines):
                current = lines[index]
                if not current.strip():
                    index += 1
                    break
                if current.startswith("|") or current.startswith("(table has "):
                    index += 1
                    continue
                break
        return "\n".join(cleaned)

    @staticmethod
    def _focus_compute_result_context(
        evidence: list[EvidenceItem],
        *,
        asset_id: int,
    ) -> list[EvidenceItem]:
        marker = f"[TABLE_COMPUTE_RESULT:asset_id={asset_id}]"
        result_items = [item for item in evidence if marker in item.text]
        if not result_items:
            return evidence
        return [
            item.model_copy(
                update={"text": _QueryPipeline._strip_post_compute_planning_context(item.text)}
            )
            for item in result_items
        ]

    @staticmethod
    def _strip_post_compute_planning_context(text: str) -> str:
        schema_pattern = re.compile(r"\nSchema:\s.*?(?=\n\S|\Z)", re.DOTALL)
        return schema_pattern.sub("", text).strip()

    @staticmethod
    def _section_diversity_filter(evidence: list[EvidenceItem], *, max_per_section: int = 2) -> list[EvidenceItem]:
        """post-grounding 章节多样性过滤：同章最多保留 max_per_section 条。"""
        if len(evidence) <= max_per_section:
            return evidence
        grouped: dict[str, list[EvidenceItem]] = {}
        for item in evidence:
            path = getattr(item, "section_path", None)
            if path and len(path) >= 2:
                key = f"{item.doc_id}:{path[0]}:{path[1]}"
            else:
                key = f"{item.doc_id}:{getattr(item, 'evidence_id', id(item))}"
            grouped.setdefault(key, []).append(item)
        result: list[EvidenceItem] = []
        for group in grouped.values():
            sorted_group = sorted(group, key=lambda x: getattr(x, "score", 0), reverse=True)
            result.extend(sorted_group[:max_per_section])
        return result

    @staticmethod
    def _strip_system_instructions(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        instruction_pattern = re.compile(
            r"<system_instruction>.*?</system_instruction>", re.DOTALL
        )
        replacement = (
            "[SYSTEM_NOTIFICATION] The data analysis query has been executed successfully. "
            "The results are injected below. You are STRICTLY FORBIDDEN from requesting "
            "further computations. Synthesize the final answer directly."
        )
        updated: list[EvidenceItem] = []
        for item in evidence:
            new_text = instruction_pattern.sub(replacement, item.text)
            if new_text != item.text:
                updated.append(item.model_copy(update={"text": new_text}))
            else:
                updated.append(item)
        return updated

    def _retrieve_payload(
        self,
        *,
        query: str,
        access_policy: AccessPolicy,
        source_scope: tuple[str, ...],
        options: QueryOptions,
    ) -> CoreRetrievalPayload | None:
        retrieve_payload = getattr(self.retrieval, "retrieve_payload", None)
        if not callable(retrieve_payload):
            return None
        return cast(CoreRetrievalPayload, retrieve_payload(
            query,
            access_policy=access_policy,
            source_scope=source_scope,
            query_options=options,
        ))

    def _build_bounded_context(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: RetrievalResult,
        merged_evidence: list[EvidenceItem],
        total_budget: int,
        evidence_budget: int,
    ) -> tuple[ContextTruncationResult, ContextPromptBuildResult]:
        current_budget = max(evidence_budget, 1)
        truncated = self._truncate_evidence(merged_evidence, budget=current_budget, options=options)
        truncated, prompt_build, current_budget = self._shrink_to_budget(
            query=query,
            options=options,
            retrieval=retrieval,
            merged_evidence=merged_evidence,
            total_budget=total_budget,
            current_budget=current_budget,
            truncated=truncated,
            prompt_variants=(("full", options.user_prompt, options.conversation_history),),
        )
        if prompt_build.token_count > total_budget:
            truncated, prompt_build, _current_budget = self._shrink_to_budget(
                query=query,
                options=options,
                retrieval=retrieval,
                merged_evidence=merged_evidence,
                total_budget=total_budget,
                current_budget=current_budget,
                truncated=truncated,
                prompt_variants=(
                    ("compact", options.user_prompt, options.conversation_history),
                    ("compact", options.user_prompt, ()),
                    ("compact", None, ()),
                    ("minimal", None, ()),
                ),
            )
        return truncated, prompt_build

    def _shrink_to_budget(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: RetrievalResult,
        merged_evidence: list[EvidenceItem],
        total_budget: int,
        current_budget: int,
        truncated: ContextTruncationResult,
        prompt_variants: Sequence[tuple[Literal["full", "compact", "minimal"], str | None, Sequence[tuple[str, str]]]],
    ) -> tuple[ContextTruncationResult, ContextPromptBuildResult, int]:
        prompt_build = self._build_prompt_variants(
            query=query,
            options=options,
            retrieval=retrieval,
            truncated=truncated,
            prompt_variants=prompt_variants,
        )
        while prompt_build.token_count > total_budget and truncated.evidence and current_budget > 1:
            overflow = prompt_build.token_count - total_budget
            next_budget = max(current_budget - max(overflow, 1), 1)
            retruncated = self._truncate_evidence(merged_evidence, budget=next_budget, options=options)
            if (
                retruncated.token_count >= truncated.token_count
                and len(retruncated.evidence) >= len(truncated.evidence)
            ):
                break
            truncated = retruncated
            current_budget = next_budget
            prompt_build = self._build_prompt_variants(
                query=query,
                options=options,
                retrieval=retrieval,
                truncated=truncated,
                prompt_variants=prompt_variants,
            )
        return truncated, prompt_build, current_budget

    def _truncate_evidence(
        self,
        merged_evidence: list[EvidenceItem],
        *,
        budget: int,
        options: QueryOptions,
    ) -> ContextTruncationResult:
        max_evidence_items = options.resolved_max_evidence_items
        if options.answer_context_top_k is not None:
            max_evidence_items = min(max_evidence_items, max(options.answer_context_top_k, 1))
        return self.truncator.truncate(
            merged_evidence,
            token_budget=budget,
            max_evidence_items=max_evidence_items,
            retrieval_profile=options.resolved_retrieval_profile.value,
        )

    def _build_prompt_variants(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: RetrievalResult,
        truncated: ContextTruncationResult,
        prompt_variants: Sequence[tuple[Literal["full", "compact", "minimal"], str | None, Sequence[tuple[str, str]]]],
    ) -> ContextPromptBuildResult:
        last_prompt: ContextPromptBuildResult | None = None
        for prompt_style, user_prompt, conversation_history in prompt_variants:
            last_prompt = self._build_prompt_from_truncation(
                query=query,
                options=options,
                retrieval=retrieval,
                truncated=truncated,
                prompt_style=prompt_style,
                user_prompt=user_prompt,
                conversation_history=conversation_history,
            )
            if last_prompt.token_count <= options.max_context_tokens:
                return last_prompt
        assert last_prompt is not None
        clipped_prompt = self.prompt_builder.token_accounting.clip(
            last_prompt.prompt,
            options.max_context_tokens,
        )
        return ContextPromptBuildResult(
            grounded_candidate=last_prompt.grounded_candidate,
            prompt=clipped_prompt,
            token_count=self.prompt_builder.token_accounting.count(clipped_prompt),
        )

    def _build_prompt_from_truncation(
        self,
        *,
        query: str,
        options: QueryOptions,
        retrieval: RetrievalResult,
        truncated: ContextTruncationResult,
        prompt_style: Literal["full", "compact", "minimal"],
        user_prompt: str | None,
        conversation_history: Sequence[tuple[str, str]],
    ) -> ContextPromptBuildResult:
        context_evidence_items = [item.as_evidence_item() for item in truncated.evidence]
        grounded_candidate = self.answer_generator.grounded_candidate(
            query,
            context_evidence_items,
            retrieval_signals=retrieval.diagnostics.retrieval_signals,
        )
        return self.prompt_builder.build(
            query=query,
            grounded_candidate=grounded_candidate,
            evidence=truncated.evidence,
            runtime_mode=retrieval.decision.runtime_mode,
            response_type=options.response_type,
            user_prompt=user_prompt,
            conversation_history=conversation_history,
            prompt_style=prompt_style,
        )


__all__ = ["_QueryPipeline"]
