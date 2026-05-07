from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.schema.query import (
    MetadataFilters,
    PolicyHints,
    QueryUnderstanding,
    StructureConstraints,
    TaskType,
)
from rag.schema.runtime import (
    AccessPolicy,
    ExecutionLocation,
    ExecutionLocationPreference,
    ExternalRetrievalPolicy,
    Residency,
    RuntimeMode,
)

_DEEP_TASK_TYPES = {TaskType.COMPARISON, TaskType.SYNTHESIS, TaskType.TIMELINE, TaskType.RESEARCH}


class ChatBindingLike(Protocol):
    backend: object
    location: str
    provider_name: str
    model_name: str | None

    def chat(self, prompt: str) -> str: ...

_QUERY_UNDERSTANDING_PROMPT = """You are the query-understanding module for a document-grounded RAG system.
Return exactly one JSON object and nothing else.
Your job is to extract both explicit user constraints and semantic retrieval intent from the query.
Do not guess hidden facts outside the query.
If the query explicitly says things like do not use web / local only / only my uploaded PDF,
place them under policy_hints.
If the query explicitly mentions page numbers, page ranges, file names, document titles, source types,
quoted phrases, heading or section hints, extract them.
Use enum values exactly:
- task_type: lookup | single_doc_qa | comparison | synthesis | timeline | research
- query_type: lookup | scoped_lookup | structure_lookup | section_lookup | special_lookup
  | comparison | summary | process | research
- structure_constraints.match_strategy: none | semantic | heading
- special_targets values: table | figure | ocr_region | image_summary | caption | formula
Return JSON matching this schema:
{
  "task_type": "lookup",
  "query_type": "lookup",
  "needs_special": false,
  "needs_structure": false,
  "needs_metadata": false,
  "needs_graph_expansion": false,
  "structure_constraints": {
    "match_strategy": "none",
    "requires_structure_match": false,
    "prefer_heading_match": false,
    "focus_terms": []
  },
  "metadata_filters": {
    "page_numbers": [],
    "page_ranges": [],
    "source_types": [],
    "document_titles": [],
    "file_names": []
  },
  "special_targets": [],
  "source_scope_hints": [],
  "quoted_terms": [],
  "policy_hints": {
    "disable_external_retrieval": false,
    "local_only": false,
    "source_type_scope": []
  }
}
If uncertain, keep fields empty or false instead of inventing.
"""
_QUERY_UNDERSTANDING_PROMPT = """You are the query-understanding module for an enterprise document-grounded RAG system.
Return exactly one JSON object and nothing else.

Your purpose is NOT to do generic information extraction.
Your purpose is to produce a retrieval-oriented understanding of the user's query for L3 planning and L4 retrieval.

Prioritize these decisions first:
1. What is the task intent?
2. How complex is the query?
3. What retrieval mode is implied?
4. Does the query need structure-aware retrieval?
5. Does the query target special objects such as tables or figures?
6. Are there explicit constraints that must become hard filters?

Important guidance:
- Most user queries will NOT include page numbers, exact file names, or exact document titles.
- Treat page numbers, page ranges, file names, document titles, and quoted phrases as OPTIONAL low-frequency explicit constraints.
- Do NOT over-focus on metadata extraction when the real signal is task intent, structure intent, or special-object intent.
- If the user asks about a process, architecture, evaluation, conclusion, comparison, or summary, reflect that in task_type, query_type, needs_structure, and structure_constraints.focus_terms.
- If the user clearly refers to tables, figures, captions, OCR text, image summaries, or formulas, mark needs_special=true and populate special_targets.
- Only set metadata filters when they are explicitly stated or clearly implied by the query.
- Do not guess hidden facts outside the query.
- If uncertain, prefer empty fields / false values over invention.

Policy handling:
- If the query explicitly says things like:
  - do not use web
  - local only
  - only my uploaded PDF
  - only this file
  then place them under policy_hints and source_scope_hints.
- Do not infer policy restrictions unless the user clearly stated them.

Task type guidance:
- lookup: fact lookup, direct answer, short retrieval
- single_doc_qa: question about one document or one tightly scoped source
- comparison: compare entities, sections, versions, or documents
- synthesis: summarize / combine evidence into one answer
- timeline: time-ordered evolution / sequence / history
- research: open-ended investigation or exploratory analysis

Query type guidance:
- lookup: plain semantic lookup
- scoped_lookup: lookup with source or scope restriction
- structure_lookup: query likely depends on document structure
- section_lookup: query is likely asking for a specific section/category of content
- special_lookup: query targets tables, figures, OCR, captions, formulas
- comparison: compare two or more things
- summary: summarize a topic/document/section
- process: workflow / procedure / how-it-works
- research: exploratory, open-ended investigation

Structure guidance:
- structure_constraints.match_strategy:
  - none: no meaningful structure signal
  - semantic: query implies a section family or topical area, but not exact heading text
  - heading: query explicitly hints at headings/sections/titles

Use enum values exactly:
- task_type: lookup | single_doc_qa | comparison | synthesis | timeline | research
- query_type: lookup | scoped_lookup | structure_lookup | section_lookup | special_lookup | comparison | summary | process | research
- structure_constraints.match_strategy: none | semantic | heading
- special_targets values: table | figure | ocr_region | image_summary | caption | formula

Return JSON matching exactly this schema:
{
  "task_type": "lookup",
  "query_type": "lookup",
  "needs_special": false,
  "needs_structure": false,
  "needs_metadata": false,
  "needs_graph_expansion": false,
  "structure_constraints": {
    "match_strategy": "none",
    "requires_structure_match": false,
    "prefer_heading_match": false,
    "focus_terms": []
  },
  "metadata_filters": {
    "page_numbers": [],
    "page_ranges": [],
    "source_types": [],
    "document_titles": [],
    "file_names": []
  },
  "special_targets": [],
  "source_scope_hints": [],
  "quoted_terms": [],
  "policy_hints": {
    "disable_external_retrieval": false,
    "local_only": false,
    "source_type_scope": []
  }
}

Additional rules:
- needs_structure should be true when the answer likely depends on a section/category/part of a document rather than generic semantic retrieval.
- needs_metadata should be true only when explicit metadata-style filtering is present or clearly required.
- needs_graph_expansion should usually remain false unless the query explicitly requires relationship/path-style reasoning across entities or linked concepts.
- structure_constraints.focus_terms should capture user-facing semantic targets such as "deployment", "evaluation results", "conclusion", "architecture", "appendix", or explicit heading/category hints.
- source_scope_hints should capture explicit source narrowing such as "this PDF", "uploaded file", "my notes", "this report".
- quoted_terms should contain exact phrases the user likely wants matched literally.

Return exactly one JSON object and nothing else.
"""


class QueryUnderstandingDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    llm_provider: str | None = None
    llm_model: str | None = None
    llm_latency_ms: float | None = None
    llm_raw_response: str | None = None
    llm_parsed_result: QueryUnderstanding | None = None
    final_understanding: QueryUnderstanding
    fallback_used: bool = False
    fallback_reason: str | None = None


class RoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_type: TaskType
    runtime_mode: RuntimeMode
    source_scope: list[str] = Field(default_factory=list)
    web_search_allowed: bool = False
    graph_expansion_allowed: bool = False
    rerank_required: bool = True


class QueryUnderstandingService:
    def __init__(
        self,
        *,
        chat_bindings: Sequence[ChatBindingLike] = (),
        enable_llm: bool = True,
    ) -> None:
        self._chat_bindings = tuple(chat_bindings)
        self._enable_llm = enable_llm
        self.last_diagnostics: QueryUnderstandingDiagnostics | None = None

    def analyze(
        self,
        query: str,
        *,
        access_policy: AccessPolicy | None = None,
        execution_location_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_FIRST,
    ) -> QueryUnderstanding:
        understanding, diagnostics = self.analyze_with_diagnostics(
            query,
            access_policy=access_policy,
            execution_location_preference=execution_location_preference,
        )
        self.last_diagnostics = diagnostics
        return understanding

    def analyze_with_diagnostics(
        self,
        query: str,
        *,
        access_policy: AccessPolicy | None = None,
        execution_location_preference: ExecutionLocationPreference = ExecutionLocationPreference.LOCAL_FIRST,
    ) -> tuple[QueryUnderstanding, QueryUnderstandingDiagnostics]:
        llm_result, llm_provider, llm_model, llm_latency_ms, raw_response, fallback_reason = self._understand_with_llm(
            query,
            access_policy=access_policy,
            execution_location_preference=execution_location_preference,
        )
        final_understanding = llm_result or self._fallback_understanding()
        diagnostics = QueryUnderstandingDiagnostics(
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_latency_ms=llm_latency_ms,
            llm_raw_response=raw_response,
            llm_parsed_result=llm_result,
            final_understanding=final_understanding,
            fallback_used=llm_result is None,
            fallback_reason=fallback_reason,
        )
        self.last_diagnostics = diagnostics
        return final_understanding, diagnostics

    def diagnostics_payload(self) -> dict[str, object]:
        if self.last_diagnostics is None:
            return {}
        return self.last_diagnostics.model_dump(mode="json")

    def build_eval_record(self, query: str) -> dict[str, object]:
        return {
            "query": query,
            "diagnostics": self.diagnostics_payload(),
        }

    def _understand_with_llm(
        self,
        query: str,
        *,
        access_policy: AccessPolicy | None,
        execution_location_preference: ExecutionLocationPreference,
    ) -> tuple[QueryUnderstanding | None, str | None, str | None, float | None, str | None, str | None]:
        if not self._enable_llm:
            return None, None, None, None, None, "llm_disabled"
        bindings = self._ordered_chat_bindings(access_policy, execution_location_preference)
        if not bindings:
            return None, None, None, None, None, "no_chat_binding"
        prompt = self._build_llm_prompt(query=query)
        fallback_reason = "llm_unavailable"
        for binding in bindings:
            started = time.perf_counter()
            try:
                raw_response = binding.chat(prompt)
            except Exception as exc:
                fallback_reason = f"chat_failed:{binding.provider_name}:{exc}"
                continue
            latency_ms = (time.perf_counter() - started) * 1000.0
            parsed = self._parse_llm_response(raw_response)
            if parsed is None:
                fallback_reason = f"invalid_json:{binding.provider_name}"
                continue
            return parsed, binding.provider_name, binding.model_name, latency_ms, raw_response, None
        return None, None, None, None, None, fallback_reason

    def _ordered_chat_bindings(
        self,
        access_policy: AccessPolicy | None,
        execution_location_preference: ExecutionLocationPreference,
    ) -> list[ChatBindingLike]:
        if not self._chat_bindings:
            return []
        if access_policy is not None and access_policy.local_only:
            preferred_locations = ("local",)
        elif execution_location_preference is ExecutionLocationPreference.LOCAL_ONLY:
            preferred_locations = ("local",)
        elif execution_location_preference is ExecutionLocationPreference.LOCAL_FIRST:
            preferred_locations = ("local", "cloud")
        else:
            preferred_locations = ("cloud", "local")
        ordered: list[ChatBindingLike] = []
        remaining = list(self._chat_bindings)
        for location in preferred_locations:
            matched = [binding for binding in remaining if binding.location == location]
            ordered.extend(self._dedupe_bindings(matched))
            remaining = [binding for binding in remaining if binding.location != location]
        ordered.extend(self._dedupe_bindings(remaining))
        return ordered

    @staticmethod
    def _dedupe_bindings(bindings: Sequence[ChatBindingLike]) -> list[ChatBindingLike]:
        seen: set[int] = set()
        ordered: list[ChatBindingLike] = []
        for binding in bindings:
            identity = id(binding.backend)
            if identity in seen:
                continue
            seen.add(identity)
            ordered.append(binding)
        return ordered

    @staticmethod
    def _build_llm_prompt(*, query: str) -> str:
        return f"{_QUERY_UNDERSTANDING_PROMPT}\nQuery: {query}\nJSON only."

    @classmethod
    def _parse_llm_response(cls, response: str) -> QueryUnderstanding | None:
        candidate = cls._extract_json_object(response)
        if candidate is None:
            return None
        try:
            payload = json.loads(candidate)
            understanding = QueryUnderstanding.model_validate(payload)
        except Exception:
            return None
        return cls._normalize_understanding(understanding)

    @staticmethod
    def _extract_json_object(response: str) -> str | None:
        stripped = response.strip()
        if not stripped:
            return None
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return stripped[start : end + 1]

    @staticmethod
    def _normalize_understanding(understanding: QueryUnderstanding) -> QueryUnderstanding:
        structure = understanding.structure_constraints
        metadata = understanding.metadata_filters
        policy = understanding.policy_hints
        normalized_structure = StructureConstraints(
            match_strategy=structure.match_strategy,
            requires_structure_match=structure.requires_structure_match,
            prefer_heading_match=structure.prefer_heading_match,
            focus_terms=_ordered_unique(structure.focus_terms),
        )
        normalized_metadata = MetadataFilters(
            page_numbers=_ordered_unique_ints(metadata.page_numbers),
            page_ranges=list(dict.fromkeys(metadata.page_ranges)),
            source_types=_ordered_unique(metadata.source_types),
            document_titles=_ordered_unique(metadata.document_titles),
            file_names=_ordered_unique(metadata.file_names),
        )
        normalized_policy = PolicyHints(
            disable_external_retrieval=policy.disable_external_retrieval,
            local_only=policy.local_only,
            source_type_scope=_ordered_unique(policy.source_type_scope),
        )
        special_targets = _ordered_unique(understanding.special_targets)
        needs_special = understanding.needs_special or bool(special_targets)
        needs_structure = understanding.needs_structure or normalized_structure.has_constraints()
        needs_metadata = understanding.needs_metadata or normalized_metadata.has_constraints()
        return understanding.model_copy(
            update={
                "needs_special": needs_special,
                "needs_structure": needs_structure,
                "needs_metadata": needs_metadata,
                "structure_constraints": normalized_structure,
                "metadata_filters": normalized_metadata,
                "special_targets": special_targets,
                "source_scope_hints": _ordered_unique(understanding.source_scope_hints),
                "quoted_terms": _ordered_unique(understanding.quoted_terms),
                "policy_hints": normalized_policy,
            }
        )

    @staticmethod
    def _fallback_understanding() -> QueryUnderstanding:
        return QueryUnderstanding(
            task_type=TaskType.LOOKUP,
            query_type="lookup",
        )


class RoutingService:
    def route(
        self,
        query: str,
        *,
        query_understanding: QueryUnderstanding,
        source_scope: Sequence[str] = (),
        access_policy: AccessPolicy | None = None,
    ) -> RoutingDecision:
        del query
        task_type = query_understanding.task_type
        runtime_mode = self._runtime_mode(task_type, source_scope, query_understanding)
        allow_external = (
            True
            if access_policy is None
            else access_policy.external_retrieval is not ExternalRetrievalPolicy.DENY
        )
        return RoutingDecision(
            task_type=task_type,
            runtime_mode=runtime_mode,
            source_scope=list(source_scope),
            web_search_allowed=allow_external and not source_scope and task_type in _DEEP_TASK_TYPES,
            graph_expansion_allowed=(runtime_mode is RuntimeMode.DEEP and query_understanding.needs_graph_expansion),
            rerank_required=True,
        )

    @staticmethod
    def _runtime_mode(
        task_type: TaskType,
        source_scope: Sequence[str],
        query_understanding: QueryUnderstanding,
    ) -> RuntimeMode:
        if query_understanding.needs_graph_expansion:
            return RuntimeMode.DEEP
        if task_type in {TaskType.COMPARISON, TaskType.TIMELINE, TaskType.RESEARCH}:
            return RuntimeMode.DEEP
        if task_type is TaskType.SYNTHESIS and len(source_scope) != 1:
            return RuntimeMode.DEEP
        return RuntimeMode.FAST


def query_policy_hint_to_access_policy(query_understanding: QueryUnderstanding) -> AccessPolicy | None:
    hints = query_understanding.policy_hints
    if not hints.has_hints():
        return None
    return AccessPolicy(
        residency=Residency.LOCAL_REQUIRED if hints.local_only else Residency.CLOUD_ALLOWED,
        external_retrieval=(
            ExternalRetrievalPolicy.DENY
            if hints.disable_external_retrieval or hints.local_only
            else ExternalRetrievalPolicy.ALLOW
        ),
        allowed_runtimes=frozenset({RuntimeMode.FAST, RuntimeMode.DEEP}),
        allowed_locations=(
            frozenset({ExecutionLocation.LOCAL})
            if hints.local_only
            else frozenset({ExecutionLocation.LOCAL, ExecutionLocation.CLOUD})
        ),
    )


def narrow_access_policy_for_query(
    default_policy: AccessPolicy,
    query_understanding: QueryUnderstanding,
) -> AccessPolicy:
    hinted_policy = query_policy_hint_to_access_policy(query_understanding)
    if hinted_policy is None:
        return default_policy
    return default_policy.narrow(hinted_policy)


def section_family_aliases(family: str) -> tuple[str, ...]:
    normalized = family.strip().lower()
    return (normalized,) if normalized else ()


def special_target_aliases(target: str) -> tuple[str, ...]:
    normalized = target.strip().lower()
    return (normalized,) if normalized else ()


def _ordered_unique(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _ordered_unique_ints(values: Iterable[int]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        normalized = int(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


__all__ = [
    "QueryUnderstandingDiagnostics",
    "QueryUnderstandingService",
    "RoutingDecision",
    "RoutingService",
    "narrow_access_policy_for_query",
    "query_policy_hint_to_access_policy",
    "section_family_aliases",
    "special_target_aliases",
]
