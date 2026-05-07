from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag.retrieval.authorization_service import AuthorizationService
from rag.schema.query import EvidenceItem
from rag.schema.runtime import AccessPolicy


def _resolve_document(get_document: object, doc_id: object) -> Any | None:
    if not callable(get_document):
        return None
    document = get_document(doc_id)
    if document is not None:
        return document
    normalized = str(doc_id).strip()
    if normalized.isdigit():
        return get_document(int(normalized))
    return None


@dataclass(slots=True)
class SynthesisService:
    metadata_repo: object | None = None
    authorization_service: AuthorizationService | object | None = None
    deny_when_document_missing: bool = True

    def filter_evidence(
        self,
        *,
        evidence: list[EvidenceItem],
        access_policy: AccessPolicy,
        user_id: str | None = None,
    ) -> list[EvidenceItem]:
        if not evidence:
            return []
        get_document = getattr(self.metadata_repo, "get_document", None)
        authorization_service = getattr(self, "authorization_service", None)
        allowed_doc_ids: set[str] | None = None
        if user_id and authorization_service is None:
            return []
        if user_id and authorization_service is not None:
            resolve_query = getattr(authorization_service, "resolve_query", None)
            if not callable(resolve_query):
                return []
            auth_context = resolve_query(
                user_id=user_id,
                access_policy=access_policy,
                source_scope=(),
            )
            if not getattr(auth_context, "resolved_user_view", False):
                return []
            if getattr(auth_context, "allowed_doc_ids", ()):
                allowed_doc_ids = {str(doc_id) for doc_id in auth_context.allowed_doc_ids}
        if not callable(get_document):
            return [] if allowed_doc_ids is None else [item for item in evidence if str(item.doc_id) in allowed_doc_ids]
        filtered: list[EvidenceItem] = []
        for item in evidence:
            if allowed_doc_ids is not None and str(item.doc_id) not in allowed_doc_ids:
                continue
            document = _resolve_document(get_document, item.doc_id)
            if document is None:
                continue
            if not self._document_visible(document):
                continue
            if not self._document_allowed(document, access_policy):
                continue
            filtered.append(item)
        return filtered

    @staticmethod
    def _document_visible(document: object) -> bool:
        if getattr(document, "is_active", True) is False:
            return False
        if getattr(document, "index_ready", True) is False:
            return False
        status = str(getattr(document, "doc_status", "") or "").strip().lower()
        return status not in {"retired", "expired", "deleted", "inactive"}

    @staticmethod
    def _document_allowed(document: object, access_policy: AccessPolicy) -> bool:
        effective_access_policy = getattr(document, "effective_access_policy", None)
        if not isinstance(effective_access_policy, AccessPolicy):
            return True
        try:
            access_policy.narrow(effective_access_policy)
        except ValueError:
            return False
        return True


__all__ = ["SynthesisService"]
