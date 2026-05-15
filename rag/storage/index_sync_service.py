from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

from rag.schema.core import ProcessingStateRecord

_PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}


class StaleProcessingStateError(RuntimeError):
    """Raised when a worker is holding an out-of-date processing lease."""


@dataclass(frozen=True, slots=True)
class IndexSyncMonitorSnapshot:
    pending: int = 0
    processing: int = 0
    failed: int = 0
    completed: int = 0
    retry_backlog: int = 0
    max_attempts: int = 0
    oldest_pending_age_seconds: float = 0.0
    alert_level: str = "ok"


@dataclass(slots=True)
class IndexSyncService:
    metadata_repo: object
    stage: str = "index_sync"

    def enqueue(
        self,
        *,
        doc_id: int,
        source_id: int,
        operation: str,
        priority: str = "normal",
        metadata_json: dict[str, object] | None = None,
        error_message: str | None = None,
    ) -> ProcessingStateRecord | None:
        save_processing_state = getattr(self.metadata_repo, "save_processing_state", None)
        if not callable(save_processing_state):
            return None
        existing = self.get(doc_id)
        attempts = 0 if existing is None else existing.attempts
        merged_metadata = dict(existing.metadata_json) if existing is not None else {}
        merged_metadata.update(metadata_json or {})
        merged_metadata["operation"] = operation
        state = ProcessingStateRecord(
            doc_id=doc_id,
            source_id=source_id if existing is None else existing.source_id,
            stage=self.stage,
            status="pending",
            attempts=attempts,
            priority=priority if existing is None else existing.priority,
            worker_id=None,
            lease_expires_at=None,
            error_message=error_message,
            metadata_json=merged_metadata,
        )
        return cast(ProcessingStateRecord, save_processing_state(state))

    def get(self, doc_id: int) -> ProcessingStateRecord | None:
        getter = getattr(self.metadata_repo, "get_processing_state", None)
        if not callable(getter):
            return None
        return cast(ProcessingStateRecord | None, getter(doc_id))

    def claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: datetime | None = None,
    ) -> ProcessingStateRecord | None:
        save_processing_state = getattr(self.metadata_repo, "save_processing_state", None)
        list_processing_states = getattr(self.metadata_repo, "list_processing_states", None)
        if not callable(save_processing_state) or not callable(list_processing_states):
            return None
        current_time = now or datetime.now(UTC)
        candidates = [
            state
            for state in list_processing_states(stage=self.stage)
            if (
                state.status == "pending" and self._retry_available(state, current_time)
            )
            or self._lease_expired(state, current_time)
        ]
        if not candidates:
            return None
        selected = min(
            candidates,
            key=lambda state: (
                _PRIORITY_ORDER.get(state.priority, 99),
                state.updated_at,
                state.doc_id,
            ),
        )
        claimed = selected.model_copy(
            update={
                "status": "processing",
                "attempts": selected.attempts + 1,
                "worker_id": worker_id,
                "lease_expires_at": current_time + timedelta(seconds=max(lease_seconds, 1)),
                "error_message": None,
                "updated_at": current_time,
            }
        )
        return cast(ProcessingStateRecord, save_processing_state(claimed))

    def mark_completed(self, doc_id: int, *, now: datetime | None = None) -> ProcessingStateRecord | None:
        save_processing_state = getattr(self.metadata_repo, "save_processing_state", None)
        if not callable(save_processing_state):
            return None
        existing = self.get(doc_id)
        if existing is None:
            return None
        completed = existing.model_copy(
            update={
                "status": "completed",
                "worker_id": None,
                "lease_expires_at": None,
                "error_message": None,
                "updated_at": now or datetime.now(UTC),
            }
        )
        return cast(ProcessingStateRecord, save_processing_state(completed))

    def mark_failed(
        self,
        doc_id: int,
        *,
        error_message: str,
        retryable: bool = True,
        now: datetime | None = None,
    ) -> ProcessingStateRecord | None:
        save_processing_state = getattr(self.metadata_repo, "save_processing_state", None)
        if not callable(save_processing_state):
            return None
        existing = self.get(doc_id)
        if existing is None:
            return None
        failed = existing.model_copy(
            update={
                "status": "pending" if retryable else "failed",
                "worker_id": None,
                "lease_expires_at": None,
                "error_message": error_message,
                "updated_at": now or datetime.now(UTC),
            }
        )
        return cast(ProcessingStateRecord, save_processing_state(failed))

    def process_next(
        self,
        *,
        worker_id: str,
        sync_handler: Callable[[ProcessingStateRecord], object],
        lease_seconds: int = 60,
        now: datetime | None = None,
    ) -> ProcessingStateRecord | None:
        claimed = self.claim_next(worker_id=worker_id, lease_seconds=lease_seconds, now=now)
        if claimed is None:
            return None
        try:
            sync_handler(claimed)
        except StaleProcessingStateError:
            return self.get(claimed.doc_id)
        except Exception as exc:
            self.mark_failed(claimed.doc_id, error_message=str(exc), retryable=True, now=now)
            raise
        return self.mark_completed(claimed.doc_id, now=now)

    def monitor_snapshot(self, *, now: datetime | None = None) -> IndexSyncMonitorSnapshot:
        list_processing_states = getattr(self.metadata_repo, "list_processing_states", None)
        if not callable(list_processing_states):
            return IndexSyncMonitorSnapshot()
        current_time = now or datetime.now(UTC)
        states = list(list_processing_states(stage=self.stage))
        pending = [state for state in states if state.status == "pending"]
        processing = [state for state in states if state.status == "processing"]
        failed = [state for state in states if state.status == "failed"]
        completed = [state for state in states if state.status == "completed"]
        retry_backlog = [state for state in pending if bool(state.error_message)]
        oldest_pending_age = 0.0
        if pending:
            oldest_pending_age = max(
                0.0,
                max((current_time - state.updated_at).total_seconds() for state in pending),
            )
        alert_level = "ok"
        if failed:
            alert_level = "critical"
        elif oldest_pending_age >= 300 or len(retry_backlog) >= 3:
            alert_level = "warning"
        return IndexSyncMonitorSnapshot(
            pending=len(pending),
            processing=len(processing),
            failed=len(failed),
            completed=len(completed),
            retry_backlog=len(retry_backlog),
            max_attempts=max((state.attempts for state in states), default=0),
            oldest_pending_age_seconds=round(oldest_pending_age, 3),
            alert_level=alert_level,
        )

    @staticmethod
    def _lease_expired(state: ProcessingStateRecord, now: datetime) -> bool:
        return state.status == "processing" and state.lease_expires_at is not None and state.lease_expires_at <= now

    @staticmethod
    def _retry_available(state: ProcessingStateRecord, now: datetime) -> bool:
        if not state.error_message:
            return True
        delay_seconds = min(5 * max(state.attempts, 1), 300)
        return state.updated_at + timedelta(seconds=delay_seconds) <= now


__all__ = ["IndexSyncMonitorSnapshot", "IndexSyncService", "StaleProcessingStateError"]
