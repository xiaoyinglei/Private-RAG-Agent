from __future__ import annotations

from dataclasses import dataclass

from rag.schema.core import ProcessingStateRecord
from rag.storage.data_contract_service import DataContractService
from rag.storage.index_sync_service import IndexSyncService


@dataclass(slots=True)
class IndexSyncWorker:
    index_sync_service: IndexSyncService
    data_contract_service: DataContractService
    worker_id: str = "index-sync-worker"

    def run_once(self, *, lease_seconds: int = 60) -> ProcessingStateRecord | None:
        return self.index_sync_service.process_next(
            worker_id=self.worker_id,
            lease_seconds=lease_seconds,
            sync_handler=self._sync_record,
        )

    def run_until_idle(self, *, max_tasks: int = 8, lease_seconds: int = 60) -> list[ProcessingStateRecord]:
        processed: list[ProcessingStateRecord] = []
        for _ in range(max(max_tasks, 0)):
            state = self.run_once(lease_seconds=lease_seconds)
            if state is None:
                break
            processed.append(state)
            if state.status == "pending":
                break
        return processed

    def _sync_record(self, state: ProcessingStateRecord) -> None:
        self.data_contract_service.sync_processing_state(state)


__all__ = ["IndexSyncWorker"]
