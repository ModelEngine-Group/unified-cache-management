# MIT License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any

from ucm.logger import init_logger
from ucm.store.pipeline.errors import StoreNotFoundError, StoreUnhealthyError
from ucm.store.ucmstore_v1 import UcmKVStoreBaseV1

logger = init_logger(__name__)


class RankConsistencyTracker:
    """Scheduler-side set of rank0 block ids whose data is missing on some rank.

    Marked blocks are excluded from external lookups: valid_prefix truncates
    prefix lookups, while membership checks let exact lookups mask only marked
    keys. A reported rank0 dump success clears the mark (clear_dumped).

    The tracker is a performance guard, not a correctness mechanism: a spurious
    clear costs at most one extra "hit -> load NotFound -> re-mark" round. The
    only ordering contract is that callers apply clear_dumped() before
    mark_missing() for metadata from the same scheduler step, so a fresh mark
    survives a stale dump success reported together with it.
    """

    _MAX_INCONSISTENT_BLOCKS = 2_000_000

    def __init__(self) -> None:
        self._inconsistent: set[bytes] = set()

    def mark_missing(self, rank0_block_ids: set[bytes]) -> int:
        added = 0
        evicted = 0
        for block_id in rank0_block_ids:
            if block_id in self._inconsistent:
                continue
            if len(self._inconsistent) >= self._MAX_INCONSISTENT_BLOCKS:
                self._inconsistent.pop()
                evicted += 1
            self._inconsistent.add(block_id)
            added += 1
        if evicted:
            logger.warning_limit(
                "Rank consistency tracker reached its %d-block limit; "
                "evicted %d inconsistent block records",
                self._MAX_INCONSISTENT_BLOCKS,
                evicted,
            )
        return added

    def clear_dumped(self, rank0_block_ids: set[bytes]) -> int:
        before = len(self._inconsistent)
        self._inconsistent -= rank0_block_ids
        return before - len(self._inconsistent)

    def __len__(self) -> int:
        return len(self._inconsistent)

    def __contains__(self, block_id: bytes) -> bool:
        return block_id in self._inconsistent

    def valid_prefix(self, block_ids: list[bytes]) -> list[bytes]:
        for index, block_id in enumerate(block_ids):
            if block_id in self._inconsistent:
                return block_ids[:index]
        return block_ids


class RankConsistencyManager:
    """Adds cross-rank consistency handling around Store operations."""

    def __init__(
        self,
        *,
        is_scheduler: bool,
        use_consistency_manager: bool = True,
    ) -> None:
        """Configure scheduler filtering and worker-side dump accounting."""
        self.enabled = use_consistency_manager
        self._tracker = (
            RankConsistencyTracker() if is_scheduler and self.enabled else None
        )
        self._missing_reqs: set[str] = set()
        self._missing_blocks: set[bytes] = set()
        self._dump_succeeded_blocks: set[bytes] = set()
        self._load_task_contexts: dict[
            int, tuple[UcmKVStoreBaseV1, dict[str, list[bytes]]]
        ] = {}
        self._dump_task_contexts: dict[
            int, tuple[UcmKVStoreBaseV1, dict[str, set[bytes]]]
        ] = {}
        self._dump_blocks_by_request: dict[str, set[bytes]] = {}
        self._dump_failed_blocks_by_request: dict[str, set[bytes]] = {}

    def lookup_on_prefix(self, store: UcmKVStoreBaseV1, block_ids: list[bytes]) -> int:
        """Exclude known-missing blocks before forwarding a prefix lookup."""
        if not self.enabled:
            return store.lookup_on_prefix(block_ids)
        if self._tracker is not None:
            block_ids = self._tracker.valid_prefix(block_ids)
        if not block_ids:
            return -1
        return store.lookup_on_prefix(block_ids)

    def lookup_on_reverse(self, store: UcmKVStoreBaseV1, block_ids: list[bytes]) -> int:
        """Exclude known-missing blocks before forwarding a reverse lookup.

        A known-missing block is treated as a miss so the reverse scan
        continues to the left.  At most one re-query per known-missing hit
        is issued; in practice the inconsistent set is small.
        """
        if not self.enabled:
            return store.lookup_on_reverse(block_ids)
        if self._tracker is None or not block_ids:
            return store.lookup_on_reverse(block_ids)
        result = store.lookup_on_reverse(block_ids)
        while result >= 0 and block_ids[result] in self._tracker:
            if result == 0:
                return -1
            result = store.lookup_on_reverse(block_ids[:result])
        return result

    def lookup_all(self, store: UcmKVStoreBaseV1, block_ids: list[bytes]) -> list[bool]:
        """Mask known-missing blocks after forwarding an exact lookup."""
        results = store.lookup(block_ids)
        if self._tracker is None:
            return results
        return [
            False if block_id in self._tracker else result
            for block_id, result in zip(block_ids, results)
        ]

    def submit_load(
        self,
        store: UcmKVStoreBaseV1,
        block_ids_by_request: dict[str, list[bytes]],
        block_ids: list[bytes],
        shard_indices: list[int],
        ptrs: Any,
    ) -> Any:
        """Retain request block IDs and classify immediate NotFound errors."""
        request_context = (
            {
                request_id: list(request_block_ids)
                for request_id, request_block_ids in block_ids_by_request.items()
            }
            if self.enabled
            else {}
        )
        try:
            task = store.load_data(block_ids, shard_indices, ptrs)
        except Exception as error:
            if self.enabled and isinstance(error, StoreNotFoundError):
                self._mark_load_context_missing(request_context)
            raise
        self._load_task_contexts[id(task)] = (store, request_context)
        return task

    def wait_load(self, task: Any) -> None:
        """Wait through the task's Store and record blocks missing at wait time."""
        task_key = id(task)
        if task_key not in self._load_task_contexts:
            raise RuntimeError("Load task was not submitted through submit_load().")
        store, request_context = self._load_task_contexts.pop(task_key)
        try:
            store.wait(task)
        except Exception as error:
            if self.enabled and isinstance(error, StoreNotFoundError):
                self._mark_load_context_missing(request_context)
            raise

    def check_load(self, task: Any) -> bool:
        """Poll a load task without blocking and retain its wait context.

        A successful poll does not remove the context. The caller must invoke
        wait_load() once after completion so the Store can surface deferred
        errors and release task resources.
        """
        task_key = id(task)
        if task_key not in self._load_task_contexts:
            raise RuntimeError("Load task was not submitted through submit_load().")
        store, request_context = self._load_task_contexts[task_key]
        try:
            return store.check(task)
        except Exception as error:
            self._load_task_contexts.pop(task_key, None)
            if self.enabled and isinstance(error, StoreNotFoundError):
                self._mark_load_context_missing(request_context)
            raise

    def submit_dump(
        self,
        store: UcmKVStoreBaseV1,
        block_ids_by_request: dict[str, set[bytes]],
        block_ids: list[bytes],
        shard_indices: list[int],
        ptrs: Any,
        event_handle: int,
    ) -> Any:
        """Track submitted blocks and the Store used by each dump task."""
        request_context = (
            {
                request_id: set(request_block_ids)
                for request_id, request_block_ids in block_ids_by_request.items()
            }
            if self.enabled
            else {}
        )
        for request_id, request_block_ids in request_context.items():
            self._dump_blocks_by_request.setdefault(request_id, set()).update(
                request_block_ids
            )
        try:
            task = store.dump_data(block_ids, shard_indices, ptrs, event_handle)
        except Exception:
            self._record_dump_failure(request_context)
            raise
        self._dump_task_contexts[id(task)] = (store, request_context)
        return task

    def wait_dump(self, task: Any) -> None:
        """Wait through the task's Store and record blocks affected by failure."""
        task_key = id(task)
        if task_key not in self._dump_task_contexts:
            raise RuntimeError("Dump task was not submitted through submit_dump().")
        store, request_context = self._dump_task_contexts.pop(task_key)
        try:
            store.wait(task)
        except StoreUnhealthyError:
            self._record_dump_failure(request_context)
            return
        except Exception:
            self._record_dump_failure(request_context)
            raise

    def finish_dump(self, request_ids: set[str]) -> None:
        """Queue successful blocks for reporting to scheduler."""
        if not self.enabled:
            return
        pending_request_ids = {
            request_id
            for _, request_context in self._dump_task_contexts.values()
            for request_id in request_context
        }
        unfinished = request_ids & pending_request_ids
        if unfinished:
            raise RuntimeError(
                f"Dump tasks are still pending for requests: {sorted(unfinished)}"
            )
        for request_id in request_ids:
            block_ids = self._dump_blocks_by_request.pop(request_id, set())
            failed_block_ids = self._dump_failed_blocks_by_request.pop(
                request_id, set()
            )
            self._dump_succeeded_blocks.update(block_ids - failed_block_ids)

    def update_worker_meta(self, worker_meta: Any) -> None:
        """Move worker events into metadata that will be sent to scheduler."""
        if not self.enabled:
            return
        worker_meta.missing_reqs.update(self._missing_reqs)
        worker_meta.missing_blocks.update(self._missing_blocks)
        worker_meta.dump_succeeded_blocks.update(self._dump_succeeded_blocks)
        self._missing_reqs.clear()
        self._missing_blocks.clear()
        self._dump_succeeded_blocks.clear()

    def apply_worker_meta(self, worker_meta: Any) -> None:
        """Apply reported dump successes and load misses on the scheduler."""
        if self._tracker is None:
            return
        total_before = len(self._tracker)
        cleared = self._tracker.clear_dumped(worker_meta.dump_succeeded_blocks)
        added = self._tracker.mark_missing(worker_meta.missing_blocks)
        total_after = len(self._tracker)
        if added or cleared:
            logger.warning_limit(
                "Rank consistency invalid block set changed after worker metadata "
                "aggregation: added=%d, cleared=%d, total_before=%d, total_after=%d",
                added,
                cleared,
                total_before,
                total_after,
            )

    def _mark_load_context_missing(
        self, block_ids_by_request: dict[str, list[bytes]]
    ) -> None:
        """Queue NotFound rank-0 block IDs for reporting to scheduler."""
        for request_id, block_ids in block_ids_by_request.items():
            self._missing_reqs.add(request_id)
            self._missing_blocks.update(block_ids)

    def _record_dump_failure(self, block_ids_by_request: dict[str, set[bytes]]) -> None:
        """Accumulate blocks covered by a failed dump task."""
        for request_id, request_block_ids in block_ids_by_request.items():
            self._dump_failed_blocks_by_request.setdefault(request_id, set()).update(
                request_block_ids
            )
