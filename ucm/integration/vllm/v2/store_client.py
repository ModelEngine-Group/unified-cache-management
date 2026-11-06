"""Python binding ownership, wire tokens and native transfer completion.

Routing, admission, staging and publication belong to ucm/v2. This module
only packs the binding's types and retains transfer descriptions until IO
has terminated. It does not implement a Python proxy or a storage backend.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
from typing import Any

import numpy as np


class StoreIOError(RuntimeError):
    """Recoverable storage failure; a failed wait has already terminated IO."""


@dataclass(frozen=True)
class AddressedToken:
    """Serializable copy of one successful native Prepare result.

    The opaque payload is copied verbatim. In particular, the connector must
    not slice a token's keys independently of its backend-owned payload.
    """

    keys: tuple[bytes, ...]
    payload: bytes

    @classmethod
    def from_native(cls, token: Any) -> AddressedToken:
        return cls(tuple(key.to_bytes() for key in token.keys), bytes(token.payload))

    def to_native(self, proxy: Any) -> Any:
        token = proxy.AddressedToken()
        token.keys = [proxy.BlockId(key) for key in self.keys]
        token.payload = self.payload
        return token


# Scheduler jobs use odd IDs; each Worker allocates even IDs for layerwise IO.
# Bulk IO uses the Scheduler job ID directly. Completion queues are Worker-local.
JOB_ID_STRIDE = 2
MAX_NATIVE_JOB_ID = (1 << 64) - 1


@dataclass(frozen=True)
class PreparedJob:
    """Scheduler-assigned job identity and the result of its Prepare."""

    job_id: int
    ucm_keys: tuple[bytes, ...]
    token: AddressedToken


@dataclass(frozen=True)
class TransferBatch:
    """Device ranges and Store byte offsets; matrices are [key, segment]."""

    ucm_keys: tuple[bytes, ...]
    device_ptrs: np.ndarray
    byte_sizes: np.ndarray
    store_offsets: np.ndarray


@dataclass
class TransferTask:
    job_id: int
    sub_job_id: int | None
    transfer_batches: tuple[TransferBatch, ...]
    native_arguments: tuple[Any, ...]
    # The Worker owns this event until the native IO is terminal.
    prerequisite_handle: int = 0
    success: bool | None = None

    @property
    def io_job_id(self) -> int:
        return self.job_id if self.sub_job_id is None else self.sub_job_id


class _StoreBinding:
    """Own the native assembler and translate recoverable call failures."""

    def __init__(self, launch_config: dict[str, Any]) -> None:
        self._proxy = importlib.import_module(
            launch_config.get("store_module", "ucm_proxy")
        )
        self._assembler = self._proxy.StoreAssembler(launch_config["store_config"])

    @contextmanager
    def _translate_errors(self) -> Iterator[None]:
        try:
            yield
        except (OSError, MemoryError) as exc:
            raise StoreIOError(str(exc)) from exc
        except self._proxy.UcmError as exc:
            # The current binding exposes Status as a text-only exception.
            # Contract/setup failures remain fatal; only explicit IO/resource
            # failures can be handled as misses or best-effort failed saves.
            status_name = str(exc).split(" ", 1)[0]
            if status_name in {
                "OsApiError",
                "OutOfMemory",
                "NoSpace",
                "NotFound",
                "Retry",
                "Timeout",
                "Unhealthy",
            }:
                raise StoreIOError(str(exc)) from exc
            raise


class StoreSchedulerClient(_StoreBinding):
    """Request metadata and Prepare/Complete calls using the native interface."""

    def __init__(self, launch_config: dict[str, Any]) -> None:
        super().__init__(launch_config)
        self._scheduler = self._assembler.get_scheduler()

    def on_new_request(self, request_id: str) -> None:
        request_context = self._proxy.RequestContext()
        request_context.request_id = request_id
        with self._translate_errors():
            self._scheduler.on_new_request(request_context)

    def on_request_finished(self, request_id: str) -> None:
        with self._translate_errors():
            self._scheduler.on_request_finished(request_id)

    def lookup_on_prefix(
        self, request_id: str, ucm_keys: tuple[bytes, ...]
    ) -> int | None:
        """Return the last prefix-hit index, -1 for a miss, or None for prefetch."""
        with self._translate_errors():
            return self._scheduler.lookup_on_prefix(request_id, b"".join(ucm_keys))

    def lookup_on_reverse(
        self, request_id: str, ucm_keys: tuple[bytes, ...]
    ) -> int | None:
        """Return the rightmost hit index, -1 for a miss, or None for prefetch."""
        with self._translate_errors():
            return self._scheduler.lookup_on_reverse(request_id, b"".join(ucm_keys))

    def prepare_get(
        self, request_id: str, ucm_keys: tuple[bytes, ...]
    ) -> AddressedToken:
        with self._translate_errors():
            return AddressedToken.from_native(
                self._scheduler.prepare_get(request_id, b"".join(ucm_keys))
            )

    def complete_get(self, request_id: str) -> None:
        with self._translate_errors():
            self._scheduler.complete_get(request_id)

    def prepare_put(
        self, request_id: str, ucm_keys: tuple[bytes, ...]
    ) -> AddressedToken:
        with self._translate_errors():
            return AddressedToken.from_native(
                self._scheduler.prepare_put(request_id, b"".join(ucm_keys))
            )

    def complete_put(self, request_id: str, success: bool) -> None:
        with self._translate_errors():
            self._scheduler.complete_put(request_id, success)

    def prepare_prefetch(self, request_id: str) -> AddressedToken:
        with self._translate_errors():
            return AddressedToken.from_native(
                self._scheduler.prepare_prefetch(request_id)
            )

    def complete_prefetch(self, request_id: str, success: bool) -> None:
        with self._translate_errors():
            self._scheduler.complete_prefetch(request_id, success)


class StoreWorkerClient(_StoreBinding):
    """Submit and drain IO; no scheduler interface or request metadata state."""

    def __init__(self, launch_config: dict[str, Any]) -> None:
        super().__init__(launch_config)
        self._worker = self._assembler.get_worker()
        self._tasks: dict[int, TransferTask] = {}
        self._sub_job_ids = count(2, JOB_ID_STRIDE)

    def _build_task(
        self,
        job: PreparedJob,
        transfer_batches: tuple[TransferBatch, ...],
        *,
        is_layerwise: bool,
        prerequisite_handle: int,
    ) -> TransferTask:
        sub_job_id = None
        if is_layerwise:
            sub_job_id = next(self._sub_job_ids)
            if sub_job_id > MAX_NATIVE_JOB_ID:
                raise OverflowError("UCM subjob IDs exhausted the uint64 range")
        # Payload order follows the Prepare input, independently of the token.
        # Keep the complete opaque token; empty payloads leave other objects
        # untouched. Only the binding conversion copies token fields.
        payloads = [self._proxy.BlockPayload() for _ in job.ucm_keys]
        payload_index_by_key = {
            ucm_key: index for index, ucm_key in enumerate(job.ucm_keys)
        }
        for transfer_batch in transfer_batches:
            for batch_key_index, ucm_key in enumerate(transfer_batch.ucm_keys):
                payload = payloads[payload_index_by_key[ucm_key]]
                payload.pointers = list(payload.pointers) + [
                    int(pointer)
                    for pointer in transfer_batch.device_ptrs[batch_key_index]
                ]
                payload.sizes = list(payload.sizes) + [
                    int(size) for size in transfer_batch.byte_sizes[batch_key_index]
                ]
                payload.offsets = list(payload.offsets) + [
                    int(offset)
                    for offset in transfer_batch.store_offsets[batch_key_index]
                ]
        native_token = job.token.to_native(self._proxy)
        io_context = self._proxy.IoContext()
        io_context.prerequisite_handle = prerequisite_handle
        return TransferTask(
            job_id=job.job_id,
            sub_job_id=sub_job_id,
            transfer_batches=transfer_batches,
            native_arguments=(payloads, native_token, io_context),
            prerequisite_handle=prerequisite_handle,
        )

    def submit_get(
        self,
        job: PreparedJob,
        transfer_batches: tuple[TransferBatch, ...],
        *,
        is_layerwise: bool = False,
        prerequisite_handle: int = 0,
    ) -> TransferTask:
        task = self._build_task(
            job,
            transfer_batches,
            is_layerwise=is_layerwise,
            prerequisite_handle=prerequisite_handle,
        )
        with self._translate_errors():
            self._worker.submit_get(task.io_job_id, *task.native_arguments)
        self._tasks[task.io_job_id] = task
        return task

    def submit_put(
        self,
        job: PreparedJob,
        transfer_batches: tuple[TransferBatch, ...],
        *,
        is_layerwise: bool = False,
        prerequisite_handle: int = 0,
    ) -> TransferTask:
        task = self._build_task(
            job,
            transfer_batches,
            is_layerwise=is_layerwise,
            prerequisite_handle=prerequisite_handle,
        )
        with self._translate_errors():
            self._worker.submit_put(task.io_job_id, *task.native_arguments)
        self._tasks[task.io_job_id] = task
        return task

    def submit_prefetch(self, job: PreparedJob) -> TransferTask:
        native_token = job.token.to_native(self._proxy)
        with self._translate_errors():
            self._worker.submit_prefetch(job.job_id, native_token)
        task = TransferTask(
            job_id=job.job_id,
            sub_job_id=None,
            transfer_batches=(),
            native_arguments=(native_token,),
        )
        self._tasks[task.io_job_id] = task
        return task

    def _consume_results(self, results: Any) -> None:
        for result in results:
            task = self._tasks.pop(result.job_id)
            task.success = bool(result.success)
            task.transfer_batches = ()
            task.native_arguments = ()

    def poll_finished(self) -> None:
        self._consume_results(self._worker.get_finished())

    def wait(self, task: TransferTask) -> None:
        if task.success is None:
            # A thrown exception or missing completion leaves the task owned:
            # neither proves that the native transfer has stopped using KV.
            self._consume_results(self._worker.wait([task.io_job_id]))
        if task.success is None:
            raise RuntimeError(f"Store wait did not terminate job {task.io_job_id}")
        if not task.success:
            raise StoreIOError(f"Store job {task.io_job_id} failed")

    def close(self) -> None:
        first_error = None
        for task in tuple(self._tasks.values()):
            try:
                self.wait(task)
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
