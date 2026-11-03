"""Python binding ownership, wire tokens and native transfer completion.

Routing, admission, staging and publication belong to ucm/v2. This module
only packs the binding's types and retains transfer descriptions until IO
has terminated. It does not implement a Python proxy or a storage backend.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
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

    def to_native(self, module: Any) -> Any:
        token = module.AddressedToken()
        token.keys = [module.BlockId(key) for key in self.keys]
        token.payload = self.payload
        return token


# Native Submit/Wait currently use one size_t id. Reserve each Scheduler job
# a range of uint64 ids: offset 0 is bulk, positive offsets are layerwise
# subjobs. Native completion queues are local to each Worker instance.
SUB_JOB_ID_BITS = 32
JOB_ID_STRIDE = 1 << SUB_JOB_ID_BITS


@dataclass(frozen=True)
class PreparedJob:
    """Scheduler-assigned job identity and the result of its Prepare."""

    job_id: int
    token: AddressedToken


@dataclass(frozen=True)
class Transfer:
    request_id: str
    ucm_chunk_ids: tuple[bytes, ...]
    chunk_indices: np.ndarray
    ptrs: np.ndarray
    sizes: np.ndarray
    ucm_chunk_offsets: np.ndarray
    record_nbytes: np.ndarray | None = None


@dataclass
class TransferTask:
    job_id: int
    sub_job_id: int | None
    transfers: tuple[Transfer, ...]
    arguments: tuple[Any, ...]
    success: bool | None = None

    @property
    def native_job_id(self) -> int:
        return self.job_id + (self.sub_job_id or 0)


class StoreClient:
    """Keep StoreAssembler alive for both of its borrowed role interfaces."""

    def __init__(self, launch_config: dict[str, Any]) -> None:
        self.module = importlib.import_module(
            launch_config.get("store_module", "ucm_proxy")
        )
        self.assembler = self.module.StoreAssembler(launch_config["store_config"])
        self.scheduler = self.assembler.get_scheduler()
        self.worker = self.assembler.get_worker()
        self._tasks: dict[int, TransferTask] = {}

    def call(self, method: Callable, *arguments: Any) -> Any:
        try:
            return method(*arguments)
        except (OSError, MemoryError) as exc:
            raise StoreIOError(str(exc)) from exc
        except self.module.UcmError as exc:
            # The current binding exposes Status as a text-only exception.
            # Contract/setup failures remain fatal; only explicit IO/resource
            # failures can be handled as misses or best-effort failed saves.
            name = str(exc).split(" ", 1)[0]
            if name in {
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

    def prepare(
        self, operation: str, request_id: str, keys: tuple[bytes, ...] = ()
    ) -> AddressedToken:
        method = getattr(self.scheduler, f"prepare_{operation}")
        arguments = (
            (request_id,) if operation == "prefetch" else (request_id, b"".join(keys))
        )
        return AddressedToken.from_native(self.call(method, *arguments))

    def submit(
        self,
        operation: str,
        job: PreparedJob,
        transfers: tuple[Transfer, ...],
        sub_job_id: int | None = None,
    ) -> TransferTask:
        module = self.module
        token = job.token
        # One Prepare may cover multiple PP objects/policies. Keep the complete
        # opaque token and its key order, submitting empty payloads for keys
        # this worker/layer does not touch. No raw pointer crosses processes.
        payloads = [module.BlockPayload() for _ in token.keys]
        positions = {key: index for index, key in enumerate(token.keys)}
        for transfer in transfers:
            for row, key in enumerate(transfer.ucm_chunk_ids):
                payload = payloads[positions[key]]
                payload.pointers = list(payload.pointers) + [
                    int(ptr) for ptr in transfer.ptrs[row]
                ]
                payload.sizes = list(payload.sizes) + [
                    int(size) for size in transfer.sizes[row]
                ]
                payload.offsets = list(payload.offsets) + [
                    int(offset) for offset in transfer.ucm_chunk_offsets[row]
                ]
                if transfer.record_nbytes is not None:
                    payload.total_size = int(transfer.record_nbytes[row])
        native_token = token.to_native(module)
        io_context = module.IoContext()
        arguments = (payloads, native_token, io_context)
        task = TransferTask(job.job_id, sub_job_id, transfers, arguments)
        self.call(
            getattr(self.worker, f"submit_{operation}"), task.native_job_id, *arguments
        )
        self._tasks[task.native_job_id] = task
        return task

    def submit_prefetch(self, job: PreparedJob) -> TransferTask:
        native_token = job.token.to_native(self.module)
        self.call(self.worker.submit_prefetch, job.job_id, native_token)
        task = TransferTask(job.job_id, None, (), (native_token,))
        self._tasks[task.native_job_id] = task
        return task

    def _consume(self, results: Any) -> None:
        for result in results:
            task = self._tasks.pop(result.job_id)
            task.success = bool(result.success)
            task.arguments = ()

    def poll_finished(self) -> None:
        self._consume(self.worker.get_finished())

    def wait(self, task: TransferTask) -> None:
        if task.success is None:
            # A thrown exception or missing completion leaves the task owned:
            # neither proves that the native transfer has stopped using KV.
            self._consume(self.worker.wait([task.native_job_id]))
        if task.success is None:
            raise RuntimeError(f"Store wait did not terminate job {task.native_job_id}")
        if not task.success:
            raise StoreIOError(f"Store job {task.native_job_id} failed")

    def close(self) -> None:
        first_error = None
        for task in tuple(self._tasks.values()):
            try:
                self.wait(task)
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
