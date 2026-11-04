"""Development stores and their CPU/device byte access boundary."""

from __future__ import annotations

import importlib
import json
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import uuid4

import numpy as np

from .ucm_store import KVCacheValue

if TYPE_CHECKING:
    import torch


@runtime_checkable
class UCMByteAccess(Protocol):
    """Copy byte ranges between integer pointers and host bytes."""

    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None: ...

    def synchronize(self) -> None: ...

    def read(self, ptr: int, size: int) -> bytes: ...

    def write(self, ptr: int, payload: bytes) -> None: ...


def _tensor_views(kv_cache: KVCacheValue) -> tuple[torch.Tensor, ...]:
    torch_module = importlib.import_module("torch")
    if isinstance(kv_cache, torch_module.Tensor):
        return (kv_cache,)
    if isinstance(kv_cache, (tuple, list)):
        tensor_views: list[torch.Tensor] = []
        for element in kv_cache:
            tensor_views.extend(_tensor_views(element))
        return tuple(tensor_views)
    raise TypeError(f"Unsupported KV cache value: {type(kv_cache).__name__}")


class TorchTensorByteAccess:
    """Resolve raw pointers against registered Torch CPU/CUDA/NPU storages."""

    def __init__(self) -> None:
        self._buffers: list[tuple[int, int, torch.Tensor]] = []
        self._devices: set[torch.device] = set()

    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        torch_module = importlib.import_module("torch")
        buffers: list[tuple[int, int, torch.Tensor]] = []
        devices: set[torch.device] = set()
        seen_storages: set[tuple[int, int]] = set()
        for kv_cache in kv_caches.values():
            for tensor in _tensor_views(kv_cache):
                storage = tensor.untyped_storage()
                storage_start = int(storage.data_ptr())
                storage_bytes = int(storage.nbytes())
                if (
                    not storage_start
                    or storage_bytes <= 0
                    or (storage_start, storage_bytes) in seen_storages
                ):
                    continue
                seen_storages.add((storage_start, storage_bytes))
                byte_view = torch_module.empty(
                    0, dtype=torch_module.uint8, device=tensor.device
                ).set_(storage, 0, (storage_bytes,), (1,))
                buffers.append(
                    (storage_start, storage_start + storage_bytes, byte_view)
                )
                devices.add(tensor.device)
        if not buffers:
            raise ValueError("No non-empty KV cache tensor storage was registered")
        self._buffers = buffers
        self._devices = devices

    def synchronize(self) -> None:
        torch_module = importlib.import_module("torch")
        for device in self._devices:
            if device.type == "cuda":
                torch_module.cuda.synchronize(device)
            elif device.type == "npu":
                npu = getattr(torch_module, "npu", None)
                if npu is None:
                    raise RuntimeError("Torch NPU support is unavailable")
                npu.synchronize(device)

    def _view(self, ptr: int, size: int) -> torch.Tensor:
        # Callers hand us numpy scalars (the batch arrays are uint64);
        # torch slicing wants plain ints.
        ptr, size = int(ptr), int(size)
        for buffer_start, buffer_end, byte_view in self._buffers:
            if buffer_start <= ptr and ptr + size <= buffer_end:
                byte_offset = ptr - buffer_start
                return byte_view[byte_offset : byte_offset + size]
        raise ValueError(
            f"Pointer range [{ptr}, {ptr + size}) is outside registered KV caches"
        )

    def read(self, ptr: int, size: int) -> bytes:
        return self._view(ptr, size).cpu().numpy().tobytes()

    def write(self, ptr: int, payload: bytes) -> None:
        torch_module = importlib.import_module("torch")
        target = self._view(ptr, len(payload))
        source = torch_module.frombuffer(bytearray(payload), dtype=torch_module.uint8)
        target.copy_(source.to(target.device))


def record_segments(
    ucm_chunk_ids: Sequence[bytes],
    offsets: np.ndarray,
    ptrs: np.ndarray,
    sizes: np.ndarray,
) -> Iterable[tuple[bytes, Iterable[tuple[int, int, int]]]]:
    if np.ndim(ptrs) == 2:
        if np.shape(offsets) != np.shape(ptrs) or np.shape(sizes) != np.shape(ptrs):
            raise ValueError("Transfer arrays must have identical [K, S] shapes")
        if ptrs.shape[1] == 1:
            # A single segment needs no row views or three map iterators.
            # Column views also work for strided/broadcast matrices.
            return (
                (bytes(ucm_chunk_id), ((int(o), int(p), int(s)),))
                for ucm_chunk_id, o, p, s in zip(
                    ucm_chunk_ids, offsets[:, 0], ptrs[:, 0], sizes[:, 0], strict=True
                )
            )
        return (
            (bytes(ucm_chunk_id), zip(map(int, o), map(int, p), map(int, s)))
            for ucm_chunk_id, o, p, s in zip(
                ucm_chunk_ids, offsets, ptrs, sizes, strict=True
            )
        )
    # Compatibility for the legacy flat test/toolkit API only.
    records: dict[bytes, list[tuple[int, int, int]]] = {}
    for ucm_chunk_id, offset, ptr, size in zip(
        ucm_chunk_ids, offsets, ptrs, sizes, strict=True
    ):
        records.setdefault(bytes(ucm_chunk_id), []).append(
            (int(offset), int(ptr), int(size))
        )
    return records.items()


class FileTestStore:
    """Synchronous file backend with explicit UCM chunk publication.

    Connector attempts accumulate disjoint ranges in a shared hidden file.
    Scheduler publishes a complete attempt with atomic create-if-absent. The
    direct dump/commit API also supports private records for standalone tests.
    """

    _SUFFIX = ".ucm"

    def __init__(
        self,
        root: str | os.PathLike[str],
        byte_access: UCMByteAccess | None = None,
    ) -> None:
        self.backend = None
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.byte_access = byte_access or TorchTensorByteAccess()
        self._pending: dict[bytes, Path] = {}
        self._pending_metadata: dict[bytes, str] = {}

    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        self.byte_access.register_tensors(kv_caches)

    def _path(self, ucm_chunk_id: bytes) -> Path:
        return self.root / f"{bytes(ucm_chunk_id).hex()}{self._SUFFIX}"

    def lookup(self, ucm_chunk_ids: Sequence[bytes]) -> tuple[bool, ...]:
        return tuple(
            self._path(ucm_chunk_id).is_file() for ucm_chunk_id in ucm_chunk_ids
        )

    def lookup_on_prefix(self, ucm_chunk_ids: Sequence[bytes]) -> int:
        return next(
            (
                row - 1
                for row, ucm_chunk_id in enumerate(ucm_chunk_ids)
                if not self._path(ucm_chunk_id).is_file()
            ),
            len(ucm_chunk_ids) - 1,
        )

    def lookup_on_reverse(self, ucm_chunk_ids: Sequence[bytes]) -> int:
        return next(
            (
                row
                for row in range(len(ucm_chunk_ids) - 1, -1, -1)
                if self._path(ucm_chunk_ids[row]).is_file()
            ),
            -1,
        )

    def dump(
        self,
        ucm_chunk_ids: Sequence[bytes],
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
        metadata: Sequence[str] = (),
        record_nbytes: np.ndarray | None = None,
    ) -> None:
        if metadata:
            self._pending_metadata.update(zip(ucm_chunk_ids, metadata, strict=True))
        self.byte_access.synchronize()
        for row, (ucm_chunk_id, segments) in enumerate(
            record_segments(ucm_chunk_ids, offsets, ptrs, sizes)
        ):
            if metadata and json.loads(metadata[row]).get("shared_attempt", False):
                self._dump_attempt(
                    ucm_chunk_id, segments, int(record_nbytes[row]), metadata[row]
                )
                self._pending_metadata.pop(ucm_chunk_id, None)
                continue
            temporary = self._pending.get(ucm_chunk_id)
            if temporary is None:
                target = self._path(ucm_chunk_id)
                temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
                mode = "xb"
            else:
                mode = "r+b"
            try:
                with temporary.open(mode) as stream:
                    self._pending[ucm_chunk_id] = temporary
                    if record_nbytes is not None and mode == "xb":
                        stream.truncate(int(record_nbytes[row]))
                    for offset, ptr, size in segments:
                        payload = self.byte_access.read(ptr, size)
                        if len(payload) != size:
                            raise RuntimeError(
                                f"Byte access returned {len(payload)} bytes, expected {size}"
                            )
                        stream.seek(offset)
                        stream.write(payload)
            except Exception:
                self._pending.pop(ucm_chunk_id, None)
                self._pending_metadata.pop(ucm_chunk_id, None)
                temporary.unlink(missing_ok=True)
                raise

    def _attempt_path(self, staging_id):
        return self.root / f".{staging_id.hex()}.attempt"

    def _dump_attempt(self, staging_id, segments, total_bytes, metadata):
        path = self._attempt_path(staging_id)
        # Every writer opens the same hidden attempt, never O_TRUNC. Repeating
        # ftruncate to the agreed full size preserves ranges written by peers.
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, total_bytes)
            for offset, ptr, count in segments:
                if offset < 0 or offset + count > total_bytes:
                    raise ValueError("Dump range exceeds global object size")
                payload = self.byte_access.read(ptr, count)
                if len(payload) != count:
                    raise ValueError("Byte access returned an incomplete range")
                written = 0
                while written < count:
                    n = os.pwrite(fd, payload[written:], offset + written)
                    if n == 0:
                        raise OSError("Attempt write made no progress")
                    written += n
        finally:
            os.close(fd)
        # Metadata is identical on all writers and is not part of visibility.
        path.with_suffix(".attempt.json").write_text(metadata)

    def commit(
        self, success: bool, staging_ids: Sequence[bytes], object_ids: Sequence[bytes]
    ) -> None:
        """Publish complete attempts on success; otherwise discard their staging.

        Both decisions are idempotent and preserve existing published objects.
        Scheduler calls this only after every required writer has terminated IO.
        """
        if not success:
            self._discard(staging_ids)
            return
        for staging_id, object_id in zip(staging_ids, object_ids, strict=True):
            source = self._pending.get(staging_id, self._attempt_path(staging_id))
            target = self._path(object_id)
            try:
                # Metadata is diagnostic; only the final object link controls visibility.
                description = self._pending_metadata.get(staging_id)
                metadata_path = self._attempt_path(staging_id).with_suffix(
                    ".attempt.json"
                )
                if description is None and metadata_path.exists():
                    description = metadata_path.read_text()
                if description is not None:
                    metadata = json.loads(description)
                    metadata.pop("shared_attempt", None)
                    try:
                        with target.with_suffix(".json").open("x") as stream:
                            json.dump(metadata, stream)
                    except FileExistsError:
                        pass
                os.link(source, target)
            except FileExistsError:
                pass
            except FileNotFoundError:
                if not target.is_file():
                    raise
            self._discard((staging_id,))

    def load(
        self,
        ucm_chunk_ids: Sequence[bytes],
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
    ) -> None:
        records = record_segments(ucm_chunk_ids, offsets, ptrs, sizes)
        for ucm_chunk_id, segments in records:
            path = self._path(ucm_chunk_id)
            try:
                record = path.read_bytes()
            except FileNotFoundError as exc:
                raise KeyError(f"UCM record not found: {ucm_chunk_id.hex()}") from exc
            for offset, ptr, size in segments:
                end = offset + size
                if end > len(record):
                    raise ValueError(
                        f"Load range [{offset}, {end}) exceeds record "
                        f"{ucm_chunk_id.hex()} size {len(record)}"
                    )
                self.byte_access.write(ptr, record[offset:end])
        self.byte_access.synchronize()

    def wait(self, task: object) -> None:
        if task is not None:
            raise ValueError(f"FileTestStore is synchronous, got task {task!r}")

    def read_record(self, ucm_chunk_id: bytes) -> tuple[bytes, str]:
        try:
            payload = self._path(ucm_chunk_id).read_bytes()
        except FileNotFoundError as exc:
            raise KeyError(ucm_chunk_id) from exc
        metadata_path = self._path(ucm_chunk_id).with_suffix(".json")
        metadata = metadata_path.read_text() if metadata_path.exists() else "{}"
        return payload, metadata

    def write_record(self, ucm_chunk_id: bytes, payload: bytes, metadata: str) -> None:
        target = self._path(ucm_chunk_id)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(payload)
            target.with_suffix(".json").write_text(metadata)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _discard(self, ucm_chunk_ids: Sequence[bytes]) -> None:
        for ucm_chunk_id in ucm_chunk_ids:
            self._attempt_path(ucm_chunk_id).unlink(missing_ok=True)
            self._attempt_path(ucm_chunk_id).with_suffix(".attempt.json").unlink(
                missing_ok=True
            )
            temporary = self._pending.pop(ucm_chunk_id, None)
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            self._pending_metadata.pop(ucm_chunk_id, None)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._discard(tuple(self._pending))


class MemoryTestStore:
    """Bounded development cache with synchronous promotion and writeback.

    Whole-record methods are private to the test Store family. Proxy never
    uses them. Failed writeback pins its immutable source until flush retries.
    """

    def __init__(
        self,
        byte_access: UCMByteAccess | None = None,
        *,
        backend: FileTestStore | MemoryTestStore | None = None,
        capacity_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be nonnegative")
        self.byte_access = byte_access or TorchTensorByteAccess()
        self.backend = backend
        self.capacity_bytes = capacity_bytes
        self.records: OrderedDict[bytes, tuple[bytes, str]] = OrderedDict()
        self._pending: dict[bytes, tuple[bytearray, str]] = {}
        self._writeback: dict[bytes, tuple[bytes, str]] = {}
        self._reused: set[bytes] = set()

    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        self.byte_access.register_tensors(kv_caches)

    def _reserve(self, size: int) -> None:
        used = sum(len(payload) for payload, _ in self.records.values())
        used += sum(len(payload) for payload, _ in self._pending.values())
        victims = []
        for ucm_chunk_id, (payload, _) in self.records.items():
            if used + size <= self.capacity_bytes:
                break
            if ucm_chunk_id not in self._writeback and ucm_chunk_id not in self._reused:
                victims.append(ucm_chunk_id)
                used -= len(payload)
        if used + size > self.capacity_bytes:
            raise MemoryError(
                "Test Store capacity exhausted by pending or pinned records"
            )
        for ucm_chunk_id in victims:
            del self.records[ucm_chunk_id]

    def lookup(self, ucm_chunk_ids: Sequence[bytes]) -> tuple[bool, ...]:
        hits = [ucm_chunk_id in self.records for ucm_chunk_id in ucm_chunk_ids]
        missing_rows = [row for row, is_hit in enumerate(hits) if not is_hit]
        if self.backend is not None and missing_rows:
            backend_hits = self.backend.lookup(
                tuple(ucm_chunk_ids[row] for row in missing_rows)
            )
            for row, is_hit in zip(missing_rows, backend_hits, strict=True):
                hits[row] = is_hit
        return tuple(hits)

    def _has_record(self, ucm_chunk_id: bytes) -> bool:
        if ucm_chunk_id in self.records:
            return True
        else:
            return self.backend is not None and bool(
                self.backend.lookup((ucm_chunk_id,))[0]
            )

    def lookup_on_prefix(self, ucm_chunk_ids: Sequence[bytes]) -> int:
        return next(
            (
                row - 1
                for row, ucm_chunk_id in enumerate(ucm_chunk_ids)
                if not self._has_record(ucm_chunk_id)
            ),
            len(ucm_chunk_ids) - 1,
        )

    def lookup_on_reverse(self, ucm_chunk_ids: Sequence[bytes]) -> int:
        return next(
            (
                row
                for row in range(len(ucm_chunk_ids) - 1, -1, -1)
                if self._has_record(ucm_chunk_ids[row])
            ),
            -1,
        )

    def read_record(self, ucm_chunk_id: bytes) -> tuple[bytes, str]:
        if ucm_chunk_id in self.records:
            self.records.move_to_end(ucm_chunk_id)
            return self.records[ucm_chunk_id]
        if self.backend is None:
            raise KeyError(ucm_chunk_id)
        record = self.backend.read_record(ucm_chunk_id)
        # A partial device load promotes the complete record, never just a layer.
        if ucm_chunk_id not in self._pending:
            try:
                self._reserve(len(record[0]))
            except MemoryError:
                pass
            else:
                self.records[ucm_chunk_id] = record
        return record

    def write_record(self, ucm_chunk_id: bytes, payload: bytes, metadata: str) -> None:
        if ucm_chunk_id not in self.records:
            self._reserve(len(payload))
            self.records[ucm_chunk_id] = (bytes(payload), metadata)
        if self.backend is not None:
            self._writeback[ucm_chunk_id] = self.records[ucm_chunk_id]
            self.backend.write_record(ucm_chunk_id, *self._writeback[ucm_chunk_id])
            del self._writeback[ucm_chunk_id]

    def dump(
        self,
        ucm_chunk_ids: Sequence[bytes],
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
        metadata: Sequence[str],
        record_nbytes: np.ndarray,
    ) -> None:
        if (
            self.backend is not None
            and metadata
            and all(json.loads(item).get("shared_attempt", False) for item in metadata)
        ):
            self.backend.byte_access = self.byte_access
            return self.backend.dump(
                ucm_chunk_ids, offsets, ptrs, sizes, metadata, record_nbytes
            )
        self.byte_access.synchronize()
        for row, (ucm_chunk_id, segments) in enumerate(
            record_segments(ucm_chunk_ids, offsets, ptrs, sizes)
        ):
            if ucm_chunk_id in self.records:
                self._reused.add(ucm_chunk_id)
                continue
            size = int(record_nbytes[row])
            if ucm_chunk_id not in self._pending:
                self._reserve(size)
                self._pending[ucm_chunk_id] = (bytearray(size), metadata[row])
            payload, description = self._pending[ucm_chunk_id]
            if len(payload) != size or description != metadata[row]:
                raise ValueError(
                    "Inconsistent record size or metadata across dump calls"
                )
            for offset, ptr, count in segments:
                if offset < 0 or offset + count > size:
                    raise ValueError("Dump range exceeds complete record size")
                data = self.byte_access.read(ptr, count)
                if len(data) != count:
                    raise ValueError("Byte access returned an incomplete range")
                payload[offset : offset + count] = data

    def commit(
        self, success: bool, staging_ids: Sequence[bytes], object_ids: Sequence[bytes]
    ) -> None:
        if not success:
            self._discard(staging_ids)
            return
        for staging_id, object_id in zip(staging_ids, object_ids, strict=True):
            # Connector attempts are staged by the shared backend. Local pending
            # buffers exist only for explicit in-process Store tests.
            local = (
                staging_id in self._pending
                or staging_id in self._reused
                or staging_id in self.records
            )
            if not local and self.backend is not None:
                self.backend.commit(True, (staging_id,), (object_id,))
                continue
            self._reused.discard(staging_id)
            if staging_id in self._pending:
                payload, metadata = self._pending.pop(staging_id)
                self.records.setdefault(object_id, (bytes(payload), metadata))
            if object_id not in self.records:
                raise KeyError(f"No pending record to commit: {staging_id.hex()}")
            if self.backend is not None:
                self._writeback[object_id] = self.records[object_id]
                self.backend.write_record(object_id, *self._writeback[object_id])
                del self._writeback[object_id]

    def load(
        self,
        ucm_chunk_ids: Sequence[bytes],
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
    ) -> None:
        for ucm_chunk_id, segments in record_segments(
            ucm_chunk_ids, offsets, ptrs, sizes
        ):
            payload, _ = self.read_record(ucm_chunk_id)
            for offset, ptr, size in segments:
                if offset < 0 or offset + size > len(payload):
                    raise ValueError("Load range exceeds record size")
                self.byte_access.write(ptr, payload[offset : offset + size])
        self.byte_access.synchronize()

    def _discard(self, ucm_chunk_ids: Sequence[bytes]) -> None:
        if self.backend is not None:
            self.backend.commit(False, ucm_chunk_ids, ucm_chunk_ids)
        for ucm_chunk_id in ucm_chunk_ids:
            self._pending.pop(ucm_chunk_id, None)
            self._reused.discard(ucm_chunk_id)

    def wait(self, task: object) -> None:
        if task is not None:
            raise ValueError("MemoryTestStore is synchronous")

    def flush(self) -> None:
        if self.backend is not None:
            for ucm_chunk_id, record in tuple(self._writeback.items()):
                self.backend.write_record(ucm_chunk_id, *record)
                del self._writeback[ucm_chunk_id]
            self.backend.flush()

    def close(self) -> None:
        self.flush()
        self._pending.clear()
        self._reused.clear()
        self.records.clear()
        if self.backend is not None:
            self.backend.close()
