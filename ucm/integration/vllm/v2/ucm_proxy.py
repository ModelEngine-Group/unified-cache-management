"""The storage-neutral synchronous Proxy boundary used by connector v2."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable
from uuid import uuid4

if TYPE_CHECKING:
    import torch

    KVCacheValue: TypeAlias = (
        torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
    )
else:
    KVCacheValue: TypeAlias = object


class UCMProxyError(RuntimeError):
    """Normalized error raised by the v2 Proxy adapter."""


class UCMProxy(Protocol):
    def lookup(self, block_ids: Sequence[bytes]) -> Sequence[bool]: ...

    def load(
        self,
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> object | None: ...

    def dump(
        self,
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> object | None: ...


@runtime_checkable
class UCMProxyWaiter(Protocol):
    def wait(self, task: object) -> None: ...


@runtime_checkable
class UCMProxyTensorRegistration(Protocol):
    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None: ...


@runtime_checkable
class UCMByteAccess(Protocol):
    """Copy byte ranges between integer pointers and host bytes."""

    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None: ...

    def synchronize(self) -> None: ...

    def read(self, ptr: int, size: int) -> bytes: ...

    def write(self, ptr: int, payload: bytes) -> None: ...


def _tensor_views(value: KVCacheValue) -> tuple[torch.Tensor, ...]:
    torch_module = importlib.import_module("torch")
    if isinstance(value, torch_module.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in value:
            result.extend(_tensor_views(item))
        return tuple(result)
    raise TypeError(f"Unsupported KV cache value: {type(value).__name__}")


class TorchTensorByteAccess:
    """Resolve raw pointers against registered Torch CPU/CUDA/NPU storages."""

    def __init__(self) -> None:
        self._buffers: list[tuple[int, int, torch.Tensor]] = []
        self._devices: set[torch.device] = set()

    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        torch_module = importlib.import_module("torch")
        buffers: list[tuple[int, int, torch.Tensor]] = []
        devices: set[torch.device] = set()
        seen: set[tuple[int, int]] = set()
        for value in kv_caches.values():
            for tensor in _tensor_views(value):
                storage = tensor.untyped_storage()
                base = int(storage.data_ptr())
                size = int(storage.nbytes())
                if not base or size <= 0 or (base, size) in seen:
                    continue
                seen.add((base, size))
                byte_view = torch_module.empty(
                    0, dtype=torch_module.uint8, device=tensor.device
                ).set_(storage, 0, (size,), (1,))
                buffers.append((base, base + size, byte_view))
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
        for base, end, tensor in self._buffers:
            if base <= ptr and ptr + size <= end:
                offset = ptr - base
                return tensor[offset : offset + size]
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


class SimpleFileUCMProxy:
    """Minimal standalone byte-range Proxy for synchronous v2 bulk I/O.

    A record is stored as one raw file named by its 16-byte key.  Dump calls
    must provide every byte in each record exactly once.  Publication uses an
    atomic rename, so lookup never observes a partially written record.
    """

    _SUFFIX = ".ucm"

    def __init__(
        self,
        root: str | os.PathLike[str],
        byte_access: UCMByteAccess | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.byte_access = byte_access or TorchTensorByteAccess()

    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        self.byte_access.register_tensors(kv_caches)

    def _path(self, key: bytes) -> Path:
        return self.root / f"{bytes(key).hex()}{self._SUFFIX}"

    def lookup(self, block_ids: Sequence[bytes]) -> tuple[bool, ...]:
        return tuple(self._path(key).is_file() for key in block_ids)

    @staticmethod
    def _records(
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> dict[bytes, list[tuple[int, int, int]]]:
        records: dict[bytes, list[tuple[int, int, int]]] = {}
        for key, offset, ptr, size in zip(block_ids, offsets, ptrs, sizes, strict=True):
            records.setdefault(bytes(key), []).append(
                (int(offset), int(ptr), int(size))
            )
        return records

    @staticmethod
    def _record_size(key: bytes, segments: list[tuple[int, int, int]]) -> int:
        cursor = 0
        for offset, _ptr, size in sorted(segments):
            if offset != cursor:
                kind = "overlap" if offset < cursor else "gap"
                raise ValueError(
                    f"Record {key.hex()} has a {kind} at byte {cursor}: "
                    f"next offset={offset}"
                )
            cursor += size
        if cursor <= 0:
            raise ValueError(f"Record {key.hex()} is empty")
        return cursor

    def dump(
        self,
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> None:
        records = self._records(block_ids, offsets, ptrs, sizes)
        self.byte_access.synchronize()
        for key, segments in records.items():
            record_size = self._record_size(key, segments)
            record = bytearray(record_size)
            for offset, ptr, size in segments:
                payload = self.byte_access.read(ptr, size)
                if len(payload) != size:
                    raise RuntimeError(
                        f"Byte access returned {len(payload)} bytes, expected {size}"
                    )
                record[offset : offset + size] = payload
            target = self._path(key)
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_bytes(record)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

    def load(
        self,
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> None:
        records = self._records(block_ids, offsets, ptrs, sizes)
        for key, segments in records.items():
            path = self._path(key)
            try:
                record = path.read_bytes()
            except FileNotFoundError as exc:
                raise KeyError(f"UCM record not found: {key.hex()}") from exc
            for offset, ptr, size in segments:
                end = offset + size
                if end > len(record):
                    raise ValueError(
                        f"Load range [{offset}, {end}) exceeds record "
                        f"{key.hex()} size {len(record)}"
                    )
                self.byte_access.write(ptr, record[offset:end])
        self.byte_access.synchronize()

    def wait(self, task: object) -> None:
        if task is not None:
            raise ValueError(f"SimpleFileUCMProxy is synchronous, got task {task!r}")


@dataclass(frozen=True)
class UCMProxyBatch:
    block_ids: tuple[bytes, ...]
    offsets: tuple[int, ...]
    ptrs: tuple[int, ...]
    sizes: tuple[int, ...]

    @property
    def total_bytes(self) -> int:
        return sum(self.sizes)


class UCMProxyAdapter:
    """Validate and normalize calls without knowing the backing Store."""

    def __init__(
        self,
        proxy: UCMProxy,
        record_sizes: dict[bytes, int] | None = None,
    ) -> None:
        self._proxy = proxy
        self._record_sizes = record_sizes or {}

    @staticmethod
    def _keys(block_ids: Sequence[bytes]) -> tuple[bytes, ...]:
        keys = tuple(bytes(key) for key in block_ids)
        invalid = [index for index, key in enumerate(keys) if len(key) != 16]
        if invalid:
            raise ValueError(
                f"UCM block IDs must be 16 bytes; invalid indexes={invalid}"
            )
        return keys

    def lookup(self, block_ids: Sequence[bytes]) -> tuple[bool, ...]:
        keys = self._keys(block_ids)
        try:
            result = tuple(bool(value) for value in self._proxy.lookup(keys))
        except Exception as exc:
            raise UCMProxyError("Proxy lookup failed") from exc
        if len(result) != len(keys):
            raise UCMProxyError(
                f"Proxy lookup returned {len(result)} results for {len(keys)} keys"
            )
        return result

    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        if isinstance(self._proxy, UCMProxyTensorRegistration):
            self._proxy.register_tensors(kv_caches)

    def _wait(self, operation: str, task: object | None) -> None:
        if task is None:
            return
        if not isinstance(self._proxy, UCMProxyWaiter):
            raise UCMProxyError(
                f"Proxy {operation} returned an asynchronous task but does not "
                "provide wait(task)"
            )
        self._proxy.wait(task)

    def _batch(
        self,
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> UCMProxyBatch:
        keys = self._keys(block_ids)
        normalized = (
            tuple(int(value) for value in offsets),
            tuple(int(value) for value in ptrs),
            tuple(int(value) for value in sizes),
        )
        lengths = {len(keys), *(len(values) for values in normalized)}
        if len(lengths) != 1:
            raise ValueError(
                "block_ids, offsets, ptrs and sizes must have identical lengths"
            )
        normalized_offsets, normalized_ptrs, normalized_sizes = normalized
        for index, (key, offset, ptr, size) in enumerate(
            zip(keys, normalized_offsets, normalized_ptrs, normalized_sizes)
        ):
            if offset < 0 or ptr <= 0 or size <= 0:
                raise ValueError(
                    f"Invalid Proxy segment at index {index}: "
                    f"offset={offset}, ptr={ptr}, size={size}"
                )
            record_size = self._record_sizes.get(key)
            if record_size is not None and offset + size > record_size:
                raise ValueError(
                    f"Proxy segment {index} exceeds record: "
                    f"offset={offset}, size={size}, record_size={record_size}"
                )
        return UCMProxyBatch(
            keys, normalized_offsets, normalized_ptrs, normalized_sizes
        )

    def load(
        self,
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> None:
        batch = self._batch(block_ids, offsets, ptrs, sizes)
        if not batch.block_ids:
            return
        try:
            task = self._proxy.load(
                batch.block_ids, batch.offsets, batch.ptrs, batch.sizes
            )
            self._wait("load", task)
        except Exception as exc:
            if isinstance(exc, UCMProxyError):
                raise
            raise UCMProxyError("Proxy load failed") from exc

    def dump(
        self,
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> None:
        batch = self._batch(block_ids, offsets, ptrs, sizes)
        if not batch.block_ids:
            return
        try:
            task = self._proxy.dump(
                batch.block_ids, batch.offsets, batch.ptrs, batch.sizes
            )
            self._wait("dump", task)
        except Exception as exc:
            if isinstance(exc, UCMProxyError):
                raise
            raise UCMProxyError("Proxy dump failed") from exc
