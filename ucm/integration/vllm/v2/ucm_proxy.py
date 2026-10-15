"""The storage-neutral transfer and completion boundary used by connector v2."""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable
from uuid import uuid4

import numpy as np

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
    """Native load/dump receive unique keys and matching [K,S] arrays.

    Read-only, non-contiguous arrays are valid. Returning a task requires wait;
    enqueue retains descriptors until wait; submit waits synchronously.
    """

    def lookup(self, block_ids: Sequence[bytes]) -> Sequence[bool]: ...

    def lookup_on_prefix(self, block_ids: Sequence[bytes]) -> int: ...

    def lookup_on_reverse(self, block_ids: Sequence[bytes]) -> int: ...

    def load(
        self,
        block_ids: Sequence[bytes],
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
    ) -> object | None: ...

    def dump(
        self,
        block_ids: Sequence[bytes],
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
    ) -> object | None: ...

    def commit(self, block_ids: Sequence[bytes]) -> object | None:
        """Publish completed UCM blocks after all their dump calls finish."""
        ...


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
        self._buffers = buffers
        self._devices = devices

    def synchronize(self) -> None:
        torch_module = importlib.import_module("torch")
        for device in self._devices:
            if device.type == "cuda":
                torch_module.cuda.synchronize(device)
            elif device.type == "npu":
                torch_module.npu.synchronize(device)

    def _view(self, ptr: int, size: int) -> torch.Tensor:
        # Callers hand us numpy scalars (the batch arrays are uint64);
        # torch slicing wants plain ints.
        ptr, size = int(ptr), int(size)
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
    """Synchronous file backend with explicit UCM block publication.

    Dump accumulates byte ranges in a private .tmp file per key. Commit
    atomically renames it to the visible .ucm file. The caller guarantees
    all required ranges have been written before committing a key.
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
        self._pending: dict[bytes, Path] = {}

    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        self.byte_access.register_tensors(kv_caches)

    def _path(self, key: bytes) -> Path:
        return self.root / f"{key.hex()}{self._SUFFIX}"

    def lookup(self, block_ids: Sequence[bytes]) -> tuple[bool, ...]:
        return tuple(self._path(key).is_file() for key in block_ids)

    def lookup_on_prefix(self, block_ids: Sequence[bytes]) -> int:
        """Index of the last contiguous present key, -1 if none present."""

        for index, key in enumerate(block_ids):
            if not self._path(key).is_file():
                return index - 1
        return len(block_ids) - 1

    def lookup_on_reverse(self, block_ids: Sequence[bytes]) -> int:
        """Index of the rightmost present key, -1 if none present."""

        for index in range(len(block_ids) - 1, -1, -1):
            if self._path(block_ids[index]).is_file():
                return index
        return -1

    @staticmethod
    def _records(
        block_ids: Sequence[bytes],
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
    ) -> Iterable[tuple[bytes, Iterable[tuple[int, int, int]]]]:
        # TransferBuilder supplies matching native [K,S] arrays. The backend
        # consumes them directly, including strided and broadcast views.
        if ptrs.shape[1] == 1:
            return (
                (key, ((int(o), int(p), int(s)),))
                for key, o, p, s in zip(
                    block_ids, offsets[:, 0], ptrs[:, 0], sizes[:, 0], strict=True
                )
            )
        return (
            (key, zip(map(int, o), map(int, p), map(int, s)))
            for key, o, p, s in zip(block_ids, offsets, ptrs, sizes, strict=True)
        )

    def dump(
        self,
        block_ids: Sequence[bytes],
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
    ) -> None:
        self.byte_access.synchronize()
        for key, segments in self._records(block_ids, offsets, ptrs, sizes):
            temporary = self._pending.get(key)
            if temporary is None:
                target = self._path(key)
                temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
                mode = "xb"
            else:
                mode = "r+b"
            try:
                with temporary.open(mode) as stream:
                    self._pending[key] = temporary
                    for offset, ptr, size in segments:
                        payload = self.byte_access.read(ptr, size)
                        if len(payload) != size:
                            raise RuntimeError(
                                f"Byte access returned {len(payload)} bytes, expected {size}"
                            )
                        stream.seek(offset)
                        stream.write(payload)
            except Exception:
                self._pending.pop(key, None)
                temporary.unlink(missing_ok=True)
                raise

    def commit(self, block_ids: Sequence[bytes]) -> None:
        """Publish only the supplied keys; already committed keys are a no-op.

        Publication is atomic per key, not across the whole key sequence.
        Private temporary names isolate independent Proxy writers.
        """
        for key in block_ids:
            temporary = self._pending.get(key)
            if temporary is None:
                if self._path(key).is_file():
                    continue
                raise KeyError(f"No pending UCM block to commit: {key.hex()}")
            os.replace(temporary, self._path(key))
            del self._pending[key]

    def load(
        self,
        block_ids: Sequence[bytes],
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
    ) -> None:
        records = self._records(block_ids, offsets, ptrs, sizes)
        for key, segments in records:
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
class UCMProxyTransfer:
    """Unique keys [K] and byte descriptors [K,S]; arrays may be read-only.

    Offsets address the complete UCM block. Backends retain arrays through
    completion and never mutate them or require contiguous input.
    """

    keys: tuple[bytes, ...]
    ptrs: np.ndarray
    sizes: np.ndarray
    ucm_block_offsets: np.ndarray


@dataclass(frozen=True)
class UCMProxyTask:
    """Keep transfer arrays alive until the backend task completes."""

    operation: str
    transfer: UCMProxyTransfer
    handle: object | None


class UCMProxyAdapter:
    """Forward native transfers and handle completion."""

    def __init__(self, proxy: UCMProxy) -> None:
        self._proxy = proxy

    def lookup(self, block_ids: Sequence[bytes]) -> tuple[bool, ...]:
        try:
            result = tuple(self._proxy.lookup(block_ids))
        except Exception as exc:
            raise UCMProxyError("Proxy lookup failed") from exc
        if len(result) != len(block_ids):
            raise UCMProxyError(
                f"Proxy lookup returned {len(result)} results for {len(block_ids)} keys"
            )
        return result

    def _scan(
        self, operation: Literal["prefix", "reverse"], block_ids: Sequence[bytes]
    ) -> int:
        try:
            method = getattr(self._proxy, f"lookup_on_{operation}")
            result = int(method(block_ids))
        except Exception as exc:
            raise UCMProxyError(f"Proxy lookup_on_{operation} failed") from exc
        if not -1 <= result < len(block_ids):
            raise UCMProxyError(
                f"Proxy lookup_on_{operation} returned {result} "
                f"for {len(block_ids)} keys"
            )
        return result

    def lookup_on_prefix(self, block_ids: Sequence[bytes]) -> int:
        return self._scan("prefix", block_ids)

    def lookup_on_reverse(self, block_ids: Sequence[bytes]) -> int:
        return self._scan("reverse", block_ids)

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

    def commit(self, block_ids: Sequence[bytes]) -> None:
        """Publish keys after their writes complete, without rescanning descriptors."""
        if not block_ids:
            return
        try:
            self._wait("commit", self._proxy.commit(block_ids))
        except Exception as exc:
            if isinstance(exc, UCMProxyError):
                raise
            raise UCMProxyError("Proxy commit failed") from exc

    def enqueue(
        self, operation: Literal["load", "dump"], transfer: UCMProxyTransfer
    ) -> UCMProxyTask:
        """Submit unchanged descriptors; the returned task retains their lifetime."""
        try:
            handle = None
            if transfer.keys:
                handle = getattr(self._proxy, operation)(
                    transfer.keys,
                    transfer.ucm_block_offsets,
                    transfer.ptrs,
                    transfer.sizes,
                )
            return UCMProxyTask(operation, transfer, handle)
        except Exception as exc:
            if isinstance(exc, UCMProxyError):
                raise
            raise UCMProxyError(f"Proxy {operation} failed") from exc

    def wait(self, task: UCMProxyTask) -> None:
        try:
            self._wait(task.operation, task.handle)
        except Exception as exc:
            if isinstance(exc, UCMProxyError):
                raise
            raise UCMProxyError(f"Proxy {task.operation} failed") from exc

    def submit(
        self, operation: Literal["load", "dump"], transfer: UCMProxyTransfer
    ) -> None:
        """Submit and wait for a bulk transfer."""
        self.wait(self.enqueue(operation, transfer))
