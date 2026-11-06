# SPDX-License-Identifier: MIT
"""Connector-independent byte oracles for model-check dump/load verification.

The checks here only use the native KV tensors (shape, stride, storage) and
vLLM's block tables. They never consume addresses computed by the connector
under test, so a wrong layer/component/token offset that the connector applies
symmetrically to both dump and load cannot cancel out, and a layer the
connector forgets to transfer stays poisoned in the target block.

Everything in this module is importable without vLLM or UCM.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch

POISON_BYTE = 0xFF
# Patterns only use bytes in [0, 250], so a poisoned (0xff) byte that a load
# did not overwrite can never equal an expected byte.
PATTERN_MODULUS = 251
# A single arithmetic sequence modulo 251 repeats every 251 bytes, so swapping
# two equal-length pieces 251 bytes apart would be invisible. Adding one term
# per coarser scale pushes the repeat to 251**SCALES; within that span every
# shift changes at least one byte.
PATTERN_SCALES = 3
PATTERN_PERIOD = PATTERN_MODULUS**PATTERN_SCALES


def byte_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return a writable uint8 alias of ``tensor`` with a trailing byte axis.

    Works for arbitrary strides (permuted Ascend views, combined K/V slices)
    where ``tensor.view(torch.uint8)`` would require a unit last stride.
    """

    element_size = tensor.element_size()
    alias = torch.empty(0, dtype=torch.uint8, device=tensor.device)
    alias.set_(
        tensor.untyped_storage(),
        tensor.storage_offset() * element_size,
        (*tensor.shape, element_size),
        (*(stride * element_size for stride in tensor.stride()), 1),
    )
    return alias


def pattern_bytes(shape: tuple[int, ...], *labels: object) -> torch.Tensor:
    """Deterministic CPU uint8 pattern in [0, 250] that depends on byte position.

    Values are a sum of one arithmetic sequence per scale -- per byte, per 251
    bytes, per 251**2 bytes -- all modulo 251, with every start and step
    derived from ``labels``.

    Two labels collide only if every start and step matches. A shift of ``d``
    bytes changes at least one byte for every ``0 < d < PATTERN_PERIOD``: a
    single sequence would hide multiples of 251, but then the next scale's
    term changes by its own non-zero step, and so on. Shifts that are exact
    multiples of ``PATTERN_PERIOD`` (251**3, about 15.8 MB) remain invisible,
    which is why callers must keep one pattern's span below it; a KV block is
    orders of magnitude smaller.
    """

    digest = hashlib.sha256(repr(labels).encode("utf-8")).digest()
    start = int.from_bytes(digest[:8], "little") % PATTERN_MODULUS
    count = 1
    for size in shape:
        count *= int(size)
    index = torch.arange(count, dtype=torch.int64)
    values = torch.full_like(index, start)
    for scale in range(PATTERN_SCALES):
        chunk = digest[8 + 8 * scale : 16 + 8 * scale]
        step = 1 + int.from_bytes(chunk, "little") % (PATTERN_MODULUS - 1)
        values += (index // PATTERN_MODULUS**scale) * step
    return (values % PATTERN_MODULUS).to(torch.uint8).reshape(shape)


def fill_pattern(tensor: torch.Tensor, *labels: object) -> None:
    """Overwrite every byte of ``tensor`` with ``pattern_bytes(..., *labels)``."""

    if tensor.numel() == 0:
        return
    target = byte_view(tensor)
    target.copy_(pattern_bytes(tuple(target.shape), *labels).to(target.device))


def poison(tensor: torch.Tensor) -> None:
    """Overwrite every byte of ``tensor`` with ``POISON_BYTE``."""

    if tensor.numel() == 0:
        return
    byte_view(tensor).fill_(POISON_BYTE)


def snapshot_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Copy the bytes of ``tensor`` to CPU (NaN-safe comparison input)."""

    return byte_view(tensor).detach().cpu().clone()


def first_mismatch(actual: torch.Tensor, expected: torch.Tensor) -> int | None:
    """Return the flat index of the first differing byte, or None if equal."""

    if actual.shape != expected.shape:
        raise AssertionError(
            f"byte shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )
    differing = (actual.reshape(-1) != expected.reshape(-1)).nonzero()
    if differing.numel() == 0:
        return None
    return int(differing[0])


@dataclass
class _StorageRecord:
    base: int
    nbytes: int
    full: torch.Tensor
    before: torch.Tensor
    allowed: torch.Tensor


class StorageSnapshot:
    """Detect writes outside the regions a load was allowed to touch.

    The KV pool of a model check is a few blocks, so copying every backing
    storage before the load is cheap.
    """

    def __init__(self, tensors: Iterable[torch.Tensor]) -> None:
        self._records: dict[tuple[str, int], _StorageRecord] = {}
        for tensor in tensors:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if key in self._records or storage.nbytes() == 0:
                continue
            full = torch.empty(0, dtype=torch.uint8, device=tensor.device)
            full.set_(storage, 0, (storage.nbytes(),), (1,))
            self._records[key] = _StorageRecord(
                base=storage.data_ptr(),
                nbytes=storage.nbytes(),
                full=full,
                before=full.detach().cpu().clone(),
                allowed=torch.zeros(storage.nbytes(), dtype=torch.bool),
            )

    def _record_for(self, tensor: torch.Tensor) -> _StorageRecord:
        key = (str(tensor.device), tensor.untyped_storage().data_ptr())
        record = self._records.get(key)
        if record is None:
            raise ValueError("tensor storage was not part of the snapshot")
        return record

    def allow_tensor(self, tensor: torch.Tensor) -> None:
        """Allow every byte covered by a (possibly strided) tensor view."""

        if tensor.numel() == 0:
            return
        record = self._record_for(tensor)
        element_size = tensor.element_size()
        record.allowed.as_strided(
            (*tensor.shape, element_size),
            (*(stride * element_size for stride in tensor.stride()), 1),
            tensor.storage_offset() * element_size,
        ).fill_(True)

    def allow_range(self, ptr: int, size: int) -> None:
        """Allow ``size`` bytes at absolute address ``ptr``."""

        ptr, size = int(ptr), int(size)
        if size <= 0:
            return
        for record in self._records.values():
            if record.base <= ptr and ptr + size <= record.base + record.nbytes:
                start = ptr - record.base
                record.allowed[start : start + size] = True
                return
        raise AssertionError(
            f"transfer range is outside every registered KV storage: "
            f"ptr={ptr:#x}, size={size}"
        )

    def contains(self, ptr: int, size: int) -> bool:
        """Whether every byte of ``[ptr, ptr + size)`` is already allowed.

        Lets a caller treat an address the component under test reported as
        something to **check** against the allowed set, instead of adding it to
        that set. Returns False for a range that leaves the registered
        storages altogether.
        """

        ptr, size = int(ptr), int(size)
        if size <= 0:
            return True
        for record in self._records.values():
            if record.base <= ptr and ptr + size <= record.base + record.nbytes:
                start = ptr - record.base
                return bool(record.allowed[start : start + size].all())
        return False

    def stray_writes(self) -> list[tuple[int, int, int]]:
        """Return ``(storage_base, first_offset, changed_bytes)`` per storage."""

        stray = []
        for record in self._records.values():
            changed = record.full.detach().cpu() != record.before
            changed &= ~record.allowed
            count = int(changed.sum())
            if count:
                stray.append((record.base, int(changed.nonzero()[0]), count))
        return stray
