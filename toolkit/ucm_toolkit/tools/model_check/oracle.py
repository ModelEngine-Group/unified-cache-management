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

    Two labels collide only if both the start and the step of their
    arithmetic sequences are equal, and a shift by ``d`` bytes inside one
    pattern is visible unless ``d`` is a multiple of 251.
    """

    digest = hashlib.sha256(repr(labels).encode("utf-8")).digest()
    start = int.from_bytes(digest[:8], "little") % PATTERN_MODULUS
    step = 1 + int.from_bytes(digest[8:16], "little") % (PATTERN_MODULUS - 1)
    count = 1
    for size in shape:
        count *= int(size)
    values = (torch.arange(count, dtype=torch.int64) * step + start) % PATTERN_MODULUS
    return values.to(torch.uint8).reshape(shape)


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
