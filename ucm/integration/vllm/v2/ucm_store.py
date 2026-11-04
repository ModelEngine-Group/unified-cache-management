"""Local storage and optional backend-chain completion contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np

if TYPE_CHECKING:
    import torch

    KVCacheValue: TypeAlias = (
        torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
    )
else:
    KVCacheValue: TypeAlias = object


class UCMStore(Protocol):
    """Stores own backend lookup, promotion, writeback and buffer lifetimes.

    Only committed records are visible. wait(dump) releases source pointers;
    commit(True, ...) publishes complete records; commit(False, ...) discards
    staging. flush drains downstream commits.
    Submission errors retain no pointers for that submission. wait must finish
    or cancel IO before raising OSError/MemoryError (recoverable failures).
    If pointers may remain owned, raise a fatal RuntimeError instead. Contract
    errors use ValueError/TypeError and must never be reported as cache misses.
    A Store with no backend performs local IO only. Dump metadata is per-record
    JSON with type (FA, WA, State) and absolute request chunk_index. Callers
    commit(True) only after every required range succeeded; commit(False) only
    after all required writers have terminated IO. Discard never removes an
    existing published object or another attempt.
    """

    backend: UCMStore | None

    def lookup(self, ucm_chunk_ids: Sequence[bytes]) -> Sequence[bool]: ...

    # Index of the last hit before the first miss, -1 if the first id misses.
    # Stops at the first miss.
    def lookup_on_prefix(self, ucm_chunk_ids: Sequence[bytes]) -> int: ...

    # Scans from the end and returns the index of the first hit, -1 if none.
    # Stops at that hit.
    def lookup_on_reverse(self, ucm_chunk_ids: Sequence[bytes]) -> int: ...
    def load(
        self,
        ucm_chunk_ids: Sequence[bytes],
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
    ) -> object | None: ...
    def dump(
        self,
        ucm_chunk_ids: Sequence[bytes],
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
        metadata: Sequence[str],
        record_nbytes: np.ndarray,
    ) -> object | None: ...
    def commit(
        self, success: bool, staging_ids: Sequence[bytes], object_ids: Sequence[bytes]
    ) -> object | None: ...
    def wait(self, task: object) -> None: ...
    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
