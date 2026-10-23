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
    commit publishes complete records; flush drains downstream commits.
    A Store with no backend performs local IO only. Dump metadata is per-record
    JSON with type (FA, WA, State) and absolute request chunk_index. Callers
    commit only after every required byte range has completed.
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
    def commit(self, ucm_chunk_ids: Sequence[bytes]) -> object | None: ...
    def abort(self, ucm_chunk_ids: Sequence[bytes]) -> None: ...
    def wait(self, task: object) -> None: ...
    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
