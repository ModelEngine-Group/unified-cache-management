"""Request-position routing and the connector transfer/completion boundary."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .ucm_store import KVCacheValue, UCMStore


class UCMProxyError(RuntimeError):
    """Normalized error raised by the v2 Proxy adapter."""


class UCMProxy(Protocol):
    """chunk_indices[K] are absolute UCM record indices within each request.

    load/dump arrays are read-only [K,S] descriptors retained until wait.
    Dump record_nbytes[K] describes complete records, including unsent layers.
    """

    def lookup(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> Sequence[bool]: ...
    def lookup_on_prefix(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> int: ...
    def lookup_on_reverse(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> int: ...
    def load(
        self,
        ucm_chunk_ids: Sequence[bytes],
        chunk_indices: np.ndarray,
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
    ) -> object | None: ...
    def dump(
        self,
        ucm_chunk_ids: Sequence[bytes],
        chunk_indices: np.ndarray,
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
        metadata: Sequence[str],
        record_nbytes: np.ndarray,
    ) -> object | None: ...
    def commit(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> object | None: ...
    def abort(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> None: ...
    def wait(self, task: object) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class UCMProxyWaiter(Protocol):
    def wait(self, task: object) -> None: ...


@runtime_checkable
class UCMProxyTensorRegistration(Protocol):
    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None: ...


@dataclass(frozen=True)
class UCMProxyTransfer:
    """Unique ucm_chunk_ids [K] and byte descriptors [K,S]; arrays may be read-only.

    Offsets address the complete UCM chunk. Backends retain arrays through
    completion and never mutate them or require contiguous input.
    """

    ucm_chunk_ids: tuple[bytes, ...]
    chunk_indices: np.ndarray
    ptrs: np.ndarray
    sizes: np.ndarray
    ucm_chunk_offsets: np.ndarray
    # Per-ucm_chunk_id JSON for dump (see UCMProxy.dump); load transfers leave it empty.
    metadata: tuple[str, ...] = ()
    record_nbytes: np.ndarray | None = None


@dataclass(frozen=True)
class UCMProxyTask:
    """Keep transfer arrays alive until the backend task completes."""

    operation: str
    transfer: UCMProxyTransfer
    handle: object | None


def normalize_ucm_chunk_ids(ucm_chunk_ids: Sequence[bytes]) -> tuple[bytes, ...]:
    # Native transfers already carry bytes: validate in one pass and
    # preserve identity for subsequent layer callbacks.
    if type(ucm_chunk_ids) is tuple and all(
        type(ucm_chunk_id) is bytes and len(ucm_chunk_id) == 16
        for ucm_chunk_id in ucm_chunk_ids
    ):
        return ucm_chunk_ids
    ucm_chunk_ids = tuple(bytes(ucm_chunk_id) for ucm_chunk_id in ucm_chunk_ids)
    invalid_rows = [
        row for row, ucm_chunk_id in enumerate(ucm_chunk_ids) if len(ucm_chunk_id) != 16
    ]
    if invalid_rows:
        raise ValueError(
            f"UCM chunk IDs must be 16 bytes; invalid indexes={invalid_rows}"
        )
    return ucm_chunk_ids


class UCMProxyAdapter:
    """Forward native transfers and handle completion."""

    def __init__(self, proxy: UCMProxy) -> None:
        self._proxy = proxy

    def lookup(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> tuple[bool, ...]:
        ucm_chunk_ids = normalize_ucm_chunk_ids(ucm_chunk_ids)
        try:
            hits = tuple(
                bool(is_hit)
                for is_hit in self._proxy.lookup(ucm_chunk_ids, chunk_indices)
            )
        except Exception as exc:
            raise UCMProxyError("Proxy lookup failed") from exc
        if len(hits) != len(ucm_chunk_ids):
            raise UCMProxyError(
                f"Proxy lookup returned {len(hits)} results for {len(ucm_chunk_ids)} ucm_chunk_ids"
            )
        return hits

    def _scan(
        self, operation: str, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> int:
        ucm_chunk_ids = normalize_ucm_chunk_ids(ucm_chunk_ids)
        try:
            scan_method = (
                self._proxy.lookup_on_prefix
                if operation == "prefix"
                else self._proxy.lookup_on_reverse
            )
            hit_row = int(scan_method(ucm_chunk_ids, chunk_indices))
        except Exception as exc:
            raise UCMProxyError(f"Proxy lookup_on_{operation} failed") from exc
        if not -1 <= hit_row < len(ucm_chunk_ids):
            raise UCMProxyError(
                f"Proxy lookup_on_{operation} returned {hit_row} "
                f"for {len(ucm_chunk_ids)} ucm_chunk_ids"
            )
        return hit_row

    def lookup_on_prefix(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> int:
        return self._scan("prefix", ucm_chunk_ids, chunk_indices)

    def lookup_on_reverse(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> int:
        return self._scan("reverse", ucm_chunk_ids, chunk_indices)

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

    def commit(self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray) -> None:
        """Publish ucm_chunk_ids after their writes complete, without rescanning descriptors."""
        if not ucm_chunk_ids:
            return
        try:
            self._wait("commit", self._proxy.commit(ucm_chunk_ids, chunk_indices))
        except Exception as exc:
            if isinstance(exc, UCMProxyError):
                raise
            raise UCMProxyError("Proxy commit failed") from exc

    def enqueue(self, operation: str, transfer: UCMProxyTransfer) -> UCMProxyTask:
        """Submit unchanged descriptors; the returned task retains their lifetime."""
        if operation not in ("load", "dump"):
            raise ValueError(f"Unknown transfer operation: {operation}")
        try:
            validate_chunk_indices(transfer.ucm_chunk_ids, transfer.chunk_indices)
            shape = transfer.ptrs.shape
            if (
                len(shape) != 2
                or shape[0] != len(transfer.ucm_chunk_ids)
                or transfer.sizes.shape != shape
                or transfer.ucm_chunk_offsets.shape != shape
            ):
                raise ValueError("Transfer arrays must have matching [K,S] shapes")
            if operation == "dump":
                lengths = transfer.record_nbytes
                if (
                    lengths is None
                    or lengths.shape != (len(transfer.ucm_chunk_ids),)
                    or lengths.dtype.kind not in "iu"
                    or np.any(lengths < 0)
                    or len(transfer.metadata) != len(transfer.ucm_chunk_ids)
                ):
                    raise ValueError(
                        "Dump requires complete record_nbytes[K] and metadata[K]"
                    )
            handle = None
            if transfer.ucm_chunk_ids and operation == "dump":
                handle = self._proxy.dump(
                    transfer.ucm_chunk_ids,
                    transfer.chunk_indices,
                    transfer.ucm_chunk_offsets,
                    transfer.ptrs,
                    transfer.sizes,
                    transfer.metadata,
                    transfer.record_nbytes,
                )
            elif transfer.ucm_chunk_ids:
                handle = self._proxy.load(
                    transfer.ucm_chunk_ids,
                    transfer.chunk_indices,
                    transfer.ucm_chunk_offsets,
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

    def submit(self, operation: str, transfer: UCMProxyTransfer) -> None:
        """Submit and wait for a bulk transfer."""
        self.wait(self.enqueue(operation, transfer))

    def abort(self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray) -> None:
        try:
            self._proxy.abort(ucm_chunk_ids, chunk_indices)
        except Exception as exc:
            raise UCMProxyError("Proxy abort failed") from exc

    def flush(self) -> None:
        self._proxy.flush()

    def close(self) -> None:
        self._proxy.close()


class PipelineProxy:
    """Forward to the top Store; the Store owns its backend chain."""

    def __init__(self, store: UCMStore) -> None:
        self.store = store

    def lookup(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> tuple[bool, ...]:
        validate_chunk_indices(ucm_chunk_ids, chunk_indices)
        hits = tuple(self.store.lookup(ucm_chunk_ids))
        if len(hits) != len(ucm_chunk_ids):
            raise ValueError("Store lookup must return one result per key")
        return hits

    def lookup_on_prefix(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> int:
        validate_chunk_indices(ucm_chunk_ids, chunk_indices)
        return int(self.store.lookup_on_prefix(ucm_chunk_ids))

    def lookup_on_reverse(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> int:
        validate_chunk_indices(ucm_chunk_ids, chunk_indices)
        return int(self.store.lookup_on_reverse(ucm_chunk_ids))

    def load(
        self,
        ucm_chunk_ids: Sequence[bytes],
        chunk_indices: np.ndarray,
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
    ) -> object | None:
        validate_chunk_indices(ucm_chunk_ids, chunk_indices)
        return self.store.load(ucm_chunk_ids, offsets, ptrs, sizes)

    def dump(
        self,
        ucm_chunk_ids: Sequence[bytes],
        chunk_indices: np.ndarray,
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
        metadata: Sequence[str],
        record_nbytes: np.ndarray,
    ) -> object | None:
        validate_chunk_indices(ucm_chunk_ids, chunk_indices)
        return self.store.dump(
            ucm_chunk_ids, offsets, ptrs, sizes, metadata, record_nbytes
        )

    def commit(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> object | None:
        validate_chunk_indices(ucm_chunk_ids, chunk_indices)
        return self.store.commit(ucm_chunk_ids)

    def abort(self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray) -> None:
        validate_chunk_indices(ucm_chunk_ids, chunk_indices)
        self.store.abort(ucm_chunk_ids)

    def wait(self, task: object) -> None:
        if not isinstance(self.store, UCMProxyWaiter):
            raise UCMProxyError("Store does not provide wait(task)")
        self.store.wait(task)

    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        if isinstance(self.store, UCMProxyTensorRegistration):
            self.store.register_tensors(kv_caches)

    def flush(self) -> None:
        self.store.flush()

    def close(self) -> None:
        self.store.close()


def validate_chunk_indices(
    ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
) -> None:
    if (
        chunk_indices.shape != (len(ucm_chunk_ids),)
        or chunk_indices.dtype.kind not in "iu"
        or np.any(chunk_indices < 0)
    ):
        raise ValueError("chunk_indices must be a nonnegative integer array [K]")


@dataclass(frozen=True)
class StoreTasks:
    # Sliced descriptors must survive even if a Store only retains raw addresses.
    parts: tuple[tuple[UCMStore, object, tuple[object, ...]], ...]


class ParallelProxy:
    """Route whole records to independent Stores by request token position."""

    def __init__(
        self,
        stores: Sequence[UCMStore],
        token_ends: Sequence[int | None],
        record_token_span: int,
    ) -> None:
        if record_token_span <= 0 or not stores or len(stores) != len(token_ends):
            raise ValueError(
                "Routes require Stores, matching ends and positive record span"
            )
        if token_ends[-1] is not None:
            raise ValueError("The final route must have token_end=null")
        route_token_ends = tuple(token_ends[:-1])
        previous_token_end = 0
        for token_end in route_token_ends:
            if (
                not isinstance(token_end, int)
                or token_end <= previous_token_end
                or token_end % record_token_span
            ):
                raise ValueError(
                    "Route ends must increase and align with record_token_span"
                )
            previous_token_end = token_end
        if len({id(store) for store in stores}) != len(stores):
            raise ValueError("Parallel routes require distinct Store instances")
        if any(store.backend is not None for store in stores):
            raise ValueError("Parallel Stores must have backend=None")
        self.stores = tuple(stores)
        self.chunk_ends = np.array(
            [token_end // record_token_span for token_end in route_token_ends],
            dtype=np.uint64,
        )

    def _store_indices(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> np.ndarray:
        """Index of the Store each row is routed to."""
        validate_chunk_indices(ucm_chunk_ids, chunk_indices)
        return np.searchsorted(
            self.chunk_ends, chunk_indices.astype(np.uint64, copy=False), side="right"
        )

    def _rows_by_store(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> Iterator[tuple[UCMStore, np.ndarray]]:
        store_indices = self._store_indices(ucm_chunk_ids, chunk_indices)
        for store_index, store in enumerate(self.stores):
            rows = np.flatnonzero(store_indices == store_index)
            if len(rows):
                yield store, rows

    def lookup(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> tuple[bool, ...]:
        hits = [False] * len(ucm_chunk_ids)
        for store, rows in self._rows_by_store(ucm_chunk_ids, chunk_indices):
            store_ucm_chunk_ids = tuple(ucm_chunk_ids[row] for row in rows)
            for row, is_hit in zip(
                rows, store.lookup(store_ucm_chunk_ids), strict=True
            ):
                hits[row] = bool(is_hit)
        return tuple(hits)

    def _scan_runs(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> tuple[tuple[UCMStore, int, tuple[bytes, ...]], ...]:
        """(Store, first row, ids) of each run of consecutive rows routed to
        the same Store, in row order.

        Scans keep the order of the ids, so a Store can have several runs
        when chunk indices are not increasing.
        """
        store_indices = self._store_indices(ucm_chunk_ids, chunk_indices)
        run_bounds = [
            0,
            *(np.flatnonzero(np.diff(store_indices)) + 1).tolist(),
            len(ucm_chunk_ids),
        ]
        return tuple(
            (
                self.stores[int(store_indices[run_start])],
                run_start,
                tuple(ucm_chunk_ids[run_start:run_end]),
            )
            for run_start, run_end in zip(run_bounds[:-1], run_bounds[1:])
            if run_end > run_start
        )

    def lookup_on_prefix(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> int:
        """Scan the runs in order and stop at the first miss."""
        last_hit_row = -1
        for store, first_row, run_ids in self._scan_runs(ucm_chunk_ids, chunk_indices):
            run_last_hit = int(store.lookup_on_prefix(run_ids))
            last_hit_row = first_row + run_last_hit
            if run_last_hit < len(run_ids) - 1:
                break
        return last_hit_row

    def lookup_on_reverse(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> int:
        """Scan the runs from the last one and stop at the first hit."""
        for store, first_row, run_ids in reversed(
            self._scan_runs(ucm_chunk_ids, chunk_indices)
        ):
            run_latest_hit = int(store.lookup_on_reverse(run_ids))
            if run_latest_hit >= 0:
                return first_row + run_latest_hit
        return -1

    def _submit(
        self,
        operation: Literal["load", "dump", "commit"],
        ucm_chunk_ids: Sequence[bytes],
        chunk_indices: np.ndarray,
        *descriptors: np.ndarray | Sequence[str],
    ) -> StoreTasks | None:
        store_tasks = []
        try:
            for store, rows in self._rows_by_store(ucm_chunk_ids, chunk_indices):
                store_ucm_chunk_ids = tuple(ucm_chunk_ids[row] for row in rows)
                arguments = (store_ucm_chunk_ids,) + tuple(
                    (
                        descriptor[rows]
                        if isinstance(descriptor, np.ndarray)
                        else tuple(descriptor[row] for row in rows)
                    )
                    for descriptor in descriptors
                )
                store_method = {
                    "load": store.load,
                    "dump": store.dump,
                    "commit": store.commit,
                }[operation]
                task = store_method(*arguments)
                if task is not None:
                    store_tasks.append((store, task, arguments))
        except Exception:
            # Earlier routes may already own source pointers; drain all of them.
            try:
                self.wait(StoreTasks(tuple(store_tasks)))
            except Exception:
                pass
            raise
        return StoreTasks(tuple(store_tasks)) if store_tasks else None

    def load(
        self,
        ucm_chunk_ids: Sequence[bytes],
        chunk_indices: np.ndarray,
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
    ) -> object | None:
        return self._submit("load", ucm_chunk_ids, chunk_indices, offsets, ptrs, sizes)

    def dump(
        self,
        ucm_chunk_ids: Sequence[bytes],
        chunk_indices: np.ndarray,
        offsets: np.ndarray,
        ptrs: np.ndarray,
        sizes: np.ndarray,
        metadata: Sequence[str],
        record_nbytes: np.ndarray,
    ) -> object | None:
        return self._submit(
            "dump",
            ucm_chunk_ids,
            chunk_indices,
            offsets,
            ptrs,
            sizes,
            metadata,
            record_nbytes,
        )

    def commit(
        self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray
    ) -> object | None:
        return self._submit("commit", ucm_chunk_ids, chunk_indices)

    def abort(self, ucm_chunk_ids: Sequence[bytes], chunk_indices: np.ndarray) -> None:
        error = None
        for store, rows in self._rows_by_store(ucm_chunk_ids, chunk_indices):
            try:
                store.abort(tuple(ucm_chunk_ids[row] for row in rows))
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def wait(self, task: StoreTasks) -> None:
        error = None
        for store, handle, _arguments in task.parts:
            try:
                store.wait(handle)
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def register_tensors(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        for store in self.stores:
            store.register_tensors(kv_caches)

    def flush(self) -> None:
        error = None
        for store in self.stores:
            try:
                store.flush()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def close(self) -> None:
        error = None
        for store in self.stores:
            try:
                store.close()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error
