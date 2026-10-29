"""UCM implementation of SGLang's direct L1 external-cache linker.

This module owns the SGLang side only: it materializes device-pool views,
converts ``PoolTransfer`` objects to neutral UCM records, and drives SGLang's
layer-ready and completion queues. Store setup and I/O stay in the adapter.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

import numpy as np
import torch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)
from sglang.srt.runtime_context import get_model, get_parallel
from sglang.srt.utils import freeze_gc, get_device_module

from ucm.integration.sglang.ucm_direct_store_adapter import (
    UcmDirectStoreAdapter,
    UcmLayerTransfer,
    UcmPoolMetadata,
)

logger = logging.getLogger(__name__)
device_module = get_device_module()
_LOOKUP_TIMING_LOG_MS = 10.0
_LOOKUP_PROGRESS_MAX_REQUESTS = 128
_LOAD_TIMING_LOG_MS = 20.0
_OFFLOAD_TIMING_LOG_MS = 100.0
_OFFLOAD_POLL_INTERVAL_S = 0.005
_SLOW_LOG_FIRST = 3
_SLOW_LOG_EVERY = 50


def _pool_group_name(
    source: str, pools: tuple[str, ...], shape_index: int, layers: tuple[int, ...]
) -> str:
    """Build a bounded, deterministic Store group name."""
    base = "__".join((source, *pools))
    if len(base) > 64:
        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
        base = f"{base[:46]}__h{digest}"
    if not layers:
        raise ValueError("A Store group must contain at least one layer")
    # Keep layer membership in the identity without expanding it in the name.
    layer_key = ",".join(map(str, layers))
    digest = hashlib.sha256(layer_key.encode("ascii")).hexdigest()[:16]
    return f"{base}__s{shape_index}__h{digest}"


@dataclass(frozen=True)
class _PoolBundle:
    """Physical pools persisted together as one UCM block payload.

    A bundle is deliberately stricter than ``indices_from_pool`` alone: its
    entries must also cover the same logical-layer set.  DSA's KV and INDEXER
    entries satisfy this, so one page/layer becomes one composite Store block.
    """

    name: str
    source_pool: PoolName
    layers: tuple[int, ...]
    entries: tuple[DevicePoolEntry, ...]


@dataclass(frozen=True)
class _LookupProgress:
    page_keys: tuple[str, ...]
    transfer_keys: tuple[tuple[object, ...], ...]
    hit_pages: int


@dataclass
class _LookupTiming:
    calls: int = 0
    slow_calls: int = 0
    total_ms: float = 0.0
    store_total_ms: float = 0.0
    max_ms: float = 0.0
    max_store_ms: float = 0.0


@dataclass
class _ForwardWaitTiming:
    calls: int = 0
    slow_calls: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    max_layer: int = -1


@dataclass(frozen=True)
class _LayerPointerLayout:
    """Fixed buffer addresses, strides and fragment schema for one pool layer."""

    bases: np.ndarray
    strides: np.ndarray
    schema: tuple[tuple[int, int], ...]


@dataclass
class _TransferBuildTimings:
    schema_ms: float = 0.0
    locations_ms: float = 0.0
    layer_meta_ms: float = 0.0
    # Kept in the log for before/after comparison; array I/O no longer
    # converts address rows to Python lists.
    row_convert_ms: float = 0.0
    address_assemble_ms: float = 0.0
    layer_meta_calls: int = 0


@dataclass
class _PendingOffload:
    sequence: int
    tasks: list[Any]
    transfers: list[UcmLayerTransfer]
    ready_events: list[object]
    queued_at: float
    submitted_at: float
    transfer_pages: int
    store_tasks: int = 0
    queue_backlog: int = 0
    queue_ms: float = 0.0
    ready_event_ms: float = 0.0
    transfer_build_ms: float = 0.0
    store_submit_ms: float = 0.0
    result: bool | None = None
    finished_at: float | None = None
    check_calls: int = 0
    check_ms: float = 0.0
    wait_ms: float = 0.0


class UcmDirectLinker(UnifiedCacheLinker):
    """Link SGLang's L1 cache pools to a UCM Store adapter.

    ``adapter`` is injectable for focused tests. In a normal process it is
    created from the UCM configuration.
    """

    def __init__(
        self,
        server_args: Any,
        params: CacheInitParams,
        *,
        components,
        extra_config: dict[str, Any] | None = None,
        forward_stream: Any = None,
        adapter: UcmDirectStoreAdapter | None = None,
    ):
        del server_args
        self._linker_extra_config = deepcopy(extra_config or {})
        self._forward_stream = forward_stream
        self.page_size = params.page_size
        self._kv_allocator = params.token_to_kv_pool_allocator
        kvcache = self._kv_allocator.get_kvcache()
        self._kv_cache = kvcache
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers
        self._bundles = self._build_pool_bundles()
        self._layer_pointer_layouts = self._build_layer_pointer_layouts()
        self._closed = False
        self._gc_frozen = False
        self._pending_loads: dict[str, list[PoolTransfer]] = {}
        self._pending_lock = threading.Lock()
        self._load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object, float] | None
        ] = Queue()
        self._completed_loads: Queue[list[str]] = Queue()
        self._requires_load_finish = hasattr(
            UnifiedCacheLinker, "finish_layer_wise_loading"
        )
        self._load_results: dict[int, Future[bool]] = {}
        self._offload_queue: Queue[
            tuple[list[PoolTransfer], list[object], float] | None
        ] = Queue()
        self._offload_results: Queue[bool] = Queue()
        self._lookup_progress: OrderedDict[str, _LookupProgress] = OrderedDict()
        self._lookup_progress_lock = threading.Lock()
        self._timing_log_lock = threading.Lock()
        self._lookup_timings: OrderedDict[str, _LookupTiming] = OrderedDict()
        self._forward_wait_timings: dict[int, _ForwardWaitTiming] = {}
        self._slow_timings: dict[str, tuple[int, float]] = {}
        self.stats = {
            "lookup": 0,
            "lookup_skipped": 0,
            "lookup_resumed": 0,
            "load": 0,
            "offload": 0,
        }

        self._tp_rank = 0
        parallel = get_parallel()
        self._tp_size = parallel.tp_size
        self._dp_rank = int(getattr(parallel, "dp_rank", None) or 0)
        self._attn_dp_rank = int(getattr(parallel, "attn_dp_rank", None) or 0)
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._tp_rank = torch.distributed.get_rank(group=tp_group)
            self._tp_size = torch.distributed.get_world_size(group=tp_group)
        self._logger = logger if self._tp_rank == 0 else logging.Logger(__name__)
        if self._tp_rank != 0:
            self._logger.disabled = True
        self._lookup_owner = True
        self._lookup_dedup_active = False
        self._lookup_logger = self._logger
        self._offload_owner = not self.pool_group.rank_replicated or self._tp_rank == 0
        self._log_pool_topology()
        try:
            self.adapter = adapter or self._create_adapter(params)
            self.adapter.register_device_pools(self._pool_metadata())
            self._select_lookup_owner(tp_group, params.attn_cp_size)
            if self._lookup_dedup_active and self._lookup_owner:
                self._lookup_logger = logger
            self._load_prefetch_layers = getattr(
                self.adapter, "load_prefetch_layers", 2
            )
            if self._tp_rank == 0:
                self._logger.info(
                    "UCM linker load prefetch layers=%d", self._load_prefetch_layers
                )
        except Exception:
            self._logger.exception(
                "UCM linker initialization failed: layers=%d tp=%d/%d replicated=%s.",
                self.num_layers,
                self._tp_rank,
                self._tp_size,
                self.pool_group.rank_replicated,
            )
            raise

        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        if self._tp_rank == 0 and self._requires_load_finish:
            self._observe_layer_waits()
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )

        self._load_thread = threading.Thread(
            target=self._load_worker, daemon=True, name=f"ucm-load-tp{self._tp_rank}"
        )
        self._offload_thread = threading.Thread(
            target=self._offload_worker,
            daemon=True,
            name=f"ucm-offload-tp{self._tp_rank}",
        )
        self._load_thread.start()
        self._offload_thread.start()

    def _should_log_timing(
        self, kind: str, duration_ms: float, threshold: float
    ) -> bool:
        if self._tp_rank != 0 or duration_ms < threshold:
            return False
        with self._timing_log_lock:
            count, maximum = self._slow_timings.get(kind, (0, 0.0))
            count += 1
            self._slow_timings[kind] = (count, max(maximum, duration_ms))
            return (
                count <= _SLOW_LOG_FIRST
                or count % _SLOW_LOG_EVERY == 0
                or duration_ms >= 2 * maximum
            )

    def _log_lookup_summary(self, rid: str, timing: _LookupTiming) -> None:
        if timing.slow_calls or timing.calls > 1:
            self._lookup_logger.info(
                "UCM linker lookup summary dp_rank=%d tp_rank=%d rid=%s calls=%d "
                "slow_calls=%d total_ms=%.3f store_total_ms=%.3f "
                "max_ms=%.3f max_store_ms=%.3f",
                self._dp_rank,
                self._tp_rank,
                rid,
                timing.calls,
                timing.slow_calls,
                timing.total_ms,
                timing.store_total_ms,
                timing.max_ms,
                timing.max_store_ms,
            )

    def _record_lookup_timing(
        self, rid: str, total_ms: float, store_ms: float
    ) -> tuple[bool, int, int]:
        if self._lookup_logger.disabled:
            return False, 0, 0
        evicted = None
        with self._timing_log_lock:
            timing = self._lookup_timings.setdefault(rid, _LookupTiming())
            previous_max_ms = timing.max_ms
            timing.calls += 1
            timing.total_ms += total_ms
            timing.store_total_ms += store_ms
            timing.max_ms = max(timing.max_ms, total_ms)
            timing.max_store_ms = max(timing.max_store_ms, store_ms)
            if total_ms >= _LOOKUP_TIMING_LOG_MS:
                timing.slow_calls += 1
            self._lookup_timings.move_to_end(rid)
            if len(self._lookup_timings) > _LOOKUP_PROGRESS_MAX_REQUESTS:
                evicted = self._lookup_timings.popitem(last=False)
            calls, slow_calls = timing.calls, timing.slow_calls
        if evicted is not None:
            self._log_lookup_summary(*evicted)
        return (
            total_ms >= _LOOKUP_TIMING_LOG_MS
            and (
                slow_calls <= _SLOW_LOG_FIRST
                or slow_calls % _SLOW_LOG_EVERY == 0
                or total_ms >= 2 * previous_max_ms
            ),
            calls,
            slow_calls,
        )

    def _finish_lookup_timing(self, rid: str) -> None:
        with self._timing_log_lock:
            timing = self._lookup_timings.pop(rid, None)
        if timing is not None:
            self._log_lookup_summary(rid, timing)

    def _slow_timing_snapshot(self, kind: str) -> tuple[int, float]:
        with self._timing_log_lock:
            return self._slow_timings.get(kind, (0, 0.0))

    def _observe_layer_waits(self) -> None:
        counter = self.layer_done_counter
        original_wait = counter.wait_until

        def observed_wait(layer: int) -> None:
            index = counter.consumer_index
            if index < 0:
                return original_wait(layer)
            started = time.perf_counter()
            try:
                return original_wait(layer)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000
                with self._timing_log_lock:
                    timing = self._forward_wait_timings.setdefault(
                        index, _ForwardWaitTiming()
                    )
                    timing.calls += 1
                    timing.slow_calls += int(elapsed_ms >= _LOAD_TIMING_LOG_MS)
                    timing.total_ms += elapsed_ms
                    if elapsed_ms > timing.max_ms:
                        timing.max_ms = elapsed_ms
                        timing.max_layer = layer

        counter.wait_until = observed_wait

    def _select_lookup_owner(self, tp_group: Any, attn_cp_size: int) -> None:
        """Elect one reader per shared Cache within this attention-TP group."""
        if (
            not self.pool_group.rank_replicated
            or attn_cp_size != 1
            or tp_group is None
            or self._tp_size <= 1
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return

        # A shared Cache is a host-local POSIX shm object. Identify both the
        # host kernel and the /dev/shm mount so isolated containers do not get
        # grouped together. Unknown identities keep their own lookup.
        config = getattr(self.adapter, "_store_config", {})
        shared_cache = (
            getattr(self.adapter, "_connector_name", None) == "UcmPipelineStore"
            and str(config.get("store_pipeline", "")).startswith("Cache|")
            and config.get("share_buffer_enable", True) is True
        )
        store_signatures = getattr(self.adapter, "_lookup_store_signatures", {})
        cache_identity = None
        if shared_cache and store_signatures:
            try:
                with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as f:
                    boot_id = f.read().strip()
                shm = os.stat("/dev/shm")
                if boot_id:
                    cache_identity = (
                        boot_id,
                        shm.st_dev,
                        shm.st_ino,
                        tuple(sorted(store_signatures.items())),
                    )
            except OSError:
                pass
        identities: list[Any] = [None] * self._tp_size
        torch.distributed.all_gather_object(identities, cache_identity, group=tp_group)
        if cache_identity is not None:
            self._lookup_dedup_active = True
            self._lookup_owner = self._tp_rank == identities.index(cache_identity)
        if self._lookup_owner:
            (logger if self._lookup_dedup_active else self._logger).info(
                "UCM linker lookup owner dp_rank=%d tp_rank=%d tp_size=%d "
                "shared_cache_identified=%s",
                self._dp_rank,
                self._tp_rank,
                self._tp_size,
                cache_identity is not None,
            )

    def _create_adapter(self, params: CacheInitParams) -> UcmDirectStoreAdapter:
        identity = {
            "model": getattr(get_model(), "model_path", None),
            "tp_rank": self._tp_rank,
            "tp_size": self._tp_size,
            "dp_rank": self._dp_rank,
            "attn_dp_rank": self._attn_dp_rank,
            "pp_rank": params.pp_rank,
            "pp_size": params.pp_size,
            "attn_cp_rank": params.attn_cp_rank,
            "attn_cp_size": params.attn_cp_size,
            "rank_replicated": self.pool_group.rank_replicated,
            "device_id": int(device_module.current_device()),
        }
        config = dict(self._linker_extra_config)
        config.setdefault("config_file", os.environ.get("UNIFIEDCACHE_CONFIG_FILE"))
        return UcmDirectStoreAdapter.from_config(config, identity)

    def _pool_metadata(self) -> list[UcmPoolMetadata]:
        metadata = []
        for bundle in self._bundles:
            for entry in bundle.entries:
                if bundle.source_pool == PoolName.KV:
                    tokens = getattr(self._kv_allocator, "size_full", None)
                    if tokens is None:
                        tokens = getattr(self._kv_allocator, "size", None)
                    if tokens is None:
                        tokens = getattr(self._kv_cache, "size", None)
                elif bundle.source_pool == PoolName.SWA:
                    tokens = getattr(self._kv_allocator, "size_swa", None)
                    if tokens is None:
                        tokens = getattr(self._kv_cache, "swa_size", None)
                else:
                    tokens = None
                if tokens is not None and int(tokens) > 0:
                    capacity_pages = (
                        int(tokens) + entry.page_size - 1
                    ) // entry.page_size
                else:
                    # Pool rows are already in page units when _row_span is 1.
                    capacity_pages = int(entry._row_count) // int(entry._row_span)
                layer_schemas = tuple(
                    (layer, self._entry_layer_schema(entry, layer))
                    for layer in bundle.layers
                )
                metadata.append(
                    UcmPoolMetadata(
                        name=f"{bundle.name}:{entry.name}",
                        pool_name=str(entry.name),
                        source_pool=str(bundle.source_pool),
                        store_group=bundle.name,
                        page_size=entry.page_size,
                        capacity_pages=capacity_pages,
                        layer_schemas=layer_schemas,
                        device_buffers=tuple(
                            (int(buffer.data_ptr()), int(buffer.nbytes))
                            for buffer in entry.kv_buffer
                            if buffer.data_ptr() and buffer.nbytes
                        ),
                    )
                )
        return metadata

    def _build_pool_bundles(
        self,
    ) -> tuple[_PoolBundle, ...]:
        """Group entries that can share one page/layer Store block schema."""
        grouped: dict[
            tuple[PoolName, tuple[tuple[int, int | tuple[int, ...]], ...]],
            list[DevicePoolEntry],
        ] = {}
        for entry in self.pools.values():
            mapping = tuple(
                (layer, mapped if isinstance(mapped, int) else tuple(mapped))
                for layer, mapped in sorted(entry.layer_mapping.items())
            )
            layers = tuple(layer for layer, _ in mapping)
            if not layers:
                continue
            if any(layer < 0 or layer >= self.num_layers for layer in layers):
                raise ValueError(
                    f"UCM pool {entry.name} has out-of-range layers {layers}."
                )
            grouped.setdefault((entry.indices_from_pool, mapping), []).append(entry)

        bundles = []
        for (source_pool, mapping), entries in grouped.items():
            layers = tuple(layer for layer, _ in mapping)
            ordered = tuple(sorted(entries, key=lambda entry: str(entry.name)))
            by_schema: dict[tuple[int, ...], list[int]] = {}
            for layer in layers:
                schema = tuple(
                    size
                    for entry in ordered
                    for size, _ in self._entry_layer_schema(entry, layer)
                )
                if not schema:
                    raise ValueError(
                        f"UCM pool group {source_pool} has no fragments at layer {layer}."
                    )
                by_schema.setdefault(schema, []).append(layer)
            for index, (_, schema_layers) in enumerate(sorted(by_schema.items())):
                bundles.append(
                    _PoolBundle(
                        _pool_group_name(
                            str(source_pool),
                            tuple(str(entry.name) for entry in ordered),
                            index,
                            tuple(schema_layers),
                        ),
                        source_pool,
                        tuple(schema_layers),
                        ordered,
                    )
                )
        return tuple(bundles)

    def _log_pool_topology(self) -> None:
        if self._tp_rank != 0:
            return
        self._logger.info(
            "UCM device-pool topology: layers=%d page_size=%d pools=%d bundles=%d "
            "tp=%d/%d replicated=%s.",
            self.num_layers,
            self.page_size,
            len(self.pools),
            len(self._bundles),
            self._tp_rank,
            self._tp_size,
            self.pool_group.rank_replicated,
        )
        for bundle in self._bundles:
            schema = tuple(
                size
                for layer in bundle.layers
                for entry in sorted(bundle.entries, key=lambda item: str(item.name))
                for size, _ in self._entry_layer_schema(entry, layer)
            )
            self._logger.info(
                "UCM pool bundle=%s source=%s pools=%s "
                "page_block_tensors=%d page_block_bytes=%d.",
                bundle.name,
                bundle.source_pool,
                [str(entry.name) for entry in bundle.entries],
                len(schema),
                sum(schema),
            )

    @staticmethod
    def _entry_layer_schema(
        entry: DevicePoolEntry, layer: int
    ) -> tuple[tuple[int, int], ...]:
        mapped = entry.layer_mapping.get(layer)
        if mapped is None:
            return ()
        buffer_indices = (mapped,) if isinstance(mapped, int) else tuple(mapped)
        return tuple(
            (int(component[index][2]), int(offsets[index]))
            for component, offsets in zip(entry.buffer_meta, entry._component_offsets)
            for index in buffer_indices
        )

    def _build_layer_pointer_layouts(
        self,
    ) -> dict[tuple[PoolName, int], _LayerPointerLayout]:
        layouts: dict[tuple[PoolName, int], _LayerPointerLayout] = {}
        for bundle in self._bundles:
            for entry in bundle.entries:
                for layer in bundle.layers:
                    key = (entry.name, layer)
                    if key in layouts:
                        continue
                    mapped = entry.layer_mapping[layer]
                    buffer_indices = (
                        (mapped,) if isinstance(mapped, int) else tuple(mapped)
                    )
                    # Match DevicePoolEntry.get_prepared_layer_range_meta's
                    # component-then-buffer order exactly.
                    items = [
                        (*component[index], offsets[index])
                        for component, offsets in zip(
                            entry.buffer_meta, entry._component_offsets
                        )
                        for index in buffer_indices
                    ]
                    bases = np.asarray([item[0] for item in items], dtype=np.uint64)
                    strides = np.asarray([item[1] for item in items], dtype=np.uint64)
                    bases.setflags(write=False)
                    strides.setflags(write=False)
                    layouts[key] = _LayerPointerLayout(
                        bases=bases,
                        strides=strides,
                        schema=tuple((int(item[2]), int(item[3])) for item in items),
                    )
        return layouts

    def _entry_layer_rows(
        self,
        entry: DevicePoolEntry,
        transfer: PoolTransfer,
        layer: int,
        locations: np.ndarray,
        timings: _TransferBuildTimings | None = None,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        layout = self._layer_pointer_layouts.get((entry.name, layer))
        if layout is None:
            return np.empty((0, 0), dtype=np.uint64), ()
        page_count = len(transfer.keys or ())
        if not entry.packed and len(layout.schema) != 1:
            raise ValueError(
                f"UCM bundle requires one address row per page for {entry.name}: "
                f"fragments={len(layout.schema)} pages={page_count}."
            )
        meta_start = time.perf_counter() if timings is not None else 0.0
        # Only the page rows change per transfer. Compute the address matrix
        # without rebuilding a Python size/offset list for every page.
        pointer_matrix = (
            locations[:, None] * layout.strides[None, :] + layout.bases[None, :]
        )
        if timings is not None:
            timings.layer_meta_ms += (time.perf_counter() - meta_start) * 1000
            timings.layer_meta_calls += 1
        if pointer_matrix.shape[0] != page_count:
            raise ValueError(
                f"UCM bundle requires one address row per page for {entry.name}: "
                f"pointers={pointer_matrix.shape[0]} pages={page_count}."
            )
        return pointer_matrix, layout.schema

    def _page_transfers(
        self,
        transfers: list[PoolTransfer],
        *,
        for_io: bool,
        timings: _TransferBuildTimings | None = None,
    ) -> list[UcmLayerTransfer]:
        by_name = {transfer.name: transfer for transfer in transfers}
        result: list[UcmLayerTransfer] = []
        for bundle in self._bundles:
            members = [
                (entry, by_name.get(entry.name))
                for entry in sorted(bundle.entries, key=lambda item: str(item.name))
            ]
            if any(transfer is None or not transfer.keys for _, transfer in members):
                continue
            page_keys = tuple(members[0][1].keys)
            hit_policy = members[0][1].hit_policy
            if any(
                tuple(transfer.keys) != page_keys or transfer.hit_policy != hit_policy
                for _, transfer in members[1:]
            ):
                raise ValueError(
                    f"UCM store bundle {bundle.name} has inconsistent source transfer metadata."
                )

            schema_start = time.perf_counter() if timings is not None else 0.0
            fragments = tuple(
                fragment
                for layer in bundle.layers
                for entry, _ in members
                for fragment in self._entry_layer_schema(entry, layer)
            )
            if not fragments:
                continue
            layer_sizes = tuple(
                sum(
                    size
                    for entry, _ in members
                    for size, _ in self._entry_layer_schema(entry, layer)
                )
                for layer in bundle.layers
            )
            if not layer_sizes or len(set(layer_sizes)) != 1:
                raise ValueError(
                    f"UCM bundle {bundle.name} has inconsistent layer sizes: "
                    f"{layer_sizes}."
                )
            layer_tensor_size_list = tuple(
                size
                for entry, _ in members
                for size, _ in self._entry_layer_schema(entry, bundle.layers[0])
            )
            if timings is not None:
                timings.schema_ms += (time.perf_counter() - schema_start) * 1000
            if for_io:
                locations_start = time.perf_counter() if timings is not None else 0.0
                prepared_locations = []
                for entry, transfer in members:
                    if transfer.host_indices is None:
                        raise ValueError(
                            f"UCM {entry.name} transfer has no device indices."
                        )
                    prepared_locations.append(
                        np.asarray(
                            entry.prepare_locations(transfer.host_indices),
                            dtype=np.uint64,
                        )
                    )
                if timings is not None:
                    timings.locations_ms += (
                        time.perf_counter() - locations_start
                    ) * 1000
                assemble_start = time.perf_counter() if timings is not None else 0.0
                # Each layer slice is contiguous and can be passed to Store
                # without converting every page's pointers to Python lists.
                tensor_count = len(layer_tensor_size_list)
                if len(fragments) != len(bundle.layers) * tensor_count:
                    raise ValueError(
                        f"UCM bundle {bundle.name} has inconsistent fragment counts."
                    )
                addresses = np.empty(
                    (len(bundle.layers), len(page_keys), tensor_count),
                    dtype=np.uint64,
                )
                if timings is not None:
                    timings.address_assemble_ms += (
                        time.perf_counter() - assemble_start
                    ) * 1000
                for layer_index, layer in enumerate(bundle.layers):
                    column = 0
                    for (entry, transfer), locations in zip(
                        members, prepared_locations
                    ):
                        entry_rows, schema = self._entry_layer_rows(
                            entry, transfer, layer, locations, timings
                        )
                        if (
                            entry_rows.shape != (len(page_keys), len(schema))
                            or not schema
                        ):
                            raise ValueError(
                                f"UCM bundle {bundle.name} is missing "
                                f"{entry.name} layer={layer}."
                            )
                        assemble_start = (
                            time.perf_counter() if timings is not None else 0.0
                        )
                        next_column = column + len(schema)
                        addresses[layer_index, :, column:next_column] = entry_rows
                        column = next_column
                        if timings is not None:
                            timings.address_assemble_ms += (
                                time.perf_counter() - assemble_start
                            ) * 1000
                    if column != tensor_count:
                        raise ValueError(
                            f"UCM bundle {bundle.name} page address/schema mismatch."
                        )
            else:
                addresses = tuple(() for _ in page_keys)
            result.append(
                UcmLayerTransfer(
                    store_group=bundle.name,
                    source_pool=str(bundle.source_pool),
                    pools=tuple(str(entry.name) for entry in bundle.entries),
                    layers=bundle.layers,
                    shard_size=layer_sizes[0],
                    tensor_size_list=layer_tensor_size_list,
                    page_keys=page_keys,
                    addresses=addresses,
                    fragments=fragments,
                    hit_policy=str(getattr(hit_policy, "value", hit_policy)),
                )
            )
        return result

    def _expand(self, transfers: list[PoolTransfer], **kwargs) -> list[PoolTransfer]:
        return self.pool_group.resolve_transfers(transfers, **kwargs)

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        started_at = time.perf_counter()
        expanded = self._expand(transfers)
        if not expanded:
            return []
        kv_transfer = next(
            (transfer for transfer in transfers if transfer.name == PoolName.KV),
            None,
        )
        page_keys = list(kv_transfer.keys or ()) if kv_transfer is not None else []
        if not page_keys:
            return []
        if not self._lookup_owner:
            # SGLang intersects each rank's restorable-boundary set. Supplying
            # every boundary is neutral while the local host owner probes its
            # shared Cache/Posix view.
            self.stats["lookup_skipped"] += 1
            return list(range(1, len(page_keys) + 1))
        build_start = time.perf_counter()
        lookup_transfers = self._page_transfers(expanded, for_io=False)
        build_ms = (time.perf_counter() - build_start) * 1000
        if not lookup_transfers:
            return []
        transfer_keys = tuple(
            (
                transfer.store_group,
                transfer.source_pool,
                transfer.pools,
                transfer.layers,
                transfer.hit_policy,
                transfer.page_keys,
            )
            for transfer in lookup_transfers
        )
        with self._lookup_progress_lock:
            previous = self._lookup_progress.get(rid)
        resume_from_pages = 0
        if (
            previous is not None
            and previous.hit_pages <= len(page_keys)
            and previous.page_keys == tuple(page_keys[: len(previous.page_keys)])
            and len(previous.transfer_keys) == len(transfer_keys)
            and all(
                old[:-1] == new[:-1] and old[-1] == new[-1][: len(old[-1])]
                for old, new in zip(previous.transfer_keys, transfer_keys)
            )
        ):
            resume_from_pages = previous.hit_pages
        lookup_stats = {"queried_pages": 0, "fallbacks": 0, "resumed_groups": 0}
        store_start = time.perf_counter()
        result = self.adapter.lookup(
            lookup_transfers,
            page_keys,
            resume_from_pages=resume_from_pages,
            lookup_stats=lookup_stats,
        )
        store_ms = (time.perf_counter() - store_start) * 1000
        with self._lookup_progress_lock:
            self._lookup_progress[rid] = _LookupProgress(
                tuple(page_keys), transfer_keys, result[-1] if result else 0
            )
            self._lookup_progress.move_to_end(rid)
            if len(self._lookup_progress) > _LOOKUP_PROGRESS_MAX_REQUESTS:
                self._lookup_progress.popitem(last=False)
        total_ms = (time.perf_counter() - started_at) * 1000
        self.stats["lookup"] += 1
        self.stats["lookup_resumed"] += int(lookup_stats["resumed_groups"] > 0)
        log_lookup, lookup_calls, slow_calls = self._record_lookup_timing(
            rid, total_ms, store_ms
        )
        if log_lookup:
            self._lookup_logger.info(
                "UCM linker lookup dp_rank=%d tp_rank=%d rid=%s pages=%d transfers=%d "
                "lookup_calls=%d slow_calls=%d hit_pages=%d queried_pages=%d resume_from_pages=%d "
                "resumed_groups=%d lookup_fallbacks=%d "
                "build_ms=%.3f store_lookup_ms=%.3f "
                "total_ms=%.3f",
                self._dp_rank,
                self._tp_rank,
                rid,
                len(page_keys),
                len(lookup_transfers),
                lookup_calls,
                slow_calls,
                result[-1] if result else 0,
                lookup_stats["queried_pages"],
                resume_from_pages,
                lookup_stats["resumed_groups"],
                lookup_stats["fallbacks"],
                build_ms,
                store_ms,
                total_ms,
            )
        return result

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        started_at = time.perf_counter()
        self._finish_lookup_timing(rid)
        with self._lookup_progress_lock:
            self._lookup_progress.pop(rid, None)
        expanded = self._expand(transfers, allow_partial=True, allow_missing_kv=True)
        if not expanded:
            return False
        with self._pending_lock:
            if rid in self._pending_loads:
                raise RuntimeError(f"UCM load for rid={rid} is already queued.")
            self._pending_loads[rid] = expanded
        enqueue_ms = (time.perf_counter() - started_at) * 1000
        if self._should_log_timing("load_enqueue", enqueue_ms, _LOAD_TIMING_LOG_MS):
            self._logger.info(
                "UCM linker load enqueue tp_rank=%d rid=%s transfers=%d "
                "enqueue_ms=%.3f",
                self._tp_rank,
                rid,
                len(expanded),
                enqueue_ms,
            )
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        with self._pending_lock:
            return self._pending_loads.pop(rid, None) is not None

    def num_completed_loads(self) -> int:
        return self._completed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        return self._completed_loads.get_nowait()

    def _freeze_gc_once(self) -> None:
        if not self._gc_frozen:
            freeze_gc("UCM direct linker")
            self._gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        with self._pending_lock:
            if not self._pending_loads:
                return -1
            pending = self._pending_loads
            self._pending_loads = {}
        self._freeze_gc_once()
        counter_index = self.layer_done_counter.update_producer()
        if self._requires_load_finish:
            self._load_results[counter_index] = Future()
        ready_event = device_module.Event()
        ready_event.record()
        self._load_queue.put((counter_index, pending, ready_event, time.perf_counter()))
        self.stats["load"] += len(pending)
        return counter_index

    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        if counter_index < 0:
            return True
        started_at = time.perf_counter()
        result = self._load_results[counter_index]
        try:
            landed = result.result()
            return self.layer_done_counter.finish(counter_index) and landed
        finally:
            wait_ms = (time.perf_counter() - started_at) * 1000
            with self._timing_log_lock:
                forward_wait = self._forward_wait_timings.pop(counter_index, None)
            if forward_wait is not None and self._should_log_timing(
                "forward_wait", forward_wait.total_ms, _LOAD_TIMING_LOG_MS
            ):
                self._logger.info(
                    "UCM linker forward layer wait tp_rank=%d batch=%d calls=%d "
                    "slow_calls=%d total_ms=%.3f max_layer=%d max_ms=%.3f",
                    self._tp_rank,
                    counter_index,
                    forward_wait.calls,
                    forward_wait.slow_calls,
                    forward_wait.total_ms,
                    forward_wait.max_layer,
                    forward_wait.max_ms,
                )
            if self._should_log_timing("load_finish", wait_ms, _LOAD_TIMING_LOG_MS):
                self._logger.info(
                    "UCM linker load finish tp_rank=%d batch=%d finish_wait_ms=%.3f",
                    self._tp_rank,
                    counter_index,
                    wait_ms,
                )
            self._load_results.pop(counter_index, None)

    def _wait_tasks(self, tasks: object) -> None:
        # Wait consumes the CacheStore task handle after returning. Do not call
        # check() afterwards: CacheStore would report -50005 (NotFound) for the
        # already-reaped task even when the transfer completed successfully.
        self.adapter.wait(tasks)

    def _load_worker(self) -> None:
        while True:
            task = self._load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready_event, queued_at = task
                landed = True
                active_layer = -1
                try:
                    worker_started = time.perf_counter()
                    batch_start = worker_started
                    queue_wait_ms = (worker_started - queued_at) * 1000
                    ready_start = worker_started
                    ready_event.synchronize()
                    ready_wait_ms = (time.perf_counter() - ready_start) * 1000
                    transfer_build_start = time.perf_counter()
                    build_timings = _TransferBuildTimings()
                    page_transfers = [
                        transfer
                        for request in pending.values()
                        for transfer in self._page_transfers(
                            request, for_io=True, timings=build_timings
                        )
                    ]
                    transfer_build_ms = (
                        time.perf_counter() - transfer_build_start
                    ) * 1000
                    transfer_build_other_ms = transfer_build_ms - sum(
                        (
                            build_timings.schema_ms,
                            build_timings.locations_ms,
                            build_timings.layer_meta_ms,
                            build_timings.row_convert_ms,
                            build_timings.address_assemble_ms,
                        )
                    )
                    transfer_prepare_start = time.perf_counter()
                    prepared_transfers = self.adapter.prepare_load(page_transfers)
                    transfer_prepare_ms = (
                        time.perf_counter() - transfer_prepare_start
                    ) * 1000
                    page_count = sum(
                        len(transfer.page_keys) for transfer in page_transfers
                    )
                    # Submit a bounded number of future layers, but publish
                    # readiness in logical-layer order. CacheStore can progress
                    # queued tasks in C++ while Python waits for this layer.
                    in_flight: dict[int, tuple[list[Any], float, float]] = {}
                    next_layer = 0
                    total_submit_ms = 0.0
                    total_wait_ms = 0.0
                    max_layer_ms = 0.0
                    max_layer_index = -1
                    slow_layer_count = 0
                    store_task_count = 0
                    try:
                        for layer in range(self.num_layers):
                            active_layer = layer
                            if page_transfers:
                                limit = min(
                                    self.num_layers,
                                    layer + self._load_prefetch_layers + 1,
                                )
                                while next_layer < limit:
                                    submit_start = time.perf_counter()
                                    tasks = self.adapter.submit_prepared_load_layer(
                                        prepared_transfers, next_layer
                                    )
                                    submit_ms = (
                                        time.perf_counter() - submit_start
                                    ) * 1000
                                    total_submit_ms += submit_ms
                                    store_task_count += len(tasks)
                                    in_flight[next_layer] = (
                                        tasks,
                                        submit_start,
                                        submit_ms,
                                    )
                                    next_layer += 1

                                # A logical layer can span several pool bundles
                                # and requests; release it only after every task
                                # for that layer has finished.
                                tasks, layer_start, submit_ms = in_flight.pop(layer)
                                store_wait_start = time.perf_counter()
                                self._wait_tasks(tasks)
                                store_wait_ms = (
                                    time.perf_counter() - store_wait_start
                                ) * 1000
                                total_wait_ms += store_wait_ms
                                layer_ms = (time.perf_counter() - layer_start) * 1000
                                slow_layer_count += int(layer_ms >= _LOAD_TIMING_LOG_MS)
                                self._should_log_timing(
                                    "load_layer", layer_ms, _LOAD_TIMING_LOG_MS
                                )
                                if layer_ms > max_layer_ms:
                                    max_layer_ms = layer_ms
                                    max_layer_index = layer
                            self.layer_done_counter.complete(counter_index, layer)
                    finally:
                        # A failed submit or wait must not leave prefetched
                        # tasks writing into device slots after this batch ends.
                        for tasks, _, _ in in_flight.values():
                            try:
                                self._wait_tasks(tasks)
                            except Exception:
                                self._logger.exception(
                                    "UCM prefetched load failed while draining"
                                )
                    finished_at = time.perf_counter()
                    worker_ms = (finished_at - batch_start) * 1000
                    total_ms = (finished_at - queued_at) * 1000
                    if self._should_log_timing(
                        "load_batch", total_ms, _LOAD_TIMING_LOG_MS
                    ):
                        slow_batches, max_batch_ms = self._slow_timing_snapshot(
                            "load_batch"
                        )
                        slow_layers_total, max_layer_total_ms = (
                            self._slow_timing_snapshot("load_layer")
                        )
                        self._logger.info(
                            "UCM linker load batch end tp_rank=%d batch=%d requests=%d "
                            "transfers=%d transfer_pages=%d store_tasks=%d "
                            "prefetch_layers=%d queue_wait_ms=%.3f ready_event_ms=%.3f "
                            "transfer_build_ms=%.3f transfer_prepare_ms=%.3f "
                            "build_schema_ms=%.3f build_locations_ms=%.3f "
                            "build_layer_meta_ms=%.3f build_row_convert_ms=%.3f "
                            "build_address_assemble_ms=%.3f build_other_ms=%.3f "
                            "build_layer_meta_calls=%d "
                            "submit_sum_ms=%.3f wait_sum_ms=%.3f "
                            "slow_layers=%d slowest_layer=%d slowest_layer_ms=%.3f "
                            "worker_ms=%.3f total_ms=%.3f "
                            "slow_batches=%d max_batch_ms=%.3f "
                            "slow_layers_total=%d max_layer_total_ms=%.3f",
                            self._tp_rank,
                            counter_index,
                            len(pending),
                            len(page_transfers),
                            page_count,
                            store_task_count,
                            self._load_prefetch_layers,
                            queue_wait_ms,
                            ready_wait_ms,
                            transfer_build_ms,
                            transfer_prepare_ms,
                            build_timings.schema_ms,
                            build_timings.locations_ms,
                            build_timings.layer_meta_ms,
                            build_timings.row_convert_ms,
                            build_timings.address_assemble_ms,
                            transfer_build_other_ms,
                            build_timings.layer_meta_calls,
                            total_submit_ms,
                            total_wait_ms,
                            slow_layer_count,
                            max_layer_index,
                            max_layer_ms,
                            worker_ms,
                            total_ms,
                            slow_batches,
                            max_batch_ms,
                            slow_layers_total,
                            max_layer_total_ms,
                        )
                except BaseException as error:
                    landed = False
                    self.layer_done_counter.fail(counter_index, error)
                    self._logger.exception(
                        "UCM layer-wise load batch failed tp_rank=%d batch=%d "
                        "layer=%d requests=%d elapsed_ms=%.3f",
                        self._tp_rank,
                        counter_index,
                        active_layer,
                        len(pending),
                        (time.perf_counter() - batch_start) * 1000,
                    )
                finally:
                    if self._requires_load_finish:
                        self._load_results[counter_index].set_result(landed)
                    else:
                        self._completed_loads.put(list(pending))
            finally:
                self._load_queue.task_done()

    def offload(
        self, transfers: list[PoolTransfer], session_id: str | None = None
    ) -> bool:
        # Newer SGLang passes session ownership for optional backend
        # attribution. UCM block identities are independent of that hint.
        del session_id
        expanded = self._expand(transfers, allow_partial=True)
        if not expanded:
            return False
        self._freeze_gc_once()
        if not self._offload_owner:
            self._offload_results.put(True)
            return True
        indices_ready = device_module.Event()
        indices_ready.record()
        ready_events = [indices_ready]
        if self._forward_stream is not None:
            kv_written = device_module.Event()
            kv_written.record(self._forward_stream)
            ready_events.append(kv_written)
        self._offload_queue.put((expanded, ready_events, time.perf_counter()))
        return True

    def _offload_worker(self) -> None:
        # Submit nodes without waiting for earlier Store tasks. Reap completed
        # tasks independently, then publish results in SGLang's node order.
        in_flight: deque[_PendingOffload] = deque()
        no_task = object()
        next_poll_at = 0.0
        sequence = 0

        def submit(task: tuple[list[PoolTransfer], list[object], float]) -> None:
            nonlocal sequence
            sequence += 1
            expanded, ready_events, queued_at = task
            transfers: list[UcmLayerTransfer] = []
            transfer_pages = 0
            started_at = time.perf_counter()
            try:
                # The index is read on CPU while building transfers. The KV
                # event can instead be waited on by Cache's D2H stream.
                ready_events[0].synchronize()
                prerequisite_handle = 0
                if len(ready_events) > 1:
                    kv_event = ready_events[1]
                    if getattr(self.adapter, "supports_dump_prerequisite", False):
                        prerequisite_handle = int(
                            getattr(kv_event, "cuda_event", 0) or 0
                        )
                    if not prerequisite_handle:
                        kv_event.synchronize()
                ready_ms = (time.perf_counter() - started_at) * 1000
                build_start = time.perf_counter()
                transfers = self._page_transfers(expanded, for_io=True)
                transfer_pages = sum(len(t.page_keys) for t in transfers)
                build_ms = (time.perf_counter() - build_start) * 1000
                submit_start = time.perf_counter()
                tasks = (
                    self.adapter.submit_offload(
                        transfers, prerequisite_handle=prerequisite_handle
                    )
                    if prerequisite_handle
                    else self.adapter.submit_offload(transfers)
                )
                submitted_at = time.perf_counter()
            except BaseException:
                finished_at = time.perf_counter()
                self._logger.exception(
                    "UCM offload submit failed tp_rank=%d node=%d queue_ms=%.3f "
                    "elapsed_ms=%.3f",
                    self._tp_rank,
                    sequence,
                    (started_at - queued_at) * 1000,
                    (finished_at - started_at) * 1000,
                )
                in_flight.append(
                    _PendingOffload(
                        sequence=sequence,
                        tasks=[],
                        transfers=transfers,
                        ready_events=[],
                        queued_at=queued_at,
                        submitted_at=finished_at,
                        transfer_pages=transfer_pages,
                        result=False,
                        finished_at=finished_at,
                    )
                )
                return

            # Keep address arrays alive until every native task is reaped.
            in_flight.append(
                _PendingOffload(
                    sequence=sequence,
                    tasks=tasks,
                    transfers=transfers,
                    ready_events=ready_events,
                    queued_at=queued_at,
                    submitted_at=submitted_at,
                    transfer_pages=transfer_pages,
                    store_tasks=len(tasks),
                    queue_backlog=self._offload_queue.qsize(),
                    queue_ms=(started_at - queued_at) * 1000,
                    ready_event_ms=ready_ms,
                    transfer_build_ms=build_ms,
                    store_submit_ms=(submitted_at - submit_start) * 1000,
                )
            )

        while True:
            try:
                task = (
                    self._offload_queue.get(
                        timeout=max(0.0, next_poll_at - time.perf_counter())
                    )
                    if in_flight
                    else self._offload_queue.get()
                )
            except Empty:
                task = no_task
            if task is None:
                # close() enqueues the sentinel after reset() drains work.
                self._offload_queue.task_done()
                return

            if task is not no_task:
                submit(task)

            now = time.perf_counter()
            if in_flight and now >= next_poll_at:
                for pending in in_flight:
                    if pending.result is not None:
                        continue
                    check_start = time.perf_counter()
                    pending.check_calls += 1
                    try:
                        completed = all(self.adapter.check(t) for t in pending.tasks)
                    except BaseException:
                        self._logger.exception(
                            "UCM offload check failed tp_rank=%d node=%d",
                            self._tp_rank,
                            pending.sequence,
                        )
                        # An error may leave native tasks active. Wait reaps
                        # every handle before tree locks are released.
                        pending.result = False
                        completed = True
                    finally:
                        pending.check_ms += (time.perf_counter() - check_start) * 1000
                    if not completed:
                        continue
                    if pending.result is None:
                        pending.result = True
                    wait_start = time.perf_counter()
                    try:
                        # Check does not consume a task handle. Wait obtains its
                        # final status and releases it after Check reports done.
                        self._wait_tasks(pending.tasks)
                    except BaseException:
                        self._logger.exception(
                            "UCM offload failed tp_rank=%d node=%d",
                            self._tp_rank,
                            pending.sequence,
                        )
                        pending.result = False
                    pending.wait_ms = (time.perf_counter() - wait_start) * 1000
                    pending.finished_at = time.perf_counter()
                    pending.tasks = []
                    pending.transfers = []
                    pending.ready_events = []
                next_poll_at = time.perf_counter() + _OFFLOAD_POLL_INTERVAL_S

            while in_flight and in_flight[0].result is not None:
                pending = in_flight.popleft()
                published_at = time.perf_counter()
                store_ms = (pending.finished_at - pending.submitted_at) * 1000
                total_ms = (published_at - pending.queued_at) * 1000
                publish_delay_ms = (published_at - pending.finished_at) * 1000
                if self._should_log_timing("offload", total_ms, _OFFLOAD_TIMING_LOG_MS):
                    slow_offloads, max_offload_ms = self._slow_timing_snapshot(
                        "offload"
                    )
                    self._logger.info(
                        "UCM linker offload complete tp_rank=%d node=%d "
                        "success=%s transfer_pages=%d store_tasks=%d "
                        "queue_backlog=%d queue_ms=%.3f ready_event_ms=%.3f "
                        "transfer_build_ms=%.3f store_submit_ms=%.3f "
                        "check_calls=%d "
                        "check_ms=%.3f store_wait_ms=%.3f store_async_ms=%.3f "
                        "publish_delay_ms=%.3f total_ms=%.3f in_flight=%d "
                        "slow_offloads=%d max_offload_ms=%.3f",
                        self._tp_rank,
                        pending.sequence,
                        pending.result,
                        pending.transfer_pages,
                        pending.store_tasks,
                        pending.queue_backlog,
                        pending.queue_ms,
                        pending.ready_event_ms,
                        pending.transfer_build_ms,
                        pending.store_submit_ms,
                        pending.check_calls,
                        pending.check_ms,
                        pending.wait_ms,
                        store_ms,
                        publish_delay_ms,
                        total_ms,
                        len(in_flight),
                        slow_offloads,
                        max_offload_ms,
                    )
                if pending.result:
                    self.stats["offload"] += 1
                self._offload_results.put(pending.result)
                self._offload_queue.task_done()
            if not in_flight:
                next_poll_at = 0.0

    def num_completed_offloads(self) -> int:
        return self._offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self._offload_results.get_nowait()

    @staticmethod
    def _drain(queue: Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return

    def reset(self) -> None:
        with self._lookup_progress_lock:
            self._lookup_progress.clear()
        with self._pending_lock:
            self._pending_loads.clear()
        self._load_queue.join()
        self._load_results.clear()
        self._offload_queue.join()
        with self._timing_log_lock:
            pending_timings = tuple(self._lookup_timings.items())
            self._lookup_timings.clear()
            self._forward_wait_timings.clear()
            slow_timings = dict(self._slow_timings)
            self._slow_timings.clear()
        for rid, timing in pending_timings:
            self._log_lookup_summary(rid, timing)
        if slow_timings and self._tp_rank == 0:
            self._logger.info(
                "UCM linker slow timings count_and_max_ms=%s", slow_timings
            )
        self._drain(self._completed_loads)
        self._drain(self._offload_results)
        self.layer_done_counter.reset()

    def close(self) -> None:
        if self._closed:
            return
        self.reset()
        self._closed = True
        self._load_queue.put(None)
        self._offload_queue.put(None)
        self._load_thread.join()
        self._offload_thread.join()
        self.adapter.close()
        if self._tp_rank == 0:
            self._logger.info("UCM direct linker stats: %s", self.stats)
