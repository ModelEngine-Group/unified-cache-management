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
_LOOKUP_PROGRESS_MAX_REQUESTS = 128
_OFFLOAD_POLL_INTERVAL_S = 0.005


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
    fragments: tuple[tuple[int, int], ...]
    tensor_size_list: tuple[int, ...]
    shard_size: int


@dataclass(frozen=True)
class _LookupProgress:
    page_keys: tuple[str, ...]
    transfer_keys: tuple[tuple[object, ...], ...]
    hit_pages: int


@dataclass(frozen=True)
class _LayerPointerLayout:
    """Fixed buffer addresses, strides and fragment schema for one pool layer."""

    bases: np.ndarray
    strides: np.ndarray
    schema: tuple[tuple[int, int], ...]


@dataclass
class _PendingOffload:
    sequence: int
    tasks: list[Any]
    transfers: list[UcmLayerTransfer]
    ready_events: list[object]
    result: bool | None = None


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
        # Scheduler-owned state; workers receive batches through _load_queue.
        self._pending_loads: dict[str, list[PoolTransfer]] = {}
        self._load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object] | None
        ] = Queue()
        self._completed_loads: Queue[list[str]] = Queue()
        self._requires_load_finish = hasattr(
            UnifiedCacheLinker, "finish_layer_wise_loading"
        )
        self._load_results: dict[int, Future[bool]] = {}
        self._offload_queue: Queue[tuple[list[PoolTransfer], list[object]] | None] = (
            Queue()
        )
        self._offload_results: Queue[bool] = Queue()
        self._lookup_progress: OrderedDict[str, _LookupProgress] = OrderedDict()

        self._tp_rank = 0
        parallel = get_parallel()
        self._tp_size = parallel.tp_size
        self._dp_rank = int(getattr(parallel, "dp_rank", None) or 0)
        self._attn_dp_rank = int(getattr(parallel, "attn_dp_rank", None) or 0)
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._tp_rank = torch.distributed.get_rank(group=tp_group)
            self._tp_size = torch.distributed.get_world_size(group=tp_group)
        self._lookup_owner = True
        self._offload_owner = not self.pool_group.rank_replicated or self._tp_rank == 0
        self._log_pool_topology()
        try:
            self.adapter = adapter or self._create_adapter(params, extra_config)
            self.adapter.register_device_pools(self._pool_metadata())
            self._select_lookup_owner(tp_group, params.attn_cp_size)
            self._load_prefetch_layers = getattr(
                self.adapter, "load_prefetch_layers", 2
            )
            logger.info(
                "UCM linker load prefetch layers=%d", self._load_prefetch_layers
            )
        except Exception:
            logger.exception(
                "UCM linker initialization failed: layers=%d tp=%d/%d replicated=%s.",
                self.num_layers,
                self._tp_rank,
                self._tp_size,
                self.pool_group.rank_replicated,
            )
            raise

        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
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

    def _select_lookup_owner(self, tp_group: Any, attn_cp_size: int) -> None:
        """Elect one reader per shared Cache within this attention-TP group."""
        if (
            not self.pool_group.rank_replicated
            or attn_cp_size != 1
            or tp_group is None
            or self._tp_size <= 1
            or not torch.distributed.is_initialized()
        ):
            return

        # Ranks on one host are assumed to share /dev/shm. Keep separate
        # readers for different hosts or Store identities.
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
                if boot_id:
                    cache_identity = (
                        boot_id,
                        tuple(sorted(store_signatures.items())),
                    )
            except OSError:
                pass
        identities: list[Any] = [None] * self._tp_size
        torch.distributed.all_gather_object(identities, cache_identity, group=tp_group)
        if cache_identity is not None:
            self._lookup_owner = self._tp_rank == identities.index(cache_identity)
        if self._lookup_owner:
            logger.info(
                "UCM linker lookup owner dp_rank=%d tp_rank=%d tp_size=%d "
                "shared_cache_identified=%s",
                self._dp_rank,
                self._tp_rank,
                self._tp_size,
                cache_identity is not None,
            )

    def _create_adapter(
        self, params: CacheInitParams, extra_config: dict[str, Any] | None
    ) -> UcmDirectStoreAdapter:
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
        config = dict(extra_config or {})
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
            for index, (tensor_sizes, schema_layers) in enumerate(
                sorted(by_schema.items())
            ):
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
                        tuple(
                            fragment
                            for layer in schema_layers
                            for entry in ordered
                            for fragment in self._entry_layer_schema(entry, layer)
                        ),
                        tensor_sizes,
                        sum(tensor_sizes),
                    )
                )
        return tuple(bundles)

    def _log_pool_topology(self) -> None:
        logger.info(
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
            logger.info(
                "UCM pool bundle=%s source=%s pools=%s "
                "page_block_tensors=%d page_block_bytes=%d.",
                bundle.name,
                bundle.source_pool,
                [str(entry.name) for entry in bundle.entries],
                len(bundle.fragments),
                bundle.shard_size * len(bundle.layers),
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
                    if not items or (not entry.packed and len(items) != 1):
                        raise ValueError(
                            f"UCM pool {entry.name} layer={layer} has an unsupported "
                            f"address layout: fragments={len(items)} packed={entry.packed}."
                        )
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
        layer: int,
        locations: np.ndarray,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        layout = self._layer_pointer_layouts[(entry.name, layer)]
        # Only the page rows change per transfer. Compute the address matrix
        # without rebuilding a Python size/offset list for every page.
        pointer_matrix = (
            locations[:, None] * layout.strides[None, :] + layout.bases[None, :]
        )
        return pointer_matrix, layout.schema

    def _page_transfers(
        self,
        transfers: list[PoolTransfer],
        *,
        for_io: bool,
    ) -> list[UcmLayerTransfer]:
        by_name = {transfer.name: transfer for transfer in transfers}
        result: list[UcmLayerTransfer] = []
        for bundle in self._bundles:
            members = [(entry, by_name.get(entry.name)) for entry in bundle.entries]
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

            fragments = bundle.fragments
            layer_tensor_size_list = bundle.tensor_size_list
            if for_io:
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
                # Each layer slice is contiguous and can be passed to Store
                # without converting every page's pointers to Python lists.
                tensor_count = len(layer_tensor_size_list)
                addresses = np.empty(
                    (len(bundle.layers), len(page_keys), tensor_count),
                    dtype=np.uint64,
                )
                for layer_index, layer in enumerate(bundle.layers):
                    column = 0
                    for (entry, transfer), locations in zip(
                        members, prepared_locations
                    ):
                        entry_rows, schema = self._entry_layer_rows(
                            entry, layer, locations
                        )
                        if entry_rows.shape != (len(page_keys), len(schema)):
                            raise ValueError(
                                f"UCM bundle {bundle.name} is missing "
                                f"{entry.name} layer={layer}."
                            )
                        next_column = column + len(schema)
                        addresses[layer_index, :, column:next_column] = entry_rows
                        column = next_column
            else:
                addresses = None
            result.append(
                UcmLayerTransfer(
                    store_group=bundle.name,
                    source_pool=str(bundle.source_pool),
                    pools=tuple(str(entry.name) for entry in bundle.entries),
                    layers=bundle.layers,
                    shard_size=bundle.shard_size,
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
            return list(range(1, len(page_keys) + 1))
        lookup_transfers = self._page_transfers(expanded, for_io=False)
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
        result = self.adapter.lookup(
            lookup_transfers,
            page_keys,
            resume_from_pages=resume_from_pages,
        )
        self._lookup_progress[rid] = _LookupProgress(
            tuple(page_keys), transfer_keys, result[-1] if result else 0
        )
        self._lookup_progress.move_to_end(rid)
        if len(self._lookup_progress) > _LOOKUP_PROGRESS_MAX_REQUESTS:
            self._lookup_progress.popitem(last=False)
        return result

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        self._lookup_progress.pop(rid, None)
        expanded = self._expand(transfers, allow_partial=True, allow_missing_kv=True)
        if not expanded:
            return False
        if rid in self._pending_loads:
            raise RuntimeError(f"UCM load for rid={rid} is already queued.")
        self._pending_loads[rid] = expanded
        return True

    def cancel_queued_load(self, rid: str) -> bool:
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
        self._load_queue.put((counter_index, pending, ready_event))
        return counter_index

    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        if counter_index < 0:
            return True
        result = self._load_results[counter_index]
        try:
            landed = result.result()
            return self.layer_done_counter.finish(counter_index) and landed
        finally:
            self._load_results.pop(counter_index, None)

    def _load_worker(self) -> None:
        while True:
            task = self._load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready_event = task
                landed = True
                active_layer = -1
                try:
                    ready_event.synchronize()
                    page_transfers = [
                        transfer
                        for request in pending.values()
                        for transfer in self._page_transfers(request, for_io=True)
                    ]
                    prepared_transfers = self.adapter.prepare_load(page_transfers)
                    # Submit a bounded number of future layers, but publish
                    # readiness in logical-layer order. CacheStore can progress
                    # queued tasks in C++ while Python waits for this layer.
                    in_flight: dict[int, list[Any]] = {}
                    next_layer = 0
                    try:
                        for layer in range(self.num_layers):
                            active_layer = layer
                            if page_transfers:
                                limit = min(
                                    self.num_layers,
                                    layer + self._load_prefetch_layers + 1,
                                )
                                while next_layer < limit:
                                    tasks = self.adapter.submit_prepared_load_layer(
                                        prepared_transfers, next_layer
                                    )
                                    in_flight[next_layer] = tasks
                                    next_layer += 1

                                # A logical layer can span several pool bundles
                                # and requests; release it only after every task
                                # for that layer has finished.
                                tasks = in_flight.pop(layer)
                                self.adapter.wait(tasks)
                            self.layer_done_counter.complete(counter_index, layer)
                    finally:
                        # A failed submit or wait must not leave prefetched
                        # tasks writing into device slots after this batch ends.
                        for tasks in in_flight.values():
                            try:
                                self.adapter.wait(tasks)
                            except Exception:
                                logger.exception(
                                    "UCM prefetched load failed while draining"
                                )
                except BaseException as error:
                    landed = False
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception(
                        "UCM layer-wise load batch failed tp_rank=%d batch=%d "
                        "layer=%d requests=%d",
                        self._tp_rank,
                        counter_index,
                        active_layer,
                        len(pending),
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
        self._offload_queue.put((expanded, ready_events))
        return True

    def _offload_worker(self) -> None:
        # Submit nodes without waiting for earlier Store tasks. Reap completed
        # tasks independently, then publish results in SGLang's node order.
        in_flight: deque[_PendingOffload] = deque()
        no_task = object()
        next_poll_at = 0.0
        sequence = 0

        def submit(task: tuple[list[PoolTransfer], list[object]]) -> None:
            nonlocal sequence
            sequence += 1
            expanded, ready_events = task
            transfers: list[UcmLayerTransfer] = []
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
                transfers = self._page_transfers(expanded, for_io=True)
                tasks = (
                    self.adapter.submit_offload(
                        transfers, prerequisite_handle=prerequisite_handle
                    )
                    if prerequisite_handle
                    else self.adapter.submit_offload(transfers)
                )
            except BaseException:
                logger.exception(
                    "UCM offload submit failed tp_rank=%d node=%d",
                    self._tp_rank,
                    sequence,
                )
                in_flight.append(
                    _PendingOffload(
                        sequence=sequence,
                        tasks=[],
                        transfers=transfers,
                        ready_events=[],
                        result=False,
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
                    try:
                        completed = all(self.adapter.check(t) for t in pending.tasks)
                    except BaseException:
                        logger.exception(
                            "UCM offload check failed tp_rank=%d node=%d",
                            self._tp_rank,
                            pending.sequence,
                        )
                        # An error may leave native tasks active. Wait reaps
                        # every handle before tree locks are released.
                        pending.result = False
                        completed = True
                    if not completed:
                        continue
                    if pending.result is None:
                        pending.result = True
                    try:
                        # Check does not consume a task handle. Wait obtains its
                        # final status and releases it after Check reports done.
                        self.adapter.wait(pending.tasks)
                    except BaseException:
                        logger.exception(
                            "UCM offload failed tp_rank=%d node=%d",
                            self._tp_rank,
                            pending.sequence,
                        )
                        pending.result = False
                    pending.tasks = []
                    pending.transfers = []
                    pending.ready_events = []
                next_poll_at = time.perf_counter() + _OFFLOAD_POLL_INTERVAL_S

            while in_flight and in_flight[0].result is not None:
                pending = in_flight.popleft()
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
        self._lookup_progress.clear()
        self._pending_loads.clear()
        self._load_queue.join()
        self._load_results.clear()
        self._offload_queue.join()
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
