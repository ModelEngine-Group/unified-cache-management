"""Device-address checks and real IO through the production UCM linker."""

import time
import uuid

from .runner import CheckFailure, _fill_pattern


def transfer_views(group, transfers):
    """Resolve byte views using SGLang's own layer/page address mapping."""
    import torch

    views = {}
    for transfer in group.resolve_transfers(transfers):
        entry = group.entry_map[transfer.name]
        locations = entry.prepare_locations(transfer.host_indices)
        for layer in entry.layer_mapping:
            pointers, sizes, _ = entry.get_prepared_layer_range_meta(locations, layer)
            for row, lengths in zip(pointers, sizes):
                for pointer, length in zip(row, lengths):
                    pointer, length = int(pointer), int(length)
                    for buffer in entry.kv_buffer:
                        offset = pointer - buffer.data_ptr()
                        if 0 <= offset and offset + length <= buffer.nbytes:
                            if not buffer.is_contiguous():
                                raise ValueError("non-contiguous native buffer")
                            views[pointer, length] = buffer.view(torch.uint8).flatten()[
                                offset : offset + length
                            ]
                            break
                    else:
                        raise ValueError("transfer address outside registered buffers")
    return list(views.values())


def probe_linker(
    config, model, requirements, *, layer_range=None, groups=None, phase=None
):
    import torch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hicache_storage import (
        PoolHitPolicy,
        PoolName,
        PoolTransfer,
    )
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
        resolve_hybrid_device_pool_group,
    )
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils import get_device_module

    from ucm.integration.sglang.ucm_direct_linker import UcmDirectLinker
    from ucm.integration.sglang.ucm_direct_store_adapter import UcmDirectStoreAdapter

    from .linker_pools import build_linker_pools
    from .sglang_compat import environment_info

    environment = environment_info()
    device = environment["device_name"]
    if device == "cpu":
        raise CheckFailure(
            "ENVIRONMENT_UNSUPPORTED",
            "linker_pool_construction",
            "accelerator required",
        )
    phase = phase or (lambda name, action: action())
    pool, allocator, request_pool = phase(
        "linker_pool_construction",
        lambda: build_linker_pools(
            config,
            model,
            requirements,
            environment,
            layer_range,
        ),
    )
    from sglang.srt.mem_cache import static_kv_storage

    # Normal service startup binds retention before constructing the cache.
    # This probe has no draft pools and therefore no additional draft reach.
    storage_of = getattr(static_kv_storage, "kv_storage_of", None)
    if storage_of is not None:
        storage = storage_of(request_pool, allocator)
        bind_retention = getattr(storage, "bind_window_retention", None)
        if bind_retention is not None:
            bind_retention(getattr(model, "sliding_window_size", None), None)
    parallel = get_parallel()
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=config.page_size,
        pp_rank=parallel.pp_rank,
        pp_size=config.pp_size,
        tp_cache_group=groups[0].cpu_group if groups else None,
        sliding_window_size=getattr(model, "sliding_window_size", None),
    )
    components = {ComponentType.FULL}
    # Runtime-discovered pools (e.g. compressed KV) need not list "swa"
    # in the config-derived requirements, but still require its component.
    if getattr(pool, "swa_kv_pool", None) is not None:
        components.add(ComponentType.SWA)
    for name, component in (("swa", ComponentType.SWA), ("mamba", ComponentType.MAMBA)):
        if name in requirements.pool_names:
            components.add(component)

    def resolve_layout():
        try:
            return resolve_hybrid_device_pool_group(
                kvcache=pool,
                params=params,
                page_size=config.page_size,
                components=components,
            )
        except ValueError as exc:
            # Classify an explicit native support restriction, not arbitrary
            # constructor/API failures, as an incompatibility.
            if str(exc).startswith(
                (
                    "The direct external linker supports ",
                    "The direct external linker does not support ",
                )
            ):
                raise CheckFailure(
                    "LINKER_POOL_UNSUPPORTED",
                    "linker_layout",
                    f"SGLang native linker rejected the pool combination: {exc}",
                ) from exc
            raise

    group = phase("linker_layout", resolve_layout)
    layout = object.__new__(UcmDirectLinker)
    layout.pools, layout.num_layers = group.entry_map, group.num_layers
    layout._kv_allocator, layout._kv_cache = allocator, pool
    layout._bundles = layout._build_pool_bundles()
    layout._build_layer_pointer_layouts()
    metadata = layout._pool_metadata()
    report = {"pool_names": [str(name) for name in group.entry_map]}
    if config.mode == "inspect":
        return report
    identity = dict(
        model=model.model_path,
        rank_replicated=group.rank_replicated,
        tp_rank=parallel.tp_rank,
        tp_size=config.tp_size,
        pp_rank=parallel.pp_rank,
        pp_size=config.pp_size,
    )
    # Cache requires at least 2048 shards with its default load reserve.
    # Summing all entries per layer is a conservative bound for grouped stores.
    layer_bytes = {}
    for item in metadata:
        for layer, schema in item.layer_schemas:
            layer_bytes[layer] = layer_bytes.get(layer, 0) + sum(
                size for size, _ in schema
            )
    gib = 1 << 30
    cache_gib = max(1, (max(layer_bytes.values(), default=0) * 2048 + gib - 1) // gib)
    adapter = UcmDirectStoreAdapter(
        connector_name=config.connector_name,
        module_path=config.connector_module_path,
        store_config={
            "storage_backends": config.storage_backends,
            "store_pipeline": "Cache|Posix",
            "cache_buffer_capacity_gb": cache_gib,
        },
        identity=identity,
    )
    linker = None
    try:
        linker = phase(
            "linker_registration",
            lambda: UcmDirectLinker(
                None,
                params,
                components=components,
                adapter=adapter,
            ),
        )

        def register_counter():
            # Mirror service registry setup, which this standalone probe skips.
            if not callable(
                getattr(request_pool, "register_layer_transfer_counter", None)
            ):
                pool.register_layer_transfer_counter(linker.layer_done_counter)
                if PoolName.MAMBA in linker.pools:
                    pool.layer_router.load_parts_layer_wise()

        phase("linker_counter_registration", register_counter)
        prefix = uuid.uuid4().hex
        if groups:
            shared = [prefix]
            tp_group = groups[0].cpu_group
            owner = torch.distributed.get_global_rank(tp_group, 0)
            torch.distributed.broadcast_object_list(shared, src=owner, group=tp_group)
            prefix = shared[0]
        keys = [f"linker-check-{prefix}-{i}" for i in range(config.pages)]
        transfers = []
        for source in sorted(set(group.sources.values()), key=str):
            entry = next(
                entry
                for entry in group.entry_map.values()
                if entry.indices_from_pool == source
            )
            if source == PoolName.KV:
                count, trailing = config.pages, False
            elif source == PoolName.MAMBA:
                count, trailing = 1, True
            elif source == PoolName.SWA:
                window = getattr(params, "swa_retain_window", None)
                if window is None:
                    window = params.sliding_window_size
                if window is None or window <= 0:
                    raise CheckFailure(
                        "CHECKER_RUNTIME_POOL_PROBE_REQUIRED",
                        "linker_transfers",
                        "Native SWA retention window is unavailable",
                    )
                count = min(
                    config.pages, (window + config.page_size - 1) // config.page_size
                )
                trailing = True
            else:
                raise CheckFailure(
                    "CHECKER_RUNTIME_POOL_PROBE_REQUIRED",
                    "linker_transfers",
                    f"No probe transfer policy for native source {source}",
                )
            indices = torch.arange(
                entry.page_size, (count + 1) * entry.page_size, device=device
            )
            transfers.append(
                PoolTransfer(
                    name=source,
                    keys=keys[-count:],
                    device_indices=indices,
                    hit_policy=(
                        PoolHitPolicy.TRAILING_PAGES
                        if trailing
                        else PoolHitPolicy.ALL_PAGES
                    ),
                )
            )
        expected = (
            [config.pages]
            if any(t.name != PoolName.KV for t in transfers)
            else list(range(1, config.pages + 1))
        )
        views = phase("linker_addresses", lambda: transfer_views(group, transfers))
        if not views:
            raise CheckFailure(
                "CHECKER_RUNTIME_POOL_PROBE_REQUIRED",
                "linker_addresses",
                "no device payload resolved",
            )
        total = sum(view.nbytes for view in views)
        if total * 2 > config.max_host_mib * 1024 * 1024:
            raise CheckFailure(
                "CHECKER_MEMORY_LIMIT",
                "linker_snapshot",
                f"snapshot/compare requires {total * 2} bytes",
            )

        def query(rid, wanted):
            actual = phase("linker_local_lookup", lambda: linker.lookup(rid, transfers))
            mask = torch.tensor(
                [int(i in actual) for i in range(config.pages + 1)], dtype=torch.int
            )
            if groups:
                for group_ in groups:
                    torch.distributed.all_reduce(
                        mask, op=torch.distributed.ReduceOp.MIN, group=group_.cpu_group
                    )
            found = mask.nonzero().flatten().tolist()
            if found != wanted:
                raise CheckFailure(
                    "LOOKUP_CONTRACT_MISMATCH",
                    "linker_lookup",
                    f"actual={found}, expected={wanted}",
                )

        phase("linker_cold_lookup", lambda: query("cold", []))
        for index, view in enumerate(views):
            pattern = torch.empty(view.numel(), dtype=torch.uint8)
            rank_seed = 0 if group.rank_replicated else parallel.tp_rank * 100003
            _fill_pattern(
                pattern, seed=20260909 + index + rank_seed + parallel.pp_rank * 1000003
            )
            view.copy_(pattern)
        snapshots = [view.cpu() for view in views]
        get_device_module().synchronize()

        def offload():
            if not linker.offload(transfers):
                raise RuntimeError("offload rejected")
            deadline = time.monotonic() + config.distributed_timeout
            while not linker.num_completed_offloads():
                if time.monotonic() >= deadline:
                    raise TimeoutError("offload timeout")
                time.sleep(0.01)
            if not linker.pop_completed_offload():
                raise RuntimeError("offload failed")

        phase("linker_offload", offload)
        phase("linker_warm_lookup", lambda: query("warm", expected))
        # Primary-prefix holes must invalidate a tail checkpoint too.
        primary = next(t for t in transfers if t.name == PoolName.KV)
        original_keys = primary.keys
        primary.keys = [f"missing-{prefix}"] + original_keys[1:]
        try:
            phase("linker_missing_primary", lambda: query("missing", []))
        finally:
            primary.keys = original_keys
        for view in views:
            view.zero_()

        def load():
            if not linker.load("restore", transfers):
                raise RuntimeError("load rejected")
            batch = linker.start_layer_wise_loading()
            if linker._requires_load_finish:
                if not linker.finish_layer_wise_loading(batch):
                    raise RuntimeError("load failed")
            else:
                deadline = time.monotonic() + config.distributed_timeout
                while not linker.num_completed_loads():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("load timeout")
                    time.sleep(0.01)
                linker.pop_completed_load()
            get_device_module().synchronize()
            for view, snapshot in zip(views, snapshots):
                if not torch.equal(view.cpu(), snapshot):
                    raise CheckFailure(
                        "DATA_MISMATCH", "linker_load", "device bytes differ"
                    )

        phase("linker_load_compare", load)
        report.update(
            bytes_compared=total,
            checks=[
                "cold_miss",
                "offload",
                "joint_lookup",
                "missing_primary",
                "layerwise_load",
                "device_data_compare",
            ],
        )
        return report
    finally:
        if linker is not None:
            linker.close()
        else:
            adapter.close()
