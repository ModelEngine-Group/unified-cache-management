"""Run with a real SGLang installation providing DevicePoolEntry (0.5.20+).

For backported releases, set ENABLE_UCM_PATCH=1 before starting pytest.
This checks the external layout API using CPU tensors; no model or GPU is needed.
"""

import types

import numpy as np
import pytest


@pytest.mark.stage(0)
@pytest.mark.platform("cpu")
@pytest.mark.feature("sglang_linker")
@pytest.mark.parametrize("rows_are_pages", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_sglang_device_pool_layout_contract(rows_are_pages, packed):
    import torch
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry

    from ucm.integration.sglang.ucm_direct_linker import UcmDirectLinker

    page_size, rows = 4, 12
    components = [
        [torch.empty((rows, 3 * (i % 2 + 1), 5), dtype=torch.float16) for i in range(4)]
    ]
    if packed:
        components.append(
            [torch.empty((rows, i + 1, 7), dtype=torch.uint8) for i in range(4)]
        )
    mapping = {1: 2, 3: (0, 1), 5: 3} if packed else {0: 0, 2: 2}
    entry = DevicePoolEntry(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        device_pool=None,
        components=components,
        layer_mapping=mapping,
        page_size=page_size,
        rows_are_pages=rows_are_pages,
        packed=packed,
    )
    linker = object.__new__(UcmDirectLinker)
    linker.pools = {PoolName.KV: entry}
    linker.num_layers = 6
    linker._kv_allocator = types.SimpleNamespace()
    linker._kv_cache = types.SimpleNamespace()
    linker._bundles = linker._build_pool_bundles()
    linker._layer_pointer_layouts = linker._build_layer_pointer_layouts()
    indices = torch.arange(3 * page_size, dtype=torch.int64)
    locations = entry.prepare_locations(indices)
    for layer in mapping:
        actual, schema = linker._entry_layer_rows(
            entry, layer, np.asarray(locations, dtype=np.uint64)
        )
        ptrs, sizes, offsets = entry.get_prepared_layer_range_meta(locations, layer)
        np.testing.assert_array_equal(actual, np.asarray(ptrs, dtype=np.uint64))
        assert schema == tuple(zip(sizes[0], offsets[0]))
        assert all(tuple(row) == tuple(sizes[0]) for row in sizes)
        assert all(tuple(row) == tuple(offsets[0]) for row in offsets)
        bases, _, _ = entry.get_prepared_layer_range_meta([0], layer)
        next_rows, _, _ = entry.get_prepared_layer_range_meta([1], layer)
        cached = linker._layer_pointer_layouts[(entry.name, layer)]
        np.testing.assert_array_equal(cached.bases, bases[0])
        np.testing.assert_array_equal(
            cached.strides, np.asarray(next_rows[0]) - np.asarray(bases[0])
        )
    for metadata in linker._pool_metadata():
        assert metadata.capacity_pages == (
            rows if rows_are_pages else rows // page_size
        )
