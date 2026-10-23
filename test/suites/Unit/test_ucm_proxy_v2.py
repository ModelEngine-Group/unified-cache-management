"""Store composition, positional routing and connector-v2 integration tests."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from test_ucm_connector_v2 import (
    ByteMemory,
    FakeRequest,
    FakeTensor,
    FullAttentionSpec,
    KVCacheSpecKind,
    KVConnectorRole,
    MemoryByteAccess,
    config,
    group,
    make_dispatcher,
    make_request_hasher,
    parse_spec,
    plan_token_starts,
    request_ucm_ids,
    ucm_ids,
    vllm_config,
)

from ucm.integration.vllm.v2.factory import create_proxy, create_store
from ucm.integration.vllm.v2.layout import (
    UCMKVCacheGroupSpec,
    UCMKVCacheLayout,
    UCMKVCacheSpec,
)
from ucm.integration.vllm.v2.parallel import ParallelLayout
from ucm.integration.vllm.v2.test_stores import FileTestStore, MemoryTestStore
from ucm.integration.vllm.v2.ucm_chunk_id import ShardKey, make_local_ucm_chunk_ids
from ucm.integration.vllm.v2.ucm_connector import UCMConnector
from ucm.integration.vllm.v2.ucm_kv_cache import UCMTransferBuilder
from ucm.integration.vllm.v2.ucm_lookup import UCMLookup
from ucm.integration.vllm.v2.ucm_proxy import (
    ParallelProxy,
    PipelineProxy,
    UCMProxyAdapter,
    UCMProxyError,
    UCMProxyTransfer,
)
from ucm.integration.vllm.v2.ucm_scheduler import (
    AttentionType,
    RequestDispatchMeta,
    UCMConnectorMetadata,
    UCMDispatcher,
    UCMGroupDispatchPlan,
)
from ucm.integration.vllm.v2.ucm_sharding import UCMSharding


def transfer(
    keys,
    indices,
    ptrs,
    *,
    offsets=None,
    sizes=None,
    record_size=8,
    attention_type=AttentionType.FA,
):
    ptrs = np.asarray(ptrs, dtype=np.uint64)
    offsets = (
        np.zeros_like(ptrs) if offsets is None else np.asarray(offsets, dtype=np.uint64)
    )
    sizes = (
        np.full_like(ptrs, record_size)
        if sizes is None
        else np.asarray(sizes, dtype=np.uint64)
    )
    return UCMProxyTransfer(
        tuple(keys),
        np.asarray(indices, dtype=np.int64),
        ptrs,
        sizes,
        offsets,
        tuple(
            json.dumps({"type": attention_type.value, "chunk_index": i})
            for i in indices
        ),
        np.full(len(keys), record_size, dtype=np.uint64),
    )


class DeferredTestStore:
    """Delay Store IO until wait, with optional submission/completion failures."""

    backend = None

    def __init__(self, store):
        self.store = store
        self.backend = store.backend
        self.fail_submit = False
        self.fail_wait = False
        self.waited = 0

    def __getattr__(self, name):
        return getattr(self.store, name)

    def dump(self, *args):
        if self.fail_submit:
            raise OSError("submit failure")
        return ("dump", args)

    def load(self, *args):
        return ("load", args)

    def commit(self, *args):
        return ("commit", args)

    def wait(self, task):
        self.waited += 1
        if self.fail_wait:
            raise OSError("wait failure")
        operation, arguments = task
        getattr(self.store, operation)(*arguments)


@pytest.mark.parametrize("mode", ["pipeline", "parallel"])
@pytest.mark.parametrize("store_kind", ["file", "memory", "deferred"])
def test_layerwise_roundtrip_through_real_proxy(tmp_path, mode, store_kind):
    memory = ByteMemory()
    access = MemoryByteAccess(memory)
    stores = []
    for i in range(2):
        store = (
            FileTestStore(tmp_path / str(i), access)
            if store_kind == "file"
            else MemoryTestStore(access)
        )
        stores.append(DeferredTestStore(store) if store_kind == "deferred" else store)
    proxy = (
        PipelineProxy(stores[0])
        if mode == "pipeline"
        else ParallelProxy(stores, [4096, None], 512)
    )
    adapter = UCMProxyAdapter(proxy)
    keys = (b"a" * 16, b"b" * 16, b"c" * 16)
    source = np.array([[100, 104], [200, 204], [300, 304]], dtype=np.uint64)
    expected = (b"abcdefgh", b"ijklmnop", b"qrstuvwx")
    for ptr, payload in zip(source[:, 0], expected):
        memory.write(ptr, payload)
    full = transfer(keys, [7, 8, 9], source, offsets=[[0, 4]] * 3, sizes=[[4, 4]] * 3)
    for column in range(2):
        layer = replace(
            full,
            ptrs=full.ptrs[:, column : column + 1],
            sizes=full.sizes[:, column : column + 1],
            ucm_chunk_offsets=full.ucm_chunk_offsets[:, column : column + 1],
        )
        for array in (
            layer.ptrs,
            layer.sizes,
            layer.ucm_chunk_offsets,
            layer.chunk_indices,
        ):
            array.flags.writeable = False
        adapter.submit("dump", layer)
        assert not any(adapter.lookup(keys, full.chunk_indices))
    adapter.commit(keys, full.chunk_indices)
    assert adapter.lookup(keys, full.chunk_indices) == (True, True, True)
    # Mixed order must restore rows after splitting routes.
    load = transfer((keys[1], keys[0], keys[2]), [8, 7, 9], [[400], [500], [600]])
    adapter.submit("load", load)
    assert [memory.read(p, 8) for p in (500, 400, 600)] == list(expected)
    adapter.close()


def test_pipeline_writeback_eviction_and_complete_record_promotion(tmp_path):
    memory = ByteMemory()
    access = MemoryByteAccess(memory)
    lower = FileTestStore(tmp_path, access)
    middle = MemoryTestStore(access, backend=lower, capacity_bytes=8)
    upper = MemoryTestStore(access, backend=middle, capacity_bytes=8)
    adapter = UCMProxyAdapter(PipelineProxy(upper))
    a, b = b"a" * 16, b"b" * 16
    for key, ptr, data, index in ((a, 100, b"abcdefgh", 7), (b, 200, b"ijklmnop", 8)):
        memory.write(ptr, data)
        batch = transfer((key,), [index], [[ptr]], attention_type=AttentionType.STATE)
        adapter.submit("dump", batch)
        assert lower.lookup((key,)) == (False,)
        adapter.commit(batch.ucm_chunk_ids, batch.chunk_indices)
        assert lower.read_record(key) == (data, batch.metadata[0])
    assert a not in upper.records and a not in middle.records
    adapter.submit("load", transfer((a,), [7], [[300]], offsets=[[4]], sizes=[[4]]))
    assert memory.read(300, 4) == b"efgh"
    assert upper.records[a][0] == b"abcdefgh"
    # An upper hit must not read from the lower Store.
    with patch.object(middle, "read_record", side_effect=AssertionError("lower read")):
        adapter.submit("load", transfer((a,), [7], [[400]]))
    assert memory.read(400, 8) == b"abcdefgh"
    adapter.close()


def test_failed_writeback_pins_source_and_flush_retries(tmp_path):
    memory = ByteMemory()
    access = MemoryByteAccess(memory)
    lower = FileTestStore(tmp_path, access)
    upper = MemoryTestStore(access, backend=lower, capacity_bytes=8)
    adapter = UCMProxyAdapter(PipelineProxy(upper))
    key = b"a" * 16
    memory.write(100, b"abcdefgh")
    batch = transfer((key,), [0], [[100]])
    adapter.submit("dump", batch)
    with patch.object(lower, "write_record", side_effect=OSError("disk failure")):
        with pytest.raises(UCMProxyError, match="commit"):
            adapter.commit(batch.ucm_chunk_ids, batch.chunk_indices)
        memory.write(100, b"XXXXXXXX")
        with pytest.raises(MemoryError):
            upper.write_record(b"b" * 16, b"ijklmnop", "{}")
        with pytest.raises(OSError, match="disk failure"):
            adapter.close()
        assert upper.records[key][0] == b"abcdefgh"
    adapter.flush()
    assert lower.read_record(key)[0] == b"abcdefgh"
    upper.write_record(b"b" * 16, b"ijklmnop", "{}")
    assert key not in upper.records
    adapter.close()


def test_parallel_never_falls_back_or_spills_to_other_route(tmp_path):
    memory = ByteMemory()
    access = MemoryByteAccess(memory)
    cache = MemoryTestStore(access, capacity_bytes=8)
    disk = FileTestStore(tmp_path, access)
    key = b"a" * 16
    disk.write_record(key, b"wrongway", "{}")
    proxy = ParallelProxy((cache, disk), [4096, None], 512)
    with (
        patch.object(disk, "lookup", side_effect=AssertionError("wrong route")),
        patch.object(disk, "load", side_effect=AssertionError("wrong route")),
        patch.object(disk, "dump", side_effect=AssertionError("wrong route")),
    ):
        assert proxy.lookup((key,), np.array([7])) == (False,)
        with pytest.raises(KeyError):
            proxy.load(
                (key,),
                np.array([7]),
                np.array([[0]]),
                np.array([[100]]),
                np.array([[8]]),
            )
        adapter = UCMProxyAdapter(proxy)
        memory.write(100, b"abcdefgh")
        batch = transfer((key,), [7], [[100]])
        adapter.submit("dump", batch)
        # Pending reservations consume capacity without becoming visible.
        with pytest.raises(UCMProxyError):
            adapter.submit("dump", transfer((b"b" * 16,), [6], [[100]]))
        adapter.abort(batch.ucm_chunk_ids, batch.chunk_indices)
        assert cache.lookup((key,)) == (False,)
    assert disk.read_record(key)[0] == b"wrongway"
    proxy.close()


def test_parallel_scans_use_original_order_and_absolute_indices():
    stores = [MemoryTestStore(MemoryByteAccess(ByteMemory())) for _ in range(2)]
    keys = tuple(bytes([i]) * 16 for i in range(4))
    indices = np.array([8, 6, 9, 7])
    stores[1].write_record(keys[0], b"", "{}")
    stores[0].write_record(keys[1], b"", "{}")
    stores[0].write_record(keys[3], b"", "{}")
    proxy = ParallelProxy(stores, [4096, None], 512)
    assert proxy.lookup(keys, indices) == (True, True, False, True)
    assert proxy.lookup_on_prefix(keys, indices) == 1
    assert proxy.lookup_on_reverse(keys, indices) == 3
    assert proxy.lookup_on_prefix(keys[1:], indices[1:]) == 0
    assert proxy.lookup((), np.array([], dtype=np.int64)) == ()


@pytest.mark.parametrize("failure", ["submit", "wait"])
def test_parallel_drains_other_route_when_one_fails(failure):
    access = MemoryByteAccess(ByteMemory())
    stores = [DeferredTestStore(MemoryTestStore(access)) for _ in range(2)]
    if failure == "submit":
        stores[1].fail_submit = True
    else:
        stores[0].fail_wait = True
    adapter = UCMProxyAdapter(ParallelProxy(stores, [4096, None], 512))
    batch = transfer((b"a" * 16, b"b" * 16), [7, 8], [[100], [200]])
    with pytest.raises(UCMProxyError):
        adapter.submit("dump", batch)
    assert stores[0].waited == 1
    assert stores[1].waited == (0 if failure == "submit" else 1)
    adapter.abort(batch.ucm_chunk_ids, batch.chunk_indices)
    assert not any(store.store._pending for store in stores)


@pytest.mark.parametrize(
    "indices",
    [np.array([-1]), np.array([1.5]), np.array([[0]]), np.array([], dtype=int)],
)
def test_invalid_positions_fail_before_store_io(indices):
    store = MemoryTestStore(MemoryByteAccess(ByteMemory()))
    adapter = UCMProxyAdapter(PipelineProxy(store))
    batch = replace(transfer((b"a" * 16,), [0], [[100]]), chunk_indices=indices)
    with pytest.raises(UCMProxyError):
        adapter.submit("dump", batch)
    assert store._pending == {}


@pytest.mark.parametrize(
    "ends", [[1, None], [0, None], [4096, 4096, None], [4096], [None, None]]
)
def test_parallel_rejects_invalid_thresholds(ends):
    stores = [MemoryTestStore() for _ in ends]
    with pytest.raises(ValueError):
        ParallelProxy(stores, ends, 512)


def test_parallel_rejects_store_with_backend():
    cache = MemoryTestStore(backend=MemoryTestStore())
    with pytest.raises(ValueError, match="backend=None"):
        ParallelProxy([cache], [None], 512)


@pytest.mark.parametrize("mode", ["pipeline", "parallel"])
def test_factory_one_proxy_per_process_ranks_differ_only_by_id(tmp_path, mode):
    settings = {
        "v2_stores": {
            "cache": {"type": "test_memory", "capacity_bytes": 8},
            "left": {"type": "test_file", "path": str(tmp_path / "left")},
            "right": {"type": "test_file", "path": str(tmp_path / "right")},
        },
        "v2_proxy": (
            {"type": "pipeline", "stores": ["cache", "left"]}
            if mode == "pipeline"
            else {
                "type": "parallel",
                "routes": [
                    {"token_end": 4096, "store": "left"},
                    {"token_end": None, "store": "right"},
                ],
            }
        ),
    }
    # Separate processes in production: each builds the same single Proxy.
    scheduler = create_proxy(settings, record_token_span=512)
    writers = [create_proxy(settings, record_token_span=512) for _ in range(2)]
    common_ids = (b"a" * 12, b"b" * 12)
    # Two TP ranks with distinct heads: one shard each, same Stores.
    ids = [
        make_local_ucm_chunk_ids(common_ids, AttentionType.FA, 0, ShardKey(head_key=tp))
        for tp in range(2)
    ]
    every_rank = (*ids[0], *ids[1])
    for i, writer in enumerate(writers):
        proxy = writer._proxy
        stores = (proxy.store,) if mode == "pipeline" else proxy.stores
        memory = ByteMemory()
        for store in stores:
            store.byte_access = MemoryByteAccess(memory)
        memory.write(100, bytes([65 + i]) * 8)
        batch = transfer(ids[i], [7, 8], [[100], [100]])
        if mode == "pipeline":
            # Cache capacity holds one record; commit each before reserving the next.
            for row in range(2):
                one = transfer((ids[i][row],), [7 + row], [[100]])
                writer.submit("dump", one)
                writer.commit(one.ucm_chunk_ids, one.chunk_indices)
        else:
            writer.submit("dump", batch)
            writer.commit(ids[i], batch.chunk_indices)
        assert scheduler.lookup(every_rank, np.tile(batch.chunk_indices, 2)) == (
            (True, True, False, False) if i == 0 else (True,) * 4
        )
        writer.submit("load", transfer(ids[i], [7, 8], [[200], [300]]))
        assert memory.read(200, 8) == bytes([65 + i]) * 8
        assert memory.read(300, 8) == bytes([65 + i]) * 8
    for writer in writers:
        writer.close()
    scheduler.close()


def test_factory_default_store_uses_its_configured_path(tmp_path):
    adapter = create_proxy({"v2_storage_path": str(tmp_path)}, record_token_span=512)
    assert adapter._proxy.store.root == tmp_path
    adapter.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"v2_stores": {}, "v2_proxy": {"type": "unknown"}},
        {"v2_stores": {}, "v2_proxy": {"type": "pipeline", "stores": ["missing"]}},
        {"v2_stores": {}, "v2_proxy": {"type": "pipeline", "stores": []}},
        {
            "v2_stores": {"cache": {"type": "test_memory"}},
            "v2_proxy": {"type": "pipeline", "stores": ["cache"]},
        },
        {
            "v2_stores": {"cache": {"type": "test_memory"}},
            "v2_proxy": {
                "type": "parallel",
                "routes": [{"token_end": None, "store": "cache"}],
            },
        },
        {
            "v2_stores": {
                "left": {"type": "test_file", "path": "same"},
                "right": {"type": "test_file", "path": "same"},
            },
            "v2_proxy": {
                "type": "parallel",
                "routes": [
                    {"token_end": 4096, "store": "left"},
                    {"token_end": None, "store": "right"},
                ],
            },
        },
    ],
)
def test_factory_rejects_invalid_or_process_local_configuration(settings):
    with pytest.raises(ValueError):
        create_proxy(settings, record_token_span=512)


def test_all_shard_lookup_accepts_hits_at_different_store_levels(tmp_path):
    access = MemoryByteAccess(ByteMemory())
    disk = FileTestStore(tmp_path, access)
    cache = MemoryTestStore(access, backend=disk)
    group_info = UCMKVCacheGroupSpec(
        0, 512, frozenset((KVCacheSpecKind.FULL_ATTENTION,))
    )
    spec = UCMKVCacheSpec((group_info,), 512, 512, "cpu")
    lookup = UCMLookup(
        spec,
        UCMProxyAdapter(PipelineProxy(cache)),
        UCMSharding(spec, ParallelLayout(tp=2)),
    )
    common_ids = (b"a" * 12,)
    (shard0,) = make_local_ucm_chunk_ids(
        common_ids, AttentionType.FA, 0, ShardKey(head_key=0)
    )
    (shard1,) = make_local_ucm_chunk_ids(
        common_ids, AttentionType.FA, 0, ShardKey(head_key=1)
    )
    cache.records[shard0] = (b"abcdefgh", "{}")
    assert lookup._prefix_hit_end_chunk(AttentionType.FA, common_ids, 0, 1) == 0
    disk.write_record(shard1, b"ijklmnop", "{}")
    assert lookup._prefix_hit_end_chunk(AttentionType.FA, common_ids, 0, 1) == 1
    cache.close()


def test_scheduler_sliced_lookup_and_wa_state_route_by_boundary():
    kinds = (
        KVCacheSpecKind.FULL_ATTENTION,
        KVCacheSpecKind.SLIDING_WINDOW,
        KVCacheSpecKind.MAMBA,
    )
    groups = tuple(
        UCMKVCacheGroupSpec(
            i,
            512,
            frozenset((kind,)),
            tail_tokens=512 if i == 1 else None,
        )
        for i, kind in enumerate(kinds)
    )
    spec = UCMKVCacheSpec(groups, 512, 512, "cpu")
    stores = [MemoryTestStore(MemoryByteAccess(ByteMemory())) for _ in range(2)]
    proxy = ParallelProxy(stores, [4096, None], 512)
    dispatcher = UCMDispatcher(
        spec,
        UCMLookup(
            spec,
            UCMProxyAdapter(proxy),
            UCMSharding(spec, ParallelLayout()),
            recompute_tokens=0,
        ),
        make_request_hasher(vllm_config(), 0),
        b"seed",
    )
    request = FakeRequest("request", 5120)
    ucm_chunk_ids_by_attention_type = request_ucm_ids(dispatcher, request)
    for ucm_chunk_ids in ucm_chunk_ids_by_attention_type:
        for index, ucm_chunk_id in enumerate(ucm_chunk_ids):
            stores[0 if index < 8 else 1].write_record(ucm_chunk_id, b"", "{}")
    assert dispatcher.lookup(request, 3072).external_hit_tokens == 2048
    # Latest State exists only in the wrong route. The common boundary backs up.
    stores[1].records.pop(ucm_chunk_ids_by_attention_type[2][9])
    stores[0].write_record(ucm_chunk_ids_by_attention_type[2][9], b"", "{}")
    assert dispatcher.lookup(request, 3072).external_hit_tokens == 1536
    stores[1].records.pop(ucm_chunk_ids_by_attention_type[1][8])
    assert dispatcher.lookup(request, 3072).external_hit_tokens == 1024


class NativeTensor(FakeTensor):
    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return int(np.prod(self.shape))

    def stride(self, index=None):
        return self._strides if index is None else self._strides[index]


def make_layout():
    names = ("model.layers.0.attn", "model.layers.1.attn")
    native_group = group(names, FullAttentionSpec(4))
    native_group.is_eagle_group = False
    native = config(native_group)
    native.kv_cache_layout = "NHD"
    spec = parse_spec(
        native,
        scheduler_block_size=4,
        hf_text_config=SimpleNamespace(model_type="llama"),
        device_type="cpu",
    )
    caches = {
        name: NativeTensor(1000 + i * 1000, (8, 1, 4, 2), (8, 8, 2, 1))
        for i, name in enumerate(names)
    }
    module = SimpleNamespace(
        KVCacheLayout={
            "NHD": SimpleNamespace(is_block_outermost=False, is_block_compact=False)
        }
    )
    with patch.dict(sys.modules, {"vllm.v1.kv_cache_layout": module}):
        layout = UCMKVCacheLayout(
            spec,
            caches,
            kv_cache_config=native,
            num_hidden_layers=2,
            use_layerwise=True,
        )
    return layout, caches, native


def test_transfer_builder_keeps_absolute_positions_and_complete_record_size():
    layout, _, _ = make_layout()
    builder = UCMTransferBuilder(layout)
    keys = (b"a" * 12, b"b" * 12)
    plan = UCMGroupDispatchPlan(
        AttentionType.FA,
        keys,
        plan_token_starts(12, 20, len(keys)),
        (np.array([1, 2]),),
    )
    metadata = UCMConnectorMetadata(
        requests={"r": RequestDispatchMeta("r", dump_plans=(plan,), load_plans=(plan,))}
    )
    for layer in layout.layer_id_by_name:
        (dump,) = builder.build_dump_transfers(metadata, layer_name=layer)
        (load,) = builder.build_load_transfers(metadata, layer_name=layer)
        np.testing.assert_array_equal(dump.chunk_indices, [3, 4])
        np.testing.assert_array_equal(load.chunk_indices, [3, 4])
        np.testing.assert_array_equal(dump.record_nbytes, [16, 16])
        assert [json.loads(item)["chunk_index"] for item in dump.metadata] == [3, 4]
        assert int(dump.sizes.sum(axis=1)[0]) == 8
        assert load.record_nbytes is None


@pytest.mark.parametrize("layerwise", [False, True])
def test_connector_save_commits_to_original_route_and_shutdown_drains(layerwise):
    layout, _, _ = make_layout()
    memory = ByteMemory()
    memory.write(1008, b"abcdefgh")
    memory.write(2008, b"ijklmnop")
    stores = [
        DeferredTestStore(MemoryTestStore(MemoryByteAccess(memory))) for _ in range(2)
    ]
    adapter = UCMProxyAdapter(ParallelProxy(stores, [12, None], 4))
    worker = UCMConnector.__new__(UCMConnector)
    worker.layout = layout
    worker.spec = layout.spec
    worker.transfer_builder = UCMTransferBuilder(layout)
    worker._proxy = adapter
    worker.use_layerwise = layerwise
    worker._layer_load_tasks = {}
    worker._dump_tasks = []
    worker._saved_layer_names = set()
    worker._save_error = None
    worker._failed_load_reqs = set()
    worker._save_complete = False
    plan = UCMGroupDispatchPlan(
        AttentionType.FA, (b"a" * 12,), plan_token_starts(12, 16, 1), (np.array([1]),)
    )
    worker._connector_metadata = UCMConnectorMetadata(
        requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
    )
    worker.wait_for_save()
    ucm_chunk_ids = ucm_ids(plan.common_ucm_chunk_ids)
    assert stores[0].store.lookup(ucm_chunk_ids) == (False,)
    assert stores[1].store.lookup(ucm_chunk_ids) == (True,)
    assert stores[1].store.records[ucm_chunk_ids[0]][0] == b"abcdefghijklmnop"
    worker.shutdown()


def test_empty_pp_marker_carries_position_and_commits_only_its_route():
    group_info = UCMKVCacheGroupSpec(0, 512, frozenset((KVCacheSpecKind.MAMBA,)))
    spec = UCMKVCacheSpec((group_info,), 512, 512, "cpu")
    stores = [MemoryTestStore(MemoryByteAccess(ByteMemory())) for _ in range(2)]
    worker = UCMConnector.__new__(UCMConnector)
    worker.spec = spec
    worker.layout = SimpleNamespace(spec=spec, group_layouts={})
    worker.transfer_builder = UCMTransferBuilder(worker.layout)
    worker._proxy = UCMProxyAdapter(ParallelProxy(stores, [4096, None], 512))
    plan = UCMGroupDispatchPlan(
        AttentionType.STATE,
        (b"s" * 12,),
        plan_token_starts(4096, 4608, 1),
        (np.array([], dtype=np.uint64),),
    )
    metadata = UCMConnectorMetadata(
        requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
    )
    (batch,) = worker._dump_empty_shards(metadata)
    np.testing.assert_array_equal(batch.chunk_indices, [8])
    ucm_chunk_ids = ucm_ids(plan.common_ucm_chunk_ids, AttentionType.STATE)
    assert batch.ucm_chunk_ids == ucm_chunk_ids
    assert stores[1].lookup(ucm_chunk_ids) == (False,)
    worker._proxy.commit(batch.ucm_chunk_ids, batch.chunk_indices)
    assert stores[1].records[ucm_chunk_ids[0]][0] == b""
    assert stores[0].lookup(ucm_chunk_ids) == (False,)


def test_reused_committed_record_is_pinned_until_its_commit():
    memory = ByteMemory()
    store = MemoryTestStore(MemoryByteAccess(memory), capacity_bytes=8)
    key = b"a" * 16
    store.write_record(key, b"abcdefgh", "{}")
    adapter = UCMProxyAdapter(PipelineProxy(store))
    batch = transfer((key,), [0], [[100]])
    adapter.submit("dump", batch)
    with pytest.raises(MemoryError):
        store.write_record(b"b" * 16, b"ijklmnop", "{}")
    adapter.commit(batch.ucm_chunk_ids, batch.chunk_indices)
    store.write_record(b"b" * 16, b"ijklmnop", "{}")
    assert store.lookup((key,)) == (False,)


@pytest.mark.parametrize("layerwise", [False, True])
def test_connector_failed_dump_drains_and_aborts_pending(layerwise):
    layout, _, _ = make_layout()
    stores = [
        DeferredTestStore(MemoryTestStore(MemoryByteAccess(ByteMemory())))
        for _ in range(2)
    ]
    stores[1].fail_submit = True
    worker = UCMConnector.__new__(UCMConnector)
    worker.layout = layout
    worker.spec = layout.spec
    worker.transfer_builder = UCMTransferBuilder(layout)
    worker._proxy = UCMProxyAdapter(ParallelProxy(stores, [12, None], 4))
    worker.use_layerwise = layerwise
    worker._layer_load_tasks = {}
    worker._dump_tasks = []
    worker._saved_layer_names = set()
    worker._save_error = None
    worker._failed_load_reqs = set()
    worker._save_complete = False
    plan = UCMGroupDispatchPlan(
        AttentionType.FA,
        (b"a" * 12, b"b" * 12),
        plan_token_starts(8, 16, 2),
        (np.array([1, 2]),),
    )
    worker._connector_metadata = UCMConnectorMetadata(
        requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
    )
    with pytest.raises(UCMProxyError):
        worker.wait_for_save()
    assert stores[0].waited == 1
    assert not any(store.store._pending for store in stores)
    assert not worker._dump_tasks
    worker.shutdown()


def test_close_drains_all_routes_even_if_one_close_fails():
    stores = [MemoryTestStore(), MemoryTestStore()]
    proxy = ParallelProxy(stores, [512, None], 512)
    with (
        patch.object(stores[0], "close", side_effect=OSError("close failure")),
        patch.object(stores[1], "close") as close,
    ):
        with pytest.raises(OSError, match="close failure"):
            proxy.close()
        close.assert_called_once()


def test_parallel_retains_sliced_arrays_until_wait():
    import weakref

    class WeakTaskStore(DeferredTestStore):
        def dump(self, *arguments):
            return tuple(
                weakref.ref(value) if isinstance(value, np.ndarray) else value
                for value in arguments
            )

        def wait(self, task):
            arguments = tuple(
                value() if isinstance(value, weakref.ReferenceType) else value
                for value in task
            )
            assert all(value is not None for value in arguments)
            self.store.dump(*arguments)

    stores = [
        WeakTaskStore(MemoryTestStore(MemoryByteAccess(ByteMemory()))) for _ in range(2)
    ]
    adapter = UCMProxyAdapter(ParallelProxy(stores, [512, None], 512))
    task = adapter.enqueue(
        "dump", transfer((b"a" * 16, b"b" * 16), [0, 1], [[100], [200]])
    )
    adapter.wait(task)
    assert all(store.store._pending for store in stores)
    adapter.close()


@pytest.mark.parametrize("role", [KVConnectorRole.WORKER, KVConnectorRole.SCHEDULER])
def test_connector_creation_is_delegated_to_factory(role):
    _, _, native = make_layout()
    cfg = vllm_config()
    cfg.instance_id = "engine"
    cfg.use_v2_model_runner = False
    cfg.model_config.hf_text_config = SimpleNamespace(
        model_type="llama",
        num_hidden_layers=2,
        to_dict=lambda: {"model_type": "llama", "num_hidden_layers": 2},
    )
    cfg.cache_config.cache_dtype = "auto"
    cfg.cache_config.mamba_cache_dtype = "auto"
    cfg.cache_config.mamba_ssm_cache_dtype = "auto"
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.parallel_config.decode_context_parallel_size = 1
    cfg.parallel_config.cp_kv_cache_interleave_size = 1
    cfg.parallel_config.world_size = 2
    adapter = UCMProxyAdapter(PipelineProxy(MemoryTestStore()))
    with (
        patch("ucm.integration.vllm.v2.ucm_connector._worker_rank", return_value=1),
        patch(
            "ucm.integration.vllm.v2.ucm_connector._worker_parallel_ranks",
            return_value=(1, 0, 0),
        ),
        patch(
            "ucm.integration.vllm.v2.ucm_connector.create_proxy", return_value=adapter
        ) as factory,
    ):
        connector = UCMConnector(cfg, role, native)
    assert connector._proxy is adapter
    assert factory.call_args.kwargs == {"record_token_span": 4}
    assert connector.parallel_layout == ParallelLayout(tp=2)
    if role == KVConnectorRole.WORKER:
        assert connector._parallel_rank == (1, 0, 0)
        # Two TP ranks with distinct heads: no replicas to share writes.
        assert set(connector._replica_index_and_count_by_class.values()) == {(0, 1)}
    else:
        assert connector._parallel_rank == (0, 0, 0)
        assert connector.dispatcher._lookup._record_shards
    connector.shutdown()


def test_late_load_failure_discards_already_enqueued_layer_dumps():
    layout, _, _ = make_layout()
    store = DeferredTestStore(MemoryTestStore(MemoryByteAccess(ByteMemory())))
    worker = UCMConnector.__new__(UCMConnector)
    worker.layout = layout
    worker.spec = layout.spec
    worker.transfer_builder = UCMTransferBuilder(layout)
    worker._proxy = UCMProxyAdapter(PipelineProxy(store))
    worker.use_layerwise = True
    worker._layer_load_tasks = {}
    worker._dump_tasks = []
    worker._saved_layer_names = set()
    worker._save_error = None
    worker._failed_load_reqs = set()
    worker._save_complete = False
    plan = UCMGroupDispatchPlan(
        AttentionType.FA, (b"a" * 12,), plan_token_starts(12, 16, 1), (np.array([1]),)
    )
    metadata = UCMConnectorMetadata(
        requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
    )
    worker._connector_metadata = metadata
    worker._save_layer("model.layers.0.attn", metadata)
    worker._failed_load_reqs.add("r")
    worker.wait_for_save()
    assert store.waited == 1
    assert not store.store._pending
    assert store.store.lookup(ucm_ids(plan.common_ucm_chunk_ids)) == (False,)
    worker.shutdown()
