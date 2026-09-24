from types import SimpleNamespace

import pytest
import torch

sglang_hicache = pytest.importorskip("sglang.srt.mem_cache.hicache_storage")

from ucm.integration.sglang.ucm_connector import (  # noqa: E402
    SglangUcmConnector,
    UnifiedCacheStoreConfig,
)
from ucm.integration.sglang.unifiedcache_store import UnifiedCacheStore  # noqa: E402


class FakeStore:
    def __init__(self, config):
        self.config = config
        self.objects = set()
        self.dump_calls = []
        self.load_calls = []
        self.closed = False

    def dump_data(self, keys, shard_indices, pointers):
        self.dump_calls.append((keys, shard_indices, pointers))
        self.objects.update(keys)
        return object()

    def load_data(self, keys, shard_indices, pointers):
        self.load_calls.append((keys, shard_indices, pointers))
        if not all(key in self.objects for key in keys):
            raise RuntimeError("missing object")
        return object()

    def wait(self, task):
        return None

    def lookup(self, keys):
        return [key in self.objects for key in keys]

    def lookup_on_prefix(self, keys):
        last = -1
        for index, key in enumerate(keys):
            if key not in self.objects:
                break
            last = index
        return last

    def lookup_on_reverse(self, keys):
        for index in range(len(keys) - 1, -1, -1):
            if keys[index] in self.objects:
                return index
        return -1

    def close(self):
        self.closed = True


class FakeHostPool:
    layout = "page_first"
    dtype = torch.bfloat16
    kv_buffer = object()

    def __init__(self, page_size, component_sizes):
        self.page_size = page_size
        self.component_sizes = component_sizes
        self.restored_pages = []

    def get_page_buffer_meta(self, host_indices):
        page_count = len(host_indices) // self.page_size
        pointers = []
        sizes = []
        for page in range(page_count):
            for component, size in enumerate(self.component_sizes):
                pointers.append(10_000 + page * 1_000 + component * 100)
                sizes.append(size)
        return pointers, sizes

    def get_data_page(self, offset, flat=True):
        return torch.full((sum(self.component_sizes),), offset, dtype=torch.uint8)

    def get_dummy_flat_data_page(self):
        return torch.zeros(sum(self.component_sizes), dtype=torch.uint8)

    def set_from_flat_data_page(self, offset, page):
        self.restored_pages.append((offset, page))


@pytest.fixture
def connector(monkeypatch, tmp_path):
    stores = []

    def create_connector(name, config, module_path):
        store = FakeStore(config)
        stores.append(store)
        return store

    monkeypatch.setattr(
        "ucm.integration.sglang.ucm_connector.UcmConnectorFactoryV1.create_connector",
        create_connector,
    )
    main_pool = FakeHostPool(page_size=2, component_sizes=[400])
    storage_config = SimpleNamespace(
        model_name="org/model", is_mla_model=False, tp_rank=1, tp_size=2
    )
    main_store = FakeStore({})
    value = SglangUcmConnector(main_store, main_pool, storage_config, [str(tmp_path)])
    value.ucm_store_config = UnifiedCacheStoreConfig(
        module_path="fake.module",
        name="FakePosix",
        config={"storage_backends": [str(tmp_path)]},
    )
    return value, stores


def test_v2_creates_one_fixed_size_store_per_pool(connector):
    value, stores = connector
    value.register_pool_v2(FakeHostPool(2, [64, 64]), sglang_hicache.PoolName.SWA)
    value.register_pool_v2(FakeHostPool(1, [128]), sglang_hicache.PoolName.INDEXER)

    assert [store.config["tensor_size"] for store in stores] == [64, 128]
    assert [store.config["shard_size"] for store in stores] == [128, 128]
    assert [store.config["block_size"] for store in stores] == [128, 128]
    assert "sglang_v2" in stores[0].config["storage_backends"][0]


def test_v2_rejects_unequal_non_mamba_components(connector):
    value, _ = connector

    with pytest.raises(ValueError, match="asymmetric pools are not supported"):
        value.register_pool_v2(FakeHostPool(2, [64, 96]), sglang_hicache.PoolName.SWA)


def test_logical_anchor_defers_store_creation_to_v2_pool(monkeypatch, tmp_path):
    stores = []

    def create_connector(name, config, module_path):
        store = FakeStore(config)
        stores.append(store)
        return store

    store_config = UnifiedCacheStoreConfig(
        module_path="fake.module",
        name="FakePosix",
        config={"storage_backends": [str(tmp_path)]},
    )
    monkeypatch.setattr(
        "ucm.integration.sglang.ucm_connector.UnifiedCacheStoreConfig.load_from_config",
        lambda storage_config, mem_pool_host: store_config,
    )
    monkeypatch.setattr(
        "ucm.integration.sglang.ucm_connector.UcmConnectorFactoryV1.create_connector",
        create_connector,
    )
    logical_pool = FakeHostPool(page_size=2, component_sizes=[])
    logical_pool.kv_buffer = None
    storage_config = SimpleNamespace(
        model_name="org/deepseek-v4", is_mla_model=True, tp_rank=0, tp_size=1
    )

    value = SglangUcmConnector.from_hicache(storage_config, logical_pool)

    assert value.store is None
    assert stores == []
    assert value.batch_set_v1(["page-0"], torch.tensor([0, 1])) == [True]
    assert value.batch_get_v1(["page-0"], torch.tensor([0, 1])) == [True]
    assert value.exists("page-0")
    assert value.batch_exists(["page-0", "page-1"]) == 2
    value.register_pool_v2(
        FakeHostPool(2, [128]), sglang_hicache.PoolName.DEEPSEEK_V4_C4
    )
    assert len(stores) == 1


def test_store_unwraps_host_pool_group_to_physical_kv_anchor(monkeypatch):
    anchor_pool = FakeHostPool(page_size=2, component_sizes=[64, 64])
    host_pool_group = SimpleNamespace(
        anchor_entry=SimpleNamespace(host_pool=anchor_pool),
        layout=anchor_pool.layout,
    )
    connector = SimpleNamespace(store=object(), mem_pool_host=None)
    seen = []

    def from_hicache(cls, storage_config, mem_pool_host):
        seen.append(mem_pool_host)
        connector.mem_pool_host = mem_pool_host
        return connector

    monkeypatch.setattr(SglangUcmConnector, "from_hicache", classmethod(from_hicache))
    value = UnifiedCacheStore(storage_config=SimpleNamespace())

    value.register_mem_pool_host(host_pool_group)

    assert seen == [anchor_pool]
    assert value.mem_pool_host is anchor_pool
    assert value.connector.mem_pool_host is anchor_pool


def test_v2_round_trip_groups_component_results_by_logical_page(connector):
    value, stores = connector
    pool_name = sglang_hicache.PoolName.SWA
    value.register_pool_v2(FakeHostPool(2, [64, 64]), pool_name)
    transfer = sglang_hicache.PoolTransfer(
        name=pool_name,
        keys=["page-0", "page-1"],
        host_indices=torch.tensor([0, 1, 2, 3]),
    )

    assert value.batch_io_v2([transfer], is_set=True) == {pool_name: [True, True]}
    assert len(stores[0].dump_calls[0][2]) == 2
    assert len(stores) == 1
    assert stores[0].dump_calls[0][2] == [
        [10_000, 10_100],
        [11_000, 11_100],
    ]
    assert value.batch_io_v2([transfer], is_set=False) == {pool_name: [True, True]}


def test_mamba_uses_one_flattened_store_and_restores_pages(connector):
    value, stores = connector
    pool_name = sglang_hicache.PoolName.MAMBA
    host_pool = FakeHostPool(2, [64, 96])
    value.register_pool_v2(host_pool, pool_name)
    transfer = sglang_hicache.PoolTransfer(
        name=pool_name,
        keys=["page-0", "page-1"],
        host_indices=torch.tensor([0, 1, 2, 3]),
    )

    assert len(stores) == 1
    assert stores[0].config["tensor_size"] == 160
    assert value.batch_io_v2([transfer], is_set=True) == {pool_name: [True, True]}
    assert len(stores[0].dump_calls[0][2]) == 2
    assert value.batch_io_v2([transfer], is_set=False) == {pool_name: [True, True]}
    assert [offset for offset, _ in host_pool.restored_pages] == [0, 2]


def test_v2_exists_intersects_all_required_components(connector):
    value, stores = connector
    pool_name = sglang_hicache.PoolName.INDEXER
    value.register_pool_v2(FakeHostPool(1, [64, 64]), pool_name)
    keys = ["page-0", "page-1", "page-2"]
    stores[0].objects.update(value._component_key(key, pool_name, 0) for key in keys)
    stores[0].objects.remove(value._component_key("page-1", pool_name, 0))
    value.mem_pool_host.kv_buffer = None  # logical KV anchor
    transfer = sglang_hicache.PoolTransfer(
        name=pool_name,
        hit_policy=sglang_hicache.PoolHitPolicy.ALL_PAGES,
    )

    result = value.batch_exists_v2(keys, [transfer])

    assert result.kv_hit_pages == 1
    assert result.restorable_prefix_pages == [1]


def test_logical_anchor_v2_keys_are_shared_across_tp_ranks(tmp_path):
    logical_pool = FakeHostPool(page_size=2, component_sizes=[])
    logical_pool.kv_buffer = None
    pool_name = sglang_hicache.PoolName.DEEPSEEK_V4_C4

    def make_connector(tp_rank):
        storage_config = SimpleNamespace(
            model_name="org/hybrid-model",
            is_mla_model=True,
            tp_rank=tp_rank,
            tp_size=4,
        )
        return SglangUcmConnector(None, logical_pool, storage_config, [str(tmp_path)])

    tp0 = make_connector(0)
    tp3 = make_connector(3)

    assert tp0._component_key("page-0", pool_name, 0) == tp3._component_key(
        "page-0", pool_name, 0
    )


def test_physical_anchor_v2_keys_remain_isolated_by_tp_rank(tmp_path):
    physical_pool = FakeHostPool(page_size=2, component_sizes=[64])
    pool_name = sglang_hicache.PoolName.SWA

    def make_connector(tp_rank):
        storage_config = SimpleNamespace(
            model_name="org/hybrid-model",
            is_mla_model=False,
            tp_rank=tp_rank,
            tp_size=4,
        )
        return SglangUcmConnector(
            FakeStore({}), physical_pool, storage_config, [str(tmp_path)]
        )

    tp0 = make_connector(0)
    tp3 = make_connector(3)

    assert tp0._component_key("page-0", pool_name, 0) != tp3._component_key(
        "page-0", pool_name, 0
    )


def test_trailing_pool_restores_prefix_from_persisted_tail_window(connector):
    value, stores = connector
    pool_name = sglang_hicache.PoolName.SWA
    value.register_pool_v2(FakeHostPool(1, [64]), pool_name)
    value.mem_pool_host.kv_buffer = None  # logical anchor: v2 pools own the payload
    keys = ["page-0", "page-1", "page-2"]
    tail_key = keys[-1]
    transfer = sglang_hicache.PoolTransfer(
        name=pool_name,
        keys=[tail_key],
        hit_policy=sglang_hicache.PoolHitPolicy.TRAILING_PAGES,
    )
    stores[0].objects.add(value._component_key(tail_key, pool_name, 0))

    result = value.batch_exists_v2(keys, [transfer])

    assert result.kv_hit_pages == 3
    assert result.extra_pool_hit_pages[pool_name] == 3
    assert result.restorable_prefix_pages == [3]


def test_trailing_pool_miss_rejects_current_prefix(connector):
    value, _ = connector
    pool_name = sglang_hicache.PoolName.SWA
    value.register_pool_v2(FakeHostPool(1, [64]), pool_name)
    value.mem_pool_host.kv_buffer = None
    transfer = sglang_hicache.PoolTransfer(
        name=pool_name,
        keys=["missing-tail-state"],
        hit_policy=sglang_hicache.PoolHitPolicy.TRAILING_PAGES,
    )

    result = value.batch_exists_v2(["page-0", "page-1"], [transfer])

    assert result.kv_hit_pages == 0
    assert result.restorable_prefix_pages == []


def test_trailing_pool_sparse_lookup_falls_back_to_older_checkpoint(connector):
    value, stores = connector
    pool_name = sglang_hicache.PoolName.SWA
    value.register_pool_v2(FakeHostPool(1, [64]), pool_name)
    value.mem_pool_host.kv_buffer = None
    keys = ["page-0", "page-1", "page-2", "page-3"]
    stores[0].objects.add(value._component_key("page-1", pool_name, 0))
    transfer = sglang_hicache.PoolTransfer(
        name=pool_name,
        keys=["__placeholder__"],
        hit_policy=sglang_hicache.PoolHitPolicy.TRAILING_PAGES,
    )

    result = value.batch_exists_v2(keys, [transfer])

    assert result.kv_hit_pages == 2
    assert result.extra_pool_hit_pages[pool_name] == 2
    assert result.restorable_prefix_pages == [2]


def test_trailing_pool_requires_complete_multi_page_window(connector):
    value, stores = connector
    pool_name = sglang_hicache.PoolName.SWA
    value.register_pool_v2(FakeHostPool(1, [64]), pool_name)
    value.mem_pool_host.kv_buffer = None
    keys = ["page-0", "page-1", "page-2", "page-3"]
    # Page 3 is a candidate endpoint, but page 2 is absent.  The older
    # two-page window ending at page 1 is the latest complete one.
    for key in ("page-0", "page-1", "page-3"):
        stores[0].objects.add(value._component_key(key, pool_name, 0))
    transfer = sglang_hicache.PoolTransfer(
        name=pool_name,
        keys=["__placeholder__", "__placeholder__"],
        hit_policy=sglang_hicache.PoolHitPolicy.TRAILING_PAGES,
    )

    result = value.batch_exists_v2(keys, [transfer])

    assert result.kv_hit_pages == 2
    assert result.extra_pool_hit_pages[pool_name] == 2
    assert result.restorable_prefix_pages == [1, 2]


def test_trailing_pool_placeholders_use_the_candidate_end_checkpoint(connector):
    value, stores = connector
    pool_name = sglang_hicache.PoolName.SWA
    value.register_pool_v2(FakeHostPool(1, [64]), pool_name)
    value.mem_pool_host.kv_buffer = None
    keys = ["page-0", "page-1", "page-2"]
    transfer = sglang_hicache.PoolTransfer(
        name=pool_name,
        keys=["__placeholder__", "__placeholder__", "__placeholder__"],
        hit_policy=sglang_hicache.PoolHitPolicy.TRAILING_PAGES,
    )
    for key in keys:
        stores[0].objects.add(value._component_key(key, pool_name, 0))

    result = value.batch_exists_v2(keys, [transfer])

    assert result.kv_hit_pages == 3
    assert result.extra_pool_hit_pages[pool_name] == 3
    assert result.restorable_prefix_pages == [3]


def test_trailing_placeholder_uses_common_all_pages_boundary(connector):
    value, stores = connector
    value.mem_pool_host.kv_buffer = None
    swa = sglang_hicache.PoolName.SWA
    c4 = sglang_hicache.PoolName.DEEPSEEK_V4_C4
    value.register_pool_v2(FakeHostPool(1, [64]), swa)
    value.register_pool_v2(FakeHostPool(1, [64]), c4)
    keys = ["page-0", "page-1", "page-2"]

    # The C4 pool limits the common primary prefix to two pages.
    for key in keys[:2]:
        stores[1].objects.add(value._component_key(key, c4, 0))
    # The two-page SWA window must end at that same common boundary.
    for key in keys[:2]:
        stores[0].objects.add(value._component_key(key, swa, 0))

    result = value.batch_exists_v2(
        keys,
        [
            sglang_hicache.PoolTransfer(
                name=c4, hit_policy=sglang_hicache.PoolHitPolicy.ALL_PAGES
            ),
            sglang_hicache.PoolTransfer(
                name=swa,
                keys=["__placeholder__", "__placeholder__"],
                hit_policy=sglang_hicache.PoolHitPolicy.TRAILING_PAGES,
            ),
        ],
    )

    assert result.kv_hit_pages == 2
    assert result.extra_pool_hit_pages[c4] == 2
    assert result.extra_pool_hit_pages[swa] == 2
    assert result.restorable_prefix_pages == [2]


def test_close_closes_primary_and_all_dynamic_stores(connector):
    value, stores = connector
    value.register_pool_v2(FakeHostPool(1, [64, 64]), sglang_hicache.PoolName.SWA)

    value.close()

    assert value.store.closed
    assert all(store.closed for store in stores)
