"""Exercise Python preparation without SGLang, torch or native UCM libraries."""

import os
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from common.sglang_utils import adapter_module, linker_module, pipeline_module


class LayoutPreparationTest(unittest.TestCase):
    def test_linker_yaml_path_requires_explicit_config(self):
        linker = object.__new__(linker_module.UcmDirectLinker)
        linker._tp_rank = linker._dp_rank = linker._attn_dp_rank = 0
        linker._tp_size = 1
        linker.pool_group = types.SimpleNamespace(rank_replicated=True)
        params = types.SimpleNamespace(
            pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1
        )
        with (
            patch.dict(os.environ, {"UNIFIEDCACHE_CONFIG_FILE": "environment.yaml"}),
            # Let a regression reading os.environ reach the assertions below.
            patch.object(linker_module, "os", os, create=True),
            patch.object(
                linker_module,
                "get_model",
                lambda: types.SimpleNamespace(model_path="model"),
                create=True,
            ),
            patch.object(
                linker_module,
                "device_module",
                types.SimpleNamespace(current_device=lambda: 0),
            ),
            patch.object(
                adapter_module.UcmDirectStoreAdapter, "from_config"
            ) as factory,
        ):
            linker._create_adapter(params, None)
            self.assertEqual(factory.call_args.args[0], {})
            linker._create_adapter(params, {"UCM_CONFIG_FILE": "explicit.yaml"})
            self.assertEqual(
                factory.call_args.args[0], {"UCM_CONFIG_FILE": "explicit.yaml"}
            )
        with self.assertRaisesRegex(ValueError, "linker_extra_config.UCM_CONFIG_FILE"):
            adapter_module.UcmDirectStoreAdapter.from_config({}, {})
        with self.assertRaisesRegex(ValueError, "renamed to.*UCM_CONFIG_FILE"):
            adapter_module.UcmDirectStoreAdapter.from_config(
                {"config_file": "legacy.yaml"}, {}
            )

    def test_group_name_uses_shortest_pool_and_shard_geometry(self):
        self.assertEqual(
            linker_module._pool_group_name(
                ("deepseek_v4_c4_indexer", "deepseek_v4_c4"), 4096, 32
            ),
            "deepseek_v4_c4_4096x32",
        )
        self.assertEqual(linker_module._pool_group_name(("bb", "aa"), 8, 2), "aa_8x2")
        self.assertEqual(
            linker_module._pool_group_name(("indexer", "kv"), 16, 4), "kv_16x4"
        )

    def test_equal_geometry_group_name_collision_is_rejected(self):
        linker = object.__new__(linker_module.UcmDirectLinker)
        linker.num_layers = 2
        linker.pools = {
            "kv": types.SimpleNamespace(
                name="kv",
                indices_from_pool="kv",
                layer_mapping={0: 0, 1: 1},
                buffer_meta=[[(100, 4, 4), (200, 8, 8)], [(300, 12, 12), (400, 8, 8)]],
                _component_offsets=[[0, 0], [0, 0]],
            )
        }
        with self.assertRaisesRegex(
            ValueError, "Duplicate UCM Store group name: kv_16x1"
        ):
            linker._build_pool_bundles()

    def test_registered_backend_paths_use_group_names(self):
        adapter = self.adapter()
        del adapter._ensure_store_schema
        roots = ["cache-a", "cache-b"]
        adapter._store_config["storage_backends"] = roots
        pools = []
        for group, names, layers, size in (
            ("kv_8x2", ("kv",), (0, 1), 8),
            ("kv_16x2", ("kv",), (0, 1), 16),
            ("kv_8x3", ("kv",), (4, 6, 8), 8),
            ("swa_8x2", ("swa",), (0, 1), 8),
            (
                "deepseek_v4_c4_16x2",
                ("deepseek_v4_c4", "deepseek_v4_c4_indexer"),
                (0, 1),
                8,
            ),
        ):
            for name in names:
                pools.append(
                    adapter_module.UcmPoolMetadata(
                        name=f"{group}:{name}",
                        pool_name=name,
                        source_pool="kv",
                        store_group=group,
                        page_size=4,
                        layer_schemas=tuple((layer, ((size, 0),)) for layer in layers),
                        device_buffers=((100, 4096),),
                    )
                )
        configs = []

        def create_connector(name, config, module_path):
            configs.append(config)
            return object()

        factory = types.SimpleNamespace(create_connector=create_connector)
        with (
            patch.object(adapter_module, "UcmConnectorFactoryV1", factory, create=True),
            patch.object(Path, "mkdir") as mkdir,
        ):
            adapter.register_device_pools(pools)
        expected = {
            group: [str(Path(root) / "sglang_linker" / group) for root in roots]
            for group in adapter._stores
        }
        self.assertEqual(adapter._backend_paths_by_group, expected)
        self.assertEqual(
            {
                group: config["storage_backends"]
                for group, config in zip(adapter._stores, configs)
            },
            expected,
        )
        self.assertEqual(mkdir.call_count, len(expected) * len(roots))

    def test_gc_zero_weight_does_not_restore_full_budget_per_group(self):
        adapter = self.adapter()
        adapter._store_config["posix_capacity_gb"] = 100
        member = types.SimpleNamespace(
            capacity_pages=0, layer_schemas=((0, ((8, 0),)),)
        )
        with self.assertRaisesRegex(ValueError, "no positive estimated pool capacity"):
            adapter._estimate_gc_capacities({"a": [member], "b": [member]})
        member.capacity_pages = 10
        member.layer_schemas = ()
        with self.assertRaisesRegex(ValueError, "no positive estimated pool capacity"):
            adapter._estimate_gc_capacities({"a": [member]})

    def test_only_stock_cache_pipeline_accepts_transfer_events(self):
        for name, module, pipeline, expected in (
            ("UcmPipelineStore", None, "Cache|Posix", True),
            ("UcmPipelineStore", "custom.backend", "Cache|Posix", False),
            ("UcmPipelineStore", None, "Posix", False),
            ("CustomStore", None, "Cache|Posix", False),
        ):
            with self.subTest(name=name, module=module, pipeline=pipeline):
                adapter = adapter_module.UcmDirectStoreAdapter(
                    connector_name=name,
                    module_path=module,
                    store_config={"store_pipeline": pipeline},
                    identity={},
                )
                self.assertEqual(adapter.supports_transfer_prerequisite, expected)

    def transfer(self, **changes):
        transfer = adapter_module.UcmLayerTransfer(
            store_group="g",
            source_pool="kv",
            pools=("kv",),
            layers=(0, 2),
            shard_size=8,
            tensor_size_list=(8,),
            page_keys=("a", "b"),
            addresses=np.array([[[100], [108]], [[200], [208]]], dtype=np.uint64),
            fragments=((8, 0), (8, 0)),
        )
        return replace(transfer, **changes)

    def adapter(self):
        adapter = adapter_module.UcmDirectStoreAdapter(
            connector_name="test",
            module_path=None,
            store_config={},
            identity={},
        )
        self.schema_calls = 0

        def ensure(*args, **kwargs):
            self.schema_calls += 1
            return self

        adapter._ensure_store_schema = ensure
        return adapter

    def test_layout_reused_but_dynamic_addresses_checked(self):
        adapter = self.adapter()
        first = adapter.prepare_load([self.transfer()])[0][0]
        second = adapter.prepare_load([self.transfer(page_keys=("c", "d"))])[0][0]
        self.assertEqual(self.schema_calls, 1)
        self.assertIs(first.shards_by_layer, second.shards_by_layer)
        self.assertNotEqual(first.block_ids, second.block_ids)
        with self.assertRaisesRegex(ValueError, "address/schema"):
            adapter.prepare_load(
                [self.transfer(addresses=np.zeros((1, 2, 1), dtype=np.uint64))]
            )
        for addresses in (None, ((100, 200), (108, 208))):
            with self.subTest(addresses=addresses):
                with self.assertRaisesRegex(ValueError, "address/schema"):
                    adapter.prepare_load([self.transfer(addresses=addresses)])
        with self.assertRaisesRegex(ValueError, "changed its layout"):
            adapter.prepare_load([self.transfer(layers=(0, 1))])

    def test_bundle_cache_preserves_composite_addresses(self):
        linker = object.__new__(linker_module.UcmDirectLinker)

        def entry(name, base, size):
            return types.SimpleNamespace(
                name=name,
                indices_from_pool="kv",
                layer_mapping={0: 0, 1: 1},
                packed=True,
                buffer_meta=[[(base, size, size), (base + 1000, size, size)]],
                _component_offsets=[[0, 0]],
                prepare_locations=lambda indices: indices,
            )

        linker.pools = {"kv": entry("kv", 100, 8), "indexer": entry("indexer", 200, 4)}
        linker.num_layers = 2
        linker._bundles = linker._build_pool_bundles()
        linker._layer_pointer_layouts = linker._build_layer_pointer_layouts()
        # Request preparation must use the cached schema, not rebuild it.
        linker._entry_layer_schema = lambda *args: self.fail("schema rebuilt")
        indices = types.SimpleNamespace(
            detach=lambda: types.SimpleNamespace(cpu=lambda: [0, 2])
        )
        transfers = [
            types.SimpleNamespace(
                name=name, keys=("a", "b"), hit_policy="all_pages", host_indices=indices
            )
            for name in linker.pools
        ]
        result = linker._page_transfers(transfers, for_io=True)[0]
        self.assertEqual(result.tensor_size_list, (4, 8))
        self.assertEqual(result.shard_size, 12)
        np.testing.assert_array_equal(
            result.addresses,
            [
                [[200, 100], [208, 116]],
                [[1200, 1100], [1208, 1116]],
            ],
        )
        lookup = linker._page_transfers(transfers, for_io=False)[0]
        self.assertIs(lookup.fragments, result.fragments)
        self.assertIsNone(lookup.addresses)

        calls = []
        original = linker._entry_layer_rows

        def rows(entry, layer, locations):
            calls.append((entry.name, layer))
            return original(entry, layer, locations)

        linker._entry_layer_rows = rows
        lazy = linker._page_transfers(transfers, for_io=True, lazy_addresses=True)[0]
        self.assertIsNone(lazy.addresses)
        self.assertEqual(calls, [])
        adapter = self.adapter()
        prepared = adapter.prepare_load([lazy])
        self.assertEqual(calls, [])
        submitted = []
        self.load_data = lambda ids, indexes, addresses: submitted.append(addresses)
        for i, layer in enumerate(lazy.layers):
            tasks = adapter.submit_prepared_load_layer(prepared, layer)
            np.testing.assert_array_equal(submitted[-1], result.addresses[i])
            self.assertIs(tasks[0].addresses, submitted[-1])
        self.assertEqual(len(calls), len(linker.pools) * len(lazy.layers))

        linker.pools["kv"].prepare_locations = lambda indices: indices[:1]
        with self.assertRaisesRegex(ValueError, "missing kv layer"):
            linker._page_transfers(transfers, for_io=True)

    def test_cpu_index_snapshot_is_shared_but_pool_rows_are_independent(self):
        copies = []
        values = [0, 1, 4, 5]

        def indices():
            source = types.SimpleNamespace()

            def cpu():
                copies.append(source)
                return np.array(values)

            source.detach = lambda: source
            source.cpu = cpu
            return source

        linker = object.__new__(linker_module.UcmDirectLinker)

        def entry(name, base, stride, divisor):
            return types.SimpleNamespace(
                name=name,
                indices_from_pool="kv",
                layer_mapping={0: 0},
                packed=True,
                buffer_meta=[[(base, stride, stride)]],
                _component_offsets=[[0]],
                prepare_locations=lambda cpu: cpu[::2] // divisor,
            )

        linker.pools = {
            "kv": entry("kv", 100, 8, 1),
            "indexer": entry("indexer", 200, 4, 2),
        }
        linker.num_layers = 1
        linker._bundles = linker._build_pool_bundles()
        linker._layer_pointer_layouts = linker._build_layer_pointer_layouts()
        shared = indices()
        transfers = [
            types.SimpleNamespace(
                name=name,
                keys=("a", "b"),
                hit_policy="all_pages",
                host_indices=shared,
            )
            for name in linker.pools
        ]
        eager = linker._page_transfers(transfers, for_io=True)[0]
        self.assertEqual(len(copies), 1)
        np.testing.assert_array_equal(eager.addresses, [[[200, 100], [208, 132]]])
        lazy = linker._page_transfers(transfers, for_io=True, lazy_addresses=True)[0]
        self.assertEqual(len(copies), 2)
        np.testing.assert_array_equal(lazy.address_builder(0), eager.addresses[0])
        linker._page_transfers(transfers, for_io=False)
        self.assertEqual(len(copies), 2)
        # An equal-valued but distinct tensor must have its own snapshot.
        transfers[1].host_indices = indices()
        linker._page_transfers(transfers, for_io=True)
        self.assertEqual(len(copies), 4)
        # Never reuse a previous call's snapshot after indices change.
        values[:] = [2, 3, 6, 7]
        fresh = linker._page_transfers(transfers, for_io=True)[0]
        np.testing.assert_array_equal(fresh.addresses, [[[204, 116], [212, 148]]])

    def test_lookup_reuses_layout_and_preserves_resume_fallback(self):
        hits = {"a", "b", "c"}
        calls = []

        class Store:
            def lookup_on_prefix(self, ids):
                calls.append(len(ids))
                return next(
                    (i - 1 for i, key in enumerate(ids) if key not in hits),
                    len(ids) - 1,
                )

        adapter = self.adapter()
        store = Store()
        adapter._ensure_store_schema = lambda *args, **kwargs: store
        adapter._block_ids = lambda transfer, keys: list(keys)
        transfer = self.transfer(addresses=None)
        keys = ["a", "b", "c", "d"]
        self.assertEqual(adapter.lookup([transfer], keys), [1, 2, 3])
        layout = adapter._transfer_layouts["g"]
        self.assertEqual(
            adapter.lookup([transfer], keys, resume_from_pages=3), [1, 2, 3]
        )
        hits.remove("c")
        self.assertEqual(adapter.lookup([transfer], keys, resume_from_pages=3), [1, 2])
        self.assertEqual(calls, [4, 2, 2, 4])
        self.assertIs(adapter._transfer_layouts["g"], layout)

    def test_central_schema_validation_rejects_empty_and_negative_sizes(self):
        adapter = self.adapter()
        del adapter._ensure_store_schema
        invalid = [
            self.transfer(layers=(), fragments=()),
            self.transfer(
                tensor_size_list=(-8,),
                shard_size=-8,
                fragments=((-8, 0), (-8, 0)),
            ),
        ]
        for transfer in invalid:
            with self.subTest(transfer=transfer):
                with self.assertRaisesRegex(ValueError, "no valid fragments"):
                    adapter._transfer_layout(transfer)
        self.assertEqual(adapter._schemas, {})
        self.assertEqual(adapter._transfer_layouts, {})

    def test_prepared_pipeline_reuses_ids_and_preserves_shards_and_event(self):
        calls = []

        class Native:
            def Load(self, ids, indexes, addresses, event=0):
                calls.append((ids, indexes, addresses, event))
                return len(calls)

            def Dump(self, ids, indexes, addresses, event):
                calls.append((ids, indexes, addresses, event))
                return len(calls)

        store = object.__new__(pipeline_module.UcmPipelineStore)
        store.store_ = Native()
        adapter = self.adapter()
        adapter._connector_name = "UcmPipelineStore"
        adapter._ensure_store_schema = lambda *args, **kwargs: store
        transfer = self.transfer()
        with patch.dict(sys.modules, {"ucm.store.pipeline.connector": pipeline_module}):
            prepared = adapter.prepare_load([transfer])
        self.assertIs(prepared[0][0], prepared[2][0])
        self.assertEqual(adapter.submit_prepared_load_layer(prepared, 1), [])
        adapter.submit_prepared_load_layer(prepared, 0, prerequisite_handle=43)
        adapter.submit_prepared_load_layer(prepared, 2, prerequisite_handle=43)
        self.assertEqual([call[3] for call in calls], [43, 43])
        self.assertIs(calls[0][0], calls[1][0])
        self.assertFalse(calls[0][0].flags.writeable)
        self.assertEqual(calls[0][0].tobytes(), b"".join(adapter._block_ids(transfer)))
        np.testing.assert_array_equal(calls[0][1], [0, 0])
        np.testing.assert_array_equal(calls[1][1], [1, 1])
        np.testing.assert_array_equal(calls[0][2], transfer.addresses[0])
        np.testing.assert_array_equal(calls[1][2], transfer.addresses[1])
        self.assertTrue(np.shares_memory(calls[0][2], transfer.addresses))
        self.assertTrue(np.shares_memory(calls[1][2], transfer.addresses))
        with patch.dict(sys.modules, {"ucm.store.pipeline.connector": pipeline_module}):
            adapter.submit_offload([transfer], prerequisite_handle=42)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[2][3], 42)
        self.assertEqual(
            calls[2][0].tobytes(), b"".join(adapter._block_ids(transfer)) * 2
        )
        np.testing.assert_array_equal(calls[2][1], [0, 0, 1, 1])
        np.testing.assert_array_equal(calls[2][2], transfer.addresses.reshape(4, 1))
        self.assertTrue(np.shares_memory(calls[2][2], transfer.addresses))

    def dump_adapter(self, fail_call=None):
        calls, waited = [], []

        class Native:
            def Dump(self, ids, indexes, addresses, event):
                calls.append((ids, indexes, addresses, event))
                if len(calls) == fail_call:
                    raise RuntimeError("dump submit failed")
                return len(calls)

            def Wait(self, task_id):
                waited.append(task_id)

        store = object.__new__(pipeline_module.UcmPipelineStore)
        store.store_ = Native()
        adapter = self.adapter()
        adapter._connector_name = "UcmPipelineStore"
        adapter._store_config["store_pipeline"] = "Cache|Posix"
        adapter._ensure_store_schema = lambda *args, **kwargs: store
        return adapter, calls, waited

    def test_dump_keeps_node_and_group_task_boundaries(self):
        adapter, calls, waited = self.dump_adapter()
        first, second = self.transfer(), self.transfer(store_group="other")
        with patch.dict(sys.modules, {"ucm.store.pipeline.connector": pipeline_module}):
            tasks = adapter.submit_offload([first, second], prerequisite_handle=42)
            following = adapter.submit_offload([first], prerequisite_handle=43)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len(following), 1)
        self.assertEqual([call[3] for call in calls], [42, 42, 43])
        for call, transfer in zip(calls, (first, second, first)):
            self.assertEqual(
                call[0].tobytes(), b"".join(adapter._block_ids(transfer)) * 2
            )
            np.testing.assert_array_equal(call[1], [0, 0, 1, 1])
            np.testing.assert_array_equal(call[2], transfer.addresses.reshape(4, 1))
            self.assertTrue(np.shares_memory(call[2], transfer.addresses))
        self.assertIs(tasks[0].addresses, calls[0][2])
        adapter.wait(tasks + following)
        self.assertEqual(waited, [1, 2, 3])

    def test_dump_submit_failure_drains_previous_groups(self):
        adapter, calls, waited = self.dump_adapter(fail_call=2)
        transfers = [self.transfer(), self.transfer(store_group="other")]
        with (
            patch.dict(sys.modules, {"ucm.store.pipeline.connector": pipeline_module}),
            self.assertLogs(adapter_module.logger, level="ERROR"),
        ):
            with self.assertRaisesRegex(RuntimeError, "dump submit failed"):
                adapter.submit_offload(transfers, prerequisite_handle=42)
        self.assertEqual(waited, [1])
        self.assertEqual(len(calls), 2)

    def test_generic_dump_submits_all_shards_through_public_method(self):
        calls = []

        class Store:
            def dump_data(self, ids, indexes, addresses):
                calls.append((ids, indexes, addresses))
                return 1

        adapter = self.adapter()
        adapter._ensure_store_schema = lambda *args, **kwargs: Store()
        transfer = self.transfer()
        tasks = adapter.submit_offload([transfer])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(calls[0][0], adapter._block_ids(transfer) * 2)
        self.assertEqual(calls[0][1], [0, 0, 1, 1])
        np.testing.assert_array_equal(calls[0][2], transfer.addresses.reshape(4, 1))

    def test_custom_pipeline_stores_keep_their_public_methods(self):
        calls, dumped = [], []

        class CustomPipeline(pipeline_module.UcmPipelineStore):
            def load_data(self, block_ids, indexes, addresses):
                calls.append((block_ids, indexes, addresses))
                return pipeline_module.UcmPipelineStoreTransTask(1)

            def dump_data(self, block_ids, indexes, addresses):
                dumped.append((block_ids, indexes, addresses))
                return pipeline_module.UcmPipelineStoreTransTask(2)

        store = object.__new__(CustomPipeline)
        adapter = self.adapter()
        adapter._connector_name = "UcmPipelineStore"
        adapter._ensure_store_schema = lambda *args, **kwargs: store
        transfer = self.transfer()
        with patch.dict(sys.modules, {"ucm.store.pipeline.connector": pipeline_module}):
            prepared = adapter.prepare_load([transfer])
        self.assertIsNone(prepared[0][0].prepared_io)
        adapter.submit_prepared_load_layer(prepared, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], [0, 0])
        np.testing.assert_array_equal(calls[0][2], transfer.addresses[0])
        self.assertTrue(np.shares_memory(calls[0][2], transfer.addresses))
        with patch.dict(sys.modules, {"ucm.store.pipeline.connector": pipeline_module}):
            dumped_tasks = adapter.submit_offload([transfer])
        self.assertEqual(len(dumped_tasks), 1)
        self.assertEqual(len(dumped), 1)
        self.assertEqual(dumped[0][0], adapter._block_ids(transfer) * 2)
        self.assertEqual(dumped[0][1], [0, 0, 1, 1])
        np.testing.assert_array_equal(dumped[0][2], transfer.addresses.reshape(4, 1))
        adapter._module_path = "custom.backend"
        self.assertIsNone(adapter._prepare_pipeline_io(store, [b"a" * 16]))

    def test_generic_fallback_and_partial_submit_drains_tasks(self):
        calls, waited = [], []

        class Store:
            def load_data(self, ids, indexes, addresses):
                calls.append((ids, indexes, addresses))
                if len(calls) == 2:
                    raise RuntimeError("submission failed")
                return 1

            def wait(self, task):
                waited.append(task)

        adapter = self.adapter()
        adapter._ensure_store_schema = lambda *args, **kwargs: Store()
        transfer = self.transfer(layers=(0, 0))
        prepared = adapter.prepare_load([transfer])
        with self.assertLogs(adapter_module.logger, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "submission failed"):
                adapter.submit_prepared_load_layer(prepared, 0)
        self.assertEqual(waited, [1])
        self.assertEqual(calls[0][0], adapter._block_ids(transfer))
        self.assertEqual(calls[1][1], [1, 1])

    def test_load_index_contains_only_relevant_groups(self):
        adapter = self.adapter()
        first = self.transfer()
        second = self.transfer(store_group="other", layers=(1, 3))
        prepared = adapter.prepare_load([first, second])
        self.assertEqual(set(prepared), {0, 1, 2, 3})
        self.assertEqual([x.transfer.store_group for x in prepared[0]], ["g"])
        self.assertEqual([x.transfer.store_group for x in prepared[1]], ["other"])


if __name__ == "__main__":
    unittest.main()
