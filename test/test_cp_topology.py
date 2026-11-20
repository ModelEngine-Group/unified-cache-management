"""CPU-only regression tests; run directly with Python (no vLLM required)."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "ucm/integration/vllm/ucm_connector.py"
# Load the real pure-Python topology class without importing device dependencies.
from dataclasses import dataclass

tree = ast.parse(MODULE.read_text(encoding="utf-8-sig"))
topology_node = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CPTopology"
)
namespace = {"dataclass": dataclass}
exec(
    compile(ast.Module(body=[topology_node], type_ignores=[]), str(MODULE), "exec"),
    namespace,
)
CPTopology = namespace["CPTopology"]


class TopologyTests(unittest.TestCase):
    def test_glm_full_grid_has_32_not_64_shards(self):
        topology = CPTopology(16, 2, 32, True)
        keys = list(range(96))
        partitions = [topology.local_keys(keys, r, [10, 20, 30]) for r in range(32)]
        self.assertEqual(sorted(sum(partitions, [])), keys)
        self.assertEqual(partitions[31], [31, 63, 95])
        self.assertEqual(topology.head_partition_count, 1)

    def test_pcp_replicas_read_same_full_cache(self):
        topology = CPTopology(16, 2, 1, True)
        self.assertEqual(topology.local_keys([1, 2], 0, [7, 8]), [1, 2])

    def test_head_identity(self):
        for tp, pcp, dcp, expected in [
            (16, 1, 4, [r // 4 for r in range(16)]),
            (16, 2, 2, list(range(16))),
            (16, 2, 32, [0] * 16),
            (16, 2, 1, list(range(16))),
        ]:
            topology = CPTopology(tp, pcp, dcp, False)
            self.assertEqual([topology.head_partition(r) for r in range(tp)], expected)
            self.assertEqual(topology.head_partition_count, len(set(expected)))

    def test_pure_dcp_mapping_preserved(self):
        topology = CPTopology(16, 1, 16, True)
        self.assertEqual(topology.local_keys(list(range(32)), 7, [1, 2]), [7, 23])

    def test_reject_invalid_topology_and_partial_metadata(self):
        for sizes in [(16, 2, 16), (16, 1, 3), (0, 1, 1)]:
            with self.assertRaises(ValueError):
                CPTopology(*sizes, True)
        topology = CPTopology(16, 2, 32, True)
        with self.assertRaises(ValueError):
            topology.local_keys(list(range(63)), 0, [1, 2])
        with self.assertRaises(ValueError):
            topology.local_keys(list(range(64)), 32, [1, 2])


class HashNamespaceTests(unittest.TestCase):
    def test_interleave_isolates_pure_dcp_and_pcp_namespaces(self):
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "build_kv_hash_meta"
        )
        ns = {
            "_get_sparse_c8_flags": lambda config: (False, False),
            "request_hasher_module": SimpleNamespace(
                _layout_source_digest=lambda: "layout"
            ),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(MODULE), "exec"), ns)

        def meta(pcp, dcp, interleave):
            config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=4,
                    prefill_context_parallel_size=pcp,
                    decode_context_parallel_size=dcp,
                    cp_kv_cache_interleave_size=interleave,
                ),
                model_config=SimpleNamespace(model="model", dtype="float16"),
            )
            return ns["build_kv_hash_meta"](config, 0)

        for pcp, dcp in [(1, 2), (2, 1), (2, 2)]:
            self.assertNotEqual(meta(pcp, dcp, 1), meta(pcp, dcp, 2))
        self.assertIn(":cp_v2:pcp=2:interleave=2:", meta(2, 2, 2))
        self.assertEqual(meta(1, 1, 1), meta(1, 1, 2))


class ConnectorMethodsTests(unittest.TestCase):
    """Execute the real mapping/lookup methods without importing torch or vLLM."""

    @classmethod
    def setUpClass(cls):
        source = MODULE
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        connector = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "UCMCPConnector"
        )
        methods = [
            n
            for n in connector.body
            if isinstance(n, ast.FunctionDef)
            and n.name
            in (
                "bind_connector_metadata",
                "_lookup_one_stage",
                "_get_mla_dump_slice",
                "_get_rank_dump_blocks",
            )
        ]

        class Parent:
            def _get_rank_dump_blocks(self, request):
                return request.dump_block_ids

            def bind_connector_metadata(self, metadata):
                self.bound = metadata

            def _lookup_one_stage(self, store, keys):
                return store.lookup_on_prefix(keys)

        body = ast.ClassDef(
            name="TestConnector",
            bases=[ast.Name(id="Parent", ctx=ast.Load())],
            keywords=[],
            body=methods,
            decorator_list=[],
        )
        test_module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                body,
            ],
            type_ignores=[],
        )
        namespace = {"Parent": Parent}
        exec(
            compile(ast.fix_missing_locations(test_module), str(source), "exec"),
            namespace,
        )
        cls.connector_cls = namespace["TestConnector"]

    def test_metadata_is_not_modified_and_can_be_bound_twice(self):
        worker = self.connector_cls()
        worker.cp_topology = CPTopology(16, 2, 32, True)
        worker.current_rank = 17
        metadata = SimpleNamespace(
            request_meta={
                "r": SimpleNamespace(
                    load_block_ids=(list(range(64)), [7, 8]),
                    dump_block_ids=(list(range(32)), [9]),
                )
            }
        )
        import threading

        metadata.opaque = threading.Lock()  # Must not require deep-copy support.
        metadata.request_meta["r"].opaque = metadata.opaque
        for _ in range(2):
            worker.bind_connector_metadata(metadata)
            self.assertIsNot(worker.bound, metadata)
            self.assertIsNot(worker.bound.request_meta, metadata.request_meta)
            self.assertIsNot(worker.bound.request_meta["r"], metadata.request_meta["r"])
            self.assertIs(worker.bound.request_meta["r"].opaque, metadata.opaque)
            self.assertEqual(
                worker.bound.request_meta["r"].load_block_ids, ([17, 49], [7, 8])
            )
            self.assertEqual(worker.bound.request_meta["r"].dump_block_ids, ([17], [9]))
        self.assertEqual(len(metadata.request_meta["r"].load_block_ids[0]), 64)

    def test_lookup_intersects_head_partitions(self):
        worker = self.connector_cls()
        worker._other_rank_hashers = [lambda key: key + 100]

        class Store:
            def lookup_on_prefix(self, keys):
                for i, key in enumerate(keys):
                    if key not in {0, 1, 2, 100}:
                        return i - 1
                return len(keys) - 1

        self.assertEqual(worker._lookup_one_stage(Store(), [0, 1, 2]), 0)
        self.assertEqual(worker._lookup_one_stage(Store(), []), -1)

    def test_cp_dump_retains_every_local_shard_block(self):
        worker = self.connector_cls()
        worker.tp_size = 4
        worker.tp_rank = 0
        keys = list(range(12))
        self.assertEqual(keys[worker._get_mla_dump_slice(3)], keys)


class PPCombinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import copy
        import math

        tree = ast.parse(MODULE.read_text(encoding="utf-8-sig"))
        connector = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "UCMLayerWiseConnector"
        )
        methods = [
            n
            for n in connector.body
            if isinstance(n, ast.FunctionDef)
            and n.name
            in (
                "_configure_pp_store",
                "_lookup_all_stages",
                "set_xfer_handshake_metadata_pp_aware",
            )
        ]
        node = ast.ClassDef(
            name="PPConnector", bases=[], keywords=[], body=methods, decorator_list=[]
        )
        namespace = {"math": math, "copy": copy, "uses_partitioned_cache": lambda: True}
        code = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                node,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(code), str(MODULE), "exec"), namespace)
        cls.connector_cls = namespace["PPConnector"]
        cls.method_globals = namespace

    def worker(self, pcp=2, local=16):
        worker = self.connector_cls()
        worker.tp_size = 4
        worker.is_mla = True
        worker.unique_id = "test"
        worker._vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=2, prefill_context_parallel_size=pcp
            )
        )
        worker._get_world_size = lambda: local
        worker._pp_stage_name = lambda rank, start, count: f"pp{rank}_{start}_{count}"
        worker._pp_stage_paths = lambda roots, rank, start, count: [
            f"/stage{rank}/{start}/{count}"
        ]
        return worker

    def test_stage_budget_uses_tp_times_pcp(self):
        worker = self.worker()
        config = dict(
            storage_backends="/shared",
            cache_buffer_capacity_gb=160,
            posix_capacity_gb=100,
        )
        worker._configure_pp_store(config, 1, 20, 18)
        self.assertEqual(config["local_rank_size"], 8)
        self.assertEqual(config["cache_buffer_capacity_gb"], 80)
        self.assertEqual(config["posix_capacity_gb"], 50)
        self.assertEqual(config["storage_backends"], ["/stage1/20/18"])

    def test_pp_without_pcp_budget_preserved(self):
        worker = self.worker(pcp=1, local=8)
        config = dict(storage_backends="/shared", cache_buffer_capacity_gb=160)
        worker._configure_pp_store(config, 0, 0, 20)
        self.assertEqual(config["local_rank_size"], 4)
        self.assertEqual(config["cache_buffer_capacity_gb"], 80)

    def test_lookup_requires_every_stage(self):
        from concurrent.futures import ThreadPoolExecutor

        worker = self.worker()
        worker._is_pp = True
        self.assertEqual(worker._lookup_all_stages([1, 2]), -1)
        worker._pp_stores = [2, 0]
        worker._lookup_one_stage = lambda store, keys: store
        with ThreadPoolExecutor(max_workers=2) as pool:
            worker._get_lookup_pool = lambda count: pool
            self.assertEqual(worker._lookup_all_stages([1, 2, 3]), 0)
            worker._pp_stores = [2, -1]
            self.assertEqual(worker._lookup_all_stages([1, 2, 3]), -1)

    def test_handshake_rejects_inconsistent_geometry(self):
        worker = self.worker()
        worker._is_pp = True
        worker.connector_configs = [
            {"ucm_connector_name": "fake", "ucm_connector_config": {}}
        ]
        worker._set_default_shm_buffer_capacity = lambda config: None
        stage = dict(layer_start=0, layer_count=20, block_size=128, shard_size=64)
        with self.assertRaisesRegex(ValueError, "heterogeneous"):
            worker.set_xfer_handshake_metadata_pp_aware(
                {(0, 0): stage, (0, 4): dict(stage, block_size=256)}
            )

    def test_handshake_reports_missing_fields_before_geometry_comparison(self):
        worker = self.worker()
        worker._is_pp = True
        worker.connector_configs = [
            {"ucm_connector_name": "fake", "ucm_connector_config": {}}
        ]
        worker._set_default_shm_buffer_capacity = lambda config: None
        stage = dict(layer_start=0, layer_count=20, block_size=128, shard_size=64)
        for coordinate in [(0, 0), (0, 1)]:
            for field in stage:
                for absent in [True, False]:
                    incomplete = dict(stage)
                    if absent:
                        incomplete.pop(field)
                    else:
                        incomplete[field] = None
                    metadata = {(0, 0): dict(stage), (0, 1): dict(stage)}
                    metadata[coordinate] = incomplete
                    with self.assertRaisesRegex(
                        ValueError, f"missing required fields:.*{field}"
                    ):
                        worker.set_xfer_handshake_metadata_pp_aware(metadata)

    def test_handshake_builds_all_stage_stores_with_unequal_layers(
        self, three_dimensional=False
    ):
        worker = self.worker()
        worker._is_pp = True
        worker._gc_owner = False
        worker.connector_configs = [
            {
                "ucm_connector_name": "fake",
                "ucm_connector_config": {
                    "storage_backends": "/shared",
                    "cache_buffer_capacity_gb": 160,
                    "store_pipeline": "Cache|Posix",
                },
            }
        ]
        worker._set_default_shm_buffer_capacity = lambda config: None
        self.method_globals["UcmConnectorFactoryV1"] = SimpleNamespace(
            create_connector=lambda name, config, module: config
        )
        self.method_globals["logger"] = SimpleNamespace(info=lambda *args: None)
        metadata = {
            (0, 0): dict(layer_start=0, layer_count=20, block_size=1280, shard_size=64),
            (1, 0): dict(
                layer_start=20, layer_count=18, block_size=1152, shard_size=64
            ),
        }
        if three_dimensional:
            metadata = {
                (pp, pcp, tp): dict(stage)
                for (pp, _), stage in metadata.items()
                for pcp in range(2)
                for tp in range(4)
            }
            self.assertEqual(len(metadata), 16)
        worker.set_xfer_handshake_metadata_pp_aware(metadata)
        self.assertEqual(len(worker._pp_stores), 2)
        self.assertEqual(worker.store["block_size"], 1280)
        self.assertEqual(worker._pp_stores[1]["block_size"], 1152)
        self.assertEqual(worker._pp_stores[1]["store_pipeline"], "Posix")
        self.assertNotEqual(
            worker.store["unique_id"], worker._pp_stores[1]["unique_id"]
        )
        previous = worker._pp_stores
        worker.set_xfer_handshake_metadata_pp_aware(metadata)
        self.assertIs(worker._pp_stores, previous)

    def test_handshake_accepts_pp_pcp_tp_keys_from_runtime(self):
        self.test_handshake_builds_all_stage_stores_with_unequal_layers(True)

    def test_three_coordinate_handshake_checks_every_worker(self):
        worker = self.worker()
        worker._is_pp = True
        worker.connector_configs = [
            {"ucm_connector_name": "fake", "ucm_connector_config": {}}
        ]
        worker._set_default_shm_buffer_capacity = lambda config: None
        stage = dict(layer_start=0, layer_count=20, block_size=128, shard_size=64)
        with self.assertRaisesRegex(ValueError, "heterogeneous"):
            worker.set_xfer_handshake_metadata_pp_aware(
                {(0, 0, 0): stage, (0, 1, 3): dict(stage, shard_size=32)}
            )

    def test_handshake_missing_representative_has_actionable_error(self):
        worker = self.worker()
        worker._is_pp = True
        worker.connector_configs = [
            {"ucm_connector_name": "fake", "ucm_connector_config": {}}
        ]
        worker._set_default_shm_buffer_capacity = lambda config: None
        with self.assertRaisesRegex(ValueError, "expected"):
            worker.set_xfer_handshake_metadata_pp_aware({(0, 1, 0): {}})


class ObjectOwnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Permit running these tests independently of ConnectorMethodsTests.
        ConnectorMethodsTests.setUpClass()

    def test_mismatched_object_mapping_rejected(self):
        worker = ConnectorMethodsTests.connector_cls()
        worker.cp_topology = CPTopology(4, 2, 2, True)
        worker._dump_tp_rank = 0
        worker.pcp_rank = 0
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            worker._get_rank_dump_blocks(SimpleNamespace(dump_block_ids=([b"key"], [])))

    def test_exactly_one_writer_per_content_partition(self):
        for tp, pcp, dcp in [
            (4, 1, 1),
            (4, 1, 2),
            (4, 1, 4),
            (4, 2, 1),
            (4, 2, 2),
            (4, 2, 8),
            (16, 2, 32),
        ]:
            for mla in [True, False]:
                topology = CPTopology(tp, pcp, dcp, mla)
                for value in range(64):
                    key = value.to_bytes(16, "big")
                    writers = {}
                    for pr in range(pcp):
                        for tr in range(tp):
                            dr = (
                                tr % dcp
                                if pcp == 1
                                else (
                                    0
                                    if dcp == 1
                                    else pr if dcp == pcp else tr * pcp + pr
                                )
                            )
                            content = (dr, topology.head_partition(tr))
                            writers.setdefault(content, 0)
                            writers[content] += topology.owns_dump_key(key, tr, pr)
                    self.assertTrue(
                        all(count == 1 for count in writers.values()),
                        (tp, pcp, dcp, mla, value, writers),
                    )

    def test_real_connector_selection_stable_for_all_layers_and_pp_stages(self):
        # Reproduce TP4/PCP2/DCP2's four competing latent replicas.
        topology = CPTopology(4, 2, 2, True)
        keys = [value.to_bytes(16, "big") for value in range(32)]
        for stage in [0, 1]:
            for dr in [0, 1]:
                all_selected = []
                for tr in range(4):
                    worker = ConnectorMethodsTests.connector_cls()
                    worker.cp_topology = topology
                    worker._dump_tp_rank = tr
                    worker.pcp_rank = dr
                    request = SimpleNamespace(dump_block_ids=(keys, list(range(32))))
                    selected, blocks = worker._get_rank_dump_blocks(request)
                    self.assertEqual(selected, [keys[i] for i in blocks])
                    all_selected.extend(selected)
                    for layer in range(39 + stage):
                        self.assertEqual(
                            worker._get_rank_dump_blocks(request)[0], selected
                        )
                    # Physical allocation and list position cannot change ownership.
                    request.dump_block_ids = (
                        list(reversed(keys)),
                        list(range(100, 132)),
                    )
                    self.assertEqual(
                        set(worker._get_rank_dump_blocks(request)[0]), set(selected)
                    )
                self.assertEqual(sorted(all_selected), keys)

    def test_full_dcp_keeps_all_distinct_shards(self):
        topology = CPTopology(4, 2, 8, True)
        for pr in range(2):
            for tr in range(4):
                self.assertTrue(topology.owns_dump_key(b"object", tr, pr))

    def test_invalid_replica_coordinate_rejected(self):
        with self.assertRaises(ValueError):
            CPTopology(4, 2, 2, True).dump_replica(4, 0)


class SyncLoadPatchTests(unittest.TestCase):
    def test_metadata_and_wrapper_preserve_load_semantics(self):
        import sys
        from functools import wraps
        from types import ModuleType
        from unittest.mock import patch

        source = (
            ROOT
            / "ucm/integration/vllm/patch/v0300/vllm_ascend/balance_sync_kv_load_patch.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        for node in functions:
            node.decorator_list = []
        namespace = {"wraps": wraps, "logger": SimpleNamespace(info=lambda *args: None)}
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
            namespace,
        )

        class Metadata:
            def __init__(self, requests):
                self.request_meta = requests

        class Multi:
            def __init__(self, children):
                self.metadata = children

        single = ModuleType("ucm.integration.vllm.ucm_connector")
        single.UCMConnectorMetadata = Metadata
        multi = ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
        )
        multi.MultiKVConnectorMetadata = Multi
        with patch.dict(sys.modules, {single.__name__: single, multi.__name__: multi}):
            check = namespace["_has_ucm_sync_load"]
            sync = Metadata(
                {"r": SimpleNamespace(load_block_ids=([b"k"], [1]), load_async=False)}
            )
            asynchronous = Metadata(
                {"r": SimpleNamespace(load_block_ids=([b"k"], [1]), load_async=True)}
            )
            empty = Metadata(
                {"r": SimpleNamespace(load_block_ids=([], []), load_async=False)}
            )
            self.assertTrue(check(sync))
            self.assertTrue(check(Multi([asynchronous, sync])))
            self.assertFalse(check(asynchronous))
            self.assertFalse(check(empty))
            self.assertFalse(check(object()))

            class Scheduler:
                def schedule(self):
                    return self.output

            namespace["patch_balance_sync_kv_load"](
                SimpleNamespace(BalanceScheduler=Scheduler)
            )
            wrapped = Scheduler.schedule
            namespace["patch_balance_sync_kv_load"](
                SimpleNamespace(BalanceScheduler=Scheduler)
            )
            self.assertIs(Scheduler.schedule, wrapped)
            scheduler = Scheduler()
            for metadata, initial, expected in [
                (sync, False, True),
                (asynchronous, False, False),
                (empty, False, False),
                (object(), False, False),
                (asynchronous, True, True),
            ]:
                output = SimpleNamespace(
                    kv_connector_metadata=metadata, has_sync_kv_loads=initial
                )
                scheduler.output = output
                self.assertIs(scheduler.schedule(), output)
                self.assertEqual(output.has_sync_kv_loads, expected)


class LayerLoadWaitTests(unittest.TestCase):
    def test_wait_consumes_task_before_counting_loaded_bytes(self):
        import time
        from unittest.mock import MagicMock

        tree = ast.parse(MODULE.read_text(encoding="utf-8-sig"))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "UCMLayerWiseConnector"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "wait_for_layer_load"
        )
        namespace = {"time": time, "ucmmetrics": MagicMock(), "logger": MagicMock()}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(MODULE), "exec"),
            namespace,
        )
        task = object()
        request = SimpleNamespace(
            load_block_ids=([b"key"], [1]), dump_block_ids=([], [])
        )
        wait = MagicMock()
        worker = SimpleNamespace(
            _connector_metadata=True,
            need_load=True,
            layer_name_to_id={"layer": 39},
            load_tasks={39: {"r": task}},
            layer_ids=[39],
            first_layer_id=39,
            _get_connector_metadata=lambda: SimpleNamespace(
                request_meta={"r": request}
            ),
            _rank_consistency=SimpleNamespace(wait_load=wait),
            _layerwise_load_bytes=0,
            kv_cache_layout=SimpleNamespace(shard_size=128),
            _record_layerwise_load_duration=lambda *args: False,
        )
        namespace["wait_for_layer_load"](worker, "layer")
        wait.assert_called_once_with(task)
        self.assertEqual(worker._layerwise_load_bytes, 128)
        self.assertEqual(worker.load_tasks, {})
        namespace["wait_for_layer_load"](worker, "layer")
        wait.assert_called_once_with(task)


class DCPRecoveryTests(unittest.TestCase):
    def test_keys_cover_entire_logical_block_and_keep_shard_order(self):
        import hashlib
        import pickle

        connector = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "UCMCPConnector"
        )
        method = next(
            n
            for n in connector.body
            if isinstance(n, ast.FunctionDef) and n.name == "_bind_request_block_hasher"
        )
        source = ast.parse(
            (MODULE.parent / "request_hasher.py").read_text(encoding="utf-8")
        )
        hasher_class = next(
            n
            for n in source.body
            if isinstance(n, ast.ClassDef) and n.name == "RequestHasher"
        )
        ns = {
            "hashlib": hashlib,
            "pickle": pickle,
            "_validate_request_hash_semantics": lambda request: None,
            "_generate_extra_keys": lambda *args: (None, 0),
            "KVConnectorRole": SimpleNamespace(SCHEDULER="scheduler"),
        }
        code = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                hasher_class,
                method,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(code), str(MODULE), "exec"), ns)
        hasher = ns["RequestHasher"]("logical_block_v1")
        worker = SimpleNamespace(
            cp_world_size=2,
            block_size=256,
            _role="scheduler",
            request_hasher=hasher,
            _seed=hasher.seed,
        )
        ns["_bind_request_block_hasher"](worker)
        a = SimpleNamespace(all_token_ids=[1] * 128 + [2] * 128)
        b = SimpleNamespace(all_token_ids=[1] * 128 + [3] * 128)
        ka, kb = worker.request_block_hasher(a), worker.request_block_hasher(b)
        self.assertEqual(len(ka), 2)
        self.assertTrue(set(ka).isdisjoint(kb))
        self.assertEqual(worker.request_block_hasher(a), ka)
        self.assertNotEqual(ka[0], ka[1])
        a.all_token_ids += [4] * 255
        self.assertEqual(worker.request_block_hasher(a), ka)
        a.all_token_ids += [4]
        self.assertEqual(len(worker.request_block_hasher(a)), 4)
        self.assertEqual(worker.request_block_hasher(a)[:2], ka)

    def test_direct_and_layerwise_load_disable_distributed_filler_for_dcp(self):
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and any(k.arg == "distribute_backend_load" for k in n.keywords)
        ]
        self.assertEqual(len(calls), 2)
        for call in calls:
            expr = next(
                k.value for k in call.keywords if k.arg == "distribute_backend_load"
            )
            evaluate = compile(ast.Expression(expr), str(MODULE), "eval")
            for mla in [True, False]:
                for dcp in [1, 2, 8]:
                    self.assertEqual(
                        eval(
                            evaluate,
                            {"self": SimpleNamespace(is_mla=mla, cp_world_size=dcp)},
                        ),
                        mla and dcp == 1,
                    )


if __name__ == "__main__":
    unittest.main()
