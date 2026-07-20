import importlib.util
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

STUBS_PATH = Path(__file__).with_name("test_ucm_connector_metrics.py")
SPEC = importlib.util.spec_from_file_location("ucm_hybrid_test_stubs", STUBS_PATH)
stubs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stubs
SPEC.loader.exec_module(stubs)

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

import ucm.integration.vllm.hla_connector as hla_module
import ucm.integration.vllm.hma_connector as hma_module
from ucm.integration.vllm.hla_connector import (
    UCMHybridLinearAttentionConnector,
    UCMHybridLinearAttentionLayerWiseConnector,
)
from ucm.integration.vllm.hma_connector import (
    FAWALoadTask,
    FAWARequestDispatchMeta,
    UCMFAWAConnector,
    UCMFAWAConnectorMetadata,
)
from ucm.integration.vllm.rank_consistency import RankConsistencyManager
from ucm.integration.vllm.ucm_connector import (
    RequestMeta,
    UCMConnector,
    UCMConnectorMetadata,
    UCMDirectConnector,
    UCMWorkerMetadata,
)
from ucm.store.pipeline.errors import StoreNotFoundError

hla_module.logger.info_once = lambda *args, **kwargs: None
hma_module.logger.info_once = lambda *args, **kwargs: None
stubs.ucm_connector_module.logger.info_once = lambda *args, **kwargs: None


def attach_rank_consistency(connector, *, scheduler_side=False, enabled=True):
    if not scheduler_side:
        connector._connector_worker_meta = UCMWorkerMetadata(
            is_mla=getattr(connector, "is_mla", False)
        )
    connector._rank_consistency = RankConsistencyManager(
        is_scheduler=scheduler_side,
        use_consistency_manager=enabled,
    )
    return connector._rank_consistency


def collect_rank_consistency_meta(manager):
    metadata = UCMWorkerMetadata()
    manager.update_worker_meta(metadata)
    return metadata


class HybridRankConsistencyTest(unittest.TestCase):
    def test_connectors_do_not_bypass_common_transfer_channels(self):
        integration_dir = STUBS_PATH.parents[1] / "ucm" / "integration" / "vllm"
        connector_path = integration_dir / "ucm_connector.py"
        manager_path = integration_dir / "rank_consistency.py"
        connector_source = connector_path.read_text(encoding="utf-8")
        manager_source = manager_path.read_text(encoding="utf-8")

        self.assertNotIn("def _consistent_", connector_source)
        self.assertEqual(manager_source.count("store.load_data("), 1)
        self.assertEqual(manager_source.count("store.dump_data("), 1)
        self.assertEqual(manager_source.count("store.wait(task)"), 2)

        for path in integration_dir.rglob("*.py"):
            if path in (connector_path, manager_path) or "patch" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("StoreNotFoundError", source, path)
            for store_name in ("store", "fa_store", "wa_store"):
                self.assertNotIn(f"self.{store_name}.lookup_on_prefix(", source, path)
                self.assertNotIn(f"self.{store_name}.lookup(", source, path)
            self.assertNotIn(".load_data(", source, path)
            self.assertNotIn(".dump_data(", source, path)
            self.assertNotIn(".wait(", source, path)

    def test_common_load_channel_marks_all_requests_missing_on_submit(self):
        first = b"a" * 16
        second = b"b" * 16

        class Store:
            def load_data(self, block_ids, shard_indices, ptrs):
                raise StoreNotFoundError("missing")

        connector = object.__new__(UCMDirectConnector)
        connector.store = Store()
        connector.tp_rank = 0
        connector.is_mla = False
        manager = attach_rank_consistency(connector)

        with self.assertRaises(StoreNotFoundError):
            manager.submit_load(
                connector.store,
                {"first": [first], "second": [second]},
                [first, second],
                [0, 0],
                object(),
            )

        metadata = collect_rank_consistency_meta(manager)
        self.assertEqual(metadata.load_failed_reqs, set())
        self.assertEqual(metadata.missing_blocks, {first, second})
        self.assertEqual(metadata.missing_reqs, {"first", "second"})

    def test_common_load_channel_uses_registered_store_when_waiting(self):
        key = b"k" * 16
        task = SimpleNamespace(task_id=1)

        class Store:
            def load_data(self, block_ids, shard_indices, ptrs):
                return task

            def wait(self, wait_task):
                self.wait_task = wait_task
                raise StoreNotFoundError("missing")

        store = Store()
        connector = object.__new__(UCMDirectConnector)
        connector.store = SimpleNamespace()
        connector.tp_rank = 0
        connector.is_mla = False
        manager = attach_rank_consistency(connector)

        submitted = manager.submit_load(
            store,
            {"req": [key]},
            [key],
            [0],
            object(),
        )

        self.assertIs(submitted, task)
        with self.assertRaises(StoreNotFoundError):
            manager.wait_load(task)
        self.assertIs(store.wait_task, task)
        self.assertEqual(collect_rank_consistency_meta(manager).missing_blocks, {key})

    def test_common_load_channel_uses_supplied_store_key(self):
        key = b"k" * 16
        store_key = b"r" * 16
        task = object()

        class Store:
            def load_data(self, block_ids, shard_indices, ptrs):
                self.block_ids = block_ids
                return task

        connector = object.__new__(UCMDirectConnector)
        connector.store = Store()
        connector.tp_rank = 1
        connector.is_mla = False
        connector.request_hasher = lambda block_id: self.fail(
            "common load channel must not hash block ids"
        )
        manager = attach_rank_consistency(connector)

        submitted = manager.submit_load(
            connector.store,
            {"req": [key]},
            [store_key],
            [0],
            object(),
        )

        self.assertIs(submitted, task)
        self.assertEqual(connector.store.block_ids, [store_key])

    def test_common_dump_channel_reports_success_after_wait(self):
        key = b"k" * 16
        task = object()

        class Store:
            def dump_data(self, block_ids, shard_indices, ptrs, event_handle):
                return task

            def wait(self, wait_task):
                self.wait_task = wait_task

        connector = object.__new__(UCMDirectConnector)
        connector.store = Store()
        connector.tp_rank = 0
        manager = attach_rank_consistency(connector)

        submitted = manager.submit_dump(
            connector.store, {"req": {key}}, [key], [0], object(), 0
        )

        self.assertIs(submitted, task)
        manager.wait_dump(task)
        manager.finish_dump({"req"})
        self.assertIs(connector.store.wait_task, task)
        self.assertEqual(
            collect_rank_consistency_meta(manager).dump_succeeded_blocks,
            {key},
        )

    def test_common_dump_channel_uses_supplied_store_key_and_keeps_original_id(self):
        key = b"k" * 16
        rank_key = b"r" * 16
        task = object()

        class Store:
            def dump_data(self, block_ids, shard_indices, ptrs, event_handle):
                self.block_ids = block_ids
                return task

            def wait(self, wait_task):
                return None

        connector = object.__new__(UCMDirectConnector)
        connector.store = Store()
        connector.tp_rank = 1
        connector.is_mla = False
        connector.request_hasher = lambda block_id: self.fail(
            "common dump channel must not hash block ids"
        )
        manager = attach_rank_consistency(connector)

        submitted = manager.submit_dump(
            connector.store, {"req": {key}}, [rank_key], [0], object(), 0
        )

        self.assertIs(submitted, task)
        manager.wait_dump(task)
        manager.finish_dump({"req"})
        self.assertEqual(connector.store.block_ids, [rank_key])
        self.assertEqual(
            collect_rank_consistency_meta(manager).dump_succeeded_blocks,
            {key},
        )

    def test_common_dump_channel_does_not_report_submit_failure(self):
        key = b"k" * 16

        class Store:
            def dump_data(self, block_ids, shard_indices, ptrs, event_handle):
                raise RuntimeError("failed")

        connector = object.__new__(UCMDirectConnector)
        connector.store = Store()
        connector.tp_rank = 0
        manager = attach_rank_consistency(connector)

        with self.assertRaisesRegex(RuntimeError, "failed"):
            manager.submit_dump(
                connector.store, {"req": {key}}, [key], [0], object(), 0
            )
        manager.finish_dump({"req"})

        self.assertEqual(
            collect_rank_consistency_meta(manager).dump_succeeded_blocks,
            set(),
        )

    def test_common_dump_channel_rejects_finish_before_wait(self):
        key = b"k" * 16

        class Store:
            def dump_data(self, block_ids, shard_indices, ptrs, event_handle):
                return object()

        connector = object.__new__(UCMDirectConnector)
        connector.store = Store()
        connector.tp_rank = 0
        manager = attach_rank_consistency(connector)

        manager.submit_dump(connector.store, {"req": {key}}, [key], [0], object(), 0)

        with self.assertRaisesRegex(RuntimeError, "still pending"):
            manager.finish_dump({"req"})

    def test_consistent_lookup_on_prefix_filters_inconsistent_suffix(self):
        valid = b"a" * 16
        missing = b"b" * 16
        trailing = b"c" * 16
        looked_up = []
        connector = object.__new__(UCMDirectConnector)
        connector.store = SimpleNamespace(
            lookup_on_prefix=lambda keys: looked_up.extend(keys) or 0
        )
        manager = attach_rank_consistency(connector, scheduler_side=True)
        manager.apply_worker_meta(UCMWorkerMetadata(missing_blocks={missing}))

        result = manager.lookup_on_prefix(connector.store, [valid, missing, trailing])

        self.assertEqual(result, 0)
        self.assertEqual(looked_up, [valid])

    def test_consistent_lookup_on_prefix_skips_store_for_empty_valid_prefix(self):
        missing = b"m" * 16
        connector = object.__new__(UCMDirectConnector)
        connector.store = SimpleNamespace(
            lookup_on_prefix=lambda keys: self.fail("lookup must be skipped")
        )
        manager = attach_rank_consistency(connector, scheduler_side=True)
        manager.apply_worker_meta(UCMWorkerMetadata(missing_blocks={missing}))

        result = manager.lookup_on_prefix(connector.store, [missing])

        self.assertEqual(result, -1)

    def test_consistent_lookup_all_passes_complete_key_list_to_store(self):
        keys = [b"a" * 16, b"b" * 16]
        looked_up = []
        connector = object.__new__(UCMDirectConnector)
        connector.store = SimpleNamespace(
            lookup=lambda block_ids: looked_up.extend(block_ids) or [True, False]
        )
        manager = attach_rank_consistency(connector, scheduler_side=True)

        result = manager.lookup_all(connector.store, keys)

        self.assertEqual(result, [True, False])
        self.assertEqual(looked_up, keys)

    def test_consistent_lookup_all_masks_only_inconsistent_keys_after_store_lookup(
        self,
    ):
        valid = b"a" * 16
        missing = b"b" * 16
        trailing = b"c" * 16
        looked_up = []
        connector = object.__new__(UCMDirectConnector)
        connector.store = SimpleNamespace(
            lookup=lambda keys: looked_up.extend(keys) or [True, True, False]
        )
        manager = attach_rank_consistency(connector, scheduler_side=True)
        manager.apply_worker_meta(UCMWorkerMetadata(missing_blocks={missing}))

        result = manager.lookup_all(connector.store, [valid, missing, trailing])

        self.assertEqual(looked_up, [valid, missing, trailing])
        self.assertEqual(result, [True, False, False])

    def test_prefetch_other_rank_hashes_prefetches_rehashed_keys(self):
        block = b"a" * 16
        prefetched = []
        connector = object.__new__(UCMDirectConnector)
        connector._other_rank_hashers = [
            lambda key: key + b"1",
            lambda key: key + b"2",
        ]
        connector.store = SimpleNamespace(prefetch=lambda keys: prefetched.extend(keys))

        connector._prefetch_other_rank_hashes([block])

        self.assertEqual(prefetched, [block + b"1", block + b"2"])

    def test_direct_connector_get_block_size(self):
        connector = object.__new__(UCMDirectConnector)
        connector.block_size = 16

        self.assertEqual(connector.get_block_size(), 16)

    def test_hla_connector_get_block_size_uses_lcm_block_size(self):
        connector = object.__new__(UCMHybridLinearAttentionConnector)
        connector.block_size = 16
        connector.group_manager = SimpleNamespace(lcm_block_size=64)

        self.assertEqual(connector.get_block_size(), 64)

    def test_hma_connector_get_block_size_uses_hash_block_size(self):
        connector = object.__new__(UCMFAWAConnector)
        connector.block_size = 16
        connector.hash_block_size = 512

        self.assertEqual(connector.get_block_size(), 512)

    def test_connector_facade_get_block_size_uses_selected_connector(self):
        connector = object.__new__(UCMConnector)
        connector.connector = SimpleNamespace(get_block_size=lambda: 128)

        self.assertEqual(connector.get_block_size(), 128)

    def test_missing_load_rewinds_with_connector_block_size(self):
        req_meta = SimpleNamespace(
            hbm_hit_block_num=2,
            total_hit_block_num=5,
            token_processed=320,
        )
        connector = object.__new__(UCMDirectConnector)
        connector.block_size = 16
        connector.get_block_size = lambda: 64
        attach_rank_consistency(connector, scheduler_side=True)
        connector.requests_meta = {"req": req_meta}
        output = SimpleNamespace(
            kv_connector_worker_meta=UCMWorkerMetadata(
                load_failed_reqs={"req"},
                missing_reqs={"req"},
            )
        )

        connector.update_connector_output(output)

        self.assertEqual(req_meta.total_hit_block_num, 2)
        self.assertEqual(req_meta.token_processed, 128)

    def test_missing_load_regenerates_dump_blocks_on_next_step(self):
        ucm_block_ids = [bytes([index]) * 16 for index in range(1, 5)]
        vllm_block_ids = [11, 12, 13, 14]
        req_meta = RequestMeta(
            ucm_block_ids=ucm_block_ids,
            hbm_hit_block_num=1,
            total_hit_block_num=3,
            num_token_ids=64,
            vllm_block_ids=vllm_block_ids,
            token_processed=64,
        )
        connector = object.__new__(UCMDirectConnector)
        connector.block_size = 16
        connector.cp_world_size = 1
        attach_rank_consistency(connector, scheduler_side=True)
        connector.requests_meta = {"req": req_meta}
        output = SimpleNamespace(
            kv_connector_worker_meta=UCMWorkerMetadata(
                load_failed_reqs={"req"},
                missing_reqs={"req"},
            )
        )

        connector.update_connector_output(output)
        dispatch_meta = connector._generate_dispatch_meta(
            req_meta,
            new_tokens=48,
            vllm_block_ids=[],
            need_load=False,
        )

        self.assertEqual(dispatch_meta.load_block_ids, ([], []))
        self.assertEqual(
            dispatch_meta.dump_block_ids,
            (ucm_block_ids[1:], vllm_block_ids[1:]),
        )

    def test_direct_first_inconsistent_key_still_tracks_request_for_dump(self):
        missing = b"m" * 16
        connector = object.__new__(UCMDirectConnector)
        connector.block_size = 1
        connector.hash_block_size = 1
        connector.cp_world_size = 1
        connector.persist_token_threshold = 0
        connector.enable_record_traces = False
        connector.requests_meta = {}
        connector.generate_hash = lambda block_size, token_ids, seed: [missing]
        connector._seed = b"seed"
        connector._other_rank_hashers = []
        manager = attach_rank_consistency(connector, scheduler_side=True)
        manager.apply_worker_meta(UCMWorkerMetadata(missing_blocks={missing}))
        connector.store = SimpleNamespace(
            lookup_on_prefix=lambda keys: self.fail("lookup must be skipped")
        )
        request = SimpleNamespace(
            request_id="repair",
            num_tokens=2,
            all_token_ids=[1, 2],
            max_tokens=1,
        )

        connector.get_num_new_matched_tokens(request, 0)

        self.assertIn("repair", connector.requests_meta)

    def test_direct_not_found_keeps_original_key_after_rank_hashing(self):
        key = b"k" * 16
        hashed_key = b"h" * 16

        class Pointers:
            shape = (1, 1)

            def reshape(self, *shape):
                return self

        class Store:
            def load_data(self, block_ids, shard_indices, ptrs):
                self.loaded = list(block_ids)
                return object()

            def wait(self, task):
                raise StoreNotFoundError("missing")

        metadata = UCMConnectorMetadata(
            request_meta={
                "req": SimpleNamespace(
                    load_block_ids=([key], [1]),
                    dump_block_ids=([], []),
                )
            }
        )
        connector = object.__new__(UCMDirectConnector)
        connector._get_connector_metadata = lambda: metadata
        connector._skip_null_vllm_blocks = False
        connector.tp_rank = 1
        connector.is_mla = False
        connector.request_hasher = lambda block_id: hashed_key
        connector.kv_cache_layout = SimpleNamespace(
            extract_block_addrs=lambda block_ids: Pointers()
        )
        connector.store = Store()
        connector._invalid_block_ids = set()
        manager = attach_rank_consistency(connector)
        connector.block_data_size = 1

        connector.start_load_kv(None)

        self.assertEqual(connector.store.loaded, [hashed_key])
        self.assertEqual(metadata.request_meta["req"].load_block_ids[0], [key])
        self.assertEqual(connector._connector_worker_meta.load_failed_reqs, {"req"})
        self.assertEqual(collect_rank_consistency_meta(manager).missing_blocks, {key})

    def test_hla_scheduler_filters_inconsistent_group_keys(self):
        valid = b"a" * 16
        missing = b"b" * 16

        class GroupManager:
            lcm_block_size = 1
            num_groups = 1
            full_attn_groups = [SimpleNamespace(group_id=0, block_size=1)]

            def compute_all_group_block_ids(self, token_ids):
                return [[valid, missing]]

            def lookup_external_hit_tokens(
                self,
                num_computed_tokens,
                group_block_ids,
                lookup_on_prefix,
                lookup_all,
            ):
                keys = group_block_ids[0]
                self.prefix_result = lookup_on_prefix(keys)
                self.exact_result = lookup_all(keys)
                return self.prefix_result + 1, self.prefix_result + 1, []

        connector = object.__new__(UCMHybridLinearAttentionConnector)
        connector.group_manager = GroupManager()
        connector.persist_token_threshold = 0
        connector.enable_record_traces = False
        connector.requests_meta = {}
        looked_up = []
        exact_looked_up = []
        connector.store = SimpleNamespace(
            lookup_on_prefix=lambda keys: looked_up.extend(keys) or 0,
            lookup=lambda keys: exact_looked_up.extend(keys) or [True] * len(keys),
            prefetch=lambda keys: None,
        )
        connector._other_rank_hashers = []
        manager = attach_rank_consistency(connector, scheduler_side=True)
        manager.apply_worker_meta(UCMWorkerMetadata(missing_blocks={missing}))
        request = SimpleNamespace(
            request_id="repair",
            num_tokens=3,
            all_token_ids=[1, 2, 3],
            max_tokens=1,
        )

        connector.get_num_new_matched_tokens(request, 0)

        self.assertEqual(looked_up, [valid])
        self.assertEqual(exact_looked_up, [valid, missing])
        self.assertEqual(connector.group_manager.prefix_result, 0)
        self.assertEqual(connector.group_manager.exact_result, [True, False])

    def test_hla_direct_successful_dump_reports_original_key(self):
        key = b"k" * 16

        class Pointers:
            shape = (1, 1)

            def reshape(self, *shape):
                return self

        metadata = UCMConnectorMetadata(
            request_meta={"req": SimpleNamespace(dump_block_ids=([key], [1]))}
        )
        connector = object.__new__(UCMHybridLinearAttentionConnector)
        connector._get_connector_metadata = lambda: metadata
        connector._async_dump_req_ids = set()
        connector._pending_dump_tasks = []
        connector._skip_null_vllm_blocks = True
        connector.is_mla = False
        connector.tp_rank = 0
        manager = attach_rank_consistency(connector)
        connector.tp_size = 1
        connector.enable_event_sync = False
        connector.device = SimpleNamespace(synchronize=lambda: None)
        connector.block_data_size = 1
        connector.kv_cache_layout = SimpleNamespace(
            extract_block_addrs=lambda block_ids: Pointers()
        )
        connector.store = SimpleNamespace(
            dump_data=lambda *args: object(),
            wait=lambda task: None,
        )

        connector.wait_for_save()
        finished, metadata = connector.request_finished_all_groups(
            SimpleNamespace(request_id="req"), ([1],)
        )

        self.assertFalse(finished)
        self.assertIsNone(metadata)
        self.assertEqual(
            collect_rank_consistency_meta(manager).dump_succeeded_blocks,
            {key},
        )

    def test_hla_layerwise_submit_not_found_reports_original_keys(self):
        key = b"k" * 16
        metadata = UCMConnectorMetadata(
            request_meta={
                "req": SimpleNamespace(
                    load_block_ids=([key], [1]),
                    dump_block_ids=([], []),
                )
            }
        )

        class Store:
            def load_data(self, block_ids, shard_indices, ptrs):
                raise StoreNotFoundError("missing")

        connector = object.__new__(UCMHybridLinearAttentionLayerWiseConnector)
        connector.request_data = [("req", [key], [key], [1])]
        connector._failure_req_ids = set()
        connector._submitted_load_rows = set()
        connector.load_tasks = defaultdict(list)
        connector.kv_cache_layout = SimpleNamespace(
            extract_block_addrs_for_row=lambda block_ids, row_id: []
        )
        connector.store = Store()
        connector._invalid_block_ids = set()
        connector._skip_null_vllm_blocks = True
        connector.tp_rank = 2
        manager = attach_rank_consistency(connector)
        connector.tp_size = 4
        connector.is_mla = False
        connector.request_hasher = lambda block_id: b"hashed"

        connector._submit_request_load_tasks_for_row(0, metadata)

        self.assertEqual(collect_rank_consistency_meta(manager).missing_blocks, {key})

    def test_hla_layerwise_not_found_ignores_null_vllm_blocks(self):
        skipped = b"s" * 16
        loaded = b"l" * 16
        metadata = UCMConnectorMetadata(
            request_meta={
                "req": SimpleNamespace(
                    load_block_ids=([skipped, loaded], [0, 1]),
                    dump_block_ids=([], []),
                )
            }
        )

        class Store:
            def load_data(self, block_ids, shard_indices, ptrs):
                raise StoreNotFoundError("missing")

        connector = object.__new__(UCMHybridLinearAttentionLayerWiseConnector)
        connector.request_data = [("req", [loaded], [loaded], [1])]
        connector._invalid_block_ids = set()
        connector._failure_req_ids = set()
        connector._submitted_load_rows = set()
        connector.load_tasks = defaultdict(list)
        connector.kv_cache_layout = SimpleNamespace(
            extract_block_addrs_for_row=lambda block_ids, row_id: []
        )
        connector.store = Store()
        connector._skip_null_vllm_blocks = True
        connector.tp_rank = 2
        manager = attach_rank_consistency(connector)
        connector.tp_size = 4
        connector.is_mla = False
        connector.request_hasher = lambda block_id: b"hashed"

        connector._submit_request_load_tasks_for_row(0, metadata)

        self.assertEqual(
            collect_rank_consistency_meta(manager).missing_blocks, {loaded}
        )

    def test_hma_scheduler_filters_inconsistent_keys(self):
        valid = b"a" * 16
        missing = b"b" * 16
        looked_up = []
        connector = object.__new__(UCMFAWAConnector)
        connector.hash_block_size = 1
        connector.persist_token_threshold = 0
        connector.load_tokens_threshold = 0
        connector.enable_record_traces = False
        connector.requests_meta = {}
        connector.generate_hash = lambda block_size, token_ids, seed: [valid, missing]
        connector._seed = b"seed"
        manager = attach_rank_consistency(connector, scheduler_side=True)
        manager.apply_worker_meta(UCMWorkerMetadata(missing_blocks={missing}))
        connector._fawa_stats_enabled = False
        connector.fa_store = SimpleNamespace(
            lookup_on_prefix=lambda keys: looked_up.extend(keys) or len(keys) - 1
        )
        connector.wa_store = SimpleNamespace(lookup=lambda keys: [True] * len(keys))
        request = SimpleNamespace(
            request_id="repair",
            num_tokens=3,
            all_token_ids=[1, 2, 3],
        )

        connector.get_num_new_matched_tokens(request, 0)

        self.assertEqual(looked_up, [valid])

    def test_hma_first_inconsistent_key_still_tracks_request_for_dump(self):
        missing = b"m" * 16
        connector = object.__new__(UCMFAWAConnector)
        connector.hash_block_size = 1
        connector.persist_token_threshold = 0
        connector.load_tokens_threshold = 0
        connector.enable_record_traces = False
        connector.requests_meta = {}
        connector.generate_hash = lambda block_size, token_ids, seed: [missing]
        connector._seed = b"seed"
        manager = attach_rank_consistency(connector, scheduler_side=True)
        manager.apply_worker_meta(UCMWorkerMetadata(missing_blocks={missing}))
        connector._fawa_stats_enabled = False
        connector.fa_store = SimpleNamespace(
            lookup_on_prefix=lambda keys: self.fail("lookup must be skipped")
        )
        connector.wa_store = SimpleNamespace(
            lookup=lambda keys: self.fail("lookup must be skipped")
        )
        request = SimpleNamespace(
            request_id="repair",
            num_tokens=2,
            all_token_ids=[1, 2],
        )

        connector.get_num_new_matched_tokens(request, 0)

        self.assertIn("repair", connector.requests_meta)

    def test_hma_wait_not_found_reports_load_keys(self):
        key = b"k" * 16
        metadata = UCMFAWAConnectorMetadata(
            request_meta={
                "req": FAWARequestDispatchMeta(
                    load_keys=[key],
                    load_vllm_block_ids=([1],),
                )
            }
        )

        class Store:
            def load_data(self, block_ids, shard_indices, ptrs):
                return object()

            def wait(self, task):
                raise StoreNotFoundError("missing")

        connector = object.__new__(UCMFAWAConnector)
        connector._get_connector_metadata = lambda: metadata
        connector._record_load_error = lambda *args: None
        connector.tp_rank = 3
        manager = attach_rank_consistency(connector)
        load_task = connector._submit_load_task("req", "FA", Store(), [key], object())
        self.assertIsNotNone(load_task)

        connector._wait_load_task(load_task)

        self.assertEqual(collect_rank_consistency_meta(manager).missing_blocks, {key})

    def test_hma_store_config_does_not_inject_global_rank(self):
        connector = object.__new__(UCMFAWAConnector)
        connector.connector_configs = [
            {"ucm_connector_name": "Fake", "ucm_connector_config": {}}
        ]
        connector.engine_id = "engine"
        connector.is_mla = False
        connector._role = KVConnectorRole.WORKER
        connector._vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(rank=7, data_parallel_rank=0)
        )
        connector._apply_sdma_direct_launch_granularity = lambda config: None

        _, _, config = connector._base_store_config("fa")

        self.assertNotIn("rank", config)

    def test_hma_owner_rank_reports_dump_success(self):
        key = b"k" * 16
        block_ids = {"req": {key}}

        class Store:
            def dump_data(self, keys, shard_indices, ptrs, event_handle):
                return object()

            def wait(self, task):
                return None

        connector = object.__new__(UCMFAWAConnector)
        connector.store = Store()
        connector.tp_rank = 3
        manager = attach_rank_consistency(connector)

        tasks = [
            manager.submit_dump(connector.store, block_ids, [key], [0], object(), 0)
            for _ in ("FA", "WA")
        ]
        for task in tasks:
            self.assertIsNotNone(task)
            manager.wait_dump(task)
        manager.finish_dump({"req"})

        self.assertEqual(
            collect_rank_consistency_meta(manager).dump_succeeded_blocks,
            {key},
        )

    def test_hma_owner_rank_does_not_report_partial_dump_success(self):
        key = b"k" * 16
        block_ids = {"req": {key}}

        class Store:
            def __init__(self):
                self.tasks = []

            def dump_data(self, keys, shard_indices, ptrs, event_handle):
                task = object()
                self.tasks.append(task)
                return task

            def wait(self, task):
                if task is self.tasks[1]:
                    raise RuntimeError("failed")

        connector = object.__new__(UCMFAWAConnector)
        connector.store = Store()
        connector.tp_rank = 3
        manager = attach_rank_consistency(connector)

        tasks = [
            manager.submit_dump(connector.store, block_ids, [key], [0], object(), 0)
            for _ in ("FA", "WA")
        ]
        for task in tasks:
            self.assertIsNotNone(task)
        manager.wait_dump(tasks[0])
        with self.assertRaisesRegex(RuntimeError, "failed"):
            manager.wait_dump(tasks[1])
        manager.finish_dump({"req"})

        self.assertEqual(
            collect_rank_consistency_meta(manager).dump_succeeded_blocks,
            set(),
        )


if __name__ == "__main__":
    unittest.main()
