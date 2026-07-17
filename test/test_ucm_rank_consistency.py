# MIT License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch


class TestStoreNotFoundError(RuntimeError):
    pass


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "ucm"
    / "integration"
    / "vllm"
    / "rank_consistency.py"
)
SPEC = importlib.util.spec_from_file_location("ucm_rank_consistency", MODULE_PATH)
rank_consistency = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rank_consistency
ucm_module = ModuleType("ucm")
ucm_module.__path__ = []
logger_module = ModuleType("ucm.logger")
logger_module.init_logger = lambda _: MagicMock()
store_package = ModuleType("ucm.store")
store_package.__path__ = []
store_module = ModuleType("ucm.store.ucmstore_v1")
store_module.UcmKVStoreBaseV1 = type("UcmKVStoreBaseV1", (), {})
pipeline_package = ModuleType("ucm.store.pipeline")
pipeline_package.__path__ = []
errors_module = ModuleType("ucm.store.pipeline.errors")
errors_module.StoreNotFoundError = TestStoreNotFoundError
original_ucm = sys.modules.get("ucm")
original_logger = sys.modules.get("ucm.logger")
original_store_package = sys.modules.get("ucm.store")
original_store_module = sys.modules.get("ucm.store.ucmstore_v1")
original_pipeline_package = sys.modules.get("ucm.store.pipeline")
original_errors_module = sys.modules.get("ucm.store.pipeline.errors")
try:
    sys.modules["ucm"] = ucm_module
    sys.modules["ucm.logger"] = logger_module
    sys.modules["ucm.store"] = store_package
    sys.modules["ucm.store.ucmstore_v1"] = store_module
    sys.modules["ucm.store.pipeline"] = pipeline_package
    sys.modules["ucm.store.pipeline.errors"] = errors_module
    SPEC.loader.exec_module(rank_consistency)
finally:
    if original_ucm is None:
        sys.modules.pop("ucm", None)
    else:
        sys.modules["ucm"] = original_ucm
    if original_logger is None:
        sys.modules.pop("ucm.logger", None)
    else:
        sys.modules["ucm.logger"] = original_logger
    if original_store_package is None:
        sys.modules.pop("ucm.store", None)
    else:
        sys.modules["ucm.store"] = original_store_package
    if original_store_module is None:
        sys.modules.pop("ucm.store.ucmstore_v1", None)
    else:
        sys.modules["ucm.store.ucmstore_v1"] = original_store_module
    if original_pipeline_package is None:
        sys.modules.pop("ucm.store.pipeline", None)
    else:
        sys.modules["ucm.store.pipeline"] = original_pipeline_package
    if original_errors_module is None:
        sys.modules.pop("ucm.store.pipeline.errors", None)
    else:
        sys.modules["ucm.store.pipeline.errors"] = original_errors_module
RankConsistencyTracker = rank_consistency.RankConsistencyTracker


class TestWorkerMetadata:
    def __init__(self):
        self.load_failed_reqs = set()
        self.missing_reqs = set()
        self.missing_blocks = set()
        self.dump_succeeded_blocks = set()

    def mark_failed(self, request_id):
        self.load_failed_reqs.add(request_id)

    def mark_missing(self, request_id, block_ids):
        self.missing_reqs.add(request_id)
        self.missing_blocks.update(block_ids)


class RankConsistencyManagerTest(unittest.TestCase):
    def _make_manager(self, *, scheduler_side=False, enabled=True):
        manager_type = getattr(rank_consistency, "RankConsistencyManager", None)
        self.assertIsNotNone(manager_type)
        return manager_type(
            is_scheduler=scheduler_side,
            use_consistency_manager=enabled,
        )

    def test_scheduler_lookup_masks_missing_blocks(self):
        valid, missing, trailing = b"a" * 16, b"b" * 16, b"c" * 16
        prefix_lookups = []
        exact_lookups = []

        class Store:
            def lookup_on_prefix(self, block_ids):
                prefix_lookups.extend(block_ids)
                return len(block_ids) - 1

            def lookup(self, block_ids):
                exact_lookups.extend(block_ids)
                return [True] * len(block_ids)

        manager = self._make_manager(scheduler_side=True)
        metadata = TestWorkerMetadata()
        metadata.missing_blocks.add(missing)
        manager.apply_worker_meta(metadata)

        prefix_result = manager.lookup_on_prefix(Store(), [valid, missing, trailing])
        exact_result = manager.lookup_all(Store(), [valid, missing, trailing])

        self.assertEqual(prefix_result, 0)
        self.assertEqual(prefix_lookups, [valid])
        self.assertEqual(exact_lookups, [valid, missing, trailing])
        self.assertEqual(exact_result, [True, False, True])

    def test_disabled_scheduler_lookup_does_not_mask_missing_blocks(self):
        valid, missing = b"a" * 16, b"b" * 16
        prefix_lookups = []

        class Store:
            def lookup_on_prefix(self, block_ids):
                prefix_lookups.extend(block_ids)
                return len(block_ids) - 1

            def lookup(self, block_ids):
                return [True] * len(block_ids)

        manager = self._make_manager(scheduler_side=True, enabled=False)
        metadata = TestWorkerMetadata()
        metadata.missing_blocks.add(missing)
        manager.apply_worker_meta(metadata)

        self.assertEqual(manager.lookup_on_prefix(Store(), [valid, missing]), 1)
        self.assertEqual(manager.lookup_all(Store(), [valid, missing]), [True, True])
        self.assertEqual(prefix_lookups, [valid, missing])

    def test_load_not_found_supplements_worker_metadata(self):
        block = b"a" * 16

        class Store:
            def load_data(self, block_ids, shard_indices, ptrs):
                raise TestStoreNotFoundError("missing")

        manager = self._make_manager()

        with self.assertRaises(TestStoreNotFoundError):
            manager.submit_load(Store(), {"request": [block]}, [block], [0], None)

        metadata = TestWorkerMetadata()
        manager.update_worker_meta(metadata)
        self.assertEqual(metadata.load_failed_reqs, set())
        self.assertEqual(metadata.missing_reqs, {"request"})
        self.assertEqual(metadata.missing_blocks, {block})

    def test_disabled_load_not_found_does_not_supplement_worker_metadata(self):
        block = b"a" * 16

        class Store:
            def load_data(self, block_ids, shard_indices, ptrs):
                raise TestStoreNotFoundError("missing")

        manager = self._make_manager(enabled=False)

        with self.assertRaises(TestStoreNotFoundError):
            manager.submit_load(Store(), {"request": [block]}, [block], [0], None)

        metadata = TestWorkerMetadata()
        manager.update_worker_meta(metadata)
        self.assertEqual(metadata.missing_reqs, set())
        self.assertEqual(metadata.missing_blocks, set())

    def test_load_submit_error_does_not_supplement_worker_metadata(self):
        block = b"a" * 16

        class Store:
            def load_data(self, block_ids, shard_indices, ptrs):
                raise RuntimeError("submit failed")

        manager = self._make_manager()

        with self.assertRaisesRegex(RuntimeError, "submit failed"):
            manager.submit_load(Store(), {"request": [block]}, [block], [0], None)

        metadata = TestWorkerMetadata()
        manager.update_worker_meta(metadata)
        self.assertEqual(metadata.load_failed_reqs, set())
        self.assertEqual(metadata.missing_reqs, set())
        self.assertEqual(metadata.missing_blocks, set())

    def test_load_wait_error_does_not_supplement_worker_metadata(self):
        block = b"a" * 16
        task = object()

        class Store:
            def load_data(self, block_ids, shard_indices, ptrs):
                return task

            def wait(self, wait_task):
                raise RuntimeError("wait failed")

        manager = self._make_manager()
        manager.submit_load(Store(), {"request": [block]}, [block], [0], None)

        with self.assertRaisesRegex(RuntimeError, "wait failed"):
            manager.wait_load(task)

        metadata = TestWorkerMetadata()
        manager.update_worker_meta(metadata)
        self.assertEqual(metadata.load_failed_reqs, set())
        self.assertEqual(metadata.missing_reqs, set())
        self.assertEqual(metadata.missing_blocks, set())

    def test_dump_result_is_aggregated_across_stores(self):
        first, second = b"a" * 16, b"b" * 16

        class Task:
            pass

        class Store:
            def __init__(self, fail_wait=False):
                self.fail_wait = fail_wait

            def dump_data(self, block_ids, shard_indices, ptrs, event_handle):
                return Task()

            def wait(self, task):
                if self.fail_wait:
                    raise RuntimeError("failed")

        manager = self._make_manager()
        first_task = manager.submit_dump(
            Store(), {"request": {first}}, [first], [0], None, 0
        )
        second_task = manager.submit_dump(
            Store(fail_wait=True),
            {"request": {second}},
            [second],
            [0],
            None,
            0,
        )

        manager.wait_dump(first_task)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            manager.wait_dump(second_task)
        manager.finish_dump({"request"})

        metadata = TestWorkerMetadata()
        manager.update_worker_meta(metadata)
        self.assertEqual(metadata.dump_succeeded_blocks, {first})

    def test_disabled_dump_does_not_report_success(self):
        block = b"a" * 16
        task = object()

        class Store:
            def dump_data(self, block_ids, shard_indices, ptrs, event_handle):
                return task

            def wait(self, wait_task):
                self.wait_task = wait_task

        store = Store()
        manager = self._make_manager(enabled=False)
        submitted = manager.submit_dump(
            store, {"request": {block}}, [block], [0], None, 0
        )
        manager.wait_dump(submitted)
        manager.finish_dump({"request"})

        metadata = TestWorkerMetadata()
        manager.update_worker_meta(metadata)
        self.assertIs(store.wait_task, task)
        self.assertEqual(metadata.dump_succeeded_blocks, set())


class RankConsistencyTrackerTest(unittest.TestCase):
    def test_default_limit_is_two_million_blocks(self):
        self.assertEqual(
            getattr(RankConsistencyTracker, "_MAX_INCONSISTENT_BLOCKS", None),
            2_000_000,
        )

    def test_mark_missing_evicts_at_limit_and_warns_once_per_batch(self):
        tracker = RankConsistencyTracker()
        first, second = b"a" * 16, b"b" * 16
        third, fourth = b"c" * 16, b"d" * 16

        with (
            patch.object(
                RankConsistencyTracker, "_MAX_INCONSISTENT_BLOCKS", 2, create=True
            ),
            patch.object(rank_consistency, "logger", create=True) as logger,
        ):
            tracker.mark_missing({first, second})
            tracker.mark_missing({third, fourth})

        self.assertEqual(len(tracker._inconsistent), 2)
        self.assertTrue(tracker._inconsistent & {third, fourth})
        logger.warning_limit.assert_called_once_with(
            "Rank consistency tracker reached its %d-block limit; "
            "evicted %d inconsistent block records",
            2,
            2,
        )

    def test_marking_existing_block_at_limit_does_not_evict_or_warn(self):
        tracker = RankConsistencyTracker()
        first, second = b"a" * 16, b"b" * 16

        with (
            patch.object(
                RankConsistencyTracker, "_MAX_INCONSISTENT_BLOCKS", 2, create=True
            ),
            patch.object(rank_consistency, "logger", create=True) as logger,
        ):
            tracker.mark_missing({first, second})
            tracker.mark_missing({first})

        self.assertEqual(tracker._inconsistent, {first, second})
        logger.warning_limit.assert_not_called()

    def test_membership_reports_only_marked_blocks(self):
        tracker = RankConsistencyTracker()
        missing, valid = b"m" * 16, b"v" * 16
        tracker.mark_missing({missing})

        self.assertIn(missing, tracker)
        self.assertNotIn(valid, tracker)

        tracker.clear_dumped({missing})

        self.assertNotIn(missing, tracker)

    def test_valid_prefix_truncates_at_first_marked_block(self):
        tracker = RankConsistencyTracker()
        prefix, missing, tail = b"a" * 16, b"b" * 16, b"c" * 16
        tracker.mark_missing({missing})

        self.assertEqual(tracker.valid_prefix([prefix, missing, tail]), [prefix])
        self.assertEqual(tracker.valid_prefix([missing, prefix]), [])
        self.assertEqual(tracker.valid_prefix([prefix, tail]), [prefix, tail])

    def test_clear_dumped_restores_lookup(self):
        tracker = RankConsistencyTracker()
        block = b"x" * 16
        tracker.mark_missing({block})
        self.assertEqual(tracker.valid_prefix([block]), [])

        tracker.clear_dumped({block})

        self.assertEqual(tracker.valid_prefix([block]), [block])

    def test_clear_dumped_ignores_unmarked_blocks(self):
        tracker = RankConsistencyTracker()
        marked, unrelated = b"m" * 16, b"u" * 16
        tracker.mark_missing({marked})

        tracker.clear_dumped({unrelated})

        self.assertEqual(tracker.valid_prefix([marked]), [])

    def test_same_step_mark_survives_stale_dump_success(self):
        # update_connector_output applies clear_dumped() before mark_missing()
        # within one scheduler step, so a fresh missing mark outlives a stale
        # dump success reported together with it.
        tracker = RankConsistencyTracker()
        block = b"x" * 16
        tracker.mark_missing({block})

        tracker.clear_dumped({block})
        tracker.mark_missing({block})

        self.assertEqual(tracker.valid_prefix([block]), [])


if __name__ == "__main__":
    unittest.main()
