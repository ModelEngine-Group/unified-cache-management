"""Hardware-free tests for workload shapes and the actual hit-source protocol."""

import contextlib
import ctypes
import importlib.util
import io
import json
import multiprocessing
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "ucm/store/test/e2e/cache_bw.py"
SPEC = importlib.util.spec_from_file_location("cache_bw_benchmark", SCRIPT)
bw = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bw
SPEC.loader.exec_module(bw)


class FakeTensor:
    def __init__(self, size):
        self.data = bytearray(size)

    def fill_(self, value):
        self.data[:] = bytes([value]) * len(self.data)

    def zero_(self):
        self.fill_(0)

    def data_ptr(self):
        return ctypes.addressof(ctypes.c_char.from_buffer(self.data))

    def __eq__(self, value):
        return FakeComparison(all(byte == value for byte in self.data))


class FakeComparison:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class MemoryStore:
    """Read real aligned seed pointers; distinguish cache hits from disk reads."""

    def __init__(self, workload, disk=None):
        self.workload = workload
        self.disk = disk
        self.cache = {}
        self.sources = []
        self.records = []
        self.store_ = self

    def wait(self, _task):
        pass

    def dump_data(self, ids, shards, pointers):
        for block_id, shard, row in zip(ids, shards, pointers):
            assert row[0] % 4096 == 0
            self.disk[block_id, shard] = ctypes.string_at(
                row[0], self.workload.shard_size
            )

    def unpack(self, ids, shards, pointers):
        # Check the real pybind buffer ABI, not just a list of fake tensors.
        ids_view, indexes_view, address_view = map(memoryview, (ids, shards, pointers))
        assert ids_view.ndim == indexes_view.ndim == 1
        assert indexes_view.itemsize == address_view.itemsize == 8
        assert address_view.shape == (len(shards), len(self.workload.tensor_sizes))
        assert address_view.c_contiguous
        assert ids_view.nbytes == 16 * len(shards)
        return [ids[index : index + 16] for index in range(0, len(ids), 16)]

    def Dump(self, ids, shards, pointers):
        ids = self.unpack(ids, shards, pointers)
        for block_id, shard, row in zip(ids, shards, pointers):
            self.cache[block_id, shard] = b"".join(
                ctypes.string_at(pointer, size)
                for pointer, size in zip(row, self.workload.tensor_sizes)
            )

    def Load(self, ids, shards, pointers):
        ids = self.unpack(ids, shards, pointers)
        self.records.append(tuple(ids))
        for block_id, shard, row in zip(ids, shards, pointers):
            key = block_id, shard
            if key in self.cache:
                self.sources.append("cache")
            else:
                if self.disk is None:
                    raise RuntimeError("cache miss without backend")
                self.cache[key] = self.disk[key]
                self.sources.append("posix")
            data = self.cache[key]
            offset = 0
            for pointer, size in zip(row, self.workload.tensor_sizes):
                ctypes.memmove(pointer, data[offset : offset + size], size)
                offset += size

    def Wait(self, task):
        self.wait(task)


def make_fake_rows(_torch, workload, _device, count):
    return [[FakeTensor(size) for size in workload.tensor_sizes] for _ in range(count)]


def process_success(_phase, _args, _workload, _run_id, rank, barrier, results):
    barrier.wait()
    results[rank] = rank + 1


def process_failure(_phase, _args, _workload, _run_id, rank, barrier, _results):
    if rank == 0:
        raise SystemExit(7)
    barrier.wait()


def process_stall(*_args):
    time.sleep(10)


class WorkerContext:
    """Use real spawned processes while replacing only the accelerator workload."""

    def __init__(self, target):
        self.context = multiprocessing.get_context("spawn")
        self.target = target

    def Barrier(self, *args, **kwargs):
        return self.context.Barrier(*args, **kwargs)

    def Process(self, *, target, args):
        return self.context.Process(target=self.target, args=args)


class CacheBandwidthTest(unittest.TestCase):
    def parse(self, mode="cache-hit", *extra):
        return bw.parse_args(["--mode", mode, *extra])

    def small(self, mode):
        return self.parse(
            mode,
            "--model",
            "custom",
            "--io-size",
            "4K",
            "--layers",
            "3",
            "--layout",
            "layerwise",
            "--blocks",
            "3",
            "--workers",
            "1",
            "--warmup",
            "1",
            "--epochs",
            "2",
            "--epoch-interval-ms",
            "0",
            "--verify",
        )

    def test_glm_layerwise_is_78_distinct_shards(self):
        args, workload = self.parse()
        self.assertEqual(workload.tensor_sizes, (131072, 16384, 32768))
        self.assertEqual(workload.shard_size, 180224)
        self.assertEqual(workload.block_size, 180224 * 78)
        self.assertEqual(workload.shards, 78)
        self.assertEqual(args.workers, 16)

    def test_glm_whole_block_and_single_layer(self):
        _, whole = self.parse(
            "cache-hit", "--layout", "whole-block", "--cache-capacity-gb", "64"
        )
        _, single = self.parse("cache-hit", "--layout", "single-layer")
        self.assertEqual(len(whole.tensor_sizes), 3 * 78)
        self.assertEqual(whole.shards, 1)
        self.assertEqual(whole.payload_size, 180224 * 78)
        self.assertEqual(single.payload_size, 180224)
        self.assertEqual(single.shards, 1)

    def test_dsv4_keeps_the_existing_fa_tensor_layout_and_padding(self):
        _, workload = self.parse("cache-hit", "--model", "dsv4")
        self.assertEqual(
            workload.tensor_sizes, tuple([131072, 16384, 256] * 21 + [4096] * 20)
        )
        self.assertEqual(workload.shards, 1)
        self.assertGreater(workload.shard_size, workload.payload_size)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse("cache-hit", "--model", "dsv4", "--layout", "layerwise")

    def test_io_size_and_tensor_size_overrides(self):
        self.assertEqual(bw.byte_size("176KiB"), 180224)
        _, workload = self.parse("cache-hit", "--tensor-sizes", "128K,16K,32K")
        self.assertEqual(workload.tensor_sizes, (131072, 16384, 32768))
        _, workload = self.parse("cache-hit", "--io-size", "4097")
        self.assertEqual(workload.payload_size, 4097)
        self.assertEqual(workload.shard_size, 8192)

    def test_config_modes_and_explicit_copy_switches(self):
        args, workload = self.parse("cache-hit", "--cache-copy", "plain")
        config = bw.store_config(args, workload, "run", 0)
        self.assertEqual(config["store_pipeline"], "Cache")
        self.assertFalse(config["cache_sdma_direct"])
        self.assertFalse(config["cache_io_aggregation"])
        self.assertNotIn("cache_load_backend_only", config)
        self.assertNotIn("cpu_affinity_cores", config)
        self.assertNotIn("cache_stream_number", config)
        args.cache_copy = "sdma"
        self.assertTrue(bw.store_config(args, workload, "run", 0)["cache_sdma_direct"])
        args.cache_copy = "aggregation"
        config = bw.store_config(args, workload, "run", 0)
        self.assertTrue(config["cache_io_aggregation"])
        self.assertFalse(config["cache_sdma_direct"])

    def test_posix_seed_and_load_have_distinct_domains_but_identical_files(self):
        args, workload = self.parse("posix-hit")
        seed = bw.store_config(args, workload, "run", 2, seed=True)
        load = bw.store_config(args, workload, "run", 2)
        self.assertEqual(seed["store_pipeline"], "Posix")
        self.assertEqual(load["store_pipeline"], "Cache|Posix")
        self.assertNotEqual(seed["unique_id"], load["unique_id"])
        self.assertEqual(seed["storage_backends"], load["storage_backends"])
        self.assertEqual(seed["tensor_size"], workload.shard_size)
        self.assertEqual(seed["block_size"], load["block_size"])
        self.assertFalse(seed["posix_gc_enable"])

    def test_mla_ownership_is_disjoint_complete_and_shared_keys_match(self):
        args, _ = self.parse()
        pieces = [set(bw.owned_indexes(args, rank)) for rank in range(16)]
        self.assertTrue(all(len(piece) == 32 for piece in pieces))
        self.assertEqual(set.union(*pieces), set(range(512)))
        self.assertEqual(sum(map(len, pieces)), 512)
        self.assertEqual(
            bw.block_ids(args, "run", 0, 0), bw.block_ids(args, "run", 0, 15)
        )
        self.assertTrue(
            set(bw.block_ids(args, "run", 0, 0)).isdisjoint(
                bw.block_ids(args, "run", 1, 0)
            )
        )

    def test_gqa_keys_are_private_and_numa_fallback_uses_worker_count(self):
        args, workload = self.parse("cache-hit", "--model", "minimax-m2.7-tp8")
        self.assertTrue(
            set(bw.block_ids(args, "run", 0, 0)).isdisjoint(
                bw.block_ids(args, "run", 0, 1)
            )
        )
        self.assertEqual(len(bw.owned_indexes(args, 1)), 512)
        config = bw.store_config(args, workload, "run", 7)
        self.assertEqual(config["cache_fallback_numa_rank"], 7)
        self.assertEqual(config["cache_fallback_numa_rank_count"], 8)
        self.assertFalse(config["share_buffer_enable"])

    def test_explicit_numa_list_and_non_power_of_two_rank_count(self):
        args, workload = self.parse(
            "cache-hit", "--workers", "3", "--cache-numa-nodes", "6,4,2"
        )
        config = bw.store_config(args, workload, "run", 1)
        self.assertEqual(config["cache_detected_numa_node"], 4)
        self.assertEqual(config["cache_load_exclusive_buffer_number"] % 3, 0)

    def test_invalid_inputs_are_rejected_before_spawning(self):
        cases = [
            ["--epochs", "0"],
            ["--warmup", "-1"],
            ["--cache-streams", "33"],
            ["--epoch-interval-ms", "nan"],
            ["--cache-numa-nodes", "0,1"],
            ["--posix-engine", "aio", "--no-direct-io"],
            ["--io-size", "0"],
            ["--tensor-sizes", "4K,garbage"],
            ["--cache-capacity-gb", "1", "--blocks", "100000"],
        ]
        for case in cases:
            with (
                self.subTest(case=case),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.parse("posix-hit" if "aio" in case else "cache-hit", *case)

    def test_posix_rejects_a_task_larger_than_its_cache_partition(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse(
                "posix-hit",
                "--model",
                "custom",
                "--workers",
                "16",
                "--blocks",
                "512",
                "--cache-capacity-gb",
                "2",
            )

    def test_cache_mode_only_loads_ready_cache_entries(self):
        args, workload = self.small("cache-hit")
        store = MemoryStore(workload)
        runtime, barrier = mock.Mock(), mock.Mock()
        results = [0.0] * (args.epochs * workload.shards)
        with (
            mock.patch.object(bw, "open_store", return_value=store),
            mock.patch.object(bw, "setup_device", return_value=(None, runtime, "mock")),
            mock.patch.object(bw, "make_rows", side_effect=make_fake_rows),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            bw.load_worker(args, workload, "run", 0, barrier, results)
        self.assertEqual(set(store.sources), {"cache"})
        self.assertEqual(len(set(store.records)), 1)
        self.assertTrue(all(cost > 0 for cost in results))

    def test_cache_parameters_are_prepared_once_and_pause_is_between_requests(self):
        args, workload = self.small("cache-hit")
        args.epoch_interval_ms = 15
        store = MemoryStore(workload)
        results = [0.0] * (args.epochs * workload.shards)
        with (
            mock.patch.object(bw, "open_store", return_value=store),
            mock.patch.object(
                bw, "setup_device", return_value=(None, mock.Mock(), "mock")
            ),
            mock.patch.object(bw, "make_rows", side_effect=make_fake_rows),
            mock.patch.object(
                bw, "prepare_transfer", wraps=bw.prepare_transfer
            ) as prepare,
            mock.patch.object(bw, "block_ids", wraps=bw.block_ids) as ids,
            mock.patch.object(bw.time, "sleep") as sleep,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            bw.load_worker(args, workload, "run", 0, mock.Mock(), results)
        # One prefill and one reusable load request, independent of epoch count.
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(ids.call_count, 2)
        self.assertEqual(sleep.call_args_list, [mock.call(0.015)] * 2)
        self.assertEqual(self.parse()[0].epoch_interval_ms, 15)

    def test_prepared_buffers_keep_shape_addresses_and_tensor_lifetime(self):
        _, workload = self.parse("cache-hit", "--layout", "single-layer")
        rows = make_fake_rows(None, workload, "mock", 2)
        ids = [b"a" * 16, b"b" * 16]
        request = bw.prepare_transfer(rows, ids, workload)
        self.assertIs(request.rows, rows)
        packed, indexes, addresses = request.tasks[0]
        self.assertEqual(packed, b"".join(ids))
        self.assertEqual(list(indexes), [0, 0])
        self.assertEqual(memoryview(addresses).shape, (2, 3))
        for row, tensors in zip(addresses, rows):
            self.assertEqual(list(row), [tensor.data_ptr() for tensor in tensors])

    def test_load_timer_only_contains_submit_and_wait(self):
        _, workload = self.small("cache-hit")
        rows = make_fake_rows(None, workload, "mock", 1)
        request = bw.prepare_transfer(rows, [b"a" * 16], workload)
        store, runtime = mock.Mock(), mock.Mock()
        events = []
        runtime.synchronize.side_effect = lambda: events.append("sync")
        store.store_.Load.side_effect = lambda *_: events.append("load")
        store.store_.Wait.side_effect = lambda *_: events.append("wait")

        def clock():
            events.append("clock")
            return len(events) * 0.001

        with (
            mock.patch.object(bw.time, "perf_counter", side_effect=clock),
            mock.patch.object(
                FakeTensor, "data_ptr", side_effect=AssertionError("late pointer")
            ),
        ):
            costs = bw.transfer_request(store, runtime, request)
        self.assertEqual(events, ["sync"] + ["clock", "load", "wait", "clock"] * 3)
        self.assertEqual(len(costs), 3)
        submitted = [call.args for call in store.store_.Load.call_args_list]
        self.assertEqual([list(args[1]) for args in submitted], [[0], [1], [2]])
        self.assertTrue(all(args[2] is submitted[0][2] for args in submitted))

    def test_every_posix_epoch_and_every_layer_is_cold(self):
        args, workload = self.small("posix-hit")
        disk = {}
        seed = MemoryStore(workload, disk)
        barrier = mock.Mock()
        with (
            mock.patch.object(bw, "open_store", return_value=seed),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            bw.seed_worker(args, workload, "run", 0, barrier)
        expected = (args.warmup + args.epochs) * args.blocks * workload.shards
        self.assertEqual(len(disk), expected)
        loader = MemoryStore(workload, disk)
        results = [0.0] * (args.epochs * workload.shards)
        with (
            mock.patch.object(bw, "open_store", return_value=loader),
            mock.patch.object(
                bw, "setup_device", return_value=(None, mock.Mock(), "mock")
            ),
            mock.patch.object(bw, "make_rows", side_effect=make_fake_rows),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            bw.load_worker(args, workload, "run", 0, barrier, results)
        self.assertEqual(set(loader.sources), {"posix"})
        self.assertEqual(len(loader.sources), expected)
        self.assertEqual(len(set(loader.records)), args.warmup + args.epochs)
        self.assertTrue(all(cost > 0 for cost in results))

    def test_verification_detects_corrupt_payload(self):
        _, workload = self.small("cache-hit")
        store = MemoryStore(workload)
        store.cache[b"a" * 16, 0] = bytes(workload.payload_size)
        rows = make_fake_rows(None, workload, "mock", 1)
        request = bw.prepare_transfer(rows, [b"a" * 16], workload)
        with self.assertRaisesRegex(RuntimeError, "data mismatch"):
            bw.transfer_request(store, mock.Mock(), request, verify=True)

    def test_dry_run_cli_does_not_need_accelerator_dependencies(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--mode", "posix-hit", "--dry-run"],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["load_tasks_per_request"], 78)
        self.assertEqual(plan["unique_seed_records"], 12)
        self.assertEqual(plan["seed_storage_bytes"], 180224 * 512 * 78 * 12)

    def test_failed_run_removes_only_its_own_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "existing-data"
            sentinel.write_text("keep", encoding="utf-8")
            with (
                mock.patch.object(
                    bw, "run_group", side_effect=RuntimeError("seed failure")
                ),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "seed failure"),
            ):
                bw.main(["--mode", "posix-hit", "--storage-backends", directory])
            self.assertEqual(list(root.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_keep_data_preserves_only_the_new_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(
                    bw, "run_group", side_effect=RuntimeError("seed failure")
                ),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "seed failure"),
            ):
                bw.main(
                    [
                        "--mode",
                        "posix-hit",
                        "--storage-backends",
                        directory,
                        "--keep-data",
                    ]
                )
            paths = list(Path(directory).iterdir())
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].name.startswith("ucm-cache-bw-"))

    def test_seed_group_finishes_before_load_group_is_created(self):
        args, workload = self.small("posix-hit")
        phases = []

        def run_group(_context, phase, _args, _workload, _run_id, results):
            phases.append(phase)
            if phase == "load":
                results[:] = [0.001] * len(results)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(bw, "parse_args", return_value=(args, workload)),
            mock.patch.object(bw, "run_group", side_effect=run_group),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            args.storage_backends = [directory]
            bw.main([])
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(phases, ["seed", "load"])

    def test_summary_pools_ranks_and_excludes_warmup_slots(self):
        args, workload = self.parse(
            "cache-hit", "--workers", "2", "--epochs", "2", "--layout", "single-layer"
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            bw.print_summary(args, workload, [0.001, 0.003, 0.005, 0.007])
        rows = output.getvalue().splitlines()
        self.assertEqual(rows[3].split()[:3], ["ALL", "4", "4.000"])
        self.assertEqual(rows[4].split()[:3], ["0", "2", "2.000"])
        self.assertEqual(rows[5].split()[:3], ["1", "2", "6.000"])

    def test_spawned_group_joins_before_returning(self):
        args, workload = self.small("cache-hit")
        args.workers = 2
        context = WorkerContext(process_success)
        results = context.context.Array("d", 2, lock=False)
        bw.run_group(context, "load", args, workload, "run", results)
        self.assertEqual(list(results), [1.0, 2.0])

    def test_failed_worker_terminates_and_joins_its_peers(self):
        args, workload = self.small("cache-hit")
        args.workers = 2
        context = WorkerContext(process_failure)
        before = {child.pid for child in multiprocessing.active_children()}
        with self.assertRaisesRegex(RuntimeError, "exited with code 7"):
            bw.run_group(context, "load", args, workload, "run", None)
        self.assertEqual(
            {child.pid for child in multiprocessing.active_children()}, before
        )

    def test_stalled_worker_times_out_and_is_joined(self):
        args, workload = self.small("cache-hit")
        args.process_timeout = 1
        context = WorkerContext(process_stall)
        before = {child.pid for child in multiprocessing.active_children()}
        with self.assertRaises(TimeoutError):
            bw.run_group(context, "load", args, workload, "run", None)
        self.assertEqual(
            {child.pid for child in multiprocessing.active_children()}, before
        )


if __name__ == "__main__":
    unittest.main()
