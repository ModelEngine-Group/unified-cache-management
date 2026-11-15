# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Runnable pynds_kv mock integration example; no NDS hardware is required."""

import importlib
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pynds_kv = importlib.import_module("pynds_kv_mock")


class MockIntegrationTest(unittest.TestCase):
    def setUp(self):
        pynds_kv.reset()

    def make_blocks(self, storage, count, block_size, first=0):
        view = memoryview(storage)
        self.addCleanup(view.release)
        blocks = []
        for index in range(first, first + count):
            block = view[index * block_size : (index + 1) * block_size]
            self.addCleanup(block.release)
            blocks.append(block)
        return blocks

    def test_init_multi_single_io_and_prefix_hits(self):
        storage = bytearray(128)
        self.assertEqual(pynds_kv.init_multi([storage], [128]), 0)
        block = self.make_blocks(storage, 1, 64)[0]
        block[:32] = bytes([7]) * 32

        self.assertEqual(pynds_kv.put(101, block, 0, 32, 5), 0)
        self.assertEqual(pynds_kv.put(103, block, 32, 32, 5), 0)
        self.assertEqual(pynds_kv.is_exists([101, 102, 103], 5), 1)
        self.assertEqual(pynds_kv.is_exists([101, 103], 5), 2)
        self.assertEqual(pynds_kv.is_exists([101, 103], 6), 0)

        block[:32] = bytes(32)
        self.assertEqual(pynds_kv.get(101, block, 0, 32, 5), 0)
        self.assertEqual(bytes(block[:32]), bytes([7]) * 32)

    def test_init_exists_and_batch_round_trip(self):
        block_size = 64
        block_ids = [1001, 1002, 1003]
        nsid = 9
        storage = bytearray(block_size * len(block_ids) * 2)
        self.assertEqual(pynds_kv.init(storage, len(storage)), 0)
        write_blocks = self.make_blocks(storage, len(block_ids), block_size)
        read_blocks = self.make_blocks(
            storage, len(block_ids), block_size, first=len(block_ids)
        )
        payloads = [bytes([index + 1]) * block_size for index in range(3)]
        for block, payload in zip(write_blocks, payloads):
            block[:] = payload

        offsets = [0] * len(block_ids)
        lengths = [block_size] * len(block_ids)
        nsids = [nsid] * len(block_ids)
        self.assertEqual(pynds_kv.is_exists(block_ids, nsid), 0)
        self.assertEqual(
            pynds_kv.batchPut(block_ids, write_blocks, offsets, lengths, nsids), 0
        )
        self.assertEqual(pynds_kv.is_exists(block_ids + [1004], nsid), 3)
        self.assertEqual(pynds_kv.is_exists(block_ids, nsid + 1), 0)
        self.assertEqual(
            pynds_kv.batchGet(block_ids, read_blocks, offsets, lengths, nsids), 0
        )
        for block, payload in zip(read_blocks, payloads):
            self.assertEqual(bytes(block), payload)

    def test_concurrent_batch_dispatch(self):
        worker_count = 4
        block_size = 32
        nsid = 11
        storage = bytearray(worker_count * block_size)
        self.assertEqual(pynds_kv.init(storage, len(storage)), 0)
        blocks = self.make_blocks(storage, worker_count, block_size)
        barrier = threading.Barrier(worker_count)

        def round_trip(worker):
            block_id = 2000 + worker
            payload = bytes([worker + 1]) * block_size
            block = blocks[worker]
            barrier.wait(timeout=5)
            block[:] = payload
            if pynds_kv.batchPut([block_id], [block], [0], [block_size], [nsid]) != 0:
                return False
            block[:] = bytes(block_size)
            if pynds_kv.batchGet([block_id], [block], [0], [block_size], [nsid]) != 0:
                return False
            return bytes(block) == payload

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(round_trip, range(worker_count)))

        self.assertTrue(all(results))
        io_calls = [
            call
            for call in pynds_kv.get_calls()
            if call.operation in {"batchPut", "batchGet"}
        ]
        self.assertEqual(len(io_calls), worker_count * 2)
        self.assertEqual(len({call.thread_id for call in io_calls}), worker_count)
        self.assertEqual(
            pynds_kv.is_exists([2000, 2001, 2002, 2003], nsid), worker_count
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
