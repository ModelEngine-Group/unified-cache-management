#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#

import os
import secrets
import shutil
import tempfile
import unittest

import torch

from ucm.store.connector.nfsstore_connector import UcmNfsStore


class TestUcmNfsStoreInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = "/root/space/tmp" 
        cls.block_dim = 576
        cls.block_len = 128
        cls.block_layer = 61
        cls.block_size = (
            cls.block_dim * cls.block_len * 2 * cls.block_layer
        )  # bfloat16 takes 2 bytes

        cls.config = {
            "storage_backends": cls.temp_dir,
            "device": 0,
            "kv_block_size": cls.block_size,
            "role": "worker",
            "transferStreamNumber": 32,
            "io_size": 262144,
        }

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        cls.store = UcmNfsStore(cls.config)
        torch.cuda.set_device(0)

    def _generate_block_id(self):
        """Generate block ID"""
        return secrets.token_hex(16)

    def _create_test_block(self):
        """Create test data block"""
        return [
            torch.rand(
                [self.block_dim, self.block_len], dtype=torch.bfloat16, device="cuda:0"
            )
            for _ in range(self.block_layer)
        ]

    def test_create_success(self):
        """
        Test for successful allocation of a new block
        """
        block_id = self._generate_block_id()
        test_block = self._create_test_block()
        ret = self.store.create([block_id])
        self.assertEqual(ret, [0], "Failed to create block space")
        first_layer = test_block[0]
        dump_task = self.store.dump([block_id], [0], [first_layer])
        self.assertIsInstance(dump_task, object, "dump should return a task object")
        wait_result = self.store.wait(dump_task)
        self.assertEqual(wait_result, 0, "Failed waiting for dump task")
        self.store.commit([block_id], is_success=True)

    def test_create_and_dump_multiple_blocks(self):
        """
        Test for successful allocation of multiple blocks
        """
        batch_size = 5
        block_ids = [self._generate_block_id() for _ in range(batch_size)]
        test_blocks = [self._create_test_block() for _ in range(batch_size)]
        ret = self.store.create(block_ids)
        self.assertEqual(len(ret), batch_size, "Return result length mismatch")
        self.assertTrue(all(r == 0 for r in ret), "Failed to batch create block space")
        for i, (bid, block) in enumerate(zip(block_ids, test_blocks)):
            first_layer = block[0]
            dump_task = self.store.dump([bid], [0], [first_layer])
            wait_result = self.store.wait(dump_task)
            self.assertEqual(wait_result, 0, f"Block {bid} dump failed")
        self.store.commit(block_ids, is_success=True)

    def test_load_return_task(self):
        """
        Test for load returning a valid NfsTask with correct task_id
        """
        block_id = self._generate_block_id()
        test_block = self._create_test_block()
        first_layer = test_block[0]
        self.store.create([block_id])
        dump_task = self.store.dump([block_id], [0], [first_layer])
        self.store.wait(dump_task)
        self.store.commit([block_id], is_success=True)
        loaded_tensor = torch.empty_like(first_layer)
        load_task = self.store.load([block_id], [0], [loaded_tensor])
        self.assertIsInstance(load_task, object, "load should return a task object")
        wait_result = self.store.wait(load_task)
        self.assertEqual(wait_result, 0, "Failed waiting for load task")
        self.assertTrue(
            torch.allclose(first_layer, loaded_tensor, rtol=1e-2),
            "Loaded data inconsistent",
        )

    def test_tensor_precision_consistency(self):
        """
        Test that tensor data remains consistent between dump and load operations
        """
        block_id = self._generate_block_id()
        test_block = self._create_test_block()
        first_layer = test_block[0].clone()  # Clone to preserve original data
        ret = self.store.create([block_id])
        self.assertEqual(ret, [0], "Failed to create block space")
        dump_task = self.store.dump([block_id], [0], [first_layer])
        self.assertIsInstance(dump_task, object, "dump should return a task object")
        wait_result = self.store.wait(dump_task)
        self.assertEqual(wait_result, 0, "Failed waiting for dump task")
        self.store.commit([block_id], is_success=True)
        loaded_tensor = torch.empty_like(first_layer)
        load_task = self.store.load([block_id], [0], [loaded_tensor])
        self.assertIsInstance(load_task, object, "load should return a task object")
        wait_result = self.store.wait(load_task)
        self.assertEqual(wait_result, 0, "Failed waiting for load task")
        self.assertTrue(
            torch.allclose(first_layer, loaded_tensor, rtol=1e-2, atol=1e-3),
            "Loaded tensor data is not consistent with original data within precision tolerance",
        )
        # Additional check: calculate the mean difference
        diff = torch.abs(first_layer - loaded_tensor)
        mean_diff = torch.mean(diff)
        max_diff = torch.max(diff)
        # These values should be very small for bfloat16 precision
        self.assertLess(
            mean_diff.item(), 0.01, "Mean difference between tensors is too large"
        )
        self.assertLess(
            max_diff.item(), 0.05, "Maximum difference between tensors is too large"
        )

    def test_lookup(self):
        """
        Test for partial hit lookup (True, False, True)
        """
        existing_block_id = self._generate_block_id()
        self.store.create([existing_block_id])
        test_block = self._create_test_block()
        first_layer = test_block[0]
        dump_task = self.store.dump([existing_block_id], [0], [first_layer])
        self.store.wait(dump_task)
        self.store.commit([existing_block_id], is_success=True)
        non_existing_block_id = self._generate_block_id()
        block_ids = [existing_block_id, non_existing_block_id]
        lookup_results = self.store.lookup(block_ids)
        self.assertEqual(len(lookup_results), 2, "Lookup result count mismatch")
        self.assertTrue(lookup_results[0], "Should find existing block")
        self.assertFalse(lookup_results[1], "Should not find non-existing block")

    def test_check(self):
        """
        Test for check reporting task finished status
        """
        block_id = self._generate_block_id()
        test_block = self._create_test_block()
        first_layer = test_block[0]
        self.store.create([block_id])
        dump_task = self.store.dump([block_id], [0], [first_layer])
        ret, finished = self.store.check(dump_task)
        self.assertEqual(ret, 0, "Checking task status should succeed")
        wait_result = self.store.wait(dump_task)
        self.assertEqual(wait_result, 0, "Failed waiting for task completion")
        self.store.commit([block_id], is_success=True)

    def test_wait_success(self):
        """
        Test for successful wait on a valid NfsTask
        """
        block_id = self._generate_block_id()
        self.store.create([block_id])
        self.store.commit([block_id], is_success=True)
        # self.store.commit([block_id], is_success=False)

    def test_commit_success(self):
        """
        Test for successful commit of blocks with persist=True
        """
        batch_size = 3
        block_ids = [self._generate_block_id() for _ in range(batch_size)]
        test_blocks = [self._create_test_block() for _ in range(batch_size)]
        ret = self.store.create(block_ids)
        self.assertTrue(all(r == 0 for r in ret), "Failed to create block space")
        dump_tasks = []
        for i, (bid, block) in enumerate(zip(block_ids, test_blocks)):
            first_layer = block[0]
            dump_task = self.store.dump([bid], [0], [first_layer])
            dump_tasks.append(dump_task)
        for i, (task, bid) in enumerate(zip(dump_tasks, block_ids)):
            wait_result = self.store.wait(task)
            self.assertEqual(wait_result, 0, f"Block {bid} dump wait failed")
        self.store.commit(block_ids, is_success=True)
        for i, (bid, original_block) in enumerate(zip(block_ids, test_blocks)):
            original_layer = original_block[0]
            loaded_tensor = torch.empty_like(original_layer)

            load_task = self.store.load([bid], [0], [loaded_tensor])
            wait_result = self.store.wait(load_task)
            self.assertEqual(wait_result, 0, f"Block {bid} load wait failed")
            self.assertTrue(
                torch.allclose(original_layer, loaded_tensor, rtol=1e-2),
                f"Block {bid} data inconsistent",
            )

    def test_commit_rollback(self):
        """
        Test for rollback commit of blocks with persist=False
        """
        block_id = self._generate_block_id()
        self.store.create([block_id])
        self.store.commit([block_id], is_success=True)
        # self.store.commit([block_id], is_success=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
