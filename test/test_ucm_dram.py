import random
import unittest
from typing import List
from unittest.mock import MagicMock

import torch
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.sampling.params import SamplingParams
from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import hash_request_tokens
from vllm_v1_request import Request

from unifiedcache.ucm_connector.ucm_dram import UcmDram, DramTask


def make_request(request_id,
                 prompt_token_ids,
                 mm_positions=None,
                 mm_hashes=None,
                 cache_salt=None):
    if mm_positions is None:
        multi_model_inputs = None
    else:
        multi_model_inputs = [MultiModalKwargs({})] * len(mm_positions)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        multi_model_inputs=multi_model_inputs,
        multi_model_hashes=mm_hashes,
        multi_model_placeholders=mm_positions,
        sampling_params=SamplingParams(max_tokens=17),
        eos_token_id=100,
        arrival_time=0,
        lora_request=None,
        cache_salt=cache_salt,
    )

class TestUcmDram(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("===> Before all tests (setUpClass)")

    @classmethod
    def tearDownClass(cls):
        print("===> After all tests (setUpClass)")

    def setUp(self):
        self.config = {
            "block_size": 4
        }
        self.scheduler_config = {
            "role": "scheduler",
            "max_cache_size": 1073741824,
            "kv_block_size": 262144
        }
        self.worker_config = {
            "role": "worker",
            "max_cache_size": 1073741824,
            "kv_block_size": 262144
        }

        self.block_number = 4
        self.block_size = int(self.config["block_size"])
        self.scheduler_dram = UcmDram(self.scheduler_config)
        self.worker_dram = UcmDram(self.worker_config)
        random.seed(20250728)
        self.request = make_request(
            request_id=1,
            prompt_token_ids=random.sample(range(0, 10000), self.block_number * self.block_size),
            mm_positions=None,
            mm_hashes=None,
        )
        block_hash_types = hash_request_tokens(sha256, self.block_size, self.request)
        self.block_hashes: List[str] = [str(x.hash_value) for x in block_hash_types]

    def test_look_up_all_hit(self):
        '''
        Test for all blocks hitten in cache
        '''
        expected = [True] * len(self.block_hashes)
        self.scheduler_dram.cached_blocks.update(self.block_hashes)
        actual = self.scheduler_dram.lookup(self.block_hashes)

        self.assertEqual(actual, expected)
    
    def test_lookup_partial_hit(self):
        '''
        Test for part of the blocks hitten in cache
        '''
        partial_index = random.randint(0,4)
        partial_hashes = self.block_hashes[:partial_index]
        self.scheduler_dram.cached_blocks.update(partial_hashes)
        actual = self.scheduler_dram.lookup(self.block_hashes)
        expected = [True] * partial_index + [False] * (self.block_size - partial_index)
        self.assertEqual(actual, expected)

    def test_lookup_none_hit(self):
        '''
        Test for none of the blocks hitten in cache
        '''
        actual = self.scheduler_dram.lookup(self.block_hashes)
        expected = [False] * len(self.block_hashes)
        self.assertEqual(actual, expected)

    def test_load_success(self):
        '''
        Test for load from cache successfully
        '''
        src_tensors = [torch.randint(0,100, (self.block_size,),dtype = torch.int8)
                       for _ in range(len(self.block_hashes))]
        offsets = [i for i in range(len(self.block_hashes))]
        dump_task = self.worker_dram.dump(self.block_hashes, offsets, src_tensors)
        self.worker_dram.wait(dump_task)
        dst_tensors = [torch.zeros(self.block_size, dtype = torch.int8)
                       for _ in range(len(self.block_hashes))]
        load_task = self.worker_dram.load(self.block_hashes, offsets, dst_tensors)

        self.assertIsInstance(load_task, DramTask)
        self.assertIsNone(load_task.event)
        for i, (src_tensor, dst_tensor) in  enumerate(zip(src_tensors, dst_tensors)):
            self.assertEqual(dst_tensor.shape[0], self.block_size)
            self.assertTrue(torch.equal(src_tensor, dst_tensor),
                            f"Block {i} loaded data is different")
    
    def test_dump_success(self):
        '''
        Test data dump successfully
        '''
        src_tensors = [torch.randint(0,100, (self.block_size,),dtype = torch.int8)
                       for _ in range(len(self.block_hashes))]
        offsets = [i for i in range(len(self.block_hashes))]
        original_data = [tensor.clone() for tensor in src_tensors]
        dump_task = self.worker_dram.dump(self.block_hashes, offsets, src_tensors)
        self.assertIsInstance(dump_task, DramTask)
        self.assertIsNone(dump_task.event)
        self.worker_dram.wait(dump_task)
        for i, block_id in enumerate(self.block_hashes):
            key = block_id + '_' + str(offsets[i])
            cached_data = self.worker_dram.dram_cache[key]
            self.assertEqual(cached_data.shape[0], self.block_size)
            self.assertTrue(torch.equal(cached_data, original_data[i]))
    
    def test_wait_success(self):
        '''
        Test wait for task successfully
        '''
        task = DramTask()
        task.event = MagicMock()
        result = self.worker_dram.wait(task)
        self.assertEqual(result, 0)
        task.event.synchronize.assert_called_once()
    
    def test_wait_failure(self):
        task = DramTask()
        task.event = None
        result = self.worker_dram.wait(task)
        self.assertEqual(result, -1)

if __name__ == '__main__':
    unittest.main()