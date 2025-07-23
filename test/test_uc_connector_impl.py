import random
import secrets
import unittest
from typing import List
from unittest.mock import Mock, patch, MagicMock

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata
)
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

from unifiedcache.integration.vllm.uc_connector_impl import UCConnectorImpl, LoadPara, SavePara, \
    UCConnectorV1Metadata, ReqMeta
from unifiedcache.ucm_connector.base import Task, UcmKVStoreBase

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
        multi_modal_inputs=multi_model_inputs,
        multi_modal_hashes=mm_hashes,
        multi_modal_placeholders=mm_positions,
        sampling_params=SamplingParams(max_tokens=17),
        eos_token_id=100,
        arrival_time=0,
        lora_request=None,
        cache_salt=cache_salt,
    )


class TestUCConnectorImpl(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("===> Before all tests (setUpClass)")

    @classmethod
    def tearDownClass(cls):
        print("===> Before all tests (tearDownClass)")

    def setUp(self):
        self.block_number = 4
        self.block_size = 8
        self.num_layers = 48
        self.total_blocks_num = 40
        self.kv_caches = {}
        for i in range(self.num_layers):
            layer_name = f"model.layers.{i}.self_attn.attn"
            kv_tensor = torch.rand((2, self.total_blocks_num, self.block_size, 4, 8), dtype=torch.bfloat16)
            self.kv_caches[layer_name] = kv_tensor

    def init_impl(self, mock_connector, use_layerwise=True) -> UCConnectorImpl:
        with patch.object(UCConnectorImpl, '__init__', return_value=None):
            ucconnectorImpl = UCConnectorImpl(None, None)
            ucconnectorImpl.block_size = self.block_size
            ucconnectorImpl.use_layerwise = use_layerwise
            ucconnectorImpl.kv_caches = self.kv_caches
            ucconnectorImpl.rank = 1
            ucconnectorImpl.is_mla = False
            ucconnectorImpl.connector = mock_connector
            ucconnectorImpl.load_paras: dict[str, LoadPara] = {}
            ucconnectorImpl.save_paras: dict[str, SavePara] = {}
            ucconnectorImpl.dump_tasks: dict[str, List[Task]] = {}
            ucconnectorImpl.load_tasks: dict[str, tuple[Task, Task]] = {}
            ucconnectorImpl.total_tp_size = 2
        return ucconnectorImpl

    def test_get_num_new_matched_tokens_hit(self):
        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_lookup(tokens: List[int]) -> List[bool]:
            return [True] * self.block_number

        mock_connector.lookup.side_effect = mock_lookup
        ucconnectorImpl = self.init_impl(mock_connector)

        random.seed(20250704)
        request1 = make_request(
            request_id=1,
            prompt_token_ids=random.sample(range(0, 10000), self.block_number * self.block_size),
            mm_positions=None,
            mm_hashes=None,
        )

        # all block dumped in ssd, external_tokens equals to full tokens num
        all_tokens_len = len(request1.all_token_ids)
        external_tokens, _ = ucconnectorImpl.get_num_new_matched_tokens(request1, 0)
        self.assertEqual(external_tokens, all_tokens_len)

    def test_get_num_new_matched_tokens_no_hit(self):
        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_lookup(tokens: List[int]) -> List[bool]:
            return [False] * self.block_number

        mock_connector.lookup.side_effect = mock_lookup
        ucconnectorImpl = self.init_impl(mock_connector)

        random.seed(20250704)
        request1 = make_request(
            request_id=1,
            prompt_token_ids=random.sample(range(0, 10000), self.block_number * self.block_size),
            mm_positions=None,
            mm_hashes=None,
        )

        # no block dumped in ssd, external_tokens equals to 0
        external_tokens, _ = ucconnectorImpl.get_num_new_matched_tokens(request1, 0)
        self.assertEqual(external_tokens, 0)

    def test_get_num_new_matched_tokens_invalid_para(self):
        with patch.object(UCConnectorImpl, '__init__', return_value=None):
            ucconnectorImpl = UCConnectorImpl(None, None)
            ucconnectorImpl.block_size = self.block_size

        request1 = make_request(
            request_id=1,
            prompt_token_ids=random.sample(range(0, 10000), self.block_number * self.block_size),
            mm_positions=None,
            mm_hashes=None,
        )

        # passing invalid params
        with self.assertRaises(AssertionError):
            external_tokens, _ = ucconnectorImpl.get_num_new_matched_tokens(request1, self.block_size + 1)

    def test_wait_for_save_not_layerwise_success(self):
        req_meta1 = MagicMock(spec=ReqMeta)
        req_meta1.request_id = "req1"
        req_meta1.save_paras = SavePara(num_blocks_need_save=self.block_number, start_save_position=0,
                                        num_blocks_to_save=self.block_number)
        req_meta1.save_paras.block_hashes = [secrets.token_hex(8) for _ in range(self.block_number)]
        req_meta1.vllm_block_ids = list(range(self.block_number))

        metadata = UCConnectorV1Metadata()
        metadata.requests = [req_meta1]

        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_dump(block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]) -> Task:
            assert len(offset) == len(src_tensor) == len(block_ids)
            return Task()

        def mock_wait(task: Task) -> int:
            return 0

        mock_connector.dump.side_effect = mock_dump
        mock_connector.wait.side_effect = mock_wait
        ucconnectorImpl = self.init_impl(mock_connector, use_layerwise=False)
        ucconnectorImpl.wait_for_save(metadata)

    def test_wait_for_save_not_layerwise_invalid_para(self):
        metadata = Mock()
        with patch.object(UCConnectorImpl, '__init__', return_value=None):
            ucconnectorImpl = UCConnectorImpl(None, None)
            ucconnectorImpl.block_size = self.block_size
            ucconnectorImpl.use_layerwise = False

        with self.assertRaises(AssertionError):
            ucconnectorImpl.wait_for_save(metadata)

    def test_start_load_kv_not_layerwise_success(self):
        req_meta1 = MagicMock(spec=ReqMeta)
        req_meta1.request_id = "req1"
        req_meta1.load_paras = LoadPara(vllm_cached_tokens=1 * self.block_size,
                                        storage_cached_tokens=self.block_number * self.block_size, can_load=True)
        req_meta1.load_paras.block_hashes = [secrets.token_hex(8) for _ in range(self.block_number)]
        req_meta1.vllm_block_ids = list(range(self.block_number))

        metadata = UCConnectorV1Metadata()
        metadata.requests = [req_meta1]

        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_load(block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]) -> Task:
            assert len(offset) == len(dst_tensor) == len(block_ids)
            return Task()

        def mock_wait(task: Task) -> int:
            return 0

        mock_connector.load.side_effect = mock_load
        mock_connector.wait.side_effect = mock_wait

        ucconnectorImpl = self.init_impl(mock_connector, use_layerwise=False)
        forward_context = Mock()
        ucconnectorImpl.start_load_kv(forward_context, metadata=metadata)

    def test_start_load_kv_invalid_para(self):
        with patch.object(UCConnectorImpl, '__init__', return_value=None):
            ucconnectorImpl = UCConnectorImpl(None, None)
            ucconnectorImpl.block_size = self.block_size

        forward_context = Mock()
        with self.assertRaises(AssertionError):
            ucconnectorImpl.start_load_kv(forward_context)

    def test_start_load_kv_layerwise_success(self):
        req_meta1 = MagicMock(spec=ReqMeta)
        req_meta1.request_id = "req1"
        req_meta1.load_paras = LoadPara(vllm_cached_tokens=1 * self.block_size,
                                        storage_cached_tokens=self.block_number * self.block_size, can_load=True)
        req_meta1.load_paras.block_hashes = [secrets.token_hex(8) for _ in range(self.block_number)]
        req_meta1.vllm_block_ids = list(range(self.block_number))

        metadata = UCConnectorV1Metadata()
        metadata.requests = [req_meta1]

        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_load(block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]) -> Task:
            assert len(offset) == len(dst_tensor) == len(block_ids)
            return Task()

        mock_connector.load.side_effect = mock_load
        ucconnectorImpl = self.init_impl(mock_connector)
        forward_context = Mock()
        ucconnectorImpl.start_load_kv(forward_context, metadata=metadata)
        assert mock_connector.load.call_count == 2

    def test_generate_layerwise_load_tasks_success(self):
        # 初始化implement
        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_load(block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]) -> Task:
            assert offset is not None and offset
            assert dst_tensor is not None and dst_tensor
            return Task()

        mock_connector.load.side_effect = mock_load
        ucconnectorImpl = self.init_impl(mock_connector)

        # 构造generate_layerwise_load_tasks所需的参数
        fetch_block_ids = list(range(self.block_number * 2))
        fetch_block_hashes = [secrets.token_hex(8) for _ in range(self.block_number * 2)]
        layer_to_tensor: dict[str, tuple[List[torch.Tensor], List[int]]] = {}
        current_layer = 0
        for layer_name, kv_layer in self.kv_caches.items():
            tensors, offsets = ucconnectorImpl.get_tensor_and_offset_layerwise(fetch_block_ids, kv_layer, layer_name)
            layer_to_tensor[layer_name] = (tensors, offsets)
            current_layer += 1
        # 调用函数后应生成num_layers层任务
        layerwise_load_task = ucconnectorImpl.generate_layerwise_load_tasks(fetch_block_hashes, layer_to_tensor)

        # 按层调用
        for i in range(self.num_layers):
            task = next(layerwise_load_task)
            assert task is not None, f"layer {i} is None"
        assert mock_connector.load.call_count == self.num_layers * 2

    def test_generate_layerwise_load_tasks_invalid_params(self):
        # 初始化implement
        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_load(block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]) -> Task:
            assert offset is not None and offset
            assert dst_tensor is not None and dst_tensor
            return Task()

        mock_connector.load.side_effect = mock_load
        ucconnectorImpl = self.init_impl(mock_connector)

        # 构造generate_layerwise_load_tasks所需的参数
        fetch_block_ids = list(range(self.block_number * 2))
        fetch_block_hashes = [secrets.token_hex(8) for _ in range(self.block_number * 2)]
        layer_to_tensor: dict[str, tuple[List[torch.Tensor], List[int]]] = {}
        for layer_name, kv_layer in self.kv_caches.items():
            tensors, offsets = ucconnectorImpl.get_tensor_and_offset_layerwise(fetch_block_ids, kv_layer, layer_name)
            layer_to_tensor[layer_name] = (tensors, offsets)
        # 调用函数后应生成num_layers层任务
        layerwise_load_task = ucconnectorImpl.generate_layerwise_load_tasks([], layer_to_tensor)
        with self.assertRaises(AssertionError) as context:
            next(layerwise_load_task)
        self.assertEqual(str(context.exception), "The block hashes need to be fetched should not be None or empty.")

        layerwise_load_task = ucconnectorImpl.generate_layerwise_load_tasks(fetch_block_hashes, None)
        with self.assertRaises(AssertionError) as context:
            next(layerwise_load_task)
        self.assertEqual(str(context.exception), "The layers of tensor need to be fetched should not be None or empty.")


if __name__ == '__main__':
    unittest.main()
