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

import random
import secrets
import unittest
from typing import List, Union
from unittest.mock import MagicMock, Mock, patch

import torch
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request

from ucm.integration.vllm.uc_connector import (
    BlockOperation,
    ReqMeta,
    RequestBlockInfo,
    UCConnectorV1Metadata,
    UnifiedCacheConnectorV1,
)
from ucm.store.ucmstore import Task, UcmKVStoreBase


def make_request(
    request_id, prompt_token_ids, mm_positions=None, mm_hashes=None, cache_salt=None
):
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
        pooling_params=None,
        eos_token_id=100,
        arrival_time=0,
        lora_request=None,
        cache_salt=cache_salt,
    )


class TestUCConnector(unittest.TestCase):

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
        self.total_tp_size = 2
        self.kv_caches = {}
        for i in range(self.num_layers):
            layer_name = f"model.layers.{i}.self_attn.attn"
            kv_tensor = torch.rand(
                (2, self.total_blocks_num, self.block_size, 4, 8), dtype=torch.bfloat16
            )
            self.kv_caches[layer_name] = kv_tensor

    def init_uc(
        self, mock_connector, metadata=Mock(), use_layerwise=True
    ) -> UnifiedCacheConnectorV1:
        with patch.object(UnifiedCacheConnectorV1, "__init__", return_value=None):
            ucconnector = UnifiedCacheConnectorV1(None, None)
            ucconnector.block_size = self.block_size
            ucconnector.use_layerwise = use_layerwise
            ucconnector.kv_caches = self.kv_caches
            ucconnector.rank = 1
            ucconnector.is_mla = False
            ucconnector.connector = mock_connector
            ucconnector.request_block_infos: dict[str, RequestBlockInfo] = {}
            ucconnector.dump_tasks: dict[str, dict[str, List[Task]]] = {}
            ucconnector.total_tp_size = self.total_tp_size
            ucconnector._connector_metadata = metadata
            ucconnector.layerwise_load_tasks: dict[
                str, dict[str, tuple[Task, Task]]
            ] = {}
            ucconnector._need_load_reqs: dict[str, Union[list[int], list[Task]]] = {}
            ucconnector._load_failed_reqs: set[str] = set()
            ucconnector._load_req_to_blocks: dict[str, set[int]] = {}
        return ucconnector

    def test_get_num_new_matched_tokens_hit_all_on_storage(self):
        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_lookup(block_hashes: List[str]) -> List[bool]:
            return [True] * self.block_number

        mock_connector.lookup.side_effect = mock_lookup
        ucconnector = self.init_uc(mock_connector)

        random.seed(20250704)
        request1 = make_request(
            request_id=1,
            prompt_token_ids=random.sample(
                range(0, 10000), self.block_number * self.block_size
            ),
            mm_positions=None,
            mm_hashes=None,
        )

        # all block dumped in ssd, external_tokens equals to full tokens num - self.block_size
        all_tokens_len = len(request1.all_token_ids)
        external_tokens, _ = ucconnector.get_num_new_matched_tokens(request1, 0)
        self.assertEqual(external_tokens, all_tokens_len - self.block_size)
        self.assertEqual(
            ucconnector.request_block_infos[request1.request_id].block_operations,
            [
                BlockOperation.LOAD,
                BlockOperation.LOAD,
                BlockOperation.LOAD,
                BlockOperation.NONE,
            ],
        )

    def test_get_num_new_matched_tokens_partial_hit(self):
        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_lookup(block_hashes: List[str]) -> List[bool]:
            return [True, False, True, False]

        mock_connector.lookup.side_effect = mock_lookup
        ucconnector = self.init_uc(mock_connector)

        random.seed(20250704)
        request1 = make_request(
            request_id=1,
            prompt_token_ids=random.sample(
                range(0, 10000), self.block_number * self.block_size
            ),
            mm_positions=None,
            mm_hashes=None,
        )

        # all block dumped in ssd, external_tokens equals to full tokens num - self.block_size
        all_tokens_len = len(request1.all_token_ids)
        external_tokens, _ = ucconnector.get_num_new_matched_tokens(request1, 0)
        self.assertEqual(external_tokens, self.block_size)
        self.assertEqual(
            ucconnector.request_block_infos[request1.request_id].block_operations,
            [
                BlockOperation.LOAD,
                BlockOperation.NONE,
                BlockOperation.NONE,
                BlockOperation.NONE,
            ],
        )

    def test_get_num_new_matched_tokens_partial_hit_with_preftxcache(self):
        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_lookup(block_hashes: List[str]) -> List[bool]:
            return [True, True, False]

        mock_connector.lookup.side_effect = mock_lookup
        ucconnector = self.init_uc(mock_connector)

        random.seed(20250704)
        request1 = make_request(
            request_id=1,
            prompt_token_ids=random.sample(
                range(0, 10000), self.block_number * self.block_size
            ),
            mm_positions=None,
            mm_hashes=None,
        )

        # 1 block hit on hbm cache, 2 block hit on ssd, 1 blocks miss
        external_tokens, _ = ucconnector.get_num_new_matched_tokens(
            request1, self.block_size
        )
        self.assertEqual(external_tokens, 2 * self.block_size)
        self.assertEqual(
            ucconnector.request_block_infos[request1.request_id].start_position, 1
        )
        self.assertEqual(
            ucconnector.request_block_infos[request1.request_id].block_operations,
            [
                BlockOperation.NONE,
                BlockOperation.LOAD,
                BlockOperation.LOAD,
                BlockOperation.NONE,
            ],
        )

    def test_get_num_new_matched_tokens_partial_hit_with_load_async(self):
        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_lookup(block_hashes: List[str]) -> List[bool]:
            return [True, True, False]

        mock_connector.lookup.side_effect = mock_lookup
        ucconnector = self.init_uc(mock_connector)
        ucconnector.kv_role = "kv_consumer"

        random.seed(20250704)
        request1 = make_request(
            request_id=1,
            prompt_token_ids=random.sample(
                range(0, 10000), self.block_number * self.block_size
            ),
            mm_positions=None,
            mm_hashes=None,
        )
        if request1.kv_transfer_params is None:
            request1.kv_transfer_params = {}
        request1.kv_transfer_params["load_async"] = True

        # 1 block hit on hbm cache, 2 block hit on ssd, 1 blocks miss
        external_tokens, load_async = ucconnector.get_num_new_matched_tokens(
            request1, self.block_size
        )
        self.assertEqual(external_tokens, 2 * self.block_size)
        self.assertEqual(
            ucconnector.request_block_infos[request1.request_id].start_position, 1
        )
        self.assertEqual(request1.kv_transfer_params["load_async"], False)
        self.assertEqual(
            ucconnector.request_block_infos[request1.request_id].block_operations,
            [
                BlockOperation.NONE,
                BlockOperation.LOAD,
                BlockOperation.LOAD,
                BlockOperation.NONE,
            ],
        )

    def test_update_state_after_alloc_create_success(self):
        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_create(block_hashes: List[str]) -> List[int]:
            return [0]

        mock_connector.create.side_effect = mock_create
        ucconnector = self.init_uc(mock_connector)
        request1 = make_request(
            request_id=1,
            prompt_token_ids=random.sample(
                range(0, 10000), self.block_number * self.block_size
            ),
            mm_positions=None,
            mm_hashes=None,
        )
        ucconnector.request_block_infos[request1.request_id] = RequestBlockInfo(
            block_hashes=[secrets.token_hex(8) for _ in range(self.block_number)],
            block_operations=[
                BlockOperation.NONE,
                BlockOperation.LOAD,
                BlockOperation.LOAD,
                BlockOperation.NONE,
            ],
            start_position=1,
        )

        vllm_blocks = Mock()
        vllm_blocks.get_unhashed_block_ids.return_value = [0, 1, 2, 3]
        ucconnector.update_state_after_alloc(request1, vllm_blocks, 2 * self.block_size)
        self.assertEqual(
            ucconnector.request_block_infos[request1.request_id].block_operations,
            [
                BlockOperation.NONE,
                BlockOperation.LOAD,
                BlockOperation.LOAD,
                BlockOperation.DUMP,
            ],
        )
        self.assertEqual(
            ucconnector.request_block_infos[request1.request_id].start_position, 0
        )

    def test_update_state_after_alloc_with_load_async(self):
        mock_connector = Mock(spec=UcmKVStoreBase)

        ucconnector = self.init_uc(mock_connector)
        request1 = make_request(
            request_id=1,
            prompt_token_ids=random.sample(
                range(0, 10000), self.block_number * self.block_size
            ),
            mm_positions=None,
            mm_hashes=None,
        )
        ucconnector.request_block_infos[request1.request_id] = RequestBlockInfo(
            block_hashes=[secrets.token_hex(8) for _ in range(self.block_number)],
            block_operations=[
                BlockOperation.NONE,
                BlockOperation.LOAD,
                BlockOperation.LOAD,
                BlockOperation.NONE,
            ],
            start_position=1,
        )
        ucconnector._need_load_reqs[request1.request_id] = []
        vllm_blocks = Mock()
        vllm_blocks.get_unhashed_block_ids.return_value = [1, 2, 3]
        ucconnector.update_state_after_alloc(request1, vllm_blocks, 2 * self.block_size)
        self.assertEqual(
            ucconnector.request_block_infos[request1.request_id].block_operations,
            [
                BlockOperation.NONE,
                BlockOperation.LOAD,
                BlockOperation.LOAD,
                BlockOperation.NONE,
            ],
        )
        self.assertEqual(
            ucconnector.request_block_infos[request1.request_id].start_position, 1
        )

    def test_build_connector_meta(self):
        mock_connector = Mock(spec=UcmKVStoreBase)

        ucconnector = self.init_uc(mock_connector)
        request1 = make_request(
            request_id=1,
            prompt_token_ids=random.sample(
                range(0, 10000), self.block_number * self.block_size
            ),
            mm_positions=None,
            mm_hashes=None,
        )
        request2 = make_request(
            request_id=2,
            prompt_token_ids=random.sample(
                range(0, 10000), self.block_number * self.block_size
            ),
            mm_positions=None,
            mm_hashes=None,
        )
        ucconnector.request_block_infos[request1.request_id] = RequestBlockInfo(
            block_hashes=[secrets.token_hex(8) for _ in range(self.block_number)],
            block_operations=[
                BlockOperation.NONE,
                BlockOperation.LOAD,
                BlockOperation.LOAD,
                BlockOperation.DUMP,
            ],
            start_position=0,
        )
        ucconnector.request_block_infos[request2.request_id] = RequestBlockInfo(
            block_hashes=[secrets.token_hex(8) for _ in range(self.block_number)],
            block_operations=[
                BlockOperation.NONE,
                BlockOperation.LOAD,
                BlockOperation.LOAD,
                BlockOperation.DUMP,
            ],
            start_position=2,
        )

        from types import SimpleNamespace

        scheduled_new_reqs = [
            SimpleNamespace(req_id=request1.request_id, block_ids=[[1, 2, 3, 4]]),
        ]

        scheduled_cached_reqs = SimpleNamespace(
            req_ids=[request2.request_id],
            new_block_ids=[[[5, 6]]],
        )

        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=scheduled_cached_reqs,
        )

        meta = ucconnector.build_connector_meta(scheduler_output)
        self.assertIsInstance(meta, UCConnectorV1Metadata)
        new_req_meta = meta.requests[0]
        self.assertEqual(
            new_req_meta.load_blocks,
            [
                (
                    ucconnector.request_block_infos[request1.request_id].block_hashes[
                        1
                    ],
                    2,
                ),
                (
                    ucconnector.request_block_infos[request1.request_id].block_hashes[
                        2
                    ],
                    3,
                ),
            ],
        )
        self.assertEqual(
            new_req_meta.dump_blocks,
            [(ucconnector.request_block_infos[request1.request_id].block_hashes[3], 4)],
        )
        cache_req_meta = meta.requests[1]
        self.assertEqual(
            cache_req_meta.load_blocks,
            [(ucconnector.request_block_infos[request2.request_id].block_hashes[2], 5)],
        )
        self.assertEqual(
            cache_req_meta.dump_blocks,
            [(ucconnector.request_block_infos[request2.request_id].block_hashes[3], 6)],
        )

    def test_get_num_new_matched_tokens_no_hit(self):
        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_lookup(blocks: List[str]) -> List[bool]:
            return [False] * self.block_number

        mock_connector.lookup.side_effect = mock_lookup
        ucconnector = self.init_uc(mock_connector)

        random.seed(20250704)
        request1 = make_request(
            request_id=1,
            prompt_token_ids=random.sample(
                range(0, 10000), self.block_number * self.block_size
            ),
            mm_positions=None,
            mm_hashes=None,
        )

        external_tokens, _ = ucconnector.get_num_new_matched_tokens(request1, 0)
        self.assertEqual(external_tokens, 0)

    def test_get_num_new_matched_tokens_invalid_para(self):
        with patch.object(UnifiedCacheConnectorV1, "__init__", return_value=None):
            ucconnector = UnifiedCacheConnectorV1(None, None)
            ucconnector.block_size = self.block_size

        request1 = make_request(
            request_id=1,
            prompt_token_ids=random.sample(
                range(0, 10000), self.block_number * self.block_size
            ),
            mm_positions=None,
            mm_hashes=None,
        )

        # passing invalid params
        with self.assertRaises(AssertionError):
            external_tokens, _ = ucconnector.get_num_new_matched_tokens(
                request1, self.block_size + 1
            )

    def test_wait_for_save_not_layerwise_success(self):
        req_meta1 = MagicMock(spec=ReqMeta)
        req_meta1.request_id = "req1"
        req_meta1.dump_blocks = [
            (secrets.token_hex(8), i) for i in range(self.block_number)
        ]

        metadata = UCConnectorV1Metadata()
        metadata.requests = [req_meta1]

        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_dump(
            block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]
        ) -> Task:
            assert len(offset) == len(src_tensor) == len(block_ids)
            return Task()

        def mock_wait(task: Task) -> int:
            return 0

        mock_connector.dump.side_effect = mock_dump
        mock_connector.wait.side_effect = mock_wait
        ucconnector = self.init_uc(
            mock_connector, metadata=metadata, use_layerwise=False
        )
        ucconnector.wait_for_save()

    def test_wait_for_save_not_layerwise_invalid_para(self):
        with patch.object(UnifiedCacheConnectorV1, "__init__", return_value=None):
            ucconnector = UnifiedCacheConnectorV1(None, None)
            ucconnector.block_size = self.block_size
            ucconnector.use_layerwise = False
            ucconnector._connector_metadata = Mock()

        with self.assertRaises(AssertionError):
            ucconnector.wait_for_save()

    def test_start_load_kv_not_layerwise_success(self):
        req_meta1 = MagicMock(spec=ReqMeta)
        req_meta1.request_id = "req1"
        req_meta1.load_blocks = [
            (secrets.token_hex(8), i) for i in range(self.block_number)
        ]
        req_meta1.load_async = False

        metadata = UCConnectorV1Metadata()
        metadata.requests = [req_meta1]

        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_load(
            block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
        ) -> Task:
            assert len(offset) == len(dst_tensor) == len(block_ids)
            return Task()

        def mock_wait(task: Task) -> int:
            return 0

        mock_connector.load.side_effect = mock_load
        mock_connector.wait.side_effect = mock_wait

        ucconnector = self.init_uc(
            mock_connector, metadata=metadata, use_layerwise=False
        )
        forward_context = Mock()
        ucconnector.start_load_kv(forward_context)

    def test_start_load_kv_invalid_para(self):
        with patch.object(UnifiedCacheConnectorV1, "__init__", return_value=None):
            ucconnector = UnifiedCacheConnectorV1(None, None)
            ucconnector.block_size = self.block_size
            ucconnector._connector_metadata = Mock()

        forward_context = Mock()
        with self.assertRaises(AssertionError):
            ucconnector.start_load_kv(forward_context)

    def test_start_load_kv_layerwise_success(self):
        req_meta1 = MagicMock(spec=ReqMeta)
        req_meta1.request_id = "req1"
        req_meta1.load_blocks = [
            (secrets.token_hex(8), i) for i in range(self.block_number)
        ]

        metadata = UCConnectorV1Metadata()
        metadata.requests = [req_meta1]

        mock_connector = Mock(spec=UcmKVStoreBase)

        def mock_load(
            block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
        ) -> Task:
            assert len(offset) == len(dst_tensor) == len(block_ids)
            return Task()

        mock_connector.load.side_effect = mock_load
        ucconnector = self.init_uc(mock_connector, metadata=metadata)
        forward_context = Mock()
        ucconnector.start_load_kv(forward_context)
        assert mock_connector.load.call_count == 2 * self.num_layers


if __name__ == "__main__":
    unittest.main()
