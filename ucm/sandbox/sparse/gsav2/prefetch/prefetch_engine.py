# Copyright (c) Huawei Technologies,. Ltd 2004-2025. All rights reserved

import enum
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm_ascend.attention.utils import get_sfa_qsfa_packed_head_dim
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

from ucm.sandbox.sparse.gsav2.prefetch import gsa_prefetch
from ucm.sandbox.sparse.gsav2.time_cal import time_us
from ucm.sparse.utils import (
    GSA_SPARSE_LEN,
    IS_INDEXER_OFFLOAD,
    PREFETCH_TYPE,
    PTOPK_PREFETCH_ENABLE,
    SPA_TOKENS_LEN,
    STREAM_NUM,
    align_to_256bytes,
)


class SequenceStage(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()


class GSAPrefetchBase:
    def __init__(
        self,
        vllm_config: VllmConfig,
        async_thread: int,
        is_log: bool,
        cpu_to_bind: List[int],
        topk_layer_map: List[int],
    ) -> None:
        self._init_basic_config(vllm_config, async_thread, is_log)
        self._init_dims_and_engine(vllm_config, cpu_to_bind, topk_layer_map)

    def _init_basic_config(
        self, vllm_config: VllmConfig, async_thread: int, is_log: bool
    ) -> None:
        self.rank = vllm_config.parallel_config.rank
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.async_thread = async_thread
        self.layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        additional_config = (
            vllm_config.additional_config
            if vllm_config.additional_config is not None
            else {}
        )
        self.enable_sparse_sfa_c8 = bool(
            additional_config.get("enable_sparse_sfa_c8", False)
        )
        self.is_log = is_log
        self.max_block_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size
        self.head_size = vllm_config.model_config.get_head_size()
        self.num_head = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.step_time = 0
        self.select_bs_index = None

    def _init_dims_and_engine(
        self, vllm_config: VllmConfig, cpu_to_bind: List[int], topk_layer_map: List[int]
    ) -> None:
        self.max_bs = vllm_config.scheduler_config.max_num_seqs
        if not self.enable_sparse_sfa_c8:
            self.k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
            self.v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
            self.dtype = vllm_config.model_config.dtype
        else:
            if get_ascend_device_type() == AscendDeviceType.A5:
                self.dtype = torch.float8_e4m3fn
                self.c8_k_scale_cache_dtype = torch.float32
            else:
                self.dtype = torch.int8
                self.c8_k_scale_cache_dtype = torch.float16
            self.k_dim = get_sfa_qsfa_packed_head_dim(
                vllm_config.model_config.hf_text_config.kv_lora_rank,
                vllm_config.model_config.hf_text_config.qk_rope_head_dim,
            )
            self.v_dim = 0

        self.element_size = torch.tensor([], dtype=self.dtype).element_size()
        self.prefetch_engine_c = gsa_prefetch.GSAPrefetchEngineC(
            [
                self.layer_num,
                self.max_block_len,
                SPA_TOKENS_LEN,
                STREAM_NUM,
                GSA_SPARSE_LEN,
                PREFETCH_TYPE,
                self.dp_rank,
                self.tp_size,
                IS_INDEXER_OFFLOAD,
                self.enable_sparse_sfa_c8,
                self.max_bs,
            ],
            [self.block_size, self.num_head, self.k_dim, self.v_dim, self.element_size],
            self.is_log,
            self.rank,
            self.async_thread,
            cpu_to_bind,
            topk_layer_map,
            os.path.join(
                "./gsa_log/GLM/KVpre",
                f"kvcache_pre_log_dp{self.dp_rank}_tp{self.rank}.log",
            ),
        )
        self._slab_host_k_ptr = []
        self._slab_host_v_ptr = []
        self.k_cache_ptr = [None] * self.layer_num
        self.v_cache_ptr = [None] * self.layer_num

    def update_topk_index(self, layer_id, req_id) -> list:
        topk_indices = self.prefetch_engine_c.get_topk_indices(layer_id, req_id)
        return topk_indices

    def del_finish_meta(self, del_req, flag: bool = True) -> None:
        if PTOPK_PREFETCH_ENABLE and flag:
            self.prefetch_engine_c.del_blocks_map(del_req)
