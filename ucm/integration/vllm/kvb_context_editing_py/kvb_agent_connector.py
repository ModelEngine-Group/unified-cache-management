import copy
import hashlib
import math
import os
import pickle
import time 
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple
import itertools
import numpy as np
import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import(
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from ucm.integration.vllm.request_meta import KvbVllmRequestMeta
from vllm.distributed.parrallel_state import get_tp_group, get_world_group
from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.integration.vllm.device import create_device
from ucm.logger import init_logger
from ucm.observability import PrometheusStatsLogger
from ucm.shared.metrics import ucmmetrics
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1
from ucm.utils import Config
from ucm.integration.vllm.perf_counter import PerfCounters
from ucm.integration.vllm.model_maker_manager import ModelMarkerManager
if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext 
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
import vllm.forward_context as global_context
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from ucm.integration.vllm.ucm_connector import UCMDirectConnector, UCMConnector, UCMConnectorMetadata
from ucm.sparse.state import has_ucm_sparse
import torch_npu
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashWithGroupId,
    KVCacheBlock,
) 
import sys

logger = init_logger(__name__)


@dataclass
class RequestMeta:
    ucm_block_ids: list[bytes] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    # local_computed_block + external_computed_block
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    vllm_block_ids: list[int] = field(default_factory=list)
    token_processed: int = 0
    stats_flag: bool = False
    chunk_prefill_len: int = 0
    kvb_vllm_request_mata: KvbVllmRequestMeta = field(default_factory=KvbVllmRequestMeta)
    request_id: str = ""

@dataclass
class RequestDispatchMeta:
    load_block_ids: tuple[
        list[bytes], list[int]
    ]  #[0] mean ucm_block_ids, [1] means vllm_block_ids
    dump_block_ids: tuple[list[bytes], list[int]]
    ucm_block_ids: list[bytes]
    need_load: bool

@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, RequestDispatchMeta] = field(default_factory=dict)

class AgentConnector(UCMDirectConnector):
    """
    This connector means synchronize
    load -> forward -> save
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: Optional["KVCacheConfig"] = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)

        ucm_config = Config(vllm_config.kv_transfer_config)
        self.launch_config = ucm_config.get_config()
        self.kvb_agent_config_config = self.launch_config.get("kvb_agent_config",{})

        if  "model_marker_path" in self.kvb_agent_config:
            self.model_marker_path = self.kvb_agent_config["model_marker_path"]
        else:
            raise RuntimeError("need set model_marker_path path.")

        self.marker_manager = ModelMarkerManager(
            config_path=self.model_marker_path,
            vllm_config=self._vllm_config
        )
        
    
    def get_tool_result_block_ids(self, request):
        if request.sampling_params.extra_args is not None:
            pruned_args = request.sampling_params.extra_args["kv_edit_args"]
            logger.debug(f"pruned_args:{pruned_args}")
        all_token_ids = request.all_token_ids
        total_blocks = len(all_token_ids) // self.block_size
        slot_indices = [True] * total_blocks

        arr = np.array(all_token_ids, dtype=np.int64)

        # 查找倒数第一个[200020, 10, 200109, 3995]的起始位置
        marker_start = None
        session_marker_start = None
        marker_matches = self.marker_manager.find_marker_positions(arr, "user")
        if len(marker_matches) > 0:
            marker_start = marker_matches[-1] # 倒数第一个
            session_marker_start = marker_matches[0]
        if not marker_start:
            logger.warning("error:maker_start is None")
            return slot_indices, False
        logger.debug(f"marker_start: {marker_start}")
        logger.debug(f"marker_manager_start:{session_marker_start}")

        # 所有起始位置（200052）
        starts = self.marker_manager.find_marker_positions(arr, "tool_call")
        # 所有结束二元组（1579，5679）的第一个位置
        ends = self.marker_manager.find_marker_positions(arr, "response")

        #配对生成(start, end)元组 （原始token位置）
        ranges = [] # 每个元素为（s,e）
        j = 0
        last_end = 0 
        for s in starts:
            if s < session_marker_start:
                continue
            if s < last_end:
                logger.warning(f"error: s < last_end, s:{e}, last_end: {last_end}")
                ranges.pop()
                continue
            while j < len(ends) and ends[j] <= s:
                j += 1
            if j >= len(ends):
                break
            e = ends[j]
            ranges.append((s,e))
            last_end = e
            j += 1

        before_ranges = []
        after_ranges =  []

        # 按原始位置分类
        for s,e in ranges:
            if e < marker_start:                      # 完全在标记之前
                before_ranges.append((s,e))
            elif s > marker_start:                    # 完全在标记之后（标记占4个token）
                after_ranges.append((s,e))
            else:                                     # 跨越标记，归入before
                logger.error(f"error: 跨越标记，all_toekn_ids:{all_token_ids}")
                return slot_indices, False
            
        result_blocks = []  # 需要裁剪（设为False）的block索引
        pruned_args = request.sampling_params.extra_args["kv_edit_args"]
        logger.debug(f"pruned_args:{pruned_args}")
        if not 'tool_call' in pruned_args:
            return slot_indices, False
        pruned_list = pruned_args['tool_call']
        logger.debug(f"pruned_list:{pruned_list}")
        all_ranges = before_ranges + after_ranges

        for index, (s,e) in enumerate(all_ranges):
            if index in pruned_list:
                start_block = (s + self.block_size - 1) // self.block_size
                end_block = e // self.block_size
                if start_block < end_block:
                    # result_blocks.extend(range(start_block, end_block))
                    result_blocks.extend([
                        block_idx for block_idx in range(start_block, end_block)
                    ])
                
        pruned = len(result_blocks) * self.block_size
        logger.debug(f"pruned token:{pruned}")
        PerfCounters.get_inst().update("tool_result_tokens", pruned)
        PerfCounters.get_inst().update("pruned_ratio", pruned / len(all_token_ids))

        pruned= True
        for i in result_blocks:
            pruned = True
            slot_indices[i] = False

        return slot_indices, pruned

    def get_vllm_load_slot_indices(self, slot_indices, hbm_hit_block_num, external_hit_blocks, num_pruned_blocks):
        vllm_load_slot_indices = [False] * len(slot_indices)
        vllm_load_slot_indices[hbm_hit_block_num + num_pruned_blocks : hbm_hit_block_num + external_hit_blocks] = [True] * (external_hit_blocks - num_pruned_blocks)
        assert sum(vllm_load_slot_indices) == sum(slot_indices)
        return vllm_load_slot_indices
    
    def get_num_new_matched_tokens(
        self,
        request:"Request",
        num_computed_tokens: int,
    )  -> tuple[int, bool]:
        assert num_computed_tokens % self.block_size == 0
        hbm_hit_block_num = num_computed_tokens // self.block_size

        ucm_block_ids = self.generate_hash(
            self.block_size, request.all_token_ids, self._seed
        )

        slot_indices, pruned = self.get_tool_result_block_ids(request)
        num_pruned_blocks = min(hbm_hit_block_num, first_pruned_index)
        first_pruned_index = sys.maxsize
        if pruned:
            first_pruned_index = slot_indices.index(False)
        hbm_hit_block_num = min(hbm_hit_block_num, first_pruned_index)

        external_hit_blocks = 0
        external_block_ids = ucm_block_ids[hbm_hit_block_num:]
        if external_block_ids:
            try:
                external_hit_blocks = (
                    self._rank_consistency.lookup_on_prefix(
                            self.store, external_block_ids
                        )
                        +1
                    )
                self._prefetch_other_rank_hashest(
                    external_block_ids[:external_hit_blocks]
                )
            
            except RuntimeError as e:
                external_hit_blocks = 0
                logger.error(f"request{request.request_id} look up error. {e}")

        slot_indices[:hbm_hit_block_num] = [False] * hbm_hit_block_num
        slot_indices[hbm_hit_block_num + external_hit_blocks:] = [False] * (len(ucm_block_ids) - hbm_hit_block_num - external_hit_blocks)

        request.kvb_vllm_request_mata.ucm_slot_indices = slot_indices
        logger.info_once(
            f"request_id: {request.request_id},"
            f"prompt len: {len(request.all_token_ids)},"
            f"total_blocks_num: {len(ucm_block_ids)},"
            f"hit hbm: {hbm_hit_block_num},"
            f"hit external: {external_hit_blocks},"
            f"num_pruned_blocks: {num_pruned_blocks}"
            f"first_pruned_index: {first_pruned_index}"
        )

        if self.metrics_config:
            ucmmetrics.update_stats(
                {"interval_lookup_hit_rates": external_hit_blocks / len(ucm_block_ids)}
            )
        
        total_hit_block_num = hbm_hit_block_num + external_hit_blocks

        request.kvb_vllm_request_mata.hbm_hit_block_num = hbm_hit_block_num
        request.kvb_vllm_request_mata.num_pruned_blocks = num_pruned_blocks
        request.kvb_vllm_request_mata.vllm_load_slot_indices = self.get_vllm_load_slot_indices(slot_indices, hbm_hit_block_num, external_hit_blocks, num_pruned_blocks)

        external_hit_tokens = sum(slot_indices) * self.block_size

        # When all the tokens are cached in ssd or hbm,
        # we need to recompute the last token. This if condition will be removed
        # once vLLM scheduler provides a better solution in the future.
        num_total_hit_tokens = total_hit_block_num * self.block_size
        if num_total_hit_tokens == request.num_tokens:
            external_hit_blocks -= 1

        self .requests_meta[request.request_id] = RequestMeta(
            ucm_block_ids = ucm_block_ids,
            hbm_hit_block_num = hbm_hit_block_num,
            total_hit_block_num = hbm_hit_block_num,
            num_token_ids = len(request.all_token_ids),
            token_processed = num_total_hit_tokens,
            kvb_vllm_request_mata = request.kvb_vllm_request_mata,
            request_id = request.request_id
        ) 
        logger.debug(f"requst_id:{request.request_id}, external_hit_blocks:{external_hit_blocks}")
        return external_hit_tokens, False

    def update_state_after_alloc(
        self, request:"Request", blocks:"KVCacheBlocks", num_external_tokens: int
    ):
        pass 
    
    def _genrate_dispatch_meta(
        self,
        req_meta: RequestMeta,
        new_tokens: int,
        vllm_block_ids: list[int],
        need_load: bool = True,
    ) -> RequestDispatchMeta:
        """
        Request Blocks layout:
        ---------------------------------------------------------------------------------------------------
        | local_computed_block(HBM hit) | external_computed_block(external hit) | new_block(need to dump) |
        ---------------------------------------------------------------------------------------------------
        |  hbm_hit_block_num            |            LOAD                       | new_blocks_num          |
        ---------------------------------------------------------------------------------------------------
        |                                total_hit_block_num                                              |
        ---------------------------------------------------------------------------------------------------
        |                                scheduled_block_num                                              |
        """

        ucm_slot_indices = req_meta.kvb_vllm_request_mata.ucm_slot_indices
        vllm_load_slot_indices = req_meta.kvb_vllm_request_mata.vllm_request_meta.vllm_load_slot_indices
        ucm_block_ids = req_meta.ucm_block_ids
        req_meta.vllm_block_ids.extend(vllm_block_ids)

        load_ucm_block_ids, load_vllm_block_ids = [], []
        dump_ucm_block_ids, dump_vllm_block_ids = [], []

        if need_load:
            load_ucm_block_ids = [ucm_block_ids[i] for i in range(len(ucm_slot_indices)) if ucm_slot_indices[i]]
            load_vllm_block_ids = [vllm_block_ids[i] for i in range(len(vllm_load_slot_indices)) if vllm_load_slot_indices[i]]
        if req_meta.token_processed < req_meta.num_token_ids:
            start_idx = req_meta.token_processed // self.block_size
            end_idx = (req_meta.token_processed + new_tokens) // self.block_size
            dump_ucm_block_ids = ucm_block_ids[start_idx:end_idx]
            dump_vllm_block_ids = req_meta.vllm_block_ids[start_idx:end_idx]
            req_meta.token_processed + new_tokens

        return RequestDispatchMeta(
            (load_ucm_block_ids,load_vllm_block_ids),
            (dump_ucm_block_ids,dump_vllm_block_ids),
            ucm_block_ids,
            need_load,
        )


class UCMAgentConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config:"VllmConfig", role: KVConnectorRole, kv_cache_config:Optional["KVCacheConfig"] = None):
        KVConnectorBase_V1.__init__(
            self,
            vllm_config=vllm_config,
            role = role,
            kv_cache_config=kv_cache_config
        )
        self.connector:KVConnectorBase_V1
        ucm_config = Config(vllm_config.kv_transfer_config)
        self.launch_config = ucm_config.get_config()
        self._setup_ucm_merics(vllm_config, role)
        logger.info(f"self.launch_config:{self.launch_config}")

        use_layerwise=(
            self.launch_config.get("use_layerwise", False)
            if self.launch_config is not None
            else False
        )
        pp_enabled = self._vllm_config.parrallel_config.pipeline_parallel_size > 1
        if pp_enabled and not use_layerwise:
            raise RuntimeError(
                "Pipeline parallelism is not supported in UCMDirectConnector, please set use_layerwise=True."
            )

        if (
            hasattr(self._vllm_config.parrallel_config,"prefill_context_parallel_size")
            and hasattr(
                self._vllm_config.parrallel_config, "decode_context_parallel_size"
            )
            and self._vllm_config.parrallel_config.prefill_context_parallel_size
            * self._vllm_config.parrallel_config.decode_context_parallel_size
            > 1
        ):
            raise RuntimeError(
                "kvb is not supported in pcp and dcp."
            )
        elif use_layerwise:
            raise RuntimeError(
                "kvb is not supported while use_layerwise=Ture."
            )
        else:
            self.connector = AgentConnector(vllm_config, role, kv_cache_config)

            


