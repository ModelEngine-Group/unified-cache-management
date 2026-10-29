import copy
import enum
import hashlib
import math
import os
import pathlib
import pickle
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from functools import cache, wraps
from itertools import accumulate
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.parallel_state import get_world_group
from vllm.forward_context import (
    ForwardContext,
)
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus

if hasattr(torch, "npu") and torch.npu.is_available():
    # Ascend-only imports are delayed so that importing this module on
    # CUDA/CPU hosts does not raise ModuleNotFoundError.
    import torch_npu
    from vllm_ascend.attention.utils import get_sfa_qsfa_packed_head_dim
    from vllm_ascend.cpu_binding import CpuAlloc
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
else:
    # Non-Ascend platform: GSA is Ascend-only and must not be selected.
    torch_npu = None
    get_sfa_qsfa_packed_head_dim = None
    CpuAlloc = None
    AscendDeviceType = None
    get_ascend_device_type = None

# 新增类的import
from ucm.integration.vllm.ucm_connector import RequestHasher
from ucm.logger import init_logger
from ucm.sandbox.sparse.gsav2.gsa import GSAMetaData, GSASharedCPUPool
from ucm.sandbox.sparse.gsav2.gsa_log import GSALogger, Logger
from ucm.sandbox.sparse.gsav2.gsa_shared_memory import (
    SharedBlockTable,
    SharedTensorManager,
    SharedTensorPool,
    SharedTensorVersion,
)
from ucm.sandbox.sparse.gsav2.prefetch.prefetch_engine import (
    GSAPrefetchBase,
    SequenceStage,
)
from ucm.sandbox.sparse.gsav2.time_cal import time_us
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
)
from ucm.sparse.utils import (
    GSA_SPARSE_LEN,
    INDEXER_OFFLOAD_BLOCKS,
    INDEXER_PREFETCH_RATIO,
    INDEXER_PREFETCH_TOKENS,
    INDEXER_SPARSE_RATE,
    INIT_LOCAL_WINDOWS,
    IS_INDEXER_OFFLOAD,
    IS_PERFORMANCE_LOG,
    IS_SHARED_MEMCPY,
    KVCACHE_BUFFER_SIZE,
    LOCAL_WINDOWS_LEN,
    MAX_INDEXER_BLOCK_NUM,
    MAX_SLOTS_BLOCK_NUM,
    PREFETCH_SPACE,
    PREFETCH_THREAD_NUM,
    PTOPK_PREFETCH_ENABLE,
    SPA_TOKENS_LEN,
    STREAM_NUM,
    VLLM_ASCEND_VERSION,
    WARMUP_TOPK_LEN,
    DSAInputStruct,
    TopkIndices,
    bind_thread_memory_policy,
    init_indexer_layer_map,
)

# GSA v2-specific threshold, kept in the sandbox instead of ucm.sparse.utils
# (whose legacy default is 8400). Configurable via env var.
SEG_PREFILL_THRESHOLD = int(os.environ.get("UCM_GSA_SEG_PREFILL_THRESHOLD", "32768"))

ReqType = Union[str, int]

logger = init_logger(__name__)


class GSAforPD(UcmSparseBase):
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        self._init_basic_config(vllm_config)
        self._init_sparse_c8_config(vllm_config)
        self._init_runtime_state()
        self._init_speculative_config(vllm_config)
        self._init_slots_num_manager(role)
        if (
            role == UcmSparseRole.WORKER
            and PTOPK_PREFETCH_ENABLE
            and self.is_decode_node
        ):
            self._init_worker_shared_tables()
            self._init_worker_logger_and_pools(vllm_config)
            self._init_worker_indexer_state(vllm_config)
            self._init_prefetch_buffers(vllm_config)

    def _init_basic_config(self, vllm_config: VllmConfig):
        self.rank = vllm_config.parallel_config.rank
        self.device = vllm_config.device_config.device_type
        self.head_size = vllm_config.model_config.get_head_size()
        self.use_mla = vllm_config.model_config.use_mla
        self.block_size = vllm_config.cache_config.block_size
        self.element_size = vllm_config.model_config.dtype.itemsize
        self.num_head = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.indexer_dtype = vllm_config.model_config.dtype
        self.max_bs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_slots = MAX_SLOTS_BLOCK_NUM

    def _init_sparse_c8_config(self, vllm_config: VllmConfig):
        additional_config = (
            vllm_config.additional_config
            if vllm_config.additional_config is not None
            else {}
        )
        self.enable_sparse_li_c8 = bool(
            additional_config.get("enable_sparse_li_c8", False)
        )
        self.enable_sparse_sfa_c8 = bool(
            additional_config.get("enable_sparse_sfa_c8", False)
        )
        if not self.enable_sparse_sfa_c8:
            self.dtype = vllm_config.model_config.dtype
        elif get_ascend_device_type() == AscendDeviceType.A5:
            self.dtype = torch.float8_e4m3fn
            self.c8_k_scale_cache_dtype = torch.float32
        else:
            self.dtype = torch.int8
            self.c8_k_scale_cache_dtype = torch.float16

    def _init_runtime_state(self):
        self.gsa_metadata = None
        self.model_input = None
        self.gsa_stats = {}
        self.decode_index = []
        self.gsa_index = []
        self._buff_host_k = None
        self._buff_host_v = None
        self._buff_device_k = None
        self._buff_device_v = None
        self._load_index_host = None
        self._load_index_device = None
        self.preempt_req_output_tokens: Dict[ReqType, int] = {}
        self.is_share_memcpy = IS_SHARED_MEMCPY
        self.version = 0
        self.indexer_sparse_num_all = MAX_INDEXER_BLOCK_NUM
        self.has_prefill = False
        self.num_padding_reqs = 0
        self.alloc_block_sch = 0
        self._store_injected = False
        self._store_inject_retries = 0
        self._store_inject_max_retries = 5
        self._block_hash_injected_reqs = set()

    def _init_speculative_config(self, vllm_config: VllmConfig):
        self.num_spec_tokens = 1
        speculative_config = vllm_config.speculative_config
        if speculative_config:
            self.num_spec_tokens += speculative_config.num_speculative_tokens
        self.is_decode_node = vllm_config.kv_transfer_config.is_kv_consumer

    def _init_slots_num_manager(self, role: UcmSparseRole):
        if (
            role == UcmSparseRole.SCHEDULER
            and PTOPK_PREFETCH_ENABLE
            and self.is_decode_node
        ):
            self.slots_num_manager = SharedTensorManager(
                "gsa_glm52_slots_num_dp" + str(self.dp_rank), False
            )
        elif self.rank == 0 and self.is_decode_node:
            self.slots_num_manager = SharedTensorManager(
                "gsa_glm52_slots_num_dp" + str(self.dp_rank),
                True,
                self.indexer_sparse_num_all,
                self.num_slots,
            )

    def _init_worker_shared_tables(self):
        if self.rank == 0:
            if IS_INDEXER_OFFLOAD:
                self.block_table_sm = SharedBlockTable(
                    "gsa_glm52_slot_dp" + str(self.dp_rank), True
                )
            self.indexer_block_table_sm = SharedBlockTable(
                "indexer_glm52_slot_dp" + str(self.dp_rank), True
            )
            self.prefetch_status_sm = SharedTensorVersion(
                "prefetch_glm52_status_dp" + str(self.dp_rank),
                self.rank,
                self.tp_size,
                True,
            )
        else:
            if IS_INDEXER_OFFLOAD:
                self.block_table_sm = SharedBlockTable(
                    "gsa_glm52_slot_dp" + str(self.dp_rank), False
                )
            self.indexer_block_table_sm = SharedBlockTable(
                "indexer_glm52_slot_dp" + str(self.dp_rank), False
            )
            self.prefetch_status_sm = SharedTensorVersion(
                "prefetch_glm52_status_dp" + str(self.dp_rank),
                self.rank,
                self.tp_size,
                False,
            )

    def _init_worker_logger_and_pools(self, vllm_config: VllmConfig):
        self.topk_layer_info, self.topk_layer_map, self.topk_layer_num = (
            init_indexer_layer_map(vllm_config)
        )
        self.local_rank = get_world_group().local_rank
        numa_node, cpu_to_bind = self.bind_cpus()
        log_type = os.getenv("GSA_LOG_TYPE", GSALogger.LOG_TYPE_NORMAL)
        self.log = GSALogger(
            log_dir="gsa_log/GLM/pyLog",
            dp_rank=self.dp_rank,
            tp_rank=self.rank,
            log_type=log_type,
            show_time=False,
        )
        self.gsa_preempt_shared_pool = GSASharedCPUPool(
            self.max_model_len // self.block_size * 2
        )
        if IS_INDEXER_OFFLOAD:
            self.gsa_cpu_shared_pool = GSASharedCPUPool(self.num_slots)
        self.prefetch_engine = GSAPrefetchBase(
            vllm_config, PREFETCH_THREAD_NUM, True, cpu_to_bind, self.topk_layer_map
        )
        bind_thread_memory_policy(numa_node)

    def _init_worker_indexer_state(self, vllm_config: VllmConfig):
        self.select_block_index = DSAInputStruct(
            (vllm_config.scheduler_config.max_num_seqs), self.device, torch.int32
        )
        self.topk_indices_len = 2048
        self.kv_caches = [None] * self.layer_num
        self.indexer_scale = [None] * self.layer_num
        self.indexer_dim = 128
        self.indexer_scale_dim = 1
        self.is_prefetch_token = False
        self.prefetch_space_token = PREFETCH_SPACE
        self.init_gsa_metadata(vllm_config)
        self.swap_index = []
        self.prefetch_topk = []
        self.warmup_topk = []
        self.is_all_gsa = False
        self.is_test = False
        self.last_test_step = 0
        self.req_ids_bs = []
        self.indexer_k_cache_sparse = [None] * self.layer_num
        if self.rank == 0:
            self.indexer_cpu_shared_pool = GSASharedCPUPool(self.indexer_sparse_num_all)
        if IS_INDEXER_OFFLOAD:
            self.indexer_last_query = None
            self.indexer_last_weight = None

    def _init_prefetch_buffers(self, vllm_config: VllmConfig):
        dump_max_num = math.ceil(self.max_num_scheduled_tokens / self.block_size) * 2
        if self.enable_sparse_li_c8:
            self.indexer_pc_dump_tmp_c8 = torch.empty(
                (
                    math.ceil(self.max_model_len / self.block_size),
                    self.block_size,
                    1,
                    128,
                ),
                dtype=torch.int8,
                device=self.device,
                pin_memory=False,
            )
            self.indexer_pc_dump_scale_tmp = torch.empty(
                (
                    math.ceil(self.max_model_len / self.block_size),
                    self.block_size,
                    1,
                    1,
                ),
                dtype=torch.float16,
                device=self.device,
                pin_memory=False,
            )
        self.k_dim = (
            get_sfa_qsfa_packed_head_dim(
                vllm_config.model_config.hf_text_config.kv_lora_rank,
                vllm_config.model_config.hf_text_config.qk_rope_head_dim,
            )
            if self.enable_sparse_sfa_c8
            else vllm_config.model_config.hf_text_config.kv_lora_rank
        )
        self.indexer_pc_dump_tmp = torch.empty(
            (math.ceil(self.max_model_len / self.block_size), self.block_size, 1, 128),
            dtype=self.indexer_dtype,
            device=self.device,
            pin_memory=False,
        )
        # 0: all_hit_blockid, 1: all_mv_blockid
        self.indexer_pc_hit_index = TopkIndices(
            (
                2,
                vllm_config.scheduler_config.max_num_seqs,
                math.ceil(self.max_model_len / self.block_size),
            ),
            self.device,
            torch.int32,
        )
        # 0: dump_vllm_id, 1: dump_gsa_id
        self.indexer_pc_dump_index = DSAInputStruct(
            (2, dump_max_num), self.device, torch.int32
        )
        self.indexer_pc_dump_num = 0
        self.indexer_pc_hit_num = []
        self.init_host_slabs_from_kv(self.num_slots)
        self.prefetch_engine.prefetch_engine_c.set_kv_buffer(
            self._buff_host_k,
            self._buff_host_v,
            self._buff_device_k,
            self._buff_device_v,
            self._load_index_host,
            self._load_index_device,
        )
        self.gsa_swap_topk_cpu_ptr = None
        self.gsa_swap_topk_npu_ptr = None
        if vllm_config.speculative_config:
            self.select_gsa_prefetch_topk = torch.zeros(
                (self.topk_layer_num, self.max_bs, 1, 2048),
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            ).contiguous()

    def init_gsa_metadata(self, vllm_config):
        self.cudagraph_capture_sizes = (
            vllm_config.compilation_config.cudagraph_capture_sizes
        )
        max_graph_bs = max(max(self.cudagraph_capture_sizes), self.max_bs)

        self.dsa_block_table = TopkIndices(  # 78 * bs * 512 * 4
            (
                self.topk_layer_num,
                self.max_bs,
                vllm_config.model_config.max_model_len // self.block_size,
            ),
            self.device,
            torch.int32,
        )
        self.dsa_seq_len = DSAInputStruct((self.max_bs), self.device, torch.int32)
        self.dsa_slot_mapping = DSAInputStruct(  # 8192 * 4
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.device,
            torch.int32,
        )

        self.gsa_prefetch_index = DSAInputStruct(
            (max_graph_bs), self.device, torch.int64
        )
        self.gsa_prefetch_topk = TopkIndices(  # 78 * bs * 2048 * 4
            (self.topk_layer_num, max_graph_bs, 1, 2048), self.device, torch.int32
        )

        self.gsa_swap_index = DSAInputStruct(
            (max_graph_bs + 2), self.device, torch.int64
        )

        self.gsa_swap_topk = TopkIndices(  # 78 * bs * 4096 * 4
            (self.topk_layer_num, (max_graph_bs + 2), 1, SPA_TOKENS_LEN),
            self.device,
            torch.int32,
        )

        self.gsa_warmup_index = DSAInputStruct(
            (self.max_bs * WARMUP_TOPK_LEN), self.device, torch.int64
        )
        self.gsa_warmup_topk = TopkIndices(
            (self.topk_layer_num, self.max_bs * WARMUP_TOPK_LEN, 1, 2048),
            self.device,
            torch.int32,
        )
        self.gsa_prefill_query = torch.tensor(
            list(
                range(
                    self.num_spec_tokens,
                    (self.max_bs + 1) * self.num_spec_tokens,
                    self.num_spec_tokens,
                )
            ),
            device=self.device,
            dtype=torch.int32,
        )
        self.gsa_topk_win_tmp = torch.zeros(
            (max_graph_bs + 2, SPA_TOKENS_LEN),
            dtype=torch.int32,
            device=self.device,
        )
        self.gsa_swap_topk_bs = self.gsa_swap_topk.cpu.shape[1]
        #### indexer
        if IS_INDEXER_OFFLOAD:
            self.indexer_last_query = TopkIndices(  # 78 * bs * 32 * 128 * 2
                (self.layer_num, max_graph_bs, 32, 128), self.device, self.dtype
            )
            self.indexer_last_weight = TopkIndices(  # 78 * bs * 32 * 2
                (self.layer_num, max_graph_bs, 32), self.device, self.dtype
            )

            self.indexer_last_q_index = DSAInputStruct(
                (max_graph_bs), self.device, torch.int64
            )

    def bind_cpus(self):
        binder = CpuAlloc(self.local_rank)
        binder.build_cpu_pools()
        binder.allocate()
        current_npu = binder.device_info.running_npu_list[self.local_rank]
        cpu_pool = binder.npu_cpu_pool.get(current_npu, [])
        if not cpu_pool:
            return None, None
        anchor_cpu = cpu_pool[0]
        numa_node = binder.cpu_node.get(anchor_cpu)

        gsa_thread_num = 20
        if IS_INDEXER_OFFLOAD:
            gsa_thread_num = 24

        if len(binder.assign_main[current_npu]) > gsa_thread_num:
            return numa_node, binder.assign_main[current_npu][-1 * gsa_thread_num :]
        else:
            return numa_node, binder.assign_main[current_npu][1:]

    def print_log(self, msg):
        self.log.info(msg)

    def _find_store_connector(self, kv_group):
        """定位持有有效 store 的 connector 对象，未就绪返回 None"""
        if type(kv_group).__name__ != "AscendMultiConnector":
            connector = kv_group
            if not hasattr(connector, "store") and hasattr(connector, "connector"):
                connector = connector.connector
            return connector
        if not hasattr(kv_group, "_connectors"):
            logger.info(
                "[GSA Store Inject] _connectors missing in AscendMultiConnector, ERROR"
            )
            return None
        for one_connect in kv_group._connectors:
            if hasattr(one_connect, "store"):
                logger.info(
                    "[GSA Store Inject] AscendMultiConnector get store. connector_type=%s",
                    type(one_connect).__name__,
                )
                return one_connect
            if hasattr(one_connect, "connector") and one_connect.connector is not None:
                connector = one_connect.connector
                if hasattr(connector, "store"):
                    return connector
                logger.info(
                    "[GSA Store Inject] connector has no store, keep searching. connector_type=%s",
                    type(connector).__name__,
                )
        return None

    def _inject_store_v1(self, connector) -> bool:
        """将 C++ store 指针注入 prefetch engine 并校验，失败返回 False"""
        store_ptr = connector.store.cc_store()
        if store_ptr == 0:
            logger.info(
                "[GSA Store Inject] cc_store() returned 0 (null pointer), "
                "C++ backend not ready, will retry later"
            )
            return False
        self.prefetch_engine.prefetch_engine_c.set_store_v1(store_ptr)
        actual_store_ptr = self.prefetch_engine.prefetch_engine_c.get_store_v1_ptr()
        cpp_self_ptr = self.prefetch_engine.prefetch_engine_c.get_self_ptr()
        logger.info(
            "[GSA Store Inject] C++ verify cpp_self=%s, expected_store=%s, actual_store=%s",
            hex(cpp_self_ptr),
            hex(store_ptr),
            hex(actual_store_ptr),
        )
        if actual_store_ptr != store_ptr:
            raise RuntimeError(
                "StoreV1 pointer verification failed: "
                f"expected={hex(store_ptr)}, actual={hex(actual_store_ptr)}"
            )
        self._store_injected = True
        logger.info(
            "[GSA Store Inject] SUCCESS store_ptr=%s, cpp_self=%s",
            hex(store_ptr),
            hex(cpp_self_ptr),
        )
        return True

    def _log_store_inject_retry(self, reason: str) -> None:
        """Log a store-injection retry and escalate to an error after
        ``_store_inject_max_retries`` consecutive attempts."""
        if self._store_inject_retries >= self._store_inject_max_retries:
            logger.error(
                "[GSA Store Inject] FAILED after %d attempts (%s); "
                "GSA merge path is not usable without the store pointer.",
                self._store_inject_retries,
                reason,
            )
            raise RuntimeError(
                f"[GSA Store Inject] store injection failed after "
                f"{self._store_inject_retries} attempts: {reason}"
            )
        logger.info(
            "[GSA Store Inject] %s (attempt %d/%d), will retry later",
            reason,
            self._store_inject_retries,
            self._store_inject_max_retries,
        )

    def _ensure_store_injected(self):
        if self._store_injected:
            return
        self._store_inject_retries += 1
        try:
            kv_group = get_kv_transfer_group()
            logger.info(
                "[GSA Store Inject] kv_group_type=%s, has_store=%s, has_connector=%s",
                type(kv_group).__name__,
                hasattr(kv_group, "store"),
                hasattr(kv_group, "connector"),
            )
            connector = self._find_store_connector(kv_group)
            if (
                connector is None
                or not hasattr(connector, "store")
                or connector.store is None
            ):
                self._log_store_inject_retry("store not ready")
                return
            self._inject_store_v1(connector)
        except Exception as e:
            if self._store_inject_retries >= self._store_inject_max_retries:
                logger.error(
                    "[GSA Store Inject] FAILED with exception after %d attempts: %s",
                    self._store_inject_retries,
                    e,
                )
                raise
            logger.info(
                "[GSA Store Inject] FAILED with exception (attempt %d/%d): %s, "
                "will retry later",
                self._store_inject_retries,
                self._store_inject_max_retries,
                e,
            )

    def _inject_block_hashes(
        self, req_id, block_idxs, pool_slots, inject_indexer=False
    ):
        req_meta = self.gsa_metadata.gsa_stats[req_id]
        if req_id in self._block_hash_injected_reqs:
            return
        if req_meta.block_hashes is None:
            return
        assert len(block_idxs) == len(pool_slots)
        ids = np.ascontiguousarray(
            np.frombuffer(b"".join(req_meta.block_hashes), dtype=np.uint8)
        )
        k_refs = [
            self._slab_preempt_k_ptr + self._slab_preempt_k_offset * s
            for s in pool_slots
        ]
        if not self.enable_sparse_sfa_c8:
            v_refs = [
                self._slab_preempt_v_ptr + self._slab_preempt_v_offset * s
                for s in pool_slots
            ]
        else:
            v_refs = []

        self.prefetch_engine.prefetch_engine_c.set_blocks_map(
            req_id, ids, len(req_meta.block_hashes), block_idxs, k_refs, v_refs
        )
        if IS_INDEXER_OFFLOAD and inject_indexer:
            i_refs = [
                self.indexer_cache_preempt_i_ptr
                + self.indexer_cache_preempt_i_offset * s
                for s in pool_slots
            ]
            self.prefetch_engine.prefetch_engine_c.set_last_indexer_blocks(
                req_id, block_idxs, i_refs
            )

        self.log.info(
            f"[GSA Block Hash Inject] FINISH set_blocks_map for req_id={req_id}"
        )
        self._block_hash_injected_reqs.add(req_id)

    def _prefetch_cache_blocks(
        self,
        index_in_batch: int,
        indexer_cache: torch.Tensor,
        vllm_indexer_cache: torch.Tensor,
        indexer_tmp: torch.Tensor,
    ) -> None:
        """将 vllm 缓存中命中块 gather 到 tmp，再 scatter 进 indexer 缓存"""
        hit_num = self.indexer_pc_hit_num[index_in_batch]
        block_view = self.block_size * vllm_indexer_cache.shape[-1]
        gathered = indexer_tmp[:hit_num].view(-1, block_view)
        torch.ops._C_ascend.gsa_gather_copy(
            gathered,
            vllm_indexer_cache.view(-1, block_view),
            self.indexer_pc_hit_index.npu[0][index_in_batch][:hit_num]
            .view(-1, 1)
            .expand((hit_num, block_view)),
        )
        torch.ops._C_ascend.gsa_scatter_copy(
            indexer_cache.view(-1, block_view),
            gathered,
            self.indexer_pc_hit_index.npu[1][index_in_batch][:hit_num]
            .view(-1, 1)
            .expand((hit_num, block_view)),
        )

    def _prefetch_scale_blocks(
        self,
        index_in_batch: int,
        indexer_scale: torch.Tensor,
        vllm_indexer_scale: torch.Tensor,
    ) -> None:
        """c8 场景：将命中块的 scale gather/scatter 到 indexer 缓存"""
        hit_num = self.indexer_pc_hit_num[index_in_batch]
        gathered = self.indexer_pc_dump_scale_tmp[:hit_num].view(-1, self.block_size)
        torch.ops._C_ascend.gsa_gather_copy(
            gathered,
            vllm_indexer_scale.view(-1, self.block_size),
            self.indexer_pc_hit_index.npu[0][index_in_batch][:hit_num]
            .view(-1, 1)
            .expand((hit_num, self.block_size)),
        )
        torch.ops._C_ascend.gsa_scatter_copy(
            indexer_scale.view(-1, self.block_size),
            gathered,
            self.indexer_pc_hit_index.npu[1][index_in_batch][:hit_num]
            .view(-1, 1)
            .expand((hit_num, self.block_size)),
        )

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        indexer_cache: Optional[torch.Tensor] = None,
        indexer_scale: Optional[torch.Tensor] = None,
        vllm_indexer_cache: Optional[torch.Tensor] = None,
        vllm_indexer_scale: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        phase: Optional[str] = None,
        k_hash: Optional[torch.Tensor] = None,
        decode_ql_nope: Optional[torch.Tensor] = None,
        decode_q_pe: Optional[torch.Tensor] = None,
        has_indexer: Optional[bool] = True,
        use_li_c8: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node:
            return query, key, value, output
        if IS_PERFORMANCE_LOG:
            start = time.perf_counter()
        layer_id = int(layer_name.split(".")[2])
        if layer_id >= self.layer_num:
            return query, key, value, output
        self.init_kvcache_ptrs(
            [key, value, vllm_indexer_cache, vllm_indexer_scale],
            indexer_cache,
            indexer_scale,
            layer_id,
        )
        if not has_indexer:
            return query, key, value, output
        indexer_tmp = (
            self.indexer_pc_dump_tmp_c8 if use_li_c8 else self.indexer_pc_dump_tmp
        )
        for req_id in self.req_ids_bs:
            index_in_batch = self.gsa_metadata.gsa_stats[req_id].index_in_batch
            if self.indexer_pc_hit_num[index_in_batch] <= 0:
                continue
            self._prefetch_cache_blocks(
                index_in_batch, indexer_cache, vllm_indexer_cache, indexer_tmp
            )
            if use_li_c8:
                self._prefetch_scale_blocks(
                    index_in_batch, indexer_scale, vllm_indexer_scale
                )
        return query, key, value, output

    def get_num_output_tokens(self, req_id):
        num_output_tokens = self.gsa_metadata.gsa_stats[req_id].num_output_tokens
        return num_output_tokens

    # @time_us
    def _init_dsa_input_buffers(self, req_ids_bs):
        dsa_seq_len = [None] * len(req_ids_bs)
        slot_mapping = [None] * len(req_ids_bs)
        self.indexer_pc_hit_num = [None] * len(req_ids_bs)
        dump_vllm_id = [None] * len(req_ids_bs)
        dump_gsa_id = [None] * len(req_ids_bs)
        self.dsa_block_table.np[:] = 0  # 保证长度一致
        self.dsa_slot_mapping.np[:] = -1
        self.dsa_seq_len.np[:] = 1
        self.indexer_pc_hit_index.np[:] = 0
        return dsa_seq_len, slot_mapping, dump_vllm_id, dump_gsa_id

    def _update_dsa_block_table(self, req_id, dsa_seq_len):
        req_meta = self.gsa_metadata.gsa_stats[req_id]
        slots = req_meta.indexer_slab_host_slot
        index_in_batch = req_meta.index_in_batch
        self.dsa_block_table.np[:, index_in_batch, : len(slots)] = slots
        dsa_seq_len[index_in_batch] = req_meta.get_seq_len()
        if IS_INDEXER_OFFLOAD and req_meta.is_indexer_init:
            dsa_seq_len[index_in_batch] = req_meta.get_indexer_sparse_len()
            if req_meta.add_blocks_len:
                req_meta.indexer_block_tables[:, req_meta.indexer_sparse_len() - 1] = (
                    slots[-1]
                )
                self.prefetch_engine.prefetch_engine_c.update_indexer_map(
                    req_id, len(req_meta.slab_host_slot) - 1, slots[-1], "add"
                )
            self.dsa_block_table.np[
                :, index_in_batch, : req_meta.indexer_sparse_len()
            ] = req_meta.indexer_block_tables[:, : req_meta.indexer_sparse_len()]
        return req_meta, slots, index_in_batch

    def _calc_decode_slot_mapping(self, req_meta, slots, index_in_batch, slot_mapping):
        if self.num_spec_tokens == 1:
            slot_mapping[index_in_batch] = [
                slots[-1] * self.block_size
                + (req_meta.get_seq_len() - 1) % self.block_size
            ]
        else:
            slot_mapping[index_in_batch] = [None] * self.num_spec_tokens
            if (
                req_meta.get_seq_len() % self.block_size < self.num_spec_tokens
                and req_meta.get_seq_len() % self.block_size != 0
            ):
                x = req_meta.get_seq_len() % self.block_size
                slot_mapping[index_in_batch] = list(
                    range(
                        slots[-2] * self.block_size
                        + self.block_size
                        - (self.num_spec_tokens - x),
                        slots[-2] * self.block_size + self.block_size,
                    )
                ) + [i + slots[-1] * self.block_size for i in range(x)]
            else:
                if req_meta.get_seq_len() % self.block_size == 0:
                    slot_mapping[index_in_batch] = list(
                        range(
                            slots[-1] * self.block_size
                            + self.block_size
                            - self.num_spec_tokens,
                            slots[-1] * self.block_size + self.block_size,
                        )
                    )
                else:
                    slot_mapping[index_in_batch] = list(
                        range(
                            slots[-1] * self.block_size
                            + (req_meta.get_seq_len() % self.block_size)
                            - self.num_spec_tokens,
                            slots[-1] * self.block_size
                            + (req_meta.get_seq_len() % self.block_size),
                        )
                    )

    def _calc_prefill_slot_mapping(
        self,
        req_id,
        req_meta,
        slots,
        index_in_batch,
        slot_mapping,
        dump_vllm_id,
        dump_gsa_id,
    ):
        slot_mapping[index_in_batch] = []
        if math.ceil(req_meta.get_seq_len() / self.block_size) == len(slots):
            if req_meta.num_computed_tokens != req_meta.slot_mapping_cal_len:
                hit_num = math.ceil(req_meta.num_computed_tokens / self.block_size)
                self.indexer_pc_hit_num[index_in_batch] = hit_num
                self.indexer_pc_hit_index.np[0][index_in_batch][:hit_num] = (
                    req_meta.blocks[:hit_num]
                )
                self.indexer_pc_hit_index.np[1][index_in_batch][:hit_num] = slots[
                    :hit_num
                ]
            dump_list = list(
                range(
                    req_meta.num_computed_tokens // self.block_size,
                    req_meta.get_seq_len() // self.block_size,
                )
            )  # 适配mtp  127时申请2个BLock
            dump_vllm_id[index_in_batch] = [req_meta.blocks[x] for x in dump_list]
            dump_gsa_id[index_in_batch] = [slots[x] for x in dump_list]
            for one_token in range(
                req_meta.num_computed_tokens, req_meta.get_seq_len()
            ):
                block_idx = int(one_token / self.block_size)
                slot_idx = one_token % self.block_size
                slot_mapping[index_in_batch].append(
                    slots[block_idx] * self.block_size + slot_idx
                )
            req_meta.slot_mapping_cal_len = req_meta.get_seq_len()
        else:
            self.log.error(
                f"[GSA] ERROR indexer_slab_host_slot is small than seq len for req_id {req_id}\n"
            )
            self.log.error(
                f"[GSA] ERROR indexer_slab_host_slot len {len(slots)} seq_len {req_meta.get_seq_len()}\n"
            )
            self.log.error(
                f"[GSA] ERROR prompt len {req_meta.num_prompt_tokens}"
                f"is swap {req_id in self.preempt_req_output_tokens}\n"
            )
            self.log.error(
                f"[GSA] ERROR num_computed_tokens {req_meta.num_computed_tokens}"
                f"num_output_tokens {req_meta.num_output_tokens}\n"
            )

    def _update_req_slot_mapping(
        self,
        req_id,
        req_meta,
        slots,
        index_in_batch,
        slot_mapping,
        dump_vllm_id,
        dump_gsa_id,
    ):
        self.indexer_pc_hit_num[index_in_batch] = 0
        dump_vllm_id[index_in_batch] = []
        dump_gsa_id[index_in_batch] = []
        if self.get_num_output_tokens(req_id) >= 1:
            self._calc_decode_slot_mapping(
                req_meta, slots, index_in_batch, slot_mapping
            )
        else:
            self._calc_prefill_slot_mapping(
                req_id,
                req_meta,
                slots,
                index_in_batch,
                slot_mapping,
                dump_vllm_id,
                dump_gsa_id,
            )

    def _copy_dsa_input_to_npu(
        self, dsa_seq_len, slot_mapping, dump_vllm_id, dump_gsa_id
    ):
        all_slot_mapping = [item for sublist in slot_mapping for item in sublist]
        dump_gsa_id_all = [item for sublist in dump_gsa_id for item in sublist]
        dump_vllm_id_all = [item for sublist in dump_vllm_id for item in sublist]
        self.indexer_pc_dump_num = len(dump_gsa_id_all)
        self.indexer_pc_dump_index.np[0][: self.indexer_pc_dump_num] = dump_vllm_id_all
        self.indexer_pc_dump_index.np[1][: self.indexer_pc_dump_num] = dump_gsa_id_all
        self.indexer_pc_dump_index.copy_to_npu(-1)
        self.indexer_pc_hit_index.copy_to_npu(-1)
        self.dsa_slot_mapping.np[: len(all_slot_mapping)] = all_slot_mapping
        self.dsa_seq_len.np[: len(dsa_seq_len)] = dsa_seq_len
        self.dsa_slot_mapping.copy_to_npu(-1)
        self.dsa_block_table.copy_to_npu(-1)
        self.dsa_seq_len.copy_to_npu(-1)
        return len(all_slot_mapping)

    def get_dsa_input(self):
        if IS_PERFORMANCE_LOG:
            time_seq_len = 0
            time_slot_mapping = 0
            time_copy_to_npu = 0
        req_ids_bs = self.req_ids_bs
        dsa_seq_len, slot_mapping, dump_vllm_id, dump_gsa_id = (
            self._init_dsa_input_buffers(req_ids_bs)
        )
        for req_id in req_ids_bs:
            if IS_PERFORMANCE_LOG:
                start = time.perf_counter()
            req_meta, slots, index_in_batch = self._update_dsa_block_table(
                req_id, dsa_seq_len
            )
            if IS_PERFORMANCE_LOG:
                time_seq_len += (time.perf_counter() - start) * 1_000_000
                start = time.perf_counter()
            self._update_req_slot_mapping(
                req_id,
                req_meta,
                slots,
                index_in_batch,
                slot_mapping,
                dump_vllm_id,
                dump_gsa_id,
            )
            if IS_PERFORMANCE_LOG:
                time_slot_mapping += (time.perf_counter() - start) * 1_000_000
        if IS_PERFORMANCE_LOG:
            self.log.info(
                f"[GSA Time] get_dsa_input for rank {self.rank} time_seq_len {time_seq_len}"
                f"time_slot_mapping {time_slot_mapping}"
            )
            start = time.perf_counter()
        result = self._copy_dsa_input_to_npu(
            dsa_seq_len, slot_mapping, dump_vllm_id, dump_gsa_id
        )
        if IS_PERFORMANCE_LOG:
            time_copy_to_npu = (time.perf_counter() - start) * 1_000_000
            self.log.info(
                f"[GSA Time] get_dsa_input for rank {self.rank} time_copy_to_npu {time_copy_to_npu}"
            )

        return result

    # @time_us
    def _update_prefetch_status_all(self):
        if len(self.need_prefetch_index) > 0:
            while self.prefetch_engine.prefetch_engine_c.get_prefetch_status():
                time.sleep(0.001)
            self.is_prefetch_token = True
        elif len(self.gsa_index) > 0:
            is_done = not self.prefetch_engine.prefetch_engine_c.get_prefetch_status()
            self.prefetch_status_sm.update_prefetch_status(
                is_done, self.prefetch_engine.step_time
            )
            self.prefetch_status_sm.update_version(self.prefetch_engine.step_time)
            self.is_prefetch_token = self.prefetch_status_sm.is_version_same(
                self.prefetch_engine.step_time
            )

    def _reset_gsa_buffers_all(self):
        self.prefetch_topk = []
        self.warmup_topk = []
        self.swap_index = []
        self.select_block_index.np[:] = 0
        self.gsa_swap_index.np[:] = 0
        self.gsa_prefetch_index.np[:] = 0

    def _init_indexer_offload_all(self):
        if not IS_INDEXER_OFFLOAD:
            return
        self.indexer_last_q_index_tmp = []
        self.indexer_last_q_index_fb = [0]
        self.need_indexer_sparse_bs = []
        self.need_indexer_sparse_reqs = []
        self.need_cal_k_mean = []
        self.indexer_last_q_index.np[:] = 0
        self.is_prefetch_indexer = False
        if self.prefetch_engine.step_time % 10 == 0:
            self.is_prefetch_indexer = True
        else:
            for req_id in self.req_ids_bs:
                req_meta = self.gsa_metadata.gsa_stats[req_id]
                if (
                    req_meta.num_prompt_tokens > INDEXER_PREFETCH_TOKENS
                    and self.get_num_output_tokens(req_id) == 0
                ):
                    self.is_prefetch_indexer = True
                    break

    def _collect_and_swap_tokens_all(self) -> tuple[int, float, float]:
        req_cnt = 0
        prefetch_req_pre = 0
        use_tokens_ptrs = []
        seq_lens = []
        swap_index = []
        use_req_ids = []
        for req_id in self.req_ids_bs:
            req_meta = self.gsa_metadata.gsa_stats[req_id]
            if req_meta.num_prompt_tokens <= SEG_PREFILL_THRESHOLD or (
                not self.get_num_output_tokens(req_id) == 0
                and req_meta.stage() == SequenceStage.PREFILL
            ):
                continue
            index_in_batch = req_meta.index_in_batch
            query_local = self.model_input["query_locals"]
            if (
                IS_INDEXER_OFFLOAD
                and req_meta.num_prompt_tokens > INDEXER_PREFETCH_TOKENS
            ):
                self._collect_indexer_meta(
                    req_id, req_meta, index_in_batch, query_local
                )
            if IS_PERFORMANCE_LOG:
                start = time.perf_counter()
            self._collect_warmup_topk(req_id, req_meta, index_in_batch, query_local)
            if req_meta.is_gsa() and req_meta.is_gsa_init:
                req_cnt = self._collect_gsa_tokens(
                    req_id,
                    req_meta,
                    index_in_batch,
                    query_local,
                    req_cnt,
                    use_tokens_ptrs,
                    seq_lens,
                    swap_index,
                    use_req_ids,
                )
            if IS_PERFORMANCE_LOG:
                prefetch_req_pre += (time.perf_counter() - start) * 1_000_000
        if IS_PERFORMANCE_LOG:
            start = time.perf_counter()
        self._pre_swap_tokens_all(use_tokens_ptrs, seq_lens, swap_index, use_req_ids)
        time_update_tokens = 0
        if IS_PERFORMANCE_LOG:
            time_update_tokens = (time.perf_counter() - start) * 1_000_000
        return req_cnt, prefetch_req_pre, time_update_tokens

    def _collect_indexer_meta(self, req_id, req_meta, index_in_batch, query_local):
        if not (
            IS_INDEXER_OFFLOAD and req_meta.num_prompt_tokens > INDEXER_PREFETCH_TOKENS
        ):
            return
        self.need_cal_k_mean.append(req_id)
        if self.is_prefetch_indexer:
            self.need_indexer_sparse_bs.append(index_in_batch)
            self.need_indexer_sparse_reqs.append(req_id)
            if self.get_num_output_tokens(req_id) == 0:
                self.indexer_last_q_index_tmp.append(
                    [query_local[index_in_batch + 1] - 1]
                )
                self.indexer_last_q_index_fb.append(1)
            else:
                self.indexer_last_q_index_tmp.append(
                    list(
                        range(
                            query_local[index_in_batch], query_local[index_in_batch + 1]
                        )
                    )
                )
                self.indexer_last_q_index_fb.append(
                    query_local[index_in_batch + 1] - query_local[index_in_batch]
                )

    def _collect_warmup_topk(self, req_id, req_meta, index_in_batch, query_local):
        if not (
            self.get_num_output_tokens(req_id) == 0
            and req_meta.num_prompt_tokens > SEG_PREFILL_THRESHOLD
        ):
            return
        need_warmup_len = min(req_meta.num_scheduled_tokens, WARMUP_TOPK_LEN)
        self.warmup_topk.append(
            [
                list(
                    range(
                        query_local[index_in_batch + 1] - need_warmup_len,
                        query_local[index_in_batch + 1],
                    )
                ),
                need_warmup_len,
                req_id,
            ]
        )

    def _collect_gsa_tokens(
        self,
        req_id,
        req_meta,
        index_in_batch,
        query_local,
        req_cnt,
        use_tokens_ptrs,
        seq_lens,
        swap_index,
        use_req_ids,
    ) -> int:
        if not (req_meta.is_gsa() and req_meta.is_gsa_init):
            return req_cnt
        if self.is_prefetch_token:
            tmp = req_meta.use_tokens
            req_meta.use_tokens = req_meta.prefetch_use_token
            req_meta.prefetch_use_token = tmp
            self.prefetch_engine.prefetch_engine_c.set_topk_indices(
                req_id, req_meta.prefetch_use_token.cpu
            )
            prefetch_index_list = list(
                range(query_local[index_in_batch], query_local[index_in_batch + 1])
            )
            self.prefetch_topk.append([prefetch_index_list, req_id])
        use_tokens_ptrs.append(req_meta.use_tokens.ptr)
        seq_lens.append(req_meta.get_spase_len())
        swap_index.append(
            [len(self.swap_index), len(self.swap_index) + self.num_spec_tokens]
        )
        use_req_ids.append(req_id)
        self.select_block_index.np[req_cnt] = index_in_batch
        req_cnt += 1
        for i in range(query_local[index_in_batch], query_local[index_in_batch + 1]):
            self.swap_index.append(i)
        return req_cnt

    def _pre_swap_tokens_all(self, use_tokens_ptrs, seq_lens, swap_index, use_req_ids):
        if not use_tokens_ptrs:
            return
        if self.gsa_swap_topk_cpu_ptr is None:
            self.gsa_swap_topk_cpu_ptr = self.gsa_swap_topk.cpu.data_ptr()
            self.gsa_swap_topk_npu_ptr = self.gsa_swap_topk.npu.data_ptr()
            self.log.info(
                f"[GSA] gsa_swap_topk_cpu_ptr {self.gsa_swap_topk_cpu_ptr}"
                f"gsa_swap_topk_npu_ptr {self.gsa_swap_topk_npu_ptr}\n"
            )
        self.prefetch_engine.prefetch_engine_c.pre_swap_tokens(
            use_tokens_ptrs,
            seq_lens,
            swap_index,
            [self.gsa_swap_topk_cpu_ptr, self.gsa_swap_topk_npu_ptr],
            use_req_ids,
            SPA_TOKENS_LEN,
            self.gsa_swap_topk_bs,
            self.num_spec_tokens,
        )

    def _copy_gsa_buffers_to_npu(self, req_cnt) -> tuple[int, int, int]:
        if len(self.swap_index) != 0:
            self.gsa_swap_topk.copy_to_npu(-1)
        self.gsa_swap_index.np[: len(self.swap_index)] = [x for x in self.swap_index]
        index_flattened = [x for sublist in self.prefetch_topk for x in sublist[0]]
        self.gsa_prefetch_index.np[: len(index_flattened)] = index_flattened
        warmup_index = []
        for info in self.warmup_topk:
            warmup_index += info[0]
        self.gsa_warmup_index.np[: len(warmup_index)] = warmup_index
        self.select_block_index.copy_to_npu(req_cnt)
        self.gsa_swap_index.np[len(self.swap_index) :] = self.num_padding_reqs
        self.gsa_swap_index.copy_to_npu(-1)
        self.gsa_prefetch_index.copy_to_npu(self.num_padding_reqs)
        self.gsa_warmup_index.copy_to_npu(len(warmup_index))
        indexer_len = 0
        if IS_INDEXER_OFFLOAD:
            flattened = [
                x for sublist in self.indexer_last_q_index_tmp for x in sublist
            ]
            self.indexer_last_q_index.np[: len(flattened)] = flattened
            self.indexer_last_q_index.copy_to_npu(self.num_padding_reqs)
            indexer_len = len(flattened)
        return len(index_flattened), len(warmup_index), indexer_len

    def _update_attn_gsa_lens(
        self,
        attn_metadata,
        gsa_slot_len,
        gsa_prefetch_len,
        gsa_warmup_len,
        indexer_len,
        req_cnt,
    ):
        gsa_swap_len = len(self.swap_index)
        for attn in attn_metadata.values():
            attn.gsa_prefetch_len = gsa_prefetch_len
            attn.gsa_swap_len = gsa_swap_len
            attn.gsa_warmup_len = gsa_warmup_len
            attn.gsa_slot_len = gsa_slot_len
            if IS_INDEXER_OFFLOAD:
                attn.indexer_last_query_len = indexer_len
            attn.select_block_index = self.select_block_index.npu[:req_cnt]
        self.log.info(
            f"[GSA] gsa_prefetch_len {gsa_prefetch_len} gsa_swap_len {gsa_swap_len}"
            f"gsa_warmup_len {gsa_warmup_len} attn.gsa_slot_len {gsa_slot_len}"
        )

    def get_model_input_all(self, attn_metadata, gsa_slot_len):
        if IS_PERFORMANCE_LOG:
            time_get_status = 0
            time_update_tokens = 0
            time_reset_np = 0
            time_copy_to_npu = 0
            prefetch_req_pre = 0
            start = time.perf_counter()
        self._update_prefetch_status_all()
        if IS_PERFORMANCE_LOG:
            time_get_status = (time.perf_counter() - start) * 1_000_000
            start = time.perf_counter()
        self._reset_gsa_buffers_all()
        self._init_indexer_offload_all()
        if IS_PERFORMANCE_LOG:
            time_reset_np = (time.perf_counter() - start) * 1_000_000
        req_cnt, prefetch_req_pre, time_update_tokens = (
            self._collect_and_swap_tokens_all()
        )
        if IS_PERFORMANCE_LOG:
            start = time.perf_counter()
        gsa_prefetch_len, gsa_warmup_len, indexer_len = self._copy_gsa_buffers_to_npu(
            req_cnt
        )
        self._update_attn_gsa_lens(
            attn_metadata,
            gsa_slot_len,
            gsa_prefetch_len,
            gsa_warmup_len,
            indexer_len,
            req_cnt,
        )
        if IS_PERFORMANCE_LOG:
            time_copy_to_npu = (time.perf_counter() - start) * 1_000_000
            self.log.info(
                f"[GSA Time] get_model_input_all for rank {self.rank} is_prefetch {self.is_prefetch_token}"
                f"time_get_status {time_get_status} time_update_tokens {time_update_tokens}"
                f"time_reset_np {time_reset_np} time_copy_to_npu {time_copy_to_npu}"
                f"prefetch_req_pre {prefetch_req_pre}"
            )

    # @time_us
    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        phase: Optional[str] = None,
        has_indexer: Optional[bool] = True,
        use_li_c8: Optional[bool] = False,
    ) -> None:
        if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node:
            return
        layer_id = int(layer_name.split(".")[2])
        if layer_id >= self.layer_num:
            return

    # @time_us
    def last_chunk_init(self, req_meta, layer_id, req_id):
        self.prefetch_engine.prefetch_engine_c.set_tokens_map(
            layer_id,
            req_id,
            req_meta.topk_buf_tmp[self.topk_layer_map[layer_id]],
            req_meta.blocks[:-1],
            req_meta.num_prompt_tokens,
            req_meta.slab_host_slot,
            self.topk_layer_info[layer_id] == layer_id,
        )

    # @time_us
    def build_gsa_metadata(
        self, scheduler_output: SchedulerOutput, requests, input_batch
    ) -> GSAMetaData:
        gsa_meta = GSAMetaData(self._vllm_config, self.topk_layer_num)
        gsa_meta.gsa_stats = self.gsa_stats
        for req_id in scheduler_output.preempted_req_ids:
            self._release_preempted_gsa_stats(req_id)
            while self.prefetch_engine.prefetch_engine_c.get_prefetch_status():
                time.sleep(0.001)
            if IS_INDEXER_OFFLOAD:
                while self.prefetch_engine.prefetch_engine_c.get_indexer_status():
                    time.sleep(0.001)
            self.prefetch_engine.del_finish_meta(req_id)
            self._block_hash_injected_reqs.discard(req_id)
            self.preempt_req_output_tokens[req_id] = len(
                requests[req_id].output_token_ids
            )
        self.model_input = gsa_meta.get_model_input(
            scheduler_output,
            self.prefetch_engine.max_block_len,
            requests,
            input_batch,
            self.prefetch_engine,
            self.preempt_req_output_tokens,
        )
        self.gsa_stats = gsa_meta.gsa_stats
        return gsa_meta

    def _release_preempted_gsa_stats(self, req_id):
        if req_id not in self.gsa_stats:
            return
        stats = self.gsa_stats.pop(req_id)
        if self.rank == 0:
            self.log.info(f"[GSA] preempted_req_ids {req_id}")
            if IS_INDEXER_OFFLOAD:
                self.gsa_cpu_shared_pool.free(stats.slab_host_slot)
            self.indexer_cpu_shared_pool.free(stats.indexer_slab_host_slot)
        self.gsa_preempt_shared_pool.free(stats.req_dump_blocks)
        self.update_slot_num()

    # @time_us
    def execute_begin(self, scheduler_output: SchedulerOutput, attn_metadata):
        """每步推理前的 GSA 预取准备：注入 store、处理请求、同步块表、构造输入"""
        if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node:
            return
        if IS_PERFORMANCE_LOG:
            start_total = time.perf_counter()
        self._ensure_store_injected()
        self.prefetch_engine.step_time += 1
        batch_size = len(scheduler_output.num_scheduled_tokens.items())
        req_ids = [0] * batch_size
        add_blocks = [None] * batch_size
        new_blocks_len = [None] * batch_size
        indexer_add_blocks = [None] * batch_size
        indexer_new_blocks_len = [None] * batch_size
        for req_id, _ in scheduler_output.num_scheduled_tokens.items():
            self._process_scheduled_req(
                req_id,
                req_ids,
                add_blocks,
                new_blocks_len,
                indexer_add_blocks,
                indexer_new_blocks_len,
            )
        self._sync_block_tables(
            req_ids,
            add_blocks,
            new_blocks_len,
            indexer_add_blocks,
            indexer_new_blocks_len,
        )
        self.req_ids_bs = req_ids
        self.gsa_stats = self.gsa_metadata.gsa_stats
        attn = next(iter(attn_metadata.values()))
        self.num_padding_reqs = attn.num_input_tokens
        gsa_slot_len = self.get_dsa_input()
        self.get_model_input_all(attn_metadata, gsa_slot_len)
        if IS_PERFORMANCE_LOG:
            time_total = (time.perf_counter() - start_total) * 1_000_000
            self.log.info(f"[GSA Time] execute_begin time_total {time_total}")

    def _process_scheduled_req(
        self,
        req_id: int,
        req_ids: list,
        add_blocks: list,
        new_blocks_len: list,
        indexer_add_blocks: list,
        indexer_new_blocks_len: list,
    ):
        """处理单个调度请求：GSA 初始化、索引登记、块长计算与共享内存分配"""
        req_meta = self.gsa_metadata.gsa_stats[req_id]
        if self.get_num_output_tokens(req_id) == 1 and req_meta.is_gsa():
            self.prefetch_engine.prefetch_engine_c.set_topk_indices(
                req_id, req_meta.prefetch_use_token.cpu
            )
            for layer_id in range(self.layer_num):
                self.last_chunk_init(req_meta, layer_id, req_id)
            req_meta.use_tokens.np[:] = req_meta.prefetch_use_token.cpu
            req_meta.last_step = self.prefetch_engine.step_time - 1
            req_meta.prefetch_topk_step = self.prefetch_engine.step_time - 1
            req_meta.is_gsa_init = True
        req_in_batch = req_meta.index_in_batch
        req_ids[req_in_batch] = req_id
        if IS_INDEXER_OFFLOAD and req_meta.num_prompt_tokens > SEG_PREFILL_THRESHOLD:
            block_len = math.ceil(req_meta.get_seq_len() / self.block_size) - len(
                req_meta.slab_host_slot
            )
        else:
            block_len = math.ceil(req_meta.get_seq_len() / self.block_size) - len(
                req_meta.indexer_slab_host_slot
            )
        add_blocks[req_in_batch] = []
        new_blocks_len[req_in_batch] = 0
        indexer_add_blocks[req_in_batch] = []
        indexer_new_blocks_len[req_in_batch] = 0
        if block_len != 0:
            if (
                IS_INDEXER_OFFLOAD
                and req_meta.num_prompt_tokens > SEG_PREFILL_THRESHOLD
            ):
                if self.rank == 0:
                    req_meta.new_block_id = self.gsa_cpu_shared_pool.allocate(block_len)
                    req_meta.slab_host_slot += req_meta.new_block_id
                    add_blocks[req_in_batch] = [x for x in req_meta.new_block_id]
                new_blocks_len[req_in_batch] = block_len
                req_meta.new_block_id = [1]
            if self.rank == 0:
                new_indexer_id = self.indexer_cpu_shared_pool.allocate(block_len)
                req_meta.indexer_slab_host_slot += new_indexer_id
                indexer_add_blocks[req_in_batch] = [x for x in new_indexer_id]
            indexer_new_blocks_len[req_in_batch] = block_len
        req_meta.add_blocks_len = block_len

    def _sync_block_tables(
        self,
        req_ids: list,
        add_blocks: list,
        new_blocks_len: list,
        indexer_add_blocks: list,
        indexer_new_blocks_len: list,
    ):
        """共享内存块表双向同步：rank0 写入，其余 rank 读取并应用到请求元数据"""
        if IS_INDEXER_OFFLOAD:
            if self.rank == 0:
                self.block_table_sm.write_add_block(add_blocks)
            else:
                add_blocks = self.block_table_sm.read_add_block(new_blocks_len)
                if add_blocks is not None:
                    self._apply_add_blocks(
                        req_ids, add_blocks, new_blocks_len, is_indexer=False
                    )
        if self.rank == 0:
            self.indexer_block_table_sm.write_add_block(indexer_add_blocks)
        else:
            indexer_blocks = self.indexer_block_table_sm.read_add_block(
                indexer_new_blocks_len
            )
            if indexer_blocks is not None:
                self._apply_add_blocks(
                    req_ids, indexer_blocks, indexer_new_blocks_len, is_indexer=True
                )

    def _apply_add_blocks(self, req_ids, add_blocks, new_blocks_len, is_indexer):
        for index_in_batch, req_id in enumerate(req_ids):
            if new_blocks_len[index_in_batch] == 0:
                continue
            req_meta = self.gsa_metadata.gsa_stats[req_id]
            if is_indexer:
                req_meta.indexer_slab_host_slot += add_blocks[index_in_batch]
            else:
                req_meta.slab_host_slot += add_blocks[index_in_batch]

    def init_dsa_data(self, model_runner):
        model_runner.gsa_swap_topk = self.gsa_swap_topk
        model_runner.gsa_prefetch_topk = self.gsa_prefetch_topk
        model_runner.gsa_warmup_topk = self.gsa_warmup_topk

        model_runner.gsa_swap_index = self.gsa_swap_index
        model_runner.gsa_prefetch_index = self.gsa_prefetch_index
        model_runner.gsa_warmup_index = self.gsa_warmup_index

        model_runner.dsa_slot_mapping = self.dsa_slot_mapping
        model_runner.dsa_block_table = self.dsa_block_table
        model_runner.dsa_seq_len = self.dsa_seq_len
        model_runner.gsa_prefill_query = self.gsa_prefill_query
        model_runner.gsa_topk_win_tmp = self.gsa_topk_win_tmp

        if IS_INDEXER_OFFLOAD:
            model_runner.indexer_last_query = self.indexer_last_query
            model_runner.indexer_last_q_index = self.indexer_last_q_index
            model_runner.indexer_last_weight = self.indexer_last_weight

        if self.log.is_info_enabled:
            self.log.info(
                f"[GSA] init dsa data successfully!! gsa_swap_index id {id(model_runner.gsa_swap_topk.npu)}\n"
            )

    def cal_indexer_k_repr_mean_prefill(self):
        if not IS_INDEXER_OFFLOAD:
            return
        req_ids_bs = self.need_cal_k_mean
        indexer_new_block = [None] * len(req_ids_bs)
        indexer_new_block_len = [None] * len(req_ids_bs)
        for index, req_id in enumerate(req_ids_bs):
            req_meta = self.gsa_metadata.gsa_stats[req_id]
            indexer_new_block[index] = []
            indexer_new_block_len[index] = 0
            self._cal_one_req_indexer_mean(
                req_meta, index, indexer_new_block, indexer_new_block_len
            )
        if self.rank == 0:
            self.indexer_block_table_sm.write_add_block(indexer_new_block)
        else:
            add_blocks = self.indexer_block_table_sm.read_add_block(
                indexer_new_block_len
            )
            self._apply_indexer_add_blocks(
                req_ids_bs, add_blocks, indexer_new_block_len
            )

    def _cal_one_req_indexer_mean(
        self, req_meta, index, indexer_new_block, indexer_new_block_len
    ):
        if not self.get_num_output_tokens(req_id) == 0:
            return
        for layer_id in range(self.layer_num):
            self.indexer_k_repr[layer_id][req_meta.slab_host_slot] = torch.mean(
                self.indexer_k_cache_sparse[layer_id][req_meta.indexer_slab_host_slot],
                dim=1,
                keepdim=True,
            )
        if self.rank == 0:
            self.indexer_cpu_shared_pool.free(req_meta.indexer_slab_host_slot)
            req_meta.indexer_slab_host_slot = self.indexer_cpu_shared_pool.allocate(
                math.ceil(req_meta.prompt_sparse_len * INDEXER_PREFETCH_RATIO) + 1
            )
            indexer_new_block[index] = [x for x in req_meta.indexer_slab_host_slot]
        indexer_new_block_len[index] = (
            math.ceil(req_meta.prompt_sparse_len * INDEXER_PREFETCH_RATIO) + 1
        )

    def _apply_indexer_add_blocks(self, req_ids_bs, add_blocks, indexer_new_block_len):
        if add_blocks is None:
            return
        for index, req_id in enumerate(req_ids_bs):
            if indexer_new_block_len[index] == 0:
                continue
            self.gsa_metadata.gsa_stats[req_id].indexer_slab_host_slot = [
                x for x in add_blocks[index]
            ]

    def cal_indexer_k_repr_mean_decode(self, rec_result=None) -> None:
        if not IS_INDEXER_OFFLOAD:
            return
        include_masks = []
        exclude_masks = []
        repre_slot_mappings = []

        req_ids_bs = self.need_cal_k_mean

        self.max_topk_len = []
        for req_id in self.need_indexer_sparse_reqs:
            req_meta = self.gsa_metadata.gsa_stats[req_id]
            include_masks.append(req_meta.include_mask)
            exclude_masks.append(req_meta.exclude_mask)
            repre_slot_mappings.append(
                req_meta.slab_host_slot[: req_meta.prompt_block_len]
            )
            if self.get_num_output_tokens(req_id) == 0:
                self.max_topk_len.append(
                    math.ceil(req_meta.prompt_sparse_len * INDEXER_PREFETCH_RATIO)
                )
            else:
                self.max_topk_len.append(req_meta.prompt_sparse_len)

        self.include_masks = make_tensor_with_pad(
            include_masks, pad=False, dtype=torch.uint8, device=self.device
        )
        self.exclude_masks = make_tensor_with_pad(
            exclude_masks, pad=True, dtype=torch.uint8, device=self.device
        )
        self.repre_slot_mappings = make_tensor_with_pad(
            repre_slot_mappings, pad=0, dtype=torch.int32, device=self.device
        )

        if self.log.is_info_enabled:
            self.log.info(
                f"[GSA] step {self.prefetch_engine.step_time} sparse len all {self.max_topk_len}\n"
            )

    # @time_us
    def cal_indexer_topk_blocks(self, rec_result=None):
        if not IS_INDEXER_OFFLOAD:
            return
        hidden_size = 128
        bs = len(self.need_indexer_sparse_bs)
        if bs == 0:
            return
        start = time.perf_counter()
        if self.num_spec_tokens != 1 and rec_result is not None:
            last_q_index_fb = list(accumulate(self.indexer_last_q_index_fb))
            result = [rec_result[i] for i in self.need_indexer_sparse_bs]
            index_list = []
            result_num = len(result)
            for i in range(result_num):
                index_list.append(last_q_index_fb[i] + result[i])
        else:
            index_list = list(range(len(self.need_indexer_sparse_bs)))

        indexer_key_repr = self.indexer_k_repr  # layer, t_b_n, 1, 1, 128
        indexer_q = self.indexer_last_query.npu[:, index_list]  # layer, bs, 64, 128
        indexer_w = self.indexer_last_weight.npu[:, index_list]  # layer, bs, 64
        curr_key_repr = indexer_key_repr[:, self.repre_slot_mappings.flatten()].view(
            self.layer_num, bs, -1, hidden_size
        )  # layer, bs, max_blk_num, 128
        indexer_q_k = torch.einsum(
            "lbik,lbjk->lbij", indexer_q, curr_key_repr
        )  # layer, bs, 64, max_blk_num
        relu_indexer_q_k = torch.relu(indexer_q_k)
        score_blks_indexer = torch.einsum(
            "lbi,lbij->lbj", indexer_w, relu_indexer_q_k
        )  # layer, bs, max_blk_num

        score_blks_indexer.masked_fill_(self.include_masks == 1, float("inf"))
        score_blks_indexer.masked_fill_(self.exclude_masks == 1, float("-inf"))
        _, topk_indices = torch.topk(
            score_blks_indexer, k=max(self.max_topk_len), dim=-1
        )  # layer indices: (bs, k)

        sorted_topk_indices_cpu = topk_indices.cpu()
        for index, req_id in enumerate(self.need_indexer_sparse_reqs):  # todo顺序
            req_meta = self.gsa_metadata.gsa_stats[req_id]
            req_meta.indexer_topk_blocks[:, : self.max_topk_len[index]] = (
                sorted_topk_indices_cpu[:, index, : self.max_topk_len[index]]
            )
        elapsed = (time.perf_counter() - start) * 1_000_000
        self.log.performance_info(f"[GSA] cal_indexer_topk_blocks cost: {elapsed}\n")

    def set_indexer_topk_blocks_prefill(self):
        if not IS_INDEXER_OFFLOAD:
            return
        if len(self.need_indexer_sparse_reqs) == 0:
            return
        start = time.perf_counter()
        for req_id in self.need_indexer_sparse_reqs:
            req_meta = self.gsa_metadata.gsa_stats[req_id]
            if self.get_num_output_tokens(req_id) == 0:
                remain_idx = req_meta.indexer_topk_blocks[
                    :, : req_meta.prompt_sparse_len
                ].tolist()
                remain_idx = [sorted(x) for x in remain_idx]
                prefetch_idx = req_meta.indexer_topk_blocks[
                    :,
                    req_meta.prompt_sparse_len : math.ceil(
                        req_meta.prompt_sparse_len * INDEXER_PREFETCH_RATIO
                    ),
                ].tolist()
                req_meta.indexer_device_slot_status = [
                    prefetch_idx[x] + remain_idx[x] for x in range(self.layer_num)
                ]
                self.prefetch_engine.prefetch_engine_c.set_indexer_map(
                    req_id,
                    req_meta.indexer_slab_host_slot,
                    remain_idx,
                    prefetch_idx,
                    req_meta.slab_host_slot,
                    req_meta.prefetch_indexer_block_tables_cpu,
                )
                self.prefetch_engine.prefetch_engine_c.indexer_init_load(
                    [req_meta.indexer_slab_host_slot[:-1]] * self.layer_num,
                    req_meta.indexer_device_slot_status,
                    req_id,
                )
                req_meta.indexer_block_tables_cpu.copy_(
                    req_meta.prefetch_indexer_block_tables_cpu
                )
                req_meta.is_indexer_init = True
        elapsed = (time.perf_counter() - start) * 1_000_000
        self.log.performance_info(
            f"[GSA] set_indexer_topk_blocks_prefill cost: {elapsed}\n"
        )

    # @time_us
    def set_indexer_topk_blocks_decode(self):
        if not IS_INDEXER_OFFLOAD:
            return
        if len(self.need_indexer_sparse_reqs) == 0:
            return
        start = time.perf_counter()
        topk_lens_input = []
        req_id_list = []
        topk_lists = []
        swap_block_table = []
        add_blocks = []

        is_indexer_load = self.prefetch_engine.prefetch_engine_c.get_indexer_status()

        for req_id in self.need_indexer_sparse_reqs:
            req_meta = self.gsa_metadata.gsa_stats[req_id]
            if self.get_num_output_tokens(req_id) == 0:
                pass
            elif not is_indexer_load:
                topk_lens_input.append(req_meta.prompt_sparse_len)
                req_id_list.append(req_id)
                topk_lists.append(
                    req_meta.indexer_topk_blocks_cpu[:, : req_meta.prompt_sparse_len]
                )
                req_meta.indexer_block_tables[:, : req_meta.prompt_sparse_len] = (
                    req_meta.prefetch_indexer_block_tables_cpu[
                        :, : req_meta.prompt_sparse_len
                    ]
                )
        if len(req_id_list) == 0:
            return

        self.prefetch_engine.prefetch_engine_c.submit_task_indexer(
            topk_lens_input, req_id_list, topk_lists, self.prefetch_engine.step_time
        )
        elapsed = (time.perf_counter() - start) * 1_000_000
        self.log.performance_info(
            f"[GSA] set_indexer_topk_blocks_decode cost: {elapsed}\n"
        )

    def update_slot_num(self):
        if self.rank == 0:
            if IS_INDEXER_OFFLOAD:
                k_num = self.gsa_cpu_shared_pool.get_free_num()
            else:
                k_num = 0
            i_num = self.indexer_cpu_shared_pool.get_free_num()
            self.slots_num_manager.update_num(i_num, k_num)

    def _warmup_dump_blocks(self):
        if IS_PERFORMANCE_LOG:
            start = time.perf_counter()
        start_offset = 0
        for _, warmup_info in enumerate(self.warmup_topk):
            req_id = warmup_info[2]
            req_meta = self.gsa_metadata.gsa_stats[req_id]
            req_meta.topk_buf_tmp = self.gsa_warmup_topk.cpu[
                :, start_offset : start_offset + warmup_info[1]
            ]
            start_offset += warmup_info[1]
            vllm_blocks = self._get_vllm_blocks(req_meta)
            req_meta.req_dump_blocks = self.gsa_preempt_shared_pool.allocate(
                req_meta.non_ucm_block_count
            )
            preempt_kv_block = vllm_blocks[-1 * req_meta.non_ucm_block_count :]
            preempt_indexer_block = req_meta.indexer_slab_host_slot[
                -1 * req_meta.non_ucm_block_count :
            ]
            self._copy_preempt_kv_blocks(
                req_meta, preempt_kv_block, preempt_indexer_block
            )
            first_non_ucm_block = req_meta.original_prompt_block_len
            non_ucm_block_idxs = list(
                range(
                    first_non_ucm_block,
                    first_non_ucm_block + req_meta.non_ucm_block_count,
                )
            )
            self._inject_block_hashes(
                req_id,
                non_ucm_block_idxs,
                req_meta.req_dump_blocks,
                inject_indexer=(req_meta.num_prompt_tokens > INDEXER_PREFETCH_TOKENS),
            )
        if IS_PERFORMANCE_LOG:
            return (time.perf_counter() - start) * 1_000_000
        return 0

    def _get_vllm_blocks(self, req_meta):
        if IS_INDEXER_OFFLOAD:
            slot_len = len(req_meta.slab_host_slot)
        else:
            slot_len = len(req_meta.indexer_slab_host_slot)
        if self.num_spec_tokens != 1 and len(req_meta.blocks) > slot_len:
            return req_meta.blocks[:-1]
        return req_meta.blocks

    def _copy_preempt_kv_blocks(
        self, req_meta, preempt_kv_block, preempt_indexer_block
    ):
        for layer_id in range(self.layer_num):
            for block_count in range(req_meta.non_ucm_block_count):
                dump_block = req_meta.req_dump_blocks[block_count]
                self._slab_preempt_k[dump_block][layer_id].copy_(
                    self.kv_caches[layer_id][0][preempt_kv_block[block_count]]
                )
                if not self.enable_sparse_sfa_c8:
                    self._slab_preempt_v[dump_block][layer_id].copy_(
                        self.kv_caches[layer_id][1][preempt_kv_block[block_count]]
                    )
                if IS_INDEXER_OFFLOAD and preempt_indexer_block is not None:
                    self.indexer_cache_preempt[dump_block][layer_id].copy_(
                        self.indexer_k_cache_sparse[layer_id][
                            preempt_indexer_block[block_count]
                        ]
                    )

    def warmup(self):
        if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node:
            return
        if len(self.warmup_topk) == 0:
            return
        if IS_PERFORMANCE_LOG:
            time_warm_up_copy = 0
            time_total = 0
            start_total = time.perf_counter()
            start = time.perf_counter()
        torch.npu.synchronize()
        warmup_topk_len = 0
        for info in self.warmup_topk:
            warmup_topk_len += info[1]
        self.gsa_warmup_topk.copy_to_cpu(-1)
        if IS_PERFORMANCE_LOG:
            time_warm_up_copy = (time.perf_counter() - start) * 1_000_000
        time_warm_up_dump = self._warmup_dump_blocks()
        self.cal_indexer_k_repr_mean_prefill()
        if IS_PERFORMANCE_LOG:
            time_total = (time.perf_counter() - start_total) * 1_000_000
            self.log.info(
                f"[GSA Time] warmup for rank {self.rank} time_total {time_total}"
                f"time_warm_up_copy {time_warm_up_copy} time_warm_up_dump {time_warm_up_dump}"
            )

    # @time_us
    def execute_finished(self, rec_result=None):
        """rec_result 代表当前每个 req 接受了多少个 token"""
        if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node:
            return
        if IS_PERFORMANCE_LOG:
            time_prepare = 0
            time_submit = 0
            time_deal_indexer = 0
            start = time.perf_counter()
            start_total = time.perf_counter()
        if len(self.prefetch_topk) == 0 and len(self.warmup_topk) == 0:
            self.cal_indexer_k_repr_mean_decode(rec_result)
            self.cal_indexer_topk_blocks(rec_result)
            self.set_indexer_topk_blocks_prefill()
            self.set_indexer_topk_blocks_decode()
            self.update_slot_num()
            if IS_PERFORMANCE_LOG and self.rank == 0:
                elapsed = (time.perf_counter() - start) * 1_000_000
                self.log.info(f"[GSA] no prefetch time {elapsed}")
            return
        if self.is_prefetch_token:
            if IS_PERFORMANCE_LOG:
                start = time.perf_counter()
            topk_len_list, req_id_list, is_index_offload, req_seq_len = (
                self._prepare_prefetch_requests(rec_result)
            )
            if IS_PERFORMANCE_LOG:
                time_prepare = (time.perf_counter() - start) * 1_000_000
                start = time.perf_counter()
            self._submit_prefetch_task(
                topk_len_list, req_id_list, is_index_offload, req_seq_len
            )
            if IS_PERFORMANCE_LOG:
                time_submit = (time.perf_counter() - start) * 1_000_000
        if IS_PERFORMANCE_LOG:
            start = time.perf_counter()
        self.cal_indexer_k_repr_mean_decode(rec_result)
        self.cal_indexer_topk_blocks(rec_result)
        self.set_indexer_topk_blocks_prefill()
        self.set_indexer_topk_blocks_decode()
        self.update_slot_num()
        if self.rank == 0:
            elapsed = (time.perf_counter() - start) * 1_000_000
        if IS_PERFORMANCE_LOG:
            time_deal_indexer = (time.perf_counter() - start) * 1_000_000
            time_total = (time.perf_counter() - start_total) * 1_000_000
            self.log.info(
                f"[GSA Time] execute_finished for rank {self.rank} "
                f"time_total {time_total} time_prepare {time_prepare} "
                f"time_submit {time_submit} time_deal_indexer {time_deal_indexer}"
            )

    def _prepare_prefetch_requests(self, rec_result):
        """拷回预取结果，并准备每个请求的序列长度/索引/offload 标记"""
        self.gsa_prefetch_topk.copy_to_cpu(-1)
        topk_len_list = [self.topk_indices_len] * len(self.prefetch_topk)
        req_id_list = [req_prefetch_topk[1] for req_prefetch_topk in self.prefetch_topk]
        is_index_offload = [False] * len(self.prefetch_topk)
        req_seq_len = [None] * len(self.prefetch_topk)
        index_list = []
        for index, req_id in enumerate(req_id_list):
            req_meta = self.gsa_metadata.gsa_stats[req_id]
            index_in_batch = req_meta.index_in_batch
            req_meta.prefetch_topk_step = self.prefetch_engine.step_time
            if self.num_spec_tokens != 1 and rec_result is not None:
                req_seq_len[index] = (
                    req_meta.get_seq_len()
                    - self.num_spec_tokens
                    + rec_result[index_in_batch]
                    + 1
                )
                index_list.append(
                    index * self.num_spec_tokens + rec_result[index_in_batch]
                )
                self.select_gsa_prefetch_topk[:, index].copy_(
                    self.gsa_prefetch_topk.cpu[
                        :, index * self.num_spec_tokens + rec_result[index_in_batch]
                    ]
                )
            else:
                req_seq_len[index] = req_meta.get_seq_len()
                index_list.append(index)
            if IS_INDEXER_OFFLOAD:
                is_index_offload[index] = req_meta.is_indexer_init
        return topk_len_list, req_id_list, is_index_offload, req_seq_len

    def _submit_prefetch_task(
        self, topk_len_list, req_id_list, is_index_offload, req_seq_len
    ):
        if sum(topk_len_list) == 0:
            return
        self.prefetch_space_token = 0
        if self.num_spec_tokens != 1:
            self.prefetch_engine.prefetch_engine_c.submit_prefetch_task(
                topk_len_list,
                list(range(len(topk_len_list))),
                req_id_list,
                self.select_gsa_prefetch_topk,
                is_index_offload,
                req_seq_len,
                self.prefetch_engine.step_time,
            )
        else:
            self.prefetch_engine.prefetch_engine_c.submit_prefetch_task(
                topk_len_list,
                list(range(len(topk_len_list))),
                req_id_list,
                self.gsa_prefetch_topk.cpu,
                is_index_offload,
                req_seq_len,
                self.prefetch_engine.step_time,
            )

    def build_sparse_meta(
        self, scheduler_output: SchedulerOutput, requests, input_batch, attn_metadata
    ) -> None:
        if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node:
            return
        self.gsa_metadata = self.build_gsa_metadata(
            scheduler_output, requests, input_batch
        )
        req_ids = list(getattr(input_batch, "req_ids", []))
        self.decode_index = [
            self.gsa_metadata.gsa_stats[rid].index_in_batch
            for rid in req_ids
            if self.gsa_metadata.gsa_stats[rid].stage() == SequenceStage.DECODE
        ]
        self.gsa_index = [
            self.gsa_metadata.gsa_stats[rid].index_in_batch
            for rid in req_ids
            if self.gsa_metadata.gsa_stats[rid].is_gsa()
        ]
        stats = self.gsa_metadata.gsa_stats
        num_output_one = [
            rid for rid in req_ids if self.get_num_output_tokens(rid) == 0
        ]
        self.need_prefetch_index = [
            stats[rid].index_in_batch
            for rid in num_output_one
            if stats[rid].num_prompt_tokens > SEG_PREFILL_THRESHOLD
        ]
        self.is_all_gsa = len(self.gsa_index) == len(req_ids)
        if len(self.decode_index) == len(req_ids):
            self.has_prefill = False
        else:
            self.has_prefill = True

    def request_begin(self, request_id: ReqType, prompt_token_ids: List[int]):
        pass

    def request_finished_in_scheduler(self, request_id: ReqType):
        if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node:
            return
        if request_id in self.preempt_req_output_tokens:
            del self.preempt_req_output_tokens[request_id]
        pass

    def request_finished_in_worker(self, request_id: ReqType):
        if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node:
            return
        while self.prefetch_engine.prefetch_engine_c.get_prefetch_status():
            time.sleep(0.001)
        if IS_INDEXER_OFFLOAD:
            while self.prefetch_engine.prefetch_engine_c.get_indexer_status():
                time.sleep(0.001)
        if request_id in self.gsa_stats:
            stats = self.gsa_stats.pop(request_id)
            if self.rank == 0:
                if IS_INDEXER_OFFLOAD and stats.slab_host_slot:
                    self.gsa_cpu_shared_pool.free(stats.slab_host_slot)
                if stats.indexer_slab_host_slot:
                    self.indexer_cpu_shared_pool.free(stats.indexer_slab_host_slot)
            self.gsa_preempt_shared_pool.free(stats.req_dump_blocks)
            self.update_slot_num()
        self.preempt_req_output_tokens.pop(request_id, None)
        self.prefetch_engine.del_finish_meta(request_id)
        self._block_hash_injected_reqs.discard(request_id)

    def update_state_after_alloc(self, request: Request, num_blocks: int):
        pass

    def begin_sch_flag(self):
        self.alloc_block_sch = 0

    def update_scheduler_free_slot(self, num):
        if num == 0:
            self.alloc_block_sch = 0
        else:
            self.alloc_block_sch += num

    def estimate_free_slots_sparse(
        self, request: Request, num_blocks_to_allocate, num_new_computed_tokens=0
    ):
        if not self.is_decode_node:
            return False
        num_blocks_to_allocate_total = (
            num_blocks_to_allocate + num_new_computed_tokens // self.block_size
        )
        free_i_num, free_k_num = self.slots_num_manager.read_num()
        free_i_num -= self.alloc_block_sch
        free_k_num -= self.alloc_block_sch
        gsa_preempted_flag = False
        if (
            IS_INDEXER_OFFLOAD
            and len(request.prompt_token_ids) > SEG_PREFILL_THRESHOLD
            and (free_k_num - num_blocks_to_allocate_total) < 20
        ):
            gsa_preempted_flag = True
        if free_i_num - num_blocks_to_allocate_total < 20:
            gsa_preempted_flag = True
        if gsa_preempted_flag:
            pass
        else:
            self.alloc_block_sch += num_blocks_to_allocate_total
        return gsa_preempted_flag

    def estimate_num_slots_sparsed(self, request: Request, num_spec_slots=0) -> int:
        if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node:
            return INVALID_SLOT
        if request.status == RequestStatus.PREEMPTED:
            self.preempt_req_output_tokens[request.request_id] = (
                request.num_output_tokens
            )

        if request.request_id in self.preempt_req_output_tokens:
            num_output_tokens = (
                request.num_output_tokens
                - self.preempt_req_output_tokens[request.request_id]
            )
        else:
            num_output_tokens = request.num_output_tokens

        if num_output_tokens == 0 or request.num_prompt_tokens <= SEG_PREFILL_THRESHOLD:
            return INVALID_SLOT
        output_len = num_output_tokens
        num_tokens_sparsed = output_len + GSA_SPARSE_LEN + num_spec_slots
        return num_tokens_sparsed

    def allocate_slots(self, kv_cache_manager, request, num_slots_sparsed):
        if not PTOPK_PREFETCH_ENABLE or not self.is_decode_node:
            return None
        coordinator = kv_cache_manager.coordinator
        block_pool = kv_cache_manager.block_pool
        kv_cache_groups = kv_cache_manager.kv_cache_config.kv_cache_groups

        if request.request_id in self.preempt_req_output_tokens:
            # handle preempt: get the TRUE output_len
            num_output_tokens = (
                request.num_output_tokens
                - self.preempt_req_output_tokens[request.request_id]
            )
        else:
            num_output_tokens = request.num_output_tokens

        if num_output_tokens == 1:
            kv_cache_manager.free(request)
            request.cache_salt = "gsa" + uuid.uuid4().hex

        new_computed_block_list = tuple([] for _ in range(len(kv_cache_groups)))
        if VLLM_ASCEND_VERSION == "0.18":
            num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=num_slots_sparsed,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=0,
                total_computed_tokens=0,
                num_tokens_main_model=num_slots_sparsed,
            )

        elif VLLM_ASCEND_VERSION == "0.13":
            num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=num_slots_sparsed,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=0,
            )
        else:
            return None
        manual_preempt = False
        gsa_preempted_flag = False
        if num_output_tokens > 1:
            gsa_preempted_flag = self.estimate_free_slots_sparse(
                request, num_blocks_to_allocate
            )
        if (
            manual_preempt
            or num_blocks_to_allocate > block_pool.get_num_free_blocks()
            or gsa_preempted_flag
        ):
            return None
        coordinator.allocate_new_blocks(request.request_id, num_slots_sparsed, 0)
        blocks = coordinator.single_type_managers[0].req_to_blocks[request.request_id]
        return KVCacheBlocks(tuple([blocks]))

    def _init_slab_preempt_kv(self):
        k_shape = (
            self.max_model_len // self.block_size * 2,
            self.layer_num,
            self.block_size,
            self.num_head,
            self.k_dim,
        )
        self._slab_preempt_k = torch.empty(
            k_shape,
            dtype=self.dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        self._slab_preempt_k_ptr = int(self._slab_preempt_k.data_ptr())
        self._slab_preempt_k_offset = (
            self.layer_num
            * self.block_size
            * self.num_head
            * self.k_dim
            * self._slab_preempt_k.element_size()
        )
        if not self.enable_sparse_sfa_c8:
            v_shape = (
                self.max_model_len // self.block_size * 2,
                self.layer_num,
                self.block_size,
                self.num_head,
                self._vllm_config.model_config.hf_text_config.qk_rope_head_dim,
            )
            self._slab_preempt_v = torch.empty(
                v_shape,
                dtype=self.dtype,
                device=torch.device("cpu"),
                pin_memory=True,
            )
            self._slab_preempt_v_ptr = int(self._slab_preempt_v.data_ptr())
            self._slab_preempt_v_offset = (
                self.layer_num
                * self.block_size
                * self.num_head
                * self._vllm_config.model_config.hf_text_config.qk_rope_head_dim
                * self._slab_preempt_v.element_size()
            )
        else:
            self._slab_preempt_v = self._slab_preempt_k
            self._slab_preempt_v_ptr = self._slab_preempt_k_ptr
            self._slab_preempt_v_offset = self._slab_preempt_k_offset

    def _init_indexer_preempt(self):
        if IS_INDEXER_OFFLOAD:
            indexer_k_shape = (
                self.max_model_len // self.block_size * 2,
                self.layer_num,
                self.block_size,
                1,
                128,
            )
            self.indexer_cache_preempt = torch.empty(
                indexer_k_shape,
                dtype=self.dtype,
                device=torch.device("cpu"),
                pin_memory=True,
            )
            self.indexer_cache_preempt_ptr = int(self.indexer_cache_preempt.data_ptr())
            self.indexer_cache_preempt_offset = (
                self.layer_num
                * self.block_size
                * 128
                * self.indexer_cache_preempt.element_size()
            )
            indexer_k_repr_shape = (self.layer_num, self.num_blocks, 1, 1, 128)
            self.indexer_k_repr = torch.empty(
                indexer_k_repr_shape,
                dtype=self.dtype,
                device=self.device,
                pin_memory=False,
            )

    def _init_kv_load_buffers(self):
        k_buffer_shape = (
            STREAM_NUM,
            KVCACHE_BUFFER_SIZE * self.block_size * self.num_head * self.max_bs,
            self.k_dim,
        )
        self._buff_host_k = torch.empty(
            k_buffer_shape,
            dtype=self.dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        self._buff_device_k = torch.empty(
            k_buffer_shape, dtype=self.dtype, device=self.device, pin_memory=False
        )
        k_buffer_shape = (
            GSA_SPARSE_LEN // self.block_size,
            self.num_head,
            self.block_size,
            self.k_dim,
        )
        self.last_chunk_k_buffer = [
            torch.empty(
                k_buffer_shape,
                dtype=self.dtype,
                device=torch.device("cpu"),
                pin_memory=True,
            )
            for _ in range(self.layer_num)
        ]
        if not self.enable_sparse_sfa_c8:
            v_buffer_shape = (
                STREAM_NUM,
                KVCACHE_BUFFER_SIZE * self.block_size * self.num_head * self.max_bs,
                self._vllm_config.model_config.hf_text_config.qk_rope_head_dim,
            )
            self._buff_host_v = torch.empty(
                v_buffer_shape,
                dtype=self.dtype,
                device=torch.device("cpu"),
                pin_memory=True,
            )
            self._buff_device_v = torch.empty(
                v_buffer_shape, dtype=self.dtype, device=self.device, pin_memory=False
            )
            v_buffer_shape = (
                GSA_SPARSE_LEN // self.block_size,
                self.num_head,
                self.block_size,
                self._vllm_config.model_config.hf_text_config.qk_rope_head_dim,
            )
            self.last_chunk_v_buffer = [
                torch.empty(
                    v_buffer_shape,
                    dtype=self.dtype,
                    device=torch.device("cpu"),
                    pin_memory=True,
                )
                for _ in range(self.layer_num)
            ]
        else:
            self._buff_host_v = self._buff_host_k
            self._buff_device_v = self._buff_device_k
            self.last_chunk_v_buffer = self.last_chunk_k_buffer
        dst_index_shape = (
            STREAM_NUM,
            2048 * self.max_bs,
        )
        self._load_index_host = torch.empty(
            dst_index_shape,
            dtype=torch.int64,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        self._load_index_device = torch.empty(
            dst_index_shape, dtype=torch.int64, device=self.device, pin_memory=False
        )

    def init_host_slabs_from_kv(self, num_slab_blocks):
        if self._buff_host_k is None:
            print(
                " ========================== initialize slab host ========================== "
            )
            self.num_blocks = num_slab_blocks
            self._init_slab_preempt_kv()
            self._init_indexer_preempt()
            self._init_kv_load_buffers()
            print(
                " ========================== initialize slab host success ========================== "
            )

    def init_kvcache_ptrs(self, kv_cache, indexer_cache, indexer_scale, layer_id):
        if self.kv_caches[layer_id] is None:
            self.kv_caches[layer_id] = kv_cache
            self.indexer_k_cache_sparse[layer_id] = indexer_cache
            self.indexer_scale[layer_id] = indexer_scale
            self.prefetch_engine.prefetch_engine_c.set_kvTensor_Layer(
                kv_cache[0],
                kv_cache[1],
                self.last_chunk_k_buffer[layer_id],
                self.last_chunk_v_buffer[layer_id],
                layer_id,
            )
            if IS_INDEXER_OFFLOAD:
                self.prefetch_engine.prefetch_engine_c.set_kvTensor_Index(
                    indexer_cache,
                )

    def shutdown(self):
        if not IS_INDEXER_OFFLOAD or self.block_table_sm is None:
            return
        try:
            self.block_table_sm.unlink()
            print(f"✅ 成功清理: block_table_sm memory", flush=True)
        except Exception as e:
            print(f"❌ block_table_sm 清理失败: {e}", flush=True)

        try:
            self.indexer_block_table_sm.unlink()
            print(f"✅ 成功清理: indexer_block_table_sm", flush=True)
        except Exception as e:
            print(f"❌ indexer_block_table_sm 清理失败: {e}", flush=True)
