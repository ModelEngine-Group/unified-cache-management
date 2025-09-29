import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List, Union

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.forward_context import ForwardContext
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request

from ucm.integration.vllm.ucm_sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
)
from ucm.store.ucmstore import Task, UcmKVStoreBase
from ucm.ucm_sparse import kvstar_retrieve
from ucm.ucm_sparse.kvstar.utils import bind_cpus, block_hash_func, get_offset

"""
--------------------------------------------------------------------------------------
| prefill                                                   | decode
| full block | full block | full block | full block | tail      | <--tail block fully cached during decode step
|            |            |            |            | block     | <-- KVStar multistep:
|init_window |                         |local window|             in long prefill, short decode: not sparse decode fully block
                                                                 TODO: in short prefill, long decode: refresh all blk repre include decode fully block, and update local window blk space
window must be fully block
--------------------------------------------------------------------------------------
"""

ReqType = Union[str, int]  # req_id的标识, 可以是str(UUID)或int(唯一), 和vllm保持一致
HashType = Union[str, int]  # 使用hashtype方便阅读, 快速确认某些管理dict以hash为key


class ReqStage(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()


# NOTE: 预留检索任务状态枚举, TODO: 支持异步检索逻辑
class RetrieveTaskStatus(enum.Enum):
    WAITING = enum.auto()
    FINISHED = enum.auto()


# NOTE: 预留异步检索任务python侧管理结构, TODO: 待根据实际需求确认
@dataclass
class RetrieveManager:
    retrieve_device: str  # CPU/XPU
    request_ids: List[ReqType]
    retrieve_tasks: dict  # task_id/request_id, task_status


# 请求级的spare meta信息
@dataclass
class ReqMeta:
    request_id: ReqType
    index_in_batch: int
    num_prompt_tokens: int
    num_output_tokens: int
    num_scheduled_tokens: int
    num_computed_tokens: int
    num_sparsed_tokens: int
    vllm_block_ids: list[int]
    token_blk_size: int
    prompt_token_ids: list[int]
    query_start_loc: int = -1
    query_len: int = -1
    retrieval_stride: int = 8
    block_hashes: list[str] = field(default_factory=list)

    def set_block_hashes(self, token_ids):
        block_hashes = []
        parent_block_hash_value = None
        for start in range(0, len(token_ids), self.token_blk_size):
            end = start + self.token_blk_size
            block_token_ids = token_ids[start:end]
            if len(block_token_ids) < self.token_blk_size:
                break
            curr_block_token_ids_tuple = tuple(block_token_ids)
            block_hash = block_hash_func(
                parent_block_hash_value, curr_block_token_ids_tuple
            )
            block_hashes.append(str(block_hash))
            parent_block_hash_value = block_hash
        return block_hashes

    @property
    def req_block_hashes(self) -> list[str]:
        if self.block_hashes:
            return self.block_hashes
        self.block_hashes = self.set_block_hashes(self.prompt_token_ids)
        return self.block_hashes

    @property
    def step(self) -> int:
        return self.num_output_tokens

    @property
    def stage(self) -> ReqStage:
        return ReqStage.DECODE if self.num_output_tokens > 0 else ReqStage.PREFILL

    @property
    def is_last_chunk(self) -> bool:
        return (
            self.num_computed_tokens + self.num_scheduled_tokens
            >= self.num_prompt_tokens
        )

    @property
    def prefill_fully_blk_num(self) -> int:
        return self.num_prompt_tokens // self.token_blk_size

    """
    MultiStep 稀疏化算法
    prefill阶段: 利用prompt最后连续8个token做topk检索, 保留25%重要block
    decode阶段:
        step1~8: 根据prefill稀疏化后的kvcache进行计算, 卸载自己的8个query
        step9~16: 继续依赖prefill阶段的topk kvcache, 触发1~8卸载下来的8个query的topk检索任务, 卸载自己的8个query
        step17~24: 根据step1~8的检索结果进行计算, 触发9~16卸载下来的8个query的topk检索任务, 卸载自己的8个query
        step25~32: 根据step9~16的检索结果进行计算, 触发17~24卸载下来的8个query的topk检索任务, 卸载自己的8个query
        ...

    计划设置两个query_group: 
        standby_group: step1~8自己的query卸载到的位置
        do_retrieve_group: 进行step9~16时, step1~8的query换到do_retrieve_group, 用于检索, step9~16自己的query卸载到standby_group
        切换逻辑放在step % RETRIEVAL_TOKEN_GROUP_SIZE == 0 的execute_begin中
    """

    @property
    def query_offload_info(self) -> list | None:
        if self.stage == ReqStage.PREFILL:
            cur_step_parse_prompt_len_end_pos = (
                self.num_computed_tokens + self.num_scheduled_tokens
            )
            if (
                cur_step_parse_prompt_len_end_pos
                < self.num_prompt_tokens - self.retrieval_stride
            ):
                return None
            # 计算应该卸载到standby_group的哪些位置
            valid_token_end_pos_in_retrieve_group = self.retrieval_stride - (
                self.num_prompt_tokens - cur_step_parse_prompt_len_end_pos
            )
            valid_token_num_in_retrieve_group = min(
                valid_token_end_pos_in_retrieve_group, self.num_scheduled_tokens
            )
            valid_token_start_pos_in_retrieve_group = (
                valid_token_end_pos_in_retrieve_group
                - valid_token_num_in_retrieve_group
            )
            return list(
                range(
                    valid_token_start_pos_in_retrieve_group,
                    valid_token_end_pos_in_retrieve_group,
                )
            )
        return [self.num_output_tokens % self.retrieval_stride]


@dataclass
class KVStarMultiStepSparseMetaData(
    UcmSparseMetadata
):  # 生命周期为一次worker step, 每次都会重新设置
    requests: List[ReqMeta]
    finished_req_ids: List[ReqType]

    def __init__(self):
        self.requests = []
        self.finished_req_ids = []

    def add_request(
        self,
        request_id: ReqType,
        index_in_batch: int,
        num_prompt_tokens: int,
        num_output_tokens: int,
        num_scheduled_tokens: int,
        num_computed_tokens: int,
        num_sparsed_tokens: int,
        vllm_block_ids: list[int],
        token_blk_size,
        query_start_loc: int,
        query_len: int,
        retrieval_stride: int,
        prompt_token_ids: list[int],
    ) -> None:
        meta = ReqMeta(
            request_id=request_id,
            index_in_batch=index_in_batch,
            num_prompt_tokens=num_prompt_tokens,
            num_output_tokens=num_output_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_sparsed_tokens=num_sparsed_tokens,
            vllm_block_ids=vllm_block_ids,
            token_blk_size=token_blk_size,
            prompt_token_ids=prompt_token_ids,
            query_start_loc=query_start_loc,
            query_len=query_len,
            retrieval_stride=retrieval_stride,
        )
        self.requests.append(meta)


class ReqPerLayerState:  # 命名风格和vllm保持一致
    """
    成员：
        - blk_repre: request的全量表征池，负责检索输出topk, 对于NPU检索, 尽可能拼成一个tensor, 抽象成GeMM, 对于CPU检索, 需要在Host侧有高效的8 steps query和key cache block表征放置方式
        - blk_hash：和ucmStore连接起来
        - blk_tables: 标记vllm PA的 block_tables中真正存储的tensor是什么，每一轮decode会增量更新此map，起到缓存作用
    方法：
    1. prefill阶段：insert_repre()
    2. decode阶段：
        - insert_repre() when step % block_size == 0
        - retrieval() when step % retrieval_stride == 0
    """

    def __init__(
        self,
        req_meta: ReqMeta,
        layer_name: str,
        rank: int,
        tp_size: int,
        store_instance: UcmKVStoreBase,
        store_name: str,
        sparse_cfg,
    ):
        # TODO: 后续若需要req_id, 作为属性添加
        self.sparse_cfg = sparse_cfg

        self.layer_name = layer_name
        self.layer_id = int(layer_name.split(".")[2])
        self.blk_repre = torch.Tensor()
        self.block_hashes = []

        self.num_tokens = 0  # the number of all_tokens, prompt+output
        self.store_instance = store_instance
        self.store_name = store_name
        self.req_meta = req_meta
        self.init_window: tuple[torch.Tensor, torch.Tensor] = None
        self.local_window: tuple[torch.Tensor, torch.Tensor] = None
        self.init_window_sz = self.sparse_cfg["init_window_sz"]
        self.local_window_sz = self.sparse_cfg["local_window_sz"]
        self.block_size = None
        self.k_cache = None
        self.v_cache = None
        self.d_pruned_index = None
        self.local_tp_rank = rank
        self.total_tp_size = tp_size
        self.blk_trans_tasks: Dict[HashType, Task] = {}
        self.standby_query_group = {}
        self.do_retrieve_query_group = {}

        self.step_group_retrieve_result: dict = {}
        self.task_waiter: dict = {}

        self.init_window_sz = self.sparse_cfg["init_window_sz"]
        self.local_window_sz = self.sparse_cfg["local_window_sz"]

        self.num_blocks_dumped = 0

    # NOTE: 这里的block_id是全量的block(卸载到UC store, 计算表征)的先后idx, 拿它去算的UC store中的hash, 注意区分和vLLM拿token计算hash不一样
    @classmethod
    def block_hash(cls, request_id, block_id):
        return f"req_{request_id}_blk_{block_id}"

    def set_block_hashes(self, token_ids):
        block_hashes = []
        parent_block_hash_value = None
        for start in range(0, len(token_ids), self.block_size):
            end = start + self.block_size
            block_token_ids = token_ids[start:end]
            if len(block_token_ids) < self.block_size:
                break
            curr_block_token_ids_tuple = tuple(block_token_ids)
            block_hash = block_hash_func(
                parent_block_hash_value, curr_block_token_ids_tuple
            )
            block_hashes.append(str(block_hash))
            parent_block_hash_value = block_hash
        return block_hashes

    def retrieval_async(self, cur_step: int, topk: int, retrieve_device="cpu"):
        """
        异步的检索逻辑
        """
        if retrieve_device == "cpu":
            # create cpu retrieve task add to c lib thread pool
            # set task flag 'wait' (until finished)
            retrieve_record = self.get_retrieve_record(cur_step)
            if topk == 0:
                self.step_group_retrieve_result[retrieve_record] = []
                return

            self.do_retrieve_query_group[retrieve_record] = (
                torch.stack(self.standby_query_group[retrieve_record])
                .to(torch.float16)
                .contiguous()
                .to("cpu")
            )
            task_id = kvstar_retrieve.AsyncRetrieveByCPU(
                self.do_retrieve_query_group[retrieve_record],
                self.blk_repre,
                self.d_pruned_index,
                topk,
                int(self.req_meta.request_id),
                kvstar_retrieve.CPU,
            )
            self.task_waiter[retrieve_record] = task_id

        else:
            # XPU, 异步逻辑, 需要创建stream&event, 然后也是记录task等
            pass

    def get_retrieve_record(self, cur_step):
        if cur_step == 1:
            retrieve_record = "prefill"
        else:
            retrieve_record = "decode" + str(
                cur_step - self.sparse_cfg["retrieval_stride"]
            )
        return retrieve_record

    def extract_block_repre(self, vllm_block_ids, prune_dim_enable=False):
        """
        生成key cache block的块级表征
        紧跟着prefill或decode的满块qkv_linear后或者attention后, 没必要异步化增加复杂度了
        """

        # 序列平均
        # 去掉了维度裁剪，因为只有部分block，不能从全局block中选取合适的dim，不能代表query的分布，暂时舍去
        # 维度裁剪
        # 启用prune_dim时才会进行全量的维度筛选
        # 之后每一次都遵循首次全量的维度筛选结果
        if vllm_block_ids[-1] < 2:
            return None
        k_cache = self.k_cache[vllm_block_ids]  # n,S,h,d
        n, S, h, d = k_cache.shape
        if prune_dim_enable and self.sparse_cfg["blk_repre_dim_prune_ratio"] < 0.98:
            k_channel_absmean = (
                k_cache.reshape(n * S, h, d).to(dtype=torch.float32).abs().mean(dim=0)
            )  # Shd -> hd
            d_pruned = round(d * self.sparse_cfg["blk_repre_dim_prune_ratio"])
            _, d_pruned_index = torch.topk(
                k_channel_absmean, k=d_pruned, dim=-1
            )  # hd -> (h, d_prune)
            k_cache_prune = torch.zeros_like(
                k_cache[:, :, :, :d_pruned]
            )  # hSd -> (n, S, h, d_prune)
            for i_h in range(h):
                k_cache_prune[:, :, i_h, :] = k_cache[:, :, i_h, d_pruned_index[i_h]]
            self.d_pruned_index = d_pruned_index.contiguous().to("cpu")
        elif (
            self.d_pruned_index is not None
        ):  # decode 单块 dump时刷新decode块表征, 不参考前面所有完整块, 仅依据prefill获知的通道直接做裁剪 NOTE: 目前不做decode稀疏化, 外层走不到
            h, d_pruned = self.d_pruned_index.shape
            d_pruned_index = self.d_pruned_index
            k_cache_prune = torch.zeros_like(
                k_cache[:, :, :, :d_pruned]
            )  # hSd -> (n, S, h, d_prune)
            for i_h in range(h):
                k_cache_prune[:, :, i_h, :] = k_cache[:, :, i_h, d_pruned_index[i_h]]
        else:  # 不裁剪维度
            d_pruned = d
            k_cache_prune = self.k_cache[vllm_block_ids]

        c = self.sparse_cfg["blk_repre_inner_token_merge"]
        M = S // c
        k_cache_new = k_cache_prune.reshape(n, M, c, h, d_pruned).mean(
            dim=2
        )  # nMchd -> nMhd

        return k_cache_new

    def prepare_init_and_local_window(self):
        vllm_block_ids = self.req_meta.vllm_block_ids
        self.k_cache[vllm_block_ids[: self.init_window_sz]] = self.init_window[0]
        self.v_cache[vllm_block_ids[: self.init_window_sz]] = self.init_window[1]

        if self.local_window is None:
            return

        self.k_cache[vllm_block_ids[-self.local_window_sz :]] = self.local_window[0]
        self.v_cache[vllm_block_ids[-self.local_window_sz :]] = self.local_window[1]

    def construct_init_and_local_window(self):
        vllm_block_ids = self.req_meta.vllm_block_ids
        # TODO: make sure we don't need to clone()
        self.init_window = (
            self.k_cache[vllm_block_ids[: self.init_window_sz]].clone(),
            self.v_cache[vllm_block_ids[: self.init_window_sz]].clone(),
        )
        local_window_sz = min(
            self.local_window_sz, len(vllm_block_ids[self.init_window_sz :])
        )
        if local_window_sz > 0:
            self.local_window = (
                self.k_cache[vllm_block_ids[-local_window_sz:]].clone(),
                self.v_cache[vllm_block_ids[-local_window_sz:]].clone(),
            )

    # NOTE: per_req, layerwise级别的attention_begin/attention_finished, 被UCMSparse级别(batch reqs)的同名函数内部按条件调用
    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        forward_context: ForwardContext,
    ) -> None:
        # -------------------------卸载query---------------------------------
        # 1. 先获取该req的query长度
        index_in_batch = self.req_meta.index_in_batch
        query_start_loc = self.req_meta.query_start_loc
        query_len = self.req_meta.query_len

        if self.req_meta.stage == ReqStage.PREFILL:
            # prefill, chunked prefill query offload, TODO: 填充pass
            self.offload_prefill_query(query, query_len, query_start_loc)
        else:  # decode阶段确定query卸载位置, 不支持投机 TODO: 如何支持
            if self.blk_repre is None:
                return
            assert (
                query_len == 1
            ), "KVStar series sparse attention doesn't support spec_decode now"
            group_record, step_idx_in_retrieve_group = self.get_decode_step_record()
            self.save_to_standby(
                group_record, step_idx_in_retrieve_group, query_start_loc, query
            )

            if self.req_meta.step % self.sparse_cfg["retrieval_stride"] == 0:
                candidate_swap_vllm_block_ids = self.get_retrieve_candidate_block_ids()
                # -------------------------decode query准备好即可触发检索，因为待检索的在step1已经下发---------------------------------
                # 对于step 1, 下发并等待prefill last 8token检索
                # 对于step 9, 下发step1~8检索任务, 等待prefill last 8token检索
                # 对于step 17, 下发step9~16检索任务, 等待step1~8检索任务
                self.retrieval_async(
                    self.req_meta.step + 1, len(candidate_swap_vllm_block_ids)
                )  # 异步逻辑
                # self.retrieval_sync(self.req_meta.step, len(candidate_swap_vllm_block_ids))
            if self.req_meta.step == 1:
                self.prepare_init_and_local_window()
                # step1 特殊操作，需要等待检索任务完成后，串行执行加载，并等待加载完成。
                candidate_swap_vllm_block_ids = self.get_retrieve_candidate_block_ids()
                self.wait_for_blk_transfer_task_done()
                self.retrieval_async(
                    self.req_meta.step, len(candidate_swap_vllm_block_ids)
                )  # 异步逻辑
                self.load_retrieve_result_async(
                    self.req_meta.step, candidate_swap_vllm_block_ids
                )
            if self.req_meta.step % self.sparse_cfg["retrieval_stride"] == 1:
                # 需要等待检索cache加载完成
                self.wait_for_blk_transfer_task_done()

        # TODO: 如果需要初始窗口和最近窗口, 按需调整可换入换出空间
        # NOTE: Some sparse attention algorithms need to modify attn_metadata here

    def offload_prefill_query(self, query, query_len, query_start_loc):
        chunk_prefill_query_offload_info = self.req_meta.query_offload_info
        # 2. 确定是否包含卸载token
        if chunk_prefill_query_offload_info:
            offload_query_len = len(chunk_prefill_query_offload_info)
            # 3. 裁剪需要offload的query
            assert query_len >= offload_query_len
            tokens_to_offload = query[
                query_start_loc
                + query_len
                - offload_query_len : query_start_loc
                + query_len
            ]
            group_record = "prefill"
            for query_relative_idx, in_query_group_idx in enumerate(
                chunk_prefill_query_offload_info
            ):
                self.save_to_standby(
                    group_record,
                    in_query_group_idx,
                    query_relative_idx,
                    tokens_to_offload,
                )

    def load_retrieve_result_async(self, load_step, candidate_swap_vllm_block_ids):
        if load_step <= self.sparse_cfg["retrieval_stride"] * 2:
            need_retrieve_record = "prefill"
        else:
            cur_group_idx = int(
                math.ceil(load_step / self.sparse_cfg["retrieval_stride"])
            )  # e.g. step 17 / 8 = 第3组
            wait_retrieve_step_idx = (cur_group_idx - 3) * self.sparse_cfg[
                "retrieval_stride"
            ] + 1
            need_retrieve_record = "decode" + str(wait_retrieve_step_idx)
        if self.step_group_retrieve_result.get(need_retrieve_record) is None:
            async_retrieve_task_id = self.task_waiter[need_retrieve_record]
            kvstar_retrieve.Wait(async_retrieve_task_id)
            task_result = kvstar_retrieve.GetTaskResult(async_retrieve_task_id)
            del self.standby_query_group[need_retrieve_record]
            del self.do_retrieve_query_group[need_retrieve_record]

            if task_result["status"] == "SUCCESS":  # 假设 0 代表 SUCCESS
                # 3. 从对象中提取出 topkIndices 列表
                topk_indices = task_result["data"]  # KVSTAR_RETRIEVE
                init_window_sz = self.sparse_cfg["init_window_sz"]
                select_blk_hashes = [
                    self.block_hashes[int(id_) + init_window_sz] for id_ in topk_indices
                ]
                self.step_group_retrieve_result[need_retrieve_record] = (
                    select_blk_hashes
                )
            else:
                print(
                    f"task: {async_retrieve_task_id}执行出问题: 结果信息: {task_result}, 对应请求layer {self.layer_id}"
                )
                assert 0  # TODO: 任务重试, 任务重下发(分配新task id), 内部GetTaskResult, task管理目前未做垃圾清理
        retrieve_result_hash_list = self.step_group_retrieve_result.get(
            need_retrieve_record
        )

        # -------------------------触发块异步加载---------------------------------
        # 第一个迭代步取完prefill的检索结果后，被头两组decode复用，第三组才开始取之后的块
        if need_retrieve_record != "prefill" or load_step == 1:
            if len(retrieve_result_hash_list) > 0:
                self.launch_transfer_task(
                    "load", retrieve_result_hash_list, candidate_swap_vllm_block_ids
                )
        return

    def get_retrieve_candidate_block_ids(self):
        candidate_swap_vllm_block_ids = self.req_meta.vllm_block_ids[
            self.init_window_sz : math.ceil(
                self.blk_repre.shape[0] * self.sparse_cfg["sparse_ratio"]
            )
            + self.init_window_sz
        ]
        return candidate_swap_vllm_block_ids

    def get_decode_step_record(self):
        cur_decode_step = self.req_meta.step
        step_idx_in_retrieve_group = (cur_decode_step - 1) % self.sparse_cfg[
            "retrieval_stride"
        ]
        belong_retrieve_group = (
            (cur_decode_step - 1) // self.sparse_cfg["retrieval_stride"]
        ) * self.sparse_cfg["retrieval_stride"] + 1
        group_record = "decode" + str(belong_retrieve_group)
        return group_record, step_idx_in_retrieve_group

    def save_to_standby(
        self, group_record, in_query_group_idx, query_relative_idx, tokens_to_offload
    ):
        if group_record not in self.standby_query_group.keys():
            self.standby_query_group[group_record] = [None] * self.sparse_cfg[
                "retrieval_stride"
            ]
        self.standby_query_group[group_record][in_query_group_idx] = tokens_to_offload[
            query_relative_idx
        ].clone()

    def compute_block_repre(self, num_blocks_need_dump):
        if self.req_meta.stage == ReqStage.PREFILL and self.req_meta.is_last_chunk:
            self.blk_repre = self.extract_block_repre(
                self.req_meta.vllm_block_ids[
                    : self.num_blocks_dumped + num_blocks_need_dump
                ],
                prune_dim_enable=True,
            )
            # NOTE: 关键, 维度剔除首尾块
            if self.blk_repre is not None:
                if self.blk_repre.shape[0] <= 2:
                    self.blk_repre = None  # NOTE: 小于保留窗口, 无需记录块表征
                else:
                    self.blk_repre = (
                        self.blk_repre[self.init_window_sz : -self.local_window_sz]
                        .to(torch.float16)
                        .contiguous()
                        .to("cpu")
                    )
            self.construct_init_and_local_window()

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        forward_context: ForwardContext,
    ) -> None:
        if self.req_meta.stage != ReqStage.PREFILL:
            if (
                self.req_meta.step >= self.sparse_cfg["retrieval_stride"] * 2
                and self.req_meta.step % self.sparse_cfg["retrieval_stride"] == 0
            ):
                # 在decode一组的最后一个迭代步完成attn计算时，启动异步load，此时旧cache已不再需要，可以换成下一组所需cache
                # decode头两组的KVCache在attn_begin时加载，此处只加载第三组开始的KVCache
                candidate_swap_vllm_block_ids = self.get_retrieve_candidate_block_ids()
                self.load_retrieve_result_async(
                    self.req_meta.step + 1, candidate_swap_vllm_block_ids
                )
            return
        # 只在prefill阶段dump cache一次
        self.maybe_register_kv_cache(forward_context)
        num_tokens_updated = (
            self.req_meta.num_computed_tokens + self.req_meta.num_scheduled_tokens
        )
        num_blocks_dumped = self.num_blocks_dumped
        num_full_blocks = num_tokens_updated // self.block_size  # 截断取整获取满块
        num_blocks_need_dump = num_full_blocks - num_blocks_dumped
        self.num_tokens = num_tokens_updated
        # 先异步卸载块, 再计算表征, 掩盖时延
        self.compute_block_repre(num_blocks_need_dump)

    # attention之后, 设置该层的一些kvcache信息到req_layerwise_state, 目前只会设置一次
    def maybe_register_kv_cache(self, forward_context: ForwardContext):
        if self.block_size:
            return
        attn = forward_context.no_compile_layers[self.layer_name]
        kv_cache = attn.kv_cache[forward_context.virtual_engine]
        # TODO: consider is_mla here
        self.k_cache = kv_cache[0]
        self.v_cache = kv_cache[1]
        self.block_size = self.k_cache.shape[1]
        self.num_key_heads = self.k_cache.shape[2]
        self.block_hashes = self.req_meta.req_block_hashes
        self.head_size = self.k_cache.shape[3]

    @classmethod
    def blk_trans_task_hash(
        cls, block_ids, store_type, tensor_type
    ):  # 生成唯一标识块传输任务的hash
        return hash((tuple(block_ids), store_type, tensor_type))

    @classmethod
    def req_state_hash(
        cls, req_id, layer_name
    ):  # 生成唯一标识req_layerwise state的hash
        return hash((req_id, layer_name))

    def update_meta(self, req_meta: ReqMeta, forward_context: ForwardContext):
        self.req_meta = req_meta

    def launch_transfer_task(self, transfer_type, block_hashes, vllm_block_ids):
        fn = getattr(self.store_instance, transfer_type)
        length = len(block_hashes)
        block_shape = (self.block_size, self.num_key_heads, self.head_size)
        precision = self.k_cache.storage().element_size()
        # TODO: consider is_mla here
        is_mla = False

        block_shape = tuple(block_shape)

        # 获取每个key或value在UCStore块内的偏移(UCStore块整合了TP域和全层)
        offsets_k = [
            get_offset(
                block_shape,
                self.local_tp_rank,
                self.total_tp_size,
                precision,
                self.layer_id,
                is_v=False,
                is_mla=is_mla,
            )
        ] * length
        offsets_v = [
            get_offset(
                block_shape,
                self.local_tp_rank,
                self.total_tp_size,
                precision,
                self.layer_id,
                is_v=True,
                is_mla=is_mla,
            )
        ] * length

        # vLLM block 位置
        key_src_tensors = [self.k_cache[id_] for id_ in vllm_block_ids]
        value_src_tensors = [self.v_cache[id_] for id_ in vllm_block_ids]

        # load or dump
        task_k = fn(block_hashes, offsets_k, key_src_tensors)
        task_v = fn(block_hashes, offsets_v, value_src_tensors)

        # 计算任务hash, 方便记录task元信息&状态
        task_k_hash = self.blk_trans_task_hash(block_hashes, transfer_type, "key")
        self.blk_trans_tasks[task_k_hash] = task_k
        task_v_hash = self.blk_trans_task_hash(block_hashes, transfer_type, "value")
        self.blk_trans_tasks[task_v_hash] = task_v

    def wait_for_blk_transfer_task_done(
        self,
    ):  # 一些异步任务等待逻辑 NOTE: 注意区分检索任务和blk传输任务
        for task_hash, task in self.blk_trans_tasks.items():
            # TODO: handle exceptions here, refer to UcmKVConnector
            ret = self.store_instance.wait(task)
        self.blk_trans_tasks.clear()


class KVStarMultiStep(UcmSparseBase):
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config=vllm_config, role=role)

        # TODO: req_states should be shared among all ranks: 涉及到某些稀疏化算法需要融合全部kvcache头, 则这个需要跨进程共享
        self.req_states: dict[str, List[ReqPerLayerState]] = (
            {}
        )  # key用于标识请求及对应层, value是该请求该层的一些稀疏化管理信息
        self.local_tp_rank = vllm_config.parallel_config.rank
        self.total_tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.total_num_hidden_layers = (
            vllm_config.model_config.hf_config.num_hidden_layers
        )
        if self.role == UcmSparseRole.WORKER:
            # TODO: 进行异步检索模块c lib库的相关init
            ratio = 0.75
            numa_nodes_num, alloc_numa_ids, phy_cpu_core_per_numa = bind_cpus(
                self.total_tp_size, self.local_tp_rank, ratio=ratio
            )

            cpu_device = kvstar_retrieve.CPU
            param = kvstar_retrieve.SetupParam(
                cpuNumaIds=alloc_numa_ids,
                physicalCorePerNuma=phy_cpu_core_per_numa,
                allocRatio=ratio,
                blkRepreSize=4096,  # 无效入参
                deviceType=cpu_device,  # 直接传递枚举对象
                totalTpSize=self.total_tp_size,
                localRankId=self.local_tp_rank,
            )
            kvstar_retrieve.Setup(param)
            self.connector_name = (
                self._vllm_config.kv_transfer_config.kv_connector_extra_config[
                    "ucm_connector_name"
                ]
            )
            self.connector = get_kv_transfer_group().connector

        else:
            self.connector = None
        # Note: 和ucm prefixcache block共用connector
        assert self._vllm_config.kv_transfer_config is not None

        # scheduler侧也记录config, 也许有用
        self.kvstar_multistep_cfg = (
            vllm_config.kv_transfer_config.kv_connector_extra_config[
                "ucm_sparse_config"
            ]["KVStarMultiStep"]
        )
        print(f"kvstar_multistep_cfg: {self.kvstar_multistep_cfg}")

        self.token_blk_size = vllm_config.cache_config.block_size

    # TODO: 按照SparseBase基类的约束, 分别实现对应的功能

    # ==============================
    # Scheduler/Worker-side 按Role区分的共有逻辑
    # ==============================

    def create_layerwise_req_state(self, req_meta, layer_name):
        layer_id = int(layer_name.split(".")[2])
        if req_meta.request_id not in self.req_states:
            if self.req_states.get(req_meta.request_id) is None:
                self.req_states[req_meta.request_id] = [
                    None
                ] * self.total_num_hidden_layers
        if self.req_states[req_meta.request_id][layer_id] is None:
            self.req_states[req_meta.request_id][layer_id] = ReqPerLayerState(
                req_meta,
                layer_name,
                self.local_tp_rank,
                self.total_tp_size,
                self.connector,
                self.connector_name,
                self.kvstar_multistep_cfg,
            )
        return self.req_states[req_meta.request_id][layer_id]

    def request_begin(self, request_id: Union[int, str], prompt_token_ids: List[int]):
        """
        This is called at the beginning of "Scheduler->add_request" function.
        """
        pass

    # ==============================
    # Worker-side methods
    # ==============================

    def request_finished_in_worker(self, request_id: ReqType):
        del self.req_states[request_id]

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
    ) -> None:
        """
        This is called at the beginning of "unified_attention".
        Sparse attention algorithm can modify forward_context.attn_metadata if necessary.
        (UC_TODO: modify dataclass is not allowed in python?)

        Modify forward_context.attn_metadata in-place

        每一次(每层)attention开始前, 包在unified_attention内

        """
        for req_meta in self._sparse_metadata.requests:
            req_layerwise_state = self.create_layerwise_req_state(req_meta, layer_name)
            req_layerwise_state.update_meta(
                req_meta, forward_context
            )  # 重新绑定本次step该请求刷新后的meta
            req_layerwise_state.attention_begin(query, key, value, forward_context)

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
    ) -> None:
        """
        This is called at the end of "unified_attention".
        每一次(每层)attention结束后, 包在unified_attention内
        比如下一层的预取, kvcache卸载或其他逻辑
        """
        for req_meta in self._sparse_metadata.requests:
            req_layerwise_state = self.create_layerwise_req_state(req_meta, layer_name)
            req_layerwise_state.update_meta(
                req_meta, forward_context
            )  # 重新绑定本次step该请求刷新后的meta
            req_layerwise_state.attention_finished(
                query, key, value, attn_output, forward_context
            )  # NOTE: 可以把attention out也传进去

    # ==============================
    # 其他的一些子类worker侧自己的函数
    # ==============================

    def build_sparse_meta(
        self, scheduler_output, requests, input_batch, attn_metadata
    ) -> None:  # 函数内bind
        """
        Build the sparse metadata for this step.
        目前梳理出的, sparse metadata所需信息基本都可在worker侧获取, 无需scheduler创建传递
        """

        sparse_meta = KVStarMultiStepSparseMetaData()

        if isinstance(attn_metadata, dict):
            attn_metadata = next(iter(attn_metadata.values()))

        """
        逻辑:
        1. 对于新请求 scheduler_output.scheduled_new_reqs(首次调度或者被打断后重算), 前者UCStore没缓存, 后者有
        2. 对于已计算过的请求(Prefill后, ChunkedPrefill首次后) scheduler_output.scheduled_cached_reqs
        这些请求, 需要在pre/post attention, model, req多个层面的稀疏化操作需要做些什么

        当前build_sparse_meta调用点在self.model forward前, vllm_ascend.worker.npu_input_batch CachedRequestState 已组装好未调度结束的请求的信息, 由此构建sparse_meta
        """
        query_start_locs = attn_metadata.query_start_loc

        for (
            req_id,
            num_scheduled_tokens,
        ) in (
            scheduler_output.num_scheduled_tokens.items()
        ):  # NOTE: num_scheduled_tokens包含投机token
            req_state = requests[req_id]

            q_start_loc = query_start_locs[input_batch.req_id_to_index[req_id]].item()
            q_len = (
                query_start_locs[input_batch.req_id_to_index[req_id] + 1].item()
                - q_start_loc
            )

            if len(req_state.prompt_token_ids) > self.token_blk_size:
                sparse_meta.add_request(
                    req_id,
                    input_batch.req_id_to_index[req_id],
                    len(req_state.prompt_token_ids),
                    len(req_state.output_token_ids),  # 当前生成的且验证过的out_token
                    num_scheduled_tokens,
                    req_state.num_computed_tokens,  # 已经计算过的token长度(即有其kvcache)
                    scheduler_output.req_sparsed_slots[
                        req_id
                    ],  # 当前给定的slot预算 (num_sparsed_tokens)
                    req_state.block_ids[
                        0
                    ],  # 当前只支持单种kvcache group, tuple [0] 元素
                    self.token_blk_size,
                    q_start_loc,
                    q_len,
                    self.kvstar_multistep_cfg["retrieval_stride"],
                    req_state.prompt_token_ids,
                )

        self._sparse_metadata = sparse_meta

    # ==============================
    # Scheduler-side methods
    # ==============================

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        """
        This is called by "Scheduler->schedule" function to estimate the number of required slots.
        """
        if request.num_output_tokens == 0:  # prefill/chunked_prefill
            return INVALID_SLOT
        block_size = self._vllm_config.cache_config.block_size

        num_prefill_fully_block = request.num_prompt_tokens // block_size
        num_prefill_keep_fixed_blk = min(
            self.kvstar_multistep_cfg["init_window_sz"]
            + self.kvstar_multistep_cfg["local_window_sz"],
            num_prefill_fully_block,
        )

        num_sparse_saved_fully_blk = math.ceil(
            (num_prefill_fully_block - num_prefill_keep_fixed_blk)
            * self.kvstar_multistep_cfg["sparse_ratio"]
        )  # same as blk_repre.shape[0] * SPARSE_RATIO

        num_blocks_dense_total = math.ceil(request.num_tokens / block_size)  # 向上取整

        num_blocks_be_compressed_prefill = (
            num_prefill_fully_block
            - num_sparse_saved_fully_blk
            - num_prefill_keep_fixed_blk
        )

        num_blocks_this_step_budget = (
            num_blocks_dense_total - num_blocks_be_compressed_prefill
        )

        tail_blk_valid_token_num = request.num_tokens % block_size
        if tail_blk_valid_token_num:
            estimate_num_slots_budget = (
                num_blocks_this_step_budget - 1
            ) * block_size + tail_blk_valid_token_num
        else:
            estimate_num_slots_budget = (
                num_blocks_this_step_budget * block_size
            )  # 接下来一步会满块, 触发block dump
        return estimate_num_slots_budget

    # TODO: 适配KVStar slots分配需求
    def allocate_slots(
        self, request, num_slots_sparsed, coordinator, block_pool, kv_cache_groups
    ):
        block_size = self._vllm_config.cache_config.block_size
        num_blocks_need = math.ceil(num_slots_sparsed / block_size)
        allocated_blocks = coordinator.get_blocks(request.request_id)[0]
        returned_blocks = []
        kept_blocks = []
        num_blocks_original = len(allocated_blocks)
        for i, block in enumerate(allocated_blocks):
            if i >= num_blocks_original - num_blocks_need:
                kept_blocks.append(block)
            else:
                returned_blocks.append(block)
            block_pool._maybe_evict_cached_block(block)
        block_pool.free_blocks(returned_blocks)

        coordinator.single_type_managers[0].req_to_blocks[
            request.request_id
        ] = kept_blocks

        new_computed_block_list = tuple([] for _ in range(len(kv_cache_groups)))
        num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_slots_sparsed,
            new_computed_blocks=new_computed_block_list,
        )
        if num_blocks_to_allocate > block_pool.get_num_free_blocks():
            return None
        coordinator.allocate_new_blocks(request.request_id, num_slots_sparsed)
        return KVCacheBlocks(tuple([kept_blocks]))
