from typing import Union, Dict, List, Optional
import enum
from dataclasses import dataclass
import math

import torch
import torch_npu

from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.worker.npu_input_batch import CachedRequestState, InputBatch

from unifiedcache.ucm_connector.factory import UcmConnectorFactory
from unifiedcache.ucm_connector.base import Task, UcmKVStoreBase

from unifiedcache.integration.vllm.ucm_sparse.base import UcmSparseBase, UcmSparseMetadata, UcmSparseRole, INVALID_SLOT

from unifiedcache.ucm_sparse.kvstar.utils import get_offset, bind_cpus
import kvstar_retrieve

# TODO: import c lib库模块, 解放GIL锁

# NOTE: 常量&宏
RETRIEVAL_TOKEN_GROUP_SIZE = 8
SPARSE_RATIO = 0.25
PRUNE_DIM_RATIO = 0.25  # 维度裁剪比例
INIT_WINDOW_SZ = 1
LOCAL_WINDOW_SZ = 1

'''
--------------------------------------------------------------------------------------
| prefill                                                   | decode
| full block | full block | full block | full block | tail      | <--tail block fully cached during decode step
|            |            |            |            | block     | <-- KVStar multistep:
|init_window |                         |local window|             in long prefill, short decode: not sparse decode fully block
                                                                 TODO: in short prefill, long decode: refresh all blk repre include decode fully block, and update local window blk space
window must be fully block
--------------------------------------------------------------------------------------
'''

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

    @property
    def step(self) -> int:
        return self.num_output_tokens

    @property
    def stage(self) -> ReqStage:
        return ReqStage.DECODE if self.num_output_tokens > 0 else ReqStage.PREFILL

    @property
    def is_last_chunk(self) -> bool:
        return self.num_computed_tokens + self.num_scheduled_tokens >= self.num_prompt_tokens

    @property
    def prefill_fully_blk_num(self) -> int:
        return self.num_prompt_tokens // self.token_blk_size

    '''
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
    '''

    @property
    def query_offload_info(self) -> list | None:
        if self.stage == ReqStage.PREFILL:
            cur_step_parse_prompt_len_end_pos = self.num_computed_tokens + self.num_scheduled_tokens
            if cur_step_parse_prompt_len_end_pos < self.num_prompt_tokens - RETRIEVAL_TOKEN_GROUP_SIZE:
                return None
            # 计算应该卸载到standby_group的哪些位置
            valid_token_end_pos_in_retrieve_group = RETRIEVAL_TOKEN_GROUP_SIZE - (self.num_prompt_tokens - cur_step_parse_prompt_len_end_pos)
            valid_token_num_in_retrieve_group = min(valid_token_end_pos_in_retrieve_group, self.num_scheduled_tokens)
            valid_token_start_pos_in_retrieve_group = valid_token_end_pos_in_retrieve_group - valid_token_num_in_retrieve_group
            return list(range(valid_token_start_pos_in_retrieve_group, valid_token_end_pos_in_retrieve_group))
        return [self.num_output_tokens % RETRIEVAL_TOKEN_GROUP_SIZE]


@dataclass
class KVStarMultiStepSparseMetaData(UcmSparseMetadata):  # 生命周期为一次worker step, 每次都会重新设置
    requests: List[ReqMeta]
    finished_req_ids: List[ReqType]

    def __init__(self):
        self.requests = []
        self.finished_req_ids = []

    def add_request(self,
                    request_id: ReqType,
                    index_in_batch: int,
                    num_prompt_tokens: int,
                    num_output_tokens: int,
                    num_scheduled_tokens: int,
                    num_computed_tokens: int,
                    num_sparsed_tokens: int,
                    vllm_block_ids: list[int],
                    token_blk_size
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
            token_blk_size=token_blk_size
        )
        self.requests.append(meta)

class ReqPerLayerState():  # 命名风格和vllm保持一致
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

    def __init__(self, req_meta: ReqMeta, layer_name: str, rank: int, tp_size: int, store_instance: UcmKVStoreBase):
        # TODO: 后续若需要req_id, 作为属性添加

        self.layer_name = layer_name
        self.layer_id = int(layer_name.split(".")[2])
        # 如果检索用NPU, 则用cat扩展块表征tensor, 类似于list append上去了, 利于提升NPU做检索的计算效率
        self.blk_repre = torch.Tensor()  # NPU: Tensor: (blks, num_key_heads_per_tp, head_size), CPU: list[Tensor], self.full_blk_num, 即CPU: len(self.blk_repre), NPU: self.blk_repre.shape[0]

        self.num_tokens = 0  # the number of all_tokens, prompt+output
        self.store_instance = store_instance

        self.req_meta = req_meta

        self.block_size = None
        self.k_cache = None
        self.v_cache = None
        self.d_pruned_index = None
        self.local_tp_rank = rank
        self.total_tp_size = tp_size

        self.blk_trans_tasks: Dict[HashType, Task] = {}

        self.standby_query_group: List[Optional[torch.Tensor]] = [None] * RETRIEVAL_TOKEN_GROUP_SIZE
        self.do_retrieve_query_group: List[Optional[torch.Tensor]] = [None] * RETRIEVAL_TOKEN_GROUP_SIZE

        self.step_group_retrieve_result: dict = {}
        self.task_waiter: dict = {}

        self.init_window_sz = INIT_WINDOW_SZ
        self.local_window_sz = LOCAL_WINDOW_SZ  # only save prefill last fully block for acc

        self.num_blocks_dumped = 0



    # NOTE: 这里的block_id是全量的block(卸载到UC store, 计算表征)的先后idx, 拿它去算的UC store中的hash, 注意区分和vLLM拿token计算hash不一样
    @classmethod
    def block_hash(cls, request_id, block_id):
        return f"req_{request_id}_blk_{block_id}"

    def retrieval_sync(self, cur_step: int, topk: int, retrieve_device='cpu'):
        """
        同步的检索逻辑
        """
        if retrieve_device == 'cpu':
            if cur_step == 1:
                retrieve_record = "prefill"
            else:
                retrieve_record = "decode" + str(cur_step - RETRIEVAL_TOKEN_GROUP_SIZE)
            if topk == 0:
                self.step_group_retrieve_result[retrieve_record] = []
                return

            # querys and key cache block repre
            retrieve_query_group = self.do_retrieve_query_group
            fully_filed_blk_num_except_fixed_window = self.blk_repre.shape[0]
            if topk >= fully_filed_blk_num_except_fixed_window:
                select_blk_hashes = [f"{self.block_hash(self.req_meta.request_id, id_ + INIT_WINDOW_SZ)}" for id_ in range(0, self.blk_repre.shape[0])]  # 只有prefill阶段的块, 且进行过块裁剪
            else:
                k_cache = self.blk_repre  # n,h,d
                q_token = torch.stack(retrieve_query_group)
                x, H, d = q_token.shape  # x: token num, H: head num, d: head dim
                n, h, _ = k_cache.shape
                g = H // h
                # 裁剪Q维度
                if self.d_pruned_index is not None:
                    h, d_pruned = self.d_pruned_index.shape
                    q_token = q_token.reshape(x, h, g, -1)
                    q_token_prune = torch.zeros_like(q_token[:, :, :, :d_pruned])
                    for i_h in range(h):
                        q_token_prune[:, i_h, :, :] = q_token[:, i_h, :, self.d_pruned_index[i_h]]
                    q_token_prune.reshape(x, H, d_pruned)
                    d = d_pruned
                    q_token = q_token_prune

                q_token = q_token.reshape(x, h, g, d)
                # 计算得分
                score = torch.einsum('xhgd,nhd->xhgn', q_token, k_cache)
                score = score.to(dtype=torch.float32)
                score = torch.nn.functional.softmax(score, dim=-1).to(q_token.dtype)  # xhgn
                score = torch.einsum('xhgn->n', score)
                # 选择block
                _, topk_index = torch.topk(score, k=topk, dim=-1)  # (n_chunk_topk)
                select_blk_hashes = [f"{self.block_hash(self.req_meta.request_id, id_ + INIT_WINDOW_SZ)}" for id_ in list(topk_index)]

            self.step_group_retrieve_result[retrieve_record] = select_blk_hashes
        else:
            # TODO: 如果用NPU, 拿tensor进行检索, 找topk个最相关blk, 对应逻辑可以放在这边
            #  如果对self.do_retrieve_query_group, self.blk_repre的数据格式有需求, 沿着这些属性修改逻辑改了就行, 以下层实现友好优先
            pass  # XPU

    def retrieval_async(self, cur_step: int, topk: int, retrieve_device='cpu'):
        """
        异步的检索逻辑
        """
        if retrieve_device == 'cpu':
            # create cpu retrieve task add to c lib thread pool
            # set task flag 'wait' (until finished)
            if cur_step == 1:
                retrieve_record = "prefill"
            else:
                retrieve_record = "decode" + str(cur_step - RETRIEVAL_TOKEN_GROUP_SIZE)
            if topk == 0:
                self.step_group_retrieve_result[retrieve_record] = []
                return
            retrieve_query_group = self.do_retrieve_query_group
            q_group_stack_tensor = torch.stack(retrieve_query_group).to(torch.float16).contiguous().to('cpu')
            task_id = kvstar_retrieve.AsyncRetrieveByCPU(q_group_stack_tensor, self.blk_repre, self.d_pruned_index,
                                                         topk, int(self.req_meta.request_id), kvstar_retrieve.CPU)
            self.task_waiter[retrieve_record] = task_id

        else:
            # XPU, 异步逻辑, 需要创建stream&event, 然后也是记录task等
            pass

    def construct_blk_mapping(self, layer_name, block_hash_hashes, vllm_block_ids):
        """
        prefill/chunked_prefill首次/短序列直到decode阶段凑出首个满块, 从无到有构建vllm<-->UC Store块映射
        """
        pass

    def update_blk_mapping(self, layer_name, block_hash_hashes, vllm_block_ids):
        """
        后续chunked_prefill, decode满块, 更新vllm<-->UC Store块映射
        """
        pass

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

        if prune_dim_enable and PRUNE_DIM_RATIO < 0.98:
            k_cache = self.k_cache[vllm_block_ids]
            n, S, h, d = k_cache.shape
            k_channel_absmean = k_cache.reshape(n * S, h, d).to(dtype=torch.float32).abs().mean(dim=0)  # Shd -> hd
            d_pruned = round(d * PRUNE_DIM_RATIO)
            _, d_pruned_index = torch.topk(k_channel_absmean, k=d_pruned, dim=-1)  # hd -> (h, d_prune)
            k_cache_prune = torch.zeros_like(k_cache[:, :, :, :d_pruned])  # hSd -> (n, S, h, d_prune)
            for i_h in range(h):
                k_cache_prune[:, :, i_h, :] = k_cache[:, :, i_h, d_pruned_index[i_h]]
            self.d_pruned_index = d_pruned_index.contiguous().to('cpu')
        elif self.d_pruned_index is not None:  # decode 单块 dump时刷新decode块表征, 不参考前面所有完整块, 仅依据prefill获知的通道直接做裁剪 NOTE: 目前不做decode稀疏化, 外层走不到
            k_cache = self.k_cache[vllm_block_ids]  # n,S,h,d

            h, d_pruned = self.d_pruned_index.shape
            d_pruned_index = self.d_pruned_index
            k_cache_prune = torch.zeros_like(k_cache[:, :, :, :d_pruned])  # hSd -> (n, S, h, d_prune)
            for i_h in range(h):
                k_cache_prune[:, :, i_h, :] = k_cache[:, :, i_h, d_pruned_index[i_h]]
        else:  # 不裁剪维度
            k_cache_prune = self.k_cache[vllm_block_ids]

        # 去掉了检索维度，因为都是在cpu上做，没有npu参与，prefill和decode归一为一个block一个向量
        k_cache_new = k_cache_prune.mean(dim=1)  # nShd -> nhd

        return k_cache_new

    def insert_repre(self, layer_name: int, block_hashes: List, key: torch.Tensor, vllm_block_ids: List[int]):
        """
        生成的key cache block的块级表征插到管理结构中
        如果是CPU, 则需要pybind11出来对应的insert接口
        """
        pass

    def free_req_repre(self):
        """
        请求finished后, 清理它的块级表征
        """
        pass

    # NOTE: per_req, layerwise级别的attention_begin/attention_finished, 被UCMSparse级别(batch reqs)的同名函数内部按条件调用
    def attention_begin(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        forward_context: ForwardContext) -> None:

        # -------------------------卸载query---------------------------------
        # 1. 先获取该req的query长度
        index_in_batch = self.req_meta.index_in_batch
        query_start_loc = forward_context.attn_metadata.query_start_loc[index_in_batch]
        query_len = forward_context.attn_metadata.query_lens[index_in_batch]

        if self.req_meta.stage == ReqStage.PREFILL:
            # prefill, chunked prefill query offload, TODO: 填充pass
            chunk_prefill_query_offload_info = self.req_meta.query_offload_info

            # 2. 确定是否包含卸载token
            if chunk_prefill_query_offload_info:
                offload_query_len = len(chunk_prefill_query_offload_info)

                # 3. 裁剪需要offload的query
                assert query_len >= offload_query_len
                tokens_to_offload = query[query_start_loc + query_len - offload_query_len: query_start_loc + query_len]

                for query_relative_idx, in_query_group_idx in enumerate(chunk_prefill_query_offload_info):
                    self.standby_query_group[in_query_group_idx] = tokens_to_offload[query_relative_idx]
        else:  # decode阶段确定query卸载位置, 不支持投机 TODO: 如何支持
            assert query_len == 1, "KVStar series sparse attention doesn't support spec_decode now"
            cur_decode_step = self.req_meta.step
            step_idx_in_retrieve_group = cur_decode_step % RETRIEVAL_TOKEN_GROUP_SIZE
            self.standby_query_group[step_idx_in_retrieve_group] = query[query_start_loc]

        if self.req_meta.step % RETRIEVAL_TOKEN_GROUP_SIZE == 1:
            for i in range(RETRIEVAL_TOKEN_GROUP_SIZE):
                self.do_retrieve_query_group[i] = self.standby_query_group[i]
                self.standby_query_group[i] = None

            if self.blk_repre is not None:
                candidate_swap_vllm_block_ids = self.req_meta.vllm_block_ids[self.init_window_sz: math.ceil(self.blk_repre.shape[0] * SPARSE_RATIO) + self.init_window_sz]

            else:
                candidate_swap_vllm_block_ids = []  # 实际无需检索 TODO: 代码重构

            # -------------------------触发检索---------------------------------
            # 对于step 1, 下发并等待prefill last 8token检索
            # 对于step 9, 下发step1~8检索任务, 等待prefill last 8token检索
            # 对于step 17, 下发step9~16检索任务, 等待step1~8检索任务
            self.retrieval_async(self.req_meta.step, len(candidate_swap_vllm_block_ids))  # 异步逻辑

            # 按需根据刷新后的topk检索结果, 实现block该步换入换出
            if self.req_meta.step <= RETRIEVAL_TOKEN_GROUP_SIZE * 2:
                need_retrieve_record = "prefill"
            else:
                cur_group_idx = int(math.ceil(self.req_meta.step / RETRIEVAL_TOKEN_GROUP_SIZE))  # e.g. step 17 / 8 = 第3组
                wait_retrieve_step_idx = (cur_group_idx - 3) * RETRIEVAL_TOKEN_GROUP_SIZE + 1
                need_retrieve_record = "decode" + str(wait_retrieve_step_idx)

            if self.step_group_retrieve_result.get(need_retrieve_record) is None:
                async_retrieve_task_id = self.task_waiter[need_retrieve_record]
                kvstar_retrieve.Wait(async_retrieve_task_id)
                task_result = kvstar_retrieve.GetTaskResult(async_retrieve_task_id)
                print(task_result)
                if task_result['status'] == 'SUCCESS':  # 假设 0 代表 SUCCESS
                    # 3. 从对象中提取出 topkIndices 列表
                    topk_indices = task_result["data"]  # KVSTAR_RETRIEVE
                    select_blk_hashes = [f"{self.block_hash(self.req_meta.request_id, int(id_) + INIT_WINDOW_SZ)}" for id_ in topk_indices]
                    self.step_group_retrieve_result[need_retrieve_record] = select_blk_hashes
                else:
                    print(f"task: {async_retrieve_task_id}执行出问题: 结果信息: {task_result}, 对应请求layer {self.layer_id}")
                    assert 0  # TODO: 任务重试, 任务重下发(分配新task id), 内部GetTaskResult, task管理目前未做垃圾清理

            retrieve_result_hash_list = self.step_group_retrieve_result.get(need_retrieve_record)
            #
            # while True: 同步逻辑
            #     retrieve_result_hash_list = self.step_group_retrieve_result.get(need_retrieve_record) # TODO: 后续用不到的之前step的检索结果pop出去
            #     if retrieve_result_hash_list is not None:
            #         break

            # -------------------------触发块加载---------------------------------
            if need_retrieve_record != "prefill" or self.req_meta.step == 1:  # prefill的检索结果可以被decode step 1~16用
                if len(retrieve_result_hash_list) > 0:
                    self.launch_transfer_task("load", retrieve_result_hash_list, candidate_swap_vllm_block_ids)

            self.wait_for_blk_transfer_task_done()  # 成员函数, 本请求layer_wise级别的task等待, 即当前层加载完成

            # TODO: 如果需要初始窗口和最近窗口, 按需调整可换入换出空间
        # NOTE: Some sparse attention algorithms need to modify attn_metadata here
        # attn_metadata应该也是layer_wise级别的, 改了不产生跨层影响

    def save_blocks(self, num_blocks_need_dump):
        if num_blocks_need_dump <= 0:
            return
        vllm_block_ids = self.req_meta.vllm_block_ids  # prefill dense token blk, decode sparse budget, must finish prefill phase task before schedule req to decode
        # num_blocks_dumped = self.blk_repre.shape[0] # NOTE: 这里无法通过self.blk_repre获取dump块数
        cur_dumped_blk_num = self.num_blocks_dumped
        need_save_blk_ids = list(range(cur_dumped_blk_num, cur_dumped_blk_num + num_blocks_need_dump))
        need_save_block_hashes = [f"{self.block_hash(self.req_meta.request_id, id_)}" for id_ in need_save_blk_ids]

        if self.req_meta.stage == ReqStage.PREFILL:
            vllm_block_ids_dump = vllm_block_ids[cur_dumped_blk_num: cur_dumped_blk_num + num_blocks_need_dump]
        else:  # 当前不支持投机, decode只能卸载尾部满块 TODO: 支持投机
            vllm_block_ids_dump = vllm_block_ids[-1:]
        self.launch_transfer_task("dump", need_save_block_hashes, vllm_block_ids_dump)  # 先异步卸载块, 再计算表征, 掩盖时延
        if self.req_meta.stage == ReqStage.PREFILL and self.req_meta.is_last_chunk:
            self.blk_repre = self.extract_block_repre(vllm_block_ids_dump, prune_dim_enable=True)
            # NOTE: 关键, 维度剔除首尾块
            if self.blk_repre is not None:
                if self.blk_repre.shape[0] <= 2:
                    self.blk_repre = None  # NOTE: 小于保留窗口, 无需记录块表征
                else:
                    self.blk_repre = self.blk_repre[self.init_window_sz: -self.local_window_sz].to(torch.float16).contiguous().to('cpu')

        self.num_blocks_dumped += num_blocks_need_dump  # NOTE: 更新, self.num_blocks_dumped是个tensor, 是引用不是数值拷贝

        # elif self.req_meta.stage == ReqStage.DECODE: NOTE: decode阶段block稠密不做稀疏
        #     self.blk_repre = torch.cat([self.blk_repre,self.extract_block_repre(vllm_block_ids_dump, prune_dim_enable=False)], dim=0)
        # TODO: 如果需要表征做成单个大tensor, 则可考虑self.blk_repres = torch.cat([self.blk_repres, blk_repres])

    # NOTE: 块卸载和计算表征放在attention_finished 是因为当前attention操作包了reshape_and_cache操作,
    #  在attention后进行卸载能够方便地从keycache中通过维度切片抽出对应满块或计算块级表征
    def attention_finished(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_output: torch.Tensor,
                           forward_context: ForwardContext) -> None:
        self.maybe_register_kv_cache(forward_context)
        index_in_batch = self.req_meta.index_in_batch
        query_len = forward_context.attn_metadata.query_lens[index_in_batch]

        if self.req_meta.stage == ReqStage.PREFILL:  # 为了支持chunked prefill
            num_tokens_updated = self.num_tokens + query_len
            num_blocks_dumped = self.num_blocks_dumped
            num_full_blocks = num_tokens_updated // self.block_size  # 截断取整获取满块
            num_blocks_need_dump = num_full_blocks - num_blocks_dumped
            self.num_tokens = num_tokens_updated
        else:  # 不支持投机 TODO: 如何支持
            assert query_len == 1, "KVStar series sparse attention doesn't support spec_decode now"
            is_last_slot = (self.req_meta.num_sparsed_tokens % self.block_size) == 0
            num_blocks_need_dump = 1 if is_last_slot else 0
        self.save_blocks(num_blocks_need_dump)  # 计算满块块级表征&卸载到UCStore

        if self.req_meta.stage == ReqStage.PREFILL and self.req_meta.is_last_chunk:
            self.wait_for_blk_transfer_task_done()  # 如果是最后的prefill阶段, 需要确保完全卸载

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
        self.head_size = self.k_cache.shape[3]

    @classmethod
    def blk_trans_task_hash(cls, block_ids, store_type, tensor_type):  # 生成唯一标识块传输任务的hash
        return hash((tuple(block_ids), store_type, tensor_type))

    @classmethod
    def req_state_hash(cls, req_id, layer_name):  # 生成唯一标识req_layerwise state的hash
        return hash((req_id, layer_name))

    def update_meta(self, req_meta: ReqMeta, forward_context: ForwardContext):
        self.req_meta = req_meta

    def launch_transfer_task(self, transfer_type, block_hashes, vllm_block_ids):
        fn = getattr(self.store_instance, transfer_type)
        length = len(block_hashes)
        block_shape = (self.block_size, self.num_key_heads, self.head_size)
        precision = self.k_cache.untyped_storage().element_size()
        # TODO: consider is_mla here
        is_mla = False
        # 获取每个key或value在UCStore块内的偏移(UCStore块整合了TP域和全层)
        offsets_k = [get_offset(block_shape, self.local_tp_rank, self.total_tp_size, precision, self.layer_id, is_v=False, is_mla=is_mla)] * length
        offsets_v = [get_offset(block_shape, self.local_tp_rank, self.total_tp_size, precision, self.layer_id, is_v=True, is_mla=is_mla)] * length

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

    def wait_for_blk_transfer_task_done(self):  # 一些异步任务等待逻辑 NOTE: 注意区分检索任务和blk传输任务
        for task_hash, task in self.blk_trans_tasks.items():
            # TODO: handle exceptions here, refer to UcmKVConnector
            ret = self.store_instance.wait(task)
        self.blk_trans_tasks.clear()


class KVStarMultiStep(UcmSparseBase):
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config=vllm_config, role=role)

        # TODO: req_states should be shared among all ranks: 涉及到某些稀疏化算法需要融合全部kvcache头, 则这个需要跨进程共享
        self.req_layerwise_states: dict[HashType, ReqPerLayerState] = {}  # key用于标识请求及对应层, value是该请求该层的一些稀疏化管理信息
        self.local_tp_rank = vllm_config.parallel_config.rank
        self.total_tp_size = vllm_config.parallel_config.tensor_parallel_size

        if self.role == UcmSparseRole.WORKER:
            # TODO: 进行异步检索模块c lib库的相关init
            ratio = 0.75
            numa_nodes_num, alloc_numa_ids, phy_cpu_core_per_numa = bind_cpus(self.total_tp_size, self.local_tp_rank, ratio=ratio)

            cpu_device = kvstar_retrieve.CPU
            param = kvstar_retrieve.SetupParam(
                cpuNumaIds=alloc_numa_ids,
                physicalCorePerNuma=phy_cpu_core_per_numa,
                allocRatio=ratio,
                blkRepreSize=4096,  # 无效入参
                deviceType=cpu_device,  # 直接传递枚举对象
                totalTpSize=self.total_tp_size,
                localRankId=self.local_tp_rank
            )
            kvstar_retrieve.Setup(param)

        # 独立于vllm常规kvconnector, 创建新uc_dram connector实例用于L/S Sparse KVCache, TODO: 合并同一个worker connector?
        config = {'max_cache_size': 5368709120, 'device': self.local_tp_rank, 'role': 'worker'}
        self.connector = UcmConnectorFactory.create_connector('UcmDram', config)

        self.token_blk_size = vllm_config.cache_config.block_size


    # TODO: 按照SparseBase基类的约束, 分别实现对应的功能

    # ==============================
    # Scheduler/Worker-side 按Role区分的共有逻辑
    # ==============================

    def request_begin(self, request_id: Union[int, str], prompt_token_ids: List[int]):
        """
        This is called at the beginning of "Scheduler->add_request" function.
        """
        pass

    # ==============================
    # Worker-side methods
    # ==============================

    def request_finished_in_worker(self, request_id: ReqType):
        pass

    def bind_sparse_metadata(self, sparse_metadata: UcmSparseMetadata):
        """
        如果对于元数据绑定逻辑有额外需求, 重写该函数, 否则后续删除
        """
        pass

    def clear_sparse_metadata(self):
        """
        如果对于model_execute之后元数据清理逻辑有额外需求, 重写该函数, 否则后续删除
        """
        pass

    def _get_sparse_metadata(self):
        """
        如果对于元数据获取逻辑(仅限在该类实例内调用)有额外需求, 重写该函数, 否则后续删除
        """
        pass

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        稀疏模块识别HBM kvcache空间逻辑
        """
        pass

    def execute_begin(self, scheduler_output: SchedulerOutput):
        """
        在model_execute开始之初执行一些逻辑
        """
        pass

    def execute_finished(self):
        """
        在model_execute结束之前执行一些逻辑
        """
        pass

    def attention_begin(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        layer_name: str, forward_context: ForwardContext) -> None:
        """
        This is called at the beginning of "unified_attention".
        Sparse attention algorithm can modify forward_context.attn_metadata if necessary.
        (UC_TODO: modify dataclass is not allowed in python?)

        Modify forward_context.attn_metadata in-place

        每一次(每层)attention开始前, 包在unified_attention内

        """
        for req_meta in self._sparse_metadata.requests:
            req_layerwise_state_hash = ReqPerLayerState.req_state_hash(req_meta.request_id, layer_name)
            if req_layerwise_state_hash not in self.req_layerwise_states.keys():
                self.req_layerwise_states[req_layerwise_state_hash] = ReqPerLayerState(req_meta, layer_name, self.local_tp_rank, self.total_tp_size, self.connector)
            req_layerwise_state = self.req_layerwise_states[req_layerwise_state_hash]
            req_layerwise_state.update_meta(req_meta, forward_context)  # 重新绑定本次step该请求刷新后的meta
            req_layerwise_state.attention_begin(query, key, value, forward_context)

    def attention_finished(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_output: torch.Tensor,
                           layer_name: str, forward_context: ForwardContext) -> None:
        """
        This is called at the end of "unified_attention".

        每一次(每层)attention结束后, 包在unified_attention内

        比如下一层的预取, kvcache卸载或其他逻辑

        """
        for req_meta in self._sparse_metadata.requests:
            req_layerwise_state_hash = ReqPerLayerState.req_state_hash(req_meta.request_id, layer_name)
            if req_layerwise_state_hash not in self.req_layerwise_states.keys():
                self.req_layerwise_states[req_layerwise_state_hash] = ReqPerLayerState(req_meta, layer_name, self.local_tp_rank, self.total_tp_size, self.connector)
            req_layerwise_state = self.req_layerwise_states[req_layerwise_state_hash]
            req_layerwise_state.update_meta(req_meta, forward_context)  # 重新绑定本次step该请求刷新后的meta
            req_layerwise_state.attention_finished(query, key, value, attn_output, forward_context)  # NOTE: 可以把attention out也传进去


    # ==============================
    # 其他的一些子类worker侧自己的函数
    # ==============================

    def wait_all_task_done(self):  # 一些异步任务等待逻辑 NOTE: 注意区分检索任务和blk传输任务, 这里是multistep策略worker侧整体的wait_all, 注意区分req_layerwise级别的wait_for
        pass

    def build_sparse_meta(self, scheduler_output: SchedulerOutput,
                          requests: Dict[str, CachedRequestState],
                          input_batch: InputBatch) -> None:  # 函数内bind
        """
        Build the sparse metadata for this step.
        目前梳理出的, sparse metadata所需信息基本都可在worker侧获取, 无需scheduler创建传递
        """

        sparse_meta = KVStarMultiStepSparseMetaData()
        '''
        逻辑:
        1. 对于新请求 scheduler_output.scheduled_new_reqs(首次调度或者被打断后重算), 前者UCStore没缓存, 后者有
        2. 对于已计算过的请求(Prefill后, ChunkedPrefill首次后) scheduler_output.scheduled_cached_reqs
        这些请求, 需要在pre/post attention, model, req多个层面的稀疏化操作需要做些什么

        当前build_sparse_meta调用点在self.model forward前, vllm_ascend.worker.npu_input_batch CachedRequestState 已组装好未调度结束的请求的信息, 由此构建sparse_meta
        '''

        for req_id, num_scheduled_tokens in scheduler_output.num_scheduled_tokens.items():  # NOTE: num_scheduled_tokens包含投机token
            req_state = requests[req_id]
            sparse_meta.add_request(
                req_id,
                input_batch.req_id_to_index[req_id],
                len(req_state.prompt_token_ids),
                len(req_state.output_token_ids),  # 当前生成的且验证过的out_token
                num_scheduled_tokens,
                req_state.num_computed_tokens,  # 已经计算过的token长度(即有其kvcache)
                scheduler_output.req_sparsed_slots[req_id],  # 当前给定的slot预算 (num_sparsed_tokens)
                req_state.block_ids[0],  # 当前只支持单种kvcache group, tuple [0] 元素
                self.token_blk_size
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
        num_prefill_keep_fixed_blk = min(INIT_WINDOW_SZ + LOCAL_WINDOW_SZ, num_prefill_fully_block)

        num_sparse_saved_fully_blk = math.ceil((num_prefill_fully_block - num_prefill_keep_fixed_blk) * SPARSE_RATIO)  # same as blk_repre.shape[0] * SPARSE_RATIO

        num_blocks_dense_total = math.ceil(request.num_tokens / block_size)  # 向上取整

        num_blocks_be_compressed_prefill = num_prefill_fully_block - num_sparse_saved_fully_blk - num_prefill_keep_fixed_blk

        num_blocks_this_step_budget = num_blocks_dense_total - num_blocks_be_compressed_prefill

        tail_blk_valid_token_num = request.num_tokens % block_size
        if tail_blk_valid_token_num:
            estimate_num_slots_budget = (num_blocks_this_step_budget - 1) * block_size + tail_blk_valid_token_num
        else:
            estimate_num_slots_budget = num_blocks_this_step_budget * block_size  # 接下来一步会满块, 触发block dump
        return estimate_num_slots_budget


    def update_state_after_alloc(self, request: Request, num_blocks: int):
        """
        Update UcmSparse state after block allocation.
        """
        pass

    def request_finished_in_scheduler(self, request_id: ReqType):
        pass
