import math
import time
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Sequence, Union

import torch
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.sequence import SequenceStage
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from unifiedcache.integration.vllm.ucm_sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
)
from unifiedcache.ucm_connector.base import Task, UcmKVStoreBase
from unifiedcache.ucm_connector.factory import UcmConnectorFactory


def stat(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()
        cost = end - start
        wrapper.time_costs.append(cost)
        return result

    wrapper.call_count = 0
    wrapper.time_costs = []
    return wrapper


ReqType = Union[str, int]
HashType = Union[str, int]

# TODO: add ESA specific config in kv_transfer_config -> extra_config
INIT_WINDOW_SZ = 1
LOCAL_WINDOW_SZ = 2
SPARSE_RATIO = 0.3
RETRIEVAL_STRIDE = 4


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

    @property
    def step(self) -> int:
        return self.num_output_tokens

    @property
    def stage(self) -> SequenceStage:
        return (
            SequenceStage.DECODE
            if self.num_output_tokens > 0
            else SequenceStage.PREFILL
        )

    @property
    def is_last_chunk(self) -> bool:
        return (
            self.num_computed_tokens + self.num_scheduled_tokens
            >= self.num_prompt_tokens
        )


@dataclass
class ESASparseMetaData(UcmSparseMetadata):
    requests: list[ReqMeta]
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
        )
        self.requests.append(meta)


def get_offset(block_shape, rank, tp_size, precision, layer_id, is_v, is_mla) -> int:
    block_size, num_key_heads_per_tp, head_size = block_shape
    k_min_data_block_size = block_size * num_key_heads_per_tp * head_size * precision
    v_min_data_block_size = k_min_data_block_size if not is_mla else 0
    layer_size = (k_min_data_block_size + v_min_data_block_size) * tp_size
    if is_mla:
        k_offset = layer_size * layer_id
    else:
        k_offset = layer_size * layer_id + layer_size // tp_size * rank
    v_offset = k_offset + k_min_data_block_size
    return v_offset if is_v else k_offset


class ReqStatePerLayer:
    # handle single request per layer

    def __init__(
        self,
        req_meta: ReqMeta,
        layer_name: str,
        rank: int,
        tp_size: int,
        store_instance: UcmKVStoreBase,
    ):
        self.layer_name = layer_name
        self.layer_id = int(layer_name.split(".")[2])
        self.block_repre: torch.Tensor = (
            None  ## shape: blks, num_key_heads_per_tp, head_size
        )
        self.init_window: tuple[torch.Tensor, torch.Tensor] = None
        self.local_window: tuple[torch.Tensor, torch.Tensor] = None
        self.store_instance = store_instance
        self.req_meta = req_meta
        self.block_size = None
        self.k_cache = None
        self.v_cache = None
        self.rank = rank
        self.tp_size = tp_size
        self.tasks: Dict[str, Task] = {}
        self.init_window_sz = INIT_WINDOW_SZ
        self.local_window_sz = LOCAL_WINDOW_SZ

    @classmethod
    def req_state_hash(cls, req_id, layer_name):
        return hash((req_id, layer_name))

    @classmethod
    def block_hash(cls, request_id, block_id):
        return f"req_{request_id}_blk_{block_id}"

    @classmethod
    def task_hash(cls, block_ids, store_type, tensor_type):
        return hash((tuple(block_ids), store_type, tensor_type))

    def update_meta(self, req_meta: ReqMeta, forward_context: ForwardContext):
        self.req_meta = req_meta

    def retrieval(self, query: torch.Tensor, top_k: int):
        if top_k >= self.block_repre.shape[0]:
            n_blocks = self.block_repre.shape[0]
            block_ids = list(
                range(self.init_window_sz, n_blocks - self.local_window_sz + 1)
            )
            block_hashes = [
                f"{self.block_hash(self.req_meta.request_id, id_)}" for id_ in block_ids
            ]
            return block_hashes
        ntokens, num_q_heads, _ = query.shape
        if num_q_heads > self.num_key_heads:
            query = query.view(ntokens, self.num_key_heads, -1, self.head_size)
            query = query.mean(2)
        elif num_q_heads < self.num_key_heads:
            query = torch.repeat_interleave(query, self.num_key_heads // num_q_heads, 1)

        retrieval_start = self.init_window_sz
        retrieval_end = self.block_repre.shape[0] - self.local_window_sz + 1
        block_repre_ = self.block_repre[retrieval_start:retrieval_end]

        scores = torch.einsum("qnd,knd->nqk", query, block_repre_)
        scores = scores.softmax(-1)
        scores = scores.sum(0).sum(0)
        topk_ret = torch.topk(scores, top_k)
        topk_index = topk_ret.indices
        topk_index = (
            topk_index.sort().values
        )  # TODO: remove this, don't need to sort in decode
        block_ids = [id.item() + self.init_window_sz for id in topk_index]
        block_hashes = [
            f"{self.block_hash(self.req_meta.request_id, id_)}" for id_ in block_ids
        ]
        return block_hashes

    def construct_init_and_local_window(self):
        vllm_block_ids = self.req_meta.vllm_block_ids
        # TODO: make sure we don't need to clone()
        self.init_window = (
            self.k_cache[vllm_block_ids[: self.init_window_sz]],
            self.v_cache[vllm_block_ids[: self.init_window_sz]],
        )
        local_window_sz = min(
            self.local_window_sz, len(vllm_block_ids[self.init_window_sz :])
        )
        if local_window_sz > 0:
            self.local_window = (
                self.k_cache[vllm_block_ids[-local_window_sz:]],
                self.v_cache[vllm_block_ids[-local_window_sz:]],
            )

    def launch_transfer_task(self, transfer_type, block_hashes, vllm_block_ids):
        fn = getattr(self.store_instance, transfer_type)
        length = len(block_hashes)
        block_shape = (self.block_size, self.num_key_heads, self.head_size)
        precision = self.k_cache.untyped_storage().element_size()
        # TODO: consider is_mla here
        is_mla = False
        offsets_k = [
            get_offset(
                block_shape,
                self.rank,
                self.tp_size,
                precision,
                self.layer_id,
                is_v=False,
                is_mla=is_mla,
            )
        ] * length
        offsets_v = [
            get_offset(
                block_shape,
                self.rank,
                self.tp_size,
                precision,
                self.layer_id,
                is_v=True,
                is_mla=is_mla,
            )
        ] * length
        key_src_tensors = [self.k_cache[id_] for id_ in vllm_block_ids]
        value_src_tensors = [self.v_cache[id_] for id_ in vllm_block_ids]
        task_k = fn(block_hashes, offsets_k, key_src_tensors)
        task_v = fn(block_hashes, offsets_v, value_src_tensors)
        task_k_hash = self.task_hash(block_hashes, transfer_type, "key")
        self.tasks[task_k_hash] = task_k
        task_v_hash = self.task_hash(block_hashes, transfer_type, "value")
        self.tasks[task_v_hash] = task_v

    def extract_block_repre(self, vllm_block_ids):
        return self.k_cache[vllm_block_ids].mean(1)

    def save_blocks(self, num_blocks_need_dump):
        if num_blocks_need_dump <= 0:
            return
        vllm_block_ids = self.req_meta.vllm_block_ids
        num_blocks_dumped = 0 if self.block_repre is None else self.block_repre.shape[0]
        block_ids = list(
            range(num_blocks_dumped, num_blocks_dumped + num_blocks_need_dump)
        )
        block_hashes = [
            f"{self.block_hash(self.req_meta.request_id, id_)}" for id_ in block_ids
        ]
        if self.req_meta.stage == SequenceStage.PREFILL:
            vllm_block_ids_dump = vllm_block_ids[
                num_blocks_dumped : num_blocks_dumped + num_blocks_need_dump
            ]
        else:
            # TODO: handle spec_decode here
            vllm_block_ids_dump = vllm_block_ids[-1:]
        self.launch_transfer_task("dump", block_hashes, vllm_block_ids_dump)
        repre = self.extract_block_repre(vllm_block_ids_dump)
        # TODO: pre-allocate can speed up here
        if self.block_repre is None:
            self.block_repre = repre
        else:
            self.block_repre = torch.cat([self.block_repre, repre])

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

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        forward_context: ForwardContext,
    ) -> None:
        if self.req_meta.step % RETRIEVAL_STRIDE != 1:
            return
        index_in_batch = self.req_meta.index_in_batch
        if isinstance(forward_context.attn_metadata, dict):
            attn_md = forward_context.attn_metadata[self.layer_name]
        else:
            attn_md = forward_context.attn_metadata
        query_start_loc = attn_md.query_start_loc[index_in_batch]
        query_len = self.req_meta.num_scheduled_tokens
        current_query = query[query_start_loc : query_start_loc + query_len]

        vllm_block_ids = self.req_meta.vllm_block_ids[
            self.init_window_sz : -self.local_window_sz
        ]
        self.wait_for_task_done()
        self.prepare_init_and_local_window()  # last dump task(possible)
        # NOTE: sync style
        topk_block_hashes = self.retrieval(current_query, len(vllm_block_ids))
        self.launch_transfer_task("load", topk_block_hashes, vllm_block_ids)

        self.wait_for_task_done()

        # NOTE: Some sparse attention algorithms need to modify attn_metadata here

    def prepare_init_and_local_window(self):
        if self.req_meta.step != 1:
            return

        vllm_block_ids = self.req_meta.vllm_block_ids
        self.k_cache[vllm_block_ids[: self.init_window_sz]] = self.init_window[0]
        self.v_cache[vllm_block_ids[: self.init_window_sz]] = self.init_window[1]

        if self.local_window is None:
            return

        self.k_cache[vllm_block_ids[-self.local_window_sz :]] = self.local_window[0]
        self.v_cache[vllm_block_ids[-self.local_window_sz :]] = self.local_window[1]

    def wait_for_task_done(self):
        for task_hash, task in self.tasks.items():
            # TODO: handle exceptions here, refer to UcmKVConnector
            ret = self.store_instance.wait(task)
        self.tasks.clear()

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        forward_context: ForwardContext,
    ) -> None:
        self.maybe_register_kv_cache(forward_context)
        num_tokens_updated = (
            self.req_meta.num_computed_tokens + self.req_meta.num_scheduled_tokens
        )
        num_blocks_dumped = 0 if self.block_repre is None else self.block_repre.shape[0]
        num_full_blocks = num_tokens_updated // self.block_size
        num_blocks_need_dump = num_full_blocks - num_blocks_dumped
        self.save_blocks(num_blocks_need_dump)
        if self.req_meta.stage == SequenceStage.PREFILL and self.req_meta.is_last_chunk:
            self.construct_init_and_local_window()
            self.wait_for_task_done()


class ESA(UcmSparseBase):
    # handle batch
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        self.req_states: dict[str, ReqStatePerLayer] = {}
        self.rank = vllm_config.parallel_config.rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.block_size = vllm_config.cache_config.block_size
        config = {"max_cache_size": 5368709120, "device": self.rank, "role": "worker"}
        self.connector = UcmConnectorFactory.create_connector("UcmDram", config)
        # TODO: consider init self.is_mla here

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
    ) -> None:
        for req_meta in self._sparse_metadata.requests:
            req_state_hash = ReqStatePerLayer.req_state_hash(
                req_meta.request_id, layer_name
            )
            if req_state_hash not in self.req_states:
                self.req_states[req_state_hash] = ReqStatePerLayer(
                    req_meta, layer_name, self.rank, self.tp_size, self.connector
                )
            req_state = self.req_states[req_state_hash]
            req_state.update_meta(req_meta, forward_context)
            req_state.attention_begin(query, key, value, forward_context)

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
    ) -> None:
        for req_meta in self._sparse_metadata.requests:
            req_state_hash = ReqStatePerLayer.req_state_hash(
                req_meta.request_id, layer_name
            )
            if req_state_hash not in self.req_states:
                self.req_states[req_state_hash] = ReqStatePerLayer(
                    req_meta, layer_name, self.rank, self.tp_size, self.connector
                )
            req_state = self.req_states[req_state_hash]
            req_state.update_meta(req_meta, forward_context)
            req_state.attention_finished(
                query, key, value, attn_output, forward_context
            )

    def wait_all_task_done(self):
        pass

    def execute_finished(self):
        pass

    def execute_finished(self):
        pass

    def build_sparse_meta(
        self,
        scheduler_output,
        requests,
        input_batch,
    ) -> UcmSparseMetadata:
        sparse_meta = ESASparseMetaData()
        for (
            req_id,
            num_scheduled_tokens,
        ) in scheduler_output.num_scheduled_tokens.items():
            req_state = requests[req_id]
            if len(req_state.prompt_token_ids) > self.block_size:
                sparse_meta.add_request(
                    req_id,
                    input_batch.req_id_to_index[req_id],
                    len(req_state.prompt_token_ids),
                    len(req_state.output_token_ids),
                    num_scheduled_tokens,
                    req_state.num_computed_tokens,
                    scheduler_output.req_sparsed_slots[req_id],
                    req_state.block_ids[0],
                )
        self._sparse_metadata = sparse_meta

    def request_begin(self, request_id: ReqType, prompt_token_ids: List[int]):
        pass

    def request_finished_in_scheduler(self, request_id: ReqType):
        pass

    def request_finished_in_worker(self, request_id: ReqType):
        pass

    def update_state_after_alloc(self, request: Request, num_blocks: int):
        pass

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        if (
            request.num_output_tokens == 0
            or request.num_prompt_tokens < self.block_size
        ):
            return INVALID_SLOT
        num_blocks = math.ceil(request.num_tokens / self.block_size)
        mid_window_sz = int(
            (num_blocks - INIT_WINDOW_SZ - LOCAL_WINDOW_SZ) * SPARSE_RATIO
        )
        flaw = request.num_tokens % self.block_size
        if flaw:
            flaw = self.block_size - flaw
        num_tokens_sparsed = (
            INIT_WINDOW_SZ + mid_window_sz + LOCAL_WINDOW_SZ
        ) * self.block_size - flaw
        return num_tokens_sparsed
