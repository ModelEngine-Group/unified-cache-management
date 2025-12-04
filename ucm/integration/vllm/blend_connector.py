import hashlib
import itertools
import os
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Self, Tuple

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from ucm.integration.vllm.ucm_connector import (
    RequestDispatchMeta,
    RequestHasher,
    RequestMeta,
    UCMConnectorMetadata,
    UCMDirectConnector,
)
from ucm.logger import init_logger
from ucm.shared.metrics import ucmmonitor
from ucm.shared.metrics.observability import UCMStatsLogger
from ucm.sparse.blend.blockwise_rope import block_wise_rope_forward
from ucm.sparse.kvstar.multistep import ReqStage
from ucm.store.factory import UcmConnectorFactory
from ucm.store.ucmstore import Task, UcmKVStoreBase
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

logger = init_logger(__name__)


@dataclass
class ChunkMetaData:
    # [start, start + len)
    start_idx_in_req: int
    chunk_tokens_len: int

    start_idx_in_req_blks: int
    chunk_blks_len: int

    cached_start_position: int

    vllm_blk_ids: List[int] = field(default_factory=list)
    chunk_blks_hash: List[str] = field(default_factory=list)
    store_hits: List[bool] = field(default_factory=list)

    @property
    def end_idx_in_req(self) -> int:
        return self.start_idx_in_req + self.chunk_tokens_len

    @property
    def end_idx_in_req_blks(self) -> int:
        return self.start_idx_in_req_blks + self.chunk_blks_len

    @property
    def cached_end_position(self) -> int:
        return self.cached_start_position + self.chunk_tokens_len

    @property
    def position_offset(self) -> int:
        return self.start_idx_in_req - self.cached_start_position

    @property
    def hits_vllm_blk_ids(self) -> List[int]:
        return list(itertools.compress(self.vllm_blk_ids, self.store_hits))

    @property
    def hits_chunk_blks_hash(self) -> List[str]:
        return list(itertools.compress(self.chunk_blks_hash, self.store_hits))

    def merge_chunk(self, temp_chunk_meta: Self):
        # current we use a fix pattern(end with a fix token id) to recognize the text token chunk
        # in some special situation, one text chunk maybe split as multi text chunk, so we should merge them into one
        self.chunk_tokens_len += temp_chunk_meta.chunk_tokens_len
        self.chunk_blks_len += temp_chunk_meta.chunk_blks_len
        self.chunk_blks_hash += temp_chunk_meta.chunk_blks_hash

    def update_meta_partial_pc(self, num_pc_part_blks: int, block_size: int) -> None:
        self.start_idx_in_req += num_pc_part_blks * block_size
        self.chunk_tokens_len -= num_pc_part_blks * block_size

        self.start_idx_in_req_blks += num_pc_part_blks
        self.chunk_blks_len -= num_pc_part_blks

        self.chunk_blks_hash = self.chunk_blks_hash[num_pc_part_blks:]
        self.store_hits = self.store_hits[num_pc_part_blks:]
        self.cached_start_position += num_pc_part_blks * block_size


class BlendStage(Enum):
    BUILD_CHUNK_CACHE = auto()
    BUILD_PREFIX_CACHE = auto()
    CACHE_BLEND = auto()

    def is_blend_cache(self):
        return self == BlendStage.CACHE_BLEND

    def is_prefix_cache(self):
        return self == BlendStage.BUILD_PREFIX_CACHE


@dataclass
class BlendRequestMeta:
    ucm_block_hashs: list[str] = field(default_factory=list)
    # hbm pc is not supported
    hbm_hit_block_num: int = 0
    # ucm pc is supported
    pc_hit_block_num: int = 0
    chunks_meta: List[ChunkMetaData] = field(default_factory=list)
    blend_stage: BlendStage = BlendStage.BUILD_PREFIX_CACHE


@dataclass
class BlendRequestDispatchMeta(RequestDispatchMeta):
    chunks_meta: List[ChunkMetaData]


@dataclass
class UCMBlendConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, BlendRequestDispatchMeta] = field(default_factory=dict)


class UCMBlendConnector(UCMDirectConnector):
    """
    This Connector means overlap:
    load l0 -> forward l0 -> save l0
               load l1    -> forward l1 -> save l1
                             load l2    -> forward l2 -> save l2
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)
        ucm_sparse_config = self.launch_config.get("ucm_sparse_config", [])
        self.blend_stage = BlendStage.BUILD_PREFIX_CACHE
        self.req2rag_load_chunks: dict[str, list[ChunkMetaData]] = {}
        if "Blend" in ucm_sparse_config:
            blend_config = ucm_sparse_config["Blend"]
            self.enable_blend = True
            self.chunk_end_token_id = blend_config["chunk_end_token_id"]
        else:
            raise "UCMBlendConnector init failed, please check your config"

        self.ucm_chunk_end_hash: int = self.request_hasher("UCM_CHUNK_END_HASH")
        self.ucm_chunk_continue_hash: int = self.request_hasher(
            "UCM_CHUNK_CONTINUE_HASH"
        )
        self.requests_blend_meta: dict[str, BlendRequestMeta] = {}
        self.cos_sin_cache: torch.Tensor = None

    def _generate_hash(self, block_size: int, token_ids: list[int]) -> list[str]:

        ret = []
        parent_block_hash_value = RequestHasher._SEED_HASH
        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            block_token_ids = token_ids[start:end]
            # Do not hash the block if it is not full.
            if len(block_token_ids) < block_size:
                break

            block_token_ids_tuple = tuple(block_token_ids)
            hash_value = self.request_hasher(
                (parent_block_hash_value, block_token_ids_tuple)
            )
            parent_block_hash_value = hash_value
            ret.append(str(hash_value))

        return ret

    def _generate_chunk_hash(
        self,
        block_size: int,
        token_ids: list[int],
        extra_end_hash: list[int] = None,
        parent_block_hash_value: int = RequestHasher._SEED_HASH,
    ) -> Tuple[list[str], list[int]]:
        assert len(token_ids) % block_size == 0
        ret = []
        hash_value = 0
        continue_hash_value = 0
        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            block_token_ids = token_ids[start:end]
            # Do not hash the block if it is not full.
            if len(block_token_ids) < block_size:
                break

            block_token_ids_tuple = tuple(block_token_ids)
            hash_value = self.request_hasher(
                (parent_block_hash_value, block_token_ids_tuple)
            )
            if end == len(token_ids) and extra_end_hash is not None:
                pc_hash_value = hash_value
                # add an end tag to stop further match when in BUILD CACHE stage
                hash_value = self.request_hasher((pc_hash_value, extra_end_hash[0]))
                for i in range(1, len(extra_end_hash)):
                    continue_hash_value = self.request_hasher(
                        (pc_hash_value, extra_end_hash[1])
                    )
            parent_block_hash_value = hash_value
            ret.append(str(hash_value))

        return ret, [hash_value, continue_hash_value]

    def _get_req_stage(self, all_prefill_tokens):

        if (
            all_prefill_tokens[-1] == self.chunk_end_token_id
            and len(all_prefill_tokens) % self.block_size == 0
        ):
            return BlendStage.BUILD_CHUNK_CACHE, all_prefill_tokens

        start_blk_idx = 0
        start_token_dix = 0
        candidate_chunks = []
        for end_blk_idx, end_token_idx in enumerate(
            range(self.block_size - 1, len(all_prefill_tokens), self.block_size)
        ):
            if all_prefill_tokens[end_token_idx] == self.chunk_end_token_id:
                chunk_token_ids = all_prefill_tokens[
                    start_token_dix : end_token_idx + 1
                ]
                candidate_chunks.append(chunk_token_ids)
        return BlendStage.CACHE_BLEND

    def _process_req(self, all_token_ids):
        """
        pre-assumption, we explicitly construct block-padded chunk req to make it cached all tokens
        beside chunk-build req, we try to split chunk from req, if no chunk exist, it just builds naive prefix cache
        if chunk found, first we should match the prefix cache as much as possible, cause, they can be fully reused
        then for other chunk blocks, if store hit num of block hash is less than threshold, we do not conduct cache blend
        finally, if there are quite many chunk block-hits, we do cache blend to get TTFT-promot
        """
        chunks_meta = []
        req_stage = BlendStage.CACHE_BLEND
        if (
            all_token_ids[-1] == self.chunk_end_token_id
            and len(all_token_ids) % self.block_size == 0
        ):
            req_stage = BlendStage.BUILD_CHUNK_CACHE

        start_blk_idx = 0
        start_token_dix = 0
        req_block_hashes = []

        next_block_hash_value = RequestHasher._SEED_HASH

        for end_blk_idx, end_token_idx in enumerate(
            range(self.block_size - 1, len(all_token_ids), self.block_size)
        ):
            chunk_start_hash = next_block_hash_value
            # only compare the last token id in each blk
            if all_token_ids[end_token_idx] == self.chunk_end_token_id:
                chunk_token_ids = all_token_ids[start_token_dix : end_token_idx + 1]
                if req_stage == BlendStage.BUILD_CHUNK_CACHE:
                    if end_token_idx == len(all_token_ids) - 1:
                        # last chunk
                        chunk_blks_hash, _ = self._generate_chunk_hash(
                            self.block_size,
                            chunk_token_ids,
                            extra_end_hash=[self.ucm_chunk_end_hash],
                            parent_block_hash_value=next_block_hash_value,
                        )
                    else:
                        chunk_blks_hash, [next_block_hash_value, _] = (
                            self._generate_chunk_hash(
                                self.block_size,
                                chunk_token_ids,
                                extra_end_hash=[self.ucm_chunk_continue_hash],
                                parent_block_hash_value=next_block_hash_value,
                            )
                        )
                else:
                    chunk_blks_hash, [next_block_hash_value, last_block_hash_value] = (
                        self._generate_chunk_hash(
                            self.block_size,
                            chunk_token_ids,
                            extra_end_hash=[
                                self.ucm_chunk_continue_hash,
                                self.ucm_chunk_end_hash,
                            ],
                            parent_block_hash_value=next_block_hash_value,
                        )
                    )
                    lookup_result = self.store.lookup([str(next_block_hash_value)])
                    if lookup_result[0]:
                        # continue to build chunk
                        pass
                    else:
                        # end build chunk
                        chunk_blks_hash[-1] = str(last_block_hash_value)

                        # reset next chunk start hash
                        next_block_hash_value = RequestHasher._SEED_HASH

                chunk_blks_len = end_blk_idx - start_blk_idx + 1
                chunk_tokens_len = chunk_blks_len * self.block_size

                rag_chunk_meta = ChunkMetaData(
                    start_idx_in_req=start_token_dix,
                    chunk_tokens_len=chunk_tokens_len,
                    start_idx_in_req_blks=start_blk_idx,
                    chunk_blks_len=chunk_blks_len,
                    chunk_blks_hash=chunk_blks_hash,
                    cached_start_position=0,
                )

                # update for next rag chunk
                start_blk_idx = end_blk_idx + 1
                start_token_dix = end_token_idx + 1

                chunks_meta.append(rag_chunk_meta)
                req_block_hashes.extend(chunk_blks_hash)

                if chunk_start_hash != RequestHasher._SEED_HASH:
                    # merge the last two chunk
                    temp_chunk_meta = chunks_meta.pop()
                    chunks_meta[-1].merge_chunk(temp_chunk_meta)

        if req_stage == BlendStage.BUILD_CHUNK_CACHE:
            return req_stage, req_block_hashes, chunks_meta

        if chunks_meta:
            # found chunk, as for suffix part(such as user question about chunk), current no need to cache hit and dump
            return BlendStage.CACHE_BLEND, req_block_hashes, chunks_meta
        else:
            return (
                BlendStage.BUILD_PREFIX_CACHE,
                self._generate_hash(self.block_size, all_token_ids),
                chunks_meta,
            )

    def _get_req_chunk_hit(
        self,
        req_stage: BlendStage,
        req_block_hashes: List[str],
        req_chunks_meta: List[ChunkMetaData],
    ):

        lookup_results = self.store.lookup(req_block_hashes)
        pc_hit_blocks = 0
        chunk_hit_blocks = 0

        if req_stage.is_prefix_cache():
            for i, hit in enumerate(lookup_results):
                if not hit:
                    break
                pc_hit_blocks += 1
            return pc_hit_blocks, sum(lookup_results)

        # for chunk cache
        for i, chunk_meta in enumerate(req_chunks_meta):
            chunk_meta.store_hits = lookup_results[
                chunk_meta.start_idx_in_req_blks : chunk_meta.end_idx_in_req_blks
            ]

        # the first chunk may be fully reused
        for i, hit in enumerate(req_chunks_meta[0].store_hits):
            if not hit:
                break
            pc_hit_blocks += 1
        req_chunks_meta[0].update_meta_partial_pc(pc_hit_blocks, self.block_size)
        if req_chunks_meta[0].chunk_tokens_len == 0:
            req_chunks_meta.pop(0)

        return pc_hit_blocks, sum(lookup_results)

    def _generate_blend_dispatch_meta(
        self,
        req_meta: BlendRequestMeta,
        new_tokens: int,
        vllm_block_ids: list[int],
    ) -> BlendRequestDispatchMeta:
        """
        Request Blocks layout:
        Stage: Build Prefix Cache or Build Chunk Cache (max one chunk per req)
        ----------------------------------------------------------------------------------------------------------
        | prefix cache (at first chunk) | other chunk cache      |
        ----------------------------------------------------------------------------------------------------------
        |            LOAD               |          DUMP          |
        ----------------------------------------------------------------------------------------------------------
        |           REUSE               |    RECOMPUTE           |
        ----------------------------------------------------------------------------------------------------------


        Stage: Cache Blend
        ----------------------------------------------------------------------------------------------------------
        | prefix cache at first chunk | other chunk cache hit  | other chunk cache miss | suffix part(question) |
        ----------------------------------------------------------------------------------------------------------
        |            LOAD             |          LOAD          |    NO NEED TO DUMP    |     NO NEED TO DUMP    |
        ----------------------------------------------------------------------------------------------------------
        |           REUSE             |   REUSE & RECOMPUTE    |       RECOMPUTE       |        RECOMPUTE       |
        ----------------------------------------------------------------------------------------------------------

        """

        # current not support chunk prefill, cause the topK high deviation KV should come from the all tokens
        pc_hit_block_num = req_meta.pc_hit_block_num
        ucm_block_hashs = req_meta.ucm_block_hashs
        # load prefix part
        load_ucm_block_ids, load_vllm_block_ids = (
            ucm_block_hashs[:pc_hit_block_num],
            vllm_block_ids[:pc_hit_block_num],
        )
        dump_ucm_block_ids, dump_vllm_block_ids = [], []

        if req_meta.blend_stage.is_blend_cache():
            # just need to load, in future we may create a multi-chunk hash to dump and reuse the blended cache
            for chunk_meta in req_meta.chunks_meta:
                chunk_meta.vllm_blk_ids = vllm_block_ids[
                    chunk_meta.start_idx_in_req_blks : chunk_meta.end_idx_in_req_blks
                ]
                load_ucm_block_ids.extend(chunk_meta.hits_chunk_blks_hash)
                load_vllm_block_ids.extend(chunk_meta.hits_vllm_blk_ids)
            return BlendRequestDispatchMeta(
                (load_ucm_block_ids, load_vllm_block_ids),
                (dump_ucm_block_ids, dump_vllm_block_ids),
                req_meta.chunks_meta,
            )

        # build cache stage
        dump_ucm_block_ids, dump_vllm_block_ids = (
            ucm_block_hashs[pc_hit_block_num:],
            vllm_block_ids[pc_hit_block_num : len(ucm_block_hashs)],
        )
        return BlendRequestDispatchMeta(
            (load_ucm_block_ids, load_vllm_block_ids),
            (dump_ucm_block_ids, dump_vllm_block_ids),
            req_meta.chunks_meta,
        )

    def _post_process_chunk_cache(self, k_cache, vllm_ids, positions):
        """
        post process loaded chunk kcache
        """
        if self.cos_sin_cache is None:
            raise "Please call setup model first."
        # triton kernl for block-wise delta rope
        block_wise_rope_forward(k_cache, vllm_ids, positions, self.cos_sin_cache)

    def _register_cos_sin_cache(self, model: "Model"):
        try:
            rotary_emb = model.model.layers[0].self_attn.rotary_emb
            self.cos_sin_cache = rotary_emb.cos_sin_cache
        except Exception:
            raise "get cos_sin_cache from model failed!  current not implemented for this model"

    def setup_model(self, model: "Model") -> None:
        self._register_cos_sin_cache(model)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:

        # current not support HBM prefix cache, cause the blended cached have a ground view of all chunks
        # so they can not apply to other req
        assert num_computed_tokens == 0
        all_token_ids = request.all_token_ids

        max_blk_num = len(all_token_ids) // self.block_size

        if max_blk_num == 0:
            return 0, False

        req_stage, req_block_hashes, req_chunks_meta = self._process_req(all_token_ids)

        pc_hit_blocks, chunk_hit_blocks = self._get_req_chunk_hit(
            req_stage, req_block_hashes, req_chunks_meta
        )

        logger.info(
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {max_blk_num}, "
            f"first chunk prefix hit: {pc_hit_blocks}, "
            f"chunks cache total hit: {chunk_hit_blocks}, "
            f"need cache blend block num: {chunk_hit_blocks - pc_hit_blocks}, "
        )
        if self.metrics_config:
            self.monitor.update_stats(
                "ConnStats",
                {"interval_lookup_hit_rates": chunk_hit_blocks / max_blk_num},
            )

        pc_hit_tokens = pc_hit_blocks * self.block_size

        # When all the tokens are cached in ssd or hbm,
        # we need to recompute the last token. This if condition will be removed
        # once vLLM scheduler provides a better solution in the future.
        if pc_hit_tokens == request.num_tokens:
            pc_hit_tokens -= 1

        self.requests_blend_meta[request.request_id] = BlendRequestMeta(
            ucm_block_hashs=req_block_hashes,
            pc_hit_block_num=pc_hit_blocks,
            chunks_meta=req_chunks_meta,
            blend_stage=req_stage,
        )

        return pc_hit_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        pass

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        requests_dispatch_meta = {}
        # for new request, we need to load and dump
        for request in scheduler_output.scheduled_new_reqs:
            request_id, vllm_block_ids = request.req_id, request.block_ids[0]
            req_meta = self.requests_blend_meta.get(request_id)
            if req_meta:
                requests_dispatch_meta[request_id] = self._generate_blend_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    vllm_block_ids,
                )

        # for cached request, there are 3 situation:
        # 1. chunked prefill: we should make sure this will not happen
        # 2. resumed: we need to handle like new request
        # 3. TODO decode stage: nothing happened
        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        if not isinstance(scheduled_cached_reqs, list):
            # >= 0.9.2
            for i, request_id in enumerate(scheduled_cached_reqs.req_ids):
                if scheduler_output.num_scheduled_tokens[request_id] == 1:
                    # decode stage
                    continue
                req_meta = self.requests_blend_meta.get(request_id)
                if req_meta:
                    requests_dispatch_meta[request_id] = (
                        self._generate_blend_dispatch_meta(
                            req_meta,
                            scheduler_output.num_scheduled_tokens[request_id],
                            scheduled_cached_reqs.new_block_ids[i][0],
                        )
                    )
        else:
            for request in scheduled_cached_reqs:
                request_id = request.request_id
                if scheduler_output.num_scheduled_tokens[request_id] == 1:
                    # decode stage
                    continue
                req_meta = self.requests_blend_meta.get(request_id)
                if req_meta:
                    requests_dispatch_meta[request_id] = (
                        self._generate_blend_dispatch_meta(
                            req_meta,
                            scheduler_output.num_scheduled_tokens[request_id],
                            request.new_block_ids[0],
                        )
                    )

        # clear finished request
        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMBlendConnectorMetadata(requests_dispatch_meta)

    def wait_for_layer_load(self, layer_name: str) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMBlendConnectorMetadata)

        all_hits_vllm_ids = []
        positions = []
        k_cache = self.kv_caches[layer_name][0]
        for request_id, request in metadata.request_meta.items():
            for chunk_meta in request.chunks_meta:
                all_hits_vllm_ids.extend(chunk_meta.hits_vllm_blk_ids)
                positions.extend(
                    [chunk_meta.position_offset] * len(chunk_meta.hits_vllm_blk_ids)
                )
        if all_hits_vllm_ids:
            vllm_ids = torch.tensor(all_hits_vllm_ids, device=k_cache.device)
            positions = torch.tensor(positions, device=k_cache.device)
            self._post_process_chunk_cache(k_cache, vllm_ids, positions)
        pass
