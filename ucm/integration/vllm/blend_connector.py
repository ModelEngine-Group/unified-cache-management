import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Self, Tuple

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from ucm.integration.vllm.ucm_connector import (
    RequestDispatchMeta,
    UCMConnectorMetadata,
    UCMDirectConnector,
)
from ucm.logger import init_logger
from ucm.shared.metrics import ucmmonitor
from ucm.sparse.blend.blockwise_rope import block_wise_rope_forward

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

logger = init_logger(__name__)


@dataclass
class ChunkMetaData:
    # [start, start + len)
    start_token_dix: int
    chunk_tokens_len: int

    start_blk_idx: int
    chunk_blks_len: int

    cached_start_position: int

    vllm_blk_ids: List[int] = field(default_factory=list)
    chunk_blks_hash: List[bytes] = field(default_factory=list)
    store_hits: List[bool] = field(default_factory=list)

    @property
    def end_token_dix(self) -> int:
        return self.start_token_dix + self.chunk_tokens_len

    @property
    def end_blk_idx(self) -> int:
        return self.start_blk_idx + self.chunk_blks_len

    @property
    def cached_end_position(self) -> int:
        return self.cached_start_position + self.chunk_tokens_len

    @property
    def position_offset(self) -> int:
        return self.start_token_dix - self.cached_start_position

    @property
    def hits_vllm_blk_ids(self) -> List[int]:
        return list(itertools.compress(self.vllm_blk_ids, self.store_hits))

    @property
    def hits_chunk_blks_hash(self) -> List[bytes]:
        return list(itertools.compress(self.chunk_blks_hash, self.store_hits))

    def merge_chunk(self, temp_chunk_meta: Self) -> None:
        # current we use a fix pattern(end with a fix token id) to recognize the text token chunk
        # in some special situation, one text chunk maybe split as multi text chunk, so we should merge them into one
        self.chunk_tokens_len += temp_chunk_meta.chunk_tokens_len
        self.chunk_blks_len += temp_chunk_meta.chunk_blks_len
        self.chunk_blks_hash += temp_chunk_meta.chunk_blks_hash

    def update_meta_partial_pc(self, num_pc_part_blks: int, block_size: int) -> None:
        if num_pc_part_blks > 0:
            self.start_token_dix += num_pc_part_blks * block_size
            self.chunk_tokens_len -= num_pc_part_blks * block_size

            self.start_blk_idx += num_pc_part_blks
            self.chunk_blks_len -= num_pc_part_blks

            self.chunk_blks_hash = self.chunk_blks_hash[num_pc_part_blks:]
            self.store_hits = self.store_hits[num_pc_part_blks:]
            self.cached_start_position += num_pc_part_blks * block_size


class BlendStage(Enum):
    BUILD_CHUNK_CACHE = auto()
    BUILD_PREFIX_CACHE = auto()
    CACHE_BLEND = auto()

    def is_blend_cache(self) -> bool:
        return self == BlendStage.CACHE_BLEND

    def is_prefix_cache(self) -> bool:
        return self == BlendStage.BUILD_PREFIX_CACHE


@dataclass
class BlendRequestMeta:
    ucm_block_hashs: list[bytes] = field(default_factory=list)
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
class UCMBlendConnectorMetadata(UCMConnectorMetadata):
    request_meta: dict[str, BlendRequestDispatchMeta] = field(default_factory=dict)


class UCMBlendConnector(UCMDirectConnector):
    """
    This Connector process chunk hash and prefix cache
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

        self.requests_blend_meta: dict[str, BlendRequestMeta] = {}
        self.cos_sin_cache: torch.Tensor = None

        # if chunk cache hits less than min_blend_threshold, no need to cache blend
        self.min_blend_threshold = 16

        # post process delta rope meta
        self.delta_rope_vllm_ids: torch.Tensor = None
        self.delta_rope_positions: torch.Tensor = None

    def _process_req(self, all_token_ids: List[int]):
        """
        pre-assumption, we explicitly construct block-padded chunk req to make it cached all tokens
        beside chunk-build req, we try to split chunk from req, if no chunk exist, it just builds naive prefix cache
        if chunk found, first we should match the prefix cache as much as possible, cause, they can be fully reused
        then for other chunk blocks, if store hit num of block hash is less than threshold, we do not conduct cache blend
        finally, if there are quite many chunk block-hits, we do cache blend to get TTFT-promot
        """
        chunks_meta = []
        prefix_block_hashes = self.generate_hash(
            self.block_size, all_token_ids, self._seed
        )
        if (
            all_token_ids[-1] == self.chunk_end_token_id
            and len(all_token_ids) % self.block_size == 0
        ):
            return (
                BlendStage.BUILD_CHUNK_CACHE,
                prefix_block_hashes,
                chunks_meta,
                [],
            )

        start_blk_idx = 0
        start_token_dix = 0
        req_chunks_hashes = []

        for end_blk_idx, end_token_idx in enumerate(
            range(self.block_size - 1, len(all_token_ids), self.block_size)
        ):
            # only compare the last token id in each blk to split chunk
            # in future we should add chunk info as llm engine input,then pass them to schedule out
            # but this will bring lots of modification to engine.
            if all_token_ids[end_token_idx] == self.chunk_end_token_id:
                chunk_token_ids = all_token_ids[start_token_dix : end_token_idx + 1]
                chunk_blks_hash = self.generate_hash(
                    self.block_size, chunk_token_ids, self._seed
                )

                chunk_blks_len = end_blk_idx - start_blk_idx + 1
                chunk_tokens_len = chunk_blks_len * self.block_size

                rag_chunk_meta = ChunkMetaData(
                    start_token_dix=start_token_dix,
                    chunk_tokens_len=chunk_tokens_len,
                    start_blk_idx=start_blk_idx,
                    chunk_blks_len=chunk_blks_len,
                    chunk_blks_hash=chunk_blks_hash,
                    cached_start_position=0,
                )

                # update for next rag chunk
                start_blk_idx = end_blk_idx + 1
                start_token_dix = end_token_idx + 1

                chunks_meta.append(rag_chunk_meta)
                req_chunks_hashes.extend(chunk_blks_hash)

        if chunks_meta:
            # found chunk, as for suffix part(such as user question about chunk), current no need to cache hit and dump
            return (
                BlendStage.CACHE_BLEND,
                prefix_block_hashes,
                chunks_meta,
                req_chunks_hashes,
            )
        else:
            return (
                BlendStage.BUILD_PREFIX_CACHE,
                prefix_block_hashes,
                chunks_meta,
                req_chunks_hashes,
            )

    def _get_req_chunk_hit(
        self,
        req_stage: BlendStage,
        prefix_block_hashes: List[str],
        req_chunks_meta: List[ChunkMetaData],
        req_chunks_hashes: List[str],
    ) -> Tuple[int, int]:

        # first perform prefix cache lookup
        pc_lookup_results = self.store.lookup(prefix_block_hashes)
        pc_hit_blocks = 0
        chunk_hit_blocks = 0

        for i, hit in enumerate(pc_lookup_results):
            if not hit:
                break
            pc_hit_blocks += 1

        if not req_stage.is_blend_cache():
            return pc_hit_blocks, chunk_hit_blocks

        # then perform chunk cache lookup
        chunk_lookup_results = self.store.lookup(req_chunks_hashes[pc_hit_blocks:])
        chunk_hit_blocks = sum(chunk_lookup_results)

        chunk_lookup_results = pc_lookup_results[:pc_hit_blocks] + chunk_lookup_results
        # for cache blend
        for i, chunk_meta in enumerate(req_chunks_meta):
            chunk_meta.store_hits = chunk_lookup_results[
                chunk_meta.start_blk_idx : chunk_meta.end_blk_idx
            ]
        first_chunk_meta = req_chunks_meta[0]
        first_chunk_meta.update_meta_partial_pc(pc_hit_blocks, self.block_size)
        # remove total pc hit chunk
        if first_chunk_meta.chunk_tokens_len == 0:
            req_chunks_meta.pop(0)

        return pc_hit_blocks, chunk_hit_blocks

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
        |           REUSE               |     RECOMPUTE          |
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
                    chunk_meta.start_blk_idx : chunk_meta.end_blk_idx
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

    def _post_process_chunk_cache(self, k_cache, vllm_ids, positions) -> None:
        """
        post process loaded chunk kcache
        """
        if self.cos_sin_cache is None:
            raise "Please call setup model first."
        # triton kernl for block-wise delta rope
        block_wise_rope_forward(k_cache, vllm_ids, positions, self.cos_sin_cache)

    def _register_cos_sin_cache(self, model: "Model") -> None:
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
        # so they can not be applied to other req
        assert num_computed_tokens == 0
        all_token_ids = request.all_token_ids

        max_blk_num = len(all_token_ids) // self.block_size

        if max_blk_num == 0:
            return 0, False

        req_stage, prefix_block_hashes, req_chunks_meta, req_chunks_hashes = (
            self._process_req(all_token_ids)
        )

        pc_hit_blocks, chunk_hit_blocks = self._get_req_chunk_hit(
            req_stage, prefix_block_hashes, req_chunks_meta, req_chunks_hashes
        )

        if chunk_hit_blocks < self.min_blend_threshold:
            req_stage = BlendStage.BUILD_PREFIX_CACHE
            req_chunks_meta = []

        req_block_hashes = prefix_block_hashes
        if req_stage.is_blend_cache():
            req_block_hashes = req_chunks_hashes

        logger.info(
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {max_blk_num}, "
            f"req_stage: {req_stage}, "
            f"first chunk prefix hit: {pc_hit_blocks}, "
            f"chunks cache total hit: {chunk_hit_blocks}, "
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

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """
        Blend need build post process meta for loaded kv cache
        """
        super().bind_connector_metadata(connector_metadata)
        all_hits_vllm_ids = []
        positions = []
        for request_id, request in connector_metadata.request_meta.items():
            for chunk_meta in request.chunks_meta:
                all_hits_vllm_ids.extend(chunk_meta.hits_vllm_blk_ids)
                positions.extend(
                    [chunk_meta.position_offset] * len(chunk_meta.hits_vllm_blk_ids)
                )
        if all_hits_vllm_ids:
            self.delta_rope_vllm_ids = torch.tensor(
                all_hits_vllm_ids, device=self.device
            )
            self.delta_rope_positions = torch.tensor(positions, device=self.device)

    def clear_connector_metadata(self) -> None:
        """Clear the post process meta"""
        super().clear_connector_metadata()
        self.delta_rope_vllm_ids = None
        self.delta_rope_positions = None

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self.delta_rope_vllm_ids is not None:
            k_cache = self.kv_caches[layer_name][0]
            self._post_process_chunk_cache(
                k_cache, self.delta_rope_vllm_ids, self.delta_rope_positions
            )
