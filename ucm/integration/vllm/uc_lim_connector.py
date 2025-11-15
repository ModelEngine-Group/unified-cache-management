import os
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.v1.request import Request, RequestStatus

from ucm.integration.vllm.uc_connector import (
    BlockOperation,
    RequestBlockInfo,
    UnifiedCacheConnectorV1,
)
from ucm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

logger = init_logger(__name__)


class UnifiedCacheLimitConnectorV1(UnifiedCacheConnectorV1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._last_miss_ratio = 0.0
        if (
            self._vllm_config.kv_transfer_config is not None
            and "hit_miss_ratio"
            in self._vllm_config.kv_transfer_config.kv_connector_extra_config
        ):
            self._last_miss_ratio = (
                self._vllm_config.kv_transfer_config.kv_connector_extra_config[
                    "hit_miss_ratio"
                ]
            )
        self._last_miss_ratio = max(0.0, min(1.0, self._last_miss_ratio))

        logger.info(f"Miss hit ratio is {self._last_miss_ratio}")

    @property
    def hit_ratio_upper(self) -> float:
        env_val = os.getenv("UC_HIT_MISS_RATIO")

        try:
            if env_val is None:
                return 1 - self._last_miss_ratio

            current_miss_ratio = float(env_val)
            current_miss_ratio = max(0.0, min(1.0, current_miss_ratio))

            if current_miss_ratio == self._last_miss_ratio:
                return 1 - current_miss_ratio

            logger.info(
                f"UC_HIT_MISS_RATIO changed from {self._last_miss_ratio} to {current_miss_ratio}"
            )
            self._last_miss_ratio = current_miss_ratio
            return 1 - current_miss_ratio

        except ValueError:
            logger.warning(
                f"UC_HIT_MISS_RATIO={env_val} is invalid, use last miss ratio {self._last_miss_ratio}"
            )
            if self._last_miss_ratio is None:
                self._last_miss_ratio = 0.0
            return 1 - self._last_miss_ratio

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        logger.info(f"get_num_new_matched_tokens request {request.request_id}.")

        hit_ratio_upper = self.hit_ratio_upper

        if request.status == RequestStatus.PREEMPTED:
            logger.info(f"Handle preempted request {request.request_id}.")
            self.request_finished(request, [])

        assert num_computed_tokens % self.block_size == 0
        block_hashes = self.hash_request_tokens(self.md5, self.block_size, request)
        if not block_hashes:
            logger.debug("Maybe tokens too short to load.")
            return 0, False

        # Lookup for all blocks
        lookup_results = self.connector.lookup(block_hashes)
        total_blocks_length = len(block_hashes)

        lookup_hits = sum(lookup_results)
        current_hit_ratio = 0.0
        if total_blocks_length > 0:
            current_hit_ratio = lookup_hits / total_blocks_length
        # Limit the hit blocks
        if current_hit_ratio > hit_ratio_upper:
            original_hits = lookup_hits
            # Align to block size
            new_hits = int(total_blocks_length * hit_ratio_upper)
            lookup_hits = min(lookup_hits, new_hits)
            logger.info(
                f"hit ratio upper: {hit_ratio_upper} is smaller than "
                f"the real hit ratio {current_hit_ratio}, "
                f"the origin hits is {original_hits}, "
                f"the new hits is {new_hits}, the final hits is {lookup_hits}"
            )

        start_position = num_computed_tokens // self.block_size
        remain_hashes = block_hashes[start_position:]
        block_operations = [BlockOperation.NONE] * len(block_hashes)
        if not remain_hashes:
            # All blocks are in HBM, or hit ratio lower than already hit in HBM
            return 0, False

        # External hit blocks equals to all hit blocks - already hit on hbm
        num_lookup_hits = 0
        lookup_hits -= start_position
        lookup_results = lookup_results[start_position:]
        for i, hit in enumerate(lookup_results):
            if hit and num_lookup_hits < lookup_hits:
                num_lookup_hits += 1
                block_operations[start_position + i] = BlockOperation.LOAD
            else:
                # TODO we will fix hole match later
                break
        logger.info(
            f"num_total_blocks: {len(block_hashes)}, "
            f"num_lookup_hits on hbm: {start_position}, "
            f"num_lookup_hits on storage except hbm: {num_lookup_hits}"
        )

        # TODO Fix load async scene

        # When all the tokens are cached in ssd or hbm,
        # we need to recompute the last token. This if condition will be removed
        # once vLLM's scheduler provides a better solution in the future.
        if (num_lookup_hits + start_position) * self.block_size == len(
            request.all_token_ids
        ):
            num_lookup_hits -= 1
            block_operations[-1] = BlockOperation.NONE

        self.request_block_infos[request.request_id] = RequestBlockInfo(
            block_hashes=block_hashes,
            block_operations=block_operations,
            start_position=start_position,
        )

        return num_lookup_hits * self.block_size, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        if request.request_id in self._need_load_reqs:
            local_block_ids = (
                # since we use unhashed blocks, so we don't need to reset start_position
                blocks.get_unhashed_block_ids()
                if num_external_tokens > 0
                else []
            )
            self._need_load_reqs[request.request_id] = local_block_ids
            return

        request_block_info = self.request_block_infos.get(request.request_id, None)
        if request_block_info:
            start_position = request_block_info.start_position
            block_operations = request_block_info.block_operations
            block_hashes = request_block_info.block_hashes
            start_create_pos = start_position + num_external_tokens // self.block_size
            remaining_hashes = block_hashes[start_create_pos:]
            if remaining_hashes:
                create_results = self.connector.create(remaining_hashes)
                if any(ret != 0 for ret in create_results):
                    logger.warning(f"\ncreate_results on storage: {create_results}\n")
                for j, ret in enumerate(create_results):
                    idx = start_create_pos + j
                    block_operations[idx] = (
                        BlockOperation.DUMP
                        if ret == 0 or ret == -50003
                        else BlockOperation.NONE
                    )
            # set start_position to 0, so that we can process from the beginning
            request_block_info.start_position = 0
