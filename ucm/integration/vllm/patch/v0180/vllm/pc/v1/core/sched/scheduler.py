from collections.abc import Iterable

from vllm.v1.request import Request, RequestStatus


class Scheduler:
    def _mamba_block_aligned_split(
        self,
        request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                num_new_tokens = last_cache_position - num_computed_tokens
        return num_new_tokens

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            req_block_id_groups = self.kv_cache_manager.get_block_ids(req_id)
            if not req_block_id_groups:
                continue
            # vLLM v0.18's recovery path assumes one KV group. Hybrid
            # Qwen3-Next has multiple groups, and UCM's HMA dispatch always
            # includes the full-attention group as group 0, so use it as the
            # recovery anchor.
            req_block_ids = req_block_id_groups[0]
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                req_num_computed_tokens = request.num_computed_tokens
            else:
                req_num_computed_tokens = request.num_cached_tokens

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    continue

                marked_invalid_block = True
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens
                request.num_external_computed_tokens -= num_affected_tokens
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    total_affected_tokens += (
                        request.num_computed_tokens - request.num_cached_tokens
                    )
                    request.num_computed_tokens = request.num_cached_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict
