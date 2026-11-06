"""Add GSA sparse-attention slots to vLLM 0.23.0 KVCacheManager.allocate_slots."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
    full_sequence_must_fit: bool = False,
    num_slots_sparsed: Union[None, int] = None,
    reserved_blocks: int = 0,
) -> KVCacheBlocks | None:
    """Add slots for a request with new tokens to append.

    Args:
        request: The request to allocate slots.
        num_new_tokens: The number of new tokens to be allocated and computed.
        num_new_computed_tokens: The number of new computed tokens just
            hitting the prefix caching, excluding external tokens.
        new_computed_blocks: The cached blocks for the above new computed
            tokens, grouped as a tuple by kv cache groups.
        num_lookahead_tokens: The number of speculative tokens to allocate.
            This is used by spec decode proposers with kv-cache such
            as eagle.
        num_external_computed_tokens: The number of tokens that their
            KV caches are not cached by vLLM but cached by the connector.
        delay_cache_blocks: Whether to skip caching the blocks. This is
            used by P/D when allocating blocks used in a KV transfer
            which will complete in a future step.
        num_encoder_tokens: The number of encoder tokens to allocate for
            cross-attention in encoder-decoder models(e.g., Whisper).
            For decoder-only models, this should be 0.
        full_sequence_must_fit: Only allocate blocks if the KV cache has enough
            free blocks to hold the full sequence, accounting for prefix cache hits
            and sliding window. Used as an admission gate to prevent over-admitting
            requests when chunked prefill would otherwise only check the first chunk
        reserved_blocks: Number of free blocks that must be left available for
            other in-flight sequences to complete. The actual allocation is only
            made if it fits within (free blocks - reserved_blocks). Used to gate
            async KV-connector loads so their initial allocation cannot consume
            blocks an already in-flight (prefilling) sequence is relying on.

    Blocks layout:
    ```
    ----------------------------------------------------------------------
    | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
    ----------------------------------------------------------------------
                                              |   < to be computed >     |
    ----------------------------------------------------------------------
                              |            < to be allocated >           |
    ----------------------------------------------------------------------
                              | < to be cached (roughly, |
                              | details below)>          |
    ----------------------------------------------------------------------
    | Prefix-cached tokens from either vLLM   |
    | or connector. Can be safely removed if  |
    | they are outside sliding window.        |
    ----------------------------------------------------------------------
    |   < cached by vLLM >    | not cached by |
                              | vLLM, but     |
    | ref_cnt  | ref_cnt not  | cached by     |
    | increased| increased yet| connector     |
    ----------------------------------------------------------------------
    ```

    Abbrivations:

    ```
    comp      = request.num_computed_tokens
    new_comp  = num_new_computed_tokens
              = len(new_computed_blocks) * block_size
    ext_comp  = num_external_computed_tokens, cached by the connector
    new       = num_new_tokens, including unverified draft tokens
    lookahead = num_lookahead_tokens
    ```

    NOTE: for new tokens which include both verified and unverified draft
    tokens, we only cache the verified tokens (by capping the number at
    `request.num_tokens`).

    The allocation has three stages:
    - Free unnecessary blocks in `comp` and check
       if we have sufficient free blocks (return None if not).
    - Handle prefix tokens (`comp + new_comp + ext_comp`):
        - Free unnecessary blocks (e.g. outside sliding window)
        - Allocate new blocks for `ext_comp` tokens inside
          sliding window
    - Allocate new blocks for tokens to be computed (`new + lookahead`)

    Returns:
        A list of new allocated blocks.
    """
    from ucm.sparse.base import INVALID_SLOT
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse
    from ucm.sparse.utils import PTOPK_PREFETCH_ENABLE

    # When loading KV data asynchronously, we may have zero new tokens to
    # compute while still allocating slots for externally computed tokens.
    if num_new_tokens == 0 and num_external_computed_tokens == 0:
        raise ValueError(
            "num_new_tokens must be greater than 0 when there are no "
            "external computed tokens"
        )

    if num_slots_sparsed != INVALID_SLOT and has_ucm_sparse():
        return get_ucm_sparse().allocate_slots(self, request, num_slots_sparsed)

    if new_computed_blocks is not None:
        new_computed_block_list = new_computed_blocks.blocks
    else:
        new_computed_block_list = self.empty_kv_cache_blocks.blocks

    # The number of computed tokens is the number of computed tokens plus
    # the new prefix caching hits
    num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
    total_computed_tokens = min(
        num_local_computed_tokens + num_external_computed_tokens,
        self.max_model_len,
    )

    if full_sequence_must_fit:
        # First check and fail if the full request sequence won't fit.
        full_num_tokens = min(request.num_tokens, self.max_model_len)

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=total_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None

    num_tokens_main_model = total_computed_tokens + num_new_tokens
    num_tokens_need_slot = min(
        num_tokens_main_model + num_lookahead_tokens, self.max_model_len
    )

    # Free the blocks that are skipped during the attention computation
    # (e.g., tokens outside the sliding window).
    # We can do this even if we cannot schedule this request due to
    # insufficient free blocks.
    # Should call this function before allocating new blocks to reduce
    # the number of evicted blocks.
    self.coordinator.remove_skipped_blocks(request.request_id, total_computed_tokens)

    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=num_tokens_need_slot,
        new_computed_blocks=new_computed_block_list,
        num_encoder_tokens=num_encoder_tokens,
        total_computed_tokens=num_local_computed_tokens + num_external_computed_tokens,
        num_tokens_main_model=num_tokens_main_model,
    )

    gsa_preempted_flag = False
    if has_ucm_sparse() and PTOPK_PREFETCH_ENABLE:
        gsa_preempted_flag = get_ucm_sparse().estimate_free_slots_sparse(
            request, num_blocks_to_allocate, num_new_computed_tokens
        )

    available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
    if num_blocks_to_allocate > available_blocks or gsa_preempted_flag:
        # Cannot allocate new blocks
        return None

    if (
        new_computed_block_list is not self.empty_kv_cache_blocks.blocks
        or num_external_computed_tokens > 0
    ):
        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.coordinator.allocate_new_computed_blocks(
            request_id=request.request_id,
            new_computed_blocks=new_computed_block_list,
            num_local_computed_tokens=num_local_computed_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
        )

    new_blocks = self.coordinator.allocate_new_blocks(
        request.request_id,
        num_tokens_need_slot,
        num_tokens_main_model,
        num_encoder_tokens,
    )

    # P/D: delay caching blocks if we have to recv from
    # remote. Update state for locally cached blocks.
    if not self.enable_caching or delay_cache_blocks:
        return self.create_kv_cache_blocks(new_blocks)

    # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
    # + num_external_computed_tokens + num_new_tokens, but must exclude
    # "non-committable" tokens (e.g., draft tokens that could be rejected).
    # Therefore, we cap the number at `request.num_tokens`, ensuring only
    # "finalized" tokens are cached.
    num_tokens_to_cache = min(
        total_computed_tokens + num_new_tokens,
        request.num_tokens,
    )
    self.coordinator.cache_blocks(request, num_tokens_to_cache)

    return self.create_kv_cache_blocks(new_blocks)


@when_imported("vllm.v1.core.kv_cache_manager")
def _patch(mod):
    if hasattr(mod.KVCacheManager, "allocate_slots"):
        _patched = FunctionType(
            allocate_slots.__code__,
            vars(mod),
            name=allocate_slots.__name__,
            argdefs=allocate_slots.__defaults__,
            closure=allocate_slots.__closure__,
        )
        _patched.__annotations__ = allocate_slots.__annotations__
        _patched.__kwdefaults__ = allocate_slots.__kwdefaults__
        patch_or_inject(mod.KVCacheManager, "allocate_slots", _patched)
        logger.info("UCM GSA KVCacheManager.allocate_slots patch applied")
