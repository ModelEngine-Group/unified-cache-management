from collections.abc import Sequence
from functools import wraps
from inspect import signature

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)

_REGISTERED = False
_MISSING = object()


def _get_num_evictable_blocks(cls, blocks: Sequence) -> int:
    return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)


def _manager_get_num_blocks_to_allocate(
    manager, *args, num_external_computed_tokens: int = 0
) -> int:
    params = signature(manager.get_num_blocks_to_allocate).parameters
    if "num_external_computed_tokens" in params:
        return manager.get_num_blocks_to_allocate(
            *args,
            num_external_computed_tokens=num_external_computed_tokens,
        )
    return manager.get_num_blocks_to_allocate(*args)


def coordinator_get_num_blocks_to_allocate(
    self,
    request_id: str,
    num_tokens: int,
    new_computed_blocks: tuple[Sequence, ...],
    num_encoder_tokens: int = 0,
    total_computed_tokens: int = 0,
    num_tokens_main_model: int | None = None,
    num_external_computed_tokens: int = 0,
) -> int:
    from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager

    if num_tokens_main_model is None:
        num_tokens_main_model = num_tokens
    if num_external_computed_tokens == 0:
        num_external_computed_tokens = getattr(
            self, "_ucm_num_external_computed_tokens", 0
        )

    num_blocks_to_allocate = 0
    for i, manager in enumerate(self.single_type_managers):
        if isinstance(manager, CrossAttentionManager):
            num_blocks_to_allocate += _manager_get_num_blocks_to_allocate(
                manager,
                request_id,
                num_encoder_tokens,
                [],
                0,
                num_encoder_tokens,
            )
        else:
            num_blocks_to_allocate += _manager_get_num_blocks_to_allocate(
                manager,
                request_id,
                num_tokens,
                new_computed_blocks[i],
                total_computed_tokens,
                num_tokens_main_model,
                num_external_computed_tokens=num_external_computed_tokens,
            )
    return num_blocks_to_allocate


def single_type_get_num_blocks_to_allocate(
    self,
    request_id: str,
    num_tokens: int,
    new_computed_blocks: Sequence,
    total_computed_tokens: int,
    num_tokens_main_model: int,
    num_external_computed_tokens: int = 0,
) -> int:
    from vllm.utils.math_utils import cdiv

    num_required_blocks = cdiv(num_tokens, self.block_size)
    num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

    if request_id in self.num_cached_block:
        assert len(new_computed_blocks) == 0
        return max(num_required_blocks - num_req_blocks, 0)

    num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
    num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
    num_skipped_blocks = num_skipped_tokens // self.block_size
    num_new_blocks = max(
        num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
        0,
    )

    num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)
    num_evictable_blocks = self._get_num_evictable_blocks(
        new_computed_blocks[num_skipped_new_computed_blocks:]
    )
    return num_new_blocks + num_evictable_blocks


def mamba_get_num_blocks_to_allocate(
    self,
    request_id: str,
    num_tokens: int,
    new_computed_blocks: Sequence,
    total_computed_tokens: int,
    num_tokens_main_model: int,
    num_external_computed_tokens: int = 0,
) -> int:
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    from vllm.v1.kv_cache_interface import MambaSpec

    assert isinstance(self.kv_cache_spec, MambaSpec)
    if len(new_computed_blocks) > 0 and new_computed_blocks[-1].block_hash in getattr(
        self, "cached_blocks_this_step", set()
    ):
        return self.block_pool.num_gpu_blocks + 1

    if self.mamba_cache_mode != "align":
        if self.num_speculative_blocks > 0:
            num_tokens += self.kv_cache_spec.block_size * self.num_speculative_blocks
        return super(MambaManager, self).get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            num_external_computed_tokens=num_external_computed_tokens,
        )

    num_tokens = num_tokens_main_model
    req_blocks = self.req_to_blocks[request_id]
    num_blocks_to_allocate = 0

    if request_id in self.num_cached_block:
        req_len_after_computed = len(req_blocks)
    else:
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            computed_blocks_after_skip = new_computed_blocks[num_skipped_blocks:]
            num_external_computed_tokens = min(
                total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )
        else:
            computed_blocks_after_skip = new_computed_blocks

        num_blocks_to_allocate += self._get_num_evictable_blocks(
            computed_blocks_after_skip
        )

        req_len_after_computed = num_skipped_blocks + len(computed_blocks_after_skip)
        if num_external_computed_tokens > 0:
            num_external_blocks = max(
                cdiv(total_computed_tokens, self.block_size) - req_len_after_computed,
                0,
            )
            num_blocks_to_allocate += num_external_blocks
            req_len_after_computed += num_external_blocks

    num_required_blocks = (
        cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
    )
    if num_required_blocks <= req_len_after_computed:
        return num_blocks_to_allocate

    req_len_before_new_alloc = req_len_after_computed
    num_skipped_blocks = num_required_blocks - self.num_speculative_blocks - 1
    if req_len_before_new_alloc < num_skipped_blocks:
        req_len_before_new_alloc = num_skipped_blocks

    if request_id in self._allocated_block_reqs:
        for block_idx in range(
            req_len_after_computed - self.num_speculative_blocks,
            req_len_after_computed,
        ):
            if block_idx < num_skipped_blocks:
                req_len_before_new_alloc += 1
            else:
                break

    num_blocks_to_allocate += max(num_required_blocks - req_len_before_new_alloc, 0)
    return num_blocks_to_allocate


def make_allocate_slots_patch(original_allocate_slots):
    @wraps(original_allocate_slots)
    def allocate_slots(self, *args, **kwargs):
        num_external_computed_tokens = kwargs.get("num_external_computed_tokens", 0)
        if len(args) >= 6:
            num_external_computed_tokens = args[5]

        coordinator = getattr(self, "coordinator", None)
        if coordinator is None:
            return original_allocate_slots(self, *args, **kwargs)

        previous = getattr(coordinator, "_ucm_num_external_computed_tokens", _MISSING)
        coordinator._ucm_num_external_computed_tokens = num_external_computed_tokens
        try:
            return original_allocate_slots(self, *args, **kwargs)
        finally:
            if previous is _MISSING:
                delattr(coordinator, "_ucm_num_external_computed_tokens")
            else:
                coordinator._ucm_num_external_computed_tokens = previous

    return allocate_slots


def register_get_free_block_patches() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    @when_imported("vllm.v1.core.kv_cache_coordinator")
    def patch_kv_cache_coordinator(mod):
        logger.debug(f"Patched {mod} called")
        patch_or_inject(
            mod.KVCacheCoordinator,
            "get_num_blocks_to_allocate",
            coordinator_get_num_blocks_to_allocate,
        )

    @when_imported("vllm.v1.core.single_type_kv_cache_manager")
    def patch_single_type_kv_cache_manager(mod):
        logger.debug(f"Patched {mod} called")
        patch_or_inject(
            mod.SingleTypeKVCacheManager,
            "_get_num_evictable_blocks",
            classmethod(_get_num_evictable_blocks),
        )
        patch_or_inject(
            mod.SingleTypeKVCacheManager,
            "get_num_blocks_to_allocate",
            single_type_get_num_blocks_to_allocate,
        )
        patch_or_inject(
            mod.MambaManager,
            "get_num_blocks_to_allocate",
            mamba_get_num_blocks_to_allocate,
        )

    @when_imported("vllm.v1.core.kv_cache_manager")
    def patch_kv_cache_manager(mod):
        logger.debug(f"Patched {mod} called")
        original = getattr(
            mod.KVCacheManager,
            "_ucm_original_allocate_slots_for_free_block_patch",
            None,
        )
        if original is None:
            original = mod.KVCacheManager.allocate_slots
            setattr(
                mod.KVCacheManager,
                "_ucm_original_allocate_slots_for_free_block_patch",
                original,
            )
        patch_or_inject(
            mod.KVCacheManager,
            "allocate_slots",
            make_allocate_slots_patch(original),
        )
