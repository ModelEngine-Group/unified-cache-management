import os
from typing import Any, Callable, NewType, Optional, Union

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    get_max_concurrency_for_kv_cache_config,
    get_num_blocks,
    get_uniform_page_size,
    may_override_num_blocks,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from ``bytes`` helps
# catch accidental misuse when passing around raw byte strings.
BlockHash = NewType("BlockHash", bytes)

# ``BlockHashWithGroupId`` combines a ``BlockHash`` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency. The helper
# functions below pack/unpack the ``BlockHash`` and group id into/from the key.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of ``bytes`` and ``int`` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
ExternalBlockHash = Union[bytes, int]

logger = init_logger(__name__)


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    kv_cache_specs: dict[str, KVCacheSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        kv_cache_specs: The KV cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    # Determine how model runners should initialize the KV cache tensors.
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # Special case: all layers have the same type of KV cache but with
        # different hidden size. Allocate different amount of memory for each
        # layer based on its hidden size.
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs

        if os.getenv("VLLM_HASH_ATTENTION") == "1":
            from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

            if vllm_config.cache_config.cache_dtype == "auto":
                dtype = vllm_config.model_config.dtype
            else:
                dtype = STR_DTYPE_TO_TORCH_DTYPE[vllm_config.cache_config.cache_dtype]
            khash_scale = dtype.itemsize * 8
            new_num_blocks = num_blocks * khash_scale // (khash_scale + 1)
            logger.info(
                "[HASH_ATTN] reduce num_blocks from %d to %d to allocate khash_cache",
                num_blocks,
                new_num_blocks,
            )
            num_blocks = new_num_blocks

        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    else:
        # General case:
        # We will have group_size memory pools, each is shared by one layer from
        # each group. As layers of different groups have different block table,
        # they will use different parts of the shared Tensor.
        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
        # (sw.1, padding) will be: (group_size = 2)
        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
        # full.1, sw.2: share another Tensor with size=available_memory//2
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(kv_cache_specs)
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )

    min_block_size = min([group.kv_cache_spec.block_size for group in kv_cache_groups])

    # Print the KV cache size and maximum concurrency.
    num_tokens = num_blocks // len(kv_cache_groups) * min_block_size
    if vllm_config.parallel_config.decode_context_parallel_size > 1:
        num_tokens *= vllm_config.parallel_config.decode_context_parallel_size
        logger.info(
            "Multiplying the GPU KV cache size by the dcp_world_size %d.",
            vllm_config.parallel_config.decode_context_parallel_size,
        )
    num_tokens_str = f"{num_tokens:,}"
    logger.info("GPU KV cache size: %s tokens", num_tokens_str)
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    logger.info(
        "Maximum concurrency for %s tokens per request: %.2fx",
        max_model_len_str,
        max_concurrency,
    )
    return kv_cache_config
