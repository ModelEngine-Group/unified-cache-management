# mypy: ignore-errors
import math

import vllm.model_executor.models.config
from vllm.logger import logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MambaModelConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size


@classmethod
def verify_and_update_config(cls, vllm_config) -> None:
    MambaModelConfig.verify_and_update_config(vllm_config)

    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config

    if cache_config.cache_dtype == "auto":
        kv_cache_dtype = model_config.dtype
    else:
        kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

    kernel_block_size = 128
    attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    attn_head_size = model_config.get_head_size()
    attn_single_token_k_page_size = (
        attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
    )

    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture,
        model_config=model_config,
    )

    mamba_shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
    mamba_dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
    mamba_sizes = []
    for shape, dtype in zip(mamba_shapes, mamba_dtypes):
        mamba_sizes.append(math.prod(shape) * get_dtype_size(dtype))
    ssm_block_page_size, conv_block_page_size = max(mamba_sizes), min(mamba_sizes)

    attn_block_size = kernel_block_size * cdiv(
        ssm_block_page_size,
        kernel_block_size * attn_single_token_k_page_size,
    )
    assert (
        attn_single_token_k_page_size * attn_block_size == ssm_block_page_size
    ), "Cannot align ssm_page_size and attn_page_size."

    if cache_config.block_size is None or cache_config.block_size < attn_block_size:
        cache_config.block_size = attn_block_size
        logger.info(
            "Setting attention block size to %d tokens to ensure that attention "
            "page size is >= mamba page size.",
            attn_block_size,
        )

    attn_page_size = (
        cache_config.block_size
        * 2
        * attn_head_size
        * attn_num_kv_heads
        * get_dtype_size(kv_cache_dtype)
    )

    if (
        cache_config.mamba_page_size_padded is None
        or cache_config.mamba_page_size_padded != attn_page_size + conv_block_page_size
    ):
        cache_config.mamba_page_size_padded = attn_page_size + conv_block_page_size
        mamba_padding_pct = (
            100 * conv_block_page_size / cache_config.mamba_page_size_padded
        )
        logger.info(
            "Padding mamba page size by %.2f%% to ensure that mamba page size "
            "and attention page size are exactly equal.",
            mamba_padding_pct,
        )

    cache_config.mamba_block_size = cache_config.block_size


def patch_hybrid_attention_mamba_model_config() -> None:
    vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = (
        verify_and_update_config
    )
