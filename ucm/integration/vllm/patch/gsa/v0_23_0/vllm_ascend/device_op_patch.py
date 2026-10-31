"""GSA device_op patches for vLLM-Ascend 0.23.0."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def indexer_select_post_process(
    sfa_impl,
    q_li: torch.Tensor,
    q_li_scale: torch.Tensor | None,
    q_li_shape_ori: tuple[Any, ...] | None,
    weights: torch.Tensor,
    kv_cache: tuple,
    attn_metadata,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    enable_sparse_li_c8: bool,
    use_torch_npu_lightning_indexer: bool,
) -> torch.Tensor:
    from ucm.sparse.state import has_ucm_sparse
    from ucm.sparse.utils import PTOPK_PREFETCH_ENABLE

    # DSV3.2 currently has graph compilation issues when using torch_npu.npu.lightning_indexer.
    # So two branches are maintained temporarily.
    # TODO: torch.ops._C_ascend.npu_lightning_indexer needs to be removed.
    packed_kv_cache = getattr(sfa_impl, "enable_sparse_sfa_c8", False)
    indexer_cache_idx = 1 if packed_kv_cache else 2
    indexer_scale_cache_idx = 2 if packed_kv_cache else 3

    if (
        has_ucm_sparse()
        and PTOPK_PREFETCH_ENABLE
        and sfa_impl.layer_id < sfa_impl.layer_num
        and (
            sfa_impl.is_pdmix
            or not sfa_impl.vllm_config.kv_transfer_config.is_kv_producer
        )
    ):
        block_table = attn_metadata.dsa_block_table[sfa_impl.cur_topk_index][
            : len(actual_seq_lengths_query)
        ]
        dsa_cache = sfa_impl.dsa_k_cache
        if enable_sparse_li_c8:
            dsa_cache_scale = sfa_impl.dsa_k_scale
    else:
        block_table = attn_metadata.block_table
        # vLLM 0.23 cannot directly use kv_cache[2],
        # because packed kv cache changes the index.
        dsa_cache = kv_cache[indexer_cache_idx]
        if enable_sparse_li_c8:
            dsa_cache_scale = kv_cache[indexer_scale_cache_idx]

    if enable_sparse_li_c8:
        assert len(kv_cache) == (3 if packed_kv_cache else 4)
        assert q_li_scale is not None
        assert q_li_shape_ori is not None
        weights = weights.to(torch.float16)
        topk_indices = torch.ops._C_ascend.npu_lightning_indexer_quant(
            query=q_li.view(q_li_shape_ori),
            key=dsa_cache,
            weights=weights,
            query_dequant_scale=q_li_scale.view(q_li_shape_ori[:-1]),
            key_dequant_scale=dsa_cache_scale.squeeze(2),  # B S N D -> B S D
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        )
    elif sfa_impl.use_torch_npu_lightning_indexer:
        topk_indices, _ = torch_npu.npu_lightning_indexer(
            query=q_li,
            key=dsa_cache,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        )
    else:
        topk_indices, _ = torch.ops._C_ascend.npu_lightning_indexer(
            query=q_li,
            key=dsa_cache,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        )
    return topk_indices


def execute_sparse_flash_attention_process(
    cls,
    sfa_impl,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_cache: tuple,
    topk_indices: torch.Tensor,
    attn_metadata,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    swap_len: torch.Tensor = None,
    *,
    block_table: torch.Tensor | None = None,
    sparse_mode: int = 3,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if block_table is None:
        block_table = attn_metadata.block_table
    if swap_len is not None:
        block_table = attn_metadata.block_table[swap_len]
    kv = kv_cache[0]

    # The kv-quant sparse attention op only accepts packed quantized KV.
    # Do not route by the feature flag alone: tests and fallback paths may
    # pass a normal BF16/FP16 KV cache even when the mocked impl exposes a
    # truthy attribute.
    use_kv_quant_sparse_attention = kv.dtype in (
        torch.int8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if use_kv_quant_sparse_attention:
        return cls.execute_kv_quant_sparse_flash_attention(
            sfa_impl,
            ql_nope,
            q_pe,
            kv,
            block_table,
            topk_indices,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            sparse_mode=sparse_mode,
            return_lse=return_lse,
        )

    key_rope = kv_cache[1]
    result = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=ql_nope,
        key=kv,
        value=kv,
        sparse_indices=topk_indices,
        scale_value=sfa_impl.scale,
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_key,
        query_rope=q_pe,
        key_rope=key_rope,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=sparse_mode,
        attention_mode=2,
        return_softmax_lse=return_lse,
    )
    if not isinstance(result, tuple):
        if return_lse:
            raise RuntimeError(
                "Sparse flash attention did not return softmax max/sum for DCP LSE merge."
            )
        return result
    attn_output, softmax_max, softmax_sum = result
    return BaseDeviceAdaptor._format_sparse_flash_attention_output(
        attn_output,
        softmax_max,
        softmax_sum,
        return_lse,
    )


def indexer_select_post_process(
    sfa_impl,
    q_li: torch.Tensor,
    q_li_scale: torch.Tensor | None,
    q_li_shape_ori: tuple[Any, ...] | None,
    weights: torch.Tensor,
    kv_cache: tuple,
    attn_metadata,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    enable_sparse_li_c8: bool,
    use_torch_npu_lightning_indexer: bool,
) -> torch.Tensor:
    from ucm.sparse.state import has_ucm_sparse
    from ucm.sparse.utils import PTOPK_PREFETCH_ENABLE

    packed_kv_cache = getattr(sfa_impl, "enable_sparse_sfa_c8", False)
    indexer_cache_idx = 1 if packed_kv_cache else 2
    indexer_scale_cache_idx = 2 if packed_kv_cache else 3
    if (
        has_ucm_sparse()
        and PTOPK_PREFETCH_ENABLE
        and sfa_impl.layer_id < sfa_impl.layer_num
        and (
            sfa_impl.is_pdmix
            or not sfa_impl.vllm_config.kv_transfer_config.is_kv_producer
        )
    ):
        block_table = attn_metadata.dsa_block_table[sfa_impl.cur_topk_index][
            : len(actual_seq_lengths_query)
        ]
        dsa_cache = sfa_impl.dsa_k_cache
        if enable_sparse_li_c8:
            dsa_k_scale = sfa_impl.dsa_k_scale.squeeze(2)
    else:
        block_table = attn_metadata.block_table
        # vLLM 0.23 cannot directly use kv_cache[2],
        # because packed kv cache changes the index.
        dsa_cache = kv_cache[indexer_cache_idx]
        if enable_sparse_li_c8:
            dsa_k_scale = kv_cache[indexer_scale_cache_idx].squeeze(2)
    if enable_sparse_li_c8:
        assert len(kv_cache) == (3 if packed_kv_cache else 4)
        assert q_li_shape_ori is not None

        if q_li_scale is not None:
            q_li_scale = q_li_scale.view(q_li_shape_ori[:-1])
            key_dequant_scale = kv_cache[indexer_scale_cache_idx].squeeze(2)

            topk_indices = torch_npu.npu_quant_lightning_indexer(
                query=q_li.view(q_li_shape_ori),
                key=dsa_cache,
                weights=weights,
                query_dequant_scale=q_li_scale,
                key_dequant_scale=dsa_k_scale,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        else:
            topk_indices, _ = torch_npu.npu_lightning_indexer(
                query=q_li.view(q_li_shape_ori),
                key=dsa_cache,
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
    else:
        topk_indices, _ = torch_npu.npu_lightning_indexer(
            query=q_li,
            key=dsa_cache,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        )
    return topk_indices


@when_imported("vllm_ascend.device.device_op")
def _patch(mod):
    if hasattr(mod.BaseDeviceAdaptor, "indexer_select_post_process"):
        _patched = FunctionType(
            indexer_select_post_process.__code__,
            vars(mod),
            name=indexer_select_post_process.__name__,
            argdefs=indexer_select_post_process.__defaults__,
            closure=indexer_select_post_process.__closure__,
        )
        _patched.__annotations__ = indexer_select_post_process.__annotations__
        _patched.__kwdefaults__ = indexer_select_post_process.__kwdefaults__
        patch_or_inject(mod.BaseDeviceAdaptor, "indexer_select_post_process", _patched)
        logger.info(
            "UCM GSA BaseDeviceAdaptor.indexer_select_post_process patch applied"
        )

    if hasattr(mod.BaseDeviceAdaptor, "execute_sparse_flash_attention_process"):
        _patched = FunctionType(
            execute_sparse_flash_attention_process.__code__,
            vars(mod),
            name=execute_sparse_flash_attention_process.__name__,
            argdefs=execute_sparse_flash_attention_process.__defaults__,
            closure=execute_sparse_flash_attention_process.__closure__,
        )
        _patched.__annotations__ = (
            execute_sparse_flash_attention_process.__annotations__
        )
        _patched.__kwdefaults__ = execute_sparse_flash_attention_process.__kwdefaults__
        patch_or_inject(
            mod.BaseDeviceAdaptor,
            "execute_sparse_flash_attention_process",
            classmethod(_patched),
        )
        logger.info(
            "UCM GSA BaseDeviceAdaptor.execute_sparse_flash_attention_process patch applied"
        )

    if hasattr(mod.A5DeviceAdaptor, "indexer_select_post_process"):
        _patched = FunctionType(
            indexer_select_post_process.__code__,
            vars(mod),
            name=indexer_select_post_process.__name__,
            argdefs=indexer_select_post_process.__defaults__,
            closure=indexer_select_post_process.__closure__,
        )
        _patched.__annotations__ = indexer_select_post_process.__annotations__
        _patched.__kwdefaults__ = indexer_select_post_process.__kwdefaults__
        patch_or_inject(mod.A5DeviceAdaptor, "indexer_select_post_process", _patched)
        logger.info("UCM GSA A5DeviceAdaptor.indexer_select_post_process patch applied")
