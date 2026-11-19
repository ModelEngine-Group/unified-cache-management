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

    # DSV3.2 currently has graph compilation issues when using torch_npu.npu.lightning_indexer.
    # So two branches are maintained temporarily.
    # TODO: torch.ops._C_ascend.npu_lightning_indexer needs to be removed.
    packed_kv_cache = getattr(sfa_impl, "enable_sparse_sfa_c8", False)
    indexer_cache_idx = 1 if packed_kv_cache else 2
    indexer_scale_cache_idx = 2 if packed_kv_cache else 3

    if (
        has_ucm_sparse()
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
            # actual_sparse_lengths: per-req 真实 topk 长度（kernel 按 bIdx=req
            # 索引读取，batchSize = len(actual_seq_lengths_query)），由 gsa.py
            # _update_attn_gsa_lens 准备（decode 分支下发 per-req 数组：swap req=
            # SPA_TOKENS_LEN，非 swap=DEFAULT_TOPK_LEN，padding req=0；prefill
            # 分支置 None——两次 SFA 的 topk 均全宽有效且 swap 子批是重排 batch，
            # 全量数组行号与其不对应），用于 kernel 跳过 topk_indices 尾部 -1
            # padding 的迭代与 padding req。
            actual_sparse_lengths=getattr(
                attn_metadata, "actual_sparse_lengths", None
            ),
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


def execute_kv_quant_sparse_flash_attention(
    cls,
    sfa_impl,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    *,
    sparse_mode: int = 3,
    return_lse: bool = False,
    actual_sparse_lengths: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """A3 (BaseDeviceAdaptor) C8 sparse attention via torch.ops._C_gsa.

    Replaces ``BaseDeviceAdaptor.execute_kv_quant_sparse_flash_attention``
    (vllm-ascend device_op.py) which calls
    ``torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention``. The GSA op
    (``torch.ops._C_gsa.npu_kv_quant_sparse_flash_attention_gsa``, registered
    in ops/torch_binding.cpp) is the same aclnn-backed kernel but lives in the
    standalone gsa_ops wheel so it no longer depends on the build-time
    vllm-ascend csrc patch.

    Parameter mapping: the GSA op accepts the same keyword args as the ascend
    op plus 6 optional ones (key_dequant_scale / value_dequant_scale / sinks /
    actual_sparse_lengths / pre_tokens / next_tokens). ``actual_sparse_lengths``
    is a per-req 真实 topk 长度数组（kernel 按 bIdx=req 索引读取，batchSize =
    len(actual_seq_lengths_query) = req 数；gsa.py _update_attn_gsa_lens 准备：
    decode 分支下发，swap req=SPA_TOKENS_LEN / 非 swap=DEFAULT_TOPK_LEN /
    padding req=0 即跳过；prefill 分支置 None——两次 SFA 的 topk 均全宽有效且
    swap 子批是重排 batch，全量数组行号与其不对应），kernel 用它跳过
    topk_indices 尾部 -1 padding 迭代与 padding req（5120-padding-1 性能关键）；
    其余 5 个 optional 默认 None，省略以匹配原调用面。``key_quant_mode`` /
    ``value_quant_mode`` 在 GSA schema 中位于 optional args 之前（见
    torch_binding.cpp:223-235），用 keyword 传递以避免位置依赖。
    """
    query = torch.cat([ql_nope, q_pe], dim=-1).contiguous()
    result = torch.ops._C_gsa.npu_kv_quant_sparse_flash_attention_gsa(
        query=query,
        key=kv,
        value=kv,
        sparse_indices=topk_indices,
        scale_value=sfa_impl.scale,
        key_quant_mode=2,
        value_quant_mode=2,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_key,
        actual_sparse_lengths=actual_sparse_lengths,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=sparse_mode,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=getattr(sfa_impl, "sfa_qsfa_tile_size", 128),
        rope_head_dim=getattr(sfa_impl, "qk_rope_head_dim", q_pe.shape[-1]),
        return_softmax_lse=return_lse,
    )
    if not isinstance(result, tuple):
        if return_lse:
            raise RuntimeError(
                "C8 sparse flash attention did not return softmax max/sum for DCP LSE merge."
            )
        return result
    attn_output, softmax_max, softmax_sum = result
    return BaseDeviceAdaptor._format_sparse_flash_attention_output(
        attn_output,
        softmax_max,
        softmax_sum,
        return_lse,
    )


# NOTE: A3 (BaseDeviceAdaptor) and A5 (A5DeviceAdaptor) intentionally use
# different indexer implementations. The function above (the first
# ``indexer_select_post_process``) is the A3/DSV3.2 path: it routes the
# lightning indexer through ``torch.ops._C_ascend.*`` (npu_lightning_indexer /
# npu_lightning_indexer_quant, not yet migrated to gsa_ops) to avoid graph
# compilation issues observed when using ``torch_npu.npu_lightning_indexer``
# on A3. The C8 sparse attention path (``execute_kv_quant_sparse_flash_attention``
# above) is routed through ``torch.ops._C_gsa.npu_kv_quant_sparse_flash_attention_gsa``
# (standalone gsa_ops wheel). The function below is the A5 path, which goes
# through ``torch_npu.*`` directly.
#
# They MUST keep distinct names. If both are named
# ``indexer_select_post_process`` the second ``def`` shadows the first at
# module import time, so ``_patch`` (which looks up the function by its
# module-level name) would always bind the A5 variant onto every adaptor,
# including BaseDeviceAdaptor on A3 -- which is the bug this rename fixes.
def indexer_select_post_process_a5(
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

    packed_kv_cache = getattr(sfa_impl, "enable_sparse_sfa_c8", False)
    indexer_cache_idx = 1 if packed_kv_cache else 2
    indexer_scale_cache_idx = 2 if packed_kv_cache else 3
    if (
        has_ucm_sparse()
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

    if hasattr(mod.BaseDeviceAdaptor, "execute_kv_quant_sparse_flash_attention"):
        # Route C8 (kv-quant) sparse attention through torch.ops._C_gsa
        # (standalone gsa_ops wheel) instead of torch.ops._C_ascend.
        _patched = FunctionType(
            execute_kv_quant_sparse_flash_attention.__code__,
            vars(mod),
            name=execute_kv_quant_sparse_flash_attention.__name__,
            argdefs=execute_kv_quant_sparse_flash_attention.__defaults__,
            closure=execute_kv_quant_sparse_flash_attention.__closure__,
        )
        _patched.__annotations__ = (
            execute_kv_quant_sparse_flash_attention.__annotations__
        )
        _patched.__kwdefaults__ = (
            execute_kv_quant_sparse_flash_attention.__kwdefaults__
        )
        patch_or_inject(
            mod.BaseDeviceAdaptor,
            "execute_kv_quant_sparse_flash_attention",
            classmethod(_patched),
        )
        logger.info(
            "UCM GSA BaseDeviceAdaptor.execute_kv_quant_sparse_flash_attention patch applied"
        )

    if hasattr(mod.A5DeviceAdaptor, "indexer_select_post_process"):
        # A5 keeps the torch_npu-based implementation; do NOT use the
        # module-level ``indexer_select_post_process`` here, which after the
        # rename above resolves to the A3 (_C_ascend) variant.
        _patched = FunctionType(
            indexer_select_post_process_a5.__code__,
            vars(mod),
            name=indexer_select_post_process_a5.__name__,
            argdefs=indexer_select_post_process_a5.__defaults__,
            closure=indexer_select_post_process_a5.__closure__,
        )
        _patched.__annotations__ = indexer_select_post_process_a5.__annotations__
        _patched.__kwdefaults__ = indexer_select_post_process_a5.__kwdefaults__
        patch_or_inject(mod.A5DeviceAdaptor, "indexer_select_post_process", _patched)
        logger.info("UCM GSA A5DeviceAdaptor.indexer_select_post_process patch applied")
