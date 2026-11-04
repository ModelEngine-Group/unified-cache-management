"""GSA AscendMLAImpl.forward patch for vLLM-Ascend 0.23.0."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def forward(
    self,
    layer_name,
    hidden_states: torch.Tensor,  # query in unified attn
    kv_cache: tuple[torch.Tensor],
    attn_metadata: M,
    need_gather_q_kv: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    assert output is not None, "Output tensor must be provided."
    if attn_metadata is None:
        # Profiling run.
        for layer in self.layer_sharding_kwargs or []:
            if is_hidden_layer(layer):
                reach_layer_for_shard_weight_series(layer)
        return output.fill_(0)

    num_actual_tokens = self.get_num_actual_tokens(attn_metadata)
    assert (
        attn_metadata.num_decodes is not None
        and attn_metadata.num_prefills is not None
        and attn_metadata.num_decode_tokens is not None
    )

    num_decode_tokens = attn_metadata.num_decode_tokens
    # Inputs and outputs may be padded for CUDA graphs
    output_padded = output
    o_proj_input_shape = (_EXTRA_CTX.num_tokens, self.num_heads * self.v_head_dim)
    o_proj_input = torch.zeros(
        o_proj_input_shape, dtype=hidden_states.dtype, device=hidden_states.device
    )

    if os.getenv("VLLM_HASH_ATTENTION") == "1":
        kv_cache, k_hash = kv_cache
    else:
        k_hash = None
    forward_context: ForwardContext = get_forward_context()

    # MLA Preprocess
    if (self.fa_quant_layer or self.enable_mlapo) and (
        attn_metadata.num_decode_tokens <= MLAPO_MAX_SUPPORTED_TOKENS
        and attn_metadata.num_prefills == 0
    ):
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states.contiguous(), need_gather_q_kv
        )
        decode_preprocess_res, prefill_preprocess_res = (
            DeviceOperator.mla_preprocess_only_decode(
                self, hidden_states, kv_cache, attn_metadata
            )
        )
    else:
        decode_preprocess_res, prefill_preprocess_res = self._mla_preprocess(
            layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv
        )
    if decode_preprocess_res is not None:
        # MLA Preprocess for decoding

        _, _, _, _ = maybe_execute_sparse_attention_begin(
            torch.cat(
                [decode_preprocess_res.ql_nope, decode_preprocess_res.q_pe], dim=-1
            ),
            decode_preprocess_res.ql_nope,
            decode_preprocess_res.q_pe,
            layer_name,
            forward_context,
            output=output,
            phase="decode",
            k_hash=k_hash,
            decode_ql_nope=decode_preprocess_res.ql_nope,
            decode_q_pe=decode_preprocess_res.q_pe,
        )

        output_decode = self._forward_decode(
            decode_preprocess_res.ql_nope,
            decode_preprocess_res.q_pe,
            decode_preprocess_res.k_nope,
            decode_preprocess_res.k_pe,
            kv_cache[0].shape[1],
            attn_metadata,
            decode_preprocess_res.dequant_scale_q_nope,
        )

        o_proj_input[:num_decode_tokens] = output_decode

        maybe_execute_sparse_attention_finished(
            torch.cat(
                [decode_preprocess_res.ql_nope, decode_preprocess_res.q_pe], dim=-1
            ),
            decode_preprocess_res.ql_nope,
            decode_preprocess_res.q_pe,
            output[:num_decode_tokens],
            layer_name,
            forward_context,
            phase="decode",
        )

    if prefill_preprocess_res is not None:
        # FIX: aicore move should be also placed on the comm stream in dbo,
        # otherwise it may affect the accuracy
        # TODO: use an elegant way to overlap

        _, _, _, _ = maybe_execute_sparse_attention_begin(
            torch.cat(
                [prefill_preprocess_res.q_nope, prefill_preprocess_res.q_pe], dim=-1
            ),
            prefill_preprocess_res.k_nope,
            prefill_preprocess_res.k_pe,
            layer_name,
            forward_context,
            phase="prefill",
            k_hash=k_hash,
        )

        output_prefill = self._forward_prefill(
            prefill_preprocess_res.q_nope,
            prefill_preprocess_res.q_pe,
            prefill_preprocess_res.k_nope,
            prefill_preprocess_res.k_pe,
            prefill_preprocess_res.value,
            kv_cache,
            attn_metadata,
        )

        o_proj_input[num_decode_tokens:num_actual_tokens] = output_prefill

        maybe_execute_sparse_attention_finished(
            torch.cat(
                [prefill_preprocess_res.q_nope, prefill_preprocess_res.q_pe], dim=-1
            ),
            prefill_preprocess_res.k_nope,
            prefill_preprocess_res.k_pe,
            output[num_decode_tokens:],
            layer_name,
            forward_context,
            phase="prefill",
        )

    # O proj
    weight_prefetch_method = get_weight_prefetch_method()
    weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
        inputs=self.o_proj.weight,
        dependency=o_proj_input,
        max_size=MAX_O_PROJ_PREFETCH_SIZE,
        linear_layer=self.o_proj,
    )
    output[...] = self.o_proj(
        o_proj_input, is_prefill=prefill_preprocess_res is not None
    )[0]

    del o_proj_input
    maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
    return output_padded


@when_imported("vllm_ascend.attention.mla_v1")
def _patch(mod):
    try:
        import os as _os

        from vllm.forward_context import ForwardContext, get_forward_context
        from vllm.model_executor.layers.attention.attention import (
            maybe_execute_sparse_attention_begin,
            maybe_execute_sparse_attention_finished,
        )

        for _name, _val in {
            "os": _os,
            "ForwardContext": ForwardContext,
            "get_forward_context": get_forward_context,
            "maybe_execute_sparse_attention_begin": maybe_execute_sparse_attention_begin,
            "maybe_execute_sparse_attention_finished": maybe_execute_sparse_attention_finished,
        }.items():
            if not hasattr(mod, _name):
                setattr(mod, _name, _val)
    except Exception as e:
        logger.warning(f"GSA mla_v1 injection failed {e}")

    if hasattr(mod.AscendMLAImpl, "forward"):
        _patched = FunctionType(
            forward.__code__,
            vars(mod),
            name=forward.__name__,
            argdefs=forward.__defaults__,
            closure=forward.__closure__,
        )
        _patched.__annotations__ = forward.__annotations__
        _patched.__kwdefaults__ = forward.__kwdefaults__
        patch_or_inject(mod.AscendMLAImpl, "forward", _patched)
        logger.info("UCM GSA AscendMLAImpl.forward patch applied")
