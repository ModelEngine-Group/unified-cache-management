import torch
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.mla_v1 import (
    MAX_O_PROJ_PREFETCH_SIZE,
    MLAPO_MAX_SUPPORTED_TOKENS,
    M,
)
from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.layer_shard_linear import (
    is_hidden_layer,
    reach_layer_for_shard_weight_series,
)
from vllm_ascend.utils import get_weight_prefetch_method


class AscendMLAImpl:
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

        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens
        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        o_proj_input_shape = (_EXTRA_CTX.num_tokens, self.num_heads * self.v_head_dim)
        o_proj_input = torch.zeros(
            o_proj_input_shape, dtype=hidden_states.dtype, device=hidden_states.device
        )

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

        if prefill_preprocess_res is not None:
            # FIX: aicore move should be also placed on the comm stream in dbo,
            # otherwise it may affect the accuracy
            # TODO: use an elegant way to overlap
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
        if has_prefill:
            maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
        return output_padded
