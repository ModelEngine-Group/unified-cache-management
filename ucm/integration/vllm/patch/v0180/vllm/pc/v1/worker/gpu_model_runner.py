from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec, MambaSpec
from vllm.v1.worker.cp_utils import get_total_cp_world_size
from vllm.v1.worker.gpu_input_batch import InputBatch


class GPUModelRunner:
    def may_reinitialize_input_batch(
        self, kv_cache_config, kernel_block_sizes: list[int]
    ) -> None:
        """
        Re-initialize InputBatch after final KV cache block sizes are known.

        For Mamba groups, only collapse to a single block when the actual
        mamba_cache_mode is "none"; align/all modes need the normal block count
        even if prefix caching is disabled.
        """
        block_sizes = []
        max_num_blocks = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            block_size = kv_cache_group.kv_cache_spec.block_size
            block_sizes.append(block_size)
            max_num_blocks_per_req = cdiv(
                max_model_len, block_size * get_total_cp_world_size()
            )
            if isinstance(kv_cache_group.kv_cache_spec, MambaSpec):
                if kv_cache_group.kv_cache_spec.mamba_cache_mode == "none":
                    max_num_blocks_per_req = 1
                max_num_blocks_per_req += (
                    kv_cache_group.kv_cache_spec.num_speculative_blocks
                )
            max_num_blocks.append(max_num_blocks_per_req)

        if (
            block_sizes != self._init_block_sizes
            or kernel_block_sizes != self._init_kernel_block_sizes
        ):
            assert self.offload_config.uva.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See "
                "https://github.com/vllm-project/vllm/pull/18298 "
                "for more details."
            )
            self._init_block_sizes = block_sizes
            self._init_kernel_block_sizes = kernel_block_sizes
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=(
                    self.input_batch.logitsprocs_need_output_token_ids
                ),
                is_pooling_model=self.is_pooling_model,
            )

        assert self._init_block_sizes == block_sizes, (
            f"InputBatch block_sizes {self._init_block_sizes} != "
            f"kv_cache block_sizes {block_sizes}"
        )
        assert self._init_kernel_block_sizes == kernel_block_sizes, (
            f"InputBatch kernel_block_sizes {self._init_kernel_block_sizes} "
            f"!= kv_cache kernel_block_sizes {kernel_block_sizes}"
        )
