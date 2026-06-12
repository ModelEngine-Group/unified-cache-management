import vllm.model_executor.models.config


def patch_hybrid_attention_mamba_model_config() -> None:
    original = (
        vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config
    )

    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        original(vllm_config)

        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "align":
            cache_config.mamba_block_size = cache_config.block_size

    vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = (
        verify_and_update_config
    )
