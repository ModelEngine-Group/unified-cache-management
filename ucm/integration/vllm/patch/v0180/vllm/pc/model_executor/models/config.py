def patch_mamba_model_config_cls(mamba_model_config_cls: type) -> None:
    """Keep explicit mamba_cache_mode even when prefix caching is disabled."""
    original = getattr(
        mamba_model_config_cls,
        "_ucm_original_verify_and_update_config",
        None,
    )
    if original is None:
        original = mamba_model_config_cls.verify_and_update_config
        setattr(
            mamba_model_config_cls,
            "_ucm_original_verify_and_update_config",
            original,
        )

    @classmethod
    def verify_and_update_config(cls, vllm_config):
        cache_config = vllm_config.cache_config
        requested_mamba_cache_mode = cache_config.mamba_cache_mode
        restore_mamba_cache_mode = (
            not cache_config.enable_prefix_caching
            and requested_mamba_cache_mode != "none"
        )

        original(vllm_config)

        if restore_mamba_cache_mode:
            cache_config.mamba_cache_mode = requested_mamba_cache_mode

    mamba_model_config_cls.verify_and_update_config = verify_and_update_config
