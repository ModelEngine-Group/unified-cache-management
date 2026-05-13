def patch_preprocess_mamba(mod) -> None:
    original = getattr(mod, "_ucm_original_preprocess_mamba", None)
    if original is None:
        original = mod.preprocess_mamba
        setattr(mod, "_ucm_original_preprocess_mamba", original)

    def preprocess_mamba(*args, **kwargs):
        cache_config = kwargs.get("cache_config")
        if cache_config is None and len(args) > 2:
            cache_config = args[2]

        if cache_config is None:
            return original(*args, **kwargs)

        original_enable_prefix_caching = cache_config.enable_prefix_caching
        cache_config.enable_prefix_caching = True
        try:
            return original(*args, **kwargs)
        finally:
            cache_config.enable_prefix_caching = original_enable_prefix_caching

    mod.preprocess_mamba = preprocess_mamba
