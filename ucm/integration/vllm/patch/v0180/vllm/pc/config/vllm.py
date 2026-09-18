def validate_mamba_block_size(self):
    return self


def patch_vllm_config_cls(vllm_config_cls: type) -> None:
    """Relax mamba_block_size validation for UCM external KV usage.

    Pydantic dataclasses keep a reference to the validator function in the
    compiled schema, so mutate the existing function object when possible.
    """
    current = getattr(vllm_config_cls, "validate_mamba_block_size", None)
    current_func = getattr(current, "__func__", current)
    patched_in_place = False

    if hasattr(current_func, "__code__"):
        try:
            current_func.__code__ = validate_mamba_block_size.__code__
            current_func.__defaults__ = validate_mamba_block_size.__defaults__
            current_func.__kwdefaults__ = validate_mamba_block_size.__kwdefaults__
            patched_in_place = True
        except (AttributeError, TypeError, ValueError):
            patched_in_place = False

    if not patched_in_place:
        setattr(
            vllm_config_cls,
            "validate_mamba_block_size",
            validate_mamba_block_size,
        )

    try:
        from pydantic.dataclasses import rebuild_dataclass

        rebuild_dataclass(vllm_config_cls, force=True)
    except Exception:
        # Older/newer pydantic builds may not support rebuild for this class.
        # In-place validator mutation above is enough for the compiled schema.
        pass
