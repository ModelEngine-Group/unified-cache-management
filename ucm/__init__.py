from ucm.integration.vllm.ucm_connector import UCMConnector

try:
    from ucm.integration.vllm.patch.apply_patch import ensure_patches_applied

    ensure_patches_applied()
except Exception as e:
    # Don't fail if patches can't be applied - might be running in environment without vLLM
    import warnings

    warnings.warn(
        f"Failed to apply vLLM patches: {e}. "
        f"If you're using vLLM, ensure it's installed and patches are compatible."
    )

__all__ = ["UCMConnector"]
