from .vllm_patch import _apply_vllm_patches
from .vllm_ascend_patch import _apply_ascend_patches

__all__ = ["_apply_vllm_patches", "_apply_ascend_patches"]
