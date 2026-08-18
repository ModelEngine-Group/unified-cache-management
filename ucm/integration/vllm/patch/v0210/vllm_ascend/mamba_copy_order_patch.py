"""Fix mamba align copy ordering on vLLM-Ascend >= 0.21.0.

vLLM-Ascend's ``patch_mamba_utils`` defers ``do_mamba_copy_block`` from
``preprocess_mamba`` (which runs in ``_prepare_inputs``, *before*
``start_load_kv``) to inside the forward context (*after* ``start_load_kv``).

This reversal causes the copy to overwrite UCM-loaded KV cache data with
zeros from the freshly-allocated old block.

This patch wraps ``preprocess_mamba`` to execute ``do_mamba_copy_block``
immediately (restoring upstream vLLM behavior), then resets
``copy_bufs.offset = 0`` so the later call in ``model_runner_v1.py``
becomes a no-op (``do_mamba_copy_block`` already checks ``offset == 0``).

Resulting execution order:
  1. ``preprocess_mamba`` → ``do_mamba_copy_block`` (compute stream)
  2. ``start_load_kv`` → ``device.synchronize()`` → load DMA (store stream)
  3. ``model_runner`` → ``do_mamba_copy_block`` (no-op, offset == 0)
  4. ``_model_forward``
"""

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.patch.worker.patch_mamba_utils")
def patch_mamba_copy_order(mod):
    """Wrap preprocess_mamba to execute do_mamba_copy_block immediately."""
    try:
        from vllm.v1.worker import mamba_utils
    except ImportError:
        logger.warning(
            "Skip mamba copy order patch: vllm.v1.worker.mamba_utils not found"
        )
        return

    if not hasattr(mamba_utils, "do_mamba_copy_block"):
        logger.warning("Skip mamba copy order patch: do_mamba_copy_block not found")
        return

    original_preprocess = mamba_utils.preprocess_mamba
    do_mamba_copy_block = mamba_utils.do_mamba_copy_block

    if getattr(original_preprocess, "_ucm_copy_order_patched", False):
        return

    def patched_preprocess(*args, **kwargs):
        original_preprocess(*args, **kwargs)
        if args:
            copy_bufs = args[-1]
        else:
            copy_bufs = kwargs.get("copy_bufs")
        if copy_bufs is not None and copy_bufs.offset > 0:
            do_mamba_copy_block(copy_bufs)
            copy_bufs.offset = 0

    patched_preprocess._ucm_copy_order_patched = True
    mamba_utils.preprocess_mamba = patched_preprocess
    logger.info(
        "UCM mamba copy order patch applied: do_mamba_copy_block restored "
        "to preprocess_mamba (before start_load_kv)"
    )
