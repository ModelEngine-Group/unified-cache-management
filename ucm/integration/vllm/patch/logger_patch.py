import logging
import os

from ucm.integration.vllm.patch.utils import when_imported


def _capture_enabled() -> bool:
    value = os.getenv("UCM_CAPTURE_VLLM_LOG", "1").strip().lower()
    return value in ("1", "true", "yes", "on")


@when_imported("vllm.logger")
def patch_logger(mod):
    if not _capture_enabled():
        return

    from ucm.logger import UcmBridgeHandler, get_vllm_capture_handler

    vllm_root = logging.getLogger("vllm")
    if any(isinstance(h, UcmBridgeHandler) for h in vllm_root.handlers):
        return
    handler = get_vllm_capture_handler()
    if handler is not None:
        vllm_root.addHandler(handler)
