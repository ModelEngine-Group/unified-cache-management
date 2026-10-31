"""Initialize UCM sparse state in vLLM 0.23.0 ``gpu_worker``.

Replicates the ``vllm/v1/worker/gpu_worker.py`` hunk of
``vllm-adapt-1020-c8.patch`` at runtime so no ``git apply`` is needed. The
upstream ``init_worker_distributed_environment`` gains a trailing
``ensure_ucm_sparse_initialized(vllm_config)`` call.
"""

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _make_patched_init(orig):
    def init_worker_distributed_environment(*args, **kwargs):
        from ucm.sparse.state import ensure_ucm_sparse_initialized

        # vllm_config is the last positional arg in the upstream signature.
        vllm_config = args[-1] if args else kwargs.get("vllm_config")
        orig(*args, **kwargs)
        if vllm_config is not None:
            ensure_ucm_sparse_initialized(vllm_config)

    return init_worker_distributed_environment


@when_imported("vllm.v1.worker.gpu_worker")
def _patch_gpu_worker(mod):
    fn = getattr(mod, "init_worker_distributed_environment", None)
    if fn is None:
        logger.warning(
            "Skip gpu_worker sparse init patch: "
            "init_worker_distributed_environment missing"
        )
        return
    patch_or_inject(mod, "init_worker_distributed_environment", _make_patched_init(fn))
    logger.info("UCM GSA gpu_worker init_worker_distributed_environment patch applied")
