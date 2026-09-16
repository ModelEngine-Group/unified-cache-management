"""Install the MiniMax M3 prefill backport for vllm-ascend 0.26.0rc1."""

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.models.minimax_m3.msa_m3")
def patch_minimax_m3_prefill(mod):
    if getattr(mod, "_ucm_m3_prefill_patched", False):
        return

    from vllm_ascend.models.minimax_m3.ops import msa_m3_triton

    kernel = getattr(msa_m3_triton, "_prepare_prefill_topk_scores_kernel", None)
    parameters = set(getattr(kernel, "arg_names", ()))
    if {"MAX_QUERY_LEN", "PROGRAMS_PER_BATCH_HEAD"} <= parameters:
        logger.info("UCM MiniMax M3 prefill: upstream #16121 already present; skipping")
        return
    if not {"BLOCK_SIZE_Q", "BLOCK_SIZE_TAIL", "BLOCK_SIZE_FORCE"} <= parameters:
        raise RuntimeError(
            "UCM MiniMax M3 prefill backport requires the vllm-ascend 0.26.0rc1 "
            "score preparation interface; check the installed source."
        )

    from ucm.integration.vllm.patch.v0260.vllm_ascend import minimax_m3_prefill

    # Replace both from-import aliases: the new kernel depends on score
    # allocation initializing the invalid tail to -inf.
    mod.minimax_m3_index_score = minimax_m3_prefill.minimax_m3_index_score
    mod.minimax_m3_index_topk = minimax_m3_prefill.minimax_m3_index_topk
    mod._ucm_m3_prefill_patched = True
    logger.info("UCM MiniMax M3 prefill backport applied (vllm-ascend PR #16121)")
