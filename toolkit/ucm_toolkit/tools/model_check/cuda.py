# SPDX-License-Identifier: MIT
"""CUDA UCM KV-cache compatibility check without checkpoint weights.

This launcher shares platform-neutral checking logic with ``common.py``. It
does not load a checkpoint and it does not
hand-write model KV-cache shapes.

The cache construction path is "runner-native no-weight": construct the vLLM
model structure without checkpoint weights, instantiate the production
ModelRunner, ask the runner for KVCacheSpec/KVCacheGroup/KVCacheConfig, then
call ModelRunner.initialize_kv_cache() to produce the real worker KV-cache
tensors. This keeps model-specific cache shape
logic inside vLLM instead of re-implementing model rules here,
while bounding the actual allocation with num_gpu_blocks_override.

The probe constructs a real vLLM Scheduler, adds one synthetic request and
calls Scheduler.schedule(). The Scheduler-owned KVCacheManager performs the
per-group allocation and SchedulerOutput supplies the UCM metadata. The source
KV blocks are filled with deterministic values, dumped by UCM, loaded into a
distinct Scheduler-allocated target block table, and compared byte-for-byte.

The four main sections are:

1. make_config: build the real vLLM configuration without an LLM engine.
2. make_cache: construct the no-weight model and production
   ModelRunner, ask the runner to generate KVCacheSpec/KVCacheGroup/
   KVCacheConfig, then allocate/reshape KV tensors through
   ModelRunner.initialize_kv_cache().
3. schedule: submit one vLLM Request to the production
   Scheduler; its KVCacheManager allocates the group-shaped block tables and
   its SchedulerOutput carries the UCM metadata.
4. verify: call UCM with that metadata and verify its
   store/load preserves the selected KV tensor blocks.

Run after editing the "User configuration" and "UCM configuration" sections::

    ucm-toolkit run model-check --model /models/Qwen2.5-14B-Instruct

The installed vLLM must expose the V1 APIs used by UCM. API absence is treated
as an unsupported environment rather than silently falling back to a
hand-written cache layout.
"""

from __future__ import annotations

import gc
import os
import time
import traceback
from typing import Any

import torch
from .config import load_config
from .common import (
    CacheFixture,
    UnsupportedEnvironment,
    current_vllm_config_context,
    log,
)
from .common import make_cache as make_common_cache
from .common import make_config as make_common_config
from .common import (
    make_worker,
    no_real_device_move_from_meta_context,
    schedule,
    verify,
)

# =========================== User configuration ===========================
config = load_config()
model = config.model
tokens = config.tokens
block_size = config.block_size
use_layerwise = config.use_layerwise
additional_config = config.additional_config
store_pipeline = config.store_pipeline
storage_backends = config.storage_backends
visible_devices = config.visible_devices
dtype = config.dtype
kv_cache_dtype = config.kv_cache_dtype
trust_remote_code = True
request_token_salt = time.time_ns() ^ os.getpid()


def _factory_kwargs_redirect_to_meta(kwargs: dict[str, Any]) -> dict[str, Any]:
    rewritten_kwargs = dict(kwargs)
    device_arg = rewritten_kwargs.get("device")
    if device_arg is None:
        # ``torch.device(\"meta\")`` is normally sufficient for factories
        # without an explicit device.  The checker wraps those factories,
        # though, and some PyTorch/vLLM combinations then bypass that ambient
        # device context.  Make the no-weight contract explicit: every factory
        # allocation made while constructing the model stays on meta.
        rewritten_kwargs["device"] = torch.device("meta")
        return rewritten_kwargs

    try:
        target_device = torch.device(device_arg)
    except Exception:
        return rewritten_kwargs
    if target_device.type != "meta":
        rewritten_kwargs["device"] = torch.device("meta")
    return rewritten_kwargs


# ---------------------------------------------------------------------------
# 1. Construct VllmConfig
# ---------------------------------------------------------------------------


def make_config() -> Any:
    return make_common_config(
        model,
        tokens,
        block_size,
        dtype,
        kv_cache_dtype,
        trust_remote_code,
        additional_config,
        store_pipeline,
        storage_backends,
        use_layerwise,
        "cuda",
    )


# ---------------------------------------------------------------------------
# 2. Generate the real vLLM KV-cache size, groups, tensors, shape and stride
# ---------------------------------------------------------------------------


def make_model(
    vllm_config: Any,
) -> tuple[Any, Any]:
    """Build the production model structure without checkpoint weights.

    The checker intentionally does not call worker.load_model(),
    model_runner.load_model(), get_model_loader().load_model(), profile_run(),
    warmup, compile, or cudagraph capture.  It constructs the model under
    ``torch.device("meta")`` so vLLM populate the static forward
    context used by the production ModelRunner KV path.
    """

    from vllm.model_executor.model_loader.utils import initialize_model

    init_vllm_config = vllm_config

    previous_dtype = torch.get_default_dtype()
    try:
        # vLLM's real load path constructs modules under the configured model
        # dtype. Keep that constructor behavior while replacing CUDA by meta.
        torch.set_default_dtype(init_vllm_config.model_config.dtype)
        init_vllm_config.compilation_config.static_forward_context.clear()
        with (
            current_vllm_config_context(init_vllm_config),
            no_real_device_move_from_meta_context(_factory_kwargs_redirect_to_meta),
            torch.device("meta"),
        ):
            model = initialize_model(init_vllm_config, prefix="")
    except Exception as exc:
        raise UnsupportedEnvironment(
            "This model/vLLM combination cannot construct its model structure on "
            "the meta device after applying the relevant platform/worker patches. "
            "The checker will not fall back to loading checkpoint weights into "
            f"HBM. Original error: {type(exc).__name__}: {exc}\n"
            f"Original traceback:\n{traceback.format_exc()}"
        ) from exc
    finally:
        torch.set_default_dtype(previous_dtype)

    return model, init_vllm_config


def select_model_runner_cls(vllm_config: Any) -> type[Any]:
    """Return CUDA's production ModelRunner class."""

    if bool(getattr(vllm_config, "use_v2_model_runner", False)):
        try:
            from vllm.v1.worker.gpu.model_runner import GPUModelRunner

            return GPUModelRunner
        except Exception as exc:
            raise UnsupportedEnvironment(
                "CUDA V2 model-runner mode is enabled, but the installed vLLM "
                "does not expose vllm.v1.worker.gpu.model_runner.GPUModelRunner. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as exc:
        raise UnsupportedEnvironment(
            "CUDA checks require vllm.v1.worker.gpu_model_runner.GPUModelRunner "
            "so KV tensors are initialized through the real GPU runner path. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return GPUModelRunner


def make_runner(
    vllm_config: Any,
    model: Any,
    device: torch.device,
) -> Any:
    """Instantiate the production ModelRunner without loading checkpoint weights."""

    runner_cls = select_model_runner_cls(vllm_config)
    try:
        with current_vllm_config_context(vllm_config):
            runner = runner_cls(vllm_config, device)
    except Exception as exc:
        raise UnsupportedEnvironment(
            "Production ModelRunner construction failed before any checkpoint "
            "loader was called. This usually means a platform patch, meta-safe "
            "model-construction workaround, or minimal scheduler config is still "
            f"missing. Original error: {type(exc).__name__}: {exc}\n"
            f"Original traceback:\n{traceback.format_exc()}"
        ) from exc

    # load_model() normally installs this attribute.  The checker has already
    # constructed the same module hierarchy on meta, so attach it explicitly for
    # runner code that needs model methods during KV initialization (e.g. Mamba).
    runner.model = model
    if bool(getattr(vllm_config, "use_v2_model_runner", False)):
        # V2 initializes these after loading checkpoint weights.  They depend on
        # model structure, not tensor values, so create them from the meta model
        # before calling its production KV-cache initialization path.
        from vllm.v1.worker.gpu.model_states import init_model_state

        runner.model_state = init_model_state(
            vllm_config, model, runner.encoder_cache, device
        )
        runner.decode_query_len = (
            runner.num_speculative_steps
            + runner.model_state.num_new_sampled_tokens_per_step
        )
    return runner


def make_cache(vllm_config: Any, active_device: torch.device) -> CacheFixture:
    return make_common_cache(
        vllm_config,
        active_device,
        tokens,
        block_size,
        "nccl",
        torch.cuda.synchronize,
        make_model,
        make_runner,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def check_config() -> None:
    if tokens <= 0 or block_size <= 0:
        raise ValueError("tokens and block_size must be positive")


def main() -> int:
    check_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    active_device = torch.device("cuda:0")
    torch.cuda.set_device(active_device)
    log(f"CUDA_VISIBLE_DEVICES={visible_devices}, device={active_device}")
    vllm_config = None
    fixture = None
    worker = None
    dispatch = None
    try:
        vllm_config = make_config()
        fixture = make_cache(vllm_config, active_device)
        # UCM's MLA shared buffer is created by the worker.  The scheduler reads
        # the worker-published id, so initialize the worker before Scheduler.
        worker = make_worker(fixture)
        dispatch = schedule(fixture, tokens, request_token_salt)
        verify(fixture, dispatch, worker, torch.cuda.synchronize)
        return 0
    finally:
        del dispatch, worker, fixture, vllm_config
        gc.collect()
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.synchronize(active_device)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
