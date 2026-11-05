# SPDX-License-Identifier: MIT
"""Ascend UCM KV-cache compatibility check without checkpoint weights.

This launcher shares platform-neutral checking logic with ``common.py``. It
does not load a checkpoint and it does not
hand-write model KV-cache shapes.

The cache construction path is "runner-native no-weight": construct the vLLM
model structure without checkpoint weights, instantiate the production
ModelRunner, ask the runner for KVCacheSpec/KVCacheGroup/KVCacheConfig, then
call ModelRunner.initialize_kv_cache() to produce the real worker KV-cache
tensors. This keeps model-specific cache shape
logic inside vLLM/vLLM-Ascend instead of re-implementing model rules here,
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
import importlib
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
        "npu",
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
    warmup, compile, or graph capture. It constructs the model under
    ``torch.device("meta")`` so vLLM/vLLM-Ascend populate the static forward
    context used by the production ModelRunner KV path.
    """

    # Match NPUWorker's no-weight-safe setup before model construction.
    from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm_ascend import ops
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
    from vllm_ascend.utils import (
        AscendDeviceType,
        adapt_patch,
        get_ascend_device_type,
        register_ascend_customop,
    )

    adapt_patch()
    ops.register_dummy_fusion_op()
    if get_ascend_device_type() != AscendDeviceType.A5:
        _register_atb_extensions()
    register_ascend_customop(vllm_config)
    init_ascend_config(vllm_config)
    init_vllm_config = vllm_config
    with current_vllm_config_context(init_vllm_config):
        init_ascend_model_parallel(init_vllm_config.parallel_config)

    previous_dtype = torch.get_default_dtype()
    try:
        # vLLM's real load path constructs modules under the configured model
        # dtype. Keep that constructor behavior while replacing the device by meta.
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


def select_model_runner_cls() -> type[Any]:
    """Return Ascend's production NPUModelRunner class."""
    try:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    except Exception as exc:
        raise UnsupportedEnvironment(
            "Ascend checks require vllm_ascend.worker.model_runner_v1.NPUModelRunner "
            "so KV tensors are initialized through the real NPU runner path. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return NPUModelRunner


def make_runner(
    vllm_config: Any,
    model: Any,
    device: torch.device,
) -> Any:
    """Instantiate the production ModelRunner without loading checkpoint weights."""

    runner_cls = select_model_runner_cls()
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
    return runner


def make_cache(vllm_config: Any, active_device: torch.device) -> CacheFixture:
    return make_common_cache(
        vllm_config,
        active_device,
        tokens,
        block_size,
        "hccl",
        torch.npu.synchronize,
        make_model,
        make_runner,
    )


def patch_groups(fixture: CacheFixture) -> None:
    """Let vLLM's single-group coordinator create managers for Ascend UniformType specs.

    vLLM-Ascend's external-store coordinator handles UniformTypeKVCacheSpecs by
    unwrapping the scheduler-equivalent inner spec before selecting a manager.
    The patched platform coordinator, however, delegates single-group configs
    back to upstream vLLM, whose module-level manager factory does not know this
    Ascend wrapper.  Patch that factory binding for this standalone checker so
    the native KVCacheManager allocation follows the Ascend manager path without
    modifying KVCacheConfig or UCM's real tensor/layout inputs.
    """

    try:
        import vllm.v1.core.kv_cache_coordinator as kv_cache_coordinator
        import vllm.v1.core.single_type_kv_cache_manager as single_type_manager
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
        from vllm_ascend.core.single_type_kv_cache_manager import (
            get_manager_for_kv_cache_spec as ascend_get_manager_for_kv_cache_spec,
        )
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
            _unwrap_spec as ascend_unwrap_uniform_spec,
        )
    except Exception as exc:
        raise UnsupportedEnvironment(
            "Ascend UniformTypeKVCacheSpecs were detected, but the checker "
            "could not import vLLM-Ascend's native UniformType unwrapping and "
            f"manager factory. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    has_uniform_group = any(
        isinstance(getattr(group, "kv_cache_spec", None), UniformTypeKVCacheSpecs)
        for group in fixture.kv_cache_config.kv_cache_groups
    )
    if not has_uniform_group:
        return

    current_factory = getattr(kv_cache_coordinator, "get_manager_for_kv_cache_spec")
    if getattr(current_factory, "_ucm_checker_ascend_uniform_patch", False):
        return

    def checker_ascend_get_manager_for_kv_cache_spec(
        kv_cache_spec: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            kv_cache_spec = ascend_unwrap_uniform_spec(kv_cache_spec)
        return ascend_get_manager_for_kv_cache_spec(kv_cache_spec, *args, **kwargs)

    checker_ascend_get_manager_for_kv_cache_spec._ucm_checker_ascend_uniform_patch = True  # type: ignore[attr-defined]

    kv_cache_coordinator.get_manager_for_kv_cache_spec = checker_ascend_get_manager_for_kv_cache_spec  # type: ignore[assignment]
    single_type_manager.get_manager_for_kv_cache_spec = checker_ascend_get_manager_for_kv_cache_spec  # type: ignore[assignment]
    log(
        "installed checker patch: upstream single-group KVCacheCoordinator "
        "will create managers for Ascend UniformTypeKVCacheSpecs through "
        "vLLM-Ascend's native unwrap + manager factory"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def check_config() -> None:
    if tokens <= 0 or block_size <= 0:
        raise ValueError("tokens and block_size must be positive")


def main() -> int:
    check_config()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    active_device = torch.device("npu:0")
    importlib.import_module("torch_npu")
    torch.npu.set_device(active_device)
    log(f"ASCEND_RT_VISIBLE_DEVICES={visible_devices}, device={active_device}")
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
        torch.npu.set_device(active_device)
        dispatch = schedule(fixture, tokens, request_token_salt, patch_groups)
        verify(fixture, dispatch, worker, torch.npu.synchronize)
        return 0
    finally:
        del dispatch, worker, fixture, vllm_config
        gc.collect()
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel

        destroy_ascend_model_parallel()
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.npu.synchronize(active_device)
        torch.npu.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
