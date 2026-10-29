# SPDX-License-Identifier: MIT
"""Ascend layout check without an NPU (vLLM-Ascend installed, no device).

Same flow as ``ascend.py``: vLLM-Ascend's own model construction, KVCacheSpec,
grouping, KVCacheConfig and ``NPUModelRunner.initialize_kv_cache()`` produce
the KV layout, and the real Scheduler allocates block tables. Only *where*
the KV tensors live changes: they are allocated in host memory, and UCM must
be built with the ``simu`` runtime (``PLATFORM`` unset) so its transfers are
host memcpy.

``cuda_sim.py`` is the same idea for vLLM's official (CUDA) layout; the
substitutions both launchers share live in ``sim_runtime.py``.

What is substituted, and why it does not change the layout:

* device selection, streams, events and synchronization are host no-ops --
  they order execution, they do not shape tensors;
* triton's Ascend arch is pinned through its own ``TRITON_ASCEND_ARCH``
  override, because vLLM-Ascend's import chain reads the arch at module level;
* kernel/extension registration that needs a device is skipped -- no kernel
  runs in this checker;
* UCM's worker device is the host ``CpuDevice``.

The SoC family is **not** simulated: vLLM-Ascend reports it from
``vllm_ascend._build_info.__device_type__``, i.e. the family its wheel was
built for, so this launcher only reports it and, when ``--ascend-soc`` is
given, asserts the match. Checking another family's layout needs a
vLLM-Ascend built for that family.

KV-cache specs, group/config values, tensor shapes, strides, dtypes and
block IDs are never replaced. A PASS shows that UCM parses and round-trips
vLLM-Ascend's layout; it does not cover NPU copy kernels, ACL event ordering
or performance.

Run::

    ucm-toolkit run model-check --platform ascend-sim --ascend-soc A2 \\
        --model /path/to/model --tokens 1024 --block-size 128 \\
        --storage-backends /path/to/ucm_storage
"""

from __future__ import annotations

import gc
import importlib
import os
from typing import Any

import torch

from .common import CacheFixture, UnsupportedEnvironment, describe_ucm_layout, log
from .common import make_cache as make_common_cache
from .common import make_worker, schedule, verify
from .config import load_config
from .sim_runtime import (
    HOST,
    disable_pin_memory,
    patch_execution_ordering,
    patch_ucm_device,
    rebind,
    require_platform,
)

config = load_config()


def _pin_triton_arch() -> None:
    """Pin triton's Ascend arch before vLLM-Ascend is imported.

    ``vllm_ascend.patch.platform`` imports vLLM's flash-linear-attention ops,
    whose module body asks triton for the current target. Without an NPU that
    call fails inside CANN, so use triton's own documented override. The image
    declares the family in ``SOC_VERSION``; nothing here is inferred from a
    device.
    """

    if os.getenv("TRITON_ASCEND_ARCH"):
        return
    soc_version = os.getenv("SOC_VERSION", "")
    prefix = "ascend"
    if not soc_version.lower().startswith(prefix):
        raise UnsupportedEnvironment(
            "ascend-sim needs the triton Ascend arch: set TRITON_ASCEND_ARCH "
            "(e.g. Ascend910B1) or provide SOC_VERSION. Without an NPU, triton "
            "cannot read the arch from the device."
        )
    arch = "Ascend" + soc_version[len(prefix) :].upper()
    try:
        from triton.backends.ascend.utils import get_ascend_arch_from_env

        os.environ["TRITON_ASCEND_ARCH"] = arch
        get_ascend_arch_from_env()  # triton validates its own override here
    except ImportError:
        os.environ["TRITON_ASCEND_ARCH"] = arch
    log(f"ascend-sim: TRITON_ASCEND_ARCH={arch} (from SOC_VERSION={soc_version})")


def _patch_npu_runtime() -> None:
    """Make torch.npu's execution-ordering API host no-ops."""

    importlib.import_module("torch_npu")
    patch_execution_ordering(torch.npu, "ascend-sim")


def _check_ascend_soc() -> None:
    """Report the installed package's SoC family and verify the request.

    ``get_ascend_device_type()`` reads the wheel's own build info, so it works
    without an NPU -- and cannot be turned into another family: the layout,
    kernels and quantization of a different SoC live in a different wheel.
    """

    utils = importlib.import_module("vllm_ascend.utils")
    package_soc = utils.get_ascend_device_type()
    requested = config.ascend_soc
    if requested is None:
        log(f"ascend-sim: SoC={package_soc.name} (from the installed package)")
        return
    try:
        requested_soc = utils.AscendDeviceType[requested.upper()]
    except KeyError as exc:
        supported = sorted(utils.AscendDeviceType.__members__)
        raise UnsupportedEnvironment(
            f"unknown --ascend-soc {requested!r}; expected one of {supported}"
        ) from exc
    if requested_soc is not package_soc:
        raise UnsupportedEnvironment(
            f"--ascend-soc {requested_soc.name} does not match the installed "
            f"vLLM-Ascend, which is built for {package_soc.name}. The SoC "
            "family comes from the package "
            "(vllm_ascend._build_info.__device_type__), not from a device, so "
            f"this installation cannot produce the {requested_soc.name} "
            f"layout. Install a vLLM-Ascend built for {requested_soc.name}."
        )
    log(f"ascend-sim: SoC={package_soc.name} (matches --ascend-soc)")


def _disable_custom_ops() -> None:
    """Report custom ops as unavailable instead of loading their extension.

    ``enable_custom_op()`` loads the ``vllm_ascend_C`` kernel extension, which
    aborts the process (a C++ ``std::logic_error``, not a Python exception)
    when no device is present. Priming its own cache yields exactly the state
    vLLM-Ascend uses where runtime custom ops are unavailable -- it already
    takes that path on SoCs that disable them. Custom ops are compute kernels
    and none runs here; a kernel-dependent op *selection* difference would
    still show up in the logged layout.
    """

    utils = importlib.import_module("vllm_ascend.utils")
    utils._CUSTOM_OP_ENABLED = False
    log("ascend-sim: custom ops reported unavailable (no kernel is executed)")


class _MetaUvaBuffer:
    """Weight-side stand-in for Engram's pinned-host codebook buffer.

    The real buffer is allocated with ``aclrtMallocHost`` and published to the
    device, which needs CANN plus tens of GiB of pinned memory. Engram maps
    these buffers to ``weight`` / ``weight_scale_inv``, so they are checkpoint
    weights -- exactly what this checker never loads. The stand-in keeps the
    shape and dtype on the meta device and consumes no memory. Nothing
    dereferences the addresses because no kernel runs.
    """

    def __init__(self, shape, dtype, device, *args: Any, **kwargs: Any) -> None:
        row_elements = int(torch.Size(tuple(shape)[1:]).numel())
        self.row_bytes = row_elements * torch.empty((), dtype=dtype).element_size()
        self.tensor = torch.empty(tuple(shape), dtype=dtype, device="meta")
        self.ptrs = torch.zeros(1, dtype=torch.int64)
        self.pointer = None

    def close(self) -> None:
        self.tensor = None
        self.ptrs = None


_UVA_BUFFER_NAMES = ("HostUvaBuffer", "SharedUvaBuffer")


def _stub_engram_weight_buffers() -> None:
    """Keep Engram's codebook weights on meta (no pinned host allocation).

    Registered as a post-import hook rather than an eager import: importing
    vLLM-Ascend's model subpackages out of order leaves its ops package
    half-initialized, which the native import chain then reports as a circular
    import. The hook runs right after the native module finishes executing, so
    importers that do ``from .npu import HostUvaBuffer`` see the stand-in.
    """

    import wrapt

    def patch(module: Any) -> None:
        replaced = [name for name in _UVA_BUFFER_NAMES if hasattr(module, name)]
        for name in replaced:
            setattr(module, name, _MetaUvaBuffer)
        if replaced:
            log(
                f"ascend-sim: {module.__name__} {sorted(replaced)} kept on meta "
                "(Engram codebook weights are out of this check's scope)"
            )

    for module_name in (
        "vllm_ascend.models.deepseek_v41.engram.npu",
        "vllm_ascend.models.deepseek_v41.engram.embedding",
    ):
        wrapt.when_imported(module_name)(patch)


def _stub_device_property_queries() -> None:
    """Replace AI-core-count queries used only as kernel launch parameters.

    ``initialize_kv_cache`` asks DeepSeek-V4.1's ring compressor to resolve its
    hardware, which reads the cube-core count from triton. That count is
    consumed only as the ``num_cores=`` launch parameter of a compressor
    kernel: it allocates nothing and sets no shape, and no kernel runs here.
    The placeholder is logged so a reader never mistakes it for the real
    hardware value.
    """

    import wrapt

    def patch(module: Any) -> None:
        module._cube_core_num = lambda: 1
        log(
            f"ascend-sim: {module.__name__}._cube_core_num() -> 1 placeholder "
            "(kernel launch parameter only; no kernel runs)"
        )

    wrapt.when_imported("vllm_ascend.ops.triton.compressor.compressor_triton")(patch)


def install_host_runtime() -> None:
    """Apply every substitution, in the order the import chain needs them."""

    _pin_triton_arch()
    _patch_npu_runtime()
    require_platform(
        "npu",
        "ascend-sim",
        "Check that vLLM-Ascend is installed and that VLLM_PLUGINS does not "
        "disable its platform plugin.",
    )
    _check_ascend_soc()
    _disable_custom_ops()
    disable_pin_memory("ascend-sim")
    _stub_engram_weight_buffers()
    _stub_device_property_queries()
    patch_ucm_device("ascend-sim")


def make_cache(vllm_config: Any) -> CacheFixture:
    from . import ascend

    def make_runner(kv_vllm_config: Any, model: Any, _device: torch.device) -> Any:
        return ascend.make_runner(kv_vllm_config, model, HOST)

    return make_common_cache(
        vllm_config,
        HOST,
        config.tokens,
        config.block_size,
        "gloo",
        lambda *args: None,
        ascend.make_model,
        make_runner,
    )


def main() -> int:
    if config.tokens <= 0 or config.block_size <= 0:
        raise ValueError("tokens and block_size must be positive")
    install_host_runtime()
    from . import ascend

    log(f"ascend-sim: KV tensors on {HOST}, UCM store runtime must be simu")
    vllm_config = None
    fixture = None
    worker = None
    dispatch = None
    try:
        vllm_config = ascend.make_config()
        fixture = make_cache(vllm_config)
        worker = make_worker(fixture)
        if config.layout_only:
            describe_ucm_layout(worker)
            log("PASS: layout-only (UCM parsed the native KV layout)")
            return 0
        dispatch = schedule(
            fixture, config.tokens, ascend.request_token_salt, None, worker
        )
        verify(fixture, dispatch, worker, lambda: None)
        return 0
    finally:
        del dispatch, worker, fixture, vllm_config
        gc.collect()
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        try:
            from vllm_ascend.distributed.parallel_state import (
                destroy_ascend_model_parallel,
            )

            destroy_ascend_model_parallel()
        except Exception:
            pass
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    raise SystemExit(main())
