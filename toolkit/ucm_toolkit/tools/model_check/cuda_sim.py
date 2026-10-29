# SPDX-License-Identifier: MIT
"""Official (CUDA) layout check without a GPU (CUDA vLLM installed, no device).

Device-less counterpart of ``cuda.py``, and the CUDA sibling of
``ascend_sim.py``: vLLM's own model construction, ``KVCacheSpec``, grouping,
``KVCacheConfig`` and ``GPUModelRunner.initialize_kv_cache()`` produce the KV
layout, and the real Scheduler allocates the block tables. Only *where* the KV
tensors live changes -- they are allocated in host memory -- so UCM must be
built with the ``simu`` runtime (``PLATFORM`` unset), whose transfers are host
memcpy.

This needs a **CUDA build of vLLM** (the one that ships ``vllm.vllm_flash_attn``):
vLLM's attention backends import their kernel entry points at module level
behind ``if current_platform.is_cuda()``, so the real backend classes -- the
ones that compute the KV shapes and layouts -- are only importable there. A
CPU build cannot stand in for them; ``cpu.py`` exists for that case and
reproduces the CUDA layout through hand-declared backend constraints instead.

What is substituted, and why it does not change the layout:

* the CUDA platform probe: without a GPU, vLLM's NVML probe reports no device
  and the CPU platform would win, so the launcher declares that the CUDA
  platform is the one being simulated;
* the compute capability: it cannot be read from a device, so it must be given
  with ``--cuda-capability``. vLLM branches on it (FlashAttention version, fp8
  support, MLA backend choice), so it is **declared, never guessed** -- the
  same rule ``ascend_sim.py`` applies to the Ascend SoC family;
* streams, events, ``synchronize`` and allocator statistics are host no-ops --
  they order execution and report usage, they do not shape tensors;
* pinned host memory is reported unavailable;
* UCM's worker device is the host ``CpuDevice``.

KV-cache specs, group/config values, tensor shapes, strides, dtypes and block
IDs are never replaced. A PASS shows that UCM parses and round-trips vLLM's
official layout; it does not cover CUDA copy kernels, stream/event ordering or
performance. Because the capability is declared rather than measured, a PASS
is a statement about the layout vLLM produces *for that capability*.

Run::

    ucm-toolkit run model-check --platform cuda-sim --cuda-capability 9.0 \\
        --model /path/to/model --tokens 1024 --block-size 128 \\
        --storage-backends /path/to/ucm_storage

Add ``--layout-only`` to stop once UCM has parsed the layout.
"""

from __future__ import annotations

import gc
import importlib
from typing import Any

import torch

from .common import CacheFixture, UnsupportedEnvironment, describe_ucm_layout, log
from .common import make_cache as make_common_cache
from .common import make_worker, schedule, verify
from .config import load_config
from .sim_runtime import (
    HOST,
    REPORTED_MEMORY_BYTES,
    device_properties,
    disable_pin_memory,
    patch_execution_ordering,
    patch_ucm_device,
    rebind,
    require_platform,
)

config = load_config()

CUDA_PLATFORM_QUALNAME = "vllm.platforms.cuda.CudaPlatform"


def _parse_capability() -> tuple[int, int]:
    """Read the declared compute capability; refuse to invent one.

    Unlike vLLM-Ascend, whose wheel records the SoC family it was built for, a
    CUDA wheel targets many architectures and the capability normally comes
    from the device. With no device there is nothing to read, and the value
    changes which attention backend and which fp8 path vLLM selects -- so the
    caller has to say which GPU generation is being simulated.
    """

    declared = config.cuda_capability
    if not declared:
        raise UnsupportedEnvironment(
            "cuda-sim needs --cuda-capability (e.g. 8.0, 9.0, 10.0): without a "
            "GPU the compute capability cannot be read, and vLLM selects its "
            "attention backend and fp8 path from it. State the GPU generation "
            "whose layout you want to check instead of letting the checker "
            "guess one."
        )
    text = str(declared).strip()
    try:
        if "." in text:
            major_text, minor_text = text.split(".", 1)
        else:  # "90" style, as nvcc writes it
            major_text, minor_text = text[:-1], text[-1]
        major, minor = int(major_text), int(minor_text)
    except ValueError as exc:
        raise UnsupportedEnvironment(
            f"cannot read --cuda-capability {declared!r}; expected forms like "
            "'9.0' or '90'"
        ) from exc
    if major <= 0:
        raise UnsupportedEnvironment(
            f"--cuda-capability {declared!r} has a non-positive major version"
        )
    return major, minor


def _force_cuda_platform() -> None:
    """Declare the CUDA platform as the one being simulated.

    vLLM resolves its platform by running each built-in probe;
    ``cuda_platform_plugin`` asks NVML for a device count, so on a GPU-less
    host it reports nothing and the CPU platform wins. Replacing the probe map
    states which platform is being simulated -- it answers "is this hardware
    present", not "what does the layout look like". Must run before anything
    touches ``vllm.platforms.current_platform``, which resolves once and
    caches.
    """

    platforms = importlib.import_module("vllm.platforms")
    if getattr(platforms, "_current_platform", None) is not None:
        raise UnsupportedEnvironment(
            "vLLM already resolved its platform before cuda-sim could declare "
            "CUDA. Run this launcher as the process entry point so nothing "
            "imports vllm.platforms.current_platform first."
        )
    if not hasattr(platforms, "builtin_platform_plugins"):
        raise UnsupportedEnvironment(
            "the installed vLLM has no vllm.platforms.builtin_platform_plugins, "
            "so cuda-sim cannot declare which platform to simulate."
        )
    platforms.builtin_platform_plugins = {
        "cuda": lambda: CUDA_PLATFORM_QUALNAME,
    }
    log("cuda-sim: CUDA declared as the simulated platform (NVML probe bypassed)")


def _declare_device_identity(major: int, minor: int) -> None:
    """Publish the declared capability through torch and the vLLM platform.

    Every reader of the device's identity -- torch, the platform class, the
    attention-backend selector -- must see the same declared value, otherwise
    the layout would be assembled from two different GPUs.
    """

    name = f"CUDA-SIM-SM{major}{minor}"
    properties = device_properties(name, major, minor)

    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda *args, **kwargs: (major, minor)
    torch.cuda.get_device_name = lambda *args, **kwargs: name
    torch.cuda.get_device_properties = lambda *args, **kwargs: properties
    torch.cuda.is_bf16_supported = lambda *args, **kwargs: major >= 8
    torch.cuda.memory_allocated = lambda *args, **kwargs: 0
    torch.cuda.max_memory_allocated = lambda *args, **kwargs: 0
    torch.cuda.memory_reserved = lambda *args, **kwargs: 0
    torch.cuda.max_memory_reserved = lambda *args, **kwargs: 0
    torch.cuda.reset_peak_memory_stats = lambda *args, **kwargs: None
    torch.cuda.memory_stats = lambda *args, **kwargs: {}

    from vllm.platforms import current_platform

    platform_cls = type(current_platform)
    try:
        from vllm.platforms.interface import DeviceCapability

        capability: Any = DeviceCapability(major=major, minor=minor)
    except Exception:
        capability = (major, minor)

    platform_cls.get_device_capability = classmethod(
        lambda cls, device_id=0: capability
    )
    platform_cls.get_device_name = classmethod(lambda cls, device_id=0: name)
    platform_cls.get_device_total_memory = classmethod(
        lambda cls, device_id=0: REPORTED_MEMORY_BYTES
    )
    platform_cls.device_count = classmethod(lambda cls: 1)
    log(
        f"cuda-sim: simulating compute capability {major}.{minor} "
        f"({name}); declared, not measured"
    )


def _patch_cuda_runtime() -> None:
    """Host no-ops for the CUDA execution-ordering and allocator-stats API."""

    patch_execution_ordering(torch.cuda, "cuda-sim")


def _require_cuda_flash_attn() -> None:
    """Fail early and clearly when this is not a CUDA build of vLLM.

    vLLM's FlashAttention backend imports its kernel entry points at module
    level behind ``if current_platform.is_cuda()``. A CPU or "+empty" build has
    no ``vllm.vllm_flash_attn``, so declaring the CUDA platform would make the
    first backend import fail deep inside vLLM with a bare ImportError.
    """

    try:
        importlib.import_module("vllm.vllm_flash_attn")
    except ImportError as exc:
        raise UnsupportedEnvironment(
            "cuda-sim needs a CUDA build of vLLM: vllm.vllm_flash_attn is "
            "missing, so vLLM's real attention backends cannot be imported "
            "once the CUDA platform is active. Install a CUDA vLLM wheel, or "
            "use the cpu launcher, which reproduces the CUDA layout through "
            f"declared backend constraints. Original error: {exc}"
        ) from exc


def _disable_kernel_availability_probes() -> None:
    """Answer kernel-availability probes from the declared capability.

    vLLM asks whether a FlashAttention build is usable before selecting it,
    and that probe reads the device. The *selection* must stay native, because
    the chosen backend decides the KV shapes -- so the probe is answered from
    the declared capability rather than skipped, and both the answer and the
    resulting backend appear in the layout log.
    """

    try:
        fa_utils = importlib.import_module("vllm.v1.attention.backends.fa_utils")
    except ImportError:
        return
    for attribute in ("flash_attn_supports_fp8", "is_fa_version_supported"):
        if hasattr(fa_utils, attribute):
            rebind(
                "vllm.v1.attention.backends.fa_utils",
                attribute,
                lambda *args, **kwargs: True,
            )
            log(f"cuda-sim: fa_utils.{attribute}() answered from the declaration")


def install_host_runtime() -> None:
    """Apply every substitution, in the order the import chain needs them."""

    major, minor = _parse_capability()
    _force_cuda_platform()
    _patch_cuda_runtime()
    require_platform(
        "cuda",
        "cuda-sim",
        "Check that this is a CUDA build of vLLM and that no plugin "
        "(VLLM_PLUGINS) overrides the platform.",
    )
    _require_cuda_flash_attn()
    _declare_device_identity(major, minor)
    _disable_kernel_availability_probes()
    disable_pin_memory("cuda-sim")
    patch_ucm_device("cuda-sim")


def make_cache(vllm_config: Any) -> CacheFixture:
    """Build the native KV layout with the tensors in host memory."""

    from . import cuda

    def make_runner(kv_vllm_config: Any, model: Any, _device: torch.device) -> Any:
        return cuda.make_runner(kv_vllm_config, model, HOST)

    return make_common_cache(
        vllm_config,
        HOST,
        config.tokens,
        config.block_size,
        "gloo",
        lambda *args: None,
        cuda.make_model,
        make_runner,
    )


def main() -> int:
    if config.tokens <= 0 or config.block_size <= 0:
        raise ValueError("tokens and block_size must be positive")
    install_host_runtime()
    from . import cuda

    log(f"cuda-sim: KV tensors on {HOST}, UCM store runtime must be simu")
    vllm_config = None
    fixture = None
    worker = None
    dispatch = None
    try:
        vllm_config = cuda.make_config()
        fixture = make_cache(vllm_config)
        # UCM's MLA shared buffer is created by the worker. The scheduler reads
        # the worker-published id, so initialize the worker before Scheduler.
        worker = make_worker(fixture)
        if config.layout_only:
            describe_ucm_layout(worker)
            log("PASS: layout-only (UCM parsed the native KV layout)")
            return 0
        dispatch = schedule(
            fixture, config.tokens, cuda.request_token_salt, None, worker
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

        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    raise SystemExit(main())
