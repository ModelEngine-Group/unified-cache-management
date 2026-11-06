# SPDX-License-Identifier: MIT
"""Shared host-runtime substitutions for the device-less launchers.

``ascend_sim.py`` and ``cuda_sim.py`` run the production vLLM / vLLM-Ascend
KV-cache path with no accelerator present. Both need the same three kinds of
substitution, so they live here and stay identical between the two platforms:

* execution-ordering APIs (streams, events, ``synchronize``) become host
  no-ops -- they order execution, they do not shape tensors;
* pinned host memory is reported unavailable -- page-locking a CPU tensor goes
  through the accelerator's pinned allocator, which initializes its runtime
  and therefore needs a driver;
* UCM's worker device becomes the host ``CpuDevice``, matching the ``simu``
  store runtime the launchers require.

Nothing here touches KV-cache specs, grouping, shapes, strides, dtypes or
block IDs.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any

import torch

from .common import UnsupportedEnvironment, log

HOST = torch.device("cpu")

# Reported free/total device memory. Large enough that vLLM's capacity checks
# pass; the KV pool is bounded separately by ``num_gpu_blocks_override``.
REPORTED_MEMORY_BYTES = 1 << 40


def rebind(module_name: str, attribute: str, replacement: Any) -> None:
    """Replace a function and every ``from x import f`` binding of it.

    Only safe for values with a unique identity (functions, classes). Never
    use it for a bool or small int: identity matching would hit unrelated
    values of the same object.
    """

    module = importlib.import_module(module_name)
    original = getattr(module, attribute)
    setattr(module, attribute, replacement)
    for loaded in list(sys.modules.values()):
        namespace = getattr(loaded, "__dict__", None)
        if not namespace:
            continue
        for name, value in list(namespace.items()):
            if value is original:
                setattr(loaded, name, replacement)


class HostStream:
    """Stand-in for a device stream: records order, executes nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "HostStream":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def wait_stream(self, *args: Any, **kwargs: Any) -> None:
        pass

    def wait_event(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_event(self, *args: Any, **kwargs: Any) -> "HostEvent":
        return HostEvent()

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True


class HostEvent:
    """Stand-in for a device event: always already complete."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record(self, *args: Any, **kwargs: Any) -> None:
        pass

    def wait(self, *args: Any, **kwargs: Any) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True

    def elapsed_time(self, *args: Any) -> float:
        return 0.0


def device_properties(
    name: str, major: int, minor: int, multi_processor_count: int = 1
) -> SimpleNamespace:
    """Device properties built from declared values, never read from hardware."""

    return SimpleNamespace(
        name=name,
        major=major,
        minor=minor,
        total_memory=REPORTED_MEMORY_BYTES,
        multi_processor_count=multi_processor_count,
        is_integrated=False,
        is_multi_gpu_board=False,
        uuid=f"{name}-sim",
        L2_cache_size=0,
        regs_per_multiprocessor=0,
        max_threads_per_multi_processor=0,
        warp_size=32,
    )


def patch_execution_ordering(namespace: Any, label: str) -> None:
    """Make one accelerator namespace's ordering API host no-ops.

    ``namespace`` is ``torch.cuda`` or ``torch.npu``. Only calls that schedule
    or wait for work are replaced; allocation and property queries are handled
    by the caller, which knows what to declare for its platform.
    """

    namespace.set_device = lambda *args, **kwargs: None
    namespace.synchronize = lambda *args, **kwargs: None
    namespace.empty_cache = lambda *args, **kwargs: None
    namespace.current_device = lambda: 0
    namespace.device_count = lambda: 1
    namespace.Stream = HostStream
    namespace.Event = HostEvent
    namespace.current_stream = lambda *args, **kwargs: HostStream()
    namespace.default_stream = lambda *args, **kwargs: HostStream()
    namespace.stream = lambda *args, **kwargs: HostStream()
    namespace.mem_get_info = lambda *args, **kwargs: (
        REPORTED_MEMORY_BYTES,
        REPORTED_MEMORY_BYTES,
    )
    log(f"{label}: device streams, events and synchronize are host no-ops")


def disable_pin_memory(label: str) -> None:
    """Report pinned host memory as unavailable.

    Allocating a page-locked CPU tensor goes through the accelerator's pinned
    allocator, which initializes its runtime and therefore needs a driver.
    Pinning only speeds up host/device DMA: it changes no shape, stride or
    dtype, and vLLM already supports platforms where it is unavailable.
    """

    torch_utils = importlib.import_module("vllm.utils.torch_utils")
    cache_clear = getattr(torch_utils.is_pin_memory_available, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    rebind("vllm.utils.torch_utils", "is_pin_memory_available", lambda: False)
    # ``PIN_MEMORY`` is a module-level snapshot of that probe. Replace it by
    # name in every module that imported it: a bool cannot be matched by
    # identity without hitting unrelated True values.
    snapshots = [
        module.__name__
        for module in list(sys.modules.values())
        if getattr(module, "PIN_MEMORY", None) is True
    ]
    for module_name in snapshots:
        sys.modules[module_name].PIN_MEMORY = False
    log(f"{label}: pinned host memory disabled ({len(snapshots)} snapshots)")


def patch_ucm_device(label: str) -> None:
    """Point UCM's worker side at the host device.

    The KV tensors live in host memory here, so UCM must address them as host
    memory and its store must be the ``simu`` runtime (built with ``PLATFORM``
    unset), whose transfers are plain memcpy.
    """

    device = importlib.import_module("ucm.integration.vllm.device")
    if not hasattr(device, "CpuDevice"):
        raise UnsupportedEnvironment(
            "the installed UCM has no ucm.integration.vllm.device.CpuDevice, "
            "so its worker side cannot address host KV tensors."
        )
    rebind("ucm.integration.vllm.device", "create_device", device.CpuDevice)
    rebind("ucm.integration.vllm.device", "get_current_device_id", lambda: 0)
    rebind(
        "ucm.integration.vllm.device",
        "get_ucm_worker_torch_device",
        lambda local_rank: HOST,
    )
    log(f"{label}: UCM worker device is the host CpuDevice")


def require_platform(device_type: str, label: str, hint: str) -> None:
    """Fail early when vLLM did not activate the platform being simulated."""

    from vllm.platforms import current_platform

    actual = getattr(current_platform, "device_type", None)
    if actual != device_type:
        raise UnsupportedEnvironment(
            f"{label} needs vLLM's {device_type!r} platform to be active, got "
            f"{type(current_platform).__name__} (device_type={actual!r}). {hint}"
        )
