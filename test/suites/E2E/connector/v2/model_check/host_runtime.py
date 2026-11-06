# SPDX-License-Identifier: MIT
"""Declare the accelerator for model/backend discovery in a card-less process.

Based on the hardware declarations in model-check PR #1579. The model-inspection
child needs the same declaration. Neither process runs accelerator kernels.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace


def declare(platform, identity):
    import torch
    import wrapt

    if "vllm.platforms" in sys.modules:
        raise RuntimeError(
            "Start model-check in a fresh interpreter, before importing vLLM"
        )

    def refuse_device_init(*args, **kwargs):
        raise RuntimeError("Model-check attempted real accelerator initialization")

    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if platform == "cuda":
        major, minor = map(int, identity.split("."))
        if not torch.version.cuda:
            raise RuntimeError("cuda-sim requires a CUDA build of PyTorch and vLLM")
        torch.cuda._lazy_init = refuse_device_init
        name = f"cuda-sim SM{major}{minor}"
        properties = SimpleNamespace(
            major=major,
            minor=minor,
            name=name,
            multi_processor_count=1,
            total_memory=1 << 40,
            is_integrated=False,
            warp_size=32,
        )
        torch.cuda.get_device_properties = lambda *args, **kwargs: properties
        torch.cuda.get_device_name = lambda *args, **kwargs: name
        torch.cuda.current_device = lambda: local_rank
        torch.cuda.device_count = lambda: world_size
        torch.cuda.get_device_capability = lambda *a, **kw: (major, minor)
        torch.cuda.is_available = lambda: True
        torch.cuda.is_bf16_supported = lambda *a, **kw: major >= 8

        @wrapt.when_imported("vllm.platforms")
        def select_cuda(module):
            module.builtin_platform_plugins.clear()
            module.builtin_platform_plugins["cuda"] = (
                lambda: "vllm.platforms.cuda.CudaPlatform"
            )

        @wrapt.when_imported("vllm.platforms.cuda")
        def declare_cuda(module):
            from vllm.platforms.interface import DeviceCapability

            cls = module.CudaPlatform
            cls.get_device_capability = classmethod(
                lambda cls, device_id=0: DeviceCapability(major, minor)
            )
            cls.get_device_name = classmethod(lambda cls, device_id=0: name)
            cls.get_device_total_memory = classmethod(
                lambda cls, device_id=0: properties.total_memory
            )
            cls.device_count = classmethod(lambda cls: world_size)
            cls.logical_device_id_to_visible_device_id = classmethod(
                lambda cls, device_id: device_id
            )
            cls.is_pin_memory_available = classmethod(lambda cls: False)

    else:
        import importlib

        arch = os.environ.get("TRITON_ASCEND_ARCH") or os.environ.get("SOC_VERSION")
        if not arch or not arch.lower().startswith("ascend"):
            raise ValueError(
                "Set TRITON_ASCEND_ARCH to the target chip, e.g. Ascend910B1"
            )
        arch = "Ascend" + arch[6:].upper()
        os.environ["TRITON_ASCEND_ARCH"] = arch
        driver = importlib.import_module("triton.backends.ascend.driver")
        for owner in vars(driver).values():
            if isinstance(owner, type) and hasattr(owner, "get_arch"):
                owner.get_arch = lambda self: arch
                if hasattr(owner, "get_aicore_num"):
                    owner.get_aicore_num = lambda self: 1

        import torch_npu  # noqa: F401

        torch.npu._lazy_init = refuse_device_init
        torch.npu.get_device_name = lambda *args, **kwargs: arch
        torch.npu.current_device = lambda: local_rank
        torch.npu.device_count = lambda: world_size

        @wrapt.when_imported("vllm_ascend.utils")
        def disable_device_kernels(module):
            module._CUSTOM_OP_ENABLED = False

        @wrapt.when_imported("vllm_ascend.platform")
        def disable_pinning(module):
            module.NPUPlatform.device_count = classmethod(lambda cls: world_size)
            module.NPUPlatform.is_pin_memory_available = classmethod(lambda cls: False)

    from vllm.platforms import current_platform

    if current_platform.device_type != platform:
        raise RuntimeError(
            f"Expected {platform}, vLLM selected {current_platform.device_type}"
        )
    if platform == "npu":
        from vllm_ascend.device.device_config import get_ascend_device_type
        from vllm_ascend.device.hardware import device_type_from_soc_version

        if device_type_from_soc_version(arch).name != identity:
            raise ValueError(
                f"Triton arch {arch} does not match --ascend-soc {identity}"
            )
        if get_ascend_device_type().name != identity:
            raise ValueError(
                f"Installed vllm-ascend is built for {get_ascend_device_type().name}, requested {identity}"
            )


def install(platform, identity):
    declare(platform, identity)
    from vllm.model_executor.models import registry

    registry._SUBPROCESS_COMMAND = [
        sys.executable,
        str(Path(__file__).resolve()),
        platform,
        identity,
    ]


if __name__ == "__main__":
    declare(*sys.argv[1:])
    from vllm.model_executor.models import registry

    registry._run()
