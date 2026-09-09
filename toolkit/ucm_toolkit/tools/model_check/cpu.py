# SPDX-License-Identifier: MIT
"""CPU（官方 vLLM CPU 构建）UCM KV-cache compatibility check without weights.

与 ascend.py/cuda.py 同一套 common 流程，只是：

1. 目标平台是官方 vLLM 的 CPU 构建（vllm-0.26.0+cpu 等）：平台按
   ``vllm_version_matches_substr("cpu")`` 自动激活 CpuPlatform，无需任何
   设备/驱动；
2. 模型在 meta device 上构造（不加载权重），runner 用官方
   GPUModelRunner（device=cpu），分布式 backend 用 gloo；
3. UCM 核心需以 **simu** 运行时编译（``export PLATFORM=simu`` 或留空执行
   ``pip install -e .``，setup.py 的 fallback 分支即
   ``-DRUNTIME_ENVIRONMENT=simu``），不依赖 NPU/CUDA 设备库。

运行（在装了官方 vllm CPU 构建 + simu 版 UCM + ucm_toolkit 的 venv 里）：

    ucm-toolkit run model-check --model /path/to/model --block-size 128 \
        --storage-backends /path/to/ucm_storage

与 ascend/cuda 的差异：

- ``make_model`` 不做任何 vllm-ascend patch（官方 vllm 没有平台插件）；
- ``patch_groups`` 在 CPU 上为空操作（官方分组/manager 工厂直接用）；
  多 group（如 DeepSeekV4 的 SWA 混合）在官方 vllm 的
  ``_get_kv_cache_groups_uniform_groups`` 存在页大小断言限制（SWA 子组页 >
  full-MLA 组页），非 GPU 平台会在此失败——这是上游代码边界，单组模型
  （如 GLM 系列）不受影响。
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
    no_real_device_move_from_meta_context,
    make_worker,
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
dtype = config.dtype
kv_cache_dtype = config.kv_cache_dtype
connector_module_path = config.connector_module_path
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
# 1. Construct VllmConfig (device="cpu")
# ---------------------------------------------------------------------------


def _model_uses_mamba(vllm_config: Any) -> bool:
    """Kimi-K3 等 hybrid（mamba+MLA）模型必须保持 prefix caching 开启：
    vllm 0.27 的 VllmConfig 校验要求 ``mamba_block_size`` 与 prefix caching
    同时存在，且 mamba 'align' 模式本身依赖 prefix caching（与官方
    capture 的布局一致）。hybrid 判定以 ModelConfig.is_hybrid 为准
    （hf config 里不一定有 mamba_block_size 字段）。"""
    return bool(getattr(getattr(vllm_config, "model_config", None), "is_hybrid", False))


def make_config() -> Any:
    # vllm 0.29：必须先中性化 CPU 平台的 MLA config 特化（强制 block=16、
    # 强制关 prefix caching/chunked prefill——Kimi 的 mamba_block_size 校验
    # 会因此失败），CUDA-sim 才能拿到 GPU 口径的配置。幂等。
    _patch_cpu_platform_config_for_sim()
    vllm_config = make_common_config(
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
        "cpu",
        connector_module_path,
    )
    # CPU 平台对 MLA 模型强制禁用 chunked prefill（vllm/platforms/cpu.py），
    # prefix caching 需与 UCM 的部署形态一致（UCM 接管前缀查找，本地 HBM
    # 命中反而会绕过 UCM 的 external load）——对 MLA 模型默认关闭。
    # 例外：带 mamba_block_size 的模型（Kimi-K3）必须显式开启（vllm 0.27
    # VllmConfig 校验：mamba-block-size 只能与 prefix caching 同时设置，
    # 且 mamba 'align' 模式本身依赖 prefix caching）。
    vllm_config.cache_config.enable_prefix_caching = _model_uses_mamba(vllm_config)
    return vllm_config


# ---------------------------------------------------------------------------
# 2. No-weight model + production ModelRunner
# ---------------------------------------------------------------------------


# CUDA-sim 约束（vllm 0.29 官方源码真实声明，与 docs/kv-layout-ascend-20260905
# tools/capture_mocked.py 的 _CUDA_SIM_CONSTRAINTS 保持一致）：
#   FLASH_ATTN (flash_attn.py): MultipleOf(16)，不声明布局 → 默认偏好 LBNHC
#   FlashMLA (mla/flashmla.py): [64]，不声明布局
# indexer/SWA/DSV4 主层不走 selector（模型代码显式引用真实 backend 类），
# 约束天然生效，无需模拟。
_CUDA_SIM_CONSTRAINTS = {
    "gqa": {"kernel_blocks": [16], "layouts": None},
    "mla": {"kernel_blocks": [64], "layouts": None},
}


def _patch_cpu_platform_config_for_sim() -> None:
    """vllm 0.29 CUDA-sim：中性化 CpuPlatform.check_and_update_config 的 MLA 特化。

    0.29 的 CPU 平台在 config 阶段对 MLA 模型做三件 CUDA 上没有的事：
    block_size 强制 16（CPU 参考解码 kernel 的限制）、强制关闭 chunked
    prefill + prefix caching（Kimi 的 mamba_block_size 校验会因此直接失败）、
    GDN mamba dtype 重置 float32。CUDA-sim 按 GPU 口径：放行原方法后按
    入参还原这三组值；其余 CPU 逻辑保持不变。幂等，可重复调用。"""
    from vllm.platforms.cpu import CpuPlatform

    if getattr(CpuPlatform.check_and_update_config, "_ucm_sim_patched", False):
        return
    orig = CpuPlatform.check_and_update_config.__func__

    def _sim_safe(cls, vllm_config):
        cc = vllm_config.cache_config
        sc = vllm_config.scheduler_config
        saved = (cc.block_size, cc.user_specified_block_size,
                 cc.enable_prefix_caching, cc.mamba_ssm_cache_dtype,
                 sc.enable_chunked_prefill, sc.max_num_batched_tokens)
        try:
            orig(cls, vllm_config)
        finally:
            (cc.block_size, cc.user_specified_block_size,
             cc.enable_prefix_caching, cc.mamba_ssm_cache_dtype,
             sc.enable_chunked_prefill, sc.max_num_batched_tokens) = saved

    _sim_safe._ucm_sim_patched = True
    CpuPlatform.check_and_update_config = classmethod(_sim_safe)


def _patch_cpu_runtime() -> None:
    """官方 vLLM CPU 构建没有 CUDA attention backend（0.29 有 CPU_MLA 但
    kernel block=16，sparse 直接被平台拒绝）。布局/逐字节比对不需要真
    kernel：注入带 CUDA 真值约束的假 backend（CUDA-sim）——0.29 的张量
    shape/stride 由 vllm 布局系统按这些约束计算（allocate_kv_cache 单底衬 +
    create_kv_cache_views），register_kv_caches 收到的视图即 GPU 形状。
    forward/metadata 仅占位（model-check 只 schedule，无 forward）。"""

    from vllm.v1.attention.backend import (
        AttentionBackend, AttentionImpl, AttentionMetadataBuilder)

    class _FakeImpl(AttentionImpl):
        def __init__(self, num_heads, head_size, scale, num_kv_heads=None,
                     alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
                     logits_soft_cap=None, attn_type="decoder",
                     kv_sharing_target_layer_name=None, **kwargs):
            self.num_heads = num_heads
            self.head_size = head_size
            self.scale = scale
            self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
            self.alibi_slopes = alibi_slopes
            self.sliding_window = sliding_window
            self.kv_cache_dtype = kv_cache_dtype
            self.logits_soft_cap = logits_soft_cap
            self.attn_type = attn_type
            self.q_pad_num_heads = None
            self.is_sparse = False

        def forward(self, *a, **k):
            raise NotImplementedError("fake backend for layout capture only")

        def build_kv_cache(self, *a, **k):
            return None

    class _FakeBuilder(AttentionMetadataBuilder):
        def __init__(self, *a, **k):
            pass

        def build(self, *a, **k):
            # model-check 只做 schedule（无 forward），metadata 不会被消费
            return None

    class _CUDASimBackend(AttentionBackend):
        _mla = False
        # CUDA-sim 约束由 _apply_sim 填充；[128] 是 0.27 及更早的旧行为兜底
        _kernel_blocks: list = [128]
        _layouts: tuple | None = None

        @classmethod
        def get_name(cls) -> str:
            return "FAKE_MLA" if cls._mla else "FAKE_GQA"

        @classmethod
        def get_impl_cls(cls) -> type:
            return _FakeImpl

        @classmethod
        def get_builder_cls(cls) -> type:
            return _FakeBuilder

        @classmethod
        def get_kv_cache_shape(cls, num_blocks, block_size, num_kv_heads, head_size,
                               cache_dtype_str="auto", **kwargs):
            # vllm 0.27 及更早按 backend 的 shape 创建 KV 缓冲（0.29 runner
            # 路径已移除此调用，仅作旧版后备）：GQA 是 K/V 合并 5D，MLA 是 4D
            if cls._mla:
                return (num_blocks, block_size, num_kv_heads, head_size)
            return (num_blocks, 2, block_size, num_kv_heads, head_size)

        @classmethod
        def is_mla(cls) -> bool:
            return cls._mla

        @classmethod
        def get_supported_kernel_block_sizes(cls) -> list:
            return list(cls._kernel_blocks)

        @classmethod
        def supported_kv_cache_layouts(cls):
            # None = 不声明（vllm 0.29 走默认布局偏好 LBNHC 优先）
            return cls._layouts

        @staticmethod
        def get_required_kv_cache_layout():
            return None

    _FakeMLABackend = type("FakeMLABackend", (_CUDASimBackend,), {"_mla": True})
    _FakeGQABackend = type(
        "FakeGQABackend", (_CUDASimBackend,), {"_mla": False})

    def _gqa_name(cls) -> str:
        # vllm 0.27+ 会把 backend 名转成 AttentionBackendEnum 校验；
        # CPU_ATTN 是合法枚举名。
        return "CPU_ATTN"

    _FakeGQABackend.get_name = classmethod(_gqa_name)

    def _apply_sim(cls, spec: dict) -> None:
        layouts = spec["layouts"]
        if layouts is not None:
            from vllm.v1.kv_cache_layout import KVCacheLayout

            layouts = tuple(getattr(KVCacheLayout, n) for n in layouts)
        cls._kernel_blocks = list(spec["kernel_blocks"])
        cls._layouts = layouts

    _apply_sim(_FakeMLABackend, _CUDA_SIM_CONSTRAINTS["mla"])
    _apply_sim(_FakeGQABackend, _CUDA_SIM_CONSTRAINTS["gqa"])

    import vllm.v1.attention.selector as sel

    def _fake_get_attn_backend(head_size, dtype, kv_cache_dtype,
                               use_mla=False, **kwargs):
        return _FakeMLABackend if use_mla else _FakeGQABackend

    sel.get_attn_backend = _fake_get_attn_backend
    sel._cached_get_attn_backend = lambda backend, attn_selector_config, num_heads=None: (
        _FakeMLABackend if attn_selector_config.use_mla else _FakeGQABackend)

    try:
        import vllm.v1.attention.backends.mla.prefill.selector as _sel_mod
        import vllm.v1.attention.backends.mla.prefill as _pf_mod
    except Exception:
        _sel_mod = _pf_mod = None

    class _FakePrefillBackend:
        def __init__(self, *a, **k):
            pass

    if _sel_mod is not None:
        _sel_mod.get_mla_prefill_backend = lambda *a, **k: _FakePrefillBackend
        _pf_mod.get_mla_prefill_backend = lambda *a, **k: _FakePrefillBackend
        import vllm.model_executor.layers.attention.mla_attention as _mla_attn_mod

        _mla_attn_mod.get_mla_prefill_backend = lambda *a, **k: _FakePrefillBackend
        try:
            # 0.27 起官方 Kimi-K3 有自己的 MLA 模块（module-level 绑定 prefill 入口）
            import vllm.models.kimi_k3.nvidia.mla as _kimi_mla_mod

            _kimi_mla_mod.get_mla_prefill_backend = (
                lambda *a, **k: _FakePrefillBackend
            )
        except Exception:
            pass

    # torch.cuda / platform capability 占位（官方模型构造会建 Stream/Event，
    # DSV4 的 fp8 einsum recipe 查平台 capability）
    import torch as _torch

    class _FakeStream:
        def __init__(self, *a, **k):
            pass

        def wait_stream(self, *a, **k):
            pass

        def record_stream(self, *a, **k):
            pass

        def synchronize(self, *a, **k):
            pass

        def query(self):
            return True

    class _FakeEvent:
        def __init__(self, *a, **k):
            pass

        def record(self, *a, **k):
            pass

        def wait(self, *a, **k):
            pass

        def synchronize(self, *a, **k):
            pass

        def query(self):
            return True

    _torch.cuda.Stream = _FakeStream  # type: ignore[misc]
    _torch.cuda.current_stream = lambda *a, **k: _FakeStream()  # type: ignore[misc]
    _torch.cuda.Event = _FakeEvent  # type: ignore[misc]
    _torch.cuda.current_event = lambda *a, **k: _FakeEvent()  # type: ignore[misc]
    _torch.cuda.get_device_capability = lambda *a, **k: (10, 0)  # type: ignore[misc]
    _torch.cuda.get_device_name = lambda *a, **k: "FAKE-CPU-SM100"  # type: ignore[misc]

    from collections import namedtuple
    from vllm.platforms import current_platform as _plat

    _Cap = namedtuple("_Cap", "major minor")

    _plat.get_device_capability = lambda: _Cap(10, 0)  # type: ignore[method-assign]


def _patch_fused_moe_stub() -> None:
    """FusedMoE 桩：不建专家权重、不选 MoE backend（kv_cache_spec 不读 MLP）。

    vllm 0.29 的 select_unquantized_moe_backend 会在 CPU 构建上拒绝部分模型的
    MoE 配置（如 Kimi-K3 的 LatentMoE，NotImplementedError）。model-check 与
    capture_mocked 同思路：在 initialize_model 之前（模型模块尚未 import，
    from-import 解析到桩）替换工厂与 RoutedExperts。桩只保留构造期被读取的
    路由/量化属性。"""

    import torch.nn as nn
    from vllm.model_executor.layers.fused_moe import layer as _fm_layer
    import vllm.model_executor.layers.fused_moe as _fm_pkg
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts as _RoutedExperts)

    def _stub_factory(*args, **kwargs):
        m = nn.Module()
        m.num_experts = kwargs.get("num_experts", 0)
        m.num_local_experts = m.num_experts
        m.gate = None
        m.experts = None
        m.shared_experts = None
        return m

    def _stub_re(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.use_grouped_topk = kwargs.get("use_grouped_topk", False)
        self.renormalize = kwargs.get("renormalize", True)
        self.topk_group = kwargs.get("topk_group", None)
        self.num_expert_group = kwargs.get("num_expert_group", None)
        self.custom_routing_function = kwargs.get("custom_routing_function", None)
        self.scoring_func = kwargs.get("scoring_func", "softmax")
        self.routed_scaling_factor = kwargs.get("routed_scaling_factor", 1.0)
        self.e_score_correction_bias = kwargs.get("e_score_correction_bias", None)
        self.apply_router_weight_on_input = kwargs.get(
            "apply_router_weight_on_input", False)
        self.quant_config = kwargs.get("quant_config", None)
        self.quant_method = kwargs.get("quant_method", None)
        self.expert_map_manager = kwargs.get("expert_map_manager", None)
        self._ascend_moe_lora_context = kwargs.get(
            "_ascend_moe_lora_context", None)

    _fm_layer.FusedMoEFactory = _stub_factory
    _fm_pkg.FusedMoEFactory = _stub_factory
    _RoutedExperts.__init__ = _stub_re


def make_model(vllm_config: Any) -> tuple[Any, Any]:
    """Build the production model structure on the meta device (official vLLM)."""

    _patch_cpu_runtime()
    _patch_fused_moe_stub()

    from vllm.model_executor.model_loader.utils import initialize_model

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(vllm_config.model_config.dtype)
        vllm_config.compilation_config.static_forward_context.clear()
        with (
            current_vllm_config_context(vllm_config),
            no_real_device_move_from_meta_context(_factory_kwargs_redirect_to_meta),
            torch.device("meta"),
        ):
            model = initialize_model(vllm_config, prefix="")
    except Exception as exc:
        raise UnsupportedEnvironment(
            "This model/vLLM combination cannot construct its model structure on "
            "the meta device with the official (CPU-build) vLLM. The checker will "
            "not fall back to loading checkpoint weights. "
            f"Original error: {type(exc).__name__}: {exc}\n"
            f"Original traceback:\n{traceback.format_exc()}"
        ) from exc
    finally:
        torch.set_default_dtype(previous_dtype)
    return model, vllm_config


def select_model_runner_cls() -> type[Any]:
    """Return the production runner class of the installed vLLM."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as exc:
        raise UnsupportedEnvironment(
            "CPU checks require vllm.v1.worker.gpu_model_runner.GPUModelRunner "
            "so KV tensors are initialized through the real runner path. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return GPUModelRunner


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
            f"loader was called. Original error: {type(exc).__name__}: {exc}\n"
            f"Original traceback:\n{traceback.format_exc()}"
        ) from exc

    # load_model() normally installs this attribute.  The checker has already
    # constructed the same module hierarchy on meta, so attach it explicitly.
    runner.model = model
    return runner


def make_cache(vllm_config: Any, active_device: torch.device) -> CacheFixture:
    return make_common_cache(
        vllm_config,
        active_device,
        tokens,
        block_size,
        "gloo",
        torch.cpu.synchronize,
        make_model,
        make_runner,
    )


# ---------------------------------------------------------------------------
# Group patch (no-op on official vLLM / CPU)
# ---------------------------------------------------------------------------


def patch_groups(fixture: CacheFixture) -> None:
    """Official vLLM needs no Ascend-style manager patch on CPU.

    Single-group configs go through the upstream manager factory directly.
    DeepseekV4-style multi-group (SWA mixed) layouts hit the upstream
    ``_get_kv_cache_groups_uniform_groups`` page-size assertion on non-GPU
    platforms (see module docstring) and fail before this point.
    """
    return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def check_config() -> None:
    if tokens <= 0 or block_size <= 0:
        raise ValueError("tokens and block_size must be positive")


def main() -> int:
    check_config()
    active_device = torch.device("cpu")
    log(f"device={active_device}")
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
        dispatch = schedule(
            fixture, tokens, request_token_salt, patch_groups, worker
        )
        verify(fixture, dispatch, worker, torch.cpu.synchronize)
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
