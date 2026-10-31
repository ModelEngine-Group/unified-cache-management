# -*- coding: utf-8 -*-
"""gsav2 UT 套件 conftest。

职责：
1. 在任何被测模块导入前，向 sys.modules 注入 vllm / vllm_ascend / torch_npu /
   ucm.logger / ucm.integration.vllm.ucm_connector / gsa_prefetch(C++ 扩展) 的 stub，
   使 5 个被测文件可在纯 CPU 环境导入与运行（NPU 环境同样生效，保证跨环境确定性）；
2. 确保平台分支 `hasattr(torch, "npu") and torch.npu.is_available()` 在任何机器上
   都能走 True 侧（fake torch.npu），False 侧由 reload 用例覆盖；
3. 提供 FakeVllmConfig / GSA 构造 fixtures，将 buffer 规模压缩到 MB 级。
"""
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# 路径：本目录 + 仓库根
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[
    4
]  # ut -> gsav2 -> sparse -> sandbox -> ucm -> UCM_feacture_gsa_v2
for p in (str(HERE), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fakes  # noqa: E402


def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _make_tensor_with_pad(data, pad=False, dtype=None, device="cpu"):
    """vllm.utils.torch_utils.make_tensor_with_pad 的等价 stub。

    pad 为填充值（pad=False → 0，pad=True → 1），空列表返回空张量。
    """
    import torch

    if dtype is None:
        dtype = torch.int64
    if not data:
        return torch.zeros((0,), dtype=dtype, device=device)
    max_len = max(len(x) for x in data)
    pad_val = int(bool(pad)) if isinstance(pad, bool) else int(pad)
    rows = []
    for x in data:
        t = torch.as_tensor(x, dtype=dtype)
        if t.numel() < max_len:
            t = torch.cat([t, torch.full((max_len - t.numel(),), pad_val, dtype=dtype)])
        rows.append(t)
    return torch.stack(rows)


class _KVCacheBlocks:
    def __init__(self, blocks):
        self.blocks = blocks


class _RequestHasher:
    """ucm_connector.RequestHasher stub：产出定长 16 字节的块哈希"""

    def __init__(self, vllm_config, rank=0):
        pass

    def make_request_block_hasher(self, block_size):
        def _hasher(request):
            tokens = list(request.all_token_ids)
            n = (len(tokens) + block_size - 1) // block_size
            return [bytes([i % 251]) * 16 for i in range(n)]

        return _hasher


def _install_pin_memory_fallback():
    """torch>=2.x 在无加速器时对 pin_memory=True 直接抛 RuntimeError。

    UT 只验证 Python 逻辑，锁页内存与逻辑无关（C++ 端在 UT 中是 mock，
    不解引用 data_ptr），因此统一把 pin_memory 降级为 False。
    """
    import torch

    _orig_zeros, _orig_empty = torch.zeros, torch.empty

    def _zeros(*args, **kwargs):
        if kwargs.get("pin_memory"):
            kwargs["pin_memory"] = False
        return _orig_zeros(*args, **kwargs)

    def _empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            kwargs["pin_memory"] = False
        return _orig_empty(*args, **kwargs)

    torch.zeros = _zeros
    torch.empty = _empty


def install_stubs():
    if sys.modules.get("ucm.sandbox.sparse.gsav2.gsa", None) is not None:
        return  # 已安装（pytest 多次导入 conftest 的场景）

    # ---------------- vllm ----------------
    _mod("vllm")
    _mod("vllm.config", VllmConfig=object)
    _mod("vllm.distributed")
    _mod(
        "vllm.distributed.kv_transfer", get_kv_transfer_group=lambda: SimpleNamespace()
    )
    _mod(
        "vllm.distributed.parallel_state",
        get_world_group=lambda: SimpleNamespace(local_rank=0),
    )
    _mod("vllm.forward_context", ForwardContext=object)
    _mod("vllm.utils")
    _mod("vllm.utils.torch_utils", make_tensor_with_pad=_make_tensor_with_pad)
    _mod("vllm.utils.platform_utils", is_pin_memory_available=lambda: False)
    _mod("vllm.v1")
    _mod("vllm.v1.core")
    _mod("vllm.v1.core.kv_cache_manager", KVCacheBlocks=_KVCacheBlocks)
    _mod("vllm.v1.core.sched")
    _mod("vllm.v1.core.sched.output", SchedulerOutput=object)
    _mod(
        "vllm.v1.request",
        Request=object,
        RequestStatus=SimpleNamespace(PREEMPTED="preempted", RUNNING="running"),
    )

    # ---------------- vllm_ascend ----------------
    _mod("vllm_ascend")
    _mod("vllm_ascend.attention")
    _mod("vllm_ascend.attention.utils", get_sfa_qsfa_packed_head_dim=lambda k, q: k + q)
    _mod("vllm_ascend.cpu_binding", CpuAlloc=fakes.FakeCpuAlloc)
    _mod(
        "vllm_ascend.utils",
        AscendDeviceType=SimpleNamespace(A5="A5", A2="A2"),
        get_ascend_device_type=lambda: "A2",
    )

    # ---------------- torch_npu ----------------
    _mod("torch_npu")

    # ---------------- ucm.logger（C 扩展 ucmlogger 的替身）----------------
    _mod("ucm.logger", init_logger=lambda name=None: logging.getLogger(name or "ucm"))

    # ---------------- ucm.integration.vllm.ucm_connector ----------------
    _mod("ucm.integration")
    _mod("ucm.integration.vllm")
    _mod("ucm.integration.vllm.ucm_connector", RequestHasher=_RequestHasher)
    # ucm/__init__.py 会执行 from ucm.integration.vllm.patch.logger_patch import
    # patch_logger —— 预置 stub 避免触发真实 patch 链（依赖真实 vllm）
    _mod("ucm.integration.vllm.patch")
    _mod("ucm.integration.vllm.patch.logger_patch", patch_logger=lambda *a, **k: None)

    # ---------------- gsa_prefetch（C++ pybind11 扩展的替身）----------------
    _mod(
        "ucm.sandbox.sparse.gsav2.prefetch.gsa_prefetch",
        GSAPrefetchEngineC=fakes.FakeEngineC,
        UCMKVPrefetch=fakes.FakeUCMPrefetch,
    )

    # ---------------- torch.ops._C_ascend（ACL 自定义 op）----------------
    # 无条件覆盖：UT 不依赖真实 ACL op（NPU 环境同理，保证跨环境确定性）
    import torch

    torch.ops._C_ascend = SimpleNamespace(
        gsa_gather_copy=lambda *a, **k: None,
        gsa_scatter_copy=lambda *a, **k: None,
    )

    # ---------------- pin_memory 降级（CPU 环境兼容）----------------
    _install_pin_memory_fallback()

    # ---------------- 平台分支 True 侧：确保 torch.npu 存在 ----------------
    if not hasattr(torch, "npu"):
        torch.npu = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None)


install_stubs()

# 被测模块：导入即覆盖其模块级代码（含 gsa.py L30-43 平台分支 True 侧）
import ucm.sandbox.sparse.gsav2.gsa  # noqa: E402,F401
import ucm.sandbox.sparse.gsav2.gsa_log  # noqa: E402,F401
import ucm.sandbox.sparse.gsav2.gsa_shared_memory  # noqa: E402,F401
import ucm.sandbox.sparse.gsav2.gsa_tensor_slot_manager_ds  # noqa: E402,F401
import ucm.sandbox.sparse.gsav2.prefetch.prefetch_engine  # noqa: E402,F401


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_config():
    """微型 VllmConfig（PD 分离 decode 节点）"""
    return fakes.FakeVllmConfig()


@pytest.fixture
def make_gsa(tmp_path, monkeypatch):
    """GSA 构造工厂：patch 共享内存类 / NUMA / 常量，并将日志目录隔离到 tmp_path"""

    def _make(role=None, **cfg_kwargs):
        import ucm.sandbox.sparse.gsav2.gsa as gsa_mod
        from ucm.sparse.base import UcmSparseRole

        if role is None:
            role = UcmSparseRole.WORKER
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gsa_mod, "MAX_SLOTS_BLOCK_NUM", 16)
        monkeypatch.setattr(gsa_mod, "MAX_INDEXER_BLOCK_NUM", 16)
        monkeypatch.setattr(
            gsa_mod, "SharedTensorManager", fakes.FakeSharedTensorManager
        )
        monkeypatch.setattr(gsa_mod, "SharedBlockTable", fakes.FakeSharedBlockTable)
        monkeypatch.setattr(
            gsa_mod, "SharedTensorVersion", fakes.FakeSharedTensorVersion
        )
        monkeypatch.setattr(gsa_mod.GSA, "bind_cpus", lambda self: (None, None))
        monkeypatch.setattr(gsa_mod, "bind_thread_memory_policy", lambda node: None)
        cfg = fakes.FakeVllmConfig(**cfg_kwargs)
        return gsa_mod.GSA(cfg, role)

    return _make


@pytest.fixture
def gsa_worker(make_gsa):
    """完整构造的 GSA WORKER 实例（PD 分离 decode 节点，微型 buffer）"""
    return make_gsa()
