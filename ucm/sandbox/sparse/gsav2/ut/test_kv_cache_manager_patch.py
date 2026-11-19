# -*- coding: utf-8 -*-
"""B-1 / P0-F6 回归测试：allocate_slots 守卫的 num_slots_sparsed 三态路由。

缺陷背景：kv_cache_manager_patch.py 守卫原为
``num_slots_sparsed != INVALID_SLOT``——参数默认值 None（未打补丁的调用方
不传该 kwarg，如 vLLM fork 的 recompute_scheduler.py:283 /
scheduler_profiling_chunk.py:354,609）!= -1 为 True，穿透守卫以
num_tokens=None 进入 GSA 分配 → coordinator TypeError。
2026-11-09 修复：补 ``num_slots_sparsed is not None`` 前置判别，None 与
INVALID_SLOT 一律回落 vLLM 原生稠密分配路径。

加载方式：被测文件按文件路径直接 exec（ut conftest 已将
ucm.integration.vllm.patch 置为无 __path__ 的 stub，需先预置其 .utils
子模块 stub 才能 import）；ucm.sparse.base / ucm.sparse.state 为
allocate_slots 调用期导入，用 monkeypatch.setitem(sys.modules, ...) 逐用例
替换、自动还原。
"""

import importlib.util
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# 预置 ucm.integration.vllm.patch.utils stub（须早于被测模块 exec）
# ---------------------------------------------------------------------------
_UTILS = types.ModuleType("ucm.integration.vllm.patch.utils")
_UTILS.patch_or_inject = lambda obj, name, value, **k: setattr(obj, name, value)
_UTILS.when_imported = lambda name: (lambda fn: fn)
sys.modules.setdefault("ucm.integration.vllm.patch.utils", _UTILS)

_MOD_PATH = (
    Path(__file__).resolve().parents[1] / "patch" / "vllm" / "kv_cache_manager_patch.py"
)


def _load_under_test():
    spec = importlib.util.spec_from_file_location("kcm_patch_under_test", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kcm_patch = _load_under_test()


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class _FakeCoordinator:
    def __init__(self):
        self.allocated = []
        self.cached = []

    def remove_skipped_blocks(self, request_id, total_computed_tokens):
        pass

    def get_num_blocks_to_allocate(self, **kwargs):
        return 2

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main, num_enc):
        self.allocated.append((request_id, num_tokens, num_tokens_main, num_enc))
        return ["blk-0", "blk-1"]

    def cache_blocks(self, request, num_tokens_to_cache):
        self.cached.append((request.request_id, num_tokens_to_cache))


class _FakeBlockPool:
    def get_num_free_blocks(self):
        return 100


class _FakeEmptyBlocks:
    def __init__(self):
        self.blocks = [()]


class _FakeKvCacheManager:
    def __init__(self):
        self.coordinator = _FakeCoordinator()
        self.block_pool = _FakeBlockPool()
        self.empty_kv_cache_blocks = _FakeEmptyBlocks()
        self.max_model_len = 1024
        self.enable_caching = True

    def create_kv_cache_blocks(self, new_blocks):
        return ("NATIVE_BLOCKS", tuple(new_blocks))


class _FakeRequest:
    request_id = "req-1"
    num_computed_tokens = 0
    num_tokens = 64


class _GsaRecorder:
    """get_ucm_sparse() 返回的探针：记录 GSA 分支实际收到的东西。"""

    def __init__(self):
        self.allocate_calls = []
        self.estimate_calls = []

    def allocate_slots(self, kv_cache_manager, request, num_slots_sparsed):
        self.allocate_calls.append((request.request_id, num_slots_sparsed))
        return "GSA_ALLOCATED"

    def estimate_free_slots_sparse(self, request, num_blocks, num_new_computed=0):
        self.estimate_calls.append(request.request_id)
        return False


def _install_sparse_stubs(monkeypatch, gsa):
    base = types.ModuleType("ucm.sparse.base")
    base.INVALID_SLOT = -1
    state = types.ModuleType("ucm.sparse.state")
    state.get_ucm_sparse = lambda: gsa
    state.has_ucm_sparse = lambda: True
    monkeypatch.setitem(sys.modules, "ucm.sparse.base", base)
    monkeypatch.setitem(sys.modules, "ucm.sparse.state", state)


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------
class TestAllocateSlotsGuard:
    def test_none_falls_back_to_native(self, monkeypatch):
        """P0-F6 核心：未打补丁调用方不传 kwarg（None）→ 原生分配，不进 GSA。"""
        gsa = _GsaRecorder()
        _install_sparse_stubs(monkeypatch, gsa)
        mgr = _FakeKvCacheManager()
        req = _FakeRequest()

        out = kcm_patch.allocate_slots(mgr, req, num_new_tokens=8)

        assert out == ("NATIVE_BLOCKS", ("blk-0", "blk-1"))
        assert gsa.allocate_calls == []  # None 不得进入 GSA 分配
        assert mgr.coordinator.allocated == [("req-1", 8, 8, 0)]
        assert gsa.estimate_calls == ["req-1"]  # :173 稀疏准入检查仍生效

    def test_invalid_slot_falls_back_to_native(self, monkeypatch):
        """显式哨兵 INVALID_SLOT（跳过/抢占/GSA 关）→ 原生分配。"""
        gsa = _GsaRecorder()
        _install_sparse_stubs(monkeypatch, gsa)
        mgr = _FakeKvCacheManager()
        req = _FakeRequest()

        out = kcm_patch.allocate_slots(
            mgr, req, num_new_tokens=8, num_slots_sparsed=-1
        )

        assert out == ("NATIVE_BLOCKS", ("blk-0", "blk-1"))
        assert gsa.allocate_calls == []

    def test_int_routes_to_gsa(self, monkeypatch):
        """合法稀疏槽数 → 进入 GSA 分配并原样透传。"""
        gsa = _GsaRecorder()
        _install_sparse_stubs(monkeypatch, gsa)
        mgr = _FakeKvCacheManager()
        req = _FakeRequest()

        out = kcm_patch.allocate_slots(
            mgr, req, num_new_tokens=8, num_slots_sparsed=42
        )

        assert out == "GSA_ALLOCATED"
        assert gsa.allocate_calls == [("req-1", 42)]
        assert mgr.coordinator.allocated == []  # 原生分配不再执行
