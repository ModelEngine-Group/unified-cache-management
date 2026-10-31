# -*- coding: utf-8 -*-
"""ucm/sandbox/sparse/gsav2/gsa_tensor_slot_manager_ds.py 单元测试。

覆盖目标：TensorManager 的命中/未命中分配、hash 映射维护、
LRU 堆回收、堆耗尽异常、free 语义、统计接口。
"""
import pytest

from ucm.sandbox.sparse.gsav2.gsa_tensor_slot_manager_ds import (
    Tensor,
    TensorManager,
)


class TestTensor:
    def test_init_defaults(self):
        t = Tensor(7)
        assert t.id == 7
        assert t.hash is None
        assert t.last_use_time == 0
        assert t.ref_count == 0


class TestTensorManagerInit:
    def test_init(self):
        tm = TensorManager(5)
        assert len(tm.tensors) == 5
        assert len(tm.heap) == 5
        assert tm.hash_to_id == {}
        assert tm.time == 0
        assert tm.get_tensor_info(0)["id"] == 0

    def test_default_num(self):
        tm = TensorManager()
        assert len(tm.tensors) == 10000


class TestAllocate:
    def test_all_miss_records_indices(self):
        tm = TensorManager(4)
        ids, miss = tm.allocate(["a", -1, "b"])
        assert len(ids) == 3
        assert miss == [0, 1, 2]

    def test_minus_one_not_mapped(self):
        tm = TensorManager(4)
        tm.allocate([-1])
        assert -1 not in tm.hash_to_id
        assert tm.get_tensor_info(0)["hash"] == -1

    def test_hit_increments_ref_and_time(self):
        tm = TensorManager(4)
        tm.allocate(["h"])  # tid 0
        before = tm.get_tensor_info(0)
        ids, miss = tm.allocate(["h"])
        assert miss == []
        assert ids == [0]
        after = tm.get_tensor_info(0)
        assert after["ref_count"] == before["ref_count"] + 1
        assert after["last_use_time"] > before["last_use_time"]

    def test_hit_and_miss_mix(self):
        tm = TensorManager(4)
        tm.allocate(["h1", "h2"])  # tid 0, 1
        ids, miss = tm.allocate(["h1", "new", -1])
        assert ids[0] == 0  # h1 命中
        assert miss == [1, 2]  # new 与 -1 未命中
        assert tm.active_count() == 4  # tid0(命中) + tid1 + 2 个新分配

    def test_old_hash_mapping_removed_on_reuse(self):
        # 堆耗尽后 free → 该 tid 成为唯一空闲项 → 必然被复用
        tm = TensorManager(2)
        tm.allocate(["a", "b"])  # tid 0(a), 1(b)，堆空
        tm.free([0])
        tm.allocate(["c"])  # 复用 tid 0
        assert "a" not in tm.hash_to_id
        assert tm.hash_to_id["c"] == 0

    def test_old_minus_one_hash_skips_removal(self):
        tm = TensorManager(1)
        tm.allocate([-1])  # tid 0, hash=-1，堆空
        tm.free([0])
        tm.allocate(["x"])  # 复用 tid 0：old_hash == -1 → 不查映射
        assert "x" in tm.hash_to_id

    def test_allocated_tensor_popped_from_heap_not_reused(self):
        # 引用计数 > 0 的 tensor 被弹出后不参与复用（堆中移除）
        tm = TensorManager(3)
        ids, _ = tm.allocate(["a", "b", "c"])  # 用尽 3 个
        assert sorted(ids) == [0, 1, 2]
        with pytest.raises(RuntimeError):
            tm.allocate(["d"])  # 堆空

    def test_heap_exhausted_raises(self):
        tm = TensorManager(2)
        tm.allocate(["a", "b"])  # 堆耗尽
        with pytest.raises(RuntimeError, match="没有空闲 tensor"):
            tm.allocate(["c"])

    def test_freed_tensor_reusable(self):
        tm = TensorManager(2)
        ids, _ = tm.allocate(["a", "b"])
        tm.free(ids)
        assert tm.get_free_count() == 2
        ids2, _ = tm.allocate(["c", "d"])
        assert sorted(ids2) == [0, 1]


class TestFree:
    def test_free_decrements_ref(self):
        tm = TensorManager(4)
        ids, _ = tm.allocate(["a", "a"])  # 引用计数 2
        assert tm.get_tensor_info(ids[0])["ref_count"] == 2
        tm.free([ids[0]])
        assert tm.get_tensor_info(ids[0])["ref_count"] == 1
        assert tm.get_free_count() == 3  # 4-1（仍占用 1 个）

    def test_free_back_to_heap(self):
        tm = TensorManager(2)
        ids, _ = tm.allocate(["a", "b"])
        assert tm.get_free_count() == 0
        tm.free(ids)
        assert tm.get_free_count() == 2
        assert tm.active_count() == 0

    def test_free_invalid_tid_prints_and_skips(self, capsys):
        tm = TensorManager(4)
        tm.free([-1])  # 越界
        tm.free([99])  # 越界
        tm.free([0])  # ref_count 已为 0：走 ref_count > 0 的 False 侧
        out = capsys.readouterr().out
        assert "tid wrong" in out
        assert tm.get_free_count() == 4

    def test_free_count_helpers(self):
        tm = TensorManager(3)
        assert tm.free_count() == tm.get_free_count() == 3
        assert tm.active_count() == 0
        tm.allocate(["a"])
        assert tm.get_free_count() == 2
        assert tm.active_count() == 1
