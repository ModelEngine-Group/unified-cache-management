"""
Pytest tests for ucm.sandbox.sparse.gsav2.gsa_tensor_slot_manager_ds.TensorManager.

Migrated from the former ``run_tests()`` / ``__main__`` block of
``gsa_tensor_slot_mamger_ds.py`` (renamed to ``gsa_tensor_slot_manager_ds.py``).
"""

import pytest

from ucm.sandbox.sparse.gsav2.gsa_tensor_slot_manager_ds import TensorManager


def test_allocate_all_miss():
    """Basic allocate with all hashes missing."""
    mgr = TensorManager(10000)
    ids, misses = mgr.allocate([100, 200, 300])
    assert len(ids) == 3
    assert misses == [0, 1, 2]
    assert mgr.active_count() == 3


def test_allocate_hit_and_miss_mixed():
    """A previously allocated hash is reused on hit."""
    mgr = TensorManager(10000)
    ids, _ = mgr.allocate([100, 200, 300])
    ids2, misses2 = mgr.allocate([100, 400])
    assert misses2 == [1]  # 100 hits, 400 misses
    assert ids2[0] == ids[0]
    assert mgr.tensors[ids2[0]].ref_count == 2


def test_allocate_minus_one_sentinel():
    """-1 never hits and never participates in hash mapping."""
    mgr = TensorManager(10000)
    ids, misses = mgr.allocate([-1, -1])
    assert len(ids) == 2
    assert misses == [0, 1]
    assert mgr.tensors[ids[0]].hash == -1


def test_free_and_reallocate_reuses_same_tensor():
    """After free, allocating the same hash hits the same tensor."""
    mgr = TensorManager(10000)
    ids, _ = mgr.allocate([100, 200, 300])
    mgr.free([ids[0]])
    assert mgr.get_tensor_info(ids[0])["ref_count"] == 1
    mgr.free([ids[0]])
    assert mgr.get_tensor_info(ids[0])["ref_count"] == 0

    ids4, misses4 = mgr.allocate([100])
    assert misses4 == []
    assert ids4[0] == ids[0]
    assert mgr.tensors[ids[0]].ref_count == 1


def test_lru_picks_least_recently_used():
    """With no never-used tensors, LRU picks the earliest freed tensor."""
    mgr_lru = TensorManager(num=3)
    ids_a, _ = mgr_lru.allocate([1, 2, 3])  # -> id0, id1, id2 (time 1,2,3)
    mgr_lru.free(ids_a)  # heap: (1,id0), (2,id1), (3,id2)
    ids_b, _ = mgr_lru.allocate([4])
    assert ids_b[0] == ids_a[0]


def test_lazy_deletion_skips_active_tensor():
    """A stale heap entry for an active tensor is lazily skipped."""
    mgr3 = TensorManager(num=3)
    ids1, _ = mgr3.allocate([1, 2, 3])  # -> id0, id1, id2
    mgr3.free([ids1[1], ids1[2]])  # heap: (2,id1), (3,id2); id0 active
    ids2, _ = mgr3.allocate([2])  # hits id1, re-activated; stale (2,id1) in heap
    assert ids2[0] == ids1[1]
    assert mgr3.tensors[ids1[1]].ref_count == 1
    ids3, _ = mgr3.allocate([4])  # must skip active id1 and pick id2
    assert ids3[0] == ids1[2]


def test_batch_refcount_consistency():
    """Bulk allocate/free keeps reference counts consistent."""
    mgr4 = TensorManager(10000)
    hash_list = list(range(500))
    ids_all, misses = mgr4.allocate(hash_list)
    assert len(ids_all) == 500
    assert misses == list(range(500))

    ids_rep, misses_rep = mgr4.allocate(hash_list)
    assert misses_rep == []
    assert all(mgr4.tensors[tid].ref_count == 2 for tid in ids_all)

    mgr4.free(ids_all)
    assert all(mgr4.tensors[tid].ref_count == 1 for tid in ids_all)

    mgr4.free(ids_all)
    assert all(mgr4.tensors[tid].ref_count == 0 for tid in ids_all)
    assert mgr4.active_count() == 0

    ids_hit, misses_hit = mgr4.allocate([0])
    assert misses_hit == []
    assert ids_hit[0] == ids_all[0]


def test_active_tensor_ids_unique():
    """Active tensor IDs are never duplicated."""
    mgr4 = TensorManager(10000)
    mgr4.allocate(list(range(500)))
    used_ids = set()
    for t in mgr4.tensors:
        if t.ref_count > 0:
            assert t.id not in used_ids
            used_ids.add(t.id)


def test_allocate_with_no_free_tensor_raises():
    """Allocating when no tensor is free raises RuntimeError."""
    mgr = TensorManager(num=2)
    mgr.allocate([1, 2])
    with pytest.raises(RuntimeError):
        mgr.allocate([3])


def test_allocate_returns_all_used_ids():
    """Returned IDs are valid tensor indices within range."""
    mgr = TensorManager(num=5)
    ids, _ = mgr.allocate([10, 20, 30, 40, 50])
    assert all(0 <= tid < 5 for tid in ids)
    assert len(set(ids)) == 5
