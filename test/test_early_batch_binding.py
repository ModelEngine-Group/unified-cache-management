"""Real CPython/C++ batch lifetime and Posix reads, with simulated H2D.

Run only against a simu build via python_with_build.py. No NPU is initialized.
PSync seeds complete before lookup; this does not certify native AIO commit
ordering or measure SSD performance.
"""

import array
import gc
from pathlib import Path
import time
import uuid
import weakref

import numpy as np
import pytest

from ucm.store.pipeline import ucmpipelinestore as native
from ucm.store.pipeline.early_connector import _ReservedLoad
from ucm.integration.vllm.early_consumer import Consumer, load_scope


@pytest.mark.parametrize("deferred", [False, True])
def test_native_pipeline_batch_keeps_store_alive_and_hands_off_before_release(tmp_path, deferred):
    store_dir = Path(native.__file__).resolve().parent.parent
    shard_bytes, shards = 4096, 128
    config = dict(storage_backends=[str(tmp_path)], device_id=0,
                  tensor_size=shard_bytes, shard_size=shard_bytes,
                  block_size=shards * shard_bytes, data_dir_shard_bytes=1,
                  posix_data_trans_concurrency=1, posix_lookup_concurrency=1,
                  posix_io_engine="psync", io_direct=False,
                  store_health={"enabled": False})
    block = uuid.uuid4().bytes
    one_id = np.frombuffer(block, dtype=np.uint8)
    ids = np.frombuffer(block * shards, dtype=np.uint8)
    indexes = array.array("Q", range(shards))
    source = np.arange(shards * shard_bytes, dtype=np.uint8).reshape(shards, shard_bytes)
    addresses = np.array([[row.ctypes.data] for row in source], dtype=np.uint64)
    seed = native.PipelineStore()
    seed.Stack("Posix", str(store_dir / "posix/libposixstore.so"), config)
    seed.Wait(seed.Dump(ids, indexes, addresses))
    assert seed.LookupOnPrefix(one_id) == 0
    del seed

    pipeline = native.PipelineStore()
    pipeline.Stack("EarlyPosix", str(store_dir / "posix/libposixstore.so"), config)
    pipeline.Stack("EarlyCache", str(store_dir / "cache/libcachestore.so"),
                   dict(config, unique_id=uuid.uuid4().hex,
                        cache_buffer_capacity_gb=1, share_buffer_enable=False,
                        cache_load_exclusive_buffer_number=128,
                        waiting_queue_depth=8, running_queue_depth=16,
                        cache_stream_number=1))
    ref = weakref.ref(pipeline)
    domain = native.EarlyDomainInfo(pipeline)
    assert domain["backing"] == ""  # private buffer must never pass owner handshake
    assert domain["block_bytes"] == shards * shard_bytes
    cold = native.EarlyReservePrefix(pipeline, one_id, deferred)
    assert cold["total_hit_blocks"] == 1
    assert cold["remaining_l2_work"] == shards
    batch = cold["batch"]
    if deferred:
        assert batch.protected_bytes == 0
        assert not batch.activate()["done"]
    del pipeline
    gc.collect()
    assert ref() is not None  # the capability owns the exact creating pipeline
    progress = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        progress = batch.step(batch.window_bytes)
        assert progress["inflight_bytes"] <= batch.window_bytes
        if progress["done"]:
            break
        time.sleep(.001)
    assert progress["done"]
    assert progress["remaining_l2_work"] == 0
    assert progress["l2_read_shards"] == shards
    assert progress["l2_read_bytes"] == shards * shard_bytes
    assert batch.snapshot()["completed_read_bytes"] == shards * shard_bytes
    assert batch.protected_bytes == shards * shard_bytes
    assert batch.protection_limit_bytes < (1 << 30)  # native load-exclusive area

    warm = native.EarlyReservePrefix(ref(), one_id, deferred)
    assert warm["remaining_l2_work"] == 0
    if deferred:
        assert warm["batch"].protected_bytes == 0
        assert warm["batch"].activate()["done"]
    assert batch.protected_bytes == shards * shard_bytes  # shared block claim
    native.EarlyReleaseReservation(ref(), warm["handle"])
    assert batch.protected_bytes == shards * shard_bytes
    del warm

    target = np.zeros_like(source)
    destinations = np.array([[row.ctypes.data] for row in target], dtype=np.uint64)
    messages = []
    observer = Consumer("w0.1", messages.append, lambda reason: (_ for _ in ()).throw(AssertionError(reason)))
    observer.bind({"request": dict(key="generation", attempt="attempt", expected=[block])}, list(range(shards)))
    route = _ReservedLoad(ref())
    route.reservations, route.observer = [cold["handle"]], observer
    context = load_scope.set(observer.scope(["request"]))
    try:
        task = route.Load(ids, indexes, destinations)
    finally:
        load_scope.reset(context)
    assert not messages  # submission/possibly fast native completion is not Wait
    native.EarlyReleaseReservation(ref(), cold["handle"])
    assert batch.protected_bytes == 0  # native load owns its own references now
    route.Wait(task)
    assert [m["op"] for m in messages] == ["consumed"]
    np.testing.assert_array_equal(source, target)
    del cold, batch, route
    gc.collect()
    assert ref() is None


def test_dump_guard_delegates_exact_native_arguments_and_rejects_before_io(monkeypatch):
    import pytest
    from unittest.mock import Mock
    pipeline = Mock()
    route = _ReservedLoad(pipeline)
    ids, indexes, addresses = object(), [0, 1], object()
    submit = Mock(return_value=19)
    monkeypatch.setattr(native, "EarlyDump", submit)
    assert route.Dump(ids, indexes, addresses, 17) == 19
    submit.assert_called_once_with(pipeline, ids, indexes, addresses, 17)
    pipeline.Dump.assert_not_called()
    errors = []
    route.observer = Consumer("w0.0", lambda message: None, errors.append)
    route.observer.domain_active = True
    submit.side_effect = native.EarlyCapacityRejected("before I/O")
    with pytest.raises(native.StoreNotFoundError):
        route.Dump(ids, indexes, addresses, 17)
    assert not errors


def test_failed_unscoped_native_io_fences_active_domain_only(monkeypatch):
    import pytest
    from unittest.mock import Mock
    for method in ("Load", "Dump", "Wait"):
        for active in (False, True):
            pipeline = Mock()
            failure = RuntimeError("native task error")
            submit = Mock(side_effect=failure)
            if method == "Wait":
                pipeline.Wait = submit
            else:
                monkeypatch.setattr(native, "Early" + method, submit)
            errors = []
            route = _ReservedLoad(pipeline)
            route.observer = Consumer("w0.0", lambda message: None, errors.append)
            route.observer.domain_active = active
            args = (7,) if method == "Wait" else (np.zeros(0, dtype=np.uint8), [], object())
            with pytest.raises(RuntimeError) as caught:
                getattr(route, method)(*args)
            assert caught.value is failure
            assert bool(errors) is active
            expected = args if method == "Wait" else (pipeline, *args, *( [()] if method == "Load" else []))
            submit.assert_called_once_with(*expected)


def test_real_no_space_binding_rejects_without_writing_and_retry_succeeds(tmp_path):
    import pytest
    store_dir = Path(native.__file__).resolve().parent.parent
    config = dict(storage_backends=[str(tmp_path)], device_id=0,
                  tensor_size=1 << 20, shard_size=1 << 20, block_size=1 << 20,
                  data_dir_shard_bytes=1, posix_data_trans_concurrency=1,
                  posix_lookup_concurrency=1, posix_io_engine="psync", io_direct=False,
                  store_health={"enabled": False})
    pipeline = native.PipelineStore()
    pipeline.Stack("EarlyPosix", str(store_dir / "posix/libposixstore.so"), config)
    pipeline.Stack("EarlyCache", str(store_dir / "cache/libcachestore.so"),
                   dict(config, unique_id=uuid.uuid4().hex,
                        cache_buffer_capacity_gb=1, share_buffer_enable=False,
                        cache_load_exclusive_buffer_number=128,
                        waiting_queue_depth=8, running_queue_depth=16,
                        cache_stream_number=1))
    ids = np.frombuffer(b"".join(i.to_bytes(16, "little") for i in range(1025)), dtype=np.uint8)
    indexes = array.array("Q", [0] * 1025)
    data = np.full(1 << 20, 0x56, dtype=np.uint8)
    addresses = np.full((1025, 1), data.ctypes.data, dtype=np.uint64)
    # v0.8.0 creates POSIX GC control files during Setup. Capacity rejection
    # must leave that initial file set unchanged and create no KV data files.
    setup_files = {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()}
    for operation in (lambda: native.EarlyLoad(pipeline, ids, indexes, addresses, []),
                      lambda: native.EarlyDump(pipeline, ids, indexes, addresses, 0)):
        with pytest.raises(native.EarlyCapacityRejected):
            operation()
        assert {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()} == setup_files
    # The failed task left no permanent LOADING owner or byte claim.
    task = native.EarlyDump(pipeline, ids[:16], array.array("Q", [0]), addresses[:1], 0)
    pipeline.Wait(task)
    destination = np.zeros_like(data)
    task = native.EarlyLoad(pipeline, ids[:16], array.array("Q", [0]),
                            np.array([[destination.ctypes.data]], dtype=np.uint64), [])
    pipeline.Wait(task)
    np.testing.assert_array_equal(data, destination)


def test_real_binding_capacity_returns_protected_prefix_not_entire_lookup(tmp_path):
    store_dir = Path(native.__file__).resolve().parent.parent
    shard = 1 << 20
    config = dict(storage_backends=[str(tmp_path)], device_id=0,
                  tensor_size=shard, shard_size=shard, block_size=2 * shard,
                  data_dir_shard_bytes=1, posix_data_trans_concurrency=1,
                  posix_lookup_concurrency=1, posix_io_engine="psync", io_direct=False,
                  store_health={"enabled": False})
    blocks = [uuid.uuid4().bytes for _ in range(3)]
    block_ids = np.frombuffer(b"".join(blocks), dtype=np.uint8)
    ids = np.frombuffer(b"".join(block for block in blocks for _ in range(2)), dtype=np.uint8)
    indices = array.array("Q", [0, 1] * 3)
    source = np.arange(6 * shard, dtype=np.uint8).reshape(6, shard)
    addresses = np.array([[row.ctypes.data] for row in source], dtype=np.uint64)
    seed = native.PipelineStore()
    seed.Stack("Posix", str(store_dir / "posix/libposixstore.so"), config)
    seed.Wait(seed.Dump(ids, indices, addresses))
    del seed
    pipeline = native.PipelineStore()
    pipeline.Stack("EarlyPosix", str(store_dir / "posix/libposixstore.so"), config)
    pipeline.Stack("EarlyCache", str(store_dir / "cache/libcachestore.so"),
                   dict(config, unique_id=uuid.uuid4().hex, cache_buffer_capacity_gb=1,
                        share_buffer_enable=False, cache_load_exclusive_buffer_number=512,
                        waiting_queue_depth=8, running_queue_depth=16, cache_stream_number=1))
    domain = native.EarlyDomainInfo(pipeline)
    assert domain["reservable_nodes"] == 512
    # Use valid native minimum sizing; another consumer holds 508 actual slots.
    # The tested request then has four slots, i.e. two complete blocks.
    held_blocks = [uuid.uuid4().bytes for _ in range(254)]
    held_ids = np.frombuffer(b"".join(b for b in held_blocks for _ in range(2)), dtype=np.uint8)
    held_ptrs = np.full((508, 1), source[0].ctypes.data, dtype=np.uint64)
    pipeline.Wait(native.EarlyDump(pipeline, held_ids, array.array("Q", [0, 1] * 254), held_ptrs, 0))
    held = native.EarlyReservePrefix(pipeline, np.frombuffer(b"".join(held_blocks), dtype=np.uint8))
    assert held["total_hit_blocks"] == 254 and held["remaining_l2_work"] == 0
    r = native.EarlyReservePrefix(pipeline, block_ids)
    assert r["total_hit_blocks"] == 3 and r["batch"].protected_bytes == 0
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        progress = r["batch"].step(r["batch"].window_bytes)
        if progress["done"]:
            break
        time.sleep(.001)
    assert progress["done"] and progress["capacity_truncated"]
    assert progress["total_hit_blocks"] == 2 and progress["inflight_bytes"] == 0
    assert r["batch"].protected_bytes == 4 * shard
    target = np.zeros((2, shard), dtype=np.uint8)
    task = native.EarlyLoad(pipeline, ids[:32], array.array("Q", [0, 1]),
                            np.array([[row.ctypes.data] for row in target], dtype=np.uint64), [r["handle"]])
    native.EarlyReleaseReservation(pipeline, r["handle"])
    pipeline.Wait(task)
    np.testing.assert_array_equal(target, source[:2])
    # Finishing does not delete reusable native DRAM data.
    warm = native.EarlyReservePrefix(pipeline, block_ids[:16])
    assert warm["total_hit_blocks"] == 1 and warm["remaining_l2_work"] == 0
    native.EarlyReleaseReservation(pipeline, warm["handle"])
    native.EarlyReleaseReservation(pipeline, held["handle"])
