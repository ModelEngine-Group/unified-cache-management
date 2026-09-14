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

from ucm.store.pipeline import ucmpipelinestore as native
from ucm.store.pipeline.early_connector import _ReservedLoad
from ucm.integration.vllm.early_consumer import Consumer, load_scope


def test_native_pipeline_batch_keeps_store_alive_and_hands_off_before_release(tmp_path):
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
    cold = native.EarlyReservePrefix(pipeline, one_id)
    assert cold["total_hit_blocks"] == 1
    assert cold["remaining_l2_work"] == shards
    batch = cold["batch"]
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

    warm = native.EarlyReservePrefix(ref(), one_id)
    assert warm["remaining_l2_work"] == 0
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


def test_dump_guard_delegates_exact_native_arguments_and_rejects_before_io():
    import pytest
    from unittest.mock import Mock
    pipeline = Mock()
    route = _ReservedLoad(pipeline)
    ids, indexes, addresses = object(), [0, 1], object()
    assert route.Dump(ids, indexes, addresses, 17) is pipeline.Dump.return_value
    pipeline.Dump.assert_called_once_with(ids, indexes, addresses, 17)
    pipeline.reset_mock()
    errors = []
    route.observer = Consumer("w0.0", lambda message: None, errors.append)
    route.observer.max_dump_task_shards = 1
    with pytest.raises(RuntimeError, match="headroom"):
        route.Dump(ids, indexes, addresses, 17)
    pipeline.Dump.assert_not_called()
    assert len(errors) == 1


def test_failed_unscoped_native_io_fences_active_domain_only():
    import pytest
    from unittest.mock import Mock
    for method in ("Load", "Dump", "Wait"):
        for active in (False, True):
            native = Mock()
            failure = RuntimeError("native task error")
            getattr(native, method).side_effect = failure
            errors = []
            route = _ReservedLoad(native)
            route.observer = Consumer("w0.0", lambda message: None, errors.append)
            if active:
                route.observer.max_task_shards = route.observer.max_dump_task_shards = 2
            args = (7,) if method == "Wait" else (np.zeros(0, dtype=np.uint8), [], object())
            with pytest.raises(RuntimeError) as caught:
                getattr(route, method)(*args)
            assert caught.value is failure  # preserve the native exception
            assert bool(errors) is active
            getattr(native, method).assert_called_once_with(*args)
