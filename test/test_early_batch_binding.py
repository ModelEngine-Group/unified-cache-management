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
    task = native.EarlyLoad(ref(), ids, indexes, destinations, [cold["handle"]])
    native.EarlyReleaseReservation(ref(), cold["handle"])
    assert batch.protected_bytes == 0  # native load owns its own references now
    ref().Wait(task)
    np.testing.assert_array_equal(source, target)
    del cold, batch
    gc.collect()
    assert ref() is None
