"""Byte-exact NPU cache transfer probe using the GLM 8-layer direct layout.

Does not initialize a model or infer semantic/numerical model correctness.
Each invocation owns a new L2 directory and native shared-memory identity.
"""
import argparse
import array
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch
import torch_npu
from ucm.store.pipeline.connector import ucmpipelinestore as native

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['native', 'early'])
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir()
(a.output / "l2").mkdir()
shard_sizes = [131072, 16384, 32768] * 8
block_bytes = sum(shard_sizes)
blocks = 27
torch.npu.set_device(0)
rng = np.random.default_rng(20260917)
expected = [rng.integers(0, 256, size=(blocks, n), dtype=np.uint8) for n in shard_sizes]
ids_list = [uuid.uuid4().bytes for _ in range(blocks)]
ids = np.frombuffer(b''.join(ids_list), dtype=np.uint8)
indexes = array.array('Q', [0] * blocks)
packed = np.stack([np.concatenate([data[i] for data in expected]) for i in range(blocks)])
config = dict(storage_backends=[str(a.output / 'l2')], device_id=0,
              tensor_size=block_bytes, tensor_size_list=shard_sizes,
              shard_size=block_bytes, block_size=block_bytes,
              data_dir_shard_bytes=2, io_direct=False, posix_io_engine='psync',
              posix_data_trans_concurrency=1, posix_lookup_concurrency=1,
              posix_open_concurrency=1, posix_commit_concurrency=1,
              store_health={'enabled': False}, posix_gc_enable=False)
store_dir = Path(native.__file__).resolve().parent.parent
seed = native.PipelineStore()
seed.Stack('Posix', str(store_dir / 'posix/libposixstore.so'), config)
seed.Wait(seed.Dump(ids, indexes, np.array([[row.ctypes.data] for row in packed], dtype=np.uint64)))
assert seed.LookupOnPrefix(ids) == blocks - 1
# Fresh native L1; only L2 is populated. Use real NPU destination addresses.
targets = [torch.zeros((blocks, n), dtype=torch.uint8, device='npu:0') for n in shard_sizes]
torch.npu.synchronize()
addresses = np.array([[t.data_ptr() + i * n for t, n in zip(targets, shard_sizes)] for i in range(blocks)], dtype=np.uint64)
config.update(unique_id=uuid.uuid4().hex, cache_buffer_capacity_gb=3,
              share_buffer_enable=True, cache_load_exclusive_buffer_number=1024,
              cache_sdma_direct=True, cache_stream_number=1, local_rank_size=1,
              gpu_kv_buffer_addrs=[t.data_ptr() for t in targets],
              gpu_kv_buffer_sizes=[t.numel() for t in targets])
config['posix_io_engine'] = 'aio'
config['io_direct'] = True
pipe = native.PipelineStore()
pipe.Stack('EarlyPosix' if a.mode == 'early' else 'Posix', str(store_dir / 'posix/libposixstore.so'), config)
pipe.Stack('EarlyCache' if a.mode == 'early' else 'Cache', str(store_dir / 'cache/libcachestore.so'), {k:v for k,v in config.items() if k != 'tensor_size'})
report = dict(mode=a.mode, block_bytes=block_bytes, blocks=blocks, sizes=shard_sizes,
              config=config, module=str(native.__file__), cases=[])
def check(case):
    torch.npu.synchronize()
    differences = [int(np.count_nonzero(t.cpu().numpy() != e)) for t, e in zip(targets, expected)]
    report['cases'].append(dict(case=case, mismatched_bytes=sum(differences), per_tensor=differences))
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('NPU_BUFFER_PROBE', case, 'mismatched_bytes', sum(differences), flush=True)
    assert not any(differences), (case, differences)
def clear():
    for t in targets:
        t.zero_()
    torch.npu.synchronize()
def load(reservations=()):
    task = native.EarlyLoad(pipe, ids, indexes, addresses, reservations) if a.mode == 'early' else pipe.Load(ids, indexes, addresses)
    pipe.Wait(task)
if a.mode == 'early':
    cold = native.EarlyReservePrefix(pipe, ids)
    assert cold['total_hit_blocks'] == blocks
    assert cold['remaining_l2_work'] == blocks
    result = native.EarlyPrefetchReserved(pipe, cold['handle'])
    report['prefetch'] = result
    load([cold['handle']])
    native.EarlyReleaseReservation(pipe, cold['handle'])
else:
    load()
check('cold_l1_hot_l2')
clear()
if a.mode == 'early':
    warm = native.EarlyReservePrefix(pipe, ids)
    assert warm['remaining_l2_work'] == 0
    # No owner reservation supplied: exercise non-owner bounded native Get.
    load()
    native.EarlyReleaseReservation(pipe, warm['handle'])
else:
    load()
check('warm_l1_without_owner_handles')
# Fresh IDs force D2H/write. Original native task granularity is unchanged.
dump_ids_list = [uuid.uuid4().bytes for _ in range(blocks)]
dump_ids = np.frombuffer(b''.join(dump_ids_list), dtype=np.uint8)
task = native.EarlyDump(pipe, dump_ids, indexes, addresses, 0) if a.mode == 'early' else pipe.Dump(dump_ids, indexes, addresses)
pipe.Wait(task)
# Native dump Wait observes D2H; separately verify persistence, not just commit visibility.
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    found = {path.name:path for path in (a.output / 'l2').rglob('*') if path.is_file()}
    if all(block.hex() in found and found[block.hex()].read_bytes() == row.tobytes() for block,row in zip(dump_ids_list,packed)):
        break
    time.sleep(.01)
else:
    raise AssertionError('dump did not persist expected bytes')
report['dump_persisted_bytes'] = packed.nbytes
clear()
ids = dump_ids
load()
check('bounded_dump_then_reload')
report['passed'] = True
(a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print('NPU_BUFFER_PROBE_PASSED', a.mode, flush=True)
