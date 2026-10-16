# Cache2 bandwidth benchmark

`cache_bw.py` replaces `cache_fake_bw.py` and `cache_posix_bw.py` with one load
benchmark. It uses no Fake backend, no `cache_load_backend_only`, and no CPU
affinity code. Run it with a built/installed UCM and the matching Torch runtime.
Logical devices `0 .. workers-1` participate in one local Cache domain.

## Cache hits

```bash
python ucm/store/test/e2e/cache_bw.py \
  --mode cache-hit --model glm-5.1 --workers 16 \
  --blocks 512 --layout layerwise --epochs 20 --warmup 3 \
  --cache-capacity-gb 32 --cache-copy sdma
```

The pipeline is just `Cache`. Workers first dump one complete request into Cache,
then repeatedly load its keys. MLA dump blocks are striped across workers by
absolute block index; GQA workers dump/load private keys. Prefill bandwidth is
printed separately and excluded from load statistics. An undersized Cache is
rejected before launch; a miss cannot silently fall through to another backend.

## Posix hits

```bash
python ucm/store/test/e2e/cache_bw.py \
  --mode posix-hit --model glm-5.1 --workers 16 \
  --blocks 512 --layout layerwise --epochs 20 --warmup 3 \
  --cache-capacity-gb 32 --cache-copy sdma \
  --storage-backends /mnt/nvme0/ucm-bw /mnt/nvme1/ucm-bw \
  --posix-engine psync --posix-concurrency 32
```

The script runs two **separate spawned process groups**, in order:

1. `Posix` workers write page-aligned host buffers, layer by layer, and wait for
   all writes/commits before exiting. No Cache is created during this phase.
2. Fresh `Cache|Posix` workers start with an empty Cache and a different Cache
   domain ID. Every warmup and measured request uses a new set of block IDs,
   each loaded exactly once. Reusing keys here would incorrectly measure Cache
   hits after the first request.

For MLA/TP16 with 512 blocks, each seed worker writes 32 unique blocks, while
each load worker requests all 512 blocks. Cache2 decides which worker owns each
backend read; the script does not divide the per-rank H2D load by 16.
Other ranks may see a shared slot become Ready after its owner finishes the
read. That is normal sharing within this cold request: one Posix read per
key/shard across the group, followed by H2D on every rank. Source metrics need
not label every rank's shard as a Posix wait.

Default Posix IO is direct, bypassing the filesystem page cache. Buffered IO is
available with `--no-direct-io`; its timings may include OS page-cache hits, so
they are not a cold-disk baseline. `--posix-engine aio` requires direct IO.
This script does not drop the machine's page cache or reserve HugeTLB pages.

Each storage backend gets a new `ucm-cache-bw-*` subdirectory. Only those
run-owned directories are deleted after workers exit, including on failure.
Use `--keep-data` to preserve them for debugging. Existing backend data is not
read, modified, or removed. Posix GC is disabled for these isolated test files.

**Check the printed `seed_storage_bytes` before running.** The example above
seeds 23 complete, distinct requests, approximately 154 GiB, rather than
reusing one request and accidentally warming Cache. Payload padding can make
Posix file bytes slightly larger than the H2D byte count.

## IO shape and model profiles

| Profile | Tensor bytes per layer/block | Layers | Default workers | Keys |
| --- | --- | --- | --- | --- |
| `glm-5.1`, `glm-5.2` | 131072 + 16384 + 32768 = 176 KiB | 78 | 16 | shared MLA |
| `minimax-m2.7-tp8` | 32768 + 32768 = 64 KiB (TP8 sizes) | 62 | 8 | private GQA |
| `dsv4` | `[131072, 16384, 256] * 21 + [4096] * 20` | one FA row | 8 | shared MLA |
| `custom` | 1 MiB | 1 | 1 | private GQA |

These are synthetic byte-layout profiles; they do not run the model. DeepSeek V4
only covers the existing FA profile, not WA, and only supports `whole-block`.
Use the MiniMax preset with TP8; use custom sizes for other TP configurations.

- `--layout layerwise`: one task per layer, on the same block IDs, with shard
  indexes `0 .. layers-1`. GLM sends **78 tasks**, each 176 KiB per block.
- `--layout whole-block`: all layer tensor views are passed in a single task.
  GLM sends **one task**, with 234 views and 176 KiB × 78 payload per block.
- `--layout single-layer`: only one layer/task. This reproduces the old script's
  single-layer microbenchmark; it does not represent a complete 78-layer request.

Except for whole-block profiles, the default is complete layerwise transfer.
Shard size is rounded up to 4 KiB for Posix, and block size includes every shard.
The H2D byte count uses actual tensor payload bytes, excluding storage padding.

Custom single-tensor IO sizes and explicit KV views are supported:

```bash
python ucm/store/test/e2e/cache_bw.py \
  --mode cache-hit --model custom --worker-mode mla --workers 16 \
  --io-size 176K --layers 78 --layout layerwise --blocks 512

python ucm/store/test/e2e/cache_bw.py \
  --mode cache-hit --model glm-5.1 \
  --tensor-sizes 128K,16K,32K --layout single-layer
```

`K/M/G` and `KiB/MiB/GiB` use binary units. `--io-size 176K` is one 180224-byte
tensor; the GLM preset preserves three separate tensor views of the same total
size, which matters for copy submission and aggregation benchmarks.

## Copy modes, NUMA, and validation

- `--cache-copy auto` (default) preserves the C++ platform defaults.
- `--cache-copy plain|aggregation|sdma` explicitly selects the Cache copy path.
- `--cache-streams N` overrides stream count; omitted means the C++ default.
- `--cache-numa-nodes 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7` places each worker's
  Cache segment explicitly. Without it, Cache2 uses contiguous rank-sliced
  placement across available memory nodes. This may differ from NPU affinity.
  No worker or UCM thread is CPU-bound by the script.
- `--verify` checks all returned bytes after each timed layer/task. Source data
  differs by key and shard; destinations are cleared before loading. This is
  for correctness checks and adds untimed device work, so omit it for pure
  bandwidth measurements.
- `--process-timeout SECONDS` bounds each process group, including initialization
  and teardown. Worker failures abort the peer barrier; remaining children are
  terminated before test directories are cleaned up.

Inspect a workload without Torch, UCM, NPU/GPU, or filesystem changes:

```bash
python ucm/store/test/e2e/cache_bw.py --mode posix-hit --model glm-5.1 --dry-run
python ucm/store/test/e2e/cache_bw.py --mode cache-hit --model dsv4 --dry-run
python ucm/store/test/e2e/cache_bw.py --help
```

## Statistics and measurement boundaries

Every request prints the average, P90 and P99 task cost, the sum of task costs, and
per-rank payload bandwidth. Final statistics show `ALL` pooled rank samples
first and each rank separately below it. Request latency and task latency have
separate distributions. Warmup, tensor allocation/filling, verification,
inter-rank barriers, and epoch pauses are excluded from transfer costs.

`request_cost` sums the sequential, synchronous layer/task timings. It measures
Store transfers, **not model TTFT**, and does not simulate computation between
layers. C++ Posix-to-Cache/H2D pipelining within a task remains active. `ALL`
bandwidth is pooled bytes divided by pooled transfer time, not aggregate physical
disk bandwidth. Percentiles with few requests are only rough samples.

Local protocol checks require only Python's standard library:

```bash
python -m unittest discover -s test -p test_cache_bw.py -v
```
