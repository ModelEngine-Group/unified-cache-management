"""Single-node Gloo launch, native rank discovery and Store shard expectations."""

import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from math import lcm
from pathlib import Path


def launch(args):
    """Let torchrun supervise all local ranks, including failures in a collective."""
    world_size = args.tp * args.pcp
    if "RANK" in os.environ:
        if (
            int(os.environ["WORLD_SIZE"]) != world_size
            or int(os.environ["LOCAL_WORLD_SIZE"]) != world_size
            or int(os.environ["LOCAL_RANK"]) != int(os.environ["RANK"])
        ):
            raise ValueError("Expected a single-node torchrun with TP × PCP processes")
        return False
    if world_size == 1:
        return False
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(Path(__file__).with_name("run.py")),
            *sys.argv[1:],
        ],
        check=True,
    )
    return True


def gather(value):
    import torch.distributed as dist

    values = [None] * dist.get_world_size()
    if len(values) == 1:
        return [value]
    dist.all_gather_object(values, value)
    return values


def broadcast(value):
    import torch.distributed as dist

    items = [value]
    if dist.get_world_size() > 1:
        dist.broadcast_object_list(items, src=0)
    return items[0]


@dataclass(frozen=True)
class Placement:
    shards: tuple  # (TP KV shard, native DCP rank), in record order
    index: int
    replicas: tuple  # global ranks sharing the local shard, in writer rotation order
    rank: int
    dcp_size: int
    interleave: int

    def local_position(self, position, shard):
        # Count tokens before 'position' owned by this DCP rank in one page.
        rounds, tail = divmod(position, self.dcp_size * self.interleave)
        return rounds * self.interleave + min(
            max(tail - shard * self.interleave, 0), self.interleave
        )


def native_tp_shards(spec, tp):
    from vllm.v1.kv_cache_interface import (
        MLAAttentionSpec,
        SlidingWindowMLASpec,
        UniformTypeKVCacheSpecs,
    )

    members = (
        spec.kv_cache_specs.values()
        if isinstance(spec, UniformTypeKVCacheSpecs)
        else [spec]
    )
    counts = []
    for member in members:
        if isinstance(member, (MLAAttentionSpec, SlidingWindowMLASpec)) or getattr(
            member, "tp_replicated", False
        ):
            counts.append(1)
        else:
            counts.append(min(tp, getattr(member, "max_tp_shards", None) or tp))
    return lcm(*counts)


def inspect_workers(worker, cfg, args):
    import torch.distributed as dist
    from vllm.distributed.parallel_state import (
        get_dcp_group,
        get_pcp_group,
        get_tp_group,
    )

    rank = dist.get_rank()
    groups = (get_tp_group(), get_pcp_group(), get_dcp_group())
    if tuple(group.world_size for group in groups) != (args.tp, args.pcp, args.dcp):
        raise AssertionError(
            "Native TP/PCP/DCP group sizes differ from the requested topology"
        )
    local = tuple(group.rank_in_group for group in groups)
    reports = gather((local, worker.get_handshake_metadata()))
    handshakes = {}
    for world_rank, ((tp, pcp, dcp), handshake) in enumerate(reports):
        if (pcp, tp) != divmod(world_rank, args.tp):
            raise AssertionError(
                f"Unexpected native coordinates for rank {world_rank}: {pcp}, {tp}"
            )
        if (handshake is not None) != (pcp == 0):
            raise AssertionError(
                f"Incorrect handshake participation on rank {world_rank}"
            )
        if handshake is not None:
            if (
                handshake.rank != (tp, 0, pcp)
                or handshake.cache_block_size
                != worker.connector_worker._cache_block_size
            ):
                raise AssertionError(f"Inconsistent handshake from rank {world_rank}")
            key = (
                (0, pcp, tp)
                if args.platform in ("ascend-sim", "ascend") and args.pcp > 1
                else (0, tp)
            )
            handshakes[key] = handshake
    if worker.context.rank != (local[0], 0, local[1]):
        raise AssertionError(f"Worker {rank} disagrees with native process groups")
    placements = {}
    for gid in worker.connector_worker.layout.store_layouts:
        spec = cfg.kv_cache_groups[gid].kv_cache_spec
        tp_shards = native_tp_shards(spec, args.tp)
        if hasattr(spec, "dcp_sharded"):
            from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size

            dcp_size = resolve_dcp_kv_block_size(spec, args.dcp) // spec.block_size
        else:
            from vllm.v1.core.kv_cache_utils import dcp_world_size_for_kv_cache_spec

            dcp_size = dcp_world_size_for_kv_cache_spec(spec, args.dcp)
        owners = defaultdict(list)
        for world_rank, ((tp, pcp, dcp), _) in enumerate(reports):
            shard = tp * tp_shards // args.tp, dcp if dcp_size > 1 else 0
            owners[shard].append(world_rank)
        shards = tuple(sorted(owners))
        index = next(i for i, shard in enumerate(shards) if rank in owners[shard])
        placements[gid] = Placement(
            shards,
            index,
            tuple(owners[shards[index]]),
            rank,
            dcp_size,
            worker.context.parallel_topology.cp_kv_cache_interleave_size,
        )
        if rank == 0:
            print(
                f"group {gid}: shard (tp,dcp) -> writer rotation {dict(sorted(owners.items()))}",
                flush=True,
            )
    return handshakes, placements


def check_dump_coverage(local_intervals, sizes):
    """Check one complete, nonoverlapping planned write across ranks per key."""
    import torch.distributed as dist

    if dist.get_world_size() == 1:
        return
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(local_intervals, gathered, dst=0)
    if gathered is None:
        return
    if any(rank.keys() - sizes.keys() for rank in gathered):
        raise AssertionError("Worker returned an unplanned Dump key")
    for key, size in sizes.items():
        cursor = 0
        for start, stop in sorted(
            span for rank in gathered for span in rank.get(key, ())
        ):
            if start != cursor:
                raise AssertionError(
                    f"Dump gap/overlap for {key.hex()}: expected offset {cursor}, got {start}"
                )
            cursor = stop
        if cursor != size:
            raise AssertionError(
                f"Incomplete distributed Dump for {key.hex()}: {cursor}/{size} bytes"
            )
