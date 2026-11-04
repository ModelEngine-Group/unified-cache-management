"""Standalone real Gloo test; no CUDA/NPU, KV tensors or engine processes."""

import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Load the same native interface stubs as the component tests. torch.distributed
# itself is real; only vLLM's already-created PP coordinator is injected.
import test_ucm_connector_v2 as fixtures
import torch.distributed as dist
import torch.multiprocessing as mp

from ucm.integration.vllm.v2.connector_worker import (
    exchange_pp_metadata,
)
from ucm.integration.vllm.v2.layout.group import GroupLayoutMetadata
from ucm.integration.vllm.v2.runtime import ParallelTopology


def worker(rank, root):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=30),
    )
    outcomes = []
    # Both native rank layouts use eight processes. Each lane has two PP ranks.
    for scenario in ("dp2", "pcp2"):
        if scenario == "dp2":
            pp_groups = [
                [dp * 4 + pp * 2 + tp for pp in range(2)]
                for dp in range(2)
                for tp in range(2)
            ]
            dp, pp, pcp, tp = rank // 4, (rank % 4) // 2, 0, rank % 2
            topology = ParallelTopology(tp_size=2, pp_size=2)
        else:
            # vLLM rank order is DP x PP x PCP x TP.
            pp_groups = [
                [pp * 4 + pcp * 2 + tp for pp in range(2)]
                for pcp in range(2)
                for tp in range(2)
            ]
            dp, pp, pcp, tp = 0, rank // 4, (rank % 4) // 2, rank % 2
            topology = ParallelTopology(tp_size=2, pp_size=2, pcp_size=2)
        own_group = None
        for ranks in pp_groups:
            process_group = dist.new_group(
                ranks, backend="gloo", timeout=timedelta(seconds=30)
            )
            if rank in ranks:
                own_group = process_group
        lane_id = dp * 4 + pcp * 2 + tp
        with patch(
            "vllm.distributed.parallel_state.get_pp_group",
            create=True,
            return_value=SimpleNamespace(cpu_group=own_group),
        ):
            for case in ("unequal", "empty", "error"):
                if (pp + tp + pcp) % 2:
                    time.sleep(0.03)
                sizes = (
                    ()
                    if case == "empty" and pp == 0
                    else (7 + pp * 11 + lane_id * 100,)
                )
                local = (
                    pp,
                    {0: GroupLayoutMetadata(sizes, (4,) * len(sizes), True, 2)},
                    "invalid view" if case == "error" and pp == 1 else None,
                )
                try:
                    stages = exchange_pp_metadata(local, topology)
                except RuntimeError:
                    assert case == "error"
                    outcomes.append("error-propagated")
                else:
                    assert case != "error"
                    assert [stage[0] for stage in stages] == [0, 1]
                    for pp_rank, groups, _ in stages:
                        expected_sizes = (
                            ()
                            if case == "empty" and pp_rank == 0
                            else (7 + pp_rank * 11 + lane_id * 100,)
                        )
                        assert groups[0].sizes == expected_sizes
                    outcomes.append("ok")
    dp, pp, tp = rank // 4, (rank % 4) // 2, rank % 2
    # Real concurrent range writes from eight independent Store instances.
    import numpy as np

    from ucm.integration.vllm.v2.test_stores import FileTestStore

    memory = fixtures.ByteMemory()
    size = 7 if pp == 0 else 18
    offset = tp * 7 if pp == 0 else 14 + tp * 18
    memory.write(100, bytes((dp * 50 + pp * 10 + tp,)) * size)
    store = FileTestStore(root, fixtures.MemoryByteAccess(memory))
    attempt = bytes((dp + 1,)) * 16
    target = bytes((dp + 11,)) * 16
    store.dump(
        (attempt,),
        np.array([[offset]], dtype=np.uint64),
        np.array([[100]], dtype=np.uint64),
        np.array([[size]], dtype=np.uint64),
        ('{"shared_attempt":true}',),
        np.array([50], dtype=np.uint64),
    )
    assert store.lookup((target,)) == (False,)
    dist.barrier()
    # Concurrent publication retries are immutable and idempotent too.
    store.commit(True, (attempt,), (target,))
    dist.barrier()
    expected = b"".join(
        bytes((dp * 50 + stage * 10 + t,)) * (7 if stage == 0 else 18)
        for stage in range(2)
        for t in range(2)
    )
    assert store.read_record(target)[0] == expected
    store.close()
    Path(root, f"{rank}.json").write_text(json.dumps(outcomes))
    dist.destroy_process_group()


if __name__ == "__main__":
    root = sys.argv[1]
    mp.spawn(worker, args=(root,), nprocs=8, join=True)
    assert all(Path(root, f"{rank}.json").is_file() for rank in range(8))
    assert not list(Path(root).glob(".*.attempt"))
    print(
        "GLOO_PP_PASS: TP2 PP2 DP2 and TP2 PP2 PCP2, unequal/empty/error declarations"
    )
