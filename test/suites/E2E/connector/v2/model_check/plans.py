"""Observe the Connector plans and transfer descriptors without opening a Store."""

from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch, sentinel

from parallel import check_dump_coverage

from ucm.integration.vllm.v2 import connector_scheduler, connector_worker
from ucm.integration.vllm.v2.runtime import make_ucm_chunk_ids


@contextmanager
def unopened_clients():
    with (
        patch.object(connector_scheduler, "StoreSchedulerClient", lambda *args: None),
        patch.object(connector_worker, "StoreWorkerClient", lambda *args: None),
    ):
        yield


def assume_prefix(role, hashes, hit_chunks):
    """Supply a Store hit snapshot to the real Connector planner."""
    available = {
        key
        for keys in role._lookup_key_groups(hashes[:hit_chunks]).values()
        for key in keys
    }

    def lookup_on_prefix(request_id, keys):
        return (
            next(
                (i - 1 for i, key in enumerate(keys) if key not in available),
                len(keys) - 1,
            ),
            False,
        )

    def lookup_on_reverse(request_id, keys):
        return (
            next((i for i in reversed(range(len(keys))) if keys[i] in available), -1),
            False,
        )

    role._store_scheduler = SimpleNamespace(
        lookup_on_prefix=lookup_on_prefix,
        lookup_on_reverse=lookup_on_reverse,
        on_new_request=lambda request_id: None,
        on_request_finished=lambda request_id: None,
    )
    # None means Prepare failed in dispatch metadata. Mark the plan observable
    # without creating a token, tracking a pending job or reporting IO success.
    role._prepare_plan_job = lambda *args, **kwargs: sentinel.unsubmitted_plan


def check_batches(worker, plans, records, hashes, *, is_dump, checkpoints=None):
    """Compare the Worker payload at the Submit boundary with native tensor slices."""
    chunk = worker.spec.chunk_size
    expected = {}
    keys_by_window = {}
    for (policy, start), record in records.items():
        (key,) = make_ucm_chunk_ids((hashes[start // chunk],), policy)
        if key in expected:
            raise AssertionError(f"Duplicate key for {policy.name} chunk {start}")
        keys_by_window[policy, start] = key
        expected[key] = record + (checkpoints or {}).get((policy, start), [])

    declared = set()
    actual = defaultdict(list)
    for plan in plans:
        for start, key in zip(plan.token_starts, plan.ucm_keys, strict=True):
            window = (plan.restore_policy, int(start))
            if window not in records or window in declared:
                raise AssertionError(f"Unexpected or duplicate plan for {window}")
            if key != keys_by_window[window]:
                raise AssertionError(f"Wrong key for {window}")
            declared.add(window)
        layers = worker.layout.layer_ids if worker.use_layerwise else (None,)
        for layer in layers:
            for batch in worker._build_transfer_batches(
                plan, is_dump=is_dump, layer_id=layer
            ):
                for key, offsets, ptrs, sizes in zip(
                    batch.ucm_keys,
                    batch.store_offsets,
                    batch.device_ptrs,
                    batch.byte_sizes,
                    strict=True,
                ):
                    actual[key].extend(
                        zip(
                            map(int, offsets),
                            map(int, ptrs),
                            map(int, sizes),
                            strict=True,
                        )
                    )

    if declared != records.keys() or not actual.keys() <= expected.keys():
        raise AssertionError(
            f"Unexpected payload or missing plan: expected {len(records)} windows, got {len(declared)}"
        )
    coverage = {}
    for key, ranges in expected.items():
        observed = sorted(actual.get(key, ()))
        if observed != sorted(ranges):
            op = "Dump" if is_dump else "Get"
            raise AssertionError(
                f"rank {worker.context.global_rank}: {op} offsets, pointers or sizes disagree for {key.hex()}"
            )
        if is_dump:
            intervals = []
            for offset, _, size in observed:
                if intervals and intervals[-1][1] == offset:
                    intervals[-1] = (intervals[-1][0], offset + size)
                else:
                    intervals.append((offset, offset + size))
            if intervals:
                coverage[key] = intervals
    if is_dump:
        sizes = {
            key: worker.layout.store_sizes_by_policy[policy]
            for (policy, _), key in keys_by_window.items()
        }
        check_dump_coverage(coverage, sizes)
