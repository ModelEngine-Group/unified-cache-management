"""Global object layout and Scheduler publication, using actual v2 components."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from test_ucm_connector_v2 import (
    ByteMemory,
    FakeTensor,
    FullAttentionSpec,
    KVConnectorRole,
    MemoryByteAccess,
    config,
    group,
    make_connector_scheduler,
    parse_spec,
)

from ucm.integration.vllm.v2.connector_scheduler import (
    RequestDispatchMeta,
    UCMConnectorMetadata,
    UCMConnectorScheduler,
    UCMGroupDispatchPlan,
)
from ucm.integration.vllm.v2.connector_worker import (
    SaveResult,
    UCMConnectorWorker,
    UCMWorkerMetadata,
    exchange_pp_metadata,
)
from ucm.integration.vllm.v2.layout import RestorePolicy, UCMKVCacheLayout
from ucm.integration.vllm.v2.layout.group import GroupLayoutMetadata
from ucm.integration.vllm.v2.layout.store_layout import (
    GroupStoreLayout,
    UCMShard,
    shard_replicas,
)
from ucm.integration.vllm.v2.runtime import (
    ParallelRank,
    ParallelTopology,
    UCMRuntimeContext,
    make_ucm_chunk_ids,
)
from ucm.integration.vllm.v2.test_stores import FileTestStore, MemoryTestStore
from ucm.integration.vllm.v2.ucm_proxy import PipelineProxy, UCMProxyAdapter


def fixture_spec(native=None):
    native = native or FullAttentionSpec(4)
    return parse_spec(
        config(group(["model.layers.0.attn"], native)),
        scheduler_block_size=4,
        device_type="cpu",
    )


@pytest.mark.parametrize(
    "tp,pcp,dcp,shards,token_sharded,pieces,replicas",
    [
        (4, 1, 1, 1, True, 1, 4),
        (4, 1, 2, 1, True, 2, 2),
        (8, 1, 4, 1, True, 4, 2),
        (4, 1, 2, None, True, 4, 1),
        (4, 1, 2, 1, False, 1, 4),
        (4, 1, 2, None, False, 4, 1),
        (4, 1, 1, 2, True, 2, 2),
        (2, 2, 1, None, True, 2, 2),
        (2, 2, 2, 1, True, 2, 2),
    ],
)
def test_valid_shard_pairs_and_exactly_one_writer_per_piece(
    tp, pcp, dcp, shards, token_sharded, pieces, replicas
):
    topology = ParallelTopology(tp_size=tp, pcp_size=pcp, dcp_size=dcp)
    owners = shard_replicas(
        topology, 0, tp if shards is None else shards, token_sharded
    )
    assert len(owners) == pieces
    assert {len(ranks) for ranks in owners.values()} == {replicas}
    for index in range(3, 19):
        assert (
            len([(shard, ranks[index % len(ranks)]) for shard, ranks in owners.items()])
            == pieces
        )
    if tp == 4 and dcp == 2 and shards == 1 and token_sharded:
        assert [
            (s.tp_shard, s.dcp_shard, tuple(r.tp_rank for r in ranks))
            for s, ranks in owners.items()
        ] == [(0, 0, (0, 2)), (0, 1, (1, 3))]


def test_mixed_tp_group_repeats_complete_stage_templates_and_pp_stays_distinct():
    spec = fixture_spec()
    pp0 = GroupLayoutMetadata((5, 3, 7, 11), (4, 4, 4, 4), True, 2)
    pp1 = GroupLayoutMetadata((13,), (4,), True, 2)
    metadata = ((0, {0: pp0}, None), (1, {0: pp1}, None))
    store = GroupStoreLayout.build(
        spec, 0, None, metadata, ParallelTopology(tp_size=2, pp_size=2), ParallelRank()
    )
    # The formerly replicated 5+3 bytes are now intentionally repeated.
    assert store.tp_shards == 2
    assert store.chunk_segment_bytes.tolist() == [[5, 3, 7, 11]]
    assert store.chunk_segment_offsets.tolist() == [[0, 5, 8, 15]]
    assert store.whole_block_nbytes == store.whole_chunk_nbytes == 78


def system(
    tmp_path,
    *,
    layerwise=False,
    pp=2,
    tp=2,
    cache=False,
    native_spec=None,
    empty_stage=False,
):
    """PP exchange is injected here; a separate subprocess test uses real Gloo."""
    topology = ParallelTopology(tp_size=tp, pp_size=pp)
    definitions = []
    for stage in range(pp):
        widths = (2, 3) if stage == 0 else (4,)
        if empty_stage and stage == 1:
            widths = ()
        names = tuple(f"model.layers.{stage * 3 + i}.attn" for i in range(len(widths)))
        native = config(
            group(
                names,
                (
                    native_spec(stage, names)
                    if callable(native_spec)
                    else native_spec or FullAttentionSpec(4)
                ),
            )
        )
        spec = parse_spec(native, scheduler_block_size=4, device_type="cpu")
        caches = {
            name: FakeTensor(0x1000 + i * 0x1000, (8, 4, width), (4 * width, width, 1))
            for i, (name, width) in enumerate(zip(names, widths, strict=True))
        }
        layout = UCMKVCacheLayout(
            spec,
            caches,
            kv_cache_config=native,
            num_hidden_layers=16,
            use_layerwise=layerwise,
        )
        definitions.append(
            (
                native,
                spec,
                caches,
                widths,
                (stage, layout.export_layout_metadata(topology.tp_size), None),
            )
        )
    stages = tuple(d[-1] for d in definitions)
    spec = definitions[0][1]
    context = UCMRuntimeContext(
        role=KVConnectorRole.SCHEDULER,
        device_type="cpu",
        engine_id="test",
        global_rank=None,
        rank=None,
        parallel_topology=topology,
        launch_config={},
        use_layerwise=layerwise,
        spec=spec,
        fingerprint="global-test",
    )
    scheduler_store = FileTestStore(tmp_path)
    adapter = UCMProxyAdapter(PipelineProxy(scheduler_store))
    with patch(
        "ucm.integration.vllm.v2.connector_scheduler.create_proxy", return_value=adapter
    ):
        scheduler = UCMConnectorScheduler(context)
    workers = []
    handshakes = {}
    for stage, (native, local_spec, caches, widths, _) in enumerate(definitions):
        for tp_rank in range(tp):
            rank = ParallelRank(tp_rank, stage, 0)
            memory = ByteMemory()
            access = MemoryByteAccess(memory)
            store = FileTestStore(tmp_path, access)
            if cache:
                store = MemoryTestStore(access, backend=store, capacity_bytes=1)
            with patch(
                "ucm.integration.vllm.v2.connector_worker.create_proxy",
                return_value=UCMProxyAdapter(PipelineProxy(store)),
            ):
                worker = UCMConnectorWorker(
                    replace(
                        context, role=KVConnectorRole.WORKER, rank=rank, spec=local_spec
                    ),
                    native,
                    num_hidden_layers=16,
                )
            with patch(
                "ucm.integration.vllm.v2.connector_worker.exchange_pp_metadata",
                return_value=stages,
            ):
                worker.register_kv_caches(caches)
            for i, width in enumerate(widths):
                for chunk, block in enumerate((1, 2)):
                    memory.write(
                        0x1000 + i * 0x1000 + block * width * 4,
                        bytes((stage * 40 + tp_rank * 10 + i + chunk * 100,))
                        * width
                        * 4,
                    )
            workers.append((worker, memory, widths))
            handshakes[(stage, tp_rank)] = worker.get_handshake_metadata()
    scheduler.set_xfer_handshake_metadata(handshakes)
    common = (b"a" * 15, b"b" * 15)
    plan = UCMGroupDispatchPlan(
        RestorePolicy.FA,
        common,
        np.array([0, 4], dtype=np.uint64),
        (np.array([1, 2], dtype=np.uint64),),
        (0,),
    )
    request = scheduler._attach_save_jobs(RequestDispatchMeta("r", dump_plans=(plan,)))
    return (
        scheduler,
        scheduler_store,
        workers,
        UCMConnectorMetadata(requests={"r": request}),
    )


def report(scheduler, metadata):
    scheduler.update_connector_output(
        SimpleNamespace(kv_connector_worker_meta=metadata)
    )


@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("finish_first", [False, True])
@pytest.mark.parametrize("cache", [False, True])
def test_shared_ranges_publish_at_request_end_and_roundtrip(
    tmp_path, layerwise, finish_first, cache
):
    scheduler, store, workers, metadata = system(
        tmp_path, layerwise=layerwise, cache=cache
    )
    keys = make_ucm_chunk_ids((b"a" * 15, b"b" * 15))
    assert workers[0][0].transfer_builder.record_nbytes[RestorePolicy.FA] == 72
    if finish_first:
        scheduler.request_finished("r")
        scheduler.requests.pop("r", None)
    try:
        for i, (worker, _, _) in enumerate(workers):
            worker.bind_connector_metadata(metadata)
            worker.wait_for_save()
            assert not worker._dump_tasks
            result = worker.build_connector_worker_meta()
            report(scheduler, result)
            report(scheduler, result)  # Repeated worker output is idempotent.
            if not finish_first or i < len(workers) - 1:
                assert store.lookup(keys) == (False, False)
            worker.clear_connector_metadata()
        scheduler.request_finished("r")
        assert store.lookup(keys) == (True, True)
        for chunk, key in enumerate(keys):
            expected = b"".join(
                bytes((stage * 40 + tp_rank * 10 + layer + chunk * 100,)) * width * 4
                for stage, widths in enumerate(((2, 3), (4,)))
                for tp_rank in range(2)
                for layer, width in enumerate(widths)
            )
            assert store.read_record(key)[0] == expected
        assert not scheduler._pending_saves
        assert not list(tmp_path.glob("*.attempt"))
        load = replace(
            metadata.requests["r"].dump_plans[0],
            vllm_block_ids_by_group=(np.array([5, 6], dtype=np.uint64),),
        )
        load_meta = UCMConnectorMetadata(
            requests={"target": RequestDispatchMeta("target", load_plans=(load,))}
        )
        for worker, memory, widths in workers:
            worker.bind_connector_metadata(load_meta)
            worker.start_load_kv(None)
            worker.wait_for_save()
            for layer, width in enumerate(widths):
                for chunk, block in enumerate((5, 6)):
                    rank = worker._parallel_rank
                    assert (
                        memory.read(
                            0x1000 + layer * 0x1000 + block * width * 4, width * 4
                        )
                        == bytes(
                            (
                                rank.pp_rank * 40
                                + rank.tp_rank * 10
                                + layer
                                + chunk * 100,
                            )
                        )
                        * width
                        * 4
                    )
            worker.clear_connector_metadata()
    finally:
        for worker, _, _ in workers:
            worker.shutdown()
        scheduler.shutdown()


def test_failed_object_does_not_hide_other_complete_chunks_and_wrong_writer_rejected(
    tmp_path,
):
    scheduler, store, workers, metadata = system(tmp_path)
    jobs = metadata.requests["r"].save_jobs
    bad = jobs[0]
    with pytest.raises(ValueError, match="expected worker rank"):
        report(
            scheduler,
            UCMWorkerMetadata(
                save_results={SaveResult(bad.attempt_id, ParallelRank(999, 0, 0), True)}
            ),
        )
    scheduler.request_finished("r")
    for worker, _, _ in workers:
        worker.bind_connector_metadata(metadata)
        worker.wait_for_save()
        result = worker.build_connector_worker_meta()
        result.save_results = {
            replace(r, success=False) if r.attempt_id == bad.attempt_id else r
            for r in result.save_results
        }
        report(scheduler, result)
        worker.clear_connector_metadata()
        worker.shutdown()
    assert store.lookup(tuple(job.ucm_block_id for job in jobs)) == (False, True)
    assert not scheduler._pending_saves
    assert not list(tmp_path.glob(".*.attempt"))
    scheduler.shutdown()


def test_two_requests_same_object_have_isolated_attempts_and_immutable_publish(
    tmp_path,
):
    scheduler, store, workers, metadata = system(tmp_path, pp=1, tp=1)
    first = metadata.requests["r"]
    other = scheduler._attach_save_jobs(
        replace(first, request_id="other", save_jobs=())
    )
    assert {j.ucm_block_id for j in first.save_jobs} == {
        j.ucm_block_id for j in other.save_jobs
    }
    assert not {j.attempt_id for j in first.save_jobs} & {
        j.attempt_id for j in other.save_jobs
    }
    worker, memory, _ = workers[0]
    worker.bind_connector_metadata(metadata)
    worker.wait_for_save()
    result = worker.build_connector_worker_meta()
    report(scheduler, result)
    scheduler.request_finished("r")
    original = [store.read_record(job.ucm_block_id)[0] for job in first.save_jobs]
    worker.clear_connector_metadata()
    memory.write(0x1000 + 8, b"Z" * 8)
    worker.bind_connector_metadata(UCMConnectorMetadata(requests={"other": other}))
    worker.wait_for_save()
    report(scheduler, worker.build_connector_worker_meta())
    scheduler.request_finished("other")
    assert [
        store.read_record(job.ucm_block_id)[0] for job in first.save_jobs
    ] == original
    worker.clear_connector_metadata()
    worker.shutdown()
    scheduler.shutdown()


@pytest.mark.parametrize("failure", ["commit_true", "commit_false"])
def test_publication_and_cleanup_io_failures_are_best_effort(tmp_path, failure):
    scheduler, store, workers, metadata = system(tmp_path, pp=1, tp=1)
    worker, _, _ = workers[0]
    worker.bind_connector_metadata(metadata)
    worker.wait_for_save()
    result = worker.build_connector_worker_meta()
    if failure == "commit_false":
        result.save_results = {replace(r, success=False) for r in result.save_results}
    report(scheduler, result)
    with patch.object(store, "commit", side_effect=OSError("test failure")):
        scheduler.request_finished("r")
    assert not scheduler._pending_saves
    assert store.lookup(
        tuple(j.ucm_block_id for j in metadata.requests["r"].save_jobs)
    ) == (False, False)
    worker.clear_connector_metadata()
    worker.shutdown()
    scheduler.shutdown()


def test_pp1_avoids_collective_and_registration_errors_are_exchanged():
    local = (0, {}, None)
    with patch(
        "torch.distributed.all_gather_object",
        side_effect=AssertionError("no collective"),
    ):
        assert exchange_pp_metadata(local, ParallelTopology()) == (local,)
    with pytest.raises(RuntimeError, match="registration failed"):
        exchange_pp_metadata((0, {}, "invalid native view"), ParallelTopology())


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "rank", "fingerprint"])
def test_handshake_rejects_missing_duplicate_or_incompatible_workers(
    tmp_path, corruption
):
    scheduler, _, workers, _ = system(tmp_path)
    declarations = {
        (w._parallel_rank.pp_rank, w._parallel_rank.tp_rank): w.get_handshake_metadata()
        for w, _, _ in workers
    }
    key = next(iter(declarations))
    value = declarations[key]
    if corruption == "missing":
        declarations.pop(key)
    elif corruption == "duplicate":
        declarations[(9, 9)] = value
    elif corruption == "rank":
        declarations[key] = replace(value, rank=ParallelRank(99))
    else:
        declarations[key] = replace(value, fingerprint="other")
    try:
        with pytest.raises(ValueError):
            scheduler.set_xfer_handshake_metadata(declarations)
    finally:
        for worker, _, _ in workers:
            worker.shutdown()
        scheduler.shutdown()


def test_real_gloo_pp_exchange_tp_groups_dp_isolation_empty_stage_and_error(tmp_path):
    script = Path(__file__).with_name("_ucm_pp_gloo_worker.py")
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "GLOO_PP_PASS" in result.stdout


@pytest.mark.parametrize("preempt", [False, True])
def test_finished_or_preempted_request_state_does_not_drop_late_save_results(
    tmp_path, preempt
):
    scheduler, store, workers, metadata = system(tmp_path, pp=1, tp=1)
    request_id = "r"
    output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(
            req_ids=(), get_block_ids=lambda _: (), boundary_state_offloads={}
        ),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        preempted_req_ids={request_id} if preempt else set(),
        finished_req_ids=set() if preempt else {request_id},
    )
    scheduler.build_connector_meta(output)
    assert scheduler._pending_saves
    worker, _, _ = workers[0]
    worker.bind_connector_metadata(metadata)
    worker.wait_for_save()
    report(scheduler, worker.build_connector_worker_meta())
    assert not scheduler._pending_saves
    assert store.lookup(
        tuple(j.ucm_block_id for j in metadata.requests["r"].save_jobs)
    ) == ((False, False) if preempt else (True, True))
    worker.clear_connector_metadata()
    worker.shutdown()
    scheduler.shutdown()


@pytest.mark.parametrize("operation", ["load", "dump"])
def test_bulk_unknown_completion_retains_source_descriptors(tmp_path, operation):
    scheduler, _, workers, metadata = system(tmp_path, pp=1, tp=1)
    worker, _, _ = workers[0]
    if operation == "load":
        request = metadata.requests["r"]
        metadata = replace(
            metadata,
            requests={
                "r": replace(
                    request, load_plans=request.dump_plans, dump_plans=(), save_jobs=()
                )
            },
        )
        # Avoid a missing file IO error; this test targets an unknown wait state.
    worker.bind_connector_metadata(metadata)
    with (
        patch.object(
            worker._proxy,
            "enqueue",
            return_value=SimpleNamespace(transfer="retained", handle=None),
        ),
        patch.object(
            worker._proxy, "wait", side_effect=RuntimeError("still owns pointers")
        ),
    ):
        with pytest.raises(RuntimeError, match="still owns pointers"):
            (
                worker.start_load_kv(None)
                if operation == "load"
                else worker.wait_for_save()
            )
        assert worker._layer_load_tasks if operation == "load" else worker._dump_tasks
        with pytest.raises(RuntimeError, match="own KV pointers"):
            worker.clear_connector_metadata()
    with patch.object(worker._proxy, "wait"):
        worker.shutdown()
    scheduler.shutdown()


@pytest.mark.parametrize("kind", ["file", "memory_file", "memory"])
def test_commit_boolean_visibility_cleanup_and_attempt_isolation(tmp_path, kind):
    memory = ByteMemory()
    access = MemoryByteAccess(memory)
    disk = FileTestStore(tmp_path, access)
    store = (
        disk
        if kind == "file"
        else MemoryTestStore(access, backend=disk if kind == "memory_file" else None)
    )
    adapter = UCMProxyAdapter(PipelineProxy(store))
    from ucm.integration.vllm.v2.ucm_proxy import UCMProxyTransfer

    target = b"k" * 16

    def dump(attempt, payload):
        memory.write(100, payload)
        adapter.submit(
            "dump",
            UCMProxyTransfer(
                (attempt,),
                np.array([0], dtype=np.uint64),
                np.array([[100]], dtype=np.uint64),
                np.array([[8]], dtype=np.uint64),
                np.array([[0]], dtype=np.uint64),
                ('{"shared_attempt":true}',),
                np.array([8], dtype=np.uint64),
            ),
        )

    original = b"a" * 16
    failed = b"b" * 16
    other = b"c" * 16
    indices = np.array([0], dtype=np.uint64)
    dump(original, b"abcdefgh")
    assert store.lookup((target,)) == (False,)
    adapter.commit(True, (original,), (target,), indices)
    assert store.lookup((target,)) == (True,)
    dump(failed, b"XXXXXXXX")
    dump(other, b"YYYYYYYY")
    adapter.commit(False, (failed,), (target,), indices)
    adapter.commit(False, (failed,), (target,), indices)  # Idempotent discard.
    assert store.read_record(target)[0] == b"abcdefgh"
    assert store.lookup((failed, other)) == (False, False)
    if kind == "memory":
        assert failed not in store._pending and other in store._pending
    else:
        assert (
            not disk._attempt_path(failed).exists()
            and disk._attempt_path(other).exists()
        )
    adapter.commit(False, (other,), (target,), indices)
    adapter.close()


def test_scheduler_retries_failed_true_commit_with_false_cleanup(tmp_path):
    scheduler, store, workers, metadata = system(tmp_path, pp=1, tp=1)
    worker, _, _ = workers[0]
    worker.bind_connector_metadata(metadata)
    worker.wait_for_save()
    report(scheduler, worker.build_connector_worker_meta())
    original = store.commit
    decisions = []

    def commit(success, staging_ids, object_ids):
        decisions.append(success)
        if success:
            raise OSError("publish failed")
        return original(False, staging_ids, object_ids)

    with patch.object(store, "commit", side_effect=commit):
        scheduler.request_finished("r")
    assert decisions == [True, False, True, False]
    assert not list(tmp_path.glob(".*.attempt"))
    assert store.lookup(
        tuple(j.ucm_block_id for j in metadata.requests["r"].save_jobs)
    ) == (False, False)
    worker.clear_connector_metadata()
    worker.shutdown()
    scheduler.shutdown()


@pytest.mark.parametrize("dcp,distinct,replicas", [(1, 1, 8), (2, 2, 4), (8, 8, 1)])
def test_mla_tp8_enumerates_only_real_data_shards(dcp, distinct, replicas):
    owners = shard_replicas(ParallelTopology(tp_size=8, dcp_size=dcp), 1, 1, True)
    assert tuple(owners) == tuple(UCMShard(1, 0, i) for i in range(distinct))
    assert all(len(ranks) == replicas for ranks in owners.values())
    assert {rank for ranks in owners.values() for rank in ranks} == {
        ParallelRank(tp, 1, 0) for tp in range(8)
    }


@pytest.mark.parametrize("chunk", [1024, 2048])
@pytest.mark.parametrize("interleave", [1, 128])
@pytest.mark.parametrize("layerwise", [False, True])
def test_dcp8_compiles_one_or_two_local_pages_and_global_block_major_offsets(
    chunk, interleave, layerwise
):
    from test_ucm_connector_v2 import MLAAttentionSpec

    from ucm.integration.vllm.v2.connector_worker import UCMTransferBuilder

    topology = ParallelTopology(
        tp_size=8, pp_size=2, dcp_size=8, cp_kv_cache_interleave_size=interleave
    )
    layouts = []
    for stage, widths in enumerate(((2, 3), (4,))):
        names = tuple(f"model.layers.{stage * 3 + i}.attn" for i in range(len(widths)))
        native = config(group(names, MLAAttentionSpec(128)))
        spec = parse_spec(
            native,
            scheduler_block_size=128,
            device_type="cpu",
            chunk_size=chunk,
            dcp_size=topology.dcp_size,
        )
        assert spec.groups[0].block_size == 128
        assert spec.groups[0].block_token_span == 1024
        assert spec.blocks_per_chunk == (chunk // 1024,)
        caches = {
            name: FakeTensor(
                0x1000 + i * 0x1000, (8, 128, width), (128 * width, width, 1)
            )
            for i, (name, width) in enumerate(zip(names, widths, strict=True))
        }
        layouts.append(
            UCMKVCacheLayout(
                spec,
                caches,
                kv_cache_config=native,
                num_hidden_layers=8,
                use_layerwise=layerwise,
            )
        )
    templates = tuple(
        (pp, layout.export_layout_metadata(topology.tp_size), None)
        for pp, layout in enumerate(layouts)
    )
    count = chunk // 1024
    plan = UCMGroupDispatchPlan(
        RestorePolicy.FA,
        (b"c" * 15,),
        np.array([3 * chunk], dtype=np.uint64),
        (np.array([2, 5][:count], dtype=np.uint64),),
        (0,),
    )
    metadata = UCMConnectorMetadata(
        requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
    )
    # All ranks share the same file template, but each supplies N local pages.
    for stage, (layout, widths) in enumerate(zip(layouts, ((2, 3), (4,)), strict=True)):
        for tp in range(8):
            rank = ParallelRank(tp, stage, 0)
            layout.build_store_layouts(topology, rank, templates)
            builder = UCMTransferBuilder(layout, rank)
            store = layout.store_layouts[0]
            assert store.whole_block_nbytes == 9216  # 8 * (128*(2+3) + 128*4)
            assert store.whole_chunk_nbytes == count * 9216
            assert int(store.chunk_segment_bytes[0].sum()) == 128 * sum(widths)
            for name in layout.layer_id_by_name if layerwise else (None,):
                (transfer,) = builder.build_dump_transfers(metadata, layer_name=name)
                selected = [
                    i
                    for i, layer_name in enumerate(layout.layer_id_by_name)
                    if name is None or name == layer_name
                ]
                expected_offsets, expected_ptrs, expected_sizes = [], [], []
                for block_index, block_id in enumerate([2, 5][:count]):
                    for i in selected:
                        prefix = sum(widths[:i]) * 128
                        shard_base = tp * sum(widths) * 128 + (5120 if stage else 0)
                        expected_offsets.append(
                            block_index * 9216 + shard_base + prefix
                        )
                        expected_ptrs.append(
                            0x1000 + i * 0x1000 + block_id * 128 * widths[i]
                        )
                        expected_sizes.append(128 * widths[i])
                assert transfer.ucm_chunk_offsets.tolist() == [expected_offsets]
                assert transfer.ptrs.tolist() == [expected_ptrs]
                assert transfer.sizes.tolist() == [expected_sizes]
                assert transfer.record_nbytes.tolist() == [count * 9216]


def test_native_groups_expand_independently_and_state_keeps_one_snapshot():
    from test_ucm_connector_v2 import MambaSpec, MLAAttentionSpec

    from ucm.integration.vllm.v2.connector_worker import UCMTransferBuilder

    topology = ParallelTopology(tp_size=2, dcp_size=2)
    names = ("model.layers.0.attn", "model.layers.1.attn", "model.layers.2.mixer")
    native = config(
        group([names[0]], MLAAttentionSpec(4)),
        group([names[1]], FullAttentionSpec(8)),
        group([names[2]], MambaSpec(4, tp_replicated=True)),
    )
    spec = parse_spec(
        native,
        scheduler_block_size=8,
        device_type="cpu",
        chunk_size=16,
        dcp_size=topology.dcp_size,
    )
    assert [g.block_size for g in spec.groups] == [4, 8, 4]
    assert [g.block_token_span for g in spec.groups] == [8, 16, 4]
    assert spec.blocks_per_chunk == (2, 1, 1)
    caches = {
        names[0]: FakeTensor(1000, (8, 4, 2), (8, 2, 1)),
        names[1]: FakeTensor(2000, (8, 8, 3), (24, 3, 1)),
        names[2]: FakeTensor(3000, (8, 1, 1, 8), (8, 8, 8, 1)),
    }
    layout = UCMKVCacheLayout(
        spec,
        caches,
        kv_cache_config=native,
        num_hidden_layers=4,
    )
    templates = ((0, layout.export_layout_metadata(topology.tp_size), None),)
    layout.build_store_layouts(topology, ParallelRank(), templates)
    builder = UCMTransferBuilder(layout)
    assert {
        policy: tuple(store.group_id for store in stores)
        for policy, stores in builder.groups_by_policy.items()
    } == {RestorePolicy.FA: (0, 1), RestorePolicy.STATE: (2,)}
    assert builder.group_byte_offsets == {0: 0, 1: 32, 2: 0}
    assert builder.record_nbytes == {RestorePolicy.FA: 80, RestorePolicy.STATE: 8}
    plans = (
        UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (b"a" * 15,),
            np.array([0], dtype=np.uint64),
            (np.array([2, 5], dtype=np.uint64), np.array([3], dtype=np.uint64)),
            (0, 1),
        ),
        UCMGroupDispatchPlan(
            RestorePolicy.STATE,
            (b"a" * 15,),
            np.array([0], dtype=np.uint64),
            (np.array([6], dtype=np.uint64),),
            (2,),
        ),
    )
    transfers = UCMTransferBuilder(layout).build_dump_transfers(
        UCMConnectorMetadata(requests={"r": RequestDispatchMeta("r", dump_plans=plans)})
    )
    assert [
        (t.ucm_chunk_offsets.tolist(), t.ptrs.tolist(), t.sizes.tolist())
        for t in transfers
    ] == [([[0, 16, 32]], [[1016, 1040, 2072]], [[8, 8, 24]]), ([[0]], [[3048]], [[8]])]


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_every_rank_must_ack_even_without_local_io(tmp_path, empty, failed):
    # MLA TP replicas and a stage with no local HBM both produce no IO, but
    # are still members of the terminal rank barrier before commit/cleanup.
    from dataclasses import fields

    from test_ucm_connector_v2 import MLAAttentionSpec

    from ucm.integration.vllm.v2.connector_scheduler import SaveJob
    from ucm.integration.vllm.v2.runtime import UCMStorageHandshake

    scheduler, store, workers, metadata = system(
        tmp_path,
        tp=2 if empty else 4,
        pp=2 if empty else 1,
        native_spec=MLAAttentionSpec(4),
        empty_stage=empty,
    )
    jobs = metadata.requests["r"].save_jobs
    assert {f.name for f in fields(SaveJob)} == {
        "request_id",
        "attempt_id",
        "ucm_block_id",
        "restore_policy",
        "chunk_index",
    }
    assert {f.name for f in fields(SaveResult)} == {"attempt_id", "rank", "success"}
    assert {f.name for f in fields(UCMStorageHandshake)} == {
        "rank",
        "fingerprint",
    }
    assert not hasattr(scheduler, "storage_layout")
    scheduler.request_finished("r")
    try:
        for index, (worker, _, _) in enumerate(workers):
            worker.bind_connector_metadata(metadata)
            if empty and worker._parallel_rank.pp_rank == 1:
                assert worker.layout.group_layouts == {}
            with patch.object(
                worker._proxy, "enqueue", wraps=worker._proxy.enqueue
            ) as enqueue:
                worker.wait_for_save()
            result = worker.build_connector_worker_meta()
            assert len(result.save_results) == len(jobs)
            assert {r.rank for r in result.save_results} == {worker._parallel_rank}
            if (empty and worker._parallel_rank.pp_rank == 1) or (
                not empty and worker._parallel_rank.tp_rank >= 2
            ):
                assert enqueue.call_count == 0
            if failed and index == 0:
                result.save_results = {
                    replace(r, success=False) for r in result.save_results
                }
                # A later duplicate True may never undo a reported False.
                report(scheduler, result)
                result.save_results = {
                    replace(r, success=True) for r in result.save_results
                }
            report(scheduler, result)
            report(scheduler, result)
            if index < len(workers) - 1:
                assert len(scheduler._pending_saves) == len(jobs)
                assert store._attempt_path(jobs[0].attempt_id).exists()
                assert store.lookup(tuple(j.ucm_block_id for j in jobs)) == (
                    False,
                    False,
                )
            worker.clear_connector_metadata()
        assert not scheduler._pending_saves
        assert store.lookup(tuple(j.ucm_block_id for j in jobs)) == (
            (False, False) if failed else (True, True)
        )
        assert not list(tmp_path.glob(".*.attempt"))
    finally:
        for worker, _, _ in workers:
            worker.shutdown()
        scheduler.shutdown()


def test_worker_layout_metadata_and_handshake_do_not_hash_hbm_allocations():
    topology = ParallelTopology(tp_size=2)

    def layout(blocks, base, width):
        native = config(
            group(["model.layers.0.attn"], FullAttentionSpec(4)), num_blocks=blocks
        )
        spec = parse_spec(native, scheduler_block_size=4, device_type="cpu")
        result = UCMKVCacheLayout(
            spec,
            {
                "model.layers.0.attn": FakeTensor(
                    base, (blocks, 4, width), (4 * width, width, 1)
                )
            },
            kv_cache_config=native,
            num_hidden_layers=4,
        )
        result.build_store_layouts(
            topology,
            ParallelRank(),
            ((0, result.export_layout_metadata(topology.tp_size), None),),
        )
        return result

    first, different_allocation = layout(8, 1000, 2), layout(16, 9000, 2)
    assert first.export_layout_metadata(
        2
    ) == different_allocation.export_layout_metadata(2)
    workers = {
        tp: UCMConnectorWorker.get_handshake_metadata(
            SimpleNamespace(
                spec=value.spec,
                layout=value,
                _parallel_rank=ParallelRank(tp),
                fingerprint="allocation-test",
                transfer_builder=object(),
            )
        )
        for tp, value in enumerate((first, different_allocation))
    }
    assert workers[0].fingerprint == workers[1].fingerprint == "allocation-test"
    assert {worker.rank for worker in workers.values()} == {
        ParallelRank(0),
        ParallelRank(1),
    }
    assert not hasattr(first, "format_digest")
    assert not hasattr(first.export_layout_metadata(2)[0], "format_digest")


@pytest.mark.parametrize("interleave", [1, 128])
def test_dcp8_never_accepts_a_128_token_chunk(interleave):
    from test_ucm_connector_v2 import MLAAttentionSpec

    topology = ParallelTopology(
        tp_size=8, dcp_size=8, cp_kv_cache_interleave_size=interleave
    )
    native = config(group(["model.layers.0.attn"], MLAAttentionSpec(128)))
    with pytest.raises(ValueError, match="whole distributed KV pages"):
        parse_spec(
            native,
            scheduler_block_size=128,
            device_type="cpu",
            chunk_size=128,
            dcp_size=topology.dcp_size,
        )


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize("layerwise", [False, True])
def test_whole_group_writer_and_roundtrip_accept_mixed_tp_redundancy(
    tmp_path, mixed, pp, layerwise
):
    from test_ucm_connector_v2 import MLAAttentionSpec

    def native(stage, names):
        # An unspecified TP limit resolves to TP size. PP1 stays replicated;
        # the complete group uses the LCM across both stages.
        return {
            name: (
                FullAttentionSpec(4)
                if mixed and stage == 0 and i == 1
                else MLAAttentionSpec(4)
            )
            for i, name in enumerate(names)
        }

    scheduler, store, workers, metadata = system(
        tmp_path, pp=pp, layerwise=layerwise, native_spec=native
    )
    keys = make_ucm_chunk_ids((b"a" * 15, b"b" * 15))
    expected_bytes = (20 + (16 if pp == 2 else 0)) * (2 if mixed else 1)
    scheduler.request_finished("r")
    try:
        for index, (worker, _, _) in enumerate(workers):
            layout = worker.layout.store_layouts[0]
            assert layout.tp_shards == (2 if mixed else 1)
            assert layout.replica_count == (1 if mixed else 2)
            assert layout.shard == UCMShard(
                worker._parallel_rank.pp_rank,
                worker._parallel_rank.tp_rank if mixed else 0,
                0,
            )
            assert layout.whole_chunk_nbytes == expected_bytes
            worker.bind_connector_metadata(metadata)
            batches = worker.transfer_builder.build_dump_transfers(metadata)
            assert len(batches) == 1
            # Every chosen row includes the complete local group, never just
            # the independently sharded member or a per-column writer subset.
            assert batches[0].ptrs.shape == (
                (2 if mixed else 1),
                len(worker.layout.group_layouts[0].payload_bytes),
            )
            worker.wait_for_save()
            report(scheduler, worker.build_connector_worker_meta())
            if index < len(workers) - 1:
                assert store.lookup(keys) == (False, False)
            worker.clear_connector_metadata()
        assert store.lookup(keys) == (True, True)
        for chunk, key in enumerate(keys):
            expected = b"".join(
                bytes((stage * 40 + tp * 10 + layer + chunk * 100,)) * width * 4
                for stage, widths in enumerate(((2, 3), (4,))[:pp])
                for tp in (range(2) if mixed else (chunk % 2,))
                for layer, width in enumerate(widths)
            )
            assert store.read_record(key)[0] == expected
            assert len(expected) == expected_bytes
        load = replace(
            metadata.requests["r"].dump_plans[0],
            vllm_block_ids_by_group=(np.array([5, 6], dtype=np.uint64),),
        )
        load_meta = UCMConnectorMetadata(
            requests={"target": RequestDispatchMeta("target", load_plans=(load,))}
        )
        for worker, memory, widths in workers:
            worker.bind_connector_metadata(load_meta)
            worker.start_load_kv(None)
            worker.wait_for_save()
            for layer, width in enumerate(widths):
                for chunk, block in enumerate((5, 6)):
                    rank = worker._parallel_rank
                    saved_tp = rank.tp_rank if mixed else chunk % 2
                    expected = (
                        bytes(
                            (rank.pp_rank * 40 + saved_tp * 10 + layer + chunk * 100,)
                        )
                        * width
                        * 4
                    )
                    assert (
                        memory.read(
                            0x1000 + layer * 0x1000 + block * width * 4, width * 4
                        )
                        == expected
                    )
            worker.clear_connector_metadata()
    finally:
        for worker, _, _ in workers:
            worker.shutdown()
        scheduler.shutdown()


@pytest.mark.parametrize("dcp", [1, 8])
def test_mla_tp8_default_chunk_maps_1024_tokens_to_expected_writers_and_ranges(dcp):
    from test_ucm_connector_v2 import MLAAttentionSpec

    from ucm.integration.vllm.v2.connector_scheduler import RequestState
    from ucm.integration.vllm.v2.connector_worker import UCMTransferBuilder

    name = "model.layers.0.attn"
    native = config(group([name], MLAAttentionSpec(128)), num_blocks=128)
    spec = parse_spec(
        native, scheduler_block_size=128 * dcp, dcp_size=dcp, device_type="cpu"
    )
    assert spec.chunk_size == 128 * dcp
    block_ids = (7, 19, 23, 31, 42, 58, 63, 80) if dcp == 1 else (7,)
    common_ids = tuple(bytes((i,)) * 15 for i in range(8 if dcp == 1 else 1))
    state = RequestState(num_token_ids=1024, common_ucm_chunk_ids=common_ids)
    request = make_connector_scheduler(spec)._build_request_plan(
        "r", state, (block_ids,), 1024
    )
    (plan,) = request.dump_plans
    assert plan.vllm_block_ids_by_group[0].tolist() == list(block_ids)
    assert plan.token_starts.tolist() == list(range(0, 1024, spec.chunk_size))
    keys = make_ucm_chunk_ids(common_ids)
    metadata = UCMConnectorMetadata(requests={"r": request})
    topology = ParallelTopology(tp_size=8, dcp_size=dcp)
    coverage = {key: [] for key in keys}
    page_bytes = 128 * 2
    for tp in range(8):
        base = 0x100000 * (tp + 1)
        layout = UCMKVCacheLayout(
            spec,
            {name: FakeTensor(base, (128, 128, 2), (256, 2, 1))},
            kv_cache_config=native,
            num_hidden_layers=1,
        )
        pp_metadata = exchange_pp_metadata(
            (0, layout.export_layout_metadata(topology.tp_size), None), topology
        )
        rank = ParallelRank(tp)
        layout.build_store_layouts(topology, rank, pp_metadata)
        builder = UCMTransferBuilder(layout, rank)
        (transfer,) = builder.build_dump_transfers(metadata)
        chunk_index = tp if dcp == 1 else 0
        offset = 0 if dcp == 1 else tp * page_bytes
        block_id = block_ids[tp] if dcp == 1 else block_ids[0]
        assert transfer.ucm_chunk_ids == (keys[chunk_index],)
        assert transfer.chunk_indices.tolist() == [chunk_index]
        assert transfer.ptrs.tolist() == [[base + block_id * page_bytes]]
        assert transfer.sizes.tolist() == [[page_bytes]]
        assert transfer.ucm_chunk_offsets.tolist() == [[offset]]
        assert transfer.record_nbytes.tolist() == [page_bytes * dcp]
        coverage[keys[chunk_index]].append((tp, offset, page_bytes))

        load_request = replace(request, load_plans=(plan,), dump_plans=())
        (load,) = builder.build_load_transfers(
            UCMConnectorMetadata(requests={"r": load_request})
        )
        assert load.ucm_chunk_ids == keys
        assert load.ptrs.tolist() == [[base + b * page_bytes] for b in block_ids]
        assert load.ucm_chunk_offsets.tolist() == [[offset] for _ in keys]
    if dcp == 1:
        assert list(coverage.values()) == [[(tp, 0, page_bytes)] for tp in range(8)]
    else:
        assert list(coverage.values()) == [
            [(tp, tp * page_bytes, page_bytes) for tp in range(8)]
        ]


@pytest.mark.parametrize("pp", [1, 2])
def test_group_tp_lcm_preserves_six_shards_and_two_replicas_on_tp12(tmp_path, pp):
    def native(stage, names):
        specs = {}
        for index, name in enumerate(names):
            spec = FullAttentionSpec(4)
            spec.max_tp_shards = (
                (2 if index == 0 else 3) if pp == 1 else (2 if stage == 0 else 3)
            )
            specs[name] = spec
        return specs

    scheduler, store, workers, metadata = system(
        tmp_path, tp=12, pp=pp, native_spec=native
    )
    keys = make_ucm_chunk_ids((b"a" * 15, b"b" * 15))
    scheduler.request_finished("r")
    try:
        for worker, _, _ in workers:
            store_layout = worker.layout.store_layouts[0]
            assert store_layout.tp_shards == 6
            assert store_layout.replica_count == 2
            assert store_layout.shard.tp_shard == worker._parallel_rank.tp_rank // 2
            worker.bind_connector_metadata(metadata)
            worker.wait_for_save()
            report(scheduler, worker.build_connector_worker_meta())
            worker.clear_connector_metadata()
        for chunk, key in enumerate(keys):
            expected = b"".join(
                bytes(
                    (stage * 40 + (2 * shard + chunk % 2) * 10 + layer + chunk * 100,)
                )
                * width
                * 4
                for stage, widths in enumerate(((2, 3), (4,))[:pp])
                for shard in range(6)
                for layer, width in enumerate(widths)
            )
            assert store.read_record(key)[0] == expected
            assert len(expected) == 6 * (20 + (16 if pp == 2 else 0))
        plan = replace(
            metadata.requests["r"].dump_plans[0],
            vllm_block_ids_by_group=(np.array([5, 6], dtype=np.uint64),),
        )
        load_metadata = UCMConnectorMetadata(
            requests={"target": RequestDispatchMeta("target", load_plans=(plan,))}
        )
        for worker, memory, widths in workers:
            worker.bind_connector_metadata(load_metadata)
            worker.start_load_kv(None)
            worker.wait_for_save()
            for chunk, block_id in enumerate((5, 6)):
                writer_tp_rank = (worker._parallel_rank.tp_rank // 2) * 2 + chunk % 2
                for layer, width in enumerate(widths):
                    expected = (
                        bytes(
                            (
                                worker._parallel_rank.pp_rank * 40
                                + writer_tp_rank * 10
                                + layer
                                + chunk * 100,
                            )
                        )
                        * width
                        * 4
                    )
                    assert (
                        memory.read(
                            0x1000 + layer * 0x1000 + block_id * width * 4, width * 4
                        )
                        == expected
                    )
            worker.clear_connector_metadata()
    finally:
        for worker, _, _ in workers:
            worker.shutdown()
        scheduler.shutdown()


def test_dcp8_attention_and_replicated_sliding_group_have_independent_block_ids():
    from test_ucm_connector_v2 import AscendSlidingWindowMLASpec, MLAAttentionSpec

    from ucm.integration.vllm.v2.connector_scheduler import RequestState
    from ucm.integration.vllm.v2.connector_worker import UCMTransferBuilder

    names = ("model.layers.0.attn", "model.layers.1.attn")
    native = config(
        group([names[0]], MLAAttentionSpec(128)),
        group([names[1]], AscendSlidingWindowMLASpec(128, 1, 256)),
        num_blocks=128,
    )
    spec = parse_spec(native, scheduler_block_size=1024, dcp_size=8, device_type="cpu")
    assert spec.chunk_size == 1024
    assert spec.blocks_per_chunk == (1, 2)
    assert [g.block_token_span for g in spec.groups] == [1024, 128]
    state = RequestState(num_token_ids=1024, common_ucm_chunk_ids=(b"a" * 15,))
    request = make_connector_scheduler(spec)._build_request_plan(
        "r", state, ((7,), (7, 19, 23, 31, 42, 58, 63, 80)), 1024
    )
    assert [
        plan.vllm_block_ids_by_group[0].tolist() for plan in request.dump_plans
    ] == [[7], [63, 80]]
    metadata = UCMConnectorMetadata(requests={"r": request})
    topology = ParallelTopology(tp_size=8, dcp_size=8)
    for tp_rank in range(8):
        layout = UCMKVCacheLayout(
            spec,
            {
                name: FakeTensor(0x100000 * (index + 1), (128, 128, 2), (256, 2, 1))
                for index, name in enumerate(names)
            },
            kv_cache_config=native,
            num_hidden_layers=2,
        )
        layout.build_store_layouts(
            topology,
            ParallelRank(tp_rank),
            ((0, layout.export_layout_metadata(8), None),),
        )
        builder = UCMTransferBuilder(layout, ParallelRank(tp_rank))
        assert builder.record_nbytes == {RestorePolicy.FA: 2048, RestorePolicy.WA: 512}
        transfers = builder.build_dump_transfers(metadata)
        assert len(transfers) == (2 if tp_rank == 0 else 1)
        assert transfers[0].ptrs.tolist() == [[0x100000 + 7 * 256]]
        assert transfers[0].ucm_chunk_offsets.tolist() == [[tp_rank * 256]]
        if tp_rank == 0:
            assert transfers[1].ptrs.tolist() == [
                [0x200000 + 63 * 256, 0x200000 + 80 * 256]
            ]
            assert transfers[1].ucm_chunk_offsets.tolist() == [[0, 256]]
            assert transfers[1].sizes.tolist() == [[256, 256]]


def test_state_policy_object_preserves_distinct_gdn_and_ple_tp_shards():
    from test_ucm_connector_v2 import MambaSpec

    from ucm.integration.vllm.v2.connector_worker import UCMTransferBuilder

    names = ("model.layers.0.mixer", "model.layers.1.ple")
    native = config(
        group([names[0]], MambaSpec(4)),
        group([names[1]], MambaSpec(4, tp_replicated=True)),
    )
    spec = parse_spec(native, scheduler_block_size=4, device_type="cpu")
    plan = UCMGroupDispatchPlan(
        RestorePolicy.STATE,
        (b"s" * 15,),
        np.array([0], dtype=np.uint64),
        (np.array([1], dtype=np.uint64), np.array([2], dtype=np.uint64)),
        (0, 1),
    )
    topology = ParallelTopology(tp_size=2)
    for tp_rank in range(2):
        layout = UCMKVCacheLayout(
            spec,
            {
                names[0]: FakeTensor(0x1000, (8, 1, 1, 8), (8, 8, 8, 1)),
                names[1]: FakeTensor(0x2000, (8, 1, 1, 8), (8, 8, 8, 1)),
            },
            kv_cache_config=native,
            num_hidden_layers=2,
        )
        rank = ParallelRank(tp_rank)
        layout.build_store_layouts(
            topology, rank, ((0, layout.export_layout_metadata(2), None),)
        )
        assert [layout.store_layouts[gid].tp_shards for gid in (0, 1)] == [2, 1]
        builder = UCMTransferBuilder(layout, rank)
        assert builder.record_nbytes == {RestorePolicy.STATE: 24}
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
        )
        (dump,) = builder.build_dump_transfers(metadata)
        assert dump.ucm_chunk_ids == (b"s" * 15 + b"\x02",)
        assert dump.ucm_chunk_offsets.tolist() == ([[0, 16]] if tp_rank == 0 else [[8]])
        assert dump.sizes.tolist() == ([[8, 8]] if tp_rank == 0 else [[8]])
        assert dump.record_nbytes.tolist() == [24]
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        (load,) = builder.build_load_transfers(metadata)
        assert load.ucm_chunk_offsets.tolist() == [[tp_rank * 8, 16]]
        assert load.sizes.tolist() == [[8, 8]]
