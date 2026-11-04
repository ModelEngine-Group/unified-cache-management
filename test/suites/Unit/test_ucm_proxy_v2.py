"""Store composition, positional routing and connector-v2 integration tests."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from test_ucm_connector_v2 import (
    ByteMemory,
    FakeRequest,
    FakeTensor,
    FullAttentionSpec,
    KVCacheSpecKind,
    KVConnectorRole,
    MemoryByteAccess,
    bind_worker,
    config,
    finish_worker_save,
    group,
    make_cache_layout,
    make_connector_scheduler,
    parse_spec,
    plan_token_starts,
    request_ucm_ids,
    seed_request_state,
    ucm_ids,
    vllm_config,
)

from ucm.integration.vllm.v2.connector_scheduler import (
    RequestDispatchMeta,
    UCMConnectorMetadata,
    UCMGroupDispatchPlan,
)
from ucm.integration.vllm.v2.connector_worker import (
    UCMConnectorWorker,
    UCMTransferBuilder,
)
from ucm.integration.vllm.v2.factory import create_proxy
from ucm.integration.vllm.v2.layout import RestorePolicy  # noqa: E402
from ucm.integration.vllm.v2.layout import (
    UCMKVCacheGroupSpec,
    UCMKVCacheSpec,
)
from ucm.integration.vllm.v2.runtime import (
    ParallelTopology,
    make_ucm_chunk_ids,
)
from ucm.integration.vllm.v2.test_stores import FileTestStore, MemoryTestStore
from ucm.integration.vllm.v2.ucm_connector import UCMConnector
from ucm.integration.vllm.v2.ucm_proxy import (
    ParallelProxy,
    PipelineProxy,
    UCMProxyAdapter,
    UCMProxyError,
    UCMProxyTransfer,
)


def test_scheduler_can_repeat_planning_without_mutating_request_state():
    from dataclasses import asdict

    spec = parse_spec(
        config(group(["model.layers.0.attn"], FullAttentionSpec(4))),
        scheduler_block_size=4,
    )
    scheduler = make_connector_scheduler(spec)
    state = seed_request_state(
        scheduler, FakeRequest("r", 12), 4, (b"a" * 15, b"b" * 15, b"c" * 15)
    )
    block_tables = ([1, 2, 3],)
    before = asdict(state)
    first = scheduler._build_request_plan("r", state, block_tables, 4)
    second = scheduler._build_request_plan("r", state, block_tables, 4)

    assert asdict(state) == before
    assert state.need_load and state.token_processed == 4
    assert block_tables == ([1, 2, 3],)
    assert first.load_plans and first.dump_plans
    for actual, repeated in zip(
        first.load_plans + first.dump_plans,
        second.load_plans + second.dump_plans,
        strict=True,
    ):
        assert actual.common_ucm_chunk_ids == repeated.common_ucm_chunk_ids
        np.testing.assert_array_equal(actual.token_starts, repeated.token_starts)
        for blocks, repeated_blocks in zip(
            actual.vllm_block_ids_by_group,
            repeated.vllm_block_ids_by_group,
            strict=True,
        ):
            np.testing.assert_array_equal(blocks, repeated_blocks)


def test_scheduler_does_not_consume_load_or_progress_when_any_plan_fails():
    from dataclasses import asdict

    spec = parse_spec(
        config(group(["model.layers.0.attn"], FullAttentionSpec(4))),
        scheduler_block_size=4,
    )
    scheduler = make_connector_scheduler(spec)
    for request_id in ("first", "second"):
        seed_request_state(
            scheduler,
            FakeRequest(request_id, 12),
            4,
            (b"a" * 15, b"b" * 15, b"c" * 15),
        )
    before = {key: asdict(state) for key, state in scheduler.requests.items()}
    output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(
            req_ids=("first", "second"),
            get_block_ids=lambda request_id: ([4, 5, 6],),
            boundary_state_offloads={},
        ),
        num_scheduled_tokens={"first": 4, "second": 4},
        total_num_scheduled_tokens=8,
        preempted_req_ids=set(),
        finished_req_ids=set(),
    )
    build = scheduler._build_request_plan

    def fail_second(request_id, *args):
        if request_id == "second":
            raise ValueError("invalid second plan")
        return build(request_id, *args)

    with patch.object(scheduler, "_build_request_plan", side_effect=fail_second):
        with pytest.raises(ValueError, match="invalid second plan"):
            scheduler.build_connector_meta(output)
    assert {key: asdict(state) for key, state in scheduler.requests.items()} == before

    with patch.object(
        scheduler,
        "_lookup_hit_tokens",
        side_effect=AssertionError("lookup during planning"),
    ):
        metadata = scheduler.build_connector_meta(output)
    assert set(metadata.requests) == {"first", "second"}
    for request_id, state in scheduler.requests.items():
        assert metadata.requests[request_id].load_plans
        assert not state.need_load
        assert state.token_processed == 8
        plan = metadata.requests[request_id].dump_plans[0]
        assert plan.vllm_block_ids_by_group[0].tolist() == [5]


def test_scheduler_owns_load_failure_cleanup():
    spec = parse_spec(
        config(group(["model.layers.0.attn"], FullAttentionSpec(4))),
        scheduler_block_size=4,
    )
    scheduler = make_connector_scheduler(spec)
    for request_id in ("failed", "other"):
        seed_request_state(scheduler, FakeRequest(request_id, 8))
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=SimpleNamespace(
                load_failed_reqs={"failed", "absent"}, save_results=set()
            )
        )
    )
    assert set(scheduler.requests) == {"other"}


def test_context_unifies_rank_identity_without_raw_configs_or_fingerprint_pollution():
    from ucm.integration.vllm.v2.runtime import ParallelRank

    cfg = vllm_config()
    native = config(group(["model.layers.0.attn"], FullAttentionSpec(4)))
    workers = []
    for global_rank, engine_id in ((0, "engine-a"), (1, "engine-b")):
        cfg.instance_id = engine_id
        adapter = UCMProxyAdapter(PipelineProxy(MemoryTestStore()))
        with (
            patch(
                "ucm.integration.vllm.v2.runtime._worker_rank", return_value=global_rank
            ),
            patch(
                "ucm.integration.vllm.v2.runtime._worker_parallel_ranks",
                return_value=ParallelRank(global_rank, 0, 0),
            ) as ranks,
            patch(
                "ucm.integration.vllm.v2.connector_worker.create_proxy",
                return_value=adapter,
            ),
        ):
            worker = UCMConnector(cfg, KVConnectorRole.WORKER, native)
        ranks.assert_called_once()
        context = worker.context
        assert context.global_rank == global_rank
        assert context.rank.tp_rank == global_rank
        assert context.rank.pp_rank == context.rank.pcp_rank == 0
        assert context.parallel_topology.tp_size == 2
        assert worker.connector_worker._parallel_rank is context.rank
        assert worker.connector_worker._kv_cache_config is native
        for field in (
            "context",
            "runtime",
            "vllm_config",
            "kv_cache_config",
            "world_size",
        ):
            assert not hasattr(context, field)
        workers.append(worker)
    assert workers[0].context.fingerprint == workers[1].context.fingerprint
    for worker in workers:
        worker.shutdown()
    with patch(
        "ucm.integration.vllm.v2.runtime._worker_parallel_ranks",
        side_effect=AssertionError("scheduler has no worker rank"),
    ):
        scheduler = UCMConnector(cfg, KVConnectorRole.SCHEDULER, native)
    assert scheduler.context.rank is None
    assert scheduler.context.global_rank is None
    scheduler.shutdown()


@pytest.mark.parametrize(
    "change",
    [
        "model",
        "revision",
        "code_revision",
        "spec_method",
        "spec_tokens",
        "sfa_c8",
        "li_c8",
    ],
)
def test_runtime_fingerprint_isolates_model_speculation_and_sparse_identity(change):
    from ucm.integration.vllm.v2.runtime import UCMRuntimeContext

    baseline = vllm_config()
    changed = vllm_config()
    native = config(group(["model.layers.0.attn"], FullAttentionSpec(4)))
    if change in ("spec_method", "spec_tokens"):
        baseline.speculative_config = SimpleNamespace(
            method="mtp", num_speculative_tokens=1
        )
        changed.speculative_config = SimpleNamespace(
            method="eagle" if change == "spec_method" else "mtp",
            num_speculative_tokens=2 if change == "spec_tokens" else 1,
        )
    elif change in ("sfa_c8", "li_c8"):
        changed.additional_config[f"enable_sparse_{change}"] = True
    else:
        setattr(
            changed.model_config,
            change,
            "another-org/model" if change == "model" else "another-revision",
        )

    before = UCMRuntimeContext.from_vllm_config(
        baseline, KVConnectorRole.SCHEDULER, native
    )
    after = UCMRuntimeContext.from_vllm_config(
        changed, KVConnectorRole.SCHEDULER, native
    )
    # In the model case only the full reference differs: leaf name and HF
    # configuration are identical, so basename-only identity would be unsafe.
    assert before.fingerprint != after.fingerprint


def test_hasher_uses_scheduler_fingerprint_after_complete_handshake():
    from test_ucm_connector_v2 import ShardingHandshakeTest

    from ucm.integration.vllm.v2.connector_scheduler import UCMConnectorScheduler
    from ucm.integration.vllm.v2.runtime import UCMRuntimeContext

    spec, topology, workers = ShardingHandshakeTest()._workers()
    context = UCMRuntimeContext(
        role=KVConnectorRole.SCHEDULER,
        device_type="cpu",
        engine_id="first-engine",
        global_rank=None,
        rank=None,
        parallel_topology=topology,
        launch_config={},
        use_layerwise=False,
        spec=spec,
        fingerprint="test",
    )
    schedulers = []

    def make(context):
        with patch(
            "ucm.integration.vllm.v2.connector_scheduler.create_proxy",
            return_value=UCMProxyAdapter(PipelineProxy(MemoryTestStore())),
        ):
            scheduler = UCMConnectorScheduler(context)
        schedulers.append(scheduler)
        return scheduler

    try:
        first = make(context)
        assert first._chunk_hasher is None
        first.set_xfer_handshake_metadata(workers)
        original_hasher = first._chunk_hasher
        from ucm.integration.vllm.request_hasher import RequestHasher

        expected = RequestHasher(context.fingerprint).make_request_block_hasher(
            spec.chunk_size
        )
        assert original_hasher(FakeRequest("r", 12)) == expected(FakeRequest("r", 12))
        original_keys = first.lookup(FakeRequest("first", 12), 0).common_ucm_chunk_ids

        # Worker arrival order and engine instance ID do not define cache identity.
        same = make(replace(context, engine_id="another-engine"))
        reordered = dict(reversed(tuple(workers.items())))
        same.set_xfer_handshake_metadata(reordered)
        assert (
            same.lookup(FakeRequest("another-request", 12), 0).common_ucm_chunk_ids
            == original_keys
        )
        first.set_xfer_handshake_metadata(reordered)
        assert first._chunk_hasher is original_hasher
        assert "first" in first.requests

        changed_workers = {
            key: replace(worker, fingerprint="another-fingerprint")
            for key, worker in workers.items()
        }
        with pytest.raises(ValueError, match="fingerprint"):
            first.set_xfer_handshake_metadata(changed_workers)
        assert first._chunk_hasher is original_hasher
        changed = make(replace(context, fingerprint="another-fingerprint"))
        changed.set_xfer_handshake_metadata(changed_workers)
        assert (
            changed.lookup(FakeRequest("first", 12), 0).common_ucm_chunk_ids
            != original_keys
        )

        # Runtime identity must not replace request-specific extra semantics.
        left, right = FakeRequest("left", 12), FakeRequest("right", 12)
        left.cache_salt, right.cache_salt = "salt-left", "salt-right"
        with patch(
            "ucm.integration.vllm.request_hasher.generate_block_hash_extra_keys",
            side_effect=lambda request, start, end, mm_index: (
                (request.cache_salt,),
                mm_index,
            ),
        ):
            left_keys = first.lookup(left, 0).common_ucm_chunk_ids
            right_keys = first.lookup(right, 0).common_ucm_chunk_ids
        assert left_keys != right_keys
    finally:
        for scheduler in schedulers:
            scheduler.shutdown()


def test_restore_policy_is_derived_from_native_kinds_and_persistence():
    from test_ucm_connector_v2 import (
        AscendSlidingWindowMLASpec,
        MambaSpec,
        MLAAttentionSpec,
        UnsupportedSpec,
    )

    from ucm.integration.vllm.v2.layout.kv_cache import groups_by_restore_policy

    spec = parse_spec(
        config(
            group(["model.layers.0.attn"], FullAttentionSpec(4)),
            group(["model.layers.1.attn"], MLAAttentionSpec(4)),
            group(["model.layers.2.attn"], AscendSlidingWindowMLASpec(4, 1, 8)),
            group(["model.layers.3.mixer"], MambaSpec(4)),
            group(
                ["model.layers.4.scratch"], UnsupportedSpec(4, prefix_cacheable=False)
            ),
        ),
        scheduler_block_size=4,
        chunk_size=8,
    )
    assert [group.tail_tokens for group in spec.groups] == [None, None, 8, None, None]
    assert [group.restore_policy for group in spec.groups] == [
        RestorePolicy.FA,
        RestorePolicy.FA,
        RestorePolicy.WA,
        RestorePolicy.STATE,
        None,
    ]
    assert replace(spec.groups[0], is_persistent=False).restore_policy is None
    assert replace(spec.groups[2], tail_tokens=0).restore_policy is None
    assert [
        (policy, tuple(group.group_id for group in groups))
        for policy, groups in groups_by_restore_policy(spec).items()
    ] == [
        (RestorePolicy.FA, (0, 1)),
        (RestorePolicy.WA, (2,)),
        (RestorePolicy.STATE, (3,)),
    ]


@pytest.mark.parametrize("interleave", [1, 4])
@pytest.mark.parametrize("use_layerwise", [False, True])
@pytest.mark.parametrize("state_tp_replicated", [False, True])
def test_hybrid_dcp_full_pages_and_state_restore_then_save(
    interleave, use_layerwise, state_tp_replicated
):
    from test_ucm_connector_v2 import MambaSpec, MLAAttentionSpec, RecordingTestStore

    from ucm.integration.vllm.v2.runtime import ParallelRank

    class Store(RecordingTestStore):
        """Records descriptors and publication; never dereferences KV pointers."""

        def commit(self, success, staging_ids, object_ids):
            if success:
                self.present.update(object_ids)

        def close(self):
            pass

    cfg = vllm_config()
    cfg.parallel_config.decode_context_parallel_size = 2
    cfg.parallel_config.cp_kv_cache_interleave_size = interleave
    cfg.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = use_layerwise
    native = config(
        group(["model.layers.0.attn"], MLAAttentionSpec(4)),
        group(
            ["model.layers.1.mixer"], MambaSpec(8, tp_replicated=state_tp_replicated)
        ),
    )
    store = Store()
    with patch(
        "ucm.integration.vllm.v2.connector_scheduler.create_proxy",
        return_value=UCMProxyAdapter(PipelineProxy(store)),
    ):
        scheduler = UCMConnector(cfg, KVConnectorRole.SCHEDULER, native)
    workers, declarations = [], {}
    try:
        for tp_rank in range(2):
            with (
                patch(
                    "ucm.integration.vllm.v2.runtime._worker_rank", return_value=tp_rank
                ),
                patch(
                    "ucm.integration.vllm.v2.runtime._worker_parallel_ranks",
                    return_value=ParallelRank(tp_rank, 0, 0),
                ),
                patch(
                    "ucm.integration.vllm.v2.connector_worker.create_proxy",
                    return_value=UCMProxyAdapter(PipelineProxy(store)),
                ),
            ):
                worker = UCMConnector(cfg, KVConnectorRole.WORKER, native)
            workers.append(worker)
            worker.register_kv_caches(
                {
                    "model.layers.0.attn": FakeTensor(
                        0x1000 + tp_rank * 0x1000, (8, 4, 2), (8, 2, 1)
                    ),
                    "model.layers.1.mixer": FakeTensor(
                        0x5000 + tp_rank * 0x1000, (8, 1, 1, 8), (8, 8, 8, 1)
                    ),
                }
            )
            declarations[(0, tp_rank)] = worker.get_handshake_metadata()
        scheduler.set_xfer_handshake_metadata(declarations)
        role = scheduler.connector_scheduler
        assert [g.block_token_span for g in role.spec.groups] == [8, 8]
        assert role.spec.chunk_size == 8
        assert role.spec.blocks_per_chunk == (1, 1)

        request = FakeRequest("r", 24)
        initial = role.lookup(request, 0)
        assert initial.external_hit_tokens == 0
        common_ids = initial.common_ucm_chunk_ids
        fa_keys = make_ucm_chunk_ids(common_ids, RestorePolicy.FA)
        state_keys = make_ucm_chunk_ids(common_ids, RestorePolicy.STATE)
        assert list(role.groups_by_policy) == [RestorePolicy.FA, RestorePolicy.STATE]
        store.present.add(fa_keys[0])
        assert role.lookup(request, 0).external_hit_tokens == 0
        store.present.add(state_keys[0])
        assert role.lookup(request, 0).external_hit_tokens == 8

        output = SimpleNamespace(
            kv_connector_block_state=SimpleNamespace(
                req_ids=("r",),
                get_block_ids=lambda request_id: ([1, 2, 3], [4, 5, 0]),
                boundary_state_offloads={"r": ((1, 6, 16),)},
            ),
            num_scheduled_tokens={"r": 8},
            total_num_scheduled_tokens=8,
            preempted_req_ids=set(),
            finished_req_ids=set(),
        )
        metadata = scheduler.build_connector_meta(output)
        expected_loads, expected_dumps = [], []
        jobs = {job.restore_policy: job for job in metadata.requests["r"].save_jobs}
        for tp_rank, worker in enumerate(workers):
            state_offset = 0 if state_tp_replicated else tp_rank * 8
            expected_loads.extend(
                [
                    (fa_keys[0], tp_rank * 8, 0x1000 + tp_rank * 0x1000 + 8, 8),
                    (state_keys[0], state_offset, 0x5000 + tp_rank * 0x1000 + 4 * 8, 8),
                    (state_keys[0], state_offset, 0x5000 + tp_rank * 0x1000 + 5 * 8, 8),
                ]
            )
            expected_dumps.append(
                (
                    jobs[RestorePolicy.FA].attempt_id,
                    tp_rank * 8,
                    0x1000 + tp_rank * 0x1000 + 2 * 8,
                    8,
                )
            )
            if not state_tp_replicated or tp_rank == 1:
                expected_dumps.append(
                    (
                        jobs[RestorePolicy.STATE].attempt_id,
                        state_offset,
                        0x5000 + tp_rank * 0x1000 + 6 * 8,
                        8,
                    )
                )
            worker.bind_connector_metadata(metadata)
            worker.start_load_kv(None)
            for layer_name in ("model.layers.0.attn", "model.layers.1.mixer"):
                worker.wait_for_layer_load(layer_name)
            worker.wait_for_save()
            result = worker.connector_worker.build_connector_worker_meta()
            if result is not None:
                role.update_connector_output(
                    SimpleNamespace(kv_connector_worker_meta=result)
                )
            worker.clear_connector_metadata()
        assert role.lookup(FakeRequest("before-finish", 24), 0).external_hit_tokens == 8
        role.request_finished("r")

        def descriptors(calls):
            return sorted(
                (key, int(offset), int(ptr), int(size))
                for keys, offsets, ptrs, sizes, *_ in calls
                for row, key in enumerate(keys)
                for offset, ptr, size in zip(
                    offsets[row], ptrs[row], sizes[row], strict=True
                )
            )

        assert descriptors(store.load_calls) == sorted(expected_loads)
        assert descriptors(store.dump_calls) == sorted(expected_dumps)
        # After both FA shards and the exact State checkpoint are published,
        # the newly computed distributed page is externally recoverable.
        assert role.lookup(FakeRequest("again", 24), 0).external_hit_tokens == 16
    finally:
        for worker in workers:
            worker.shutdown()
        scheduler.shutdown()


def test_sliding_window_keeps_local_geometry_alongside_dcp_attention():
    from test_ucm_connector_v2 import AscendSlidingWindowMLASpec

    spec = parse_spec(
        config(
            group(["model.layers.0.attn"], FullAttentionSpec(4)),
            group(["model.layers.1.attn"], AscendSlidingWindowMLASpec(4, 1, 8)),
        ),
        scheduler_block_size=8,
        dcp_size=2,
    )
    assert [g.block_token_span for g in spec.groups] == [8, 4]
    assert [g.dcp_sharded for g in spec.groups] == [True, False]
    assert spec.blocks_per_chunk == (1, 2)


def transfer(
    keys,
    indices,
    ptrs,
    *,
    offsets=None,
    sizes=None,
    record_size=8,
    restore_policy=RestorePolicy.FA,
):
    ptrs = np.asarray(ptrs, dtype=np.uint64)
    offsets = (
        np.zeros_like(ptrs) if offsets is None else np.asarray(offsets, dtype=np.uint64)
    )
    sizes = (
        np.full_like(ptrs, record_size)
        if sizes is None
        else np.asarray(sizes, dtype=np.uint64)
    )
    return UCMProxyTransfer(
        tuple(keys),
        np.asarray(indices, dtype=np.int64),
        ptrs,
        sizes,
        offsets,
        tuple(
            json.dumps({"type": restore_policy.value, "chunk_index": i})
            for i in indices
        ),
        np.full(len(keys), record_size, dtype=np.uint64),
    )


class DeferredTestStore:
    """Delay Store IO until wait, with optional submission/completion failures."""

    backend = None

    def __init__(self, store):
        self.store = store
        self.backend = store.backend
        self.fail_submit = False
        self.fail_wait = False
        self.waited = 0

    def __getattr__(self, name):
        return getattr(self.store, name)

    def register_tensors(self, kv_caches):
        self.store.register_tensors(kv_caches)

    def dump(self, *args):
        if self.fail_submit:
            raise OSError("submit failure")
        return ("dump", args)

    def load(self, *args):
        return ("load", args)

    def commit(self, *args):
        return ("commit", args)

    def wait(self, task):
        self.waited += 1
        if self.fail_wait:
            raise OSError("wait failure")
        operation, arguments = task
        getattr(self.store, operation)(*arguments)


@pytest.mark.parametrize("mode", ["pipeline", "parallel"])
@pytest.mark.parametrize("store_kind", ["file", "memory", "deferred"])
def test_layerwise_roundtrip_through_real_proxy(tmp_path, mode, store_kind):
    memory = ByteMemory()
    access = MemoryByteAccess(memory)
    stores = []
    for i in range(2):
        store = (
            FileTestStore(tmp_path / str(i), access)
            if store_kind == "file"
            else MemoryTestStore(access)
        )
        stores.append(DeferredTestStore(store) if store_kind == "deferred" else store)
    proxy = (
        PipelineProxy(stores[0])
        if mode == "pipeline"
        else ParallelProxy(stores, [4096, None], 512)
    )
    adapter = UCMProxyAdapter(proxy)
    keys = (b"a" * 16, b"b" * 16, b"c" * 16)
    source = np.array([[100, 104], [200, 204], [300, 304]], dtype=np.uint64)
    expected = (b"abcdefgh", b"ijklmnop", b"qrstuvwx")
    for ptr, payload in zip(source[:, 0], expected):
        memory.write(ptr, payload)
    full = transfer(keys, [7, 8, 9], source, offsets=[[0, 4]] * 3, sizes=[[4, 4]] * 3)
    for column in range(2):
        layer = replace(
            full,
            ptrs=full.ptrs[:, column : column + 1],
            sizes=full.sizes[:, column : column + 1],
            ucm_chunk_offsets=full.ucm_chunk_offsets[:, column : column + 1],
        )
        for array in (
            layer.ptrs,
            layer.sizes,
            layer.ucm_chunk_offsets,
            layer.chunk_indices,
        ):
            array.flags.writeable = False
        adapter.submit("dump", layer)
        assert not any(adapter.lookup(keys, full.chunk_indices))
    adapter.commit(True, keys, keys, full.chunk_indices)
    assert adapter.lookup(keys, full.chunk_indices) == (True, True, True)
    # Mixed order must restore rows after splitting routes.
    load = transfer((keys[1], keys[0], keys[2]), [8, 7, 9], [[400], [500], [600]])
    adapter.submit("load", load)
    assert [memory.read(p, 8) for p in (500, 400, 600)] == list(expected)
    adapter.close()


def test_pipeline_writeback_eviction_and_complete_record_promotion(tmp_path):
    memory = ByteMemory()
    access = MemoryByteAccess(memory)
    lower = FileTestStore(tmp_path, access)
    middle = MemoryTestStore(access, backend=lower, capacity_bytes=8)
    upper = MemoryTestStore(access, backend=middle, capacity_bytes=8)
    adapter = UCMProxyAdapter(PipelineProxy(upper))
    a, b = b"a" * 16, b"b" * 16
    for key, ptr, data, index in ((a, 100, b"abcdefgh", 7), (b, 200, b"ijklmnop", 8)):
        memory.write(ptr, data)
        batch = transfer((key,), [index], [[ptr]], restore_policy=RestorePolicy.STATE)
        adapter.submit("dump", batch)
        assert lower.lookup((key,)) == (False,)
        adapter.commit(
            True, batch.ucm_chunk_ids, batch.ucm_chunk_ids, batch.chunk_indices
        )
        assert lower.read_record(key) == (data, batch.metadata[0])
    assert a not in upper.records and a not in middle.records
    adapter.submit("load", transfer((a,), [7], [[300]], offsets=[[4]], sizes=[[4]]))
    assert memory.read(300, 4) == b"efgh"
    assert upper.records[a][0] == b"abcdefgh"
    # An upper hit must not read from the lower Store.
    with patch.object(middle, "read_record", side_effect=AssertionError("lower read")):
        adapter.submit("load", transfer((a,), [7], [[400]]))
    assert memory.read(400, 8) == b"abcdefgh"
    adapter.close()


def test_failed_writeback_pins_source_and_flush_retries(tmp_path):
    memory = ByteMemory()
    access = MemoryByteAccess(memory)
    lower = FileTestStore(tmp_path, access)
    upper = MemoryTestStore(access, backend=lower, capacity_bytes=8)
    adapter = UCMProxyAdapter(PipelineProxy(upper))
    key = b"a" * 16
    memory.write(100, b"abcdefgh")
    batch = transfer((key,), [0], [[100]])
    adapter.submit("dump", batch)
    with patch.object(lower, "write_record", side_effect=OSError("disk failure")):
        with pytest.raises(UCMProxyError, match="commit"):
            adapter.commit(
                True, batch.ucm_chunk_ids, batch.ucm_chunk_ids, batch.chunk_indices
            )
        memory.write(100, b"XXXXXXXX")
        with pytest.raises(MemoryError):
            upper.write_record(b"b" * 16, b"ijklmnop", "{}")
        with pytest.raises(OSError, match="disk failure"):
            adapter.close()
        assert upper.records[key][0] == b"abcdefgh"
    adapter.flush()
    assert lower.read_record(key)[0] == b"abcdefgh"
    upper.write_record(b"b" * 16, b"ijklmnop", "{}")
    assert key not in upper.records
    adapter.close()


def test_parallel_never_falls_back_or_spills_to_other_route(tmp_path):
    memory = ByteMemory()
    access = MemoryByteAccess(memory)
    cache = MemoryTestStore(access, capacity_bytes=8)
    disk = FileTestStore(tmp_path, access)
    key = b"a" * 16
    disk.write_record(key, b"wrongway", "{}")
    proxy = ParallelProxy((cache, disk), [4096, None], 512)
    with (
        patch.object(disk, "lookup", side_effect=AssertionError("wrong route")),
        patch.object(disk, "load", side_effect=AssertionError("wrong route")),
        patch.object(disk, "dump", side_effect=AssertionError("wrong route")),
    ):
        assert proxy.lookup((key,), np.array([7])) == (False,)
        with pytest.raises(KeyError):
            proxy.load(
                (key,),
                np.array([7]),
                np.array([[0]]),
                np.array([[100]]),
                np.array([[8]]),
            )
        adapter = UCMProxyAdapter(proxy)
        memory.write(100, b"abcdefgh")
        batch = transfer((key,), [7], [[100]])
        adapter.submit("dump", batch)
        # Pending reservations consume capacity without becoming visible.
        with pytest.raises(UCMProxyError):
            adapter.submit("dump", transfer((b"b" * 16,), [6], [[100]]))
        adapter.commit(
            False, batch.ucm_chunk_ids, batch.ucm_chunk_ids, batch.chunk_indices
        )
        assert cache.lookup((key,)) == (False,)
    assert disk.read_record(key)[0] == b"wrongway"
    proxy.close()


def test_parallel_scans_use_original_order_and_absolute_indices():
    stores = [MemoryTestStore(MemoryByteAccess(ByteMemory())) for _ in range(2)]
    keys = tuple(bytes([i]) * 16 for i in range(4))
    indices = np.array([8, 6, 9, 7])
    stores[1].write_record(keys[0], b"", "{}")
    stores[0].write_record(keys[1], b"", "{}")
    stores[0].write_record(keys[3], b"", "{}")
    proxy = ParallelProxy(stores, [4096, None], 512)
    assert proxy.lookup(keys, indices) == (True, True, False, True)
    assert proxy.lookup_on_prefix(keys, indices) == 1
    assert proxy.lookup_on_reverse(keys, indices) == 3
    assert proxy.lookup_on_prefix(keys[1:], indices[1:]) == 0
    assert proxy.lookup((), np.array([], dtype=np.int64)) == ()


@pytest.mark.parametrize("failure", ["submit", "wait"])
def test_parallel_drains_other_route_when_one_fails(failure):
    access = MemoryByteAccess(ByteMemory())
    stores = [DeferredTestStore(MemoryTestStore(access)) for _ in range(2)]
    if failure == "submit":
        stores[1].fail_submit = True
    else:
        stores[0].fail_wait = True
    adapter = UCMProxyAdapter(ParallelProxy(stores, [4096, None], 512))
    batch = transfer((b"a" * 16, b"b" * 16), [7, 8], [[100], [200]])
    with pytest.raises(UCMProxyError):
        adapter.submit("dump", batch)
    assert stores[0].waited == 1
    assert stores[1].waited == (0 if failure == "submit" else 1)
    if failure == "wait":
        with pytest.raises(UCMProxyError, match="commit"):
            adapter.commit(
                False, batch.ucm_chunk_ids, batch.ucm_chunk_ids, batch.chunk_indices
            )
        stores[0].fail_wait = False
    adapter.commit(False, batch.ucm_chunk_ids, batch.ucm_chunk_ids, batch.chunk_indices)
    assert not any(store.store._pending for store in stores)


@pytest.mark.parametrize(
    "indices",
    [np.array([-1]), np.array([1.5]), np.array([[0]]), np.array([], dtype=int)],
)
def test_invalid_positions_fail_before_store_io(indices):
    store = MemoryTestStore(MemoryByteAccess(ByteMemory()))
    adapter = UCMProxyAdapter(PipelineProxy(store))
    batch = replace(transfer((b"a" * 16,), [0], [[100]]), chunk_indices=indices)
    with pytest.raises(ValueError, match="chunk_indices"):
        adapter.submit("dump", batch)
    assert store._pending == {}


@pytest.mark.parametrize(
    "ends", [[1, None], [0, None], [4096, 4096, None], [4096], [None, None]]
)
def test_parallel_rejects_invalid_thresholds(ends):
    stores = [MemoryTestStore() for _ in ends]
    with pytest.raises(ValueError):
        ParallelProxy(stores, ends, 512)


def test_parallel_rejects_store_with_backend():
    cache = MemoryTestStore(backend=MemoryTestStore())
    with pytest.raises(ValueError, match="backend=None"):
        ParallelProxy([cache], [None], 512)


@pytest.mark.parametrize("mode", ["pipeline", "parallel"])
def test_factory_one_proxy_per_process_ranks_differ_only_by_id(tmp_path, mode):
    settings = {
        "v2_stores": {
            "cache": {"type": "test_memory", "capacity_bytes": 8},
            "left": {"type": "test_file", "path": str(tmp_path / "left")},
            "right": {"type": "test_file", "path": str(tmp_path / "right")},
        },
        "v2_proxy": (
            {"type": "pipeline", "stores": ["cache", "left"]}
            if mode == "pipeline"
            else {
                "type": "parallel",
                "routes": [
                    {"token_end": 4096, "store": "left"},
                    {"token_end": None, "store": "right"},
                ],
            }
        ),
    }
    # Separate processes in production: each builds the same single Proxy.
    scheduler = create_proxy(settings, record_token_span=512)
    writers = [create_proxy(settings, record_token_span=512) for _ in range(2)]
    common_ids = (b"a" * 15, b"b" * 15)
    # Two independent policy objects use the same configured Stores.
    ids = [
        make_ucm_chunk_ids(common_ids, policy)
        for policy in (RestorePolicy.FA, RestorePolicy.STATE)
    ]
    every_rank = (*ids[0], *ids[1])
    for i, writer in enumerate(writers):
        proxy = writer._proxy
        stores = (proxy.store,) if mode == "pipeline" else proxy.stores
        memory = ByteMemory()
        for store in stores:
            store.byte_access = MemoryByteAccess(memory)
        memory.write(100, bytes([65 + i]) * 8)
        batch = transfer(ids[i], [7, 8], [[100], [100]])
        if mode == "pipeline":
            # Cache capacity holds one record; commit each before reserving the next.
            for row in range(2):
                one = transfer((ids[i][row],), [7 + row], [[100]])
                writer.submit("dump", one)
                writer.commit(
                    True, one.ucm_chunk_ids, one.ucm_chunk_ids, one.chunk_indices
                )
        else:
            writer.submit("dump", batch)
            writer.commit(True, ids[i], ids[i], batch.chunk_indices)
        assert scheduler.lookup(every_rank, np.tile(batch.chunk_indices, 2)) == (
            (True, True, False, False) if i == 0 else (True,) * 4
        )
        writer.submit("load", transfer(ids[i], [7, 8], [[200], [300]]))
        assert memory.read(200, 8) == bytes([65 + i]) * 8
        assert memory.read(300, 8) == bytes([65 + i]) * 8
    for writer in writers:
        writer.close()
    scheduler.close()


def test_factory_default_store_uses_its_configured_path(tmp_path):
    adapter = create_proxy({"v2_storage_path": str(tmp_path)}, record_token_span=512)
    assert adapter._proxy.store.root == tmp_path
    adapter.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"v2_stores": {}, "v2_proxy": {"type": "unknown"}},
        {"v2_stores": {}, "v2_proxy": {"type": "pipeline", "stores": ["missing"]}},
        {"v2_stores": {}, "v2_proxy": {"type": "pipeline", "stores": []}},
        {
            "v2_stores": {"cache": {"type": "test_memory"}},
            "v2_proxy": {"type": "pipeline", "stores": ["cache"]},
        },
        {
            "v2_stores": {"cache": {"type": "test_memory"}},
            "v2_proxy": {
                "type": "parallel",
                "routes": [{"token_end": None, "store": "cache"}],
            },
        },
        {
            "v2_stores": {
                "left": {"type": "test_file", "path": "same"},
                "right": {"type": "test_file", "path": "same"},
            },
            "v2_proxy": {
                "type": "parallel",
                "routes": [
                    {"token_end": 4096, "store": "left"},
                    {"token_end": None, "store": "right"},
                ],
            },
        },
    ],
)
def test_factory_rejects_invalid_or_process_local_configuration(settings):
    with pytest.raises(ValueError):
        create_proxy(settings, record_token_span=512)


def test_all_shard_lookup_accepts_hits_at_different_store_levels(tmp_path):
    access = MemoryByteAccess(ByteMemory())
    disk = FileTestStore(tmp_path, access)
    cache = MemoryTestStore(access, backend=disk)
    group_info = UCMKVCacheGroupSpec(
        0, 512, frozenset((KVCacheSpecKind.FULL_ATTENTION,))
    )
    spec = UCMKVCacheSpec((group_info,), 512, "cpu")
    scheduler = make_connector_scheduler(spec, cache)
    common_ids = (b"a" * 15,)
    (shard0,) = make_ucm_chunk_ids(common_ids, RestorePolicy.FA)
    (shard1,) = make_ucm_chunk_ids(common_ids, RestorePolicy.FA)
    cache.records[shard0] = (b"abcdefgh", "{}")
    assert scheduler._lookup_hit_tokens(common_ids, 513, 0) == 512
    cache.records.clear()
    disk.write_record(shard1, b"ijklmnop", "{}")
    assert scheduler._lookup_hit_tokens(common_ids, 513, 0) == 512
    cache.close()


def test_scheduler_sliced_lookup_and_wa_state_route_by_boundary():
    kinds = (
        KVCacheSpecKind.FULL_ATTENTION,
        KVCacheSpecKind.SLIDING_WINDOW,
        KVCacheSpecKind.MAMBA,
    )
    groups = tuple(
        UCMKVCacheGroupSpec(
            i,
            512,
            frozenset((kind,)),
            tail_tokens=512 if i == 1 else None,
        )
        for i, kind in enumerate(kinds)
    )
    spec = UCMKVCacheSpec(groups, 512, "cpu")
    stores = [MemoryTestStore(MemoryByteAccess(ByteMemory())) for _ in range(2)]
    proxy = ParallelProxy(stores, [4096, None], 512)
    scheduler = make_connector_scheduler(spec, recompute_tokens=0)
    scheduler._proxy = UCMProxyAdapter(proxy)

    request = FakeRequest("request", 5120)
    ucm_chunk_ids_by_restore_policy = request_ucm_ids(scheduler, request)
    for ucm_chunk_ids in ucm_chunk_ids_by_restore_policy:
        for index, ucm_chunk_id in enumerate(ucm_chunk_ids):
            stores[0 if index < 8 else 1].write_record(ucm_chunk_id, b"", "{}")
    assert scheduler.lookup(request, 3072).external_hit_tokens == 2048
    # Latest State exists only in the wrong route. The common boundary backs up.
    stores[1].records.pop(ucm_chunk_ids_by_restore_policy[2][9])
    stores[0].write_record(ucm_chunk_ids_by_restore_policy[2][9], b"", "{}")
    assert scheduler.lookup(request, 3072).external_hit_tokens == 1536
    stores[1].records.pop(ucm_chunk_ids_by_restore_policy[1][8])
    assert scheduler.lookup(request, 3072).external_hit_tokens == 1024


class NativeTensor(FakeTensor):
    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return int(np.prod(self.shape))

    def stride(self, index=None):
        return self._strides if index is None else self._strides[index]


def make_layout():
    names = ("model.layers.0.attn", "model.layers.1.attn")
    native_group = group(names, FullAttentionSpec(4))
    native_group.is_eagle_group = False
    native = config(native_group)
    native.kv_cache_layout = "NHD"
    spec = parse_spec(
        native,
        scheduler_block_size=4,
        hf_text_config=SimpleNamespace(model_type="llama"),
        device_type="cpu",
    )
    caches = {
        name: NativeTensor(1000 + i * 1000, (8, 1, 4, 2), (8, 8, 2, 1))
        for i, name in enumerate(names)
    }
    module = SimpleNamespace(
        KVCacheLayout={
            "NHD": SimpleNamespace(is_block_outermost=False, is_block_compact=False)
        }
    )
    with patch.dict(sys.modules, {"vllm.v1.kv_cache_layout": module}):
        layout = make_cache_layout(
            spec,
            caches,
            kv_cache_config=native,
            num_hidden_layers=2,
            use_layerwise=True,
        )
    return layout, caches, native


def test_transfer_builder_keeps_absolute_positions_and_complete_record_size():
    layout, _, _ = make_layout()
    builder = UCMTransferBuilder(layout)
    keys = (b"a" * 15, b"b" * 15)
    plan = UCMGroupDispatchPlan(
        RestorePolicy.FA,
        keys,
        plan_token_starts(12, 20, len(keys)),
        (np.array([1, 2]),),
        group_ids=(0,),
    )
    metadata = UCMConnectorMetadata(
        requests={"r": RequestDispatchMeta("r", dump_plans=(plan,), load_plans=(plan,))}
    )
    for layer in layout.layer_id_by_name:
        (dump,) = builder.build_dump_transfers(metadata, layer_name=layer)
        (load,) = builder.build_load_transfers(metadata, layer_name=layer)
        np.testing.assert_array_equal(dump.chunk_indices, [3, 4])
        np.testing.assert_array_equal(load.chunk_indices, [3, 4])
        np.testing.assert_array_equal(dump.record_nbytes, [16, 16])
        assert [json.loads(item)["chunk_index"] for item in dump.metadata] == [3, 4]
        assert int(dump.sizes.sum(axis=1)[0]) == 8
        assert load.record_nbytes is None


@pytest.mark.parametrize("layerwise", [False, True])
def test_connector_save_commits_to_original_route_and_shutdown_drains(layerwise):
    layout, _, _ = make_layout()
    memory = ByteMemory()
    memory.write(1008, b"abcdefgh")
    memory.write(2008, b"ijklmnop")
    stores = [
        DeferredTestStore(MemoryTestStore(MemoryByteAccess(memory))) for _ in range(2)
    ]
    adapter = UCMProxyAdapter(ParallelProxy(stores, [12, None], 4))
    worker = UCMConnectorWorker.__new__(UCMConnectorWorker)
    worker.select_dump_rows = lambda group_id, indices: np.arange(len(indices))
    worker.layout = layout
    worker.spec = layout.spec
    worker.transfer_builder = UCMTransferBuilder(layout)
    worker._proxy = adapter
    worker.use_layerwise = layerwise
    worker._layer_load_tasks = {}
    worker._dump_tasks = []
    worker._dump_submitted_layers = set()
    worker._failed_dump_attempts = set()
    worker._failed_load_reqs = set()
    worker._step_save_finalized = False
    plan = UCMGroupDispatchPlan(
        RestorePolicy.FA,
        (b"a" * 15,),
        plan_token_starts(12, 16, 1),
        (np.array([1]),),
        group_ids=(0,),
    )
    bind_worker(
        worker,
        UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
        ),
    )
    finish_worker_save(worker)
    ucm_chunk_ids = ucm_ids(plan.common_ucm_chunk_ids)
    assert stores[0].store.lookup(ucm_chunk_ids) == (False,)
    assert stores[1].store.lookup(ucm_chunk_ids) == (True,)
    assert stores[1].store.records[ucm_chunk_ids[0]][0] == b"abcdefghijklmnop"
    worker.shutdown()


def test_empty_pp_group_creates_no_transfer_or_commit_target():
    spec = UCMKVCacheSpec(
        (UCMKVCacheGroupSpec(0, 512, frozenset((KVCacheSpecKind.MAMBA,))),),
        512,
        "cpu",
    )
    from ucm.integration.vllm.v2.layout.group import GroupLayoutMetadata
    from ucm.integration.vllm.v2.layout.store_layout import GroupStoreLayout
    from ucm.integration.vllm.v2.runtime import ParallelRank

    metadata = (
        (0, {0: GroupLayoutMetadata((), (), True, 1)}, None),
        (1, {0: GroupLayoutMetadata((8,), (1,), False, 1)}, None),
    )
    local = GroupStoreLayout.build(
        spec, 0, None, metadata, ParallelTopology(pp_size=2), ParallelRank()
    )
    stores = {0: local}
    builder = UCMTransferBuilder(
        SimpleNamespace(spec=spec, group_layouts={}, store_layouts=stores),
        ParallelRank(),
    )
    plan = UCMGroupDispatchPlan(
        RestorePolicy.STATE,
        (b"s" * 15,),
        np.array([4096]),
        (np.array([]),),
        group_ids=(0,),
    )
    metadata = UCMConnectorMetadata(
        requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
    )
    assert builder.build_dump_transfers(metadata) == ()
    assert builder.build_dump_transfers(metadata) == ()


def test_reused_committed_record_is_pinned_until_its_commit():
    memory = ByteMemory()
    store = MemoryTestStore(MemoryByteAccess(memory), capacity_bytes=8)
    key = b"a" * 16
    store.write_record(key, b"abcdefgh", "{}")
    adapter = UCMProxyAdapter(PipelineProxy(store))
    batch = transfer((key,), [0], [[100]])
    adapter.submit("dump", batch)
    with pytest.raises(MemoryError):
        store.write_record(b"b" * 16, b"ijklmnop", "{}")
    adapter.commit(True, batch.ucm_chunk_ids, batch.ucm_chunk_ids, batch.chunk_indices)
    store.write_record(b"b" * 16, b"ijklmnop", "{}")
    assert store.lookup((key,)) == (False,)


@pytest.mark.parametrize("layerwise", [False, True])
def test_connector_failed_dump_drains_and_aborts_pending(layerwise):
    layout, _, _ = make_layout()
    stores = [
        DeferredTestStore(MemoryTestStore(MemoryByteAccess(ByteMemory())))
        for _ in range(2)
    ]
    stores[1].fail_submit = True
    worker = UCMConnectorWorker.__new__(UCMConnectorWorker)
    worker.select_dump_rows = lambda group_id, indices: np.arange(len(indices))
    worker.layout = layout
    worker.spec = layout.spec
    worker.transfer_builder = UCMTransferBuilder(layout)
    worker._proxy = UCMProxyAdapter(ParallelProxy(stores, [12, None], 4))
    worker.use_layerwise = layerwise
    worker._layer_load_tasks = {}
    worker._dump_tasks = []
    worker._dump_submitted_layers = set()
    worker._failed_dump_attempts = set()
    worker._failed_load_reqs = set()
    worker._step_save_finalized = False
    plan = UCMGroupDispatchPlan(
        RestorePolicy.FA,
        (b"a" * 15, b"b" * 15),
        plan_token_starts(8, 16, 2),
        (np.array([1, 2]),),
        group_ids=(0,),
    )
    bind_worker(
        worker,
        UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
        ),
    )
    finish_worker_save(worker)
    # Layerwise fallback drains both layers even after another route fails.
    assert stores[0].waited == (3 if layerwise else 2)
    assert not any(store.store._pending for store in stores)
    assert not worker._dump_tasks
    worker.shutdown()


def test_close_drains_all_routes_even_if_one_close_fails():
    stores = [MemoryTestStore(), MemoryTestStore()]
    proxy = ParallelProxy(stores, [512, None], 512)
    with (
        patch.object(stores[0], "close", side_effect=OSError("close failure")),
        patch.object(stores[1], "close") as close,
    ):
        with pytest.raises(OSError, match="close failure"):
            proxy.close()
        close.assert_called_once()


def test_parallel_retains_sliced_arrays_until_wait():
    import weakref

    class WeakTaskStore(DeferredTestStore):
        def dump(self, *arguments):
            return tuple(
                weakref.ref(value) if isinstance(value, np.ndarray) else value
                for value in arguments
            )

        def wait(self, task):
            arguments = tuple(
                value() if isinstance(value, weakref.ReferenceType) else value
                for value in task
            )
            assert all(value is not None for value in arguments)
            self.store.dump(*arguments)

    stores = [
        WeakTaskStore(MemoryTestStore(MemoryByteAccess(ByteMemory()))) for _ in range(2)
    ]
    adapter = UCMProxyAdapter(ParallelProxy(stores, [512, None], 512))
    task = adapter.enqueue(
        "dump", transfer((b"a" * 16, b"b" * 16), [0, 1], [[100], [200]])
    )
    adapter.wait(task)
    assert all(store.store._pending for store in stores)
    adapter.close()


@pytest.mark.parametrize("role", [KVConnectorRole.WORKER, KVConnectorRole.SCHEDULER])
def test_connector_creation_is_delegated_to_factory(role):
    _, _, native = make_layout()
    cfg = vllm_config()
    cfg.instance_id = "engine"
    cfg.use_v2_model_runner = False
    cfg.model_config.hf_text_config = SimpleNamespace(
        model_type="llama",
        num_hidden_layers=2,
        to_dict=lambda: {"model_type": "llama", "num_hidden_layers": 2},
    )
    cfg.cache_config.cache_dtype = "auto"
    cfg.cache_config.mamba_cache_dtype = "auto"
    cfg.cache_config.mamba_ssm_cache_dtype = "auto"
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.parallel_config.decode_context_parallel_size = 1
    cfg.parallel_config.cp_kv_cache_interleave_size = 1
    cfg.parallel_config.world_size = 2
    adapter = UCMProxyAdapter(PipelineProxy(MemoryTestStore()))
    with (
        patch("ucm.integration.vllm.v2.runtime._worker_rank", return_value=1),
        patch(
            "ucm.integration.vllm.v2.runtime._worker_parallel_ranks",
            return_value=(1, 0, 0),
        ),
        patch(
            (
                "ucm.integration.vllm.v2.connector_worker.create_proxy"
                if role == KVConnectorRole.WORKER
                else "ucm.integration.vllm.v2.connector_scheduler.create_proxy"
            ),
            return_value=adapter,
        ) as factory,
    ):
        connector = UCMConnector(cfg, role, native)
    connector = connector.connector_worker or connector.connector_scheduler
    assert connector._proxy is adapter
    assert factory.call_args.kwargs == {"record_token_span": 4}
    assert connector.parallel_topology == ParallelTopology(tp_size=2)
    if role == KVConnectorRole.WORKER:
        assert connector._parallel_rank == (1, 0, 0)
        assert not hasattr(connector, "_replicas")
    else:
        assert connector._chunk_hasher is None
        assert list(connector.groups_by_policy) == [RestorePolicy.FA]
    connector.shutdown()


def test_late_load_failure_discards_already_enqueued_layer_dumps():
    layout, _, _ = make_layout()
    store = DeferredTestStore(MemoryTestStore(MemoryByteAccess(ByteMemory())))
    worker = UCMConnectorWorker.__new__(UCMConnectorWorker)
    worker.select_dump_rows = lambda group_id, indices: np.arange(len(indices))
    worker.layout = layout
    worker.spec = layout.spec
    worker.transfer_builder = UCMTransferBuilder(layout)
    worker._proxy = UCMProxyAdapter(PipelineProxy(store))
    worker.use_layerwise = True
    worker._layer_load_tasks = {}
    worker._dump_tasks = []
    worker._dump_submitted_layers = set()
    worker._failed_dump_attempts = set()
    worker._failed_load_reqs = set()
    worker._step_save_finalized = False
    plan = UCMGroupDispatchPlan(
        RestorePolicy.FA,
        (b"a" * 15,),
        plan_token_starts(12, 16, 1),
        (np.array([1]),),
        group_ids=(0,),
    )
    metadata = UCMConnectorMetadata(
        requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
    )
    bind_worker(worker, metadata)
    worker._save_layer("model.layers.0.attn", worker._get_connector_metadata())
    worker._failed_load_reqs.add("r")
    finish_worker_save(worker)
    assert store.waited == 2  # dump completion, then commit(False) cleanup
    assert not store.store._pending
    assert store.store.lookup(ucm_ids(plan.common_ucm_chunk_ids)) == (False,)
    worker.shutdown()


@pytest.mark.parametrize("policy", [RestorePolicy.FA, RestorePolicy.STATE])
@pytest.mark.parametrize(
    "failure", [OSError("storage unavailable"), ValueError("invalid reply")]
)
def test_lookup_io_failures_are_misses_but_contract_errors_are_fatal(policy, failure):
    from test_ucm_connector_v2 import MambaSpec

    native = FullAttentionSpec(4) if policy is RestorePolicy.FA else MambaSpec(4)
    spec = parse_spec(
        config(group(["model.layers.0.attn"], native)), scheduler_block_size=4
    )
    store = MemoryTestStore()
    scheduler = make_connector_scheduler(spec, store)
    operation = (
        "lookup_on_prefix" if policy is RestorePolicy.FA else "lookup_on_reverse"
    )
    with patch.object(store, operation, side_effect=failure):
        if isinstance(failure, ValueError):
            with pytest.raises(ValueError, match="invalid reply"):
                scheduler.lookup(FakeRequest("r", 16), 0)
        else:
            result = scheduler.lookup(FakeRequest("r", 16), 0)
            assert result.external_hit_tokens == 0
            assert not scheduler.requests["r"].need_load
            assert scheduler.requests["r"].common_ucm_chunk_ids


def test_fatal_wait_drains_other_tasks_and_preserves_unsafe_descriptors():
    layout, _, _ = make_layout()
    worker = UCMConnectorWorker.__new__(UCMConnectorWorker)
    worker._failed_dump_attempts = set()
    worker._layer_load_tasks = {}
    worker._dump_tasks = [object(), object(), object()]
    tasks = tuple(worker._dump_tasks)
    from unittest.mock import Mock

    worker._proxy = Mock()
    worker._proxy.wait.side_effect = [OSError("unwrapped backend error"), None, None]
    # The role only accepts terminal UCMProxyError from its adapter. Unknown
    # completion is fatal and must keep references; remaining tasks still drain.
    with pytest.raises(OSError):
        worker._drain_dump_tasks()
    assert worker._proxy.wait.call_count == 3
    assert worker._dump_tasks == [tasks[0]]
    with pytest.raises(RuntimeError, match="transfers own KV"):
        worker.clear_connector_metadata()


@pytest.mark.parametrize("failure_at", ["wait", "commit_true", "commit_false"])
def test_best_effort_save_finalizes_after_terminal_storage_failure(failure_at):
    layout, _, _ = make_layout()
    store = DeferredTestStore(MemoryTestStore(MemoryByteAccess(ByteMemory())))
    worker = UCMConnectorWorker.__new__(UCMConnectorWorker)
    worker.layout, worker.spec = layout, layout.spec
    worker.transfer_builder = UCMTransferBuilder(layout)
    worker._proxy = UCMProxyAdapter(PipelineProxy(store))
    worker.use_layerwise = True
    worker._layer_load_tasks, worker._dump_tasks = {}, []
    worker._dump_submitted_layers, worker._failed_load_reqs = set(), set()
    worker._failed_dump_attempts, worker._step_save_finalized = set(), False
    worker.select_dump_rows = lambda group_id, indices: np.arange(len(indices))
    plan = UCMGroupDispatchPlan(
        RestorePolicy.FA, (b"a" * 15,), np.array([12]), (np.array([1]),), group_ids=(0,)
    )
    bind_worker(
        worker,
        UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
        ),
    )
    if failure_at in ("wait", "commit_false"):
        store.fail_wait = True
    with patch.object(
        store,
        "wait" if failure_at == "wait" else "commit",
        side_effect=OSError("storage failed"),
    ):
        finish_worker_save(worker)
    assert worker._step_save_finalized
    assert not worker._dump_tasks
    assert store.store.lookup(ucm_ids(plan.common_ucm_chunk_ids)) == (False,)
    worker.clear_connector_metadata()


@pytest.mark.parametrize(
    "change", ["first_fa", "second_fa", "state", "scratch", "capacity"]
)
def test_scheduler_fingerprint_and_keys_use_stored_group_block_sizes(change):
    from test_ucm_connector_v2 import MambaSpec, UnsupportedSpec

    from ucm.integration.vllm.request_hasher import RequestHasher
    from ucm.integration.vllm.v2.runtime import UCMRuntimeContext

    cfg = vllm_config()
    cfg.kv_transfer_config.kv_connector_extra_config["chunk_size"] = 8
    baseline_specs = [FullAttentionSpec(4), FullAttentionSpec(4)]
    changed_specs = [FullAttentionSpec(4), FullAttentionSpec(4)]
    if change == "first_fa":
        changed_specs[0] = FullAttentionSpec(8)
    elif change == "second_fa":
        changed_specs[1] = FullAttentionSpec(8)
    elif change == "state":
        baseline_specs[1], changed_specs[1] = MambaSpec(4), MambaSpec(8)
    elif change == "scratch":
        baseline_specs[1] = UnsupportedSpec(4, prefix_cacheable=False)
        changed_specs[1] = UnsupportedSpec(8, prefix_cacheable=False)
    native_configs = [
        config(
            *(
                group([f"model.layers.{group_id}.cache"], group_spec)
                for group_id, group_spec in enumerate(group_specs)
            ),
            num_blocks=16 if change == "capacity" and config_index == 1 else 8,
        )
        for config_index, group_specs in enumerate((baseline_specs, changed_specs))
    ]
    contexts = [
        UCMRuntimeContext.from_vllm_config(cfg, KVConnectorRole.SCHEDULER, native)
        for native in native_configs
    ]
    assert contexts[0].spec.chunk_size == contexts[1].spec.chunk_size == 8
    different = change in ("first_fa", "second_fa", "state")
    assert (contexts[0].fingerprint != contexts[1].fingerprint) is different
    keys = [
        tuple(
            RequestHasher(context.fingerprint).make_request_block_hasher(8)(
                FakeRequest("r", 16)
            )
        )
        for context in contexts
    ]
    assert (keys[0] != keys[1]) is different


def test_stored_format_version_changes_scheduler_fingerprint_and_keys():
    from ucm.integration.vllm.request_hasher import RequestHasher
    from ucm.integration.vllm.v2.runtime import UCM_CHUNK_ID_VERSION, UCMRuntimeContext

    cfg = vllm_config()
    native = config(group(["model.layers.0.attn"], FullAttentionSpec(4)))
    before = UCMRuntimeContext.from_vllm_config(cfg, KVConnectorRole.SCHEDULER, native)
    with patch(
        "ucm.integration.vllm.v2.runtime.UCM_CHUNK_ID_VERSION", UCM_CHUNK_ID_VERSION + 1
    ):
        after = UCMRuntimeContext.from_vllm_config(
            cfg, KVConnectorRole.SCHEDULER, native
        )
    assert len(before.fingerprint) == len(after.fingerprint) == 64
    assert before.fingerprint != after.fingerprint
    keys = [
        tuple(
            RequestHasher(context.fingerprint).make_request_block_hasher(4)(
                FakeRequest("r", 16)
            )
        )
        for context in (before, after)
    ]
    assert keys[0] != keys[1]


def test_pp_layer_membership_does_not_add_worker_cache_identity():
    from ucm.integration.vllm.v2.runtime import ParallelRank, UCMRuntimeContext

    cfg = vllm_config()
    cfg.parallel_config.pipeline_parallel_size = 2
    cfg.parallel_config.world_size = 4
    native_configs = [
        config(
            group(
                [f"model.layers.{stage * 3 + i}.attn" for i in range(2 - stage)],
                FullAttentionSpec(4),
            )
        )
        for stage in range(2)
    ]
    scheduler = UCMRuntimeContext.from_vllm_config(
        cfg, KVConnectorRole.SCHEDULER, native_configs[0]
    )
    for pp_rank, native in enumerate(native_configs):
        with (
            patch(
                "ucm.integration.vllm.v2.runtime._worker_rank", return_value=pp_rank * 2
            ),
            patch(
                "ucm.integration.vllm.v2.runtime._worker_parallel_ranks",
                return_value=ParallelRank(0, pp_rank, 0),
            ),
        ):
            worker = UCMRuntimeContext.from_vllm_config(
                cfg, KVConnectorRole.WORKER, native
            )
        assert worker.fingerprint == scheduler.fingerprint


@pytest.mark.parametrize("has_fa", [False, True])
@pytest.mark.parametrize("state_boundaries,expected", [({9, 6, 5}, 20), ({9, 6}, 0)])
def test_lookup_requires_one_boundary_shared_by_swa_and_state(
    has_fa, state_boundaries, expected
):
    from test_ucm_connector_v2 import AscendSlidingWindowMLASpec, MambaSpec

    native_groups = [
        group(["model.layers.1.attn"], AscendSlidingWindowMLASpec(4, 1, 4)),
        group(["model.layers.2.mixer"], MambaSpec(4)),
    ]
    if has_fa:
        native_groups.insert(0, group(["model.layers.0.attn"], FullAttentionSpec(4)))
    spec = parse_spec(config(*native_groups), scheduler_block_size=4)
    store = MemoryTestStore()
    scheduler = make_connector_scheduler(spec, store)
    request = FakeRequest("r", 41)
    common_ids = tuple(bytes((i,)) * 15 for i in range(10))
    boundaries = {
        RestorePolicy.FA: set(range(1, 11)),
        RestorePolicy.WA: {10, 8, 5},
        RestorePolicy.STATE: state_boundaries,
    }
    try:
        for policy in scheduler.groups_by_policy:
            for boundary in boundaries[policy]:
                (key,) = make_ucm_chunk_ids((common_ids[boundary - 1],), policy)
                store.write_record(key, b"", "{}")
        # Taking min(last WA hit, last State hit) would incorrectly return 9.
        assert (
            scheduler._lookup_hit_tokens(common_ids, len(request.all_token_ids), 0)
            == expected
        )
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize(
    "change", ["device", "dtype", "chunk", "layerwise", "json_order"]
)
def test_fingerprint_hashes_all_configuration_in_one_schema(change):
    from ucm.integration.vllm.v2.runtime import UCMRuntimeContext

    cfg = vllm_config()
    native = config(group(["model.layers.0.attn"], FullAttentionSpec(4)))
    before = UCMRuntimeContext.from_vllm_config(cfg, KVConnectorRole.SCHEDULER, native)
    if change == "device":
        cfg.device_config.device_type = "npu"
    elif change == "dtype":
        cfg.model_config.dtype = "float16"
    elif change == "chunk":
        cfg.kv_transfer_config.kv_connector_extra_config["chunk_size"] = 8
    elif change == "layerwise":
        cfg.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
    else:
        fields = dict(
            reversed(tuple(cfg.model_config.hf_text_config.to_dict().items()))
        )
        cfg.model_config.hf_text_config.to_dict = lambda: fields
    after = UCMRuntimeContext.from_vllm_config(cfg, KVConnectorRole.SCHEDULER, native)
    assert (
        len(bytes.fromhex(before.fingerprint))
        == len(bytes.fromhex(after.fingerprint))
        == 32
    )
    assert (before.fingerprint != after.fingerprint) == (change != "json_order")


@pytest.mark.parametrize(
    "layerwise,copy_before_load,fail_load",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (False, True, True),
        (True, True, True),
    ],
)
def test_state_restore_populates_boundary_and_running_slots(
    layerwise, copy_before_load, fail_load
):
    """Both State destinations finish loading, or both report terminal IO failure."""
    import torch
    from test_ucm_connector_v2 import MambaSpec

    from ucm.integration.vllm.v2.runtime import ParallelRank
    from ucm.integration.vllm.v2.test_stores import TorchTensorByteAccess

    native = config(group(["model.layers.0.mixer"], MambaSpec(4)))
    spec = parse_spec(native, scheduler_block_size=4, device_type="npu")
    scheduler = make_connector_scheduler(spec, MemoryTestStore())
    context = replace(
        scheduler.context,
        role=KVConnectorRole.WORKER,
        rank=ParallelRank(),
        global_rank=0,
        use_layerwise=layerwise,
    )
    store = DeferredTestStore(MemoryTestStore(TorchTensorByteAccess()))
    adapter = UCMProxyAdapter(PipelineProxy(store))
    with patch(
        "ucm.integration.vllm.v2.connector_worker.create_proxy", return_value=adapter
    ):
        worker = UCMConnectorWorker(context, native, num_hidden_layers=1)
    pages = torch.zeros((8, 64, 4), dtype=torch.uint8)
    pages[1].fill_(17)
    pages[2].fill_(29)
    try:
        worker.register_kv_caches({"model.layers.0.mixer": pages})
        common_ids = (b"a" * 15, b"b" * 15, b"c" * 15)
        (key,) = make_ucm_chunk_ids((common_ids[1],), RestorePolicy.STATE)
        checkpoint = bytes(range(256))
        store.write_record(key, checkpoint, "{}")
        state = seed_request_state(scheduler, FakeRequest("r", 12), 8, common_ids)
        request = scheduler._build_request_plan("r", state, ((0, 1, 2, 0),), 4)
        worker.bind_connector_metadata(UCMConnectorMetadata(requests={"r": request}))
        if copy_before_load:
            pages[2].copy_(pages[1])  # Runner copy occurred before the external load.
        store.fail_wait = fail_load
        worker.start_load_kv(None)
        if fail_load:
            assert worker.get_block_ids_with_load_errors() == {1, 2}
            assert worker.build_connector_worker_meta().load_failed_reqs == {"r"}
            assert not worker._layer_load_tasks
            return
        assert pages[1].numpy().tobytes() == checkpoint
        assert pages[2].numpy().tobytes() == checkpoint
        if not copy_before_load:
            pages[2].copy_(pages[1])
            assert pages[2].numpy().tobytes() == checkpoint
        assert not pages[0].any() and not pages[3:].any()
        assert not worker._layer_load_tasks
    finally:
        worker.shutdown()
        scheduler.shutdown()
