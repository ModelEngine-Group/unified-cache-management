"""CPU regression coverage for FAWA dump ownership across the TP*PCP domain.

With PCP>1 and DCP=1 every PCP peer holds a full replica of the KV cache, so
each canonical block must be dumped by exactly one of the TP*PCP workers.
Duplicate dumps from PCP peers race on shared Cache2 slots (see the FAWA
207000 incident); missing dumps silently lose prefix-cache hits.
"""

import ast
import itertools
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_ucm_v41_compat import HMA_PATH, UCM_PATH, _load_hma_methods

WORKER = "worker"
SCHEDULER = "scheduler"


def _load_class_method(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if getattr(n, "name", None) == class_name)
    node = next(n for n in cls.body if getattr(n, "name", None) == method_name)
    namespace = {}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def _is_self_assign(stmt, attr):
    return (
        isinstance(stmt, ast.Assign)
        and isinstance(stmt.targets[0], ast.Attribute)
        and stmt.targets[0].attr == attr
    )


def _load_dump_ownership_init():
    """Extract the ownership block of UCMFAWAConnector.__init__ as a function.

    The block spans ``self.pcp_world_size = 1`` .. ``self.dump_owner_rank = ...``
    and only depends on ``role``, the PCP/DCP groups and TP attributes.
    """
    tree = ast.parse(HMA_PATH.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if getattr(n, "name", None) == "UCMFAWAConnector")
    init = next(n for n in cls.body if getattr(n, "name", None) == "__init__")
    start = next(
        i for i, s in enumerate(init.body) if _is_self_assign(s, "pcp_world_size")
    )
    end = next(
        i for i, s in enumerate(init.body) if _is_self_assign(s, "dump_owner_rank")
    )
    func = ast.parse("def _init_dump_ownership(self, role):\n    pass").body[0]
    func.body = init.body[start : end + 1]
    module = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(module)
    warnings = []
    namespace = {
        "KVConnectorRole": NS(WORKER=WORKER, SCHEDULER=SCHEDULER),
        "logger": NS(warning=lambda *args: warnings.append(args)),
    }
    exec(compile(module, str(HMA_PATH), "exec"), namespace)
    return namespace["_init_dump_ownership"], warnings


@pytest.fixture
def fake_cp_groups(monkeypatch):
    """Install fake vllm.distributed PCP/DCP groups; returns a setter."""
    state = {"pcp": None, "dcp": None}

    def get_pcp_group():
        assert state["pcp"] is not None, "pcp group is not initialized"
        return state["pcp"]

    def get_dcp_group():
        assert state["dcp"] is not None, "dcp group is not initialized"
        return state["dcp"]

    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pcp_group = get_pcp_group
    distributed.get_dcp_group = get_dcp_group
    vllm = types.ModuleType("vllm")
    vllm.distributed = distributed
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)

    def configure(pcp_size, pcp_rank, dcp_size=1, initialized=True):
        if not initialized:
            state["pcp"] = state["dcp"] = None
            return
        state["pcp"] = NS(world_size=pcp_size, rank_in_group=pcp_rank)
        state["dcp"] = NS(world_size=dcp_size, rank_in_group=0)

    return configure


def _ownership(
    fake_cp_groups,
    *,
    tp,
    pcp,
    tp_rank,
    pcp_rank,
    dcp=1,
    role=WORKER,
    initialized=True,
):
    init, warnings = _load_dump_ownership_init()
    fake_cp_groups(pcp, pcp_rank, dcp, initialized)
    # vLLM layout is ExternalDP x DP x PP x PCP x TP; UCM reads tp_rank from
    # parallel_config.rank, i.e. the worker rank inside the DP replica.
    connector = NS(
        _vllm_config=NS(parallel_config=NS(prefill_context_parallel_size=pcp)),
        tp_size=tp,
        tp_rank=pcp_rank * tp + tp_rank,
    )
    init(connector, role)
    return connector, warnings


GRIDS = [(1, 1), (4, 1), (1, 2), (2, 2), (4, 2), (2, 4), (8, 2)]


@pytest.mark.parametrize("tp,pcp", GRIDS)
def test_worker_owner_ranks_are_a_bijection_onto_tp_times_pcp(fake_cp_groups, tp, pcp):
    owners = []
    for pcp_rank, tp_rank in itertools.product(range(pcp), range(tp)):
        connector, warnings = _ownership(
            fake_cp_groups, tp=tp, pcp=pcp, tp_rank=tp_rank, pcp_rank=pcp_rank
        )
        assert connector.dump_owner_size == tp * pcp
        # Owner equals the worker's position inside the TP x PCP grid.
        assert connector.dump_owner_rank == pcp_rank * tp + tp_rank
        assert warnings == []
        owners.append(connector.dump_owner_rank)
    assert sorted(owners) == list(range(tp * pcp))


@pytest.mark.parametrize("tp", [1, 4])
def test_pcp1_keeps_the_upstream_tp_only_domain(fake_cp_groups, tp):
    for tp_rank in range(tp):
        connector, _ = _ownership(
            fake_cp_groups, tp=tp, pcp=1, tp_rank=tp_rank, pcp_rank=0
        )
        assert (connector.pcp_world_size, connector.pcp_rank) == (1, 0)
        assert connector.dump_owner_size == tp
        assert connector.dump_owner_rank == tp_rank


def test_dcp_sharded_kv_is_not_merged_across_pcp(fake_cp_groups):
    connector, warnings = _ownership(
        fake_cp_groups, tp=4, pcp=2, tp_rank=1, pcp_rank=1, dcp=2
    )
    assert (connector.dump_owner_size, connector.dump_owner_rank) == (4, 1)
    assert len(warnings) == 1 and "DCP>1" in warnings[0][0]


def test_unavailable_pcp_group_falls_back_to_tp_only(fake_cp_groups):
    connector, warnings = _ownership(
        fake_cp_groups, tp=4, pcp=2, tp_rank=3, pcp_rank=1, initialized=False
    )
    assert (connector.dump_owner_size, connector.dump_owner_rank) == (4, 3)
    assert len(warnings) == 1 and "unavailable" in warnings[0][0]


def test_scheduler_never_queries_pcp_groups(fake_cp_groups):
    # The scheduler does not dump and has no PCP group; it must not touch it.
    connector, warnings = _ownership(
        fake_cp_groups,
        tp=4,
        pcp=2,
        tp_rank=0,
        pcp_rank=0,
        role=SCHEDULER,
        initialized=False,
    )
    assert (connector.dump_owner_size, connector.dump_owner_rank) == (4, 0)
    assert warnings == []


def _fawa_slice():
    return _load_hma_methods({"_get_mla_dump_slice"})["_get_mla_dump_slice"]


@pytest.mark.parametrize("owner_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("dump_block_start", [0, 1, 3, 7, 13, 64])
@pytest.mark.parametrize("num_keys", [0, 1, 3, 8, 17])
def test_dump_slice_partitions_keys_by_absolute_block_position(
    owner_size, dump_block_start, num_keys
):
    get_slice = _fawa_slice()
    indices = list(range(num_keys))
    seen = []
    for owner in range(owner_size):
        connector = NS(dump_owner_rank=owner, dump_owner_size=owner_size)
        selected = indices[get_slice(connector, dump_block_start)]
        # Ownership follows the absolute hash-block number, so chunked
        # prefills that dump one request over several steps stay stable.
        assert all((dump_block_start + i) % owner_size == owner for i in selected)
        seen.extend(selected)
    assert sorted(seen) == indices  # every key exactly once


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("dump_block_start", [0, 1, 5, 13])
def test_dump_slice_matches_upstream_when_pcp_is_one(tp, dump_block_start):
    fawa_slice = _fawa_slice()
    upstream_slice = _load_class_method(
        UCM_PATH, "UCMDirectConnector", "_get_mla_dump_slice"
    )
    for tp_rank in range(tp):
        fawa = NS(dump_owner_rank=tp_rank, dump_owner_size=tp)
        upstream = NS(tp_rank=tp_rank, tp_size=tp)
        assert fawa_slice(fawa, dump_block_start) == upstream_slice(
            upstream, dump_block_start
        )


# ---------------------------------------------------------------------------
# wait_for_save end to end: FA rows, block-wise WA and chunk-wise WA.
# ---------------------------------------------------------------------------

WA_TAIL_BLOCKS = 2
# (request_id, dump_hash_start, num_dump_keys)
REQUESTS = [("a", 0, 5), ("b", 3, 1), ("c", 13, 11), ("d", 7, 0), ("e", 2, 4)]


def _requests():
    meta = {}
    for request_id, start, count in REQUESTS:
        positions = list(range(start, start + count))
        meta[request_id] = NS(
            dump_keys=[f"{request_id}:{p}".encode() for p in positions],
            dump_hash_start=start,
            dump_hash_end=start + count,
            # group 0 = FA (one vLLM block per hash block),
            # group 1 = WA (tail blocks per hash block).
            dump_vllm_block_ids=(
                [1000 + p for p in positions],
                [
                    2000 + p * WA_TAIL_BLOCKS + t
                    for p in positions
                    for t in range(WA_TAIL_BLOCKS)
                ],
            ),
        )
    return meta


def _position(key):
    return int(key.decode().split(":")[1])


ALL_KEYS = sorted(k for r in _requests().values() for k in r.dump_keys)


def _run_wait_for_save(owner_rank, owner_size, *, wa_block_wise):
    methods = _load_hma_methods({"wait_for_save", "_get_mla_dump_slice"})
    methods["UCMFAWAConnectorMetadata"] = NS
    methods["ucmmetrics"] = NS(update_stats=lambda _stats: None)
    submitted = {}
    fa_rows = []
    wa_rows = []

    def extract_fa(keys, _start, _end, vllm_block_ids, hash_positions=None):
        fa_rows.append((list(keys), vllm_block_ids, hash_positions))
        return np.zeros((len(keys), 2), dtype=np.uint64)

    def extract_wa(keys, vllm_block_ids):
        wa_rows.append((list(keys), vllm_block_ids))
        return np.zeros((len(keys), 2), dtype=np.uint64)

    def submit(label, _store, keys, _ptrs, _event, blocks_by_request):
        submitted[label] = (list(keys), blocks_by_request)
        return NS(key_count=len(keys))

    metadata = NS(request_meta=_requests())
    connector = NS(
        fa_group_ids=[0],
        window_group_ids=[1],
        group_metas={1: NS(tail_blocks=WA_TAIL_BLOCKS)},
        fa_store=object(),
        wa_store=object(),
        wa_dump_block_wise=wa_block_wise,
        dump_owner_rank=owner_rank,
        dump_owner_size=owner_size,
        tp_dump_tasks={},
        file_size={"FA": 1, "WA": 1},
        _get_connector_metadata=lambda: metadata,
        _poll_completed_dump_tasks=lambda: None,
        _extract_fa_ptr=extract_fa,
        _extract_wa_ptr=extract_wa,
        _get_dump_event_handle=lambda: 1,
        _submit_dump_task=submit,
    )
    connector._get_mla_dump_slice = types.MethodType(
        methods["_get_mla_dump_slice"], connector
    )
    methods["wait_for_save"](connector)
    return submitted, fa_rows, wa_rows


@pytest.mark.parametrize("tp,pcp", [(1, 2), (2, 2), (4, 2), (2, 4)])
@pytest.mark.parametrize("wa_block_wise", [True, False])
def test_wait_for_save_dumps_each_block_once_across_tp_times_pcp(
    tp, pcp, wa_block_wise
):
    owner_size = tp * pcp
    fa_keys, wa_keys = [], []
    for owner in range(owner_size):
        submitted, fa_rows, wa_rows = _run_wait_for_save(
            owner, owner_size, wa_block_wise=wa_block_wise
        )
        rank_fa = submitted.get("FA", ([], {}))[0]
        rank_wa = submitted.get("WA", ([], {}))[0]
        assert all(_position(k) % owner_size == owner for k in rank_fa)

        # FA pointer rows stay aligned with the selected keys.
        for keys, vllm_block_ids, hash_positions in fa_rows:
            positions = [_position(k) for k in keys]
            assert list(hash_positions) == positions
            assert list(vllm_block_ids[0]) == [1000 + p for p in positions]
        if wa_block_wise:
            assert rank_wa == rank_fa
            # WA tail blocks are selected per hash block, in key order.
            for keys, vllm_block_ids in wa_rows:
                expected = [
                    2000 + _position(k) * WA_TAIL_BLOCKS + t
                    for k in keys
                    for t in range(WA_TAIL_BLOCKS)
                ]
                assert list(vllm_block_ids[1]) == expected
        fa_keys.extend(rank_fa)
        wa_keys.extend(rank_wa)

    assert sorted(fa_keys) == ALL_KEYS
    if wa_block_wise:
        assert sorted(wa_keys) == ALL_KEYS
    else:
        # Chunk-wise WA dumps only each request's final boundary, exactly once.
        expected = sorted(r.dump_keys[-1] for r in _requests().values() if r.dump_keys)
        assert sorted(wa_keys) == expected


def test_single_owner_dumps_everything():
    submitted, _, _ = _run_wait_for_save(0, 1, wa_block_wise=True)
    assert sorted(submitted["FA"][0]) == ALL_KEYS
    assert sorted(submitted["WA"][0]) == ALL_KEYS
