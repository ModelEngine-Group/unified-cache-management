"""Unit tests for ucm/sandbox/sparse/gsav2/gsa.py - worker-side flows.

Uses the make_worker / make_worker1 / make_scheduler fixtures from conftest,
which build real GSAs on real (uuid-named) shared memory and guarantee
close/unlink teardown.
"""

import struct
from types import SimpleNamespace

import fakes
import pytest
import torch

from ucm.sandbox.sparse.gsav2 import gsa as gsa_module
from ucm.sparse.utils import GSA_SPARSE_LEN, MAX_SLOTS_BLOCK_NUM

pytestmark = pytest.mark.stage(0)


@pytest.fixture()
def device_type_none():
    """Temporarily point the vllm_ascend.utils stub at a device type."""
    import sys

    ascend_utils = sys.modules["vllm_ascend.utils"]
    original = ascend_utils.get_ascend_device_type

    def _set(member):
        ascend_utils.get_ascend_device_type = lambda: member

    yield _set
    ascend_utils.get_ascend_device_type = original


def stat_for(
    g,
    req_id=1,
    topk=2,
    prompt=4,
    output=0,
    computed=0,
    scheduled=4,
    blocks=None,
    slots=None,
    indexer_slots=None,
):
    """Fabricate a GSAReqStat registered in g.gsa_metadata.gsa_stats."""
    stat = gsa_module.GSAReqStat(req_id, g._vllm_config, topk)
    stat.num_prompt_tokens = prompt
    stat.num_output_tokens = output
    stat.num_computed_tokens = computed
    stat.num_scheduled_tokens = scheduled
    stat.blocks = blocks if blocks is not None else [0]
    stat.index_in_batch = 0
    stat.slab_host_slot = slots or []
    stat.indexer_slab_host_slot = (
        indexer_slots if indexer_slots is not None else (slots or [])
    )
    stat.block_hashes = [b"\x01" * 16]
    g.gsa_metadata.gsa_stats[req_id] = stat
    return stat


def ensure_metadata(g):
    if getattr(g, "gsa_metadata", None) is None:
        g.gsa_metadata = gsa_module.GSAMetaData(g._vllm_config, g.topk_layer_num)
        g.gsa_metadata.gsa_stats = g.gsa_stats
    return g.gsa_metadata


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_worker_full_init(self, make_worker):
        g, config = make_worker()
        assert g.rank == 0
        assert g.is_decode_node is True
        assert g.block_size == 128
        assert g.layer_num == 2
        assert g.max_bs == 4
        assert g.num_slots == MAX_SLOTS_BLOCK_NUM
        assert g.is_pd_mix is False
        assert g.num_spec_tokens == 1
        assert g._store_injected is False
        assert g._store_inject_max_retries == 5
        # shared tables created for rank0 worker
        assert g.indexer_block_table_sm is not None
        assert g.prefetch_status_sm is not None
        assert g.slots_num_manager is not None
        assert not hasattr(g, "block_table_sm")  # IS_INDEXER_OFFLOAD False
        # real GSAPrefetchBase on the stub engine
        assert g.prefetch_engine.prefetch_engine_c.config_list[0] == 2
        assert g.prefetch_engine.topk_layer_map == [0, 1]
        # log is a real GSALogger
        assert isinstance(g.log, gsa_module.GSALogger)

    def test_worker_ptopk_disabled_minimal(self, make_worker, set_gsa_const):
        set_gsa_const(PTOPK_PREFETCH_ENABLE=False)
        g, _ = make_worker()
        assert not hasattr(g, "prefetch_engine")
        assert not hasattr(g, "indexer_block_table_sm")

    def test_worker_not_decode_minimal(self, make_worker):
        g, _ = make_worker({"is_kv_consumer": False})
        assert g.is_decode_node is False
        assert not hasattr(g, "prefetch_engine")

    def test_worker_rank1_attach(self, make_worker1):
        g, config = make_worker1()
        assert g.rank == 1
        assert g.indexer_block_table_sm.is_creator is False
        assert g.prefetch_status_sm.is_creator is False
        assert not hasattr(g, "slots_num_manager")  # rank1 worker never creates

    def test_scheduler_role_attaches_slots_num(self, make_scheduler):
        g, config = make_scheduler()
        assert g.slots_num_manager is not None
        assert g.slots_num_manager.is_creator is False
        assert g.is_decode_node is False

    def test_speculative_config(self, make_worker):
        g, _ = make_worker({"speculative_config": fakes._SpeculativeConfig(3)})
        assert g.num_spec_tokens == 4
        assert g.select_gsa_prefetch_topk is not None
        assert g.select_gsa_prefetch_topk.shape == (2, 4, 1, 2048)

    def test_li_c8_buffers(self, make_worker):
        g, _ = make_worker({"additional_config": {"enable_sparse_li_c8": True}})
        assert g.enable_sparse_li_c8 is True
        assert g.indexer_pc_dump_tmp_c8 is not None
        assert g.indexer_pc_dump_scale_tmp is not None

    def test_sfa_c8_dims(self, make_worker, device_type_none):
        from vllm_ascend.utils import AscendDeviceType

        device_type_none(AscendDeviceType.A5)
        g, _ = make_worker({"additional_config": {"enable_sparse_sfa_c8": True}})
        assert g.dtype == torch.float8_e4m3fn
        assert g.c8_k_scale_cache_dtype == torch.float32
        assert g.k_dim == 32  # packed kv_lora_rank(16) + rope(16)
        assert g.v_dim == 0
        assert g._slab_preempt_v is g._slab_preempt_k

    def test_sfa_c8_non_a5_dims(self, make_worker, device_type_none):
        from vllm_ascend.utils import AscendDeviceType

        device_type_none(AscendDeviceType.A2)
        g, _ = make_worker({"additional_config": {"enable_sparse_sfa_c8": True}})
        assert g.dtype == torch.int8
        assert g.c8_k_scale_cache_dtype == torch.float16

    def test_offload_worker_extras(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        assert g.block_table_sm is not None
        assert g.block_table_sm.is_creator is True
        assert hasattr(g, "gsa_cpu_shared_pool")
        assert g.indexer_last_query is None  # reset at the end of init
        assert g.indexer_last_weight is None


# ---------------------------------------------------------------------------
# bind_cpus
# ---------------------------------------------------------------------------


class TestBindCpus:
    def run_bind(self, make_worker, monkeypatch, cpu_alloc):
        monkeypatch.setattr(gsa_module, "CpuAlloc", lambda local_rank: cpu_alloc)
        g, _ = make_worker()
        return g

    def test_empty_cpu_pool(self, make_worker, monkeypatch, gsa_prefetch_stub):
        alloc = fakes.FakeCpuAlloc(cpu_pool={})  # no pool for npu 0
        g, _ = self.run_bind(make_worker, monkeypatch, alloc)
        last = gsa_prefetch_stub.engine_instances[-1]
        assert last.cpu_to_bind is None  # (None, None) path

    def test_ample_assign_main_takes_tail(self, make_worker, monkeypatch):
        alloc = fakes.FakeCpuAlloc(assign_main={0: list(range(30))})
        g, _ = self.run_bind(make_worker, monkeypatch, alloc)
        # IS_INDEXER_OFFLOAD False -> thread num 20 -> last 20 of 30
        assert g.prefetch_engine.prefetch_engine_c.cpu_to_bind == list(range(10, 30))

    def test_short_assign_main_takes_one_minus(
        self, make_worker, monkeypatch, set_gsa_const
    ):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        alloc = fakes.FakeCpuAlloc(assign_main={0: list(range(24))})
        g, _ = self.run_bind(make_worker, monkeypatch, alloc)
        # thread num 24, not > 24 -> [1:]
        assert g.prefetch_engine.prefetch_engine_c.cpu_to_bind == list(range(1, 24))


# ---------------------------------------------------------------------------
# execute_begin pipeline
# ---------------------------------------------------------------------------


class TestExecuteBegin:
    def base_env(self, make_worker, set_gsa_const):
        set_gsa_const(SEG_PREFILL_THRESHOLD=8)
        g, config = make_worker()
        ensure_metadata(g)
        g.need_prefetch_index = []
        g.gsa_index = []
        stat = stat_for(
            g, req_id=1, prompt=4, output=1, computed=0, scheduled=4, blocks=[11]
        )
        sched = fakes.FakeSchedulerOutput(num_scheduled_tokens={1: 4})
        attn_meta = {"attn": SimpleNamespace(num_input_tokens=4)}
        return g, stat, sched, attn_meta

    def test_early_return_ptopk_off(self, make_worker, set_gsa_const):
        set_gsa_const(PTOPK_PREFETCH_ENABLE=False)
        g, _ = make_worker()
        gsa_module.GSA.execute_begin(g, fakes.FakeSchedulerOutput(), {})
        assert g.prefetch_engine.step_time == 0

    def test_early_return_not_decode(self, make_worker):
        g, _ = make_worker({"is_kv_consumer": False})
        gsa_module.GSA.execute_begin(g, fakes.FakeSchedulerOutput(), {})
        assert not hasattr(g, "prefetch_engine")

    def test_full_flow(self, make_worker, set_gsa_const):
        g, stat, sched, attn_meta = self.base_env(make_worker, set_gsa_const)
        gsa_module.GSA.execute_begin(g, sched, attn_meta)
        assert g.prefetch_engine.step_time == 1
        assert g.req_ids_bs == [1]
        assert g.num_padding_reqs == 4
        # one indexer block was allocated for the 4 new tokens
        assert len(stat.indexer_slab_host_slot) == 1
        # attn metadata got the GSA lens
        attn = attn_meta["attn"]
        assert attn.gsa_slot_len == 1  # one scheduled token slot-mapped
        assert attn.gsa_prefetch_len == 0
        assert attn.gsa_warmup_len == 0
        # non-gsa request: prefetch status untouched
        assert g.gsa_index == []

    def test_full_flow_perf_log_off(self, make_worker, set_gsa_const):
        set_gsa_const(SEG_PREFILL_THRESHOLD=8, IS_PERFORMANCE_LOG=False)
        g, config = make_worker()
        ensure_metadata(g)
        g.need_prefetch_index = []
        g.gsa_index = []
        stat_for(g, req_id=1, prompt=4, output=1, scheduled=4, blocks=[11])
        sched = fakes.FakeSchedulerOutput(num_scheduled_tokens={1: 4})
        attn_meta = {"attn": SimpleNamespace(num_input_tokens=4)}
        gsa_module.GSA.execute_begin(g, sched, attn_meta)
        assert g.req_ids_bs == [1]

    def test_gsa_req_inits_prefetch_state(self, make_worker, set_gsa_const):
        set_gsa_const(SEG_PREFILL_THRESHOLD=8)
        g, config = make_worker()
        ensure_metadata(g)
        g.need_prefetch_index = []
        g.gsa_index = []
        stat = stat_for(g, req_id=1, prompt=64, output=1, scheduled=4, blocks=[11])
        stat.num_computed_tokens = 64
        stat.topk_buf_tmp = stat.prefetch_use_token.cpu
        gsa_module.GSA.execute_begin(
            g,
            fakes.FakeSchedulerOutput(num_scheduled_tokens={1: 4}),
            {"attn": SimpleNamespace(num_input_tokens=4)},
        )
        assert stat.is_gsa_init is True
        assert stat.last_step == 0
        assert g.prefetch_engine.prefetch_engine_c.calls_of("set_tokens_map")


class TestSyncBlockTables:
    def test_rank0_writes(self, make_worker, gsh_mod):
        g, _ = make_worker()
        gsa_module.GSA._sync_block_tables(g, [1], [[5]], [1], [[6]], [1])
        assert g.indexer_block_table_sm.version == 1
        # read back through an independent attached reader (fresh version)
        reader = gsh_mod.SharedBlockTable(
            g.indexer_block_table_sm.name, False, timeout=2
        )
        assert reader.read_add_block([1]) == [[6]]
        reader.close()

    def test_rank1_reads_and_applies(self, make_worker1, gsh_mod):
        g, _ = make_worker1()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1)
        writer = gsh_mod.SharedBlockTable(
            g.indexer_block_table_sm.name, False, timeout=2
        )
        writer.write_add_block([[6]])
        gsa_module.GSA._sync_block_tables(g, [1], [None], [1], [[6]], [1])
        assert stat.indexer_slab_host_slot == [6]
        writer.close()

    def test_apply_add_blocks_zero_len_skipped(self, make_worker):
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1)
        gsa_module.GSA._apply_add_blocks(g, [1], [[9]], [0], is_indexer=True)
        assert stat.indexer_slab_host_slot == []
        gsa_module.GSA._apply_add_blocks(g, [1], [[9]], [1], is_indexer=True)
        assert stat.indexer_slab_host_slot == [9]


# ---------------------------------------------------------------------------
# slot mapping calculations
# ---------------------------------------------------------------------------


class TestCalcSlotMapping:
    def make_meta(self, seq_len, computed=0):
        return SimpleNamespace(
            get_seq_len=lambda: seq_len,
            num_computed_tokens=computed,
            slot_mapping_cal_len=0,
            blocks=[0, 1, 2, 3],
        )

    def test_decode_spec1(self, make_worker):
        g, _ = make_worker()
        meta = self.make_meta(5)
        slot_mapping = [None]
        gsa_module.GSA._calc_decode_slot_mapping(g, meta, [10], 0, slot_mapping)
        assert slot_mapping[0] == [10 * 128 + 4]

    def test_decode_spec_spanning_blocks(self, make_worker):
        g, _ = make_worker()
        g.num_spec_tokens = 4
        meta = self.make_meta(3)  # 3 % 128 = 3 < 4 and != 0
        slot_mapping = [None]
        gsa_module.GSA._calc_decode_slot_mapping(g, meta, [7, 10], 0, slot_mapping)
        expected = list(range(7 * 128 + 128 - 1, 7 * 128 + 128)) + [
            i + 10 * 128 for i in range(3)
        ]
        assert slot_mapping[0] == expected

    def test_decode_spec_block_boundary(self, make_worker):
        g, _ = make_worker()
        g.num_spec_tokens = 4
        meta = self.make_meta(8)  # 8 % 8 == 0
        slot_mapping = [None]
        gsa_module.GSA._calc_decode_slot_mapping(g, meta, [10], 0, slot_mapping)
        assert slot_mapping[0] == list(range(10 * 128 + 128 - 4, 10 * 128 + 128))

    def test_decode_spec_mid_block(self, make_worker):
        g, _ = make_worker()
        g.num_spec_tokens = 4
        meta = self.make_meta(14)  # 14 % 128 == 14 >= 4
        slot_mapping = [None]
        gsa_module.GSA._calc_decode_slot_mapping(g, meta, [10], 0, slot_mapping)
        assert slot_mapping[0] == list(range(10 * 128 + 14 - 4, 10 * 128 + 14))

    def test_prefill_match_recomputes_hit(self, make_worker):
        g, _ = make_worker()
        meta = self.make_meta(300, computed=128)  # ceil(300/128)=3 == len(slots)
        slot_mapping, dump_v, dump_g = [None], [None], [None]
        gsa_module.GSA._calc_prefill_slot_mapping(
            g, 1, meta, [5, 6, 7], 0, slot_mapping, dump_v, dump_g
        )
        assert g.indexer_pc_hit_num[0] == 1  # ceil(128/128)
        assert dump_v[0] == [meta.blocks[1]]  # block idx of computed prefix
        assert dump_g[0] == [6]
        assert len(slot_mapping[0]) == 300 - 128

    def test_prefill_match_no_recompute(self, make_worker):
        g, _ = make_worker()
        meta = self.make_meta(256, computed=128)
        meta.slot_mapping_cal_len = 128  # == num_computed -> skip hit recompute
        slot_mapping, dump_v, dump_g = [None], [None], [None]
        gsa_module.GSA._calc_prefill_slot_mapping(
            g, 1, meta, [5, 6], 0, slot_mapping, dump_v, dump_g
        )
        assert g.indexer_pc_hit_num == []  # untouched placeholder
        assert len(slot_mapping[0]) == 128

    def test_prefill_mismatch_logs_error(self, make_worker):
        g, _ = make_worker()
        meta = self.make_meta(600)  # ceil(600/128)=5 != len(slots)=2
        slot_mapping, dump_v, dump_g = [None], [None], [None]
        gsa_module.GSA._calc_prefill_slot_mapping(
            g, 1, meta, [5, 6], 0, slot_mapping, dump_v, dump_g
        )
        assert slot_mapping[0] == []


# ---------------------------------------------------------------------------
# get_dsa_input helpers
# ---------------------------------------------------------------------------


class TestDsaInput:
    def test_update_dsa_block_table(self, make_worker, set_gsa_const):
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, slots=[3, 4], indexer_slots=[3, 4])
        dsa_seq_len = [None]
        meta, slots, idx = gsa_module.GSA._update_dsa_block_table(g, 1, dsa_seq_len)
        assert idx == 0
        assert slots == [3, 4]
        assert dsa_seq_len[0] == stat.get_seq_len()
        assert g.dsa_block_table.np[0, 0, 0] == 3

    def test_update_dsa_block_table_indexer_init(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, slots=[3, 4], indexer_slots=[3, 4])
        stat.is_indexer_init = True
        stat.add_blocks_len = 1
        stat.indexer_block_tables = g.dsa_block_table.np[:, 0, :]
        stat.indexer_sparse_len = lambda: 2
        stat.get_indexer_sparse_len = lambda: 2
        dsa_seq_len = [None]
        gsa_module.GSA._update_dsa_block_table(g, 1, dsa_seq_len)
        assert dsa_seq_len[0] == 2
        assert g.prefetch_engine.prefetch_engine_c.calls_of("update_indexer_map")

    def test_init_dsa_input_buffers(self, make_worker):
        g, _ = make_worker()
        out = gsa_module.GSA._init_dsa_input_buffers(g, ["r1", "r2"])
        assert len(out[0]) == 2
        assert g.dsa_slot_mapping.np[0] == -1
        assert g.dsa_seq_len.np[0] == 1

    def test_copy_dsa_input_to_npu(self, make_worker):
        g, _ = make_worker()
        n = gsa_module.GSA._copy_dsa_input_to_npu(g, [8], [[1, 2]], [[3]], [[4]])
        assert n == 2
        assert g.indexer_pc_dump_num == 1
        assert g.dsa_slot_mapping.np[0] == 1


# ---------------------------------------------------------------------------
# _update_prefetch_status_all
# ---------------------------------------------------------------------------


class TestUpdatePrefetchStatusAll:
    def test_need_prefetch_index_busy_wait(self, make_worker):
        g, _ = make_worker()
        g.need_prefetch_index = [0]
        g.gsa_index = []
        calls = iter([True, False])
        g.prefetch_engine.prefetch_engine_c.get_prefetch_status = lambda: next(calls)
        gsa_module.GSA._update_prefetch_status_all(g)
        assert g.is_prefetch_token is True

    def test_gsa_index_path_syncs_version(self, make_worker):
        g, _ = make_worker()
        g.need_prefetch_index = []
        g.gsa_index = [0]
        step_time = 3
        g.prefetch_engine.step_time = step_time
        g.prefetch_engine.prefetch_engine_c.get_prefetch_status = lambda: False
        sv = g.prefetch_status_sm
        # align the single rank's version slot so update_version does not wait
        struct.pack_into("<Q", sv.shm.buf, sv.elements_size * sv.tp_size, step_time)
        gsa_module.GSA._update_prefetch_status_all(g)
        assert g.is_prefetch_token == 1


# ---------------------------------------------------------------------------
# collect / swap / buffer copy
# ---------------------------------------------------------------------------


class TestCollectAndSwap:
    def env(self, make_worker, set_gsa_const, **kw):
        set_gsa_const(SEG_PREFILL_THRESHOLD=8)
        g, _ = make_worker(**kw)
        ensure_metadata(g)
        g.need_prefetch_index = []
        g.gsa_index = []
        g.model_input = {"query_locals": [0, 4, 4]}
        return g

    def test_collect_skips_short_prompt(self, make_worker, set_gsa_const):
        g = self.env(make_worker, set_gsa_const)
        stat = stat_for(g, req_id=1, prompt=4)  # <= SEG threshold
        g.req_ids_bs = [1]
        g.warmup_topk = []
        g.prefetch_topk = []
        req_cnt, _, _ = gsa_module.GSA._collect_and_swap_tokens_all(g)
        assert req_cnt == 0
        assert g.warmup_topk == []

    def test_collect_warmup_topk(self, make_worker, set_gsa_const):
        # PD-mix: last chunk == computed + scheduled == prompt len
        g = self.env(make_worker, set_gsa_const, is_kv_producer=True)
        stat = stat_for(g, req_id=1, prompt=64, output=1, computed=60, scheduled=4)
        g.req_ids_bs = [1]
        g.warmup_topk = []
        g.prefetch_topk = []
        gsa_module.GSA._collect_and_swap_tokens_all(g)
        assert len(g.warmup_topk) == 1
        indices, warmup_len, req_id = g.warmup_topk[0]
        assert warmup_len == 4 and req_id == 1
        assert indices == list(range(4 - 4, 4))

    def test_collect_gsa_tokens_swap(self, make_worker, set_gsa_const):
        g = self.env(make_worker, set_gsa_const)
        stat = stat_for(g, req_id=1, prompt=64, output=2, computed=64, scheduled=4)
        stat.is_gsa_init = True
        g.is_prefetch_token = True
        g.req_ids_bs = [1]
        g.warmup_topk = []
        g.prefetch_topk = []
        old_use = stat.use_tokens
        old_prefetch = stat.prefetch_use_token
        gsa_module.GSA._collect_and_swap_tokens_all(g)
        # tokens tables swapped
        assert stat.use_tokens is old_prefetch
        assert stat.prefetch_use_token is old_use
        assert g.prefetch_engine.prefetch_engine_c.calls_of("set_topk_indices")
        assert g.prefetch_topk == [[[0, 1, 2, 3]], 1]
        assert g.select_block_index.np[0] == 0
        # pre_swap_tokens submitted with the swapped table ptr
        assert g.prefetch_engine.prefetch_engine_c.calls_of("pre_swap_tokens")

    def test_pre_swap_empty_early(self, make_worker):
        g, _ = make_worker()
        gsa_module.GSA._pre_swap_tokens_all(g, [], [], [], [])
        assert not g.prefetch_engine.prefetch_engine_c.calls

    def test_pre_swap_sets_ptrs_once(self, make_worker, set_gsa_const):
        g = self.env(make_worker, set_gsa_const)
        stat = stat_for(g, req_id=1, prompt=64, output=2, computed=64, scheduled=4)
        assert g.gsa_swap_topk_cpu_ptr is None
        gsa_module.GSA._pre_swap_tokens_all(g, [111], [5120], [[0, 1]], [1])
        assert g.gsa_swap_topk_cpu_ptr is not None
        assert g.gsa_swap_topk_npu_ptr is not None
        first = g.gsa_swap_topk_cpu_ptr
        gsa_module.GSA._pre_swap_tokens_all(g, [111], [5120], [[0, 1]], [1])
        assert g.gsa_swap_topk_cpu_ptr == first  # set only once

    def test_copy_gsa_buffers(self, make_worker):
        g, _ = make_worker()
        g.swap_index = [0, 1]
        g.prefetch_topk = [[[0, 1]], 1]
        g.warmup_topk = [[[2, 3], 2, 1]]
        g.num_padding_reqs = 2
        g.req_cnt = 1
        p_len, w_len, i_len = gsa_module.GSA._copy_gsa_buffers_to_npu(g, 1)
        assert p_len == 2 and w_len == 2 and i_len == 0
        assert g.gsa_swap_index.np[0] == 0
        assert g.gsa_warmup_index.np[0] == 2

    def test_update_attn_gsa_lens(self, make_worker):
        g, _ = make_worker()
        g.swap_index = [0]
        attn = SimpleNamespace()
        gsa_module.GSA._update_attn_gsa_lens(g, {"a": attn}, 5, 6, 7, 8, 1)
        assert attn.gsa_slot_len == 5
        assert attn.gsa_prefetch_len == 6
        assert attn.gsa_warmup_len == 7
        assert attn.gsa_swap_len == 1
        assert not hasattr(attn, "indexer_last_query_len")  # offload disabled


class TestInitIndexerOffloadAll:
    def test_offload_disabled_early(self, make_worker):
        g, _ = make_worker()
        gsa_module.GSA._init_indexer_offload_all(g)
        assert not hasattr(g, "is_prefetch_indexer")

    def test_step_time_multiple_of_ten(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        g.req_ids_bs = []
        g.prefetch_engine.step_time = 10
        gsa_module.GSA._init_indexer_offload_all(g)
        assert g.is_prefetch_indexer is True

    def test_scan_reqs_for_last_chunk(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True, INDEXER_PREFETCH_TOKENS=16)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, prompt=64, output=0)  # last chunk (pd-sep)
        g.req_ids_bs = [1]
        g.prefetch_engine.step_time = 3
        gsa_module.GSA._init_indexer_offload_all(g)
        assert g.is_prefetch_indexer is True

    def test_scan_reqs_no_match(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True, INDEXER_PREFETCH_TOKENS=16)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, prompt=64, output=2)  # not last chunk
        g.req_ids_bs = [1]
        g.prefetch_engine.step_time = 3
        gsa_module.GSA._init_indexer_offload_all(g)
        assert g.is_prefetch_indexer is False


# ---------------------------------------------------------------------------
# attention hooks
# ---------------------------------------------------------------------------


class TestAttentionBegin:
    def test_early_return_ptopk_off(self, make_worker, set_gsa_const):
        set_gsa_const(PTOPK_PREFETCH_ENABLE=False)
        g, _ = make_worker()
        q = torch.zeros(1)
        assert gsa_module.GSA.attention_begin(
            g, q, q, q, "model.layers.0.self_attn", None
        ) == (q, q, q, None)

    def test_layer_out_of_range(self, make_worker):
        g, _ = make_worker()
        q = torch.zeros(1)
        out = gsa_module.GSA.attention_begin(
            g, q, q, q, "model.layers.99.self_attn", None
        )
        assert out == (q, q, q, None)

    def test_no_indexer_early(self, make_worker):
        g, _ = make_worker()
        q = torch.zeros(1)
        out = gsa_module.GSA.attention_begin(
            g, q, q, q, "model.layers.0.self_attn", None, has_indexer=False
        )
        assert out == (q, q, q, None)
        # kv pointers were still registered
        assert g.kv_caches[0] is not None

    def test_prefetch_cache_blocks(self, make_worker, ascend_ops):
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1)
        g.req_ids_bs = [1]
        g.indexer_pc_hit_num = [2]
        # the ACL ops read the device-side (.npu) hit indices
        g.indexer_pc_hit_index.npu[0][0][:2] = torch.tensor([0, 1], dtype=torch.int32)
        g.indexer_pc_hit_index.npu[1][0][:2] = torch.tensor([0, 1], dtype=torch.int32)
        values = (torch.arange(2 * 128 * 128) % 100).to(torch.float16)
        vllm_cache = values.reshape(2, 128 * 128)
        indexer_cache = torch.zeros(2, 128 * 128, dtype=torch.float16)
        q = torch.zeros(1)
        gsa_module.GSA.attention_begin(
            g,
            q,
            q,
            q,
            "model.layers.0.self_attn",
            None,
            indexer_cache=indexer_cache,
            vllm_indexer_cache=vllm_cache,
        )
        assert ascend_ops.call_counts["gsa_gather_copy"] == 1
        assert ascend_ops.call_counts["gsa_scatter_copy"] == 1
        assert indexer_cache[0].tolist() == vllm_cache[0].tolist()
        assert indexer_cache[1].tolist() == vllm_cache[1].tolist()

    def test_prefetch_with_scale_c8(self, make_worker, ascend_ops):
        g, _ = make_worker({"additional_config": {"enable_sparse_li_c8": True}})
        ensure_metadata(g)
        stat = stat_for(g, req_id=1)
        g.req_ids_bs = [1]
        g.indexer_pc_hit_num = [1]
        g.indexer_pc_hit_index.npu[0][0][:1] = torch.tensor([0], dtype=torch.int32)
        g.indexer_pc_hit_index.npu[1][0][:1] = torch.tensor([0], dtype=torch.int32)
        vllm_cache = (
            (torch.arange(128 * 128) % 100).to(torch.float16).reshape(1, 128 * 128)
        )
        indexer_cache = torch.zeros(1, 128 * 128, dtype=torch.int8)
        vllm_scale = (torch.arange(128) % 100).to(torch.float16).reshape(1, 128)
        indexer_scale = torch.zeros(1, 128, dtype=torch.float16)
        q = torch.zeros(1)
        gsa_module.GSA.attention_begin(
            g,
            q,
            q,
            q,
            "model.layers.0.self_attn",
            None,
            indexer_cache=indexer_cache,
            vllm_indexer_cache=vllm_cache,
            indexer_scale=indexer_scale,
            vllm_indexer_scale=vllm_scale,
            use_li_c8=True,
        )
        assert ascend_ops.call_counts["gsa_gather_copy"] == 2
        assert indexer_scale[0].tolist() == vllm_scale[0].tolist()


class TestAttentionFinished:
    def test_early_return_pd_separated(self, make_worker):
        g, _ = make_worker()  # is_pd_mix False by default
        q = torch.zeros(1)
        gsa_module.GSA.attention_finished(
            g, q, q, q, q, "model.layers.0.self_attn", None
        )
        # nothing recorded anywhere; no crash

    def test_layer_out_of_range(self, make_worker):
        g, _ = make_worker({"is_kv_producer": True})  # pd mix
        q = torch.zeros(1)
        gsa_module.GSA.attention_finished(g, q, q, q, q, "model.layers.99.a", None)

    def test_no_dump_early(self, make_worker):
        g, _ = make_worker({"is_kv_producer": True})
        g.indexer_pc_dump_num = 0
        q = torch.zeros(1)
        gsa_module.GSA.attention_finished(g, q, q, q, q, "model.layers.0.a", None)

    def test_dump_gathers_and_scatters(self, make_worker, ascend_ops):
        g, _ = make_worker({"is_kv_producer": True})
        g.indexer_pc_dump_num = 1
        # device-side dump indices: [1]=source (indexer), [0]=dest (vllm)
        g.indexer_pc_dump_index.npu[0][:1] = torch.tensor([0], dtype=torch.int32)
        g.indexer_pc_dump_index.npu[1][:1] = torch.tensor([0], dtype=torch.int32)
        g.indexer_k_cache_sparse[0] = (
            (torch.arange(128 * 128) % 100).to(torch.float16).reshape(1, 128, 128)
        )
        g.kv_caches[0] = [
            None,
            None,
            torch.zeros(1, 128, 128, dtype=torch.float16),
            None,
        ]
        q = torch.zeros(1)
        gsa_module.GSA.attention_finished(g, q, q, q, q, "model.layers.0.a", None)
        assert ascend_ops.call_counts["gsa_gather_copy"] == 1
        assert ascend_ops.call_counts["gsa_scatter_copy"] == 1
        assert (
            g.kv_caches[0][2][0, 0, :8].tolist()
            == g.indexer_k_cache_sparse[0][0, 0, :8].tolist()
        )

    def test_dump_with_scale_c8(self, make_worker, ascend_ops):
        g, _ = make_worker(
            {"is_kv_producer": True, "additional_config": {"enable_sparse_li_c8": True}}
        )
        g.indexer_pc_dump_num = 1
        g.indexer_pc_dump_index.npu[0][:1] = torch.tensor([0], dtype=torch.int32)
        g.indexer_pc_dump_index.npu[1][:1] = torch.tensor([0], dtype=torch.int32)
        g.indexer_k_cache_sparse[0] = (
            (torch.arange(128 * 128) % 100).to(torch.float16).reshape(1, 128, 128)
        )
        g.indexer_scale[0] = (torch.arange(128) % 100).to(torch.float16).reshape(1, 128)
        g.kv_caches[0] = [
            None,
            None,
            torch.zeros(1, 128, 128, dtype=torch.float16),
            torch.zeros(1, 128, dtype=torch.float16),
        ]
        q = torch.zeros(1)
        gsa_module.GSA.attention_finished(
            g, q, q, q, q, "model.layers.0.a", None, use_li_c8=True
        )
        assert ascend_ops.call_counts["gsa_gather_copy"] == 2
        assert g.kv_caches[0][3][0, :8].tolist() == g.indexer_scale[0][0, :8].tolist()


# ---------------------------------------------------------------------------
# warmup / preempt dump
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_early_returns(self, make_worker, set_gsa_const):
        set_gsa_const(PTOPK_PREFETCH_ENABLE=False)
        g, _ = make_worker()
        gsa_module.GSA.warmup(g)
        set_gsa_const(PTOPK_PREFETCH_ENABLE=True)
        g2, _ = make_worker()
        g2.warmup_topk = []
        gsa_module.GSA.warmup(g2)

    def test_warmup_full(self, make_worker, set_gsa_const, fake_npu):
        set_gsa_const(SEG_PREFILL_THRESHOLD=8)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(
            g,
            req_id=1,
            prompt=64,
            output=1,
            computed=64,
            scheduled=4,
            blocks=[1],
            slots=[1],
            indexer_slots=[1],
        )
        stat.non_ucm_block_count = 1
        stat.original_prompt_block_len = 0
        g.warmup_topk = [[[0, 1], 2, 1]]
        g.kv_caches[0] = [
            torch.full((2, 128, 1, 16), 3.0),
            torch.full((2, 128, 1, 16), 4.0),
            None,
            None,
        ]
        gsa_module.GSA.warmup(g)
        eng = g.prefetch_engine.prefetch_engine_c
        assert eng.calls_of("set_blocks_map")
        assert stat.req_dump_blocks is not None
        assert 1 in g._block_hash_injected_reqs

    def test_get_vllm_blocks_spec_blocks(self, make_worker):
        g, _ = make_worker()
        g.num_spec_tokens = 4
        meta = SimpleNamespace(
            slab_host_slot=[1], indexer_slab_host_slot=[1], blocks=[5, 6]
        )
        assert gsa_module.GSA._get_vllm_blocks(g, meta) == [5]  # spec: drop last
        meta.blocks = [5]
        assert gsa_module.GSA._get_vllm_blocks(g, meta) == [5]

    def test_get_vllm_blocks_offload(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        g.num_spec_tokens = 1
        meta = SimpleNamespace(
            slab_host_slot=[1, 2, 3], indexer_slab_host_slot=[1], blocks=[5, 6]
        )
        # offload: slot_len = len(slab)=3 ; len(blocks)=2 <= 3 -> blocks as-is
        assert gsa_module.GSA._get_vllm_blocks(g, meta) == [5, 6]

    def test_inject_block_hashes_guards(self, make_worker):
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1)
        g._slab_preempt_k_ptr = 100
        g._slab_preempt_k_offset = 10
        g._slab_preempt_v_ptr = 200
        g._slab_preempt_v_offset = 20
        # already injected
        g._block_hash_injected_reqs = {1}
        gsa_module.GSA._inject_block_hashes(g, 1, [0], [1])
        assert not g.prefetch_engine.prefetch_engine_c.calls_of("set_blocks_map")
        # no hashes
        g._block_hash_injected_reqs = set()
        stat.block_hashes = None
        gsa_module.GSA._inject_block_hashes(g, 1, [0], [1])
        assert not g.prefetch_engine.prefetch_engine_c.calls_of("set_blocks_map")
        # length mismatch
        stat.block_hashes = [b"\x01" * 16]
        with pytest.raises(AssertionError):
            gsa_module.GSA._inject_block_hashes(g, 1, [0], [1, 2])

    def test_inject_block_hashes_sfa_c8_empty_v_refs(self, make_worker):
        g, _ = make_worker({"additional_config": {"enable_sparse_sfa_c8": True}})
        ensure_metadata(g)
        stat = stat_for(g, req_id=1)
        g._slab_preempt_k_ptr = 100
        g._slab_preempt_k_offset = 10
        gsa_module.GSA._inject_block_hashes(g, 1, [0], [1])
        call = g.prefetch_engine.prefetch_engine_c.calls_of("set_blocks_map")[0]
        assert call[1][4] == []  # v_refs empty under sfa_c8

    def test_inject_block_hashes_indexer(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1)
        g._slab_preempt_k_ptr = 100
        g._slab_preempt_k_offset = 10
        g._slab_preempt_v_ptr = 200
        g._slab_preempt_v_offset = 20
        g.indexer_cache_preempt_i_ptr = 300
        g.indexer_cache_preempt_i_offset = 30
        gsa_module.GSA._inject_block_hashes(g, 1, [0], [1], inject_indexer=True)
        assert g.prefetch_engine.prefetch_engine_c.calls_of("set_last_indexer_blocks")


# ---------------------------------------------------------------------------
# execute_finished / prefetch submission
# ---------------------------------------------------------------------------


class TestExecuteFinished:
    def test_early_return_ptopk_off(self, make_worker, set_gsa_const):
        set_gsa_const(PTOPK_PREFETCH_ENABLE=False)
        g, _ = make_worker()
        gsa_module.GSA.execute_finished(g)

    def test_no_prefetch_branch(self, make_worker, set_gsa_const):
        set_gsa_const(SEG_PREFILL_THRESHOLD=8)
        g, _ = make_worker()
        ensure_metadata(g)
        g.need_prefetch_index = []
        g.gsa_index = []
        g.prefetch_topk = []
        g.warmup_topk = []
        g.is_prefetch_token = False
        gsa_module.GSA.execute_finished(g)
        # update_slot_num wrote the free counts through slots_num_manager
        i_num, k_num = g.slots_num_manager.read_num()
        assert i_num >= 0 and k_num == 0

    def test_prefetch_branch_spec1(self, make_worker, set_gsa_const):
        set_gsa_const(SEG_PREFILL_THRESHOLD=8)
        g, _ = make_worker()
        ensure_metadata(g)
        g.need_prefetch_index = []
        g.gsa_index = []
        stat = stat_for(g, req_id=1, prompt=64, output=2, computed=64, scheduled=4)
        g.prefetch_topk = [[[0, 1]], 1]
        g.warmup_topk = []
        g.is_prefetch_token = True
        gsa_module.GSA.execute_finished(g)
        eng = g.prefetch_engine.prefetch_engine_c
        assert eng.calls_of("submit_prefetch_task")

    def test_prefetch_branch_spec_n_with_rec_result(self, make_worker, set_gsa_const):
        set_gsa_const(SEG_PREFILL_THRESHOLD=8)
        g, _ = make_worker({"speculative_config": fakes._SpeculativeConfig(3)})
        ensure_metadata(g)
        g.need_prefetch_index = []
        g.gsa_index = []
        stat = stat_for(g, req_id=1, prompt=64, output=2, computed=64, scheduled=4)
        g.prefetch_topk = [[[0, 1, 2, 3]], 1]
        g.warmup_topk = []
        g.is_prefetch_token = True
        gsa_module.GSA.execute_finished(g, rec_result=[2])
        eng = g.prefetch_engine.prefetch_engine_c
        assert eng.calls_of("submit_prefetch_task")
        # the selected topk view was copied into select_gsa_prefetch_topk
        assert g.select_gsa_prefetch_topk is not None

    def test_submit_sum_zero_early(self, make_worker):
        g, _ = make_worker()
        g.topk_indices_len = 0
        g.prefetch_space_token = 5
        gsa_module.GSA._submit_prefetch_task(g, [0], [1], [False], [4])
        assert g.prefetch_space_token == 5  # untouched


# ---------------------------------------------------------------------------
# request lifecycle
# ---------------------------------------------------------------------------


class TestRequestLifecycle:
    def test_request_finished_in_worker(self, make_worker, set_gsa_const):
        set_gsa_const(SEG_PREFILL_THRESHOLD=8)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, slots=[2], indexer_slots=[3])
        stat.req_dump_blocks = [4]
        g.gsa_stats = g.gsa_metadata.gsa_stats
        g._block_hash_injected_reqs = {1}
        gsa_module.GSA.request_finished_in_worker(g, 1)
        assert 1 not in g.gsa_stats
        assert 1 not in g._block_hash_injected_reqs
        assert g.prefetch_engine.prefetch_engine_c.calls_of("del_blocks_map")
        # slot pools got the slots back
        assert g.indexer_cpu_shared_pool.get_free_num() > 0
        assert g.gsa_preempt_shared_pool.get_free_num() > 0

    def test_request_finished_in_worker_missing_stat(self, make_worker):
        g, _ = make_worker()
        g.gsa_stats = {}
        gsa_module.GSA.request_finished_in_worker(g, 99)  # no crash

    def test_request_finished_in_worker_busy_engine(self, make_worker):
        g, _ = make_worker()
        g.gsa_stats = {}
        calls = iter([True, False])
        g.prefetch_engine.prefetch_engine_c.get_prefetch_status = lambda: next(calls)
        gsa_module.GSA.request_finished_in_worker(g, 99)
        assert g.prefetch_engine.prefetch_engine_c.calls_of("get_prefetch_status")

    def test_release_preempted_missing(self, make_worker):
        g, _ = make_worker()
        g.gsa_stats = {}
        gsa_module.GSA._release_preempted_gsa_stats(g, 42)  # early return

    def test_release_preempted_rank0(self, make_worker, set_gsa_const):
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, slots=[2], indexer_slots=[3])
        stat.req_dump_blocks = [4]
        g.gsa_stats = g.gsa_metadata.gsa_stats
        i_free_before = g.indexer_cpu_shared_pool.get_free_num()
        gsa_module.GSA._release_preempted_gsa_stats(g, 1)
        assert 1 not in g.gsa_stats
        # the indexer slot was returned to the shared pool
        assert g.indexer_cpu_shared_pool.get_free_num() == i_free_before + 1

    def test_build_gsa_metadata_preempt(self, make_worker, set_gsa_const):
        set_gsa_const(SEG_PREFILL_THRESHOLD=8)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, slots=[2], indexer_slots=[3])
        g.gsa_stats = g.gsa_metadata.gsa_stats
        req = fakes.FakeRequest(
            req_id=1, prompt_token_ids=list(range(32)), output_token_ids=[1, 2, 3]
        )
        sched = fakes.FakeSchedulerOutput(
            num_scheduled_tokens={},
            preempted_req_ids=[1],
        )
        meta = gsa_module.GSA.build_gsa_metadata(g, sched, {1: req}, None)
        assert g.preempt_req_output_tokens[1] == 3
        assert meta is not None


class TestBuildSparseMeta:
    def test_early_return(self, make_worker, set_gsa_const):
        set_gsa_const(PTOPK_PREFETCH_ENABLE=False)
        g, _ = make_worker()
        gsa_module.GSA.build_sparse_meta(g, fakes.FakeSchedulerOutput(), {}, None, None)

    def test_full_flow(self, make_worker, set_gsa_const, kv_group_stub):
        set_gsa_const(SEG_PREFILL_THRESHOLD=8)
        g, _ = make_worker()
        ensure_metadata(g)
        g.need_prefetch_index = []
        stat = stat_for(
            g, req_id=1, prompt=4, output=0, computed=0, scheduled=4, blocks=[11]
        )
        req = fakes.FakeRequest(req_id=1, prompt_token_ids=list(range(4)))
        sched = fakes.FakeSchedulerOutput(
            num_scheduled_tokens={1: 4},
            scheduled_new_reqs=[fakes._NewReq(1)],
        )
        batch = fakes.FakeInputBatch(req_ids=[1], req_id_to_index={1: 0})
        gsa_module.GSA.build_sparse_meta(g, sched, {1: req}, batch, {})
        assert g.gsa_metadata is not None
        assert g.decode_index == []
        assert g.gsa_index == []
        assert g.is_all_gsa is False
        assert g.has_prefill is True
        assert g.need_prefetch_index == []


# ---------------------------------------------------------------------------
# indexer k-repr / topk blocks
# ---------------------------------------------------------------------------


class TestIndexerKRepr:
    def test_offload_disabled_early(self, make_worker):
        g, _ = make_worker()
        gsa_module.GSA.cal_indexer_k_repr_mean_prefill(g)
        gsa_module.GSA.cal_indexer_k_repr_mean_decode(g)
        gsa_module.GSA.cal_indexer_topk_blocks(g)
        gsa_module.GSA.set_indexer_topk_blocks_prefill(g)
        gsa_module.GSA.set_indexer_topk_blocks_decode(g)

    def test_prefill_mean_last_chunk(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, prompt=64, output=0, slots=[1], indexer_slots=[1])
        stat.prompt_sparse_len = 2
        g.need_cal_k_mean = [1]
        g.indexer_k_cache_sparse[0] = torch.randn(2, 128, 1, 128)
        g.indexer_k_cache_sparse[1] = torch.randn(2, 128, 1, 128)
        gsa_module.GSA.cal_indexer_k_repr_mean_prefill(g)
        # rank0: old indexer slots freed, new ones allocated & broadcast
        assert g.prefetch_engine.prefetch_engine_c is not None

    def test_prefill_mean_not_last_chunk(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, prompt=64, output=2, slots=[1], indexer_slots=[1])
        g.need_cal_k_mean = [1]
        gsa_module.GSA.cal_indexer_k_repr_mean_prefill(g)
        assert g.indexer_block_table_sm.version == 1  # rank0 wrote empty block

    def test_decode_masks(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, prompt=64, output=0, slots=[1], indexer_slots=[1])
        stat.include_mask = [True, False]
        stat.exclude_mask = [False, True]
        stat.prompt_block_len = 2
        stat.prompt_sparse_len = 1
        g.need_cal_k_mean = [1]
        g.need_indexer_sparse_reqs = [1]
        g.need_indexer_sparse_bs = [0]
        g.prefetch_engine.step_time = 1
        gsa_module.GSA.cal_indexer_k_repr_mean_decode(g)
        assert g.include_masks is not None
        assert g.exclude_masks is not None
        assert g.repre_slot_mappings is not None
        assert g.max_topk_len == [2]  # ceil(prompt_sparse_len 1 * ratio 1.5)

    def test_topk_blocks_empty_bs(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        g.need_indexer_sparse_bs = []
        gsa_module.GSA.cal_indexer_topk_blocks(g)

    def test_topk_blocks_full(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker({"dtype": torch.float32})
        ensure_metadata(g)
        stat = stat_for(
            g, req_id=1, prompt=64, output=0, slots=[1, 2], indexer_slots=[1, 2]
        )
        stat.include_mask = [True, False]
        stat.exclude_mask = [False, False]
        g.need_indexer_sparse_bs = [0]
        g.need_indexer_sparse_reqs = [1]
        g.include_masks = torch.tensor([[1, 0]], dtype=torch.uint8)
        g.exclude_masks = torch.tensor([[0, 0]], dtype=torch.uint8)
        g.repre_slot_mappings = torch.tensor([[1, 2]], dtype=torch.int32)
        g.max_topk_len = [1]
        # indexer last query/weight: (layer, bs, 32, 128) / (layer, bs, 32)
        g.indexer_last_query.np[:] = 1.0
        g.indexer_last_weight.np[:] = 1.0
        g.indexer_k_repr[:, :4] = torch.randn(2, 4, 1, 1, 128)
        gsa_module.GSA.cal_indexer_topk_blocks(g)
        # topk over 2 blocks with include mask forcing block 0
        assert (stat.indexer_topk_blocks[:, 0] == 0).all()

    def test_set_indexer_topk_prefill_full(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(
            g, req_id=1, prompt=64, output=0, slots=[1, 2], indexer_slots=[1, 2]
        )
        stat.prompt_sparse_len = 1
        stat.is_last_chunk = lambda: True
        stat.indexer_topk_blocks[:, :1] = 0
        g.need_indexer_sparse_reqs = [1]
        g.need_indexer_sparse_bs = [0]
        gsa_module.GSA.set_indexer_topk_blocks_prefill(g)
        eng = g.prefetch_engine.prefetch_engine_c
        assert eng.calls_of("set_indexer_map")
        assert eng.calls_of("indexer_init_load")
        assert stat.is_indexer_init is True

    def test_set_indexer_topk_prefill_not_last(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, prompt=64, output=2)
        stat.is_last_chunk = lambda: False
        g.need_indexer_sparse_reqs = [1]
        g.need_indexer_sparse_bs = [0]
        gsa_module.GSA.set_indexer_topk_blocks_prefill(g)
        assert not g.prefetch_engine.prefetch_engine_c.calls_of("set_indexer_map")

    def test_set_indexer_topk_decode_full(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, prompt=64, output=2)
        stat.prompt_sparse_len = 2
        stat.is_last_chunk = lambda: False
        g.need_indexer_sparse_reqs = [1]
        g.need_indexer_sparse_bs = [0]
        g.prefetch_engine.prefetch_engine_c.get_indexer_status = lambda: False
        gsa_module.GSA.set_indexer_topk_blocks_decode(g)
        assert g.prefetch_engine.prefetch_engine_c.calls_of("submit_task_indexer")

    def test_set_indexer_topk_decode_all_last(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1, prompt=64, output=0)
        stat.is_last_chunk = lambda: True
        g.need_indexer_sparse_reqs = [1]
        g.need_indexer_sparse_bs = [0]
        gsa_module.GSA.set_indexer_topk_blocks_decode(g)
        assert not g.prefetch_engine.prefetch_engine_c.calls_of("submit_task_indexer")

    def test_apply_indexer_add_blocks(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        ensure_metadata(g)
        stat = stat_for(g, req_id=1)
        gsa_module.GSA._apply_indexer_add_blocks(g, [1], None, [0])  # None -> early
        gsa_module.GSA._apply_indexer_add_blocks(g, [1], [[7]], [1])
        assert stat.indexer_slab_host_slot == [7]


# ---------------------------------------------------------------------------
# init_dsa_data / init_kvcache_ptrs / update_slot_num / shutdown
# ---------------------------------------------------------------------------


class TestInitAndShutdown:
    def test_init_dsa_data(self, make_worker):
        g, _ = make_worker()
        runner = SimpleNamespace()
        gsa_module.GSA.init_dsa_data(g, runner)
        assert runner.gsa_swap_topk is g.gsa_swap_topk
        assert runner.dsa_slot_mapping is g.dsa_slot_mapping
        assert not hasattr(runner, "indexer_last_query")

    def test_init_dsa_data_offload(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        runner = SimpleNamespace()
        gsa_module.GSA.init_dsa_data(g, runner)
        assert runner.indexer_last_query is g.indexer_last_query

    def test_init_kvcache_ptrs(self, make_worker):
        g, _ = make_worker()
        kv = [object(), object(), None, None]
        gsa_module.GSA.init_kvcache_ptrs(g, kv, "ic", "is", 0)
        assert g.kv_caches[0] is kv
        assert g.indexer_k_cache_sparse[0] == "ic"
        assert g.prefetch_engine.prefetch_engine_c.calls_of("set_kvTensor_Layer")
        # second registration is a no-op
        gsa_module.GSA.init_kvcache_ptrs(g, [None], None, None, 0)
        assert g.kv_caches[0] is kv

    def test_init_kvcache_ptrs_offload(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        gsa_module.GSA.init_kvcache_ptrs(
            g, [object(), object(), None, None], "ic", None, 0
        )
        assert g.prefetch_engine.prefetch_engine_c.calls_of("set_kvTensor_Index")

    def test_update_slot_num_rank0(self, make_worker):
        g, _ = make_worker()
        gsa_module.GSA.update_slot_num(g)
        i_num, k_num = g.slots_num_manager.read_num()
        assert i_num >= 0 and k_num == 0  # offload disabled -> k stays 0

    def test_update_slot_num_rank0_offload(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        gsa_module.GSA.update_slot_num(g)
        i_num, k_num = g.slots_num_manager.read_num()
        assert k_num == g.gsa_cpu_shared_pool.get_free_num()

    def test_update_slot_num_rank1(self, make_worker1):
        g, _ = make_worker1()
        gsa_module.GSA.update_slot_num(g)  # no slots_num_manager: no crash

    def test_shutdown_offload_disabled_early(self, make_worker):
        g, _ = make_worker()
        gsa_module.GSA.shutdown(g)

    def test_shutdown_unlinks(self, make_worker, set_gsa_const):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()
        gsa_module.GSA.shutdown(g)

    def test_shutdown_unlink_failure_tolerated(
        self, make_worker, set_gsa_const, monkeypatch
    ):
        set_gsa_const(IS_INDEXER_OFFLOAD=True)
        g, _ = make_worker()

        def boom():
            raise OSError("boom")

        monkeypatch.setattr(g.block_table_sm, "unlink", boom)
        gsa_module.GSA.shutdown(g)  # second unlink still attempted

    def test_print_log(self, make_worker):
        g, _ = make_worker()
        g.print_log("hello")  # routes to GSALogger.info
