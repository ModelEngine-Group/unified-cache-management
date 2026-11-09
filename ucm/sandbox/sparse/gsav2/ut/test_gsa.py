# -*- coding: utf-8 -*-
"""ucm/sandbox/sparse/gsav2/gsa.py 单元测试。

覆盖策略：
- C++ 扩展（GSAPrefetchEngineC / UCMKVPrefetch / torch.ops._C_ascend）使用
  conftest 注入的 mock；
- 模块常量（IS_INDEXER_OFFLOAD / PTOPK_PREFETCH_ENABLE / IS_PERFORMANCE_LOG /
  SEG_PREFILL_THRESHOLD 等）通过 monkeypatch 构造分支矩阵；
- 平台分支（L30-43）由 import 时的 True 侧 + reload 用例的 False 侧共同覆盖。
"""
import importlib
import math
import types
from types import SimpleNamespace

import fakes
import numpy as np
import pytest
import torch

import ucm.sandbox.sparse.gsav2.gsa as gsa_mod
from ucm.sparse.base import UcmSparseRole


# ===========================================================================
# 模块级函数与小类
# ===========================================================================
class TestModuleLevel:
    def test_compute_parent_block_hash(self):
        v1 = gsa_mod.compute_parent_block_hash("m", 2, "bf16")
        v2 = gsa_mod.compute_parent_block_hash("m", 2, "bf16")
        assert v1 == v2 and isinstance(v1, int)
        assert gsa_mod.compute_parent_block_hash("m", 2, "bf16", 1) != v1

    def test_inject_store_v1_ptr_without_instance(self, monkeypatch):
        monkeypatch.setattr(gsa_mod, "GSA_INSTANCE", None)
        gsa_mod.inject_store_v1_ptr(123)
        assert gsa_mod._pending_store_ptr == 123

    def test_inject_store_v1_ptr_with_instance(self, monkeypatch):
        calls = []
        inst = SimpleNamespace(_inject_store_ptr=lambda p: calls.append(p))
        monkeypatch.setattr(gsa_mod, "GSA_INSTANCE", inst)
        gsa_mod.inject_store_v1_ptr(456)
        assert calls == [456]
        assert gsa_mod._pending_store_ptr == 456

    def test_platform_branch_reload(self, monkeypatch):
        """覆盖 L30-43 两侧：True 侧在 import 时已执行，此处覆盖 False 侧并还原"""
        import torch as _torch

        assert gsa_mod.torch_npu is not None  # import 时为 True 侧（stub）
        monkeypatch.delattr(_torch, "npu")
        importlib.reload(gsa_mod)
        assert gsa_mod.torch_npu is None
        assert gsa_mod.get_sfa_qsfa_packed_head_dim is None
        assert gsa_mod.CpuAlloc is None
        assert gsa_mod.AscendDeviceType is None
        assert gsa_mod.get_ascend_device_type is None
        # 还原为 True 侧，保证后续用例使用 stub 后的绑定
        monkeypatch.undo()
        importlib.reload(gsa_mod)
        assert gsa_mod.torch_npu is not None


class TestSmallClasses:
    def test_gsa_block_table(self):
        bt = gsa_mod.GSABlockTable((4, 8), torch.int32)
        assert bt.len == 0
        bt.np[0, 0] = 7
        assert bt.cpu[0, 0].item() == 7
        bt.update_len(4)
        assert bt.len == 4

    def test_gsa_shared_cpu_pool(self):
        pool = gsa_mod.GSASharedCPUPool(5)
        assert pool.get_free_num() == 4  # slot 0 保留
        alloc = pool.allocate(2)
        assert len(alloc) == 2 and pool.get_free_num() == 2
        pool.free(alloc)
        assert pool.get_free_num() == 4
        assert pool.allocated == set()

    def test_gsa_shared_cpu_pool_allocate_assert(self):
        pool = gsa_mod.GSASharedCPUPool(3)
        with pytest.raises(AssertionError):
            pool.allocate(5)  # 仅 2 个空闲

    def test_gsa_req_block_table(self):
        rbt = gsa_mod.GSAReqBlockTable((2, 4))
        assert rbt.cpu.shape == (2, 4)
        assert rbt.ptr == rbt.cpu.data_ptr()
        rbt.np[:] = 3
        assert rbt.cpu.sum().item() == 24

    def test_gsa_hash_request(self):
        hr = gsa_mod.GSAHashRequest([1, 2, 3])
        assert hr.all_token_ids == [1, 2, 3]
        assert hr.mm_features is None
        assert hr.lora_request is None
        assert hr.cache_salt is None
        assert hr.prompt_embeds is None


# ===========================================================================
# GSAReqStat
# ===========================================================================
class TestGSAReqStat:
    def _stat(self, config=None, **kw):
        cfg = config or fakes.FakeVllmConfig()
        stat = gsa_mod.GSAReqStat("r1", cfg, 2)
        for k, v in kw.items():
            setattr(stat, k, v)
        return stat

    def test_init_basic_fields(self):
        stat = self._stat()
        assert stat.req_id == "r1"
        assert stat.blocks == []
        assert stat.num_computed_tokens == 0
        assert stat.num_prompt_tokens == 0
        assert stat.num_output_tokens == 0
        assert stat.original_prompt_block_len == -1
        assert stat.block_size == 128
        assert stat.layer_num == 2
        assert stat.use_tokens.cpu.shape == (2, 5120)
        assert stat.prefetch_use_token.cpu.shape == (2, 5120)
        assert stat.is_gsa_init is False
        assert stat.is_pd_mix is False  # PD 分离（仅 consumer）
        assert stat.block_hashes is None

    def test_init_with_indexer_offload(self, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        stat = self._stat()
        assert stat.indexer_block_tables.shape == (2, 16)
        assert stat.prefetch_indexer_block_tables.shape == (2, 16)
        assert stat.indexer_topk_blocks.shape == (2, 16)
        assert stat.include_mask == []
        assert stat.prompt_block_len == 0
        assert stat.prompt_sparse_len == 0
        assert stat.is_indexer_init is False

    def test_pd_mix_flag(self):
        stat = self._stat(fakes.FakeVllmConfig(producer=True, consumer=True))
        assert stat.is_pd_mix is True

    def test_step_and_stage(self):
        stat = self._stat(num_prompt_tokens=100, num_computed_tokens=50)
        assert stat.step() == 0
        assert stat.stage() == gsa_mod.SequenceStage.PREFILL
        stat.num_output_tokens = 3
        assert stat.step() == 3
        stat.num_computed_tokens = 100
        assert stat.stage() == gsa_mod.SequenceStage.DECODE

    def test_indexer_sparse_len(self):
        stat = self._stat(
            num_computed_tokens=100,
            num_scheduled_tokens=28,
            prompt_block_len=4,
            prompt_sparse_len=2,
        )
        assert stat.get_seq_len() == 128
        assert stat.indexer_sparse_len() == 2 + 1 - 4  # 2+ceil(128/128)-4

    def test_is_gsa_combinations(self, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        # 长 prompt + decode → True
        stat = self._stat(num_prompt_tokens=100, num_computed_tokens=100)
        assert stat.is_gsa() is True
        # 长 prompt + prefill → False
        stat = self._stat(num_prompt_tokens=100, num_computed_tokens=50)
        assert stat.is_gsa() is False
        # 短 prompt → False
        stat = self._stat(num_prompt_tokens=10, num_computed_tokens=10)
        assert stat.is_gsa() is False

    def test_is_indexer_sparse_combinations(self, monkeypatch):
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 64)
        stat = self._stat(num_prompt_tokens=100, num_computed_tokens=100)
        assert stat.is_indexer_sparse() is True
        stat = self._stat(num_prompt_tokens=100, num_computed_tokens=50)
        assert stat.is_indexer_sparse() is False
        stat = self._stat(num_prompt_tokens=10, num_computed_tokens=10)
        assert stat.is_indexer_sparse() is False

    def test_is_last_chunk_pd_mix(self):
        cfg = fakes.FakeVllmConfig(producer=True, consumer=True)
        stat = self._stat(
            cfg, num_prompt_tokens=100, num_computed_tokens=90, num_scheduled_tokens=10
        )
        assert stat.is_last_chunk() is True  # 90+10 == 100
        stat.num_scheduled_tokens = 5
        assert stat.is_last_chunk() is False

    def test_is_last_chunk_pd_separated(self):
        stat = self._stat(num_output_tokens=0)
        assert stat.is_last_chunk() is True
        stat.num_output_tokens = 1
        assert stat.is_last_chunk() is False

    def test_is_decode_only(self):
        # PD 分离：num_output_tokens >= 1
        stat = self._stat(num_output_tokens=1)
        assert stat.is_decode_only() is True
        stat.num_output_tokens = 0
        assert stat.is_decode_only() is False
        # PD 混合：stage == DECODE
        stat = self._stat(
            fakes.FakeVllmConfig(producer=True, consumer=True),
            num_prompt_tokens=100,
            num_computed_tokens=100,
        )
        assert stat.is_decode_only() is True
        stat.num_computed_tokens = 50
        assert stat.is_decode_only() is False

    def test_set_block_hashes(self):
        stat = self._stat()
        req = gsa_mod.GSAHashRequest(list(range(300)))
        stat.set_block_hashes(req)
        n_hash = len(stat.block_hashes)
        assert n_hash == math.ceil(300 / 128)
        stat.set_block_hashes(req)  # 已有 → 早退
        assert len(stat.block_hashes) == n_hash

    def test_get_spase_len(self, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        stat = self._stat(
            num_prompt_tokens=100, num_computed_tokens=100, num_output_tokens=5
        )
        assert stat.get_spase_len() == 5 + gsa_mod.GSA_SPARSE_LEN
        stat.num_prompt_tokens = 10  # 非 GSA → num_computed_tokens
        assert stat.get_spase_len() == 100

    def test_add_req_new_normal(self):
        stat = self._stat()
        add = fakes.FakeAddReqState(
            block_ids=[[5, 6]],
            prompt_token_ids=list(range(200)),
            output_token_ids=[1],
            num_computed_tokens=200,
        )
        stat.add_req_new(1, add, 0, {})
        assert stat.blocks == [5, 6]
        assert stat.num_prompt_tokens == 200
        assert stat.num_output_tokens == 1
        assert stat.original_prompt_tokens == 200
        assert stat.non_ucm_block_count == 1
        assert stat.original_prompt_block_len == 1
        assert len(stat.block_hashes) == 2  # ceil(200/128)

    def test_add_req_new_preempt(self):
        stat = self._stat()
        add = fakes.FakeAddReqState(
            block_ids=[[5, 6]],
            prompt_token_ids=list(range(200)),
            output_token_ids=[1, 2, 3, 4],
            num_computed_tokens=200,
        )
        stat.add_req_new(1, add, 0, {"r1": 3})
        assert stat.num_prompt_tokens == 203
        assert stat.num_output_tokens == 1  # 4 - 3
        assert stat.original_prompt_tokens == 200
        assert stat.non_ucm_block_count == math.ceil(203 / 128) - 1  # 2-1

    def test_add_req_new_with_indexer_offload(self, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 64)
        stat = self._stat()
        add = fakes.FakeAddReqState(prompt_token_ids=list(range(2000)))
        stat.add_req_new(1, add, 0, {})
        # prompt_block_len = (2000-1)//128 = 15
        assert stat.prompt_block_len == 15
        # prompt_sparse_len = max(ceil((15-1-5)*0.2+1+5), 64) = 64
        assert stat.prompt_sparse_len == 64
        assert len(stat.include_mask) == 15
        assert stat.include_mask[0] is True
        assert stat.include_mask[-1] is True

    def test_init_mask_short_prompt(self, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 32768)
        stat = self._stat(num_prompt_tokens=100, prompt_block_len=1)
        stat.init_mask()  # 短 prompt → 直接返回
        assert stat.include_mask == []

    def test_init_mask_full(self, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 64)
        stat = self._stat(num_prompt_tokens=100, prompt_block_len=6)
        stat.init_mask()
        assert len(stat.include_mask) == 6
        assert stat.include_mask[0] is True
        assert all(stat.include_mask[-5:])  # INIT_LOCAL_WINDOWS = 5
        assert stat.exclude_mask == [False] * 6

    def test_update_req_state(self):
        stat = self._stat()
        add = fakes.FakeAddReqState(
            block_ids=[[7, 8]],
            output_token_ids=[1, 2],
            num_computed_tokens=9,
        )
        stat.update_req_state(2, add, 3, {})
        assert stat.num_computed_tokens == 9
        assert stat.num_scheduled_tokens == 2
        assert stat.num_output_tokens == 2
        assert stat.index_in_batch == 3
        assert stat.blocks == [7, 8]
        stat.update_req_state(2, add, 3, {"r1": 1})
        assert stat.num_output_tokens == 1

    def test_get_indexer_sparse_len(self, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 64)
        stat = self._stat(
            num_prompt_tokens=2000,
            num_computed_tokens=1920,
            num_scheduled_tokens=128,
            prompt_block_len=15,
            prompt_sparse_len=8,
        )
        # sparse_len = (15-8)*128 = 896；seq = 2048 → 2048-896
        assert stat.get_indexer_sparse_len() == 2048 - 7 * 128
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 32768)
        stat.num_prompt_tokens = 100
        stat.num_computed_tokens = 100
        stat.num_scheduled_tokens = 28
        assert stat.get_indexer_sparse_len() == 128  # 早退：get_seq_len()

    def test_get_indexer_remain_len(self, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 64)
        stat = self._stat(
            num_prompt_tokens=100,
            num_computed_tokens=100,
            num_scheduled_tokens=28,
            prompt_block_len=4,
            prompt_sparse_len=2,
        )
        # sparse_len = (4 - ceil(2*1.5)) * 128 = 128；seq = 128 → 0
        assert stat.get_indexer_remain_len() == 0
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 32768)
        stat.num_prompt_tokens = 100
        assert stat.get_indexer_remain_len() == 128  # 早退：get_seq_len()


# ===========================================================================
# GSAMetaData
# ===========================================================================
class TestGSAMetaData:
    def _meta(self):
        return gsa_mod.GSAMetaData(fakes.FakeVllmConfig(), 2)

    def test_init(self):
        md = self._meta()
        assert md.gsa_stats == {}
        assert md.block_size == 128
        assert md.device == "cpu"
        assert md.use_mla is True
        assert md.topk_layer_num == 2

    def _state(self, req_id="r1", prompt=200, output=1, computed=200):
        return fakes.FakeAddReqState(
            req_id=req_id,
            block_ids=[[5, 6]],
            prompt_token_ids=list(range(prompt)),
            output_token_ids=list(range(output)),
            num_computed_tokens=computed,
        )

    def test_get_model_input_new_req(self):
        md = self._meta()
        so = fakes.FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 2},
            new_reqs=[SimpleNamespace(req_id="r1")],
        )
        model_input = md.get_model_input(
            so,
            2048,
            {"r1": self._state()},
            fakes.FakeInputBatch({"r1": 0}),
            None,
            {},
        )
        assert "r1" in md.gsa_stats
        assert model_input["query_locals"] == [0, 2]

    def test_get_model_input_resumed_and_cached(self):
        md = self._meta()
        so = fakes.FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 2, "r2": 3},
            cached_req_ids=["r1", "r2"],
            resumed_req_ids=["r1", "r2"],
        )
        md.get_model_input(
            so,
            2048,
            {"r1": self._state(), "r2": self._state()},
            fakes.FakeInputBatch({"r1": 0, "r2": 1}),
            None,
            {},
        )
        assert set(md.gsa_stats) == {"r1", "r2"}
        # 再次调度（else 分支：update_req_state）
        so2 = fakes.FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1},
            cached_req_ids=["r1"],
        )
        mi = md.get_model_input(
            so2, 2048, {"r1": self._state()}, fakes.FakeInputBatch({"r1": 0}), None, {}
        )
        assert mi["query_locals"][0] == 0

    def test_get_model_input_re_add_deletes_old(self):
        md = self._meta()
        so = fakes.FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 2},
            new_reqs=[SimpleNamespace(req_id="r1")],
        )
        md.get_model_input(
            so, 2048, {"r1": self._state()}, fakes.FakeInputBatch({"r1": 0}), None, {}
        )
        old = md.gsa_stats["r1"]
        md.get_model_input(
            so, 2048, {"r1": self._state()}, fakes.FakeInputBatch({"r1": 0}), None, {}
        )
        assert md.gsa_stats["r1"] is not old

    def test_trans_input_tensor(self):
        md = self._meta()
        md.gsa_stats["a"] = SimpleNamespace(index_in_batch=0)
        md.gsa_stats["b"] = SimpleNamespace(index_in_batch=1)
        mi = md.trans_input_tensor(
            SimpleNamespace(num_scheduled_tokens={"a": 2, "b": 3})
        )
        assert mi["query_locals"] == [0, 2, 5]


# ===========================================================================
# GSA 构造
# ===========================================================================
class TestGSAConstruction:
    def test_worker_full_init(self, gsa_worker):
        g = gsa_worker
        assert g.rank == 0 and g.device == "cpu"
        assert g.block_size == 128 and g.layer_num == 2
        assert g.num_slots == 16  # patched MAX_SLOTS_BLOCK_NUM
        assert g.is_decode_node is True
        assert g.is_pd_mix is False
        assert g.num_spec_tokens == 1
        assert g.topk_layer_num == 2
        assert isinstance(g.indexer_block_table_sm, fakes.FakeSharedBlockTable)
        assert isinstance(g.prefetch_status_sm, fakes.FakeSharedTensorVersion)
        assert isinstance(g.ucm_prefetch_engine, fakes.FakeUCMPrefetch)
        assert isinstance(g.prefetch_engine.prefetch_engine_c, fakes.FakeEngineC)
        assert g.gsa_swap_topk_bs == g.gsa_swap_topk.cpu.shape[1]
        assert g.gsa_prefill_query.tolist() == [1, 2, 3, 4]
        assert not hasattr(g, "block_table_sm")  # IS_INDEXER_OFFLOAD=False
        assert not hasattr(g, "select_gsa_prefetch_topk")  # 无 speculative

    def test_worker_init_ptopk_disabled(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "PTOPK_PREFETCH_ENABLE", False)
        g = make_gsa()
        assert not hasattr(g, "prefetch_engine")
        assert not hasattr(g, "log")

    def test_worker_init_not_decode_node(self, make_gsa):
        g = make_gsa(consumer=False)
        assert g.is_decode_node is False
        assert not hasattr(g, "prefetch_engine")
        assert not hasattr(g, "slots_num_manager")  # rank0 但非 decode

    def test_scheduler_role(self, make_gsa):
        g = make_gsa(role=UcmSparseRole.SCHEDULER)
        assert isinstance(g.slots_num_manager, fakes.FakeSharedTensorManager)
        assert g.slots_num_manager.is_creator is False

    def test_worker_rank_not_zero(self, make_gsa):
        g = make_gsa(rank=1)
        assert g.rank == 1
        # 非 rank0：block table 由 FakeSharedBlockTable(is_create=False) 附加
        assert g.indexer_block_table_sm.is_creator is False
        assert g.prefetch_status_sm.is_creator is False

    def test_sparse_c8_config_combinations(self, make_gsa, monkeypatch):
        # 均关闭：模型 dtype
        g = make_gsa()
        assert g.dtype == torch.float16
        # 仅 li_c8
        g = make_gsa(additional_config={"enable_sparse_li_c8": True})
        assert g.enable_sparse_li_c8 is True
        assert g.dtype == torch.float16
        # sfa_c8 + 非 A5
        g = make_gsa(additional_config={"enable_sparse_sfa_c8": True})
        assert g.dtype == torch.int8
        assert g.c8_k_scale_cache_dtype == torch.float16
        assert g.k_dim == 32
        # sfa_c8 + A5
        monkeypatch.setattr(gsa_mod, "get_ascend_device_type", lambda: "A5")
        g = make_gsa(additional_config={"enable_sparse_sfa_c8": True})
        assert g.dtype == torch.float8_e4m3fn
        assert g.c8_k_scale_cache_dtype == torch.float32

    def test_speculative_config(self, make_gsa):
        g = make_gsa(speculative=fakes.FakeSpeculativeConfig(2))
        assert g.num_spec_tokens == 3
        assert g.select_gsa_prefetch_topk.shape == (2, 4, 1, 2048)

    def test_pd_mix_config(self, make_gsa):
        g = make_gsa(producer=True, consumer=True)
        assert g.is_pd_mix is True

    def test_init_indexer_offload_worker(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        assert hasattr(g, "block_table_sm")
        assert hasattr(g, "gsa_cpu_shared_pool")
        assert hasattr(g, "indexer_last_query")
        assert hasattr(g, "indexer_k_repr")
        assert hasattr(g, "indexer_cache_preempt")
        assert g.indexer_cpu_shared_pool.get_free_num() == 15

    def test_bind_cpus_paths(self, monkeypatch):
        monkeypatch.setattr(gsa_mod, "CpuAlloc", fakes.FakeCpuAlloc)
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", False)
        g = object.__new__(gsa_mod.GSA)
        g.local_rank = 0
        # 空 cpu_pool
        fakes.FakeCpuAlloc.cpu_pool = {0: []}
        assert gsa_mod.GSA.bind_cpus(g) == (None, None)
        # assign_main 充足 → 取尾部 20 个
        fakes.FakeCpuAlloc.cpu_pool = {0: [3]}
        fakes.FakeCpuAlloc.cpu_node = {3: 1}
        fakes.FakeCpuAlloc.assign_main = {0: list(range(30))}
        numa, cpus = gsa_mod.GSA.bind_cpus(g)
        assert numa == 1 and cpus == list(range(10, 30))
        # assign_main 不足 → 取 [1:]
        fakes.FakeCpuAlloc.assign_main = {0: [7, 8, 9]}
        numa, cpus = gsa_mod.GSA.bind_cpus(g)
        assert numa == 1 and cpus == [8, 9]

    def test_bind_cpus_indexer_thread_num(self, monkeypatch):
        monkeypatch.setattr(gsa_mod, "CpuAlloc", fakes.FakeCpuAlloc)
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = object.__new__(gsa_mod.GSA)
        g.local_rank = 0
        fakes.FakeCpuAlloc.cpu_pool = {0: [3]}
        fakes.FakeCpuAlloc.cpu_node = {3: 2}
        fakes.FakeCpuAlloc.assign_main = {0: list(range(30))}
        numa, cpus = gsa_mod.GSA.bind_cpus(g)
        assert numa == 2 and len(cpus) == 24  # IS_INDEXER_OFFLOAD → 24 线程

    def test_print_log(self, gsa_worker):
        gsa_worker.print_log("hello")  # 写入 tmp 下的 normal 日志

    def test_init_gsa_metadata_early_return(self, gsa_worker, monkeypatch):
        # 构造路径下该分支不可达（外层已过滤），直接调用覆盖早退
        monkeypatch.setattr(gsa_mod, "PTOPK_PREFETCH_ENABLE", False)
        gsa_worker.init_gsa_metadata(fakes.FakeVllmConfig())


# ===========================================================================
# store 注入
# ===========================================================================
class TestStoreInjection:
    def test_find_store_connector_plain_with_store(self, gsa_worker):
        conn = fakes.make_plain_connector(with_store=True)
        assert gsa_worker._find_store_connector(conn) is conn

    def test_find_store_connector_plain_with_inner(self, gsa_worker):
        conn = fakes.make_plain_connector(
            with_store=False, with_inner=True, inner_with_store=True
        )
        assert gsa_worker._find_store_connector(conn) is conn.connector

    def test_find_store_connector_plain_with_inner_no_store(self, gsa_worker):
        conn = fakes.make_plain_connector(
            with_store=False, with_inner=True, inner_with_store=False
        )
        assert gsa_worker._find_store_connector(conn) is conn.connector

    def test_find_store_connector_multi_no_connectors(self, gsa_worker):
        conn = fakes.make_multi_connector("no_connectors")
        assert gsa_worker._find_store_connector(conn) is None

    def test_find_store_connector_multi_direct_store(self, gsa_worker):
        conn = fakes.AscendMultiConnector(
            connectors=[fakes.make_plain_connector(with_store=True)]
        )
        assert gsa_worker._find_store_connector(conn) is conn._connectors[0]

    def test_find_store_connector_multi_inner_store(self, gsa_worker):
        conn = fakes.AscendMultiConnector(
            connectors=[
                fakes.make_plain_connector(
                    with_store=False, with_inner=True, inner_with_store=True
                )
            ]
        )
        assert gsa_worker._find_store_connector(conn) is conn._connectors[0].connector

    def test_find_store_connector_multi_inner_no_store(self, gsa_worker):
        conn = fakes.AscendMultiConnector(
            connectors=[
                fakes.make_plain_connector(
                    with_store=False, with_inner=True, inner_with_store=False
                )
            ]
        )
        assert gsa_worker._find_store_connector(conn) is None

    def test_inject_store_v1_null_ptr(self, gsa_worker):
        conn = fakes.make_plain_connector(store_ptr=0)
        assert gsa_worker._inject_store_v1(conn) is False
        assert gsa_worker._store_injected is False

    def test_inject_store_v1_success(self, gsa_worker):
        g = gsa_worker
        g.prefetch_engine.prefetch_engine_c.store_v1_ptr = 4096
        assert g._inject_store_v1(fakes.make_plain_connector(store_ptr=4096))
        assert g._store_injected is True
        assert g.ucm_prefetch_engine.calls == [("store_init", 4096, g.layer_num)]

    def test_inject_store_v1_verification_failure(self, gsa_worker):
        g = gsa_worker
        g.prefetch_engine.prefetch_engine_c.store_v1_ptr = 8192  # 不匹配
        with pytest.raises(RuntimeError, match="verification failed"):
            g._inject_store_v1(fakes.make_plain_connector(store_ptr=4096))

    def test_log_store_inject_retry_below_max(self, gsa_worker):
        gsa_worker._store_inject_retries = 1
        gsa_worker._log_store_inject_retry("some reason")  # 只记日志
        assert gsa_worker._store_inject_retries == 1

    def test_log_store_inject_retry_at_max_raises(self, gsa_worker):
        gsa_worker._store_inject_retries = 5
        with pytest.raises(RuntimeError, match="failed after"):
            gsa_worker._log_store_inject_retry("boom")

    def _patch_group(self, monkeypatch, group):
        monkeypatch.setattr(gsa_mod, "get_kv_transfer_group", lambda: group)

    def test_ensure_store_injected_already(self, gsa_worker, monkeypatch):
        gsa_worker._store_injected = True
        called = []
        self._patch_group(monkeypatch, lambda: called.append(1))  # 不应被调用
        gsa_worker._ensure_store_injected()
        assert not called

    def test_ensure_store_injected_happy_path(self, gsa_worker, monkeypatch):
        g = gsa_worker
        g.prefetch_engine.prefetch_engine_c.store_v1_ptr = 4096
        self._patch_group(monkeypatch, fakes.make_plain_connector(store_ptr=4096))
        g._ensure_store_injected()
        assert g._store_injected is True
        assert g._store_inject_retries == 1

    def test_ensure_store_injected_connector_none(self, gsa_worker, monkeypatch):
        g = gsa_worker
        conn = fakes.make_multi_connector("no_connectors")
        self._patch_group(monkeypatch, conn)
        g._ensure_store_injected()
        assert g._store_injected is False

    def test_ensure_store_injected_no_store_attr(self, gsa_worker, monkeypatch):
        self._patch_group(monkeypatch, SimpleNamespace())
        gsa_worker._ensure_store_injected()
        assert gsa_worker._store_injected is False

    def test_ensure_store_injected_store_none(self, gsa_worker, monkeypatch):
        self._patch_group(monkeypatch, SimpleNamespace(store=None))
        gsa_worker._ensure_store_injected()
        assert gsa_worker._store_injected is False

    def test_ensure_store_injected_cc_store_zero(self, gsa_worker, monkeypatch):
        g = gsa_worker
        g.prefetch_engine.prefetch_engine_c.store_v1_ptr = 4096
        self._patch_group(monkeypatch, fakes.make_plain_connector(store_ptr=0))
        g._ensure_store_injected()
        assert g._store_injected is False

    def test_ensure_store_injected_mismatch_swallowed(self, gsa_worker, monkeypatch):
        g = gsa_worker
        g.prefetch_engine.prefetch_engine_c.store_v1_ptr = 8192
        self._patch_group(monkeypatch, fakes.make_plain_connector(store_ptr=4096))
        g._ensure_store_injected()  # 异常被捕获（未达重试上限）
        assert g._store_injected is False

    def test_ensure_store_injected_retry_limit(self, gsa_worker, monkeypatch):
        g = gsa_worker
        g._store_inject_retries = 5
        self._patch_group(monkeypatch, fakes.make_multi_connector("no_connectors"))
        with pytest.raises(RuntimeError, match="failed after"):
            g._ensure_store_injected()

    def test_ensure_store_injected_exception_below_max(self, gsa_worker, monkeypatch):
        g = gsa_worker

        def _boom():
            raise ValueError("kv group not ready")

        monkeypatch.setattr(gsa_mod, "get_kv_transfer_group", _boom)
        g._ensure_store_injected()  # 捕获并记日志
        assert g._store_injected is False

    def test_ensure_store_injected_exception_at_max(self, gsa_worker, monkeypatch):
        g = gsa_worker
        g._store_inject_retries = 5

        def _boom():
            raise ValueError("kv group not ready")

        monkeypatch.setattr(gsa_mod, "get_kv_transfer_group", _boom)
        with pytest.raises(ValueError):
            g._ensure_store_injected()


# ===========================================================================
# 状态构造 helper
# ===========================================================================
def _prime_metadata(
    g,
    req_id="r1",
    index_in_batch=0,
    prompt=200,
    computed=200,
    scheduled=1,
    output=1,
    blocks=None,
    preempt=False,
):
    """构造带单个请求的 GSAMetaData 并挂到 GSA 实例上"""
    md = gsa_mod.GSAMetaData(g._vllm_config, 2)
    stat = gsa_mod.GSAReqStat(req_id, g._vllm_config, 2)
    stat.index_in_batch = index_in_batch
    stat.num_prompt_tokens = prompt
    stat.num_computed_tokens = computed
    stat.num_scheduled_tokens = scheduled
    stat.num_output_tokens = output
    stat.blocks = blocks if blocks is not None else [1, 2]
    stat.topk_buf_tmp = torch.zeros((2, 4), dtype=torch.float16)
    if preempt:
        stat.is_pd_mix = False  # PD 分离语义
    md.gsa_stats[req_id] = stat
    g.gsa_metadata = md
    g.gsa_stats = md.gsa_stats
    g.model_input = {"query_locals": [0, scheduled]}
    return md, stat


def _make_kv_caches(g, n_blocks=32):
    for layer in range(g.layer_num):
        g.kv_caches[layer] = [
            torch.zeros((n_blocks, 128, 1, 16), dtype=torch.float16),
            torch.zeros((n_blocks, 128, 1, 16), dtype=torch.float16),
            torch.zeros((n_blocks, 128, 1, 128), dtype=torch.float16),
            torch.zeros((n_blocks, 128, 1, 1), dtype=torch.float16),
        ]


def _register_kvcaches(g, n_blocks=32):
    """经 init_kvcache_ptrs 正式登记（同时覆盖其首次注册分支）"""
    k_dim = getattr(g, "k_dim", 16)
    for layer in range(g.layer_num):
        kv_cache = [
            torch.zeros((n_blocks, 128, 1, k_dim), dtype=torch.float16),
            torch.zeros((n_blocks, 128, 1, 16), dtype=torch.float16),
            torch.zeros((n_blocks, 128, 1, 128), dtype=torch.float16),
            torch.zeros((n_blocks, 128, 1, 1), dtype=torch.float16),
        ]
        g.init_kvcache_ptrs(kv_cache, kv_cache[2], kv_cache[3], layer)


def _record_ops(monkeypatch):
    """把 torch.ops._C_ascend 的 copy op 替换为计数记录器"""
    rec = {"gather": 0, "scatter": 0}
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "gsa_gather_copy",
        lambda *a, **k: rec.__setitem__("gather", rec["gather"] + 1),
    )
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "gsa_scatter_copy",
        lambda *a, **k: rec.__setitem__("scatter", rec["scatter"] + 1),
    )
    return rec


# ===========================================================================
# block hash 注入
# ===========================================================================
class TestInjectBlockHashes:
    def _prepare(self, g, block_hashes=(b"a" * 16, b"b" * 16)):
        _, stat = _prime_metadata(g)
        stat.block_hashes = list(block_hashes)
        g._slab_preempt_k_ptr = 1000
        g._slab_preempt_k_offset = 10
        g._slab_preempt_v_ptr = 2000
        g._slab_preempt_v_offset = 20
        return stat

    def test_already_injected_early_return(self, gsa_worker):
        g = gsa_worker
        self._prepare(g)
        g._block_hash_injected_reqs.add("r1")
        g._inject_block_hashes("r1", [0], [1])
        assert not g.prefetch_engine.prefetch_engine_c.called("set_blocks_map")

    def test_block_hashes_none_early_return(self, gsa_worker):
        g = gsa_worker
        self._prepare(g)
        g.gsa_metadata.gsa_stats["r1"].block_hashes = None
        g._inject_block_hashes("r1", [0], [1])
        assert not g.prefetch_engine.prefetch_engine_c.called("set_blocks_map")

    def test_normal_injection_sfa_off(self, gsa_worker):
        g = gsa_worker
        self._prepare(g)
        g._inject_block_hashes("r1", [0, 1], [2, 3])
        calls = g.prefetch_engine.prefetch_engine_c.called("set_blocks_map")
        assert len(calls) == 1
        args = calls[0][1]
        # v_refs 非空（sfa_c8 关闭），k_refs 按 slot 偏移
        assert args[4] == [1000 + 10 * 2, 1000 + 10 * 3]
        assert args[5] == [2000 + 20 * 2, 2000 + 20 * 3]
        assert "r1" in g._block_hash_injected_reqs

    def test_injection_sfa_on_empty_v_refs(self, make_gsa):
        g = make_gsa(additional_config={"enable_sparse_sfa_c8": True})
        self._prepare(g)
        g._inject_block_hashes("r1", [0, 1], [2, 3])
        args = g.prefetch_engine.prefetch_engine_c.called("set_blocks_map")[0][1]
        assert args[5] == []  # sfa_c8 开启 → v_refs 为空

    def test_injection_with_indexer(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        self._prepare(g)
        # 源码 _init_indexer_preempt 写的是 indexer_cache_preempt_ptr，
        # _inject_block_hashes 读的是 *_i_ptr —— 手动补设以覆盖该路径
        g.indexer_cache_preempt_i_ptr = 3000
        g.indexer_cache_preempt_i_offset = 30
        g._inject_block_hashes("r1", [0], [1], inject_indexer=True)
        calls = g.prefetch_engine.prefetch_engine_c.called("set_last_indexer_blocks")
        assert len(calls) == 1
        assert calls[0][1][2] == [3000 + 30]

    def test_injection_indexer_flag_off(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        self._prepare(g)
        g._inject_block_hashes("r1", [0], [1], inject_indexer=False)
        assert not g.prefetch_engine.prefetch_engine_c.called("set_last_indexer_blocks")


# ===========================================================================
# attention 钩子
# ===========================================================================
class TestAttentionHooks:
    def test_begin_ptopk_disabled(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "PTOPK_PREFETCH_ENABLE", False)
        out = gsa_worker.attention_begin("q", "k", "v", "model.layers.1.x", None)
        assert out == ("q", "k", "v", None)

    def test_begin_layer_out_of_range(self, gsa_worker):
        out = gsa_worker.attention_begin("q", "k", "v", "model.layers.9.x", None)
        assert out == ("q", "k", "v", None)

    def test_begin_no_indexer(self, gsa_worker):
        out = gsa_worker.attention_begin(
            "q", "k", "v", "model.layers.1.x", None, has_indexer=False
        )
        assert out == ("q", "k", "v", None)

    def test_begin_full_path(self, gsa_worker, monkeypatch):
        g = gsa_worker
        rec = _record_ops(monkeypatch)
        _, stat = _prime_metadata(g)
        g.req_ids_bs = ["r1"]
        g.indexer_pc_hit_num = [1]
        g.indexer_pc_hit_index.np[:] = 0
        _register_kvcaches(g)
        out = g.attention_begin(
            "q",
            "k",
            "v",
            "model.layers.1.self_attn",
            None,
            indexer_cache=g.kv_caches[1][2],
            indexer_scale=g.kv_caches[1][3],
            vllm_indexer_cache=g.kv_caches[1][2],
            vllm_indexer_scale=g.kv_caches[1][3],
            output="o",
        )
        assert out == ("q", "k", "v", "o")
        assert g.prefetch_engine.prefetch_engine_c.called("set_kvTensor_Layer")
        assert rec == {"gather": 1, "scatter": 1}

    def test_begin_hit_num_zero_skips(self, gsa_worker, monkeypatch):
        g = gsa_worker
        rec = _record_ops(monkeypatch)
        _prime_metadata(g)
        g.req_ids_bs = ["r1"]
        g.indexer_pc_hit_num = [0]
        g.attention_begin("q", "k", "v", "model.layers.1.x", None)
        assert rec == {"gather": 0, "scatter": 0}

    def test_begin_li_c8_path(self, make_gsa, monkeypatch):
        g = make_gsa(additional_config={"enable_sparse_li_c8": True})
        rec = _record_ops(monkeypatch)
        _prime_metadata(g)
        g.req_ids_bs = ["r1"]
        g.indexer_pc_hit_num = [1]
        g.indexer_pc_hit_index.np[:] = 0
        _register_kvcaches(g)
        g.attention_begin(
            "q",
            "k",
            "v",
            "model.layers.0.x",
            None,
            indexer_cache=g.kv_caches[0][2],
            indexer_scale=g.kv_caches[0][3],
            vllm_indexer_cache=g.kv_caches[0][2],
            vllm_indexer_scale=g.kv_caches[0][3],
            use_li_c8=True,
        )
        # li_c8：cache 块 + scale 块各一次 gather/scatter
        assert rec == {"gather": 2, "scatter": 2}

    def test_finished_pd_separated_noop(self, gsa_worker):
        g = gsa_worker  # is_pd_mix False
        g.attention_finished("q", "k", "v", "o", "model.layers.1.x", None)
        assert not g.prefetch_engine.prefetch_engine_c.called("gsa_gather_copy")

    def test_finished_layer_out_of_range(self, make_gsa):
        g = make_gsa(producer=True, consumer=True)  # pd_mix
        g.attention_finished("q", "k", "v", "o", "model.layers.9.x", None)

    def test_finished_no_dump(self, make_gsa):
        g = make_gsa(producer=True, consumer=True)
        g.indexer_pc_dump_num = 0
        g.attention_finished(
            "q", "k", "v", "o", "model.layers.1.x", None, has_indexer=True
        )

    def test_finished_no_indexer(self, make_gsa):
        g = make_gsa(producer=True, consumer=True)
        g.indexer_pc_dump_num = 3
        g.attention_finished(
            "q", "k", "v", "o", "model.layers.1.x", None, has_indexer=False
        )

    def test_finished_full_path(self, make_gsa, monkeypatch):
        g = make_gsa(producer=True, consumer=True)
        rec = _record_ops(monkeypatch)
        _register_kvcaches(g)
        g.indexer_pc_dump_num = 2
        g.indexer_pc_dump_index.np[:] = 0
        g.attention_finished("q", "k", "v", "o", "model.layers.1.x", None)
        assert rec == {"gather": 1, "scatter": 1}

    def test_finished_li_c8_path(self, make_gsa, monkeypatch):
        g = make_gsa(
            producer=True,
            consumer=True,
            additional_config={"enable_sparse_li_c8": True},
        )
        rec = _record_ops(monkeypatch)
        _register_kvcaches(g)
        g.indexer_pc_dump_num = 2
        g.indexer_pc_dump_index.np[:] = 0
        g.attention_finished(
            "q", "k", "v", "o", "model.layers.1.x", None, use_li_c8=True
        )
        # li_c8：KV 块 + scale 块各一次 gather/scatter
        assert rec == {"gather": 2, "scatter": 2}


# ===========================================================================
# DSA 输入构造
# ===========================================================================
class TestDsaInput:
    def test_init_dsa_input_buffers(self, gsa_worker):
        g = gsa_worker
        ret = g._init_dsa_input_buffers(["r1"])
        assert ret[0] == [None]
        assert g.indexer_pc_hit_num == [None]
        assert (g.dsa_block_table.np == 0).all()
        assert (g.dsa_slot_mapping.np == -1).all()

    def test_update_dsa_block_table_basic(self, gsa_worker):
        g = gsa_worker
        _, stat = _prime_metadata(g)
        stat.indexer_slab_host_slot = [1, 2]
        dsa_seq_len = [None]
        req_meta, slots, ib = g._update_dsa_block_table("r1", dsa_seq_len)
        assert req_meta is stat and slots == [1, 2] and ib == 0
        assert dsa_seq_len[0] == 201  # 200 + 1
        assert g.dsa_block_table.np[0, 0, :2].tolist() == [1, 2]

    def test_update_dsa_block_table_indexer_init(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 64)
        g = make_gsa()
        _, stat = _prime_metadata(g, prompt=384, computed=384, scheduled=0)
        stat.prompt_block_len = 4
        stat.prompt_sparse_len = 2
        stat.indexer_slab_host_slot = [3, 4, 5]
        stat.is_indexer_init = True
        stat.add_blocks_len = 1
        dsa_seq_len = [None]
        g._update_dsa_block_table("r1", dsa_seq_len)
        # indexer_sparse_len = 2 + ceil(384/128) - 4 = 1
        assert stat.indexer_block_tables[0][0] == 5
        assert dsa_seq_len[0] == stat.get_indexer_sparse_len()
        assert g.prefetch_engine.prefetch_engine_c.called("update_indexer_map")

    def test_update_dsa_block_table_indexer_no_add(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 64)
        g = make_gsa()
        _, stat = _prime_metadata(g, prompt=384, computed=384, scheduled=0)
        stat.prompt_block_len = 4
        stat.prompt_sparse_len = 2
        stat.indexer_slab_host_slot = [3, 4, 5]
        stat.is_indexer_init = True
        stat.add_blocks_len = 0  # 不更新增量块
        dsa_seq_len = [None]
        g._update_dsa_block_table("r1", dsa_seq_len)
        assert not g.prefetch_engine.prefetch_engine_c.called("update_indexer_map")

    def test_calc_decode_slot_mapping_spec1(self, gsa_worker):
        g = gsa_worker
        _, stat = _prime_metadata(g)  # seq_len = 201
        slot_mapping = [None]
        g._calc_decode_slot_mapping(stat, [1, 2], 0, slot_mapping)
        # slots[-1]*128 + (201-1)%128 = 256 + 72
        assert slot_mapping[0] == [2 * 128 + (201 - 1) % 128]

    def _spec_gsa(self, make_gsa):
        return make_gsa(speculative=fakes.FakeSpeculativeConfig(2))  # spec=3

    def test_calc_decode_slot_mapping_spec_cross_block(self, make_gsa):
        g = self._spec_gsa(make_gsa)
        _, stat = _prime_metadata(g, computed=130, scheduled=0)  # x=2
        slot_mapping = [None]
        g._calc_decode_slot_mapping(stat, [1, 2, 3], 0, slot_mapping)
        expected = list(range(2 * 128 + 128 - (3 - 2), 2 * 128 + 128)) + [
            i + 3 * 128 for i in range(2)
        ]
        assert slot_mapping[0] == expected

    def test_calc_decode_slot_mapping_spec_block_boundary(self, make_gsa):
        g = self._spec_gsa(make_gsa)
        _, stat = _prime_metadata(g, computed=128, scheduled=0)  # x=0
        slot_mapping = [None]
        g._calc_decode_slot_mapping(stat, [1, 2, 3], 0, slot_mapping)
        assert slot_mapping[0] == list(range(3 * 128 + 128 - 3, 3 * 128 + 128))

    def test_calc_decode_slot_mapping_spec_mid_block(self, make_gsa):
        g = self._spec_gsa(make_gsa)
        _, stat = _prime_metadata(g, computed=131, scheduled=0)  # x=3
        slot_mapping = [None]
        g._calc_decode_slot_mapping(stat, [1, 2, 3], 0, slot_mapping)
        assert slot_mapping[0] == list(range(3 * 128 + 3 - 3, 3 * 128 + 3))

    def test_calc_prefill_slot_mapping_match(self, gsa_worker, monkeypatch):
        g = gsa_worker
        _, stat = _prime_metadata(g, computed=0, scheduled=256, prompt=256, output=0)
        stat.num_computed_tokens = 0
        dump_vllm_id, dump_gsa_id = [None], [None]
        slot_mapping = [None]
        g._calc_prefill_slot_mapping(
            "r1", stat, [10, 11], 0, slot_mapping, dump_vllm_id, dump_gsa_id
        )
        assert dump_vllm_id[0] == stat.blocks[:2]
        assert dump_gsa_id[0] == [10, 11]
        assert slot_mapping[0] == [10 * 128 + j for j in range(128)] + [
            11 * 128 + j for j in range(128)
        ]
        assert stat.slot_mapping_cal_len == 256
        # 再次调用：num_computed == cal_len → 不重算 hit（hit_num 记 0）
        g.indexer_pc_hit_num = [None]
        g.indexer_pc_hit_index.np[:] = 0
        g._calc_prefill_slot_mapping(
            "r1", stat, [10, 11], 0, slot_mapping, dump_vllm_id, dump_gsa_id
        )
        assert g.indexer_pc_hit_num[0] == 0

    def test_calc_prefill_slot_mapping_mismatch_logs_error(self, gsa_worker):
        g = gsa_worker
        _, stat = _prime_metadata(g, computed=0, scheduled=256, prompt=256, output=0)
        g._calc_prefill_slot_mapping("r1", stat, [10], 0, [None], [None], [None])
        # slots 数量不足 → 4 条 error 日志
        assert (
            not hasattr(stat, "slot_mapping_cal_len") or stat.slot_mapping_cal_len == 0
        )

    def test_update_req_slot_mapping_decode_vs_prefill(self, gsa_worker):
        g = gsa_worker
        # decode-only
        _, stat = _prime_metadata(g)  # PD 分离 + output=1
        g.indexer_pc_hit_num = [None]
        slot_mapping, dv, dg = [None], [None], [None]
        g._update_req_slot_mapping("r1", stat, [1, 2], 0, slot_mapping, dv, dg)
        assert len(slot_mapping[0]) == 1 and dv[0] == [] and dg[0] == []
        # prefill
        _, stat = _prime_metadata(g, computed=0, scheduled=256, prompt=256, output=0)
        g.indexer_pc_hit_num = [None]
        g._update_req_slot_mapping("r1", stat, [10, 11], 0, slot_mapping, dv, dg)
        assert len(slot_mapping[0]) == 256

    def test_copy_dsa_input_to_npu(self, gsa_worker):
        g = gsa_worker
        ret = g._copy_dsa_input_to_npu([10], [[128, 129]], [[5]], [[6]])
        assert ret == 2
        assert g.indexer_pc_dump_num == 1
        assert g.indexer_pc_dump_index.np[0][0] == 5
        assert g.indexer_pc_dump_index.np[1][0] == 6

    def test_get_dsa_input(self, gsa_worker, monkeypatch):
        g = gsa_worker
        _prime_metadata(g)
        stat = g.gsa_metadata.gsa_stats["r1"]
        stat.indexer_slab_host_slot = [1, 2]
        g.req_ids_bs = ["r1"]
        g.need_prefetch_index = []
        g.gsa_index = []
        assert g.get_dsa_input() == 1

    def test_get_dsa_input_perf_off(self, gsa_worker, monkeypatch):
        g = gsa_worker
        monkeypatch.setattr(gsa_mod, "IS_PERFORMANCE_LOG", False)
        _prime_metadata(g)
        g.gsa_metadata.gsa_stats["r1"].indexer_slab_host_slot = [1, 2]
        g.req_ids_bs = ["r1"]
        g.need_prefetch_index = []
        g.gsa_index = []
        assert g.get_dsa_input() == 1


# ===========================================================================
# get_model_input_all 管线
# ===========================================================================
class TestModelInputPipeline:
    def test_update_prefetch_status_need_prefetch_busy_wait(self, gsa_worker):
        g = gsa_worker
        g.need_prefetch_index = [0]
        g.gsa_index = []
        g.prefetch_engine.prefetch_engine_c.prefetch_status = [True, False]
        g._update_prefetch_status_all()
        assert g.is_prefetch_token is True

    def test_update_prefetch_status_gsa_index_path(self, gsa_worker):
        g = gsa_worker
        g.need_prefetch_index = []
        g.gsa_index = [0]
        g.prefetch_status_sm.version_same = True
        g._update_prefetch_status_all()
        sm = g.prefetch_status_sm
        assert sm.status_calls and sm.status_calls[0][0] is True  # is_done
        assert sm.version_calls  # update_version 已调用
        assert g.is_prefetch_token is True

    def test_update_prefetch_status_gsa_index_version_mismatch(self, gsa_worker):
        g = gsa_worker
        g.need_prefetch_index = []
        g.gsa_index = [0]
        g.prefetch_status_sm.version_same = False
        g._update_prefetch_status_all()
        assert g.is_prefetch_token is False

    def test_update_prefetch_status_both_empty_noop(self, gsa_worker):
        g = gsa_worker
        g.need_prefetch_index = []
        g.gsa_index = []
        g._update_prefetch_status_all()  # 不应触碰任何状态
        assert g.prefetch_status_sm.status_calls == []

    def test_reset_gsa_buffers_all(self, gsa_worker):
        g = gsa_worker
        g.prefetch_topk = [1]
        g.warmup_topk = [2]
        g.swap_index = [3]
        g.select_block_index.np[:] = 9
        g.gsa_swap_index.np[:] = 9
        g.gsa_prefetch_index.np[:] = 9
        g._reset_gsa_buffers_all()
        assert g.prefetch_topk == [] and g.warmup_topk == [] and g.swap_index == []
        assert (g.select_block_index.np == 0).all()
        assert (g.gsa_swap_index.np == 0).all()
        assert (g.gsa_prefetch_index.np == 0).all()

    def test_init_indexer_offload_all_disabled(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", False)
        gsa_worker._init_indexer_offload_all()  # 早退

    def _offload_gsa(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        return make_gsa()

    def test_init_indexer_offload_step_time_multiple(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        g.prefetch_engine.step_time = 10  # 10 % 10 == 0
        g._init_indexer_offload_all()
        assert g.is_prefetch_indexer is True
        assert g.indexer_last_q_index_fb == [0]
        assert g.need_indexer_sparse_bs == []
        assert (g.indexer_last_q_index.np == 0).all()

    def test_init_indexer_offload_loop_match(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 64)
        g = self._offload_gsa(make_gsa, monkeypatch)
        _, stat = _prime_metadata(g, output=0)  # PD 分离 → last chunk
        g.req_ids_bs = ["r1"]
        g.prefetch_engine.step_time = 1  # 非 10 倍数 → 走循环
        g._init_indexer_offload_all()
        assert g.is_prefetch_indexer is True

    def test_init_indexer_offload_loop_no_match(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 64)
        g = self._offload_gsa(make_gsa, monkeypatch)
        _, stat = _prime_metadata(g, output=1)  # 非 last chunk
        g.req_ids_bs = ["r1"]
        g.prefetch_engine.step_time = 1
        g._init_indexer_offload_all()
        assert g.is_prefetch_indexer is False

    def test_collect_and_swap_skip_short_prompt(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        _prime_metadata(g, prompt=10, computed=10, scheduled=1, output=1)
        g.req_ids_bs = ["r1"]
        req_cnt, _, _ = g._collect_and_swap_tokens_all()
        assert req_cnt == 0

    def test_collect_and_swap_skip_prefill_not_last(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        # PD 分离：output 0 → last chunk，但 stage PREFILL → 跳过
        _prime_metadata(g, prompt=200, computed=50, scheduled=10, output=0)
        g.req_ids_bs = ["r1"]
        req_cnt, _, _ = g._collect_and_swap_tokens_all()
        assert req_cnt == 0

    def test_collect_and_swap_full(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        _, stat = _prime_metadata(g, output=0)  # last chunk → warmup 也收集
        stat.is_gsa_init = True
        g.req_ids_bs = ["r1"]
        req_cnt, _, _ = g._collect_and_swap_tokens_all()
        assert req_cnt == 1
        assert g.warmup_topk == [[[0], 1, "r1"]]
        assert g.swap_index == [0]
        assert g.select_block_index.np[0] == 0
        assert g.prefetch_engine.prefetch_engine_c.called("pre_swap_tokens")

    def test_collect_and_swap_indexer_meta(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 64)
        g = make_gsa()
        _, stat = _prime_metadata(g, output=0)  # last chunk → 收集 indexer meta
        stat.is_gsa_init = False  # 不收集 gsa tokens
        g.req_ids_bs = ["r1"]
        g.prefetch_engine.step_time = 10  # 10 的倍数 → is_prefetch_indexer
        g._init_indexer_offload_all()  # 初始化 indexer 状态列表
        req_cnt, _, _ = g._collect_and_swap_tokens_all()
        assert req_cnt == 0
        assert g.need_cal_k_mean == ["r1"]
        assert g.need_indexer_sparse_bs == [0]
        assert g.need_indexer_sparse_reqs == ["r1"]

    def test_collect_and_swap_perf_off(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        monkeypatch.setattr(gsa_mod, "IS_PERFORMANCE_LOG", False)
        g = gsa_worker
        _, stat = _prime_metadata(g, output=0)
        stat.is_gsa_init = True
        g.req_ids_bs = ["r1"]
        req_cnt, _, time_upd = g._collect_and_swap_tokens_all()
        assert req_cnt == 1
        assert time_upd == 0  # perf 关闭时不计时

    def test_collect_indexer_meta_matrix(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "INDEXER_PREFETCH_TOKENS", 64)
        g = self._offload_gsa(make_gsa, monkeypatch)

        def _case(is_prefetch_indexer, output, expect_tmp, expect_fb):
            g.need_cal_k_mean = []
            g.need_indexer_sparse_bs = []
            g.need_indexer_sparse_reqs = []
            g.indexer_last_q_index_tmp = []
            g.indexer_last_q_index_fb = [0]
            g.is_prefetch_indexer = is_prefetch_indexer
            _, stat = _prime_metadata(g, output=output)
            g._collect_indexer_meta("r1", stat, 0, [0, 1])
            assert g.need_cal_k_mean == ["r1"]
            if is_prefetch_indexer:
                assert g.need_indexer_sparse_bs == [0]
                assert g.need_indexer_sparse_reqs == ["r1"]
                assert g.indexer_last_q_index_tmp == [expect_tmp]
                assert g.indexer_last_q_index_fb == expect_fb
            else:
                assert g.need_indexer_sparse_bs == []
                assert g.indexer_last_q_index_tmp == []

        _case(True, 0, [0], [0, 1])  # last chunk
        _case(True, 1, [0], [0, 1])  # 非 last chunk（range 形式）
        _case(False, 0, None, None)  # 非 prefetch indexer

    def test_collect_indexer_meta_short_prompt_early_return(
        self, make_gsa, monkeypatch
    ):
        g = self._offload_gsa(make_gsa, monkeypatch)
        g.need_cal_k_mean = []
        _, stat = _prime_metadata(g, prompt=10, computed=10, output=0)
        g._collect_indexer_meta("r1", stat, 0, [0, 1])  # 10 <= 32768
        assert g.need_cal_k_mean == []

    def test_collect_warmup_topk_branches(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        g.warmup_topk = []
        # 非 last chunk → 早退
        _, stat = _prime_metadata(g, output=1)
        g._collect_warmup_topk("r1", stat, 0, [0, 10])
        assert g.warmup_topk == []
        # last chunk 但 prompt <= SEG → 早退
        _, stat = _prime_metadata(g, prompt=10, computed=10, output=0)
        g._collect_warmup_topk("r1", stat, 0, [0, 10])
        assert g.warmup_topk == []
        # 完整路径：scheduled 5 → 取尾部 5 个
        _, stat = _prime_metadata(g, scheduled=5, output=0)
        g._collect_warmup_topk("r1", stat, 0, [0, 10])
        assert g.warmup_topk == [[[5, 6, 7, 8, 9], 5, "r1"]]

    def test_collect_gsa_tokens_prefetch_mode(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        _, stat = _prime_metadata(g, output=1)
        stat.is_gsa_init = True
        g.is_prefetch_token = True
        g.prefetch_topk = []
        before = stat.use_tokens
        before_prefetch = stat.prefetch_use_token
        req_cnt = g._collect_gsa_tokens("r1", stat, 0, [0, 1], 0, [], [], [], [])
        assert req_cnt == 1
        assert stat.use_tokens is before_prefetch  # 双缓冲交换
        assert stat.prefetch_use_token is before
        assert g.prefetch_engine.prefetch_engine_c.called("set_topk_indices")
        assert g.prefetch_topk == [[[0], "r1"]]

    def test_collect_gsa_tokens_normal_mode(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        _, stat = _prime_metadata(g, output=1)
        stat.is_gsa_init = True
        g.is_prefetch_token = False
        use_ptrs, seq_lens, swap_index, use_req_ids = [], [], [], []
        g.select_block_index.np[:] = 0
        req_cnt = g._collect_gsa_tokens(
            "r1", stat, 0, [0, 1], 0, use_ptrs, seq_lens, swap_index, use_req_ids
        )
        assert req_cnt == 1
        assert use_ptrs == [stat.use_tokens.ptr]
        assert seq_lens == [1 + gsa_mod.GSA_SPARSE_LEN]
        assert swap_index == [[0, 1]]
        assert use_req_ids == ["r1"]
        assert g.swap_index == [0]

    def test_collect_gsa_tokens_not_init_early_return(self, gsa_worker):
        g = gsa_worker
        _, stat = _prime_metadata(g, output=1)
        stat.is_gsa_init = False
        assert g._collect_gsa_tokens("r1", stat, 0, [0, 1], 0, [], [], [], []) == 0

    def test_pre_swap_tokens_all_empty(self, gsa_worker):
        gsa_worker._pre_swap_tokens_all([], [], [], [])  # 早退
        assert gsa_worker.gsa_swap_topk_cpu_ptr is None

    def test_pre_swap_tokens_all_ptr_init_and_reuse(self, gsa_worker):
        g = gsa_worker
        stat_ptr = object()
        g._pre_swap_tokens_all([stat_ptr], [100], [[0, 1]], ["r1"])
        cpu_ptr, npu_ptr = g.gsa_swap_topk_cpu_ptr, g.gsa_swap_topk_npu_ptr
        assert cpu_ptr is not None and npu_ptr is not None
        # 再次调用：ptr 已初始化，直接提交
        g._pre_swap_tokens_all([stat_ptr], [100], [[0, 1]], ["r1"])
        assert g.gsa_swap_topk_cpu_ptr == cpu_ptr
        assert len(g.prefetch_engine.prefetch_engine_c.called("pre_swap_tokens")) == 2

    def test_copy_gsa_buffers_to_npu_empty(self, gsa_worker):
        g = gsa_worker
        g.prefetch_topk = []
        g.warmup_topk = []
        g.swap_index = []
        g.num_padding_reqs = 4
        assert g._copy_gsa_buffers_to_npu(0) == (0, 0, 0)
        assert (g.gsa_swap_index.np[:] == 4).all()

    def test_copy_gsa_buffers_to_npu_full(self, gsa_worker):
        g = gsa_worker
        g.swap_index = [0, 1]
        g.prefetch_topk = [[[2], "r1"]]
        g.warmup_topk = [[[3, 4], 2, "r1"]]
        g.num_padding_reqs = 1
        prefetch_len, warmup_len, indexer_len = g._copy_gsa_buffers_to_npu(1)
        assert (prefetch_len, warmup_len, indexer_len) == (1, 2, 0)
        assert g.gsa_prefetch_index.np[0] == 2
        assert g.gsa_warmup_index.np[:2].tolist() == [3, 4]
        assert g.gsa_swap_index.np[:2].tolist() == [0, 1]
        assert (g.gsa_swap_index.np[2:] == 1).all()

    def test_copy_gsa_buffers_to_npu_offload(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        g.swap_index = []
        g.prefetch_topk = []
        g.warmup_topk = []
        g.indexer_last_q_index_tmp = [[7, 8]]
        g.num_padding_reqs = 0
        assert g._copy_gsa_buffers_to_npu(0)[2] == 2
        assert g.indexer_last_q_index.np[:2].tolist() == [7, 8]

    def test_update_attn_gsa_lens(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        attn0, attn1 = SimpleNamespace(), SimpleNamespace()
        g.swap_index = [0]
        g.select_block_index.np[0] = 3
        g._update_attn_gsa_lens(
            {"a": attn0, "b": attn1},
            gsa_slot_len=5,
            gsa_prefetch_len=6,
            gsa_warmup_len=7,
            indexer_len=8,
            req_cnt=1,
        )
        for attn in (attn0, attn1):
            assert attn.gsa_prefetch_len == 6
            assert attn.gsa_swap_len == 1
            assert attn.gsa_warmup_len == 7
            assert attn.gsa_slot_len == 5
            assert attn.indexer_last_query_len == 8
            # select_block_index 传递的是 npu 侧切片（此处仅验证形状/归属）
            assert attn.select_block_index.shape[0] == 1
            assert (
                attn.select_block_index.data_ptr()
                == g.select_block_index.npu[:1].data_ptr()
            )

    def test_get_model_input_all_full(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        _, stat = _prime_metadata(g, output=0)
        stat.is_gsa_init = True
        g.req_ids_bs = ["r1"]
        g.need_prefetch_index = []
        g.gsa_index = []
        g.num_padding_reqs = 1
        attn = SimpleNamespace()
        g.get_model_input_all({"a": attn}, gsa_slot_len=1)
        assert attn.gsa_swap_len == 1
        assert attn.gsa_slot_len == 1
        assert g.prefetch_engine.prefetch_engine_c.called("pre_swap_tokens")

    def test_get_model_input_all_perf_off(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        monkeypatch.setattr(gsa_mod, "IS_PERFORMANCE_LOG", False)
        g = gsa_worker
        _, stat = _prime_metadata(g, output=0)
        stat.is_gsa_init = True
        g.req_ids_bs = ["r1"]
        g.need_prefetch_index = []
        g.gsa_index = []
        g.num_padding_reqs = 1
        attn = SimpleNamespace()
        g.get_model_input_all({"a": attn}, gsa_slot_len=1)  # 无计时日志路径
        assert attn.gsa_swap_len == 1


# ===========================================================================
# build_gsa_metadata / execute_begin
# ===========================================================================
class TestBuildMetaAndExecute:
    def _prepare(self, g, req_id="r1", **kw):
        md, stat = _prime_metadata(g, **kw)
        stat.slab_host_slot = []
        stat.indexer_slab_host_slot = []
        stat.req_dump_blocks = []
        g.gsa_stats = md.gsa_stats
        return stat

    def test_build_gsa_metadata_preempted(self, gsa_worker):
        g = gsa_worker
        stat = self._prepare(g)
        g._block_hash_injected_reqs.add("r1")
        so = fakes.FakeSchedulerOutput(
            num_scheduled_tokens={}, preempted_req_ids=["r1"]
        )
        requests = {"r1": SimpleNamespace(output_token_ids=[1, 2, 3])}
        gsa_meta = g.build_gsa_metadata(so, requests, fakes.FakeInputBatch({}))
        assert "r1" not in g.gsa_stats
        assert g.preempt_req_output_tokens["r1"] == 3
        # del_finish_meta 下沉到 C 层的调用名为 del_blocks_map
        assert g.prefetch_engine.prefetch_engine_c.called("del_blocks_map")
        assert "r1" not in g._block_hash_injected_reqs
        assert g.gsa_stats is gsa_meta.gsa_stats

    def test_build_gsa_metadata_preempted_busy_wait(self, gsa_worker):
        g = gsa_worker
        stat = self._prepare(g)
        ec = g.prefetch_engine.prefetch_engine_c
        ec.prefetch_status = [True, False]  # prefetch 忙等一次
        so = fakes.FakeSchedulerOutput(
            num_scheduled_tokens={}, preempted_req_ids=["r1"]
        )
        requests = {"r1": SimpleNamespace(output_token_ids=[])}
        g.build_gsa_metadata(so, requests, fakes.FakeInputBatch({}))
        assert len(ec.called("get_prefetch_status")) == 2

    def test_build_gsa_metadata_preempted_indexer_busy(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        stat = self._prepare(g)
        ec = g.prefetch_engine.prefetch_engine_c
        ec.prefetch_status = [False]
        ec.indexer_status = [True, False]  # 忙等一次后空闲
        so = fakes.FakeSchedulerOutput(
            num_scheduled_tokens={}, preempted_req_ids=["r1"]
        )
        requests = {"r1": SimpleNamespace(output_token_ids=[])}
        g.build_gsa_metadata(so, requests, fakes.FakeInputBatch({}))
        assert len(ec.called("get_indexer_status")) == 2

    def test_build_gsa_metadata_normal_flow(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        self._prepare(g)
        g.gsa_stats = {}
        so = fakes.FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1},
            new_reqs=[SimpleNamespace(req_id="r1")],
        )
        add = fakes.FakeAddReqState(
            block_ids=[[5, 6]],
            prompt_token_ids=list(range(200)),
            output_token_ids=[1],
            num_computed_tokens=200,
        )
        gsa_meta = g.build_gsa_metadata(
            so, {"r1": add}, fakes.FakeInputBatch({"r1": 0})
        )
        assert "r1" in gsa_meta.gsa_stats
        assert g.model_input["query_locals"][1] == 1

    def test_release_preempted_not_in_stats(self, gsa_worker):
        g = gsa_worker
        g.gsa_stats = {}
        g._release_preempted_gsa_stats("missing")  # 早退
        assert g.gsa_stats == {}

    def test_release_preempted_rank0(self, gsa_worker):
        g = gsa_worker
        stat = self._prepare(g)
        i_free = g.indexer_cpu_shared_pool.get_free_num()
        g._release_preempted_gsa_stats("r1")
        assert "r1" not in g.gsa_stats
        assert g.indexer_cpu_shared_pool.get_free_num() == i_free
        assert g.slots_num_manager.update_calls  # update_slot_num 已执行

    def test_release_preempted_rank_nonzero(self, make_gsa):
        g = make_gsa(rank=1)
        md, stat = _prime_metadata(g)
        stat.req_dump_blocks = []
        g.gsa_stats = md.gsa_stats
        g._release_preempted_gsa_stats("r1")
        assert "r1" not in g.gsa_stats
        assert not hasattr(g, "slots_num_manager")  # rank≠0 不更新

    def test_execute_begin_early_return_ptopk_off(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "PTOPK_PREFETCH_ENABLE", False)
        gsa_worker.execute_begin(fakes.FakeSchedulerOutput(), {})  # 早退

    def test_execute_begin_early_return_not_decode(self, make_gsa):
        g = make_gsa(consumer=False)
        g.execute_begin(fakes.FakeSchedulerOutput(), {})  # 早退

    def test_execute_begin_full(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        stat = self._prepare(g)  # prompt 200, computed 200, output 1
        stat.blocks = [1, 2]
        stat.topk_buf_tmp = torch.zeros((2, 8), dtype=torch.float16)
        g.need_prefetch_index = []
        g.gsa_index = []
        so = fakes.FakeSchedulerOutput(num_scheduled_tokens={"r1": 1})
        attn = SimpleNamespace(num_input_tokens=1)
        g.execute_begin(so, {"a": attn})
        assert g.req_ids_bs == ["r1"]
        assert stat.is_gsa_init is True
        # last_chunk_init 对每层调用一次 set_tokens_map
        assert (
            len(g.prefetch_engine.prefetch_engine_c.called("set_tokens_map"))
            == g.layer_num
        )
        # block_len = ceil(201/128) - 0 = 2 → rank0 分配 indexer slots
        assert len(stat.indexer_slab_host_slot) == 2
        assert g.num_padding_reqs == 1
        assert attn.gsa_swap_len == 1  # get_model_input_all 已写回 attn
        assert g.prefetch_engine.prefetch_engine_c.called("pre_swap_tokens")

    def _call_process(self, g, req_id):
        g._process_scheduled_req(req_id, [None], [None], [None], [None], [None])

    def test_process_scheduled_req_offload_long_prompt(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = make_gsa()
        stat = self._prepare(g)
        stat.slab_host_slot = [5]  # 已有 1 块 → block_len = 2-1 = 1
        self._call_process(g, "r1")
        # rank0：gsa pool 与 indexer pool 各分配 1 块
        assert len(stat.slab_host_slot) == 2
        assert len(stat.indexer_slab_host_slot) == 1
        assert stat.add_blocks_len == 1

    def test_process_scheduled_req_offload_rank_nonzero(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = make_gsa(rank=1)
        stat = self._prepare(g)
        stat.slab_host_slot = []
        req_ids, add_blocks, new_len = [None], [None], [None]
        i_add, i_len = [None], [None]
        g._process_scheduled_req("r1", req_ids, add_blocks, new_len, i_add, i_len)
        assert add_blocks[0] == []  # 非 rank0：无 gsa 池分配（保持空）
        assert new_len[0] == 2
        assert i_add[0] == [] and i_len[0] == 2

    def test_process_scheduled_req_block_len_zero(self, gsa_worker):
        g = gsa_worker
        stat = self._prepare(g)
        stat.indexer_slab_host_slot = [1, 2]  # 已够 → block_len 0
        before = g.indexer_cpu_shared_pool.get_free_num()
        self._call_process(g, "r1")
        assert stat.add_blocks_len == 0
        assert g.indexer_cpu_shared_pool.get_free_num() == before

    def test_sync_block_tables_rank0_non_offload(self, gsa_worker):
        g = gsa_worker
        g._sync_block_tables(["r1"], [None], [None], [[7]], [1])
        assert g.indexer_block_table_sm.written == [[7]]

    def test_sync_block_tables_rank0_offload(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        g._sync_block_tables(["r1"], [[7]], [1], [[8]], [1])
        assert g.block_table_sm.written == [[7]]
        assert g.indexer_block_table_sm.written == [[8]]

    def test_sync_block_tables_rank_nonzero_read_none(self, make_gsa):
        g = make_gsa(rank=1)
        md, stat = _prime_metadata(g)
        g.gsa_stats = md.gsa_stats
        g.indexer_block_table_sm.read_result = None
        g._sync_block_tables(["r1"], [None], [None], [None], [None])
        assert stat.slab_host_slot == []
        assert stat.indexer_slab_host_slot == []

    def test_sync_block_tables_rank_nonzero_apply(self, make_gsa):
        g = make_gsa(rank=1)
        md, stat = _prime_metadata(g)
        g.gsa_stats = md.gsa_stats
        g.indexer_block_table_sm.read_result = [[7, 8]]
        g._sync_block_tables(["r1"], [None], [None], [None], [2])
        assert stat.indexer_slab_host_slot == [7, 8]

    def test_sync_block_tables_offload_rank_nonzero(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa(rank=1)
        md, stat = _prime_metadata(g)
        g.gsa_stats = md.gsa_stats
        g.block_table_sm.read_result = [[5]]
        g.indexer_block_table_sm.read_result = [[6]]
        g._sync_block_tables(["r1"], [None], [1], [None], [1])
        assert stat.slab_host_slot == [5]
        assert stat.indexer_slab_host_slot == [6]

    def test_apply_add_blocks(self, gsa_worker):
        g = gsa_worker
        md, stat = _prime_metadata(g)
        g.gsa_metadata = md
        g._apply_add_blocks(["r1", "r2"], [[7, 8], [9]], [2, 0], is_indexer=False)
        assert stat.slab_host_slot == [7, 8]
        # 第二项 new_blocks_len == 0 → continue（r2 不在 stats 也不报错）

    def test_apply_add_blocks_indexer(self, gsa_worker):
        g = gsa_worker
        md, stat = _prime_metadata(g)
        g.gsa_metadata = md
        g._apply_add_blocks(["r1"], [[7]], [1], is_indexer=True)
        assert stat.indexer_slab_host_slot == [7]

    def test_init_dsa_data_non_offload(self, gsa_worker):
        g = gsa_worker
        mr = SimpleNamespace()
        g.init_dsa_data(mr)
        assert mr.gsa_swap_topk is g.gsa_swap_topk
        assert mr.dsa_slot_mapping is g.dsa_slot_mapping
        assert not hasattr(mr, "indexer_last_query")

    def test_init_dsa_data_offload(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        mr = SimpleNamespace()
        g.init_dsa_data(mr)
        assert mr.indexer_last_query is g.indexer_last_query
        assert mr.indexer_last_q_index is g.indexer_last_q_index

    def test_init_dsa_data_log_disabled(self, gsa_worker):
        g = gsa_worker
        calls = []
        g.log = SimpleNamespace(
            is_info_enabled=False, info=lambda *a, **k: calls.append(a)
        )
        g.init_dsa_data(SimpleNamespace())
        assert calls == []


# ===========================================================================
# indexer offload 计算
# ===========================================================================
class TestIndexer:
    def _offload_gsa(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        for layer in range(g.layer_num):
            g.indexer_k_cache_sparse[layer] = torch.zeros(
                (16, 128, 1, 128), dtype=torch.float16
            )
        return g

    def _indexer_stat(self, g, output=0, sparse_len=8):
        md, stat = _prime_metadata(g, output=output)
        stat.slab_host_slot = [1]
        stat.indexer_slab_host_slot = [2]
        stat.prompt_sparse_len = sparse_len
        return stat

    # ---------------- cal_indexer_k_repr_mean_prefill ----------------
    def test_mean_prefill_early_return(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", False)
        gsa_worker.cal_indexer_k_repr_mean_prefill()

    def test_mean_prefill_rank0(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        stat = self._indexer_stat(g, output=0)  # last chunk
        g.need_cal_k_mean = ["r1"]
        g.cal_indexer_k_repr_mean_prefill()
        # rank0：释放旧 slot，按 ceil(8*1.5)+1=13 分配
        assert len(stat.indexer_slab_host_slot) == 13
        assert g.indexer_block_table_sm.written is not None
        # k_repr 已按 slab_host_slot 写入均值
        assert (g.indexer_k_repr[0][1] == 0).all()

    def test_mean_prefill_rank0_not_last_chunk(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        stat = self._indexer_stat(g, output=1)  # 非 last chunk → 早退
        g.need_cal_k_mean = ["r1"]
        g.cal_indexer_k_repr_mean_prefill()
        assert stat.indexer_slab_host_slot == [2]  # 未重新分配
        assert g.indexer_block_table_sm.written == [[]]  # 空表仍会写出

    def test_mean_prefill_rank_nonzero(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa(rank=1)
        for layer in range(g.layer_num):
            g.indexer_k_cache_sparse[layer] = torch.zeros(
                (16, 128, 1, 128), dtype=torch.float16
            )
        stat = self._indexer_stat(g, output=0)
        g.need_cal_k_mean = ["r1"]
        g.indexer_block_table_sm.read_result = [[7]]
        g.cal_indexer_k_repr_mean_prefill()
        assert stat.indexer_slab_host_slot == [7]  # 应用对端结果

    def test_apply_indexer_add_blocks_none(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        g._apply_indexer_add_blocks(["r1"], None, [1])  # 早退

    def test_apply_indexer_add_blocks_skips_zero_len(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        md, stat1 = _prime_metadata(g)
        md.gsa_stats["r2"] = gsa_mod.GSAReqStat("r2", g._vllm_config, 2)
        stat1.indexer_slab_host_slot = [9]  # len==0 的请求不应被覆盖
        g._apply_indexer_add_blocks(["r1", "r2"], [[], [7]], [0, 1])
        assert stat1.indexer_slab_host_slot == [9]
        assert md.gsa_stats["r2"].indexer_slab_host_slot == [7]

    # ---------------- cal_indexer_k_repr_mean_decode ----------------
    def test_mean_decode_early_return(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", False)
        gsa_worker.cal_indexer_k_repr_mean_decode()

    def test_mean_decode_full(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        stat = self._indexer_stat(g, output=0, sparse_len=2)
        stat.slab_host_slot = [3, 4, 5, 6]
        stat.prompt_block_len = 4
        stat.include_mask = [True, False, True, False]
        stat.exclude_mask = [False] * 4
        g.need_cal_k_mean = ["r1"]
        g.need_indexer_sparse_reqs = ["r1"]
        g.cal_indexer_k_repr_mean_decode()
        assert g.max_topk_len == [3]  # ceil(2*1.5)
        assert g.include_masks.tolist() == [[1, 0, 1, 0]]
        # 单行数据即最大长度：无 padding 发生
        assert g.exclude_masks.tolist() == [[0, 0, 0, 0]]
        assert g.repre_slot_mappings.tolist() == [[3, 4, 5, 6]]

    def test_mean_decode_not_last_chunk(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        stat = self._indexer_stat(g, output=1, sparse_len=4)
        stat.slab_host_slot = [3]
        stat.prompt_block_len = 1
        g.need_cal_k_mean = []
        g.need_indexer_sparse_reqs = ["r1"]
        g.cal_indexer_k_repr_mean_decode()
        assert g.max_topk_len == [4]  # 非 last chunk → prompt_sparse_len

    # ---------------- cal_indexer_topk_blocks ----------------
    def test_topk_blocks_early_returns(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", False)
        gsa_worker.cal_indexer_topk_blocks()  # offload 关闭
        g = gsa_worker
        g.need_indexer_sparse_bs = []
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g.cal_indexer_topk_blocks()  # bs == 0

    def _setup_topk(self, g, stat_kw=None):
        # 源码疑似问题：_init_worker_indexer_state 中 indexer_last_query 的
        # None 赋值晚于 init_gsa_metadata 的创建语句，构造完成后恒为 None；
        # 此处手动重建以覆盖 cal_indexer_topk_blocks 主路径
        g.indexer_last_query = gsa_mod.TopkIndices(
            (g.layer_num, 4, 32, 128), g.device, g.dtype
        )
        g.indexer_last_weight = gsa_mod.TopkIndices(
            (g.layer_num, 4, 32), g.device, g.dtype
        )
        g.max_topk_len = [3]  # 通常由 cal_indexer_k_repr_mean_decode 设置
        stat = self._indexer_stat(g, **(stat_kw or {}))
        stat.slab_host_slot = [3, 4, 5, 6]
        stat.prompt_block_len = 4
        stat.prompt_sparse_len = 2
        stat.include_mask = [False, False, False, False]
        stat.exclude_mask = [False, False, False, False]
        g.need_indexer_sparse_bs = [0]
        g.need_indexer_sparse_reqs = ["r1"]
        g.repre_slot_mappings = gsa_mod.make_tensor_with_pad(
            [[3, 4, 5, 6]], pad=0, dtype=torch.int32, device="cpu"
        )
        g.include_masks = gsa_mod.make_tensor_with_pad(
            [[False] * 4], pad=False, dtype=torch.uint8, device="cpu"
        )
        g.exclude_masks = gsa_mod.make_tensor_with_pad(
            [[False] * 4], pad=True, dtype=torch.uint8, device="cpu"
        )
        # 有区分度的 key 表征 → topk 结果确定
        torch.manual_seed(42)
        for layer in range(g.layer_num):
            g.indexer_k_repr[layer].normal_()
        return stat

    def test_topk_blocks_spec1_full(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        stat = self._setup_topk(g)
        g.cal_indexer_topk_blocks()
        topk = stat.indexer_topk_blocks[:, :3]
        assert topk.shape == (2, 3)
        # 三个互异的块索引
        for row in topk:
            assert len(set(row.tolist())) == 3
            assert all(0 <= x < 4 for x in row.tolist())

    def test_topk_blocks_include_mask_inf(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        stat = self._setup_topk(g)
        # include 位置 0 → inf：topk 第一名必为块 0
        g.include_masks = gsa_mod.make_tensor_with_pad(
            [[True, False, False, False]], pad=False, dtype=torch.uint8, device="cpu"
        )
        g.cal_indexer_topk_blocks()
        assert stat.indexer_topk_blocks[:, 0].tolist() == [0, 0]

    def test_topk_blocks_spec_with_rec_result(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa(speculative=fakes.FakeSpeculativeConfig(2))  # spec=3
        for layer in range(g.layer_num):
            g.indexer_k_cache_sparse[layer] = torch.zeros(
                (16, 128, 1, 128), dtype=torch.float16
            )
        stat = self._setup_topk(g)
        g.indexer_last_q_index_fb = [0, 2]
        # last_q_index_fb[0] + rec_result[0] = 0+1 = 1 → 取 last_query[:, 1]
        g.cal_indexer_topk_blocks(rec_result=[1])
        assert stat.indexer_topk_blocks[:, :3].shape == (2, 3)

    # ---------------- set_indexer_topk_blocks_prefill ----------------
    def test_prefill_topk_early_returns(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", False)
        gsa_worker.set_indexer_topk_blocks_prefill()
        g = gsa_worker
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g.need_indexer_sparse_reqs = []
        g.set_indexer_topk_blocks_prefill()

    def test_prefill_topk_not_last_chunk_continue(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        stat = self._indexer_stat(g, output=1)  # 非 last chunk → continue
        g.need_indexer_sparse_reqs = ["r1"]
        g.set_indexer_topk_blocks_prefill()
        assert stat.is_indexer_init is False
        assert not g.prefetch_engine.prefetch_engine_c.called("set_indexer_map")

    def test_prefill_topk_full(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        stat = self._indexer_stat(g, output=0)
        stat.slab_host_slot = [5, 6, 7]
        stat.indexer_slab_host_slot = [9, 8]
        stat.prompt_sparse_len = 2
        # 预填 topk：remain 区 [0,1]，prefetch 区 [2]
        stat.indexer_topk_blocks[:] = 0
        stat.indexer_topk_blocks[:, 0] = 1
        stat.indexer_topk_blocks[:, 1] = 0
        stat.indexer_topk_blocks[:, 2] = 2
        g.need_indexer_sparse_reqs = ["r1"]
        g.set_indexer_topk_blocks_prefill()
        assert stat.is_indexer_init is True
        # prefetch_idx [2] + 排序后 remain [0,1] + last_block_idx 2
        assert stat.indexer_device_slot_status == [[2, 0, 1, 2]] * 2
        assert stat.prefetch_indexer_block_tables[:, 2].tolist() == [8, 8]
        calls = g.prefetch_engine.prefetch_engine_c.called("set_indexer_map")
        assert len(calls) == 1
        assert g.prefetch_engine.prefetch_engine_c.called("indexer_init_load")
        assert (
            stat.indexer_block_tables == stat.prefetch_indexer_block_tables_cpu.numpy()
        ).all()

    # ---------------- set_indexer_topk_blocks_decode ----------------
    def test_decode_topk_early_returns(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", False)
        gsa_worker.set_indexer_topk_blocks_decode()
        g = gsa_worker
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g.need_indexer_sparse_reqs = []
        g.set_indexer_topk_blocks_decode()

    def test_decode_topk_indexer_busy_skip(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        stat = self._indexer_stat(g, output=1)  # 非 last chunk
        g.need_indexer_sparse_reqs = ["r1"]
        g.prefetch_engine.prefetch_engine_c.indexer_status = [True]
        g.set_indexer_topk_blocks_decode()  # is_indexer_load=True → 不提交
        assert not g.prefetch_engine.prefetch_engine_c.called("submit_task_indexer")

    def test_decode_topk_submit(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        stat = self._indexer_stat(g, output=1)  # 非 last chunk → 提交
        stat.prompt_sparse_len = 2
        g.need_indexer_sparse_reqs = ["r1"]
        g.prefetch_engine.prefetch_engine_c.indexer_status = [False]
        g.set_indexer_topk_blocks_decode()
        calls = g.prefetch_engine.prefetch_engine_c.called("submit_task_indexer")
        assert len(calls) == 1
        assert calls[0][1][0] == [2]  # topk_lens_input
        assert calls[0][1][1] == ["r1"]

    def test_decode_topk_last_chunk_pass(self, make_gsa, monkeypatch):
        g = self._offload_gsa(make_gsa, monkeypatch)
        stat = self._indexer_stat(g, output=0)  # last chunk → pass
        g.need_indexer_sparse_reqs = ["r1"]
        g.prefetch_engine.prefetch_engine_c.indexer_status = [False]
        g.set_indexer_topk_blocks_decode()  # req_id_list 空 → return
        assert not g.prefetch_engine.prefetch_engine_c.called("submit_task_indexer")

    # ---------------- update_slot_num ----------------
    def test_update_slot_num_rank0_offload(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        k_free = g.gsa_cpu_shared_pool.get_free_num()
        i_free = g.indexer_cpu_shared_pool.get_free_num()
        g.update_slot_num()
        assert g.slots_num_manager.update_calls == [(i_free, k_free)]

    def test_update_slot_num_rank0_non_offload(self, gsa_worker):
        g = gsa_worker
        i_free = g.indexer_cpu_shared_pool.get_free_num()
        g.update_slot_num()
        assert g.slots_num_manager.update_calls == [(i_free, 0)]

    def test_update_slot_num_rank_nonzero(self, make_gsa):
        g = make_gsa(rank=1)
        g.update_slot_num()  # 早退
        assert not hasattr(g, "slots_num_manager")


# ===========================================================================
# warmup
# ===========================================================================
class TestWarmup:
    def _warmup_setup(self, g, offload=False):
        _, stat = _prime_metadata(g, output=0)  # last chunk
        stat.non_ucm_block_count = 2
        stat.original_prompt_block_len = 1
        stat.blocks = [1, 2]
        stat.block_hashes = [bytes([i]) * 16 for i in range(2)]
        stat.indexer_slab_host_slot = [3, 4]
        if offload:
            stat.slab_host_slot = [5, 6]
        g.warmup_topk = [[[0, 1, 2], 3, "r1"]]
        g.req_ids_bs = ["r1"]
        _register_kvcaches(g)
        return stat

    def test_warmup_early_returns(self, gsa_worker, monkeypatch):
        g = gsa_worker
        g.warmup_topk = []
        g.warmup()  # warmup_topk 空 → 早退
        monkeypatch.setattr(gsa_mod, "PTOPK_PREFETCH_ENABLE", False)
        g.warmup()  # 开关关闭 → 早退

    def test_warmup_full(self, gsa_worker, monkeypatch):
        g = gsa_worker
        self._warmup_setup(g)
        g.warmup()  # 非 offload → cal_indexer_k_repr_mean_prefill 早退
        stat = g.gsa_metadata.gsa_stats["r1"]
        assert len(stat.req_dump_blocks) == 2
        assert g.prefetch_engine.prefetch_engine_c.called("set_blocks_map")
        assert len(g.gsa_preempt_shared_pool.allocated) == 2

    def test_warmup_perf_off(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_PERFORMANCE_LOG", False)
        g = gsa_worker
        self._warmup_setup(g)
        g.warmup()  # 无计时路径
        assert g.gsa_metadata.gsa_stats["r1"].req_dump_blocks

    def test_warmup_dump_perf_off_returns_zero(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_PERFORMANCE_LOG", False)
        g = gsa_worker
        self._warmup_setup(g)
        assert g._warmup_dump_blocks() == 0

    def test_warmup_dump_perf_on_returns_time(self, gsa_worker):
        g = gsa_worker
        self._warmup_setup(g)
        assert g._warmup_dump_blocks() >= 0.0  # 返回耗时（us）

    def test_warmup_dump_offload_indexer_copy(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        for layer in range(g.layer_num):
            g.indexer_k_cache_sparse[layer] = torch.zeros(
                (16, 128, 1, 128), dtype=torch.float16
            )
        stat = self._warmup_setup(g, offload=True)
        # offload + inject_indexer：prompt 200 <= 32768 → False 侧
        g._warmup_dump_blocks()
        assert not g.prefetch_engine.prefetch_engine_c.called("set_last_indexer_blocks")
        # prompt > INDEXER_PREFETCH_TOKENS → inject_indexer True 侧
        stat.num_prompt_tokens = 40000
        stat.block_hashes = [bytes([i]) * 16 for i in range(2)]
        g._block_hash_injected_reqs.clear()
        g.indexer_cache_preempt_i_ptr = 3000
        g.indexer_cache_preempt_i_offset = 30
        g._warmup_dump_blocks()
        assert g.prefetch_engine.prefetch_engine_c.called("set_last_indexer_blocks")

    def test_warmup_dump_sfa_c8(self, make_gsa):
        g = make_gsa(additional_config={"enable_sparse_sfa_c8": True})
        stat = self._warmup_setup(g)
        g._warmup_dump_blocks()
        # sfa_c8：_slab_preempt_v 与 k 共享，不额外分配 v 缓冲
        assert g._slab_preempt_v is g._slab_preempt_k
        assert g._slab_preempt_v_ptr == g._slab_preempt_k_ptr

    def test_get_vllm_blocks_branches(self, gsa_worker, make_gsa, monkeypatch):
        g = gsa_worker
        stat = SimpleNamespace(
            blocks=[1, 2, 3], slab_host_slot=[], indexer_slab_host_slot=[2]
        )
        # 非 offload + spec==1 → 原样返回
        assert g._get_vllm_blocks(stat) == [1, 2, 3]
        # spec>1 且 blocks 多于 slots → blocks[:-1]
        gs = make_gsa(speculative=fakes.FakeSpeculativeConfig(1))  # spec=2
        assert gs._get_vllm_blocks(stat) == [1, 2]
        # spec>1 但 blocks 不多 → 原样
        stat2 = SimpleNamespace(
            blocks=[1, 2], slab_host_slot=[], indexer_slab_host_slot=[1, 2, 3]
        )
        assert gs._get_vllm_blocks(stat2) == [1, 2]
        # offload → 用 slab_host_slot 长度判断
        stat3 = SimpleNamespace(
            blocks=[1, 2, 3, 4], slab_host_slot=[1], indexer_slab_host_slot=[]
        )
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        assert gs._get_vllm_blocks(stat3) == [1, 2, 3]

    def test_copy_preempt_kv_blocks_no_v(self, make_gsa):
        g = make_gsa(additional_config={"enable_sparse_sfa_c8": True})
        stat = self._warmup_setup(g)
        stat.non_ucm_block_count = 1
        stat.req_dump_blocks = [0]
        g._copy_preempt_kv_blocks(stat, [1], [3])  # sfa → 跳过 v copy
        assert (g._slab_preempt_k[0][0] == 0).all()


# ===========================================================================
# execute_finished 与预取任务提交
# ===========================================================================
class TestExecuteFinished:
    def test_early_return_ptopk_off(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "PTOPK_PREFETCH_ENABLE", False)
        gsa_worker.execute_finished()

    def test_early_return_not_decode(self, make_gsa):
        make_gsa(consumer=False).execute_finished()

    def test_no_prefetch_branch(self, gsa_worker):
        g = gsa_worker
        g.prefetch_topk = []
        g.warmup_topk = []
        g.is_prefetch_token = False
        g.need_cal_k_mean = []
        g.need_indexer_sparse_reqs = []
        g.need_indexer_sparse_bs = []
        g.execute_finished()
        # 非 offload rank0：update_slot_num 已执行（k_num=0）
        assert g.slots_num_manager.update_calls

    def test_prefetch_token_branch(self, gsa_worker):
        g = gsa_worker
        _, stat = _prime_metadata(g, output=1)
        stat.is_gsa_init = True
        g.prefetch_topk = [[[0], "r1"]]
        g.warmup_topk = []
        g.is_prefetch_token = True
        g.execute_finished()
        assert g.prefetch_engine.prefetch_engine_c.called("submit_prefetch_task")

    def test_prefetch_token_branch_perf_off(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_PERFORMANCE_LOG", False)
        g = gsa_worker
        _, stat = _prime_metadata(g, output=1)
        g.prefetch_topk = []
        g.warmup_topk = []
        g.is_prefetch_token = False
        g.execute_finished()  # 无计时路径

    def test_prepare_prefetch_requests_spec1(self, gsa_worker):
        g = gsa_worker
        _, stat = _prime_metadata(g, output=1)
        g.prefetch_topk = [[[0], "r1"]]
        topk_lens, req_ids, offloads, seq_lens = g._prepare_prefetch_requests(None)
        assert topk_lens == [2048]
        assert req_ids == ["r1"]
        assert offloads == [False]
        assert seq_lens == [201]
        assert stat.prefetch_topk_step == g.prefetch_engine.step_time

    def test_prepare_prefetch_requests_spec_with_rec(self, make_gsa):
        g = make_gsa(speculative=fakes.FakeSpeculativeConfig(2))  # spec=3
        _, stat = _prime_metadata(g, output=1)
        g.prefetch_topk = [[[0, 1, 2], "r1"]]
        rec_result = [2]
        topk_lens, req_ids, offloads, seq_lens = g._prepare_prefetch_requests(
            rec_result
        )
        # seq = 201 - 3 + 2 + 1 = 201
        assert seq_lens == [201]
        assert g.select_gsa_prefetch_topk[:, 0].shape == (g.topk_layer_num, 1, 2048)

    def test_prepare_prefetch_requests_offload_flag(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        _, stat = _prime_metadata(g, output=1)
        stat.is_indexer_init = True
        g.prefetch_topk = [[[0], "r1"]]
        _, _, offloads, _ = g._prepare_prefetch_requests(None)
        assert offloads == [True]

    def test_submit_prefetch_task_sum_zero(self, gsa_worker):
        g = gsa_worker
        g.topk_indices_len = 0
        g.prefetch_space_token = 5
        g._submit_prefetch_task([0], ["r1"], [False], [None])
        assert not g.prefetch_engine.prefetch_engine_c.called("submit_prefetch_task")
        assert g.prefetch_space_token == 5  # 未重置

    def test_submit_prefetch_task_spec1(self, gsa_worker):
        g = gsa_worker
        g.prefetch_space_token = 5
        g._submit_prefetch_task([2048], ["r1"], [False], [201])
        calls = g.prefetch_engine.prefetch_engine_c.called("submit_prefetch_task")
        assert len(calls) == 1
        assert calls[0][1][3] is g.gsa_prefetch_topk.cpu
        assert g.prefetch_space_token == 0

    def test_submit_prefetch_task_spec_gt1(self, make_gsa):
        g = make_gsa(speculative=fakes.FakeSpeculativeConfig(2))
        g._submit_prefetch_task([2048], ["r1"], [False], [201])
        calls = g.prefetch_engine.prefetch_engine_c.called("submit_prefetch_task")
        assert calls[0][1][3] is g.select_gsa_prefetch_topk


# ===========================================================================
# 调度侧接口
# ===========================================================================
class _FakeCoordinator:
    def __init__(self, num_blocks=4):
        self.num_blocks = num_blocks
        self.calls = []

    def get_num_blocks_to_allocate(self, **kwargs):
        self.calls.append(kwargs)
        return self.num_blocks

    def allocate_new_blocks(self, req_id, num_tokens, num_encoder):
        self.calls.append(("allocate", req_id, num_tokens))

    single_type_managers = property(
        lambda self: [SimpleNamespace(req_to_blocks={1: [10, 11]})]
    )


class _FakeBlockPool:
    def __init__(self, free_blocks=100):
        self.free_blocks = free_blocks

    def get_num_free_blocks(self):
        return self.free_blocks


def _fake_kv_cache_manager(coordinator=None, block_pool=None):
    return SimpleNamespace(
        coordinator=coordinator or _FakeCoordinator(),
        block_pool=block_pool or _FakeBlockPool(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object(), object()]),
        free=lambda req: None,
    )


class _SelfRefNewReq:
    """build_sparse_meta 把 scheduled_new_reqs 的元素（NewRequestData 对象）
    原样传给 _load_kv_dram，而后者以 req_id 为键查 gsa_stats —— 疑似源码
    问题（元素与键类型不一致）。用自引用对象使两者兼容以覆盖该路径。"""

    def __init__(self):
        self.req_id = self


class TestSchedulerSide:
    def _build_setup(self, g, monkeypatch, seg=32768, prompt=0):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", seg)
        g._store_injected = True  # 跳过 store 注入
        md = gsa_mod.GSAMetaData(g._vllm_config, 2)
        stat = gsa_mod.GSAReqStat("r1", g._vllm_config, 2)
        stat.num_prompt_tokens = prompt
        md.gsa_stats["r1"] = stat
        g.gsa_stats = md.gsa_stats
        # 注：build_sparse_meta 会把 scheduled_new_reqs 的元素（NewRequestData
        # 对象，非 req_id）直接传给 _load_kv_dram —— 疑似源码问题，此处用
        # cached 路径规避（new-req 路径已在 TestBuildMetaAndExecute 覆盖）
        so = fakes.FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 1},
            cached_req_ids=["r1"],
        )
        add = fakes.FakeAddReqState(
            block_ids=[[5, 6]],
            prompt_token_ids=list(range(max(prompt, 1))),
            output_token_ids=[1],
            num_computed_tokens=200,
        )
        requests = {"r1": add}
        input_batch = fakes.FakeInputBatch({"r1": 0})
        return so, requests, input_batch

    def test_build_sparse_meta_early_return(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "PTOPK_PREFETCH_ENABLE", False)
        gsa_worker.build_sparse_meta(fakes.FakeSchedulerOutput(), {}, None, {})

    def test_build_sparse_meta_full(self, gsa_worker, monkeypatch):
        g = gsa_worker
        so, requests, input_batch = self._build_setup(g, monkeypatch)
        g.build_sparse_meta(so, requests, input_batch, {})
        stats = g.gsa_metadata.gsa_stats
        assert "r1" in stats
        # prompt 0 <= SEG(32768)：非 GSA、无预取触发
        assert g.need_prefetch_index == []
        assert g.gsa_index == []
        assert g.decode_index == [0]  # stage == DECODE
        assert g.has_prefill is False
        assert g.is_all_gsa is False

    def test_build_sparse_meta_gsa_request(self, gsa_worker, monkeypatch):
        g = gsa_worker
        so, requests, input_batch = self._build_setup(
            g, monkeypatch, seg=64, prompt=200
        )
        # PD 分离的预取触发条件是 output == 0（请求刚到 decode 节点）
        requests["r1"].output_token_ids = []
        g.build_sparse_meta(so, requests, input_batch, {})
        assert g.need_prefetch_index == [0]
        assert g.gsa_index == [0]
        assert g.is_all_gsa is True

    def test_build_sparse_meta_new_reqs_and_has_prefill(self, gsa_worker, monkeypatch):
        g = gsa_worker
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 32768)
        g._store_injected = True
        nr = _SelfRefNewReq()
        md = gsa_mod.GSAMetaData(g._vllm_config, 2)
        md.gsa_stats[nr] = gsa_mod.GSAReqStat(nr, g._vllm_config, 2)
        g.gsa_stats = md.gsa_stats
        so = fakes.FakeSchedulerOutput(num_scheduled_tokens={nr: 1}, new_reqs=[nr])
        add = fakes.FakeAddReqState(
            block_ids=[[5]],
            prompt_token_ids=list(range(200)),
            output_token_ids=[],
            num_computed_tokens=0,
        )
        g.build_sparse_meta(so, {nr: add}, fakes.FakeInputBatch({nr: 0}), {})
        # prefill 请求（prompt 200 > computed 0）→ has_prefill True
        assert g.has_prefill is True
        assert g.decode_index == []
        assert g.is_all_gsa is False

    def test_build_sparse_meta_loads_kv_dram(self, gsa_worker, monkeypatch):
        g = gsa_worker
        # resumed / new reqs → _load_kv_dram（短 prompt 早退）
        so, requests, input_batch = self._build_setup(g, monkeypatch)
        so.scheduled_new_reqs = []
        so.scheduled_cached_reqs.resumed_req_ids = ["r1"]
        so.scheduled_cached_reqs.req_ids = ["r1"]
        stats = gsa_mod.GSAMetaData(g._vllm_config, 2)
        stat = gsa_mod.GSAReqStat("r1", g._vllm_config, 2)
        stat.num_prompt_tokens = 200
        stats.gsa_stats["r1"] = stat
        g.gsa_stats = stats.gsa_stats
        g.gsa_metadata = stats
        g.build_sparse_meta(so, requests, input_batch, {})

    def test_request_begin_and_update_state(self, gsa_worker):
        gsa_worker.request_begin("r1", [1, 2, 3])
        gsa_worker.update_state_after_alloc(SimpleNamespace(), 4)

    def test_request_finished_in_scheduler(self, gsa_worker, monkeypatch):
        g = gsa_worker
        g.preempt_req_output_tokens["r1"] = 5
        g.request_finished_in_scheduler("r1")
        assert "r1" not in g.preempt_req_output_tokens
        g.request_finished_in_scheduler("missing")  # 不存在 → pass
        monkeypatch.setattr(gsa_mod, "PTOPK_PREFETCH_ENABLE", False)
        g.request_finished_in_scheduler("r1")  # 开关关闭 → 早退

    def test_request_finished_in_worker_early_return(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "PTOPK_PREFETCH_ENABLE", False)
        gsa_worker = make_gsa()
        gsa_worker.request_finished_in_worker("r1")
        make_gsa(consumer=False).request_finished_in_worker("r1")

    def test_request_finished_in_worker_rank0(self, gsa_worker):
        g = gsa_worker
        md, stat = _prime_metadata(g)
        stat.slab_host_slot = [1]
        stat.indexer_slab_host_slot = [2]
        stat.req_dump_blocks = [3]
        g.gsa_stats = md.gsa_stats
        g.request_finished_in_worker("r1")
        assert "r1" not in g.gsa_stats
        assert 3 in g.gsa_preempt_shared_pool.free_slots  # dump 块已释放
        assert g.prefetch_engine.prefetch_engine_c.called("del_blocks_map")
        assert "r1" not in g._block_hash_injected_reqs

    def test_request_finished_in_worker_offload_empty_slots(
        self, make_gsa, monkeypatch
    ):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        md, stat = _prime_metadata(g)
        stat.slab_host_slot = []
        stat.indexer_slab_host_slot = []
        stat.req_dump_blocks = []
        g.gsa_stats = md.gsa_stats
        g.request_finished_in_worker("r1")  # 空 slot 组合 → 跳过 free

    def test_request_finished_in_worker_rank_nonzero(self, make_gsa):
        g = make_gsa(rank=1)
        md, stat = _prime_metadata(g)
        stat.req_dump_blocks = []
        g.gsa_stats = md.gsa_stats
        g.request_finished_in_worker("r1")
        assert "r1" not in g.gsa_stats

    def test_request_finished_in_worker_busy_wait(self, gsa_worker):
        g = gsa_worker
        md, stat = _prime_metadata(g)
        stat.req_dump_blocks = []
        g.gsa_stats = md.gsa_stats
        ec = g.prefetch_engine.prefetch_engine_c
        ec.prefetch_status = [True, False]  # 忙等一次
        g.request_finished_in_worker("r1")
        assert len(ec.called("get_prefetch_status")) == 2

    def test_request_finished_in_worker_offload_slots(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        md, stat = _prime_metadata(g)
        stat.slab_host_slot = [1]
        stat.indexer_slab_host_slot = [2]
        stat.req_dump_blocks = [3]
        g.gsa_stats = md.gsa_stats
        g.request_finished_in_worker("r1")
        assert 1 in g.gsa_cpu_shared_pool.free_slots
        assert 2 in g.indexer_cpu_shared_pool.free_slots
        assert 3 in g.gsa_preempt_shared_pool.free_slots

    def test_request_finished_in_worker_indexer_busy(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        md, stat = _prime_metadata(g)
        stat.req_dump_blocks = []
        g.gsa_stats = md.gsa_stats
        ec = g.prefetch_engine.prefetch_engine_c
        ec.prefetch_status = [False]
        ec.indexer_status = [True, False]  # indexer 忙等一次
        g.request_finished_in_worker("r1")
        assert len(ec.called("get_indexer_status")) == 2

    def test_begin_sch_flag_and_update(self, gsa_worker):
        g = gsa_worker
        g.update_scheduler_free_slot(7)
        assert g.alloc_block_sch == 7
        g.update_scheduler_free_slot(3)
        assert g.alloc_block_sch == 10
        g.begin_sch_flag()
        assert g.alloc_block_sch == 0
        g.update_scheduler_free_slot(0)  # 归零分支
        assert g.alloc_block_sch == 0

    def test_estimate_free_slots_not_decode(self, make_gsa):
        g = make_gsa(consumer=False)
        req = fakes.FakeRequest(prompt_token_ids=list(range(100)))
        assert g.estimate_free_slots_sparse(req, 10) is False

    def test_estimate_free_slots_branches(self, gsa_worker, monkeypatch):
        g = gsa_worker
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        req = fakes.FakeRequest(prompt_token_ids=list(range(100)))
        g.begin_sch_flag()
        # free_i 100-0=100：100-50=50 ≥ 20 → False，累计 alloc
        assert g.estimate_free_slots_sparse(req, 50, 0) is False
        assert g.alloc_block_sch == 50
        # free_i 50：50-30=20 不 < 20 → False，继续累计
        assert g.estimate_free_slots_sparse(req, 30, 0) is False
        assert g.alloc_block_sch == 80
        # free_i 20：20-10=10 < 20 → True，不累计
        assert g.estimate_free_slots_sparse(req, 10, 0) is True
        assert g.alloc_block_sch == 80

    def test_estimate_free_slots_offload_k_low(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = make_gsa()
        req = fakes.FakeRequest(prompt_token_ids=list(range(100)))
        # free_k(100) - total(90) < 20 → preempted
        assert g.estimate_free_slots_sparse(req, 90, 0) is True

    def test_estimate_num_slots_invalid_paths(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "PTOPK_PREFETCH_ENABLE", False)
        req = fakes.FakeRequest(num_output_tokens=5, prompt_token_ids=list(range(100)))
        assert gsa_worker.estimate_num_slots_sparsed(req) == gsa_mod.INVALID_SLOT

    def test_estimate_num_slots_preempted(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        req = fakes.FakeRequest(
            num_output_tokens=5,
            prompt_token_ids=list(range(100)),
            status=gsa_mod.RequestStatus.PREEMPTED,
        )
        # preempt 记录 5 → 修正后 output 5-5=0 → INVALID_SLOT
        assert g.estimate_num_slots_sparsed(req) == gsa_mod.INVALID_SLOT
        assert g.preempt_req_output_tokens[1] == 5

    def test_estimate_num_slots_normal(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        req = fakes.FakeRequest(
            num_output_tokens=5,
            num_prompt_tokens=200,
            prompt_token_ids=list(range(200)),
        )
        assert g.estimate_num_slots_sparsed(req) == 5 + gsa_mod.GSA_SPARSE_LEN + 3

    def test_estimate_num_slots_preempt_adjusted(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        g.preempt_req_output_tokens[1] = 3
        req = fakes.FakeRequest(
            num_output_tokens=5,
            num_prompt_tokens=200,
            prompt_token_ids=list(range(200)),
        )
        # 修正后 output = 5-3 = 2
        assert g.estimate_num_slots_sparsed(req) == 2 + gsa_mod.GSA_SPARSE_LEN + 3

    def test_estimate_num_slots_zero_output_or_short_prompt(self, gsa_worker):
        g = gsa_worker
        req = fakes.FakeRequest(num_output_tokens=0, prompt_token_ids=list(range(100)))
        assert g.estimate_num_slots_sparsed(req) == gsa_mod.INVALID_SLOT
        req = fakes.FakeRequest(num_output_tokens=2, prompt_token_ids=[1, 2])
        assert g.estimate_num_slots_sparsed(req) == gsa_mod.INVALID_SLOT

    def test_allocate_slots_early_return(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "PTOPK_PREFETCH_ENABLE", False)
        assert (
            gsa_worker.allocate_slots(_fake_kv_cache_manager(), fakes.FakeRequest(), 10)
            is None
        )

    def test_allocate_slots_success_output_one(self, gsa_worker):
        g = gsa_worker
        mgr = _fake_kv_cache_manager()
        req = fakes.FakeRequest(num_output_tokens=1, prompt_token_ids=list(range(100)))
        result = g.allocate_slots(mgr, req, 10)
        assert req.cache_salt is not None and req.cache_salt.startswith("gsa")
        assert result.blocks == ([10, 11],)
        kwargs = mgr.coordinator.calls[0]
        assert kwargs["num_tokens"] == 10
        assert "num_tokens_main_model" in kwargs  # 0.18 分支

    def test_allocate_slots_version_013(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "VLLM_ASCEND_VERSION", "0.13")
        g = gsa_worker
        mgr = _fake_kv_cache_manager()
        req = fakes.FakeRequest(num_output_tokens=1, prompt_token_ids=list(range(100)))
        result = g.allocate_slots(mgr, req, 10)
        assert result is not None
        assert "num_tokens_main_model" not in mgr.coordinator.calls[0]

    def test_allocate_slots_unsupported_version(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "VLLM_ASCEND_VERSION", "0.9")
        g = gsa_worker
        req = fakes.FakeRequest(num_output_tokens=1, prompt_token_ids=list(range(100)))
        assert g.allocate_slots(_fake_kv_cache_manager(), req, 10) is None

    def test_allocate_slots_preempt_adjusted(self, gsa_worker):
        g = gsa_worker
        g.preempt_req_output_tokens[1] = 3
        mgr = _fake_kv_cache_manager()
        req = fakes.FakeRequest(
            num_output_tokens=5, prompt_token_ids=list(range(100))  # 修正为 2
        )
        result = g.allocate_slots(mgr, req, 10)
        assert result is not None
        assert req.cache_salt is None  # 修正后 ≠1 → 不重置 cache_salt

    def test_allocate_slots_insufficient_free_blocks(self, gsa_worker):
        g = gsa_worker
        mgr = _fake_kv_cache_manager(
            coordinator=_FakeCoordinator(num_blocks=200),
            block_pool=_FakeBlockPool(free_blocks=10),
        )
        req = fakes.FakeRequest(num_output_tokens=1, prompt_token_ids=list(range(100)))
        assert g.allocate_slots(mgr, req, 10) is None

    def test_allocate_slots_sparse_preempted(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        g = gsa_worker
        g.begin_sch_flag()
        mgr = _fake_kv_cache_manager(coordinator=_FakeCoordinator(num_blocks=90))
        req = fakes.FakeRequest(num_output_tokens=5, prompt_token_ids=list(range(100)))
        # num_output > 1 → estimate：free_i(100)-90 < 20 → 判定抢占 → None
        assert g.allocate_slots(mgr, req, 10) is None

    def test_get_rank_load_hash(self, gsa_worker):
        g = gsa_worker  # tp_size = 1
        md, stat = _prime_metadata(g)
        stat.block_hashes = [b"a" * 16, b"b" * 16, b"c" * 16]
        assert g._get_rank_load_hash("r1") == stat.block_hashes

    def test_get_rank_load_hash_tp2(self, make_gsa):
        g = make_gsa(tp_size=2)
        md, stat = _prime_metadata(g)
        stat.block_hashes = [b"a" * 16, b"b" * 16, b"c" * 16]
        assert g._get_rank_load_hash("r1") == [b"a" * 16, b"b" * 16]

    def test_load_kv_dram_pd_mix_early_return(self, make_gsa):
        make_gsa(producer=True, consumer=True)._load_kv_dram("r1")

    def test_load_kv_dram_not_decode_early_return(self, make_gsa):
        make_gsa(consumer=False)._load_kv_dram("r1")

    def test_load_kv_dram_short_prompt_early_return(self, gsa_worker):
        _prime_metadata(gsa_worker)  # prompt 200 <= 32768
        gsa_worker._load_kv_dram("r1")

    def test_load_kv_dram_full_hit(self, gsa_worker, monkeypatch):
        g = gsa_worker
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        md, stat = _prime_metadata(g)
        stat.block_hashes = [b"a" * 16, b"b" * 16, b"c" * 16]
        lookups = []

        def _lookup(ids):
            lookups.append(len(ids))
            return [True] * (len(ids) // 16)

        g.ucm_prefetch_engine.ucm_lookup = _lookup
        g._load_kv_dram("r1")
        assert lookups == [48]  # 一次命中
        assert g.ucm_prefetch_engine.calls == [("ucm_load_ssd", 48)]

    def test_load_kv_dram_partial_retry(self, gsa_worker, monkeypatch):
        g = gsa_worker
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        md, stat = _prime_metadata(g)
        stat.block_hashes = [b"a" * 16, b"b" * 16, b"c" * 16]
        script = iter([[True, False, True], [True]])
        g.ucm_prefetch_engine.ucm_lookup = lambda ids: next(script)
        g._load_kv_dram("r1")
        # 第二次 lookup 只带 miss 的 1 个 hash（16 字节）
        assert g.ucm_prefetch_engine.calls == [("ucm_load_ssd", 48)]

    def test_load_kv_dram_timeout(self, gsa_worker, monkeypatch):
        g = gsa_worker
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        monkeypatch.setattr(gsa_mod, "UCM_LOOKUP_MAX_RETRY", 1)
        monkeypatch.setattr(gsa_mod, "UCM_LOOKUP_TIME_SPACE", 0)
        md, stat = _prime_metadata(g)
        stat.block_hashes = [b"a" * 16, b"b" * 16, b"c" * 16]
        g.ucm_prefetch_engine.ucm_lookup = lambda ids: [False] * (len(ids) // 16)
        g._load_kv_dram("r1")  # 重试超限 → 记 error 并返回
        assert g.ucm_prefetch_engine.calls == []  # 未提交 ssd 加载


# ===========================================================================
# host slab / 生命周期
# ===========================================================================
class TestBuffers:
    def test_init_host_slabs_skip_when_ready(self, gsa_worker, capsys):
        gsa_worker.init_host_slabs_from_kv(16)  # _buff_host_k 已存在 → 跳过
        assert "initialize slab host" not in capsys.readouterr().out

    def test_init_host_slabs_from_scratch(self, make_gsa, monkeypatch, capsys):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", False)
        g = make_gsa(consumer=False)  # 非 decode worker：buffer 未初始化
        assert g._buff_host_k is None
        g.k_dim = 16  # 正常由 _init_prefetch_buffers 设置（非 decode 未设）
        g.init_host_slabs_from_kv(16)
        out = capsys.readouterr().out
        assert "initialize slab host success" in out
        assert g._buff_host_k is not None
        assert g._slab_preempt_k_ptr > 0
        assert g._slab_preempt_v_ptr > 0
        assert not hasattr(g, "indexer_cache_preempt")

    def test_init_host_slabs_indexer_offload(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa(consumer=False)
        g.k_dim = 16
        g.init_host_slabs_from_kv(16)
        assert g.indexer_cache_preempt is not None
        assert g.indexer_cache_preempt_ptr > 0
        assert g.indexer_k_repr.shape == (g.layer_num, 16, 1, 1, 128)

    def test_init_kvcache_ptrs_first_and_second(self, gsa_worker):
        g = gsa_worker
        kv_cache = [
            torch.zeros((8, 128, 1, 16), dtype=torch.float16),
            torch.zeros((8, 128, 1, 16), dtype=torch.float16),
            torch.zeros((8, 128, 1, 128), dtype=torch.float16),
            torch.zeros((8, 128, 1, 1), dtype=torch.float16),
        ]
        g.init_kvcache_ptrs(kv_cache, kv_cache[2], kv_cache[3], 0)
        assert g.kv_caches[0] is kv_cache
        assert g.indexer_k_cache_sparse[0] is kv_cache[2]
        assert g.prefetch_engine.prefetch_engine_c.called("set_kvTensor_Layer")
        # 二次注册同层 → 早退（不再新增调用）
        n = len(g.prefetch_engine.prefetch_engine_c.called("set_kvTensor_Layer"))
        g.init_kvcache_ptrs(kv_cache, kv_cache[2], kv_cache[3], 0)
        assert (
            len(g.prefetch_engine.prefetch_engine_c.called("set_kvTensor_Layer")) == n
        )

    def test_init_kvcache_ptrs_offload_index(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        kv_cache = [
            torch.zeros((8, 128, 1, 16), dtype=torch.float16),
            torch.zeros((8, 128, 1, 16), dtype=torch.float16),
            torch.zeros((8, 128, 1, 128), dtype=torch.float16),
            torch.zeros((8, 128, 1, 1), dtype=torch.float16),
        ]
        g.init_kvcache_ptrs(kv_cache, kv_cache[2], kv_cache[3], 0)
        assert g.prefetch_engine.prefetch_engine_c.called("set_kvTensor_Index")

    def test_init_slab_preempt_kv_sfa_shares_v(self, make_gsa):
        g = make_gsa(additional_config={"enable_sparse_sfa_c8": True})
        assert g._slab_preempt_v is g._slab_preempt_k
        assert g._slab_preempt_v_ptr == g._slab_preempt_k_ptr
        assert g._slab_preempt_v_offset == g._slab_preempt_k_offset

    def test_shutdown_non_offload(self, gsa_worker, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", False)
        gsa_worker.shutdown()  # short-circuit 返回

    def test_shutdown_block_table_none(self, make_gsa, monkeypatch):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        g.block_table_sm = None
        g.shutdown()  # block_table_sm None → 早退

    def test_shutdown_success(self, make_gsa, monkeypatch, capsys):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()
        g.shutdown()
        out = capsys.readouterr().out
        assert "block_table_sm memory" in out
        assert "indexer_block_table_sm" in out

    def test_shutdown_unlink_error(self, make_gsa, monkeypatch, capsys):
        monkeypatch.setattr(gsa_mod, "IS_INDEXER_OFFLOAD", True)
        g = make_gsa()

        def _boom():
            raise OSError("segment busy")

        g.block_table_sm.unlink = _boom
        g.indexer_block_table_sm.unlink = _boom
        g.shutdown()  # 异常被捕获并打印
        out = capsys.readouterr().out
        assert "清理失败" in out
