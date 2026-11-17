# -*- coding: utf-8 -*-
"""ucm/sandbox/sparse/gsav2/prefetch/prefetch_engine.py 单元测试。

GSAPrefetchEngineC（C++ 扩展）在 UT 中是 conftest 注入的 FakeEngineC，
本文件验证 Python 侧的配置解析、dtype/k_dim 分支与接口转发。
"""
import fakes
import pytest
import torch

import ucm.sandbox.sparse.gsav2.prefetch.prefetch_engine as pe_mod


class TestSequenceStage:
    def test_enum_members(self):
        assert pe_mod.SequenceStage.PREFILL != pe_mod.SequenceStage.DECODE


class TestGSAPrefetchBaseInit:
    def test_init_basic_config(self):
        pe = pe_mod.GSAPrefetchBase(fakes.FakeVllmConfig(), 2, True, [0, 1], [0, 1])
        assert pe.rank == 0
        assert pe.dp_rank == 0
        assert pe.tp_size == 1
        assert pe.async_thread == 2
        assert pe.layer_num == 2
        assert pe.enable_sparse_sfa_c8 is False
        assert pe.is_log is True
        assert pe.max_block_len == 2048
        assert pe.block_size == 128
        assert pe.head_size == 16
        assert pe.num_head == 1
        assert pe.step_time == 0
        assert pe.select_bs_index is None

    def test_init_without_sfa_c8_dims(self):
        pe = pe_mod.GSAPrefetchBase(fakes.FakeVllmConfig(), 2, False, None, [0, 1])
        # 非 c8：k/v 维度直接取 MLA 配置，dtype 沿用模型 dtype
        assert pe.k_dim == 16
        assert pe.v_dim == 16
        assert pe.dtype == torch.float16
        assert pe.element_size == 2

    def test_init_with_sfa_c8_non_a5(self):
        cfg = fakes.FakeVllmConfig(additional_config={"enable_sparse_sfa_c8": True})
        pe = pe_mod.GSAPrefetchBase(cfg, 2, False, None, [0, 1])
        # 非 A5 平台：int8 + fp16 scale，k_dim 走 packed head dim
        assert pe.dtype == torch.int8
        assert pe.c8_k_scale_cache_dtype == torch.float16
        assert pe.k_dim == 32  # stub: kv_lora_rank(16) + qk_rope_head_dim(16)
        assert pe.v_dim == 0
        assert pe.element_size == 1

    def test_init_with_sfa_c8_a5(self, monkeypatch):
        monkeypatch.setattr(pe_mod, "get_ascend_device_type", lambda: "A5")
        cfg = fakes.FakeVllmConfig(additional_config={"enable_sparse_sfa_c8": True})
        pe = pe_mod.GSAPrefetchBase(cfg, 2, False, None, [0, 1])
        assert pe.dtype == torch.float8_e4m3fn
        assert pe.c8_k_scale_cache_dtype == torch.float32
        assert pe.element_size == 1

    def test_engine_c_constructed_with_expected_args(self):
        cfg = fakes.FakeVllmConfig()
        pe = pe_mod.GSAPrefetchBase(cfg, 3, False, [4, 5], [0, 1])
        engine = pe.prefetch_engine_c
        assert isinstance(engine, fakes.FakeEngineC)
        cfg_args = engine.init_args[0]
        dims_args = engine.init_args[1]
        assert cfg_args[:6] == [pe.layer_num, pe.max_block_len, 5120, 1, 8192, 0]
        assert cfg_args[6] == 0  # dp_rank
        assert cfg_args[7] == 1  # tp_size
        assert cfg_args[8] is False  # IS_INDEXER_OFFLOAD
        assert cfg_args[9] is False  # enable_sparse_sfa_c8
        assert cfg_args[10] == 4  # max_bs
        assert dims_args == [128, 1, 16, 16, 2]
        assert engine.init_args[7].endswith("kvcache_pre_log_dp0_tp0.log")

    def test_ptr_lists_initialized(self):
        pe = pe_mod.GSAPrefetchBase(fakes.FakeVllmConfig(), 2, False, None, [0, 1])
        assert pe._slab_host_k_ptr == []
        assert pe._slab_host_v_ptr == []
        assert pe.k_cache_ptr == [None, None]
        assert pe.v_cache_ptr == [None, None]


class TestGSAPrefetchBaseAPI:
    def _pe(self):
        return pe_mod.GSAPrefetchBase(fakes.FakeVllmConfig(), 2, False, None, [0, 1])

    def test_update_topk_index(self):
        pe = self._pe()
        pe.prefetch_engine_c.topk_indices_result = [1, 2, 3]
        assert pe.update_topk_index(1, "req-1") == [1, 2, 3]
        assert pe.prefetch_engine_c.called("get_topk_indices")

    def test_del_finish_meta_calls_del_blocks_map(self):
        pe = self._pe()
        pe.del_finish_meta("req-1")
        assert pe.prefetch_engine_c.called("del_blocks_map")

    def test_del_finish_meta_flag_false_skips(self):
        pe = self._pe()
        pe.del_finish_meta("req-1", flag=False)
        assert not pe.prefetch_engine_c.called("del_blocks_map")

    def test_del_finish_meta_ptopk_disabled_skips(self, monkeypatch):
        monkeypatch.setattr(pe_mod, "PTOPK_PREFETCH_ENABLE", False)
        pe = self._pe()
        pe.del_finish_meta("req-1")  # flag 默认 True，但开关关闭
        assert not pe.prefetch_engine_c.called("del_blocks_map")
