# -*- coding: utf-8 -*-
"""针对 GSA_ANALYSIS_REPORT.md（仓库根）中确认缺陷的回归测试。

覆盖的缺陷编号（详见报告）：
- S1-1 / A2：build_sparse_meta 把 NewRequestData 对象当作 req_id 传给
  _load_kv_dram（gsa.py:2383-2384）→ PD 分离 + 新请求必现 KeyError。
  修复计划 P0-F2：改为 self._load_kv_dram(new_req.req_id)。
- S1-2：_init_slots_num_manager 的 SCHEDULER 分支缺少 is_decode_node 守卫
  （gsa.py:586）→ 跨机 PD 分离时
    P 节点 attach D 节点 shm，120s 超时启动崩溃。修复计划 P0-F4。
- S2-2：estimate_num_slots_sparsed 默认 num_spec_slots=3（取混合基线），
  PD 分离基线为 0 —— 行为漂移文档化。

约定：
- "当前行为"用例钉住现状（BugDocumented 风格，防止无意识变更）；
- "期望修复"用例标记 @pytest.mark.xfail(strict=True)——对应修复落地后
  XPASS(strict) 报错，提示移除标记并保留该用例作为正向回归。
"""
import fakes
import pytest

import ucm.sandbox.sparse.gsav2.gsa as gsa_mod
from ucm.sparse.base import UcmSparseRole


# ===========================================================================
# S1-1 / A2（已修复 P0-F2）：build_sparse_meta 传对象而非 req_id
# ===========================================================================
class _PlainNewReq:
    """模拟 vLLM NewRequestData：req_id 是字符串，对象本身不是键。

    与 test_gsa.py 历史上的 _SelfRefNewReq（自引用 hack）不同，本类还原真实
    键控方式（GSAMetaData.get_model_input 以 new_req.req_id 查 requests /
    input_batch / num_scheduled_tokens）。
    """

    def __init__(self, req_id):
        self.req_id = req_id


class TestBuildSparseMetaNewReqArg:
    def _run(self, gsa_worker, monkeypatch):
        g = gsa_worker
        monkeypatch.setattr(gsa_mod, "IS_OPEN_GSA", True)
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 32768)
        g._store_injected = True  # 跳过 store 注入重试
        nr = _PlainNewReq("r_new")
        recorded = []
        # 实例级打桩：只记录收到的参数，绕过内部 gsa_stats 查找
        g._load_kv_dram = lambda req_id: recorded.append(req_id)
        so = fakes.FakeSchedulerOutput(
            num_scheduled_tokens={"r_new": 1}, new_reqs=[nr]
        )
        add = fakes.FakeAddReqState(
            block_ids=[[5]],
            prompt_token_ids=list(range(200)),
            output_token_ids=[],
            num_computed_tokens=0,
        )
        g.build_sparse_meta(so, {"r_new": add}, fakes.FakeInputBatch({"r_new": 0}), {})
        return recorded, nr

    def test_fixed_req_id_string_passed(self, gsa_worker, monkeypatch):
        """回归守护（P0-F2 已修复）：_load_kv_dram 必须收到 req_id 字符串。

        修复前 gsa.py:2383-2384 原样把 NewRequestData 对象传给 _load_kv_dram，
        真实运行时 gsa_stats[对象] 必然 KeyError（PD 分离 + decode 节点 +
        任意新请求即触发）。
        """
        recorded, _ = self._run(gsa_worker, monkeypatch)
        assert recorded == ["r_new"]
        assert all(isinstance(r, str) for r in recorded)


# ===========================================================================
# S1-2 / P0-F4：SCHEDULER 分支缺少 is_decode_node 守卫（2026-11-09 已修复，
# 原钉缺陷行为的用例已删，翻转原 xfail 用例为正向回归）
# ===========================================================================
class TestSchedulerSlotsNumGuard:
    def _make_p_node_scheduler(self, make_gsa, monkeypatch):
        # PD 分离 P 节点：producer-only（is_kv_consumer=False → is_decode_node=False）
        monkeypatch.setattr(gsa_mod, "IS_OPEN_GSA", True)
        return make_gsa(
            role=UcmSparseRole.SCHEDULER, consumer=False, producer=True
        )

    def test_fixed_scheduler_skips_attach(self, make_gsa, monkeypatch):
        """修复后行为：非 decode 的 scheduler 不创建/attach slots_num shm。"""
        g = self._make_p_node_scheduler(make_gsa, monkeypatch)
        assert not hasattr(g, "slots_num_manager")


# ===========================================================================
# S2-2：estimate_num_slots_sparsed 的 num_spec_slots 默认值漂移
# ===========================================================================
class TestEstimateNumSlotsSparsedDefault:
    def test_pd_separated_default_is_three(self, gsa_worker, monkeypatch):
        """钉住行为漂移：PD 分离（gsa_worker 默认 consumer-only）下默认
        num_spec_slots=3（gsa.py:2472，合并时取混合基线值，分离基线为 0）。
        v0110 补丁单参调用时每请求多估 3 token。
        若裁决改回 0，请同步更新本断言。"""
        monkeypatch.setattr(gsa_mod, "IS_OPEN_GSA", True)
        monkeypatch.setattr(gsa_mod, "SEG_PREFILL_THRESHOLD", 64)
        req = fakes.FakeRequest(num_prompt_tokens=200, num_output_tokens=1)
        assert gsa_worker.estimate_num_slots_sparsed(req) == (
            1 + gsa_mod.GSA_SPARSE_LEN + 3
        )
