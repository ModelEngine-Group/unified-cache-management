# -*- coding: utf-8 -*-
"""gsav2 跨语言绑定契约测试（静态，无 NPU / C++ 环境亦可运行）。

背景（见仓库根 GSA_ANALYSIS_REPORT.md 缺陷 A1 / A3，均已修复）：
1. [已修复 P0-F1] gsa.py:1024 原调用 set_last_indexer_blocks（复数），而
   pybinds.cpp:152 只导出 set_last_indexer_block（单数）——调用点与 UT 断言
   已全部改为单数，账本条目已移除；
2. [已修复 P0-F3] prefetch_engine.py 原 update_topk_index 调用 get_topk_indices，
   pybinds.cpp 绑定表中不存在——经全仓核查无任何生产调用方，方法已整体删除；
3. （背景同 2）ut/fakes.py 曾显式实现 get_topk_indices（非 __getattr__ 兜底），
   在单测中掩盖真实绑定的缺失——已随 A3 一并删除，本契约测试防止同类问题再现。

本文件做的事：
- 解析 prefetch/src/pybinds.cpp 的 .def("...") 得到两个类的真实绑定集合；
- 扫描 gsa.py / prefetch_engine.py 中
  prefetch_engine_c.<name>( 与 ucm_prefetch_engine.<name>( 的调用点；
  （示例代码文件不纳入扫描。）
- 维护一份"调用了但未导出"的缺陷账本 KNOWN_MISMATCHES：
  * 账本之外的新缺失 → test_no_new_unbound_calls 立即失败；
  * 账本内缺陷被修复后 → test_ledger_entries_still_valid 失败，提示清理账本；
- 校验 FakeEngineC / FakeUCMPrefetch 的显式绑定风格方法名 ⊆ 真实绑定 ∪ 账本，
  防止单测 fake 再次掩盖缺失绑定。
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
GSAV2 = HERE.parent
PYBINDS_CPP = GSAV2 / "prefetch" / "src" / "pybinds.cpp"

# 生产代码文件（避免在 UT 进程 import 生产模块，用源码文本扫描）。
PROD_FILES = {
    "gsa.py": GSAV2 / "gsa.py",
    "prefetch_engine.py": GSAV2 / "prefetch" / "prefetch_engine.py",
}

# ---------------------------------------------------------------------------
# 已知缺陷账本：(文件名, 方法名) —— 调用了但 pybinds.cpp 未导出。
# 修复对应缺陷后 test_ledger_entries_still_valid 会失败并提示从这里移除条目。
# ---------------------------------------------------------------------------
KNOWN_MISMATCHES = {
    # A1（set_last_indexer_blocks 复数误调，P0-F1）已修复：gsa.py:1024
    # 调用点已改为单数（示例文件中的同类调用点亦已同步修正）。
    # A3（get_topk_indices 绑定缺失，P0-F3）已修复：update_topk_index 无任何
    # 生产调用方，方法与 FakeEngineC.get_topk_indices 已整体删除。
    # 账本当前为空——所有 prefetch_engine_c 调用均有对应 .def 绑定，
    # 由 test_no_new_unbound_calls 持续看护；新缺失会立即使其失败。
}

_DEF_RE = re.compile(r'\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)"')
_ENGINE_CALL_RE = re.compile(r"prefetch_engine_c\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_UCM_CALL_RE = re.compile(r"ucm_prefetch_engine\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# FakeEngineC / FakeUCMPrefetch 中认定为"模拟绑定"的方法名前缀
# （called / init_args 等纯测试辅助设施不属于绑定风格，自动排除）。
_BINDING_VERB_PREFIXES = (
    "get_",
    "set_",
    "submit_",
    "indexer_",
    "pre_swap_",
    "update_",
    "del_",
    "store_init",
    "ucm_",
)


def _class_segment(text: str, class_name: str) -> str:
    marker = f"pybind11::class_<ucmprefetch::{class_name}>"
    start = text.index(marker)
    nxt = text.find("pybind11::class_", start + len(marker))
    return text[start : nxt if nxt != -1 else len(text)]


def parse_bindings():
    """返回 (engine_bindings, ucm_bindings) 两个集合。"""
    text = PYBINDS_CPP.read_text(encoding="utf-8")
    engine_block = _class_segment(text, "GSAPrefetchEngineC")
    ucm_block = _class_segment(text, "UCMKVPrefetch")
    return set(_DEF_RE.findall(engine_block)), set(_DEF_RE.findall(ucm_block))


def scan_engine_calls():
    """返回 {(文件名, 方法名)}：生产代码中对 prefetch_engine_c.<name>( 的调用。"""
    calls = set()
    for label, path in PROD_FILES.items():
        src = path.read_text(encoding="utf-8")
        for name in _ENGINE_CALL_RE.findall(src):
            calls.add((label, name))
    return calls


def scan_ucm_calls():
    """返回 {(文件名, 方法名)}：生产代码中对 ucm_prefetch_engine.<name>( 的调用。"""
    calls = set()
    for label, path in PROD_FILES.items():
        src = path.read_text(encoding="utf-8")
        for name in _UCM_CALL_RE.findall(src):
            calls.add((label, name))
    return calls


def _fake_binding_style_methods(fake_cls):
    """fake 类上显式定义、且命名风格像 pybind 绑定的方法名集合。"""
    import inspect

    names = set()
    for name, member in vars(fake_cls).items():
        if name.startswith("_") or not inspect.isfunction(member):
            continue
        if name.startswith(_BINDING_VERB_PREFIXES):
            names.add(name)
    return names


class TestEngineBindingContract:
    def test_no_new_unbound_calls(self):
        """账本之外不允许出现任何"调用了但未导出"的引擎方法。"""
        engine_bindings, _ = parse_bindings()
        missing = (
            scan_engine_calls()
            - {(f, n) for f, n in KNOWN_MISMATCHES}
            - {(f, n) for f, n in scan_engine_calls() if n in engine_bindings}
        )
        assert not missing, (
            f"以下 prefetch_engine_c 调用在 pybinds.cpp 中没有对应 .def 绑定，"
            f"运行时会 AttributeError：{sorted(missing)}\n"
            f"（若为已知缺陷请登记到 KNOWN_MISMATCHES；若是新缺陷请修复调用名或补绑定。）"
        )

    def test_ledger_entries_still_valid(self):
        """账本内条目必须仍处于"调用了但未导出"状态。

        修复对应缺陷（如 gsa.py:1024 改为 set_last_indexer_block）后本测试失败，
        提示从 KNOWN_MISMATCHES 移除已修复条目，防止账本腐烂。
        """
        engine_bindings, _ = parse_bindings()
        discovered = scan_engine_calls()
        stale = [
            entry
            for entry in KNOWN_MISMATCHES
            if entry not in discovered or entry[1] in engine_bindings
        ]
        assert not stale, (
            f"账本条目已失效（对应缺陷已修复或调用点已删除）：{sorted(stale)}\n"
            f"请从 KNOWN_MISMATCHES 中移除，使 test_no_new_unbound_calls 接管看护。"
        )

    def test_ucm_prefetch_calls_have_bindings(self):
        """ucm_prefetch_engine（UCMKVPrefetch）调用点必须有对应绑定。"""
        _, ucm_bindings = parse_bindings()
        missing = {(f, n) for f, n in scan_ucm_calls() if n not in ucm_bindings}
        assert (
            not missing
        ), f"以下 ucm_prefetch_engine 调用没有对应绑定：{sorted(missing)}"

    def test_fake_engine_methods_are_real_bindings(self):
        """FakeEngineC / FakeUCMPrefetch 的绑定风格方法不得超出真实绑定 ∪ 账本。

        防止单测 fake 显式实现一个并不存在的绑定（历史案例 get_topk_indices，
        已随 A3 整体删除），在 UT 中掩盖生产代码的 AttributeError。
        """
        import fakes

        engine_bindings, ucm_bindings = parse_bindings()
        allowed_engine = engine_bindings | {n for _, n in KNOWN_MISMATCHES}
        engine_fakes = _fake_binding_style_methods(fakes.FakeEngineC)
        unknown = engine_fakes - allowed_engine
        assert (
            not unknown
        ), f"FakeEngineC 实现了真实绑定中不存在且未登记账本的方法：{sorted(unknown)}"
        ucm_fakes = _fake_binding_style_methods(fakes.FakeUCMPrefetch)
        unknown_ucm = ucm_fakes - ucm_bindings
        assert (
            not unknown_ucm
        ), f"FakeUCMPrefetch 实现了真实绑定中不存在的方法：{sorted(unknown_ucm)}"

    def test_binding_set_nonempty_and_key_names_present(self):
        """解析结果健全性：关键绑定必须在位（防止 pybinds.cpp 结构变化导致空集假绿）。"""
        engine_bindings, ucm_bindings = parse_bindings()
        assert {"submit_prefetch_task", "set_blocks_map", "set_tokens_map"} <= (
            engine_bindings
        )
        assert "set_last_indexer_block" in engine_bindings
        assert {
            "store_init",
            "ucm_lookup",
            "ucm_load_ssd",
            "ucm_check",
            "ucm_wait",
        } <= (ucm_bindings)


class TestMergeBaselineDocs:
    def test_baseline_num_spec_slots_default(self):
        """钉住合并基线裁决（报告 S2-2）：gsa.py 的
        estimate_num_slots_sparsed 默认 num_spec_slots=3（合并时取混合基线值，
        分离基线为 0）——该默认值的变更应有意识地裁决并同步更新本断言。
        （示例代码文件已移出本套件断言范围。）
        """

        def _default(path):
            src = path.read_text(encoding="utf-8")
            m = re.search(
                r"def\s+estimate_num_slots_sparsed\s*\([^)]*num_spec_slots\s*=\s*(\d+)",
                src,
            )
            assert m, f"{path.name} 中未找到 estimate_num_slots_sparsed 定义"
            return int(m.group(1))

        assert _default(GSAV2 / "gsa.py") == 3


if __name__ == "__main__":  # 独立运行：py -3 -X utf8 test_binding_contract.py
    engine_bindings, ucm_bindings = parse_bindings()
    discovered = scan_engine_calls()
    missing = {
        (f, n)
        for f, n in discovered
        if n not in engine_bindings and (f, n) not in KNOWN_MISMATCHES
    }
    ledger_stale = [
        e for e in KNOWN_MISMATCHES if e not in discovered or e[1] in engine_bindings
    ]
    ucm_missing = {(f, n) for f, n in scan_ucm_calls() if n not in ucm_bindings}
    print(f"[contract] engine bindings: {len(engine_bindings)}")
    print(f"[contract] engine call sites: {len(discovered)} (files: 2)")
    print(f"[contract] 未导出且不在账本: {sorted(missing) or '无'}")
    print(f"[contract] 账本失效条目: {sorted(ledger_stale) or '无'}")
    print(f"[contract] ucm 未导出调用: {sorted(ucm_missing) or '无'}")
    assert not missing, f"发现账本外的未绑定调用：{sorted(missing)}"
    assert not ledger_stale, f"账本条目已失效：{sorted(ledger_stale)}"
    assert not ucm_missing, f"UCM 未导出调用：{sorted(ucm_missing)}"
    print("[contract] OK")
