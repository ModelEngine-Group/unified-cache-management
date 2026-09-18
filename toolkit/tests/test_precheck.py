"""Unit tests for the precheck tool logic (parsers, config, selection, reporter).

Pure-stdlib ``unittest``; no hardware or ucm required. The check-function
decision tests mock ``subprocess``/``os``/``import`` so they are deterministic
across hosts.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ucm_toolkit.tools.precheck import checks as checks_mod
from ucm_toolkit.tools.precheck.bandwidth import (
    ComboResult,
    bandwidth_detail,
    bandwidth_status,
    best_per_metric,
    pick_best,
)
from ucm_toolkit.tools.precheck.checks import (
    check_accelerator_driver,
    check_aio_resources,
    check_kernel_version,
)
from ucm_toolkit.tools.precheck.config import PrecheckConfig
from ucm_toolkit.tools.precheck.parseutil import (
    compare_versions,
    normalize_vllm_ascend_version,
    parse_int_list,
    parse_npu_smi_versions,
    parse_size,
    parse_size_list,
    parse_str_list,
    parse_version,
    strip_build,
)
from ucm_toolkit.tools.precheck.reporter import (
    FAIL,
    INFO,
    STATUS_FAIL,
    STATUS_INFO,
    STATUS_OK,
    STATUS_PASS,
    STATUS_SKIP,
    STATUS_WARN,
    WARN,
    CheckResult,
    overall_failed,
)


class TestVersionParsing(unittest.TestCase):
    def test_parse_version_strips_suffix(self):
        self.assertEqual(parse_version("5.15.0-25-generic"), (5, 15, 0))
        self.assertEqual(parse_version("8.0"), (8, 0))
        self.assertEqual(parse_version("25.2.0"), (25, 2, 0))

    def test_parse_version_garbage(self):
        self.assertEqual(parse_version(""), ())
        self.assertEqual(parse_version(None), ())

    def test_compare_versions_pad(self):
        self.assertEqual(compare_versions((5, 15), (5, 10)), 1)
        self.assertEqual(compare_versions((5, 10, 0), (5, 10, 0)), 0)
        self.assertEqual(compare_versions((5, 9), (5, 10)), -1)

    def test_strip_build(self):
        self.assertEqual(strip_build("0.18.0+empty"), "0.18.0")
        self.assertIsNone(strip_build(None))

    def test_normalize_vllm_ascend(self):
        self.assertEqual(normalize_vllm_ascend_version("0.18.0rc1+local"), "0.18.0")
        self.assertEqual(normalize_vllm_ascend_version("0.11.0.post1"), "0.11.0")


class TestKernelBoundary(unittest.TestCase):
    """>= 5.10 series (major.minor) — validated on openEuler 5.10.0-216."""

    FLOOR_MM = parse_version("5.10")[:2]

    def _mm(self, release):
        v = parse_version(release)
        return v[:2] if len(v) >= 2 else v + (0,) * (2 - len(v))

    def _passes(self, release):
        return compare_versions(self._mm(release), self.FLOOR_MM) >= 0

    def test_boundary(self):
        self.assertTrue(self._passes("5.10.0"))
        self.assertTrue(self._passes("5.10.0-216.0.0.115.oe2203sp4.aarch64"))
        self.assertTrue(self._passes("5.11"))
        self.assertFalse(self._passes("5.9"))
        self.assertTrue(self._passes("6.8.0-25-generic"))


class TestSizeParsing(unittest.TestCase):
    def test_suffixes(self):
        self.assertEqual(parse_size("180k"), 180 * 1024)
        self.assertEqual(parse_size("1m"), 1024 * 1024)
        self.assertEqual(parse_size("2g"), 2 * 1024**3)
        self.assertEqual(parse_size("4MiB"), 4 * 1024**2)

    def test_lists(self):
        self.assertEqual(parse_size_list("180k,1m"), [184320, 1048576])
        self.assertEqual(parse_int_list("1,16"), [1, 16])
        self.assertEqual(parse_str_list("psync, aio"), ["psync", "aio"])

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            parse_size("nope")


class TestNpuSmi(unittest.TestCase):
    def test_banner_version_fallback(self):
        # Real openEuler 25.5.2 firmware: version only in the banner.
        sample = (
            "+------------------------------------------------------+\n"
            "| npu-smi 25.5.2                   Version: 25.5.2       |\n"
            "+----------------------------+---------------+----------+\n"
        )
        parsed = parse_npu_smi_versions(sample)
        self.assertEqual(parsed["tool"], "25.5.2")
        self.assertEqual(parsed["version"], "25.5.2")
        self.assertEqual(parsed["hdk"], "25.5.2")

    def test_hdk_label_wins(self):
        sample = "npu-smi 24.1.0\nNPU0 Driver Version: 25.3.0 HDK Version: 25.3.0\n"
        parsed = parse_npu_smi_versions(sample)
        self.assertEqual(parsed["hdk"], "25.3.0")
        self.assertEqual(parsed["driver"], "25.3.0")


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        # The runtime defaults come from the shipped precheck.defaults.json
        # (code constants are the fallback).
        cfg = PrecheckConfig.default()
        self.assertEqual(cfg.kernel_min, "5.10")
        self.assertEqual(cfg.cuda_min_compute_cap, 8.0)
        self.assertEqual(cfg.ascend_min_hdk, "25.2.0")
        self.assertEqual(cfg.shm_min_gib, 512.0)
        self.assertEqual(cfg.bandwidth.shard_sizes, [184320, 8388608])
        self.assertEqual(cfg.bandwidth.worker_counts, [1, 8, 16])
        self.assertEqual(cfg.bandwidth.engines, ["psync", "aio"])
        self.assertEqual(cfg.bandwidth.modes, ["dump", "read", "mix"])
        self.assertEqual(cfg.bandwidth.block_number, 32)
        self.assertEqual(cfg.bandwidth.dump_epochs, 8)
        self.assertEqual(cfg.bandwidth.load_epochs, 8)
        self.assertEqual(cfg.bandwidth.mixed_epochs, 8)
        self.assertEqual(cfg.bandwidth.rw_ratio, 4)
        self.assertEqual(cfg.bandwidth.barrier_timeout, 60)
        self.assertEqual(cfg.bandwidth.combo_timeout, 120)
        self.assertEqual(cfg.bandwidth.aio_queue_depth, 4096)
        self.assertEqual(cfg.bandwidth.threshold_gb, 8.0)

    def test_bare_falls_back_to_code_constants(self):
        # PrecheckConfig() (no JSON) uses the code-constant fallback.
        cfg = PrecheckConfig()
        self.assertEqual(cfg.bandwidth.shard_sizes, [184320, 8388608])
        self.assertEqual(cfg.bandwidth.engines, ["psync", "aio"])
        self.assertEqual(cfg.shm_min_gib, 512.0)

    def test_from_dict_sizes_as_strings(self):
        cfg = PrecheckConfig.from_dict(
            {
                "mount_path": "/mnt/x",
                "bandwidth": {"shard_sizes": ["180k", "1m"], "threshold_gb": 10.0},
            }
        )
        self.assertEqual(cfg.mount_path, "/mnt/x")
        self.assertEqual(cfg.bandwidth.shard_sizes, [184320, 1048576])
        self.assertEqual(cfg.bandwidth.threshold_gb, 10.0)


class TestBandwidthSelection(unittest.TestCase):
    def _combo(self, comprehensive, ok=True, mixed=0.0):
        c = ComboResult(
            shard_size=1024,
            worker_count=1,
            engine="aio",
            ok=ok,
            error="" if ok else "err",
        )
        c.comprehensive = c.dump_bw = c.load_bw = comprehensive
        c.mixed_bw = mixed
        c.rw_ratio = 4 if mixed > 0 else 0
        return c

    def test_pick_best(self):
        best = pick_best([self._combo(5.0), self._combo(12.0), self._combo(9.0)])
        self.assertEqual(best.comprehensive, 12.0)

    def test_pick_best_ignores_failed(self):
        best = pick_best([self._combo(5.0, ok=False), self._combo(8.0)])
        self.assertEqual(best.comprehensive, 8.0)

    def test_status_all_metrics_pass(self):
        combos = [self._combo(8.0, mixed=8.0)]
        mb = best_per_metric(combos)
        self.assertEqual(bandwidth_status(mb, 8.0), STATUS_PASS)

    def test_status_any_metric_below_warns(self):
        # dump=5.0 < 8.0 threshold, load=10.0 >= 8.0 → WARN
        combos = [self._combo(5.0, mixed=10.0)]
        mb = best_per_metric(combos)
        self.assertEqual(bandwidth_status(mb, 8.0), STATUS_WARN)

    def test_status_no_combos(self):
        self.assertEqual(bandwidth_status([], 8.0), STATUS_WARN)

    def test_detail_shows_per_metric(self):
        combos = [self._combo(4.0, mixed=10.0)]
        mb = best_per_metric(combos)
        detail = bandwidth_detail(mb, 8.0)
        self.assertIn("dump", detail)
        self.assertIn("load", detail)
        self.assertIn("mixed", detail)
        self.assertIn("WARN", detail)

    def test_best_per_metric_different_combos(self):
        # combo A: high dump, low load; combo B: low dump, high load
        a = self._combo(10.0)
        a.dump_bw, a.load_bw = 10.0, 2.0
        a.comprehensive = 6.0
        b = self._combo(10.0)
        b.dump_bw, b.load_bw = 2.0, 10.0
        b.comprehensive = 6.0
        mb = best_per_metric([a, b])
        dump_best = dict((lbl, c) for lbl, c, _ in mb)
        self.assertIs(dump_best["dump"], a)
        self.assertIs(dump_best["load"], b)

    def test_mixed_is_headline_when_present(self):
        c = self._combo(6.0, mixed=15.0)
        from ucm_toolkit.tools.precheck.bandwidth import headline_bw

        self.assertEqual(headline_bw(c), 15.0)
        # pick_best prefers the combo with higher mixed.
        lo = self._combo(20.0)  # high comprehensive, no mixed
        hi = self._combo(5.0, mixed=25.0)
        self.assertIs(pick_best([lo, hi]), hi)


class TestReporterExitCodes(unittest.TestCase):
    def _r(self, severity, status):
        return CheckResult(name="x", severity=severity, status=status)

    def test_no_fail_exits_zero(self):
        self.assertFalse(
            overall_failed([self._r(FAIL, STATUS_PASS), self._r(WARN, STATUS_WARN)])
        )

    def test_fail_exits_nonzero(self):
        self.assertTrue(overall_failed([self._r(FAIL, STATUS_FAIL)]))

    def test_strict_promotes_warn(self):
        self.assertFalse(overall_failed([self._r(WARN, STATUS_WARN)]))
        self.assertTrue(overall_failed([self._r(WARN, STATUS_WARN)], strict=True))


class TestAioResources(unittest.TestCase):
    def test_reads_aio_limits(self):
        from unittest.mock import MagicMock, patch

        def mock_file(data):
            m = MagicMock()
            m.__enter__.return_value.read.return_value = data
            return m

        with patch(
            "builtins.open",
            side_effect=[mock_file("65536\n"), mock_file("4096\n")],
        ):
            r = check_aio_resources(PrecheckConfig())
        self.assertEqual(r.severity, INFO)
        self.assertEqual(r.status, STATUS_INFO)
        self.assertIn("max-nr=65536", r.value)
        self.assertIn("max_aio_workers=15", r.value)

    def test_unavailable_returns_info(self):
        r = check_aio_resources(PrecheckConfig())
        # On non-Linux (Windows test host) the files don't exist.
        self.assertEqual(r.severity, INFO)
        self.assertEqual(r.status, STATUS_INFO)


class TestKernelCheckRemediation(unittest.TestCase):
    """RFC #1208: FAIL items must carry remediation advice."""

    def _cfg(self, floor="5.10"):
        cfg = PrecheckConfig()
        cfg.kernel_min = floor
        return cfg

    def _patch_uname(self, release):
        return patch.object(checks_mod, "_run", return_value=(0, release + "\n", ""))

    def test_pass(self):
        with self._patch_uname("5.15.0"):
            r = check_kernel_version(self._cfg())
        self.assertEqual(r.status, STATUS_PASS)
        self.assertEqual(r.remediation, "")

    def test_fail_carries_remediation(self):
        with self._patch_uname("5.9.0"):
            r = check_kernel_version(self._cfg())
        self.assertEqual(r.status, STATUS_FAIL)
        self.assertTrue(r.remediation)
        self.assertIn("upgrade", r.remediation)


class TestAcceleratorDriver(unittest.TestCase):
    def _cfg(self, cap=8.0, hdk="25.2.0"):
        cfg = PrecheckConfig()
        cfg.cuda_min_compute_cap = cap
        cfg.ascend_min_hdk = hdk
        return cfg

    def test_ascend_banner_version_pass(self):
        # Real openEuler 25.5.2 banner (no per-card HDK/Driver label).
        real_out = (
            "+------------------------------------------------------+\n"
            "| npu-smi 25.5.2                   Version: 25.5.2       |\n"
            "+----------------------------+---------------+----------+\n"
        )
        with (
            patch.object(checks_mod, "_have", side_effect=lambda c: c == "npu-smi"),
            patch.object(checks_mod, "_run", return_value=(0, real_out, "")),
        ):
            r = check_accelerator_driver(self._cfg())
        self.assertEqual(r.status, STATUS_PASS)
        self.assertIn("HDK=25.5.2", r.value)

    def test_ascend_below_floor_warns_with_remediation(self):
        with (
            patch.object(checks_mod, "_have", side_effect=lambda c: c == "npu-smi"),
            patch.object(
                checks_mod, "_run", return_value=(0, "Driver Version: 25.1.0\n", "")
            ),
        ):
            r = check_accelerator_driver(self._cfg())
        self.assertEqual(r.status, STATUS_WARN)
        self.assertTrue(r.remediation)
        self.assertIn("upgrade", r.remediation)

    def test_neither_smi_fails_with_remediation(self):
        # No GPU/NPU driver is a hard failure (UCM needs an accelerator), not a
        # benign skip that lets the overall result pass.
        with patch.object(checks_mod, "_have", return_value=False):
            r = check_accelerator_driver(self._cfg())
        self.assertEqual(r.severity, FAIL)
        self.assertEqual(r.status, STATUS_FAIL)
        self.assertTrue(r.remediation)


if __name__ == "__main__":
    unittest.main()
