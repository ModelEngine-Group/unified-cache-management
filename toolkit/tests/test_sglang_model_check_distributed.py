"""TP/PP topology and launch contracts; real-device coverage is opt-in."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit.errors import ToolkitError
from ucm_toolkit.tools.sglang_model_check.adapter import SglangModelCheckTool
from ucm_toolkit.tools.sglang_model_check.config import CONFIG_ENV
from ucm_toolkit.tools.sglang_model_check.distributed import (
    _phase, rank_coordinates, require_supported_model, validate_partitions,
)
from ucm_toolkit.tools.sglang_model_check.runner import CheckFailure
from ucm_toolkit.tools.sglang_model_check.result import CheckResult


class DistributedCheckTest(unittest.TestCase):
    def test_rank_coordinates_and_complete_pp_partition(self):
        self.assertEqual([rank_coordinates(rank, 2, 2) for rank in range(4)],
                         [(0, 0), (1, 0), (0, 1), (1, 1)])
        validate_partitions([(0, 2), (0, 2), (2, 5), (2, 5)], 5, 2, 2)
        for ranges in ([(0, 5)] * 4, [(0, 2), (0, 3), (2, 5), (2, 5)],
                       [(0, 2), (0, 2), (3, 5), (3, 5)], [(0, 2)]):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                validate_partitions(ranges, 5, 2, 2)

    def test_known_hybrid_is_allowed_but_specialized_or_unknown_is_not(self):
        facts = dict(attention_arch="MLA", has_linear_attention=False, is_dsa=False,
                     is_deepseek_v4=False, is_minimax_sparse=False, is_hybrid_swa=False)
        require_supported_model(facts)
        require_supported_model({**facts, "has_linear_attention": True})
        require_supported_model({**facts, "attention_arch": "MHA", "is_hybrid_swa": True})
        for field in ("is_dsa", "is_deepseek_v4", "is_minimax_sparse"):
            with self.subTest(field=field), self.assertRaises(CheckFailure):
                require_supported_model({**facts, field: True})
        with self.assertRaises(CheckFailure):
            require_supported_model({"attention_arch": "MHA"})

    def test_phase_propagates_remote_failure_before_next_phase(self):
        def gather(output, local, group):
            output[:] = [local, ("ROUNDTRIP_MISMATCH", "rank 1: mismatch")]
        dist = SimpleNamespace(all_gather_object=gather, get_rank=lambda: 0)
        with self.assertRaises(CheckFailure) as raised:
            _phase(dist, object(), 2, "load_compare", lambda: "local success")
        self.assertEqual(raised.exception.code, "ROUNDTRIP_MISMATCH")
        self.assertIn("rank 1", raised.exception.reason)

    def test_distributed_context_does_not_inject_single_rank_values(self):
        from ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector import server_args_guard

        runtime = SimpleNamespace(get_server_args=lambda: object(), get_parallel=None)
        with patch("ucm_toolkit.tools.sglang_model_check.sglang_compat.inspector.importlib.import_module",
                   return_value=runtime), patch.object(runtime, "get_parallel") as get_parallel:
            with server_args_guard(single_rank=False):
                pass
            get_parallel.assert_not_called()

    @patch("ucm_toolkit.tools.sglang_model_check.adapter.run_command", return_value=0)
    @patch("ucm_toolkit.tools.sglang_model_check.adapter.importlib.util.find_spec", return_value=object())
    def test_default_or_explicit_one_uses_original_single_process(self, find_spec, run_command):
        for options in ([], ["--tp-size", "1", "--pp-size", "1"]):
            with self.subTest(options=options):
                SglangModelCheckTool().run(["--model", "model", "--platform", "cuda"] + options)
                command = run_command.call_args.args[0]
                env = run_command.call_args.kwargs["env"]
                self.assertEqual(command, [sys.executable, "-m", "ucm_toolkit.tools.sglang_model_check.runner"])
                config = json.loads(env[CONFIG_ENV])
                self.assertEqual((config["tp_size"], config["pp_size"]), (1, 1))
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    @patch("ucm_toolkit.tools.sglang_model_check.adapter.run_command", return_value=0)
    @patch("ucm_toolkit.tools.sglang_model_check.adapter.importlib.util.find_spec", return_value=object())
    def test_launch_uses_distinct_devices_and_torchrun(self, find_spec, run_command):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,4,6,7"}):
            code = SglangModelCheckTool().run([
                "--model", "model", "--mode", "roundtrip", "--storage-backends", "/tmp/check",
                "--platform", "cuda", "--tp-size", "2", "--pp-size", "2",
            ])
        self.assertEqual(code, 0)
        command, env = run_command.call_args.args[0], run_command.call_args.kwargs["env"]
        self.assertIn("torch.distributed.run", command)
        self.assertIn("--nproc-per-node=4", command)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,4,6,7")
        self.assertTrue(command[-1].endswith(".distributed"))
        config = json.loads(env[CONFIG_ENV])
        self.assertEqual((config["tp_size"], config["pp_size"]), (2, 2))

    @patch("ucm_toolkit.tools.sglang_model_check.adapter.importlib.util.find_spec", return_value=object())
    def test_invalid_distributed_requests_are_rejected(self, find_spec):
        base = ["--model", "model", "--mode", "roundtrip", "--storage-backends", "/tmp/check",
                "--platform", "cuda", "--tp-size", "2"]
        for extra in (["--device-id", "0"], ["--device-id", "0,0"],
                      ["--skip-meta-model"], ["--mode", "inspect"], ["--pp-size", "0"]):
            with self.subTest(extra=extra), self.assertRaises(ToolkitError):
                SglangModelCheckTool().run(base + extra)

    def test_summary_describes_tp_pp_scope_not_inference(self):
        result = CheckResult(model="model", mode="roundtrip")
        result.validation = dict(scope="tp_pp_host_storage_contract", tp_size=2, pp_size=2)
        result.succeed("roundtrip")
        summary = result.to_dict()["summary"]
        self.assertIn("TP=2 PP=2", summary)
        self.assertEqual(summary, "TP=2 PP=2: structure, communication and UCM storage checks passed.")


@unittest.skipUnless(os.environ.get("UCM_SGLANG_TP_PP_TEST_MODEL") and
                     os.environ.get("UCM_SGLANG_TP_PP_TEST_BACKENDS"),
                     "set a real MHA/MLA model, dedicated backend and four visible devices")
class RealDistributedTest(unittest.TestCase):
    def test_tp_and_pp_on_real_devices(self):
        for tp, pp in ((2, 1), (1, 2), (2, 2)):
            with self.subTest(tp=tp, pp=pp):
                command = [sys.executable, "-m", "ucm_toolkit", "run", "sglang-model-check",
                    "--model", os.environ["UCM_SGLANG_TP_PP_TEST_MODEL"], "--mode", "roundtrip",
                    "--storage-backends", os.environ["UCM_SGLANG_TP_PP_TEST_BACKENDS"],
                    "--tp-size", str(tp), "--pp-size", str(pp), "--max-host-mib", "2048", "--verbose"]
                subprocess.run(command, check=True, timeout=360)


if __name__ == "__main__":
    unittest.main()
