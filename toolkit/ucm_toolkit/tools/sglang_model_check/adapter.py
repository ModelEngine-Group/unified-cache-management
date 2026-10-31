"""Toolkit adapter for the offline SGLang compatibility checker."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

from ...errors import ToolkitError
from ...registry import ToolAdapter
from ...runner import run_command
from .config import CONFIG_ENV, CheckConfig


class SglangModelCheckTool(ToolAdapter):
    name = "sglang-model-check"
    aliases = ("sglang_model_check",)
    description = (
        "Check SGLang model/UCM cache compatibility without starting a service "
        "or loading checkpoint weights."
    )
    buildable = False

    def add_run_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", required=True, help="model path or HF model id")
        parser.add_argument(
            "--mode",
            choices=("inspect", "roundtrip"),
            default="inspect",
            help="inspect configuration/structure (default), or explicitly select roundtrip for host-cache IO",
        )
        parser.add_argument(
            "--device-id",
            help="visible device IDs; defaults to 0, or the first TP*PP visible devices",
        )
        parser.add_argument("--tp-size", type=int, default=1)
        parser.add_argument("--pp-size", type=int, default=1)
        parser.add_argument(
            "--distributed-timeout",
            type=int,
            default=300,
            help="multi-rank worker/collective timeout in seconds",
        )
        parser.add_argument(
            "--platform",
            choices=("auto", "cuda", "rocm", "ascend", "xpu", "cpu"),
            default="auto",
            help="accelerator platform used for compatibility reporting and visibility",
        )
        parser.add_argument(
            "--page-size",
            type=int,
            help="positive page size; omitted: choose by cache family/platform",
        )
        parser.add_argument("--pages", type=int, default=2)
        parser.add_argument(
            "--max-host-mib",
            type=int,
            default=512,
            help="upper bound for estimated host probe and comparison buffers",
        )
        parser.add_argument(
            "--layout",
            default="page_first",
            help="SGLang Host KV Cache layout to probe",
        )
        parser.add_argument("--dtype", default="auto")
        parser.add_argument(
            "--kv-cache-dtype",
            default="auto",
            choices=("auto", "bfloat16", "float16", "float32", "fp8_e4m3"),
            help="physical cache dtype; quantized formats require native pool descriptors",
        )
        parser.add_argument(
            "--dsv4-attn-backend",
            default="auto",
            choices=("auto", "flashmla", "flashmla_mega", "trtllm"),
        )
        parser.add_argument(
            "--fp4-indexer",
            action="store_true",
            help="probe SGLang's DeepSeek V4 FP4 indexer payload/scale format",
        )
        parser.add_argument(
            "--c128-page-size",
            type=int,
            default=16,
            help="NPU DeepSeek V4 physical C128 page size",
        )
        parser.add_argument(
            "--trust-remote-code",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        parser.add_argument(
            "--skip-meta-model",
            action="store_true",
            help="skip meta-device model construction (weaker config-only result)",
        )
        parser.add_argument(
            "--storage-backends",
            help="colon-separated paths passed to UcmPipelineStore; required for roundtrip",
        )
        parser.add_argument("--connector-name", default="UcmPipelineStore")
        parser.add_argument("--connector-module-path")
        parser.add_argument("--output", help="also write the JSON result to this path")
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="include detection sources, full environment details and traceback",
        )

    def _parser(self, show_all: bool = False) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="ucm-toolkit run sglang-model-check",
            description=self.description,
        )
        self.add_run_args(parser)
        common = {
            "model",
            "mode",
            "storage_backends",
            "output",
            "verbose",
            "help",
            "tp_size",
            "pp_size",
        }
        if not show_all:
            for action in parser._actions:
                if action.dest not in common:
                    action.help = argparse.SUPPRESS
        parser.add_argument(
            "--help-all",
            action="store_true",
            help="show all advanced and model-specific options",
        )
        return parser

    def run(self, tool_args: list[str]) -> int:
        try:
            show_all = "--help-all" in tool_args
            parser = self._parser(show_all=show_all)
            if show_all:
                parser.print_help()
                return 0
            args = parser.parse_args(tool_args)
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1

        if (
            (args.page_size is not None and args.page_size <= 0)
            or args.pages <= 0
            or args.max_host_mib <= 0
        ):
            raise ToolkitError(
                "--page-size, --pages and --max-host-mib must be positive"
            )
        if args.mode == "roundtrip" and not args.storage_backends:
            raise ToolkitError("--storage-backends is required in roundtrip mode")
        if args.c128_page_size <= 0 or args.c128_page_size % 16:
            raise ToolkitError("--c128-page-size must be a positive multiple of 16")
        if min(args.tp_size, args.pp_size, args.distributed_timeout) <= 0:
            raise ToolkitError(
                "--tp-size, --pp-size and --distributed-timeout must be positive"
            )
        world_size = args.tp_size * args.pp_size
        if world_size > 1 and (args.mode != "roundtrip" or args.skip_meta_model):
            raise ToolkitError(
                "TP/PP validation requires --mode roundtrip and meta-model construction"
            )
        if importlib.util.find_spec("sglang") is None:
            raise ToolkitError("sglang-model-check cannot find the sglang package")

        args.page_size = args.page_size or 0
        env = os.environ.copy()
        # Keep stdout/stderr suitable for machine-readable checker output. Model
        # generation settings such as top_p are irrelevant to this offline check.
        env["TRANSFORMERS_VERBOSITY"] = "error"
        env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        if not args.verbose:
            # Child-only settings: retain warnings/errors and leave the caller's
            # environment and production logger configuration untouched.
            env["UCM_LOG_LEVEL"] = "warning"
            env["UC_LOGGER_LEVEL"] = "warning"
        platform = args.platform
        if platform == "auto":
            ascend_env = env.get("ASCEND_RT_VISIBLE_DEVICES") or env.get(
                "ASCEND_VISIBLE_DEVICES"
            )
            platform = (
                "ascend"
                if ascend_env or importlib.util.find_spec("torch_npu") is not None
                else "cuda"
            )
        if world_size > 1 and platform not in {"cuda", "rocm", "ascend"}:
            raise ToolkitError("TP/PP validation supports CUDA/ROCm/Ascend only")
        visibility = (
            "ASCEND_RT_VISIBLE_DEVICES"
            if platform == "ascend"
            else "CUDA_VISIBLE_DEVICES"
        )
        if args.device_id is None:
            visible = env.get(
                visibility,
                env.get("ASCEND_VISIBLE_DEVICES", "") if platform == "ascend" else "",
            )
            args.device_id = (
                (
                    ",".join(visible.split(",")[:world_size])
                    if visible
                    else ",".join(str(i) for i in range(world_size))
                )
                if world_size > 1
                else "0"
            )
        devices = [device.strip() for device in args.device_id.split(",")]
        args.device_id = ",".join(devices)
        if world_size > 1 and (
            not all(devices)
            or len(devices) != world_size
            or len(set(devices)) != world_size
        ):
            raise ToolkitError("TP*PP distinct visible devices are required")
        if platform in {"cuda", "rocm"}:
            env["CUDA_VISIBLE_DEVICES"] = args.device_id
            env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        elif platform == "ascend":
            env["ASCEND_RT_VISIBLE_DEVICES"] = args.device_id
            env.pop("CUDA_VISIBLE_DEVICES", None)
        elif platform == "xpu":
            env["ZE_AFFINITY_MASK"] = args.device_id
        config = CheckConfig.from_mapping(vars(args))
        env[CONFIG_ENV] = config.to_json()
        if world_size > 1:
            env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
            return run_command(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nnodes=1",
                    f"--nproc-per-node={world_size}",
                    "--max-restarts=0",
                    "--module",
                    f"{__package__}.distributed",
                ],
                env=env,
            )
        module = f"{__package__}.runner"
        return run_command([sys.executable, "-m", module], env=env)

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        packages = ("torch", "sglang", "ucm")
        missing = [name for name in packages if importlib.util.find_spec(name) is None]
        status = "OK" if not missing else f"MISSING ({', '.join(missing)})"
        print(f"{self.name}: {status}")
        return 0 if not missing else 1
