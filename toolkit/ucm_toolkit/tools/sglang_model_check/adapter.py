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
        parser.add_argument("--device-id", default="0")
        parser.add_argument(
            "--platform",
            choices=("auto", "cuda", "rocm", "ascend", "xpu", "cpu"),
            default="auto",
            help="accelerator platform used for compatibility reporting and visibility",
        )
        parser.add_argument("--page-size", type=int,
                            help="positive page size; omitted: choose by cache family/platform")
        parser.add_argument("--pages", type=int, default=2)
        parser.add_argument("--max-host-mib", type=int, default=512,
                            help="upper bound for estimated host probe and comparison buffers")
        parser.add_argument(
            "--layout",
            default="page_first",
            help="SGLang Host KV Cache layout to probe",
        )
        parser.add_argument("--dtype", default="auto")
        parser.add_argument("--kv-cache-dtype", default="auto",
                            choices=("auto", "bfloat16", "float16", "float32", "fp8_e4m3"),
                            help="physical cache dtype; quantized formats require native pool descriptors")
        parser.add_argument("--dsv4-attn-backend", default="auto",
                            choices=("auto", "flashmla", "flashmla_mega", "trtllm"))
        parser.add_argument("--fp4-indexer", action="store_true",
                            help="probe SGLang's DeepSeek V4 FP4 indexer payload/scale format")
        parser.add_argument("--c128-page-size", type=int, default=16,
                            help="NPU DeepSeek V4 physical C128 page size")
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

    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="ucm-toolkit run sglang-model-check",
            description=self.description,
        )
        self.add_run_args(parser)
        common = {"model", "mode", "storage_backends", "output", "verbose", "help"}
        advanced = parser.add_argument_group("advanced options (see --help-all)")
        for action in list(parser._actions):
            if action.dest not in common:
                for group in parser._action_groups:
                    if action in group._group_actions:
                        group._group_actions.remove(action)
                advanced._group_actions.append(action)
        parser.add_argument("--help-all", action="store_true",
                            help="show all advanced and model-specific options")
        return parser

    def run(self, tool_args: list[str]) -> int:
        try:
            parser = self._parser()
            if "--help-all" in tool_args:
                parser.print_help()
                return 0
            for group in parser._action_groups:
                if group.title.startswith("advanced options"):
                    for action in group._group_actions:
                        action.help = argparse.SUPPRESS
            args = parser.parse_args(tool_args)
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1

        if (args.page_size is not None and args.page_size <= 0) or args.pages <= 0 or args.max_host_mib <= 0:
            raise ToolkitError("--page-size, --pages and --max-host-mib must be positive")
        if args.mode == "roundtrip" and not args.storage_backends:
            raise ToolkitError("--storage-backends is required in roundtrip mode")
        if args.c128_page_size <= 0 or args.c128_page_size % 16:
            raise ToolkitError("--c128-page-size must be a positive multiple of 16")
        if importlib.util.find_spec("sglang") is None:
            raise ToolkitError("sglang-model-check cannot find the sglang package")

        config = CheckConfig(
            model=args.model,
            mode=args.mode,
            device_id=args.device_id,
            platform=args.platform,
            page_size=args.page_size or 0,
            pages=args.pages,
            max_host_mib=args.max_host_mib,
            layout=args.layout,
            dtype=args.dtype,
            kv_cache_dtype=args.kv_cache_dtype,
            dsv4_attn_backend=args.dsv4_attn_backend,
            fp4_indexer=args.fp4_indexer,
            c128_page_size=args.c128_page_size,
            trust_remote_code=args.trust_remote_code,
            skip_meta_model=args.skip_meta_model,
            storage_backends=args.storage_backends,
            connector_name=args.connector_name,
            connector_module_path=args.connector_module_path,
            output=args.output,
            verbose=args.verbose,
        )
        env = os.environ.copy()
        env[CONFIG_ENV] = config.to_json()
        # Keep stdout/stderr suitable for machine-readable checker output. Model
        # generation settings such as top_p are irrelevant to this offline check.
        env["TRANSFORMERS_VERBOSITY"] = "error"
        env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
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
        if platform in {"cuda", "rocm"}:
            env["CUDA_VISIBLE_DEVICES"] = args.device_id
            env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        elif platform == "ascend":
            env["ASCEND_RT_VISIBLE_DEVICES"] = args.device_id
            env.pop("CUDA_VISIBLE_DEVICES", None)
        elif platform == "xpu":
            env["ZE_AFFINITY_MASK"] = args.device_id
        module = f"{__package__}.runner"
        return run_command([sys.executable, "-m", module], env=env)

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        packages = ("torch", "sglang", "ucm")
        missing = [name for name in packages if importlib.util.find_spec(name) is None]
        status = "OK" if not missing else f"MISSING ({', '.join(missing)})"
        print(f"{self.name}: {status}")
        return 0 if not missing else 1
