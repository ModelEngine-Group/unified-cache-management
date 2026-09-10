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
            help="inspect requirements, or additionally perform UCM host-cache IO",
        )
        parser.add_argument("--device-id", default="0")
        parser.add_argument("--page-size", type=int, default=64)
        parser.add_argument("--pages", type=int, default=2)
        parser.add_argument(
            "--layout",
            default="page_first",
            help="SGLang Host KV Cache layout to probe",
        )
        parser.add_argument("--dtype", default="auto")
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

    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="ucm-toolkit run sglang-model-check",
            description=self.description,
        )
        self.add_run_args(parser)
        return parser

    def run(self, tool_args: list[str]) -> int:
        try:
            args = self._parser().parse_args(tool_args)
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1

        if args.page_size <= 0 or args.pages <= 0:
            raise ToolkitError("--page-size and --pages must be positive")
        if args.mode == "roundtrip" and not args.storage_backends:
            raise ToolkitError("--storage-backends is required in roundtrip mode")
        if importlib.util.find_spec("sglang") is None:
            raise ToolkitError("sglang-model-check cannot find the sglang package")

        config = CheckConfig(
            model=args.model,
            mode=args.mode,
            device_id=args.device_id,
            page_size=args.page_size,
            pages=args.pages,
            layout=args.layout,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            skip_meta_model=args.skip_meta_model,
            storage_backends=args.storage_backends,
            connector_name=args.connector_name,
            connector_module_path=args.connector_module_path,
            output=args.output,
        )
        env = os.environ.copy()
        env[CONFIG_ENV] = config.to_json()
        env["CUDA_VISIBLE_DEVICES"] = args.device_id
        env["ASCEND_RT_VISIBLE_DEVICES"] = args.device_id
        module = f"{__package__}.runner"
        return run_command([sys.executable, "-m", module], env=env)

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        packages = ("torch", "sglang", "ucm")
        missing = [name for name in packages if importlib.util.find_spec(name) is None]
        status = "OK" if not missing else f"MISSING ({', '.join(missing)})"
        print(f"{self.name}: {status}")
        return 0 if not missing else 1
