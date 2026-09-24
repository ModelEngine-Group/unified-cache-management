"""Toolkit adapter for the UCM model compatibility checker."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

from ...errors import ToolkitError
from ...registry import ToolAdapter
from ...runner import run_command
from .config import (
    ADDITIONAL_CONFIG_ENV,
    BLOCK_SIZE_ENV,
    DEVICE_ENV,
    DTYPE_ENV,
    KV_CACHE_DTYPE_ENV,
    MODEL_ENV,
    STORAGE_BACKENDS_ENV,
    STORE_PIPELINE_ENV,
    TOKENS_ENV,
    USE_LAYERWISE_ENV,
)


def _json_object(value: str) -> dict[str, object]:
    """Parse one command-line JSON object."""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {exc}") from exc
    if not isinstance(decoded, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return decoded


def _detect_platform() -> str:
    """Detect the installed serving stack without importing torch or vLLM."""
    if importlib.util.find_spec("vllm_ascend") is not None:
        return "ascend"
    if importlib.util.find_spec("vllm") is not None:
        return "cuda"
    raise ToolkitError(
        "model-check cannot detect an installed vLLM or vLLM-Ascend stack"
    )


class ModelCheckTool(ToolAdapter):
    """Launch the CUDA or Ascend checker in an isolated child process."""

    name = "model-check"
    aliases = ("model_check",)
    description = (
        "Check a model's vLLM KV-cache layout and UCM dump/load compatibility "
        "without loading checkpoint weights."
    )
    buildable = False

    def add_run_args(self, parser: argparse.ArgumentParser) -> None:
        """Register model-check configuration arguments."""
        parser.add_argument(
            "--model",
            help="model directory or Hugging Face model identifier",
        )
        parser.add_argument("--tokens", type=int, help="synthetic request length")
        parser.add_argument(
            "--block-size", type=int, help="vLLM KV-cache block size in tokens"
        )
        parser.add_argument(
            "--layerwise",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="enable or disable UCM layerwise mode",
        )
        parser.add_argument(
            "--additional-config",
            type=_json_object,
            help="vLLM additional_config as a JSON object",
        )
        parser.add_argument("--store-pipeline", help="UCM store pipeline")
        parser.add_argument(
            "--storage-backends",
            help="colon-separated UCM storage backend paths",
        )
        parser.add_argument(
            "--device-id",
            default="0",
            help="physical accelerator id exposed to the checker process",
        )
        parser.add_argument("--dtype", help="vLLM model dtype")
        parser.add_argument("--kv-cache-dtype", help="vLLM KV-cache dtype")

    def _build_run_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="ucm-toolkit run model-check",
            description=self.description,
        )
        self.add_run_args(parser)
        return parser

    def run(self, tool_args: list[str]) -> int:
        """Run the selected checker without importing its runtime dependencies."""
        try:
            args = self._build_run_parser().parse_args(tool_args)
        except SystemExit as exc:
            if isinstance(exc.code, int):
                return exc.code
            return 0 if exc.code is None else 1

        env = os.environ.copy()
        platform = _detect_platform()
        string_options = (
            ("model", MODEL_ENV),
            ("tokens", TOKENS_ENV),
            ("block_size", BLOCK_SIZE_ENV),
            ("store_pipeline", STORE_PIPELINE_ENV),
            ("storage_backends", STORAGE_BACKENDS_ENV),
            ("dtype", DTYPE_ENV),
            ("kv_cache_dtype", KV_CACHE_DTYPE_ENV),
        )
        for option, env_name in string_options:
            value = getattr(args, option)
            if value is not None:
                env[env_name] = str(value)
        if args.additional_config is not None:
            env[ADDITIONAL_CONFIG_ENV] = json.dumps(args.additional_config)
        if args.layerwise is not None:
            env[USE_LAYERWISE_ENV] = str(args.layerwise).lower()
        env[DEVICE_ENV] = args.device_id
        visible_devices_env = (
            "CUDA_VISIBLE_DEVICES"
            if platform == "cuda"
            else "ASCEND_RT_VISIBLE_DEVICES"
        )
        env[visible_devices_env] = args.device_id
        module = f"{__package__}.{platform}"
        return run_command([sys.executable, "-m", module], env=env)

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        """Report whether at least one supported serving stack is importable."""
        common = ("torch", "vllm", "ucm")
        missing_common = [
            name for name in common if importlib.util.find_spec(name) is None
        ]
        cuda_ok = not missing_common
        ascend_ok = cuda_ok and importlib.util.find_spec("vllm_ascend") is not None

        common_status = "OK" if cuda_ok else f"MISSING ({', '.join(missing_common)})"
        print(f"{self.name}: cuda {common_status}")
        ascend_status = "OK" if ascend_ok else "MISSING (vllm_ascend or common stack)"
        print(f"{self.name}: ascend {ascend_status}")
        return 0 if cuda_ok or ascend_ok else 1
