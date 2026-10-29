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
    ASCEND_SOC_ENV,
    BLOCK_SIZE_ENV,
    CONNECTOR_MODULE_PATH_ENV,
    CUDA_CAPABILITY_ENV,
    DEVICE_ENV,
    DTYPE_ENV,
    KV_CACHE_DTYPE_ENV,
    LAYOUT_ONLY_ENV,
    LEGACY_CONNECTOR_MODULE,
    MODEL_ENV,
    STORAGE_BACKENDS_ENV,
    STORE_PIPELINE_ENV,
    TOKENS_ENV,
    TOKENS_PER_CHUNK_ENV,
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


# Launcher module per platform.  ``ascend-sim`` uses vLLM-Ascend's native
# KV-cache construction without an NPU: tensors live in host memory.
PLATFORM_MODULES = {
    "ascend": "ascend",
    "ascend-sim": "ascend_sim",
    "cuda": "cuda",
    "cuda-sim": "cuda_sim",
    "cpu": "cpu",
}
# Platforms whose checker needs no accelerator device.
DEVICELESS_PLATFORMS = ("cpu", "ascend-sim", "cuda-sim")


def _detect_platform() -> str:
    """Detect the installed serving stack without importing torch or vLLM."""
    if importlib.util.find_spec("vllm_ascend") is not None:
        return "ascend"
    if importlib.util.find_spec("vllm") is not None:
        try:
            from importlib.metadata import version as _pkg_version

            if "+cpu" in _pkg_version("vllm"):
                return "cpu"
        except Exception:
            pass
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
        for dimension in ("tp", "pp", "pcp", "dcp"):
            parser.add_argument(f"--{dimension}", type=int, default=1)
        parser.add_argument("--tokens", type=int, help="synthetic request length")
        parser.add_argument(
            "--tokens-per-chunk", type=int, help="connector-v2 tokens per UCM chunk"
        )
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
        parser.add_argument(
            "--connector-module-path",
            default=LEGACY_CONNECTOR_MODULE,
            help="module containing the UCMConnector facade",
        )
        parser.add_argument(
            "--platform",
            choices=("auto", *PLATFORM_MODULES),
            default="auto",
            help=(
                "checker launcher; auto detects the installed stack. "
                "ascend-sim and cuda-sim build the native KV layout of their "
                "platform in host memory and need no accelerator"
            ),
        )
        parser.add_argument(
            "--ascend-soc",
            help=(
                "ascend-sim only: assert the installed vLLM-Ascend is built for "
                "this SoC family, e.g. A2, A3, A5"
            ),
        )
        parser.add_argument(
            "--cuda-capability",
            help=(
                "cuda-sim only: compute capability whose layout is simulated, "
                "e.g. 9.0 (required there; it cannot be read without a GPU)"
            ),
        )
        parser.add_argument(
            "--layout-only",
            action="store_true",
            help="stop after UCM parsed the native KV layout (no dump/load)",
        )

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

        if min(args.tp, args.pp, args.pcp, args.dcp) < 1 or args.tp % args.dcp:
            raise ToolkitError("Parallel sizes must be positive and DCP must divide TP")
        env = os.environ.copy()
        for dimension in ("tp", "pp", "pcp", "dcp"):
            env[f"UCM_MODEL_CHECK_{dimension.upper()}"] = str(getattr(args, dimension))
        import time

        env["UCM_MODEL_CHECK_TOKEN_SALT"] = str(time.time_ns())
        env.setdefault("VLLM_DIST_IDENT", env["UCM_MODEL_CHECK_TOKEN_SALT"])
        platform = _detect_platform() if args.platform == "auto" else args.platform
        string_options = (
            ("model", MODEL_ENV),
            ("tokens", TOKENS_ENV),
            ("tokens_per_chunk", TOKENS_PER_CHUNK_ENV),
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
        env[CONNECTOR_MODULE_PATH_ENV] = args.connector_module_path
        if args.layout_only:
            env[LAYOUT_ONLY_ENV] = "true"
        if args.ascend_soc is not None:
            env[ASCEND_SOC_ENV] = args.ascend_soc
        if args.cuda_capability is not None:
            env[CUDA_CAPABILITY_ENV] = args.cuda_capability
        if platform == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = args.device_id
        elif platform == "ascend":
            env["ASCEND_RT_VISIBLE_DEVICES"] = args.device_id
        # cpu / ascend-sim / cuda-sim: no device-visibility variable needed
        module = f"{__package__}.{PLATFORM_MODULES[platform]}"
        world = args.tp * args.pp * args.pcp
        if world > 1:
            if (
                platform not in DEVICELESS_PLATFORMS
                and len(args.device_id.split(",")) < world
            ):
                raise ToolkitError(f"Need {world} visible devices for this topology")
            return run_command(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={world}",
                    "--module",
                    module,
                ],
                env=env,
            )
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
        print(f"{self.name}: ascend-sim {ascend_status} (no NPU required)")
        return 0 if cuda_ok or ascend_ok else 1
