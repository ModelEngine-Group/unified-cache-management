"""POSIX AIO adapter interface."""

from __future__ import annotations

import argparse
import importlib.machinery
import math
import os
import sys
from pathlib import Path

from ... import registry
from ...errors import ScriptNotFoundError
from ...registry import ToolAdapter
from ...runner import run_command
from .model_profile import (
    ModelProfileError,
    compute_io_profile,
    detect_architecture,
    load_config,
)

IMPORT_MODE_ENV = "UCM_TOOLKIT_POSIX_AIO_IMPORT"


def _path_resolves_to_repo_root(path: str, repo_root: Path) -> bool:
    """Return whether a sys.path entry points at the repository root."""
    try:
        candidate = Path(path or os.getcwd()).resolve()
    except OSError:
        return False
    return candidate == repo_root


def _ucm_is_available_without_repo_root(repo_root: Path) -> bool:
    """Return whether ucm can be imported without the source tree root."""
    search_path = [
        path for path in sys.path if not _path_resolves_to_repo_root(path, repo_root)
    ]
    return importlib.machinery.PathFinder.find_spec("ucm", search_path) is not None


def _prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    """Prepend a path to PYTHONPATH in a child process environment."""
    value = str(path)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        value if not pythonpath else os.pathsep.join([value, pythonpath])
    )


def _drop_repo_root_from_pythonpath(env: dict[str, str], repo_root: Path) -> None:
    """Remove repository-root entries from child PYTHONPATH."""
    pythonpath = env.get("PYTHONPATH")
    if not pythonpath:
        return

    entries = [
        entry
        for entry in pythonpath.split(os.pathsep)
        if not _path_resolves_to_repo_root(entry, repo_root)
    ]
    if entries:
        env["PYTHONPATH"] = os.pathsep.join(entries)
    else:
        env.pop("PYTHONPATH", None)


class PosixAioTool(ToolAdapter):
    """Adapter for ucm/store/test/e2e/posixstore_aio_test.py."""

    name = "posix-aio"
    aliases = ("posix_aio",)
    description = "Run the POSIX AIO store test script."
    buildable = False
    script_path = "ucm/store/test/e2e/posixstore_aio_test.py"

    def add_run_args(self, parser: argparse.ArgumentParser) -> None:
        """Register POSIX AIO run arguments."""
        parser.add_argument(
            "-w",
            "--worker-number",
            type=int,
            help="worker number: number of worker processes to start concurrently.",
        )
        parser.add_argument(
            "-s",
            "--shard-size",
            type=int,
            help=(
                "shard size: POSIX store I/O size. In layerwise mode, this is "
                "the K/V tensor size for one layer of one block. In non-layerwise "
                "mode, this is the K/V tensor size for all layers of one block."
            ),
        )
        parser.add_argument(
            "-n",
            "--shard-number",
            type=int,
            help="shard number: number of layers in layerwise mode; use 1 in non-layerwise mode.",
        )
        parser.add_argument(
            "-b",
            "--block-number",
            type=int,
            help="block number: total number of blocks.",
        )
        parser.add_argument(
            "-d",
            "--dump-epoch-number",
            type=int,
            help="dump epoch number: number of dump epochs.",
        )
        parser.add_argument(
            "-l",
            "--load-epoch-number",
            type=int,
            help="load epoch number: number of load epochs.",
        )
        parser.add_argument(
            "-o",
            "--storage-backend",
            action="append",
            help="storage backend: storage backend path; may be repeated.",
        )
        parser.add_argument(
            "--model",
            help=(
                "model directory (or config.json) for model-driven mode. When set, "
                "shard-size/shard-number/block-number are computed from the model "
                "config and override any manually provided values."
            ),
        )
        parser.add_argument(
            "--tp",
            type=int,
            default=1,
            help="tensor parallel size; divides num_kv_heads per rank for GQA.",
        )
        parser.add_argument(
            "--input-len",
            type=int,
            default=4096,
            help="request input length; block_number = ceil(input_len / block_size).",
        )
        parser.add_argument(
            "--layerwise",
            action="store_true",
            help="layerwise mode: one shard = one layer. Default is non-layerwise (one shard = all layers).",
        )
        parser.add_argument(
            "--block-size",
            type=int,
            default=128,
            help="vLLM paged block size in tokens; used to derive block_number from input-len.",
        )
        parser.add_argument(
            "--kv-dtype",
            help=(
                "override KV dtype: bfloat16/bf16, float16/fp16, float32/fp32, "
                "float8_e4m3fn/fp8, float8_e5m2, int8. Default: config torch_dtype "
                "or bfloat16."
            ),
        )
        parser.add_argument(
            "--posix-data-trans-concurrency",
            type=int,
            default=32,
            help="posix data transfer concurrency (psync worker count). Default 32.",
        )
        parser.add_argument(
            "--posix-io-engine",
            choices=["psync", "aio"],
            default="aio",
            help="posix io engine: psync or aio. Default aio.",
        )
        parser.add_argument(
            "--io-direct",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="use O_DIRECT for aligned file I/O (default true; --no-io-direct to disable).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="print the computed profile and the forwarded command without launching the script.",
        )

    def _build_run_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="ucm-toolkit run posix-aio",
            description="Run the POSIX AIO store benchmark.",
        )
        self.add_run_args(parser)
        return parser

    @staticmethod
    def _forward_args(args: argparse.Namespace) -> list[str]:
        forwarded: list[str] = []
        option_names = (
            "worker_number",
            "shard_size",
            "shard_number",
            "block_number",
            "dump_epoch_number",
            "load_epoch_number",
            "posix_data_trans_concurrency",
            "posix_io_engine",
        )
        for option_name in option_names:
            value = getattr(args, option_name)
            if value is not None:
                forwarded.extend([f"--{option_name.replace('_', '-')}", str(value)])
        forwarded.append("--io-direct" if args.io_direct else "--no-io-direct")
        if args.storage_backend is not None:
            for path in args.storage_backend:
                forwarded.extend(["--storage-backend", path])
        return forwarded

    def _make_env(self) -> dict[str, str]:
        env = os.environ.copy()
        repo_root = registry.repo_root()
        import_mode = env.get(IMPORT_MODE_ENV, "auto").strip().lower()
        if import_mode == "source":
            _prepend_pythonpath(env, repo_root)
        elif import_mode == "installed" or _ucm_is_available_without_repo_root(
            repo_root
        ):
            _drop_repo_root_from_pythonpath(env, repo_root)
        else:
            _prepend_pythonpath(env, repo_root)
        return env

    def _launch(self, forwarded_args: list[str]) -> int:
        script = registry.resolve_repo_path(self.script_path or "")
        if not script.exists():
            raise ScriptNotFoundError(str(script))
        env = self._make_env()
        return run_command([sys.executable, str(script), *forwarded_args], env=env)

    def _build_model_driven_args(self, args: argparse.Namespace) -> list[str] | None:
        """Compute shard/block sizing from the model config and return forwarded args.

        Returns ``None`` (after printing a message) when the model is missing or
        its architecture is unsupported; the caller turns that into exit code 1.
        """
        try:
            cfg = load_config(args.model)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return None

        arch = detect_architecture(cfg)
        if arch not in ("GQA", "MLA", "DSA"):
            print(
                f"warning: architecture '{arch}' not supported, only GQA and MLA "
                f"family (MLA/DSA) are supported now",
                file=sys.stderr,
            )
            return None

        try:
            profile = compute_io_profile(
                cfg,
                tp=args.tp,
                block_size=args.block_size,
                layerwise=args.layerwise,
                kv_dtype=args.kv_dtype,
            )
        except ModelProfileError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return None

        block_number = (
            math.ceil(args.input_len / args.block_size) if args.block_size else 0
        )

        overridden = [
            name
            for name, value in (
                ("shard-size", args.shard_size),
                ("shard-number", args.shard_number),
                ("block-number", args.block_number),
            )
            if value is not None
        ]
        if overridden:
            print(
                f"warning: model-driven mode overrides manually set: "
                f"{', '.join(overridden)}",
                file=sys.stderr,
            )

        self._print_summary(args, profile, block_number)

        forwarded: list[str] = []
        if args.worker_number is not None:
            forwarded.extend(["--worker-number", str(args.worker_number)])
        forwarded.extend(
            [
                "--shard-size",
                str(profile["shard_size"]),
                "--shard-number",
                str(profile["shard_number"]),
                "--block-number",
                str(block_number),
            ]
        )
        if args.dump_epoch_number is not None:
            forwarded.extend(["--dump-epoch-number", str(args.dump_epoch_number)])
        if args.load_epoch_number is not None:
            forwarded.extend(["--load-epoch-number", str(args.load_epoch_number)])
        forwarded.extend(["--posix-io-engine", str(args.posix_io_engine)])
        forwarded.extend(
            ["--posix-data-trans-concurrency", str(args.posix_data_trans_concurrency)]
        )
        forwarded.append("--io-direct" if args.io_direct else "--no-io-direct")
        if args.storage_backend is not None:
            for path in args.storage_backend:
                forwarded.extend(["--storage-backend", path])
        return forwarded

    @staticmethod
    def _print_summary(
        args: argparse.Namespace, profile: dict, block_number: int
    ) -> None:
        store_block = profile["store_block_size"]
        total_bytes = store_block * block_number
        kv_heads_line = ""
        if profile["num_kv_heads_per_rank"] is not None:
            kv_heads_line = (
                f"  num_kv_heads/rank : {profile['num_kv_heads_per_rank']} "
                f"(tp={profile['tensor_parallel']})\n"
            )
        print("UCM Store IO Info:")
        print(f"  model             : {args.model}")
        print(f"  architecture      : {profile['architecture']}")
        print(f"  num_hidden_layers : {profile['num_hidden_layers']}")
        print(f"  head_dim          : {profile['head_dim']}")
        if kv_heads_line:
            print(kv_heads_line, end="")
        print(
            f"  dtype             : {profile['dtype']} "
            f"({profile['elem_size']} bytes/elem)"
        )
        print(f"  block_size(tokens): {profile['block_size_tokens']}")
        print(f"  input_len         : {args.input_len}")
        print(f"  layerwise         : {profile['layerwise']}")
        plb = profile["per_layer_block_bytes"]
        ss = profile["shard_size"]
        print(
            f"  per_layer/block   : {plb} bytes ({plb / 1024:.2f} KB)"
            f"  |  {profile['per_layer_formula']}"
        )
        print(
            f"  shard size        : {ss} bytes ({ss / 1024:.2f} KB)"
            f"  |  {profile['shard_size_formula']}"
        )
        print(f"  shards per file   : {profile['shard_number']}")
        print(
            f"  file size         : {store_block} bytes "
            f"({store_block / 1024:.2f} KB / {store_block / 1024 / 1024:.2f} MB)"
            f"  |  = shard size * shards per file"
        )
        print(
            f"  block_number      : {block_number} "
            f"(= ceil({args.input_len}/{profile['block_size_tokens']}))"
        )
        print(
            f"  total data/worker : {total_bytes} bytes "
            f"(~{total_bytes / 1024 ** 3:.3f} GiB)"
        )

    def run(self, tool_args: list[str]) -> int:
        """Run the POSIX AIO test script."""
        parser = self._build_run_parser()
        args = parser.parse_args(tool_args)
        if args.model:
            forwarded = self._build_model_driven_args(args)
            if forwarded is None:
                return 1
        else:
            forwarded = self._forward_args(args)
        if args.dry_run:
            script = registry.resolve_repo_path(self.script_path or "")
            cmd = [sys.executable, str(script), *forwarded]
            print("[dry-run] would run: " + " ".join(cmd))
            return 0
        return self._launch(forwarded)

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        """Inspect POSIX AIO script availability."""
        script = registry.resolve_repo_path(self.script_path or "")
        status = "OK" if script.exists() else "MISSING"
        print(f"{self.name}: {script} {status}")
        return 0 if script.exists() else 1
