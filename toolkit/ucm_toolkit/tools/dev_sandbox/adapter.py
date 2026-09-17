"""dev-sandbox adapter interfaces."""

from __future__ import annotations

import argparse
import importlib.resources
import os
import shutil
from pathlib import Path

from ...errors import (
    BinaryNotFoundError,
    BuildDirNotFoundError,
    CommandNotFoundError,
    ToolkitError,
)
from ...registry import ToolAdapter
from ...runner import check_command, command_exists, run_command

# Friendly-mode copy-case matrix: (model_type, iodirect, sdma) -> Ascend case
# name. Sourced from toolkit/src/dev-sandbox/module/copy/ascend/*.cc. MLA does
# not distinguish iodirect: the shared-memory path is the same whether or not
# O_DIRECT is requested, so the iodirect=true cells reuse the iodirect=false
# case for each sdma value.
COPY_CASE_MATRIX: dict[tuple[str, bool, bool], str] = {
    ("gqa", False, False): "all_host_to_all_device_ce_multi_stream",
    ("gqa", True, False): "all_odirect_host_to_all_device_ce_multi_stream",
    ("gqa", False, True): "all_host_to_all_device_ffts_direct_h2d",
    ("gqa", True, True): "all_odirect_host_to_all_device_ffts_direct_h2d",
    ("mla", False, False): "one_share_host_to_all_device_ce_multi_stream",
    ("mla", True, False): "one_share_host_to_all_device_ce_multi_stream",
    ("mla", False, True): "one_share_host_to_all_device_ffts_direct_h2d",
    ("mla", True, True): "one_share_host_to_all_device_ffts_direct_h2d",
}


def _parse_bool(value: str) -> bool:
    """Parse a true/false CLI value (case-insensitive)."""
    text = value.strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


class DevSandboxTool(ToolAdapter):
    """Build adapter for the dev-sandbox CMake project (shipped in the wheel)."""

    name = "dev-sandbox"
    aliases = ("dev_sandbox",)
    description = "Build the CMake-based dev-sandbox test project."
    buildable = True
    subcommands = {
        "copy": "module/copy/copy",
        "trans": "module/trans/trans",
        "aio": "module/aio/aio",
    }

    def _source_dir(self) -> Path:
        """Installed location of the dev-sandbox CMake sources (package data)."""
        return Path(importlib.resources.files("ucm_toolkit._native.dev_sandbox"))

    def _build_dir(self) -> Path:
        """Default build output dir (env override > in-package native/build)."""
        override = os.environ.get("UCM_TOOLKIT_DEV_SANDBOX_BUILD_DIR")
        if override:
            return Path(override)
        return self._source_dir() / "build"

    def add_build_args(self, parser: argparse.ArgumentParser) -> None:
        """Register dev-sandbox build arguments."""
        parser.add_argument("--build-type", default="Release", help="CMake build type")
        parser.add_argument(
            "--jobs", "-j", type=int, default=None, help="Build parallelism"
        )
        parser.add_argument(
            "--build-dir", default=None, help="Override dev-sandbox build directory"
        )
        parser.add_argument(
            "--cmake-arg",
            action="append",
            default=[],
            help="Extra CMake configure argument; may be repeated",
        )

    def build(self, args: argparse.Namespace) -> int:
        """Build dev-sandbox."""
        if not command_exists("cmake"):
            raise CommandNotFoundError("cmake")

        source_dir = self._source_dir()
        build_dir = Path(args.build_dir) if args.build_dir else self._build_dir()
        # Wheel installs land in a read-only site-packages; fail early with a
        # hint instead of letting cmake error out on an unwritable -B path.
        try:
            build_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            raise ToolkitError(
                f"cannot create build directory {build_dir} (read-only install? "
                f"pass --build-dir <writable path>)"
            ) from exc
        cmake_args = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={args.build_type}",
        ]
        cmake_args.extend(args.cmake_arg or [])
        check_command(cmake_args)

        build_cmd = ["cmake", "--build", str(build_dir)]
        if args.jobs:
            build_cmd.extend(["-j", str(args.jobs)])
        check_command(build_cmd)
        return 0

    def run(self, tool_args: list[str]) -> int:
        """Run a native subcommand (copy/trans/aio) or the friendly selector."""
        if not tool_args or tool_args[0] in ("-h", "--help"):
            self._print_run_help()
            return 0

        first = tool_args[0]
        if first in self.subcommands:
            # Native passthrough: copy/trans/aio + raw native args.
            return self._run_native(first, tool_args[1:])
        if first.startswith("-"):
            # Friendly mode: map model-type/iodirect/sdma to a copy case,
            # then forward the remaining args verbatim to the copy binary.
            return self._run_friendly(tool_args)
        # Unknown bare token: let _binary_path raise the "unknown subcommand" error.
        return self._run_native(first, tool_args[1:])

    def _run_native(self, subcommand: str, native_args: list[str]) -> int:
        """Locate the subcommand binary and forward native args verbatim."""
        binary = self._binary_path(subcommand)
        return run_command([str(binary), *native_args])

    def _run_friendly(self, tool_args: list[str]) -> int:
        """Map --model-type/--iodirect/--sdma to a copy case and forward the
        remaining args verbatim to the copy binary."""
        parser = argparse.ArgumentParser(
            prog="ucm-toolkit run dev-sandbox",
            description=(
                "Friendly mode: map model-type/iodirect/sdma to an Ascend "
                "copy case via a lookup table, then forward the rest of the "
                "args verbatim to the copy binary."
            ),
        )
        parser.add_argument("--model-type", choices=["gqa", "mla"], required=True)
        parser.add_argument(
            "--iodirect", type=_parse_bool, required=True, help="true/false"
        )
        parser.add_argument(
            "--sdma", type=_parse_bool, required=True, help="true/false"
        )
        args, passthrough = parser.parse_known_args(tool_args)

        case = COPY_CASE_MATRIX.get((args.model_type, args.iodirect, args.sdma))
        if case is None:
            raise ToolkitError(
                f"no Ascend copy case for model-type={args.model_type} "
                f"iodirect={args.iodirect} sdma={args.sdma}"
            )
        return self._run_native("copy", ["-t", case, *passthrough])

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        """Inspect dev-sandbox source/build availability."""
        source_dir = self._source_dir()
        build_dir = self._build_dir()
        ok = True
        print(f"{self.name}:")
        print(
            f"  source_dir: {source_dir} {'OK' if source_dir.exists() else 'MISSING'}"
        )
        if not source_dir.exists():
            ok = False
        print(f"  build_dir:  {build_dir} {'OK' if build_dir.exists() else 'MISSING'}")
        for name, relpath in self.subcommands.items():
            binary = build_dir / relpath
            if not binary.exists() and Path(str(binary) + ".exe").exists():
                binary = Path(str(binary) + ".exe")
            status = "OK" if binary.exists() else "MISSING"
            print(f"  {name:<5}: {binary} {status}")
        return 0 if ok else 1

    def clean(self, args: argparse.Namespace | None = None) -> int:
        """Clean dev-sandbox build artifacts."""
        build_dir = self._build_dir()
        dry_run = bool(getattr(args, "dry_run", False))
        if dry_run:
            print(f"would remove: {build_dir}")
            return 0
        if not build_dir.exists():
            print(f"{self.name}: build directory does not exist: {build_dir}")
            return 0
        shutil.rmtree(build_dir)
        print(f"removed: {build_dir}")
        return 0

    def _binary_path(self, subcommand: str) -> Path:
        try:
            relpath = self.subcommands[subcommand]
        except KeyError as exc:
            choices = ", ".join(sorted(self.subcommands))
            raise ToolkitError(
                f"unknown dev-sandbox subcommand: {subcommand}\n"
                f"available subcommands: {choices}"
            ) from exc

        build_dir = self._build_dir()
        if not build_dir.exists():
            raise BuildDirNotFoundError(str(build_dir))
        binary = build_dir / relpath
        if binary.exists():
            return binary
        exe_binary = Path(str(binary) + ".exe")
        if exe_binary.exists():
            return exe_binary
        raise BinaryNotFoundError(
            str(binary),
            "run `ucm-toolkit build dev-sandbox` first",
        )

    def _print_run_help(self) -> None:
        print("usage: ucm-toolkit run dev-sandbox SUBCOMMAND [native args...]")
        print("       ucm-toolkit run dev-sandbox --model-type {gqa|mla} \\")
        print(
            "              --iodirect {true|false} --sdma {true|false} "
            "[copy native args...]"
        )
        print()
        print("Native subcommands (detailed control over copy/trans/aio):")
        for name in sorted(self.subcommands):
            print(f"  {name}")
        print()
        print("Friendly mode (map model-type/iodirect/sdma to a copy case,")
        print("remaining args are forwarded verbatim to the copy binary):")
        print("  --model-type {gqa|mla}     required selector")
        print("  --iodirect   {true|false}  required selector")
        print("  --sdma       {true|false}  required selector")
        print(
            "  e.g. --model-type gqa --iodirect false --sdma false "
            "-s 16K -n 512 -i 128 -d 8"
        )
        print()
        print("model-type/iodirect/sdma -> Ascend copy case:")
        for (mt, iod, sdma), case in COPY_CASE_MATRIX.items():
            label = case if case else "(none)"
            print(f"  {mt:<3} iodirect={str(iod):<5} sdma={str(sdma):<5} -> {label}")
