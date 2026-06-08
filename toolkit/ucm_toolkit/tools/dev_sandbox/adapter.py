"""dev-sandbox adapter interfaces."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ... import registry
from ...errors import BinaryNotFoundError, BuildDirNotFoundError, CommandNotFoundError, ToolkitError
from ...registry import ToolAdapter
from ...runner import check_command, command_exists, run_command


class DevSandboxTool(ToolAdapter):
    """Build adapter for toolkit/src/dev-sandbox."""

    name = "dev-sandbox"
    aliases = ("dev_sandbox",)
    description = "Build the CMake-based dev-sandbox test project."
    buildable = True
    source_dir = "toolkit/src/dev-sandbox"
    build_dir = "toolkit/src/dev-sandbox/build"
    subcommands = {
        "copy": "module/copy/copy",
        "trans": "module/trans/trans",
        "aio": "module/aio/aio",
    }

    def add_build_args(self, parser: argparse.ArgumentParser) -> None:
        """Register dev-sandbox build arguments."""
        parser.add_argument("--build-type", default="Release", help="CMake build type")
        parser.add_argument("--jobs", "-j", type=int, default=None, help="Build parallelism")
        parser.add_argument("--build-dir", default=None, help="Override dev-sandbox build directory")
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

        source_dir = registry.resolve_repo_path(self.source_dir or "")
        build_dir_value = args.build_dir or self.build_dir
        build_dir = registry.resolve_repo_path(build_dir_value or "")
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

        if args.build_dir:
            registry.update_tool_field(self.name, "build_dir", args.build_dir)
        return 0

    def run(self, tool_args: list[str]) -> int:
        """Run a dev-sandbox subcommand with raw native arguments."""
        if not tool_args or tool_args[0] in ("-h", "--help"):
            self._print_run_help()
            return 0

        subcommand = tool_args[0]
        native_args = tool_args[1:]
        binary = self._binary_path(subcommand)
        return run_command([str(binary), *native_args])

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        """Inspect dev-sandbox source/build availability."""
        source_dir = registry.resolve_repo_path(self.source_dir or "")
        build_dir = registry.resolve_repo_path(self.build_dir or "")
        ok = True
        print(f"{self.name}:")
        print(f"  source_dir: {source_dir} {'OK' if source_dir.exists() else 'MISSING'}")
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
        build_dir = registry.resolve_repo_path(self.build_dir or "")
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

        build_dir = registry.resolve_repo_path(self.build_dir or "")
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
        print()
        print("Available subcommands:")
        for name in sorted(self.subcommands):
            print(f"  {name}")
