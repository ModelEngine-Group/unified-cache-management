"""Tool registry and adapter interfaces for ucm-toolkit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

from .errors import ToolNotBuildableError, UnknownToolError


class ToolAdapter:
    """Base interface for toolkit tools."""

    name: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()
    description: ClassVar[str] = ""
    buildable: ClassVar[bool] = False

    source_dir: ClassVar[str | None] = None
    build_dir: ClassVar[str | None] = None
    binary_relpath: ClassVar[str | None] = None
    script_path: ClassVar[str | None] = None
    subcommands: ClassVar[dict[str, str]] = {}

    def add_build_args(self, parser: argparse.ArgumentParser) -> None:
        """Register build-specific CLI arguments."""

    def build(self, args: argparse.Namespace) -> int:
        """Build this tool."""
        raise ToolNotBuildableError(self.name)

    def add_run_args(self, parser: argparse.ArgumentParser) -> None:
        """Register run-specific CLI arguments."""

    def run(self, tool_args: list[str]) -> int:
        """Run this tool with raw tool arguments."""
        raise NotImplementedError

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        """Inspect tool availability and configuration."""
        raise NotImplementedError

    def clean(self, args: argparse.Namespace | None = None) -> int:
        """Clean tool-generated artifacts."""
        print(f"{self.name}: nothing to clean")
        return 0


_TOOLS: dict[str, ToolAdapter] = {}
_ALIASES: dict[str, str] = {}


def register(tool: ToolAdapter) -> None:
    """Register a tool and its aliases."""
    if tool.name in _TOOLS:
        raise RegistryUpdateError(f"duplicate tool registration: {tool.name}")
    _TOOLS[tool.name] = tool
    for alias in tool.aliases:
        if alias in _ALIASES:
            raise RegistryUpdateError(f"duplicate tool alias: {alias}")
        _ALIASES[alias] = tool.name


def get(name: str) -> ToolAdapter:
    """Return a registered tool by name or alias."""
    canonical = _ALIASES.get(name, name)
    try:
        return _TOOLS[canonical]
    except KeyError as exc:
        raise UnknownToolError(name) from exc


def list_tools() -> list[ToolAdapter]:
    """Return registered top-level tools."""
    return [_TOOLS[name] for name in sorted(_TOOLS)]


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def init_builtin_tools() -> None:
    """Register built-in top-level toolkit tools."""
    if _TOOLS:
        return
    from .tools.dev_sandbox import DevSandboxTool
    from .tools.metrics_view import MetricsViewTool
    from .tools.nic_monitor import NicMonitorTool
    from .tools.posix_aio import PosixAioTool
    from .tools.precheck import PrecheckTool

    register(DevSandboxTool())
    register(MetricsViewTool())
    register(PosixAioTool())
    register(NicMonitorTool())
    register(PrecheckTool())
