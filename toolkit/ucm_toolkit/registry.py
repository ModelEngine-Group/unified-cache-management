"""Tool registry and adapter interfaces for ucm-toolkit."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import ClassVar

from .errors import RegistryUpdateError, ToolNotBuildableError, UnknownToolError


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


def update_tool_field(tool_name: str, field_name: str, value: str) -> None:
    """Persistently update an approved string field on a registered tool."""
    if field_name != "build_dir":
        raise RegistryUpdateError(f"field cannot be updated: {field_name}")
    if not isinstance(value, str) or "\n" in value or "\r" in value:
        raise RegistryUpdateError("field value must be a single-line string")

    tool = get(tool_name)
    if not hasattr(tool.__class__, field_name):
        raise RegistryUpdateError(f"{tool.name} has no field: {field_name}")

    module = __import__(tool.__class__.__module__, fromlist=["__file__"])
    source_file = Path(module.__file__ or "")
    if not source_file.exists():
        raise RegistryUpdateError(f"cannot locate source file for {tool.name}")

    text = source_file.read_text(encoding="utf-8")
    class_header = (
        rf"class\s+{re.escape(tool.__class__.__name__)}\b[\s\S]*?(?=^class\s|\Z)"
    )
    match = re.search(class_header, text, flags=re.MULTILINE)
    if not match:
        raise RegistryUpdateError(f"cannot find class {tool.__class__.__name__}")

    class_text = match.group(0)
    field_re = re.compile(
        rf"^(\s*){re.escape(field_name)}\s*=\s*(['\"])(.*?)\2", re.MULTILINE
    )
    field_match = field_re.search(class_text)
    if not field_match:
        raise RegistryUpdateError(f"cannot find field {field_name}")

    quote = field_match.group(2)
    escaped = value.replace("\\", "\\\\").replace(quote, "\\" + quote)
    new_class_text = (
        class_text[: field_match.start()]
        + f"{field_match.group(1)}{field_name} = {quote}{escaped}{quote}"
        + class_text[field_match.end() :]
    )
    new_text = text[: match.start()] + new_class_text + text[match.end() :]
    source_file.write_text(new_text, encoding="utf-8")
    setattr(tool.__class__, field_name, value)


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a repository-relative or absolute path."""
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root() / path


def init_builtin_tools() -> None:
    """Register built-in top-level toolkit tools."""
    if _TOOLS:
        return
    from .tools.dev_sandbox import DevSandboxTool
    from .tools.metrics_view import MetricsViewTool
    from .tools.model_check import ModelCheckTool
    from .tools.nic_monitor import NicMonitorTool
    from .tools.posix_aio import PosixAioTool
    from .tools.precheck import PrecheckTool

    register(DevSandboxTool())
    register(MetricsViewTool())
    register(ModelCheckTool())
    register(PosixAioTool())
    register(NicMonitorTool())
    register(PrecheckTool())
