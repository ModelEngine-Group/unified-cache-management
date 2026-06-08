"""Shared exception types for ucm-toolkit."""

from __future__ import annotations


class ToolkitError(Exception):
    """Base class for expected toolkit errors."""

    exit_code: int = 1

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class UnknownToolError(ToolkitError):
    """Raised when a tool name or alias is not registered."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"unknown tool: {tool_name}")


class ToolNotBuildableError(ToolkitError):
    """Raised when build is requested for a non-buildable tool."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"{tool_name} is not buildable")


class BuildDirNotFoundError(ToolkitError):
    """Raised when a configured build directory cannot be found."""

    def __init__(self, build_dir: str) -> None:
        super().__init__(f"build directory not found: {build_dir}")


class BinaryNotFoundError(ToolkitError):
    """Raised when a tool binary cannot be found."""

    def __init__(self, binary_path: str, hint: str | None = None) -> None:
        message = f"binary not found: {binary_path}"
        if hint:
            message = f"{message}\nhint: {hint}"
        super().__init__(message)


class ScriptNotFoundError(ToolkitError):
    """Raised when a script-backed tool cannot locate its script."""

    def __init__(self, script_path: str) -> None:
        super().__init__(f"script not found: {script_path}")


class CommandNotFoundError(ToolkitError):
    """Raised when an external command is not available."""

    def __init__(self, command: str) -> None:
        super().__init__(f"command not found: {command}")


class CommandFailedError(ToolkitError):
    """Raised when an external command exits unsuccessfully."""

    def __init__(self, command: str, returncode: int) -> None:
        super().__init__(
            f"command failed with exit code {returncode}: {command}",
            exit_code=returncode or 1,
        )


class RegistryUpdateError(ToolkitError):
    """Raised when a controlled registry/tool field update fails."""
