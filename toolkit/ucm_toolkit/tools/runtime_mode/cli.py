"""runtime-mode command-line interface (online UCM mode control).

Reads and writes the runtime-mode control files polled by UCM-enabled
engines, so operators can switch between ``enabled`` / ``lite`` /
``disabled`` without restarting the service. Mounted as
``ucm-toolkit run runtime-mode``.

The file format and path resolution mirror ``ucm/runtime_mode.py``: JSON
``{"mode": "lite"}`` (or a bare mode string), path precedence
``--control-file`` > ``UCM_RUNTIME_CONTROL_FILE`` environment variable >
``<tempdir>/ucm_runtime_control_<engine_id>.json``. Writes are atomic
(temporary file + ``os.replace``) so a poller can never observe a
half-written file.

Examples::

    ucm-toolkit run runtime-mode set disabled
    ucm-toolkit run runtime-mode set lite --engine-id engine-0
    ucm-toolkit run runtime-mode set enabled --engine-id engine-0_dp3
    ucm-toolkit run runtime-mode get --engine-id engine-0
    ucm-toolkit run runtime-mode list
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import threading
from typing import List, Optional

MODES = ("enabled", "lite", "disabled")

CONTROL_FILE_PATTERN = "ucm_runtime_control_*.json"
ENV_CONTROL_FILE = "UCM_RUNTIME_CONTROL_FILE"

MODE_ALIASES = {
    "enabled": "enabled",
    "enable": "enabled",
    "on": "enabled",
    "true": "enabled",
    "1": "enabled",
    "full": "enabled",
    "lite": "lite",
    "shadow": "lite",
    "trace": "lite",
    "disabled": "disabled",
    "disable": "disabled",
    "off": "disabled",
    "false": "disabled",
    "0": "disabled",
    "bypass": "disabled",
}

MODE_DESCRIPTIONS = {
    "enabled": "full behavior: external lookup + KV load/dump",
    "lite": "shadow: record as-if hit rates, no KV load/dump",
    "disabled": "bypass: answer misses immediately, no load/dump",
}


def parse_mode(value: object) -> Optional[str]:
    """Return the canonical mode for ``value`` or ``None``."""
    if not isinstance(value, str):
        return None
    return MODE_ALIASES.get(value.strip().lower())


def parse_control_content(content: str) -> Optional[str]:
    """Parse control-file content (bare string or JSON ``{"mode": ...}``)."""
    text = content.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        return parse_mode(payload.get("mode"))
    if isinstance(payload, str):
        return parse_mode(payload)
    return parse_mode(text)


def normalize_engine_id(engine_id: str) -> str:
    """Strip the data-parallel suffix the same way the connector does."""
    return engine_id.rsplit("_dp", 1)[0]


def default_control_file(engine_id: str = "") -> str:
    """Default per-engine control-file path in the system temp directory."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(engine_id or "").strip())
    safe = safe or "ucm"
    return os.path.join(tempfile.gettempdir(), f"ucm_runtime_control_{safe}.json")


def resolve_control_file(explicit: Optional[str], engine_id: str = "") -> str:
    """Resolve the control-file path (explicit > env var > default)."""
    if explicit:
        return explicit
    env_path = os.getenv(ENV_CONTROL_FILE)
    if env_path and env_path.strip():
        return env_path.strip()
    return default_control_file(normalize_engine_id(engine_id))


def read_mode(path: str) -> Optional[str]:
    """Read and parse a control file; ``None`` when missing or invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return None
    return parse_control_content(content)


def write_mode(path: str, mode: str) -> None:
    """Atomically write ``{"mode": ...}`` to ``path``."""
    payload = json.dumps({"mode": mode})
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(payload)
    os.replace(tmp_path, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="runtime-mode",
        description=(
            "Switch the online runtime mode of UCM-enabled engines "
            "(enabled | lite | disabled) without restarting them."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes:\n"
            "  enabled   full behavior: external lookup + KV load/dump\n"
            "  lite      shadow: record as-if hit rates, no KV load/dump\n"
            "  disabled  bypass: answer misses immediately, no load/dump\n"
            "\n"
            "The engine picks up changes within its poll interval "
            "(default 1s).\n"
            "engine_id matches the vLLM kv_transfer_config engine_id; a "
            "trailing _dpN suffix is ignored."
        ),
    )
    sub = p.add_subparsers(dest="command")

    p_set = sub.add_parser(
        "set", help="write a mode to an engine's control file (atomic)"
    )
    p_set.add_argument("mode", help="enabled | lite | disabled (aliases allowed)")
    p_set.add_argument("--engine-id", default="", help="engine id (default: none)")
    p_set.add_argument(
        "--control-file",
        metavar="PATH",
        help="explicit control file (default: resolve from engine id / env)",
    )

    p_get = sub.add_parser("get", help="read the current mode from a control file")
    p_get.add_argument("--engine-id", default="", help="engine id (default: none)")
    p_get.add_argument(
        "--control-file",
        metavar="PATH",
        help="explicit control file (default: resolve from engine id / env)",
    )

    p_list = sub.add_parser(
        "list", help="discover control files in the temp directory and their modes"
    )

    return p


def cmd_set(args: argparse.Namespace) -> int:
    mode = parse_mode(args.mode)
    if mode is None:
        print(
            f"error: invalid mode {args.mode!r}; choices: {', '.join(MODES)}",
            file=sys.stderr,
        )
        return 2
    path = resolve_control_file(args.control_file, args.engine_id)
    old_mode = read_mode(path)
    try:
        write_mode(path, mode)
    except OSError as exc:
        print(f"error: cannot write {path}: {exc}", file=sys.stderr)
        return 1
    print(f"file:     {path}")
    print(f"old mode: {old_mode if old_mode else '-'}")
    print(f"new mode: {mode}")
    print("note:     engines observe the change within their poll interval")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    path = resolve_control_file(args.control_file, args.engine_id)
    if not os.path.exists(path):
        print(f"error: control file not found: {path}", file=sys.stderr)
        return 1
    mode = read_mode(path)
    if mode is None:
        print(
            f"error: control file is unreadable or has no valid mode: {path}",
            file=sys.stderr,
        )
        return 1
    print(f"file: {path}")
    print(f"mode: {mode}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    directory = tempfile.gettempdir()
    print(f"control files in {directory}:")
    found = 0
    try:
        names = sorted(os.listdir(directory))
    except OSError as exc:
        print(f"error: cannot list {directory}: {exc}", file=sys.stderr)
        return 1
    for name in names:
        if not re.match(r"^ucm_runtime_control_.*\.json$", name):
            continue
        found += 1
        path = os.path.join(directory, name)
        mode = read_mode(path)
        state = mode if mode else "(unreadable)"
        print(f"  {name:<50} {state}")
    if not found:
        print("  (none; a control file appears once a mode is first set, or")
        print("   when the engine was configured with runtime_control_file)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "set":
        return cmd_set(args)
    if args.command == "get":
        return cmd_get(args)
    if args.command == "list":
        return cmd_list(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
