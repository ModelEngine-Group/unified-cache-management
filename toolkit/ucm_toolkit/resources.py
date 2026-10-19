"""Locate immutable tool resources in wheels or an editable checkout."""

from __future__ import annotations

from pathlib import Path

from .errors import ScriptNotFoundError


def source_root() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    if (root / "version.ini").is_file() and (root / "toolkit/pyproject.toml").is_file():
        return root
    return None


def resource_path(name: str) -> Path:
    installed = Path(__file__).resolve().parent / "_resources" / name
    if installed.exists():
        return installed
    root = source_root()
    sources = {
        "dev-sandbox": "toolkit/src/dev-sandbox",
        "nic_monitor_pro.sh": "toolkit/src/nic_monitor/nic_monitor_pro.sh",
        "posixstore_aio_test.py": "ucm/store/test/e2e/posixstore_aio_test.py",
    }
    if root is not None and name in sources:
        source = root / sources[name]
        if source.exists():
            return source
    raise ScriptNotFoundError(str(installed))
