"""Parsing helpers for version strings, byte sizes, and SMI tool output.

Pure stdlib; imported by both the cheap checks and the (optional) bandwidth
benchmark so the parsing logic is unit-testable without any hardware.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

VersionTuple = Tuple[int, ...]

_NUM_RE = re.compile(r"\d+")
_VER_TOKEN_RE = re.compile(r"\d+(?:\.\d+)*")
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmMgGtT]i?[bB]?)?\s*$")


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


def parse_version(v: Optional[str]) -> VersionTuple:
    """Return the leading integer components of a version string.

    ``"5.15.0-25-generic"`` -> ``(5, 15, 0)``; ``"8.0"`` -> ``(8, 0)``;
    ``None``/non-numeric -> empty tuple.
    """
    if not v:
        return ()
    m = _VER_TOKEN_RE.search(v)
    if not m:
        return ()
    nums = _NUM_RE.findall(m.group(0))
    return tuple(int(n) for n in nums)


def compare_versions(a: VersionTuple, b: VersionTuple) -> int:
    """Tuple comparison padded to equal length; -1/0/1."""
    n = max(len(a), len(b))
    pa = a + (0,) * (n - len(a))
    pb = b + (0,) * (n - len(b))
    if pa < pb:
        return -1
    if pa > pb:
        return 1
    return 0


def strip_build(v: Optional[str]) -> Optional[str]:
    """Strip PEP 440 build metadata (``+local``) from a version string."""
    if not v:
        return None
    return str(v).strip().split("+", 1)[0]


def normalize_vllm_ascend_version(v: Optional[str]) -> Optional[str]:
    """Normalize a vllm-ascend version: drop ``.postN`` and ``rcN`` suffixes.

    Mirrors ``ucm/integration/vllm/patch/apply_patch.py``: ``0.18.0rc1`` ->
    ``0.18.0``; ``0.11.0.post1`` -> ``0.11.0``.
    """
    v = strip_build(v)
    if not v:
        return None
    v = v.split(".post", 1)[0]
    v = v.split("rc", 1)[0]
    return v.rstrip(".") or None


# ---------------------------------------------------------------------------
# Byte sizes
# ---------------------------------------------------------------------------

_UNIT_FACTORS = {
    "": 1,
    "k": 1024,
    "m": 1024**2,
    "g": 1024**3,
    "t": 1024**4,
}
_IUNIT_FACTORS = {
    "": 1,
    "i": 1,
    "ki": 1024,
    "mi": 1024**2,
    "gi": 1024**3,
    "ti": 1024**4,
}


def parse_size(s) -> int:
    """Parse a human byte size to an integer byte count.

    Accepts ints (``1048576``) and suffixed strings: ``180k``/``180K``,
    ``1m``/``1M``, ``2g``, ``1Ki``, ``4MiB`` (case-insensitive; trailing ``b``
    optional). All suffixed sizes are binary (1024-based). Raises
    ``ValueError`` on garbage.
    """
    if isinstance(s, (int, float)):
        return int(s)
    if not isinstance(s, str):
        raise ValueError(f"cannot parse size from {s!r}")
    m = _SIZE_RE.match(s)
    if not m:
        raise ValueError(f"unrecognized size {s!r}")
    num = float(m.group(1))
    unit = m.group(2) or ""
    # Strip an optional trailing b/B (the "bytes" marker) so "4MiB"/"4MB" work.
    if unit and unit[-1] in "bB":
        unit = unit[:-1]
    unit = unit.lower()
    factor = _IUNIT_FACTORS.get(unit) or _UNIT_FACTORS.get(unit)
    if factor is None:
        raise ValueError(f"unrecognized size unit {unit!r} in {s!r}")
    return int(num * factor)


def parse_size_list(s: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated list of sizes (``"180k,1m"``); None if unset."""
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [parse_size(p) for p in parts]


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated list of ints (``"1,16"``); None if unset."""
    if s is None:
        return None
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def parse_str_list(s: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated list of strings; None if unset."""
    if s is None:
        return None
    return [p.strip() for p in s.split(",") if p.strip()]


# ---------------------------------------------------------------------------
# nvidia-smi / npu-smi parsing
# ---------------------------------------------------------------------------


def parse_nvidia_smi_csv(stdout: str) -> List[List[str]]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader`` rows.

    Returns each non-empty line as a list of stripped cells. The pre-check
    invocations always pass ``noheader``, so no header is expected; if a header
    is present anyway it is harmless (``extract_nvidia_compute_cap`` only picks
    cells matching ``\\d+.\\d+``).
    """
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    return [[c.strip() for c in ln.split(",")] for ln in lines]


def extract_nvidia_compute_cap(rows: List[dict], cap_index: int) -> Optional[float]:
    """Best-effort compute-capability extraction from parsed CSV rows.

    ``rows`` are the lists returned by :func:`parse_nvidia_smi_csv`;
    ``cap_index`` is the 0-based position of the compute_cap column.
    """
    for cells in rows:
        if cap_index < len(cells):
            m = re.search(r"\d+\.\d+", cells[cap_index])
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    continue
    return None


_NPU_ANY_RE = re.compile(r"(\d+\.\d+\.\d+)")
_NPU_HDK_RE = re.compile(r"hdk[^0-9]*(\d+\.\d+\.\d+)", re.IGNORECASE)
_NPU_DRIVER_RE = re.compile(r"driver[^0-9]*(\d+\.\d+\.\d+)", re.IGNORECASE)
_NPU_VERSION_RE = re.compile(r"version[^0-9]*(\d+\.\d+\.\d+)", re.IGNORECASE)
_NPU_TOOL_RE = re.compile(r"npu-smi[^0-9]*(\d+\.\d+\.\d+)", re.IGNORECASE)


def parse_npu_smi_versions(stdout: str) -> dict:
    """Heuristically extract Ascend HDK / driver / tool versions from
    ``npu-smi info`` output.

    The field layout varies across firmware. Resolution order for the HDK
    value (the one compared against the floor):

    1. an explicit ``HDK Version`` label;
    2. a ``Driver Version`` label (HDK and driver track together);
    3. the banner ``Version: X.Y.Z`` field (on firmware that exposes no
       per-card version label — e.g. ``npu-smi 25.5.2   Version: 25.5.2`` —
       this banner ``Version`` IS the firmware/driver version).

    The ``npu-smi X.Y.Z`` banner *prefix* is captured separately as ``tool``
    (the SMI tool build version) and is deliberately NOT used as the HDK on its
    own, since on some firmware it differs from the driver. Returns a dict with
    keys ``hdk``, ``driver``, ``version`` (banner label), ``tool`` (each
    possibly None) and ``raw_versions`` (all ``\\d+.\\d+.\\d+`` tokens, order).
    """
    result = {
        "hdk": None,
        "driver": None,
        "version": None,
        "tool": None,
        "raw_versions": [],
    }
    if not stdout:
        return result

    for line in stdout.splitlines():
        m = _NPU_ANY_RE.search(line)
        if m:
            result["raw_versions"].append(m.group(1))

    m = _NPU_HDK_RE.search(stdout)
    if m:
        result["hdk"] = m.group(1)
    m = _NPU_DRIVER_RE.search(stdout)
    if m:
        result["driver"] = m.group(1)
    m = _NPU_VERSION_RE.search(stdout)
    if m:
        result["version"] = m.group(1)
    m = _NPU_TOOL_RE.search(stdout)
    if m:
        result["tool"] = m.group(1)

    # HDK proxy chain: explicit HDK label -> driver label -> banner "Version:".
    if result["hdk"] is None:
        result["hdk"] = result["driver"] or result["version"]
    return result
