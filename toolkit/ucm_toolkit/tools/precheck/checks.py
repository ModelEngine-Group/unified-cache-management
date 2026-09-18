"""Concrete environment checks. Each function returns a :class:`CheckResult`.

The cheap checks (versions, driver, kernel, memory) are pure stdlib and run on
any host. The bandwidth check lives in :mod:`pre_check.bandwidth` because it
pulls in numpy and the ucm C++ store.
"""

from __future__ import annotations

import importlib
import os
import platform
import re
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import List, Optional, Tuple

from .config import PrecheckConfig
from .parseutil import (
    compare_versions,
    extract_nvidia_compute_cap,
    normalize_vllm_ascend_version,
    parse_npu_smi_versions,
    parse_nvidia_smi_csv,
    parse_version,
    strip_build,
)
from .reporter import (
    FAIL,
    INFO,
    STATUS_FAIL,
    STATUS_INFO,
    STATUS_OK,
    STATUS_PASS,
    STATUS_SKIP,
    STATUS_WARN,
    WARN,
    CheckResult,
)


def _run(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    """Run a command, returning (returncode, stdout, stderr). Never raises."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout or "", r.stderr or ""
    except (FileNotFoundError, OSError):
        return -1, "", f"command not found: {' '.join(cmd)}"
    except subprocess.TimeoutExpired:
        return -1, "", f"command timed out after {timeout}s: {' '.join(cmd)}"


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


# ---------------------------------------------------------------------------
# 1. Serving stack (vllm / vllm-ascend / sglang) — INFO
# ---------------------------------------------------------------------------


def _dist_version(dist: str) -> Optional[str]:
    try:
        return pkg_version(dist)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _module_version(module: str) -> Optional[str]:
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    return getattr(mod, "__version__", None)


def check_serving_stack() -> CheckResult:
    """Print installed vllm/vllm-ascend (or sglang) versions. Never fails."""
    parts: List[str] = []
    raw: dict = {}

    vllm = _dist_version("vllm") or _module_version("vllm")
    if vllm:
        parts.append(f"vllm={strip_build(vllm)}")
        raw["vllm"] = vllm

    ascend_raw = _dist_version("vllm-ascend") or _module_version("vllm_ascend")
    ascend_norm = normalize_vllm_ascend_version(ascend_raw)
    if ascend_raw:
        parts.append(f"vllm-ascend={ascend_raw}")
        raw["vllm_ascend_raw"] = ascend_raw
        raw["vllm_ascend_norm"] = ascend_norm

    sglang = _dist_version("sglang") or _module_version("sglang")
    if sglang:
        parts.append(f"sglang={sglang}")
        raw["sglang"] = sglang

    if not parts:
        return CheckResult(
            name="serving_stack",
            severity=INFO,
            status=STATUS_INFO,
            value="-",
            detail="no vllm / vllm-ascend / sglang detected",
            raw=raw,
        )
    return CheckResult(
        name="serving_stack",
        severity=INFO,
        status=STATUS_INFO,
        value=", ".join(parts),
        detail="installed serving stack",
        raw=raw,
    )


# ---------------------------------------------------------------------------
# 2. uc-manager — INFO
# ---------------------------------------------------------------------------


def check_uc_manager_version() -> CheckResult:
    """Print installed uc-manager version. Never fails."""
    raw: dict = {}
    ver = _dist_version("uc-manager") or _dist_version("uc_manager")
    if ver is None:
        ver = _module_version("uc_manager")
    if ver is None:
        # Last resort: pip show (covers non-importable installs).
        rc, out, _ = _run([sys_python(), "-m", "pip", "show", "uc-manager"])
        if rc == 0:
            m = re.search(r"^Version:\s*(\S+)", out, re.M)
            if m:
                ver = m.group(1)
    if ver:
        raw["version"] = ver
        return CheckResult(
            name="uc_manager",
            severity=INFO,
            status=STATUS_INFO,
            value=ver,
            detail="uc-manager package version",
            raw=raw,
        )
    return CheckResult(
        name="uc_manager",
        severity=INFO,
        status=STATUS_INFO,
        value="-",
        detail="uc-manager not installed",
        raw=raw,
    )


def sys_python() -> str:
    return os.environ.get("PRECHECK_PYTHON") or "python3"


# ---------------------------------------------------------------------------
# 3. Accelerator driver — WARN
# ---------------------------------------------------------------------------


def check_accelerator_driver(cfg: PrecheckConfig) -> CheckResult:
    """Check NVIDIA compute capability or Ascend HDK version.

    Auto-detects the platform: prefers ``nvidia-smi`` when present, else falls
    back to ``npu-smi info``. With neither present the check FAILs (UCM needs an
    accelerator). CUDA warns if any GPU's compute capability is below the
    configured floor (default 8.0); Ascend warns if the HDK version is not
    strictly newer than the configured floor (25.2.0).
    """
    if _have("nvidia-smi"):
        return _check_nvidia(cfg)
    if _have("npu-smi"):
        return _check_ascend(cfg)
    # UCM requires a GPU or NPU; no accelerator driver is a hard failure, not a
    # benign skip (otherwise a CPU-only host would report an overall PASS).
    return CheckResult(
        name="accelerator_driver",
        severity=FAIL,
        status=STATUS_FAIL,
        value="-",
        detail="no nvidia-smi or npu-smi found (no GPU/NPU driver detected)",
        remediation="install the NVIDIA or Ascend driver so nvidia-smi/npu-smi "
        "is available on PATH",
        raw={"platform": "none"},
    )


def _check_nvidia(cfg: PrecheckConfig) -> CheckResult:
    rc, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0:
        return CheckResult(
            name="accelerator_driver",
            severity=WARN,
            status=STATUS_WARN,
            value="-",
            threshold=f"compute_cap >= {cfg.cuda_min_compute_cap}",
            detail=f"nvidia-smi failed: {err.strip() or 'rc=%d' % rc}",
            remediation="ensure the NVIDIA driver is loaded and nvidia-smi is on PATH",
            raw={"platform": "nvidia", "rc": rc, "stderr": err},
        )
    rows = parse_nvidia_smi_csv(out)
    # columns: name(0), driver_version(1), compute_cap(2)
    caps: List[float] = []
    driver_versions: List[str] = []
    names: List[str] = []
    for cells in rows:
        if len(cells) >= 1:
            names.append(cells[0])
        if len(cells) >= 2:
            driver_versions.append(cells[1])
    cap = extract_nvidia_compute_cap(rows, 2) if rows else None
    # Re-scan every row so multi-GPU min-cap is reported, not just the first.
    for cells in rows:
        if len(cells) >= 3:
            m = re.search(r"\d+\.\d+", cells[2])
            if m:
                caps.append(float(m.group(0)))
    min_cap = min(caps) if caps else None

    detail = (
        f"driver={driver_versions[0] if driver_versions else '-'}, "
        f"{len(names)} GPU(s): {', '.join(names) if names else '-'}"
    )
    value = f"compute_cap={min_cap}" if min_cap is not None else "compute_cap=?"

    floor = cfg.cuda_min_compute_cap
    if min_cap is None:
        return CheckResult(
            name="accelerator_driver",
            severity=WARN,
            status=STATUS_WARN,
            value=value,
            threshold=f"compute_cap >= {floor}",
            detail=detail + " (could not parse compute capability)",
            remediation="ensure nvidia-smi reports a compute_cap (e.g. 8.0)",
            raw={"platform": "nvidia", "rows": rows},
        )
    if min_cap < floor:
        return CheckResult(
            name="accelerator_driver",
            severity=WARN,
            status=STATUS_WARN,
            value=value,
            threshold=f"compute_cap >= {floor}",
            detail=detail + f" (below floor {floor})",
            remediation=(
                f"use a GPU with compute capability >= {floor} " "(Ampere or newer)"
            ),
            raw={"platform": "nvidia", "min_cap": min_cap, "rows": rows},
        )
    return CheckResult(
        name="accelerator_driver",
        severity=WARN,
        status=STATUS_PASS,
        value=value,
        threshold=f"compute_cap >= {floor}",
        detail=detail,
        raw={"platform": "nvidia", "min_cap": min_cap, "rows": rows},
    )


def _check_ascend(cfg: PrecheckConfig) -> CheckResult:
    rc, out, err = _run(["npu-smi", "info"])
    if rc != 0:
        return CheckResult(
            name="accelerator_driver",
            severity=WARN,
            status=STATUS_WARN,
            value="-",
            threshold=f"HDK > {cfg.ascend_min_hdk}",
            detail=f"npu-smi info failed: {err.strip() or 'rc=%d' % rc}",
            remediation="ensure the Ascend driver is loaded and npu-smi is on PATH",
            raw={"platform": "ascend", "rc": rc, "stderr": err},
        )
    parsed = parse_npu_smi_versions(out)
    hdk = parsed.get("hdk")
    floor = parse_version(cfg.ascend_min_hdk)
    value = f"HDK={hdk or '?'}"
    raw = {"platform": "ascend", **parsed}

    if hdk is None:
        return CheckResult(
            name="accelerator_driver",
            severity=WARN,
            status=STATUS_WARN,
            value=value,
            threshold=f"HDK > {cfg.ascend_min_hdk}",
            detail=(
                "could not extract an HDK/driver version; heuristic parse "
                f"found: {parsed['raw_versions'] or 'none'}"
            ),
            remediation="upgrade npu-smi to a version that reports an HDK/driver label",
            raw=raw,
        )
    if compare_versions(parse_version(hdk), floor) <= 0:
        return CheckResult(
            name="accelerator_driver",
            severity=WARN,
            status=STATUS_WARN,
            value=value,
            threshold=f"HDK > {cfg.ascend_min_hdk}",
            detail=f"HDK {hdk} is not strictly newer than {cfg.ascend_min_hdk}",
            remediation=(
                f"upgrade the Ascend HDK/driver to a version newer "
                f"than {cfg.ascend_min_hdk}"
            ),
            raw=raw,
        )
    return CheckResult(
        name="accelerator_driver",
        severity=WARN,
        status=STATUS_PASS,
        value=value,
        threshold=f"HDK > {cfg.ascend_min_hdk}",
        detail=f"Ascend HDK/driver version OK",
        raw=raw,
    )


# ---------------------------------------------------------------------------
# 4. Kernel version — FAIL (hard)
# ---------------------------------------------------------------------------


def check_kernel_version(cfg: PrecheckConfig) -> CheckResult:
    """Require the running kernel's major.minor to be at least the floor.

    The floor ``5.10`` means "the 5.10 LTS series or newer": ``5.10.0-216`` (a
    backported openEuler 5.10 LTS build) passes, ``5.11`` passes, ``5.9`` fails.
    This matches real UCM deployments (openEuler 22.03 ships a 5.10.0-<build>
    kernel). The floor is configurable via ``--kernel-min``.
    """
    release = None
    src = None
    rc, out, _ = _run(["uname", "-r"])
    if rc == 0 and out.strip():
        release = out.strip()
        src = "uname -r"
    if not release:
        release = platform.release()
        src = "platform.release()"

    kv = parse_version(release)
    floor = parse_version(cfg.kernel_min)
    if not kv:
        return CheckResult(
            name="kernel",
            severity=FAIL,
            status=STATUS_FAIL,
            value=release or "-",
            threshold=f">= {cfg.kernel_min}",
            detail=f"could not parse kernel release ({src}={release!r})",
            remediation="ensure uname -r reports a parseable version (e.g. 5.10.x)",
            raw={"release": release, "source": src},
        )
    # Compare on major.minor so 5.10.0-<build> is "5.10", not below it.
    kv_mm = kv[:2] if len(kv) >= 2 else kv + (0,) * (2 - len(kv))
    floor_mm = floor[:2] if len(floor) >= 2 else floor + (0,) * (2 - len(floor))
    value = ".".join(str(p) for p in kv)
    if compare_versions(kv_mm, floor_mm) >= 0:
        return CheckResult(
            name="kernel",
            severity=FAIL,
            status=STATUS_PASS,
            value=value,
            threshold=f">= {cfg.kernel_min}",
            detail=f"kernel {release} ({src})",
            raw={"release": release},
        )
    return CheckResult(
        name="kernel",
        severity=FAIL,
        status=STATUS_FAIL,
        value=value,
        threshold=f">= {cfg.kernel_min}",
        detail=f"kernel {release} ({'.'.join(map(str, kv_mm))}) is older than "
        f"{cfg.kernel_min}",
        remediation=(
            f"upgrade to kernel {cfg.kernel_min} or newer "
            f"(e.g. openEuler 22.03 LTS-SP4 ships a 5.10 LTS kernel)"
        ),
        raw={"release": release},
    )


# ---------------------------------------------------------------------------
# 5. /dev/shm size — WARN (UCM KV-cache page cache needs a large shm)
# ---------------------------------------------------------------------------


def _shm_total_avail() -> Tuple[Optional[int], Optional[int]]:
    """Return (total_bytes, available_bytes) for /dev/shm via statvfs.

    ``os.statvfs`` is POSIX-only, so this returns ``(None, None)`` on platforms
    that lack it (or where /dev/shm is absent).
    """
    if not hasattr(os, "statvfs"):
        return None, None
    try:
        st = os.statvfs("/dev/shm")
        return st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize
    except OSError:
        return None, None


def _gib(bytes_: Optional[int]) -> str:
    return f"{bytes_ / (1024 ** 3):.2f} GiB" if bytes_ is not None else "-"


def check_memory_shm(cfg: PrecheckConfig) -> CheckResult:
    """Check /dev/shm size; warn if below the UCM minimum (RAM is hidden)."""
    shm_min = cfg.shm_min_gib
    shm_total, shm_avail = _shm_total_avail()
    threshold = f">= {shm_min:.0f} GiB"
    raw = {"shm_total_bytes": shm_total, "shm_avail_bytes": shm_avail}
    if shm_total is None:
        return CheckResult(
            name="memory_shm",
            severity=WARN,
            status=STATUS_WARN,
            value="-",
            threshold=threshold,
            detail="could not determine /dev/shm size",
            remediation="ensure /dev/shm is mounted and readable",
            raw=raw,
        )
    shm_gib = shm_total / (1024**3)
    value = f"shm={shm_gib:.2f} GiB ({_gib(shm_avail)} free)"
    if shm_gib < shm_min:
        return CheckResult(
            name="memory_shm",
            severity=WARN,
            status=STATUS_WARN,
            value=value,
            threshold=threshold,
            detail=(
                f"/dev/shm {shm_gib:.1f} GiB is below the "
                f"{shm_min:.0f} GiB UCM minimum"
            ),
            remediation=(
                f"enlarge /dev/shm to >= {shm_min:.0f} GiB "
                "(UCM KV-cache page cache needs it)"
            ),
            raw=raw,
        )
    return CheckResult(
        name="memory_shm",
        severity=WARN,
        status=STATUS_PASS,
        value=value,
        threshold=threshold,
        detail="/dev/shm size OK",
        raw=raw,
    )


# ---------------------------------------------------------------------------
# AIO resource pool (INFO: display only)
# ---------------------------------------------------------------------------


def check_aio_resources(cfg: PrecheckConfig) -> CheckResult:
    """Check kernel AIO resource pool (INFO — display only, never fails).

    Each aio store context calls ``io_setup(queueDepth)`` which reserves
    ``queueDepth`` events from the kernel's aio pool (``aio-max-nr``).
    The maximum number of concurrent aio workers is::

        max_aio_workers = (aio_max_nr - aio_nr) // queueDepth

    This helps the operator decide how many workers to use for the aio
    bandwidth engine and whether to raise ``/proc/sys/fs/aio-max-nr``.
    """
    try:
        with open("/proc/sys/fs/aio-max-nr") as f:
            aio_max_nr = int(f.read().strip())
        with open("/proc/sys/fs/aio-nr") as f:
            aio_nr = int(f.read().strip())
    except (OSError, ValueError):
        return CheckResult(
            name="aio_resources",
            severity=INFO,
            status=STATUS_INFO,
            value="-",
            detail="aio kernel resources not available (non-Linux or aio disabled)",
            raw={},
        )
    available = aio_max_nr - aio_nr
    qd = cfg.bandwidth.aio_queue_depth
    max_workers = available // qd if qd > 0 else 0
    return CheckResult(
        name="aio_resources",
        severity=INFO,
        status=STATUS_INFO,
        value=f"max-nr={aio_max_nr} nr={aio_nr} available={available} "
        f"max_aio_workers={max_workers}",
        detail=f"each aio context needs {qd} events (queueDepth); "
        f"{available} available → {max_workers} concurrent aio workers",
        raw={
            "aio_max_nr": aio_max_nr,
            "aio_nr": aio_nr,
            "available": available,
            "aio_queue_depth": qd,
            "max_aio_workers": max_workers,
        },
    )


# ---------------------------------------------------------------------------
# Registry of cheap checks (bandwidth added by the CLI orchestrator)
# ---------------------------------------------------------------------------

CHECK_NAMES = {
    "serving_stack": "Serving stack (vllm/vllm-ascend/sglang) version",
    "uc_manager": "uc-manager version",
    "accelerator_driver": "Accelerator driver / compute capability",
    "kernel": "Kernel version",
    "memory_shm": "Memory & /dev/shm size",
    "aio_resources": "Kernel AIO resource pool",
    "bandwidth": "Storage bandwidth (posix aio/psync)",
}
