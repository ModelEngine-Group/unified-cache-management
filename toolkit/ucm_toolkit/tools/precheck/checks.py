"""Concrete environment checks. Each function returns a :class:`CheckResult`.

The cheap checks (versions, driver, kernel, memory) are pure stdlib and run on
any host. The bandwidth check lives in :mod:`pre_check.bandwidth` because it
pulls in numpy and the ucm C++ store.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distributions
from importlib.metadata import version as pkg_version
from typing import List, Optional, Tuple

from .config import PrecheckConfig
from .parseutil import (
    compare_versions,
    extract_nvidia_compute_cap,
    normalize_vllm_ascend_version,
    parse_ascend_install_info,
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


def _is_ucm_distribution(name: str) -> bool:
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    return re.search(r"(?:^|-)uc-manager(?:-|$)", normalized) is not None


def installed_ucm_distributions() -> dict[str, str]:
    """Return names and versions matching uc-manager, including Forks and variants."""
    installed = {}
    for dist in distributions():
        name = dist.metadata["Name"] or ""
        if _is_ucm_distribution(name):
            installed[name] = dist.version
    return dict(sorted(installed.items()))


def _ucm_distributions_from_pip() -> Tuple[dict, Optional[str], Optional[str]]:
    """List UCM distributions through the ``pip``/``pip3`` found on PATH.

    ``ucm-toolkit`` and the serving stack are sometimes installed in different
    Python environments. importlib.metadata then correctly reports nothing for
    the toolkit interpreter even though ``pip list`` in the user's shell sees
    ``uc-manager-cann910-a5``. Querying pip is a display-only fallback and makes
    that common split-environment deployment visible instead of showing ``-``.
    """
    pip = shutil.which("pip") or shutil.which("pip3")
    if pip is None:
        return {}, None, "pip/pip3 not found on PATH"

    command = [pip, "list", "--format=json"]
    rc, out, err = _run(command, timeout=15)
    if rc != 0:
        return {}, pip, err.strip() or f"pip list failed with rc={rc}"

    # Pip normally emits pure JSON on stdout and sends LD_PRELOAD and similar
    # warnings to stderr. Restricting json.loads to the outer array also makes
    # this tolerant of unexpected prefix/suffix text.
    start, end = out.find("["), out.rfind("]")
    if start < 0 or end <= start:
        return {}, pip, "pip list did not return JSON rows"
    try:
        rows = json.loads(out[start : end + 1])
    except (TypeError, ValueError) as exc:
        return {}, pip, f"could not parse pip list output: {exc}"
    if not isinstance(rows, list):
        return {}, pip, "pip list output was not a JSON array"

    installed: dict = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not name or not _is_ucm_distribution(str(name)):
            continue
        installed[str(name)] = str(row.get("version") or "")
    return dict(sorted(installed.items())), pip, None


def check_uc_manager_version() -> CheckResult:
    """Report installed UCM package names and versions without importing UCM."""
    installed = installed_ucm_distributions()
    raw: dict = {
        "distributions": installed,
        "source": "importlib.metadata",
        "python": sys.executable,
    }
    if not installed:
        pip_installed, pip_command, pip_error = _ucm_distributions_from_pip()
        if pip_command:
            raw["pip_command"] = pip_command
        if pip_error:
            raw["pip_error"] = pip_error
        if pip_installed:
            installed = pip_installed
            raw["distributions"] = installed
            raw["source"] = "pip"
        else:
            raw["source"] = "importlib.metadata+pip"

    if installed:
        if len(installed) == 1:
            raw["version"] = next(iter(installed.values()))
        source = raw["source"]
        return CheckResult(
            name="uc_manager",
            severity=INFO,
            status=STATUS_INFO,
            value=", ".join(f"{name}={version}" for name, version in installed.items()),
            detail=f"installed UCM packages (source: {source})",
            raw=raw,
        )
    return CheckResult(
        name="uc_manager",
        severity=INFO,
        status=STATUS_INFO,
        value="-",
        detail="uc-manager is not visible to the ucm-toolkit Python or pip on PATH",
        raw=raw,
    )


# ---------------------------------------------------------------------------
# 3. Accelerator driver — WARN
# ---------------------------------------------------------------------------


def check_accelerator_driver(cfg: PrecheckConfig) -> CheckResult:
    """Check NVIDIA compute capability or Ascend HDK version.

    Auto-detects the platform: prefers ``nvidia-smi`` when present, else falls
    back to ``npu-smi -v``. With neither present the check FAILs (UCM needs an
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


def _read_text_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _check_ascend(cfg: PrecheckConfig) -> CheckResult:
    command = ["npu-smi", "-v"]
    rc, out, err = _run(command)
    if rc != 0:
        return CheckResult(
            name="accelerator_driver",
            severity=WARN,
            status=STATUS_WARN,
            value="-",
            threshold=f"HDK > {cfg.ascend_min_hdk}",
            detail=f"npu-smi -v failed: {err.strip() or 'rc=%d' % rc}",
            remediation="ensure the Ascend driver is loaded and npu-smi is on PATH",
            raw={
                "platform": "ascend",
                "command": "npu-smi -v",
                "rc": rc,
                "stderr": err,
            },
        )
    parsed = parse_npu_smi_versions(out)
    # ``npu-smi -v`` reports the version directly and works across the Ascend
    # firmware layouts observed in the field. Prefer its explicit Version field;
    # some builds print only the ``npu-smi X.Y.Z`` form.
    hdk = parsed.get("hdk") or parsed.get("tool")
    version_source = "npu-smi -v"
    floor = parse_version(cfg.ascend_min_hdk)
    value = f"HDK={hdk or '?'}"
    raw = {
        "platform": "ascend",
        "command": "npu-smi -v",
        **parsed,
        "version_source": version_source,
    }

    if hdk is None:
        return CheckResult(
            name="accelerator_driver",
            severity=WARN,
            status=STATUS_WARN,
            value=value,
            threshold=f"HDK > {cfg.ascend_min_hdk}",
            detail=(
                "could not extract an HDK/driver version from npu-smi -v; "
                f"heuristic parse found: {parsed['raw_versions'] or 'none'}"
            ),
            remediation=(
                "ensure the Ascend driver is installed and npu-smi -v reports "
                "a version"
            ),
            raw=raw,
        )
    if compare_versions(parse_version(hdk), floor) <= 0:
        return CheckResult(
            name="accelerator_driver",
            severity=WARN,
            status=STATUS_WARN,
            value=value,
            threshold=f"HDK > {cfg.ascend_min_hdk}",
            detail=(
                f"HDK {hdk} is not strictly newer than {cfg.ascend_min_hdk} "
                f"(source: {version_source})"
            ),
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
        detail=f"Ascend HDK/driver version OK (source: {version_source})",
        raw=raw,
    )


def _ascend_toolkit_install_info_paths() -> List[str]:
    """Return likely CANN install-info paths for the current architecture."""
    arch = platform.machine()
    if not arch:
        return []

    roots = []
    toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME")
    if toolkit_home:
        roots.append(toolkit_home)
    roots.append("/usr/local/Ascend/ascend-toolkit/latest")

    filename = "ascend_toolkit_install.info"
    arch_dir = f"{arch}-linux"
    paths = []
    for root in roots:
        # The standard layout puts the file below ``<arch>-linux``. Some
        # deployments set ASCEND_TOOLKIT_HOME to that arch-specific directory.
        paths.append(os.path.join(root, arch_dir, filename))
        paths.append(os.path.join(root, filename))
    return list(dict.fromkeys(paths))


def check_ascend_cann_version() -> CheckResult:
    """Report CANN package metadata as an informational Ascend check.

    CANN's ``ascend_toolkit_install.info`` contains useful diagnostics such as
    ``package_name``, ``version`` and ``innerversion``. The file is optional
    from precheck's perspective: if it is missing, this item remains INFO and
    does not affect the exit code.
    """
    if _have("nvidia-smi") or not _have("npu-smi"):
        return CheckResult(
            name="ascend_cann",
            severity=INFO,
            status=STATUS_SKIP,
            value="-",
            detail="not an Ascend environment",
            raw={"platform": "nvidia" if _have("nvidia-smi") else "none"},
        )

    arch = platform.machine() or "unknown"
    paths = _ascend_toolkit_install_info_paths()
    for path in paths:
        text = _read_text_file(path)
        if text is None:
            continue

        info = parse_ascend_install_info(text)
        package_name = info.get("package_name", "")
        version = info.get("version", "")
        innerversion = info.get("innerversion", "")
        raw = {
            "path": path,
            "arch": arch,
            "package_name": package_name,
            "version": version,
            "innerversion": innerversion,
        }
        if version:
            return CheckResult(
                name="ascend_cann",
                severity=INFO,
                status=STATUS_INFO,
                value=(
                    f"CANN={version}, package={package_name or '-'}, "
                    f"inner={innerversion or '-'}"
                ),
                detail=(
                    f"{package_name or 'Ascend CANN toolkit'}, "
                    f"inner version={innerversion or '-'}"
                ),
                raw=raw,
            )
        return CheckResult(
            name="ascend_cann",
            severity=INFO,
            status=STATUS_INFO,
            value=(
                f"CANN=?, package={package_name or '-'}, "
                f"inner={innerversion or '-'}"
            ),
            detail="install info is readable but has no version field",
            raw=raw,
        )

    return CheckResult(
        name="ascend_cann",
        severity=INFO,
        status=STATUS_INFO,
        value="CANN=not found",
        detail="CANN toolkit install info is informational and was not found",
        raw={"arch": arch, "paths": paths},
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
    "ascend_cann": "Ascend CANN version",
    "kernel": "Kernel version",
    "memory_shm": "Memory & /dev/shm size",
    "aio_resources": "Kernel AIO resource pool",
    "bandwidth": "Storage bandwidth (posix aio/psync)",
}
