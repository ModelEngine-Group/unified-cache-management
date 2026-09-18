# Shared UCM CPU affinity patch helpers for vllm-ascend.
import os
from pathlib import Path

import psutil
from vllm.logger import logger

from ucm.integration.vllm.patch.utils import patch_or_inject

_UCM_THREAD_PREFIX = "ucm_"
_UCM_HEALTH_THREAD_NAME = "ucm_health_mon"
_TASK_ROOT = Path("/proc/self/task")


def _logger():
    import vllm_ascend.cpu_binding as cpu_binding

    return getattr(cpu_binding, "logger", logger)


def _ucm_affinity_enabled() -> bool:
    return os.getenv("VLLM_CPU_AFFINITY") == "1"


def _current_npu(self) -> int:
    """Resolve the NPU this rank is actually bound to.

    vllm-ascend >= v0.26.0rc1 carries the resolved device ID as
    ``CpuAlloc.current_npu`` (upstream fix #15518); older versions index the
    discovered NPU list by local rank.
    """
    npu = getattr(self, "current_npu", None)
    if npu is not None:
        return npu
    return self.device_info.running_npu_list[self.rank_id]


def _split_contiguous_halves(cores: list[int]) -> tuple[list[int], list[int]]:
    ordered = sorted(set(cores))
    if len(ordered) < 2:
        return ordered, []

    segments: list[list[int]] = []
    segment = [ordered[0]]
    for core in ordered[1:]:
        if core == segment[-1] + 1:
            segment.append(core)
        else:
            segments.append(segment)
            segment = [core]
    segments.append(segment)

    worker_cores: list[int] = []
    ucm_cores: list[int] = []
    for segment in segments:
        middle = max(1, len(segment) // 2)
        worker_cores.extend(segment[:middle])
        ucm_cores.extend(segment[middle:])

    if not ucm_cores:
        middle = max(1, len(ordered) // 2)
        worker_cores = ordered[:middle]
        ucm_cores = ordered[middle:]

    return worker_cores, ucm_cores


def _task_snapshot() -> list[tuple[int, str]]:
    tasks: list[tuple[int, str]] = []
    try:
        entries = list(_TASK_ROOT.iterdir())
    except OSError as error:
        _logger().warning("Failed to enumerate tasks for UCM CPU binding: %s", error)
        return tasks

    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            name = (entry / "comm").read_text(encoding="utf-8").strip()
        except (OSError, ProcessLookupError):
            continue
        tasks.append((int(entry.name), name))
    return tasks


def _split_health_cores(ucm_cores: list[int]) -> tuple[list[int], list[int]]:
    if len(ucm_cores) < 2:
        return ucm_cores, ucm_cores
    return ucm_cores[:-1], ucm_cores[-1:]


def _ucm_thread_cores(
    name: str, ucm_cores: list[int], health_cores: list[int]
) -> list[int]:
    if name == _UCM_HEALTH_THREAD_NAME and health_cores:
        return health_cores
    return ucm_cores


def _bind_ucm_threads(self) -> None:
    """Pin every ``ucm_*`` thread of this process to the UCM cores."""
    current_npu = _current_npu(self)
    ucm_cores = getattr(self, "assign_ucm", {}).get(current_npu, [])
    health_cores = getattr(self, "assign_ucm_health", {}).get(current_npu, [])
    if not _ucm_affinity_enabled() or not (ucm_cores or health_cores):
        return

    bound_ucm = 0
    bound_health = 0
    for tid, name in _task_snapshot():
        if not name.startswith(_UCM_THREAD_PREFIX):
            continue
        cores = _ucm_thread_cores(name, ucm_cores, health_cores)
        if not cores:
            continue
        self.bind(str(tid), cores, False)
        if name == _UCM_HEALTH_THREAD_NAME:
            bound_health += 1
        else:
            bound_ucm += 1
    _logger().info(
        "[UCM CPU Affinity] vllm-ascend bound %s UCM tasks to cores %s",
        bound_ucm,
        ucm_cores,
    )
    _logger().info(
        "[UCM CPU Affinity] vllm-ascend bound %s health tasks to cores %s",
        bound_health,
        health_cores,
    )


def assign_cpu_roles(
    self,
    npu: int,
    main: list[int],
    acl: list[int],
    rel: list[int],
) -> None:
    if _ucm_affinity_enabled():
        worker_cores, ucm_cores = _split_contiguous_halves(main)
        if ucm_cores:
            main = worker_cores
        ucm_cores, health_cores = _split_health_cores(ucm_cores)
        self.assign_ucm[npu] = ucm_cores
        self.assign_ucm_health[npu] = health_cores
    else:
        self.assign_ucm[npu] = []
        self.assign_ucm_health[npu] = []

    self.assign_main[npu] = main
    self.assign_acl[npu] = acl
    self.assign_rel[npu] = rel


# ---------------------------------------------------------------------------
# Legacy replacements (vllm-ascend <= 0.25.1). Re-implement the allocation and
# binding rules inline; kept byte-for-byte identical to the historical shared
# implementation so the per-version v0191..v0251 installers keep working.
# ---------------------------------------------------------------------------


def allocate(self) -> None:
    self.assign_ucm = {}
    self.assign_ucm_health = {}

    import vllm_ascend.cpu_binding as cpu_binding

    min_cpus_per_npu = getattr(cpu_binding, "MIN_CPUS_PER_NPU", 5)

    for npu, pool in self.npu_cpu_pool.items():
        if len(pool) < min_cpus_per_npu:
            raise RuntimeError(
                "The number of CPUs is insufficient. Each NPU requires at "
                f"least {min_cpus_per_npu} CPUs."
            )

        assign_cpu_roles(self, npu, pool[2:-2], [pool[-2]], [pool[-1]])


def print_plan(self) -> None:
    cpu_logger = _logger()
    cpu_logger.info("The CPU allocation plan is as follows:")
    current_npu = self.device_info.running_npu_list[self.rank_id]
    main = " ".join(map(str, self.assign_main[current_npu]))
    ucm = " ".join(map(str, getattr(self, "assign_ucm", {}).get(current_npu, [])))
    ucm_health = " ".join(
        map(str, getattr(self, "assign_ucm_health", {}).get(current_npu, []))
    )
    acl = " ".join(map(str, self.assign_acl[current_npu]))
    rel = str(self.assign_rel[current_npu]) if self.assign_rel[current_npu] else ""
    cpu_logger.info(
        "NPU%s: main=[%s]  ucm=[%s]  ucm_health=[%s]  acl=[%s]  release=[%s]",
        current_npu,
        main,
        ucm,
        ucm_health,
        acl,
        rel,
    )


def bind_threads(self) -> None:
    import vllm_ascend.cpu_binding as cpu_binding

    thread_message, _ = cpu_binding.execute_command(["ps", "-Te"])
    threads_map = cpu_binding.CpuAlloc.get_threads_map(thread_message)
    main_pid = str(psutil.Process().pid)
    current_npu = self.device_info.running_npu_list[self.rank_id]
    self.bind(main_pid, self.assign_main[current_npu], True)

    _bind_ucm_threads(self)

    for acl_thread in threads_map.get(main_pid, {}).get("acl_thread", []):
        self.bind(acl_thread, self.assign_acl[current_npu], False)
    for release_thread in threads_map.get(main_pid, {}).get("release_thread", []):
        self.bind(release_thread, self.assign_rel[current_npu], False)
    # self.bind_memory(main_pid, current_npu)


# ---------------------------------------------------------------------------
# Fixed replacements (vllm-ascend >= 0.26.0). Version-independent: each
# function calls the original upstream ``CpuAlloc`` method captured at install
# time, then augments it. Allocation rules (Ascend 950 clusters, IRQ
# reservation, device mapping via ``self.current_npu``) stay entirely in
# upstream code, so no per-version maintenance is needed from 0.26.0 onward.
# ---------------------------------------------------------------------------


def allocate_fixed(self, orig_allocate=None) -> None:
    """Reserve UCM cores on top of the upstream allocation plan."""
    if orig_allocate is None:
        allocate(self)
        return
    orig_allocate(self)

    self.assign_ucm = {}
    self.assign_ucm_health = {}
    if _ucm_affinity_enabled():
        for npu, main in self.assign_main.items():
            worker_cores, ucm_cores = _split_contiguous_halves(main)
            if ucm_cores:
                self.assign_main[npu] = worker_cores
            ucm_cores, health_cores = _split_health_cores(ucm_cores)
            self.assign_ucm[npu] = ucm_cores
            self.assign_ucm_health[npu] = health_cores
    else:
        for npu in self.assign_main:
            self.assign_ucm[npu] = []
            self.assign_ucm_health[npu] = []


def print_plan_fixed(self, orig_print_plan=None) -> None:
    """Print the upstream plan, followed by the UCM core reservation."""
    if orig_print_plan is None:
        print_plan(self)
        return
    orig_print_plan(self)

    cpu_logger = _logger()
    current_npu = _current_npu(self)
    ucm = " ".join(map(str, getattr(self, "assign_ucm", {}).get(current_npu, [])))
    ucm_health = " ".join(
        map(str, getattr(self, "assign_ucm_health", {}).get(current_npu, []))
    )
    cpu_logger.info(
        "NPU%s: ucm=[%s]  ucm_health=[%s]",
        current_npu,
        ucm,
        ucm_health,
    )


def bind_threads_fixed(self, orig_bind_threads=None) -> None:
    """Bind UCM threads after the upstream binding logic.

    Calling the original first keeps every upstream behavior (Ascend 950
    clusters, IRQ reservation, memory placement) intact; UCM threads are
    pinned afterwards.
    """
    if orig_bind_threads is None:
        bind_threads(self)
        return
    orig_bind_threads(self)
    _bind_ucm_threads(self)


def _mark_fixed(func):
    """Tag a replacement so install wraps it around the original method."""
    setattr(func, "_ucm_fixed", True)
    return func


allocate_fixed = _mark_fixed(allocate_fixed)
bind_threads_fixed = _mark_fixed(bind_threads_fixed)
print_plan_fixed = _mark_fixed(print_plan_fixed)


def _wrap_if_fixed(func, orig):
    """Wrap *func* with the captured original method when it is a fixed
    replacement; return per-version replacements untouched."""
    if not getattr(func, "_ucm_fixed", False):
        return func
    if orig is None:
        return func

    def wrapped(self, *args, **kwargs):
        return func(self, orig, *args, **kwargs)

    return wrapped


def install_cpu_binding_patch(
    mod,
    allocate_func=allocate,
    bind_threads_func=bind_threads,
    print_plan_func=print_plan,
) -> None:
    cpu_logger = getattr(mod, "logger", logger)
    cpu_logger.debug(f"Patched {mod} called")

    if getattr(mod.CpuAlloc, "_ucm_cpu_binding_patched", False):
        return

    if not hasattr(mod.CpuAlloc, "bind_threads"):
        cpu_logger.warning("Skip CPU binding patch: CpuAlloc.bind_threads is missing")
        return

    cls = mod.CpuAlloc
    allocate_func = _wrap_if_fixed(allocate_func, getattr(cls, "allocate", None))
    print_plan_func = _wrap_if_fixed(print_plan_func, getattr(cls, "print_plan", None))
    bind_threads_func = _wrap_if_fixed(
        bind_threads_func, getattr(cls, "bind_threads", None)
    )

    patch_or_inject(cls, "allocate", allocate_func)
    patch_or_inject(cls, "print_plan", print_plan_func)
    patch_or_inject(cls, "bind_threads", bind_threads_func)
    setattr(cls, "_ucm_cpu_binding_patched", True)
    cpu_logger.info(
        "UCM CPU binding patch applied: CpuAlloc.allocate/print_plan/bind_threads"
    )
