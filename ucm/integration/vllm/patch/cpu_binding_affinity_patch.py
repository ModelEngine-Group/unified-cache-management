# Shared UCM CPU affinity patch helpers for vllm-ascend.
import os
from pathlib import Path

import psutil
from vllm.logger import logger

from ucm.integration.vllm.patch.utils import patch_or_inject

_UCM_THREAD_PREFIX = "ucm_"
_TASK_ROOT = Path("/proc/self/task")


def _logger():
    import vllm_ascend.cpu_binding as cpu_binding

    return getattr(cpu_binding, "logger", logger)


def _ucm_affinity_enabled() -> bool:
    return os.getenv("VLLM_CPU_AFFINITY") == "1"


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
        self.assign_ucm[npu] = ucm_cores
    else:
        self.assign_ucm[npu] = []

    self.assign_main[npu] = main
    self.assign_acl[npu] = acl
    self.assign_rel[npu] = rel


def allocate(self) -> None:
    self.assign_ucm = {}

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
    acl = " ".join(map(str, self.assign_acl[current_npu]))
    rel = str(self.assign_rel[current_npu]) if self.assign_rel[current_npu] else ""
    cpu_logger.info(
        "NPU%s: main=[%s]  ucm=[%s]  acl=[%s]  release=[%s]",
        current_npu,
        main,
        ucm,
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

    ucm_cores = getattr(self, "assign_ucm", {}).get(current_npu, [])
    if _ucm_affinity_enabled() and ucm_cores:
        bound_ucm = 0
        for tid, name in _task_snapshot():
            if not name.startswith(_UCM_THREAD_PREFIX):
                continue
            self.bind(str(tid), ucm_cores, False)
            bound_ucm += 1
        _logger().info(
            "[UCM CPU Affinity] vllm-ascend bound %s UCM tasks to cores %s",
            bound_ucm,
            ucm_cores,
        )

    for acl_thread in threads_map.get(main_pid, {}).get("acl_thread", []):
        self.bind(acl_thread, self.assign_acl[current_npu], False)
    for release_thread in threads_map.get(main_pid, {}).get("release_thread", []):
        self.bind(release_thread, self.assign_rel[current_npu], False)
    # self.bind_memory(main_pid, current_npu)


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

    patch_or_inject(mod.CpuAlloc, "allocate", allocate_func)
    patch_or_inject(mod.CpuAlloc, "print_plan", print_plan_func)
    patch_or_inject(mod.CpuAlloc, "bind_threads", bind_threads_func)
    setattr(mod.CpuAlloc, "_ucm_cpu_binding_patched", True)
    cpu_logger.info(
        "UCM CPU binding patch applied: CpuAlloc.allocate/print_plan/bind_threads"
    )
