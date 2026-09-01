"""UCM-aware CPU role allocation for vLLM-Ascend 0.26.0.

Upstream ``vllm_ascend.cpu_binding`` is unchanged relative to 0.23.0, so the
Ascend 950 topology-aware path and the default IRQ-reservation path reuse the
0.23.0 allocation rules verbatim.
"""

import psutil

from ucm.integration.vllm.patch.cpu_binding_affinity_patch import (
    assign_cpu_roles,
)
from ucm.integration.vllm.patch.cpu_binding_affinity_patch import (
    bind_threads as bind_ucm_threads,
)
from ucm.integration.vllm.patch.cpu_binding_affinity_patch import (
    print_plan as print_ucm_plan,
)


def allocate(self) -> None:
    """Reserve UCM cores while retaining 0.26.0 device-specific rules."""
    if self._is_ascend_950():
        # Ascend 950 supplies one topology-aware CPU cluster per NPU. Keep that
        # cluster assignment, but partition the cluster so UCM I/O workers do
        # not contend with the vLLM worker threads when affinity is enabled.
        self.assign_ucm = {}
        self.assign_ucm_health = {}
        for npu, cpu_pool in self.npu_cpu_pool.items():
            assign_cpu_roles(self, npu, cpu_pool, [], [])
        return

    self.assign_ucm = {}
    self.assign_ucm_health = {}
    reserve_irq_cpus = self._reserve_irq_cpus()
    min_cpus_per_npu = self._min_cpus_per_npu()
    for npu, cpu_pool in self.npu_cpu_pool.items():
        if len(cpu_pool) < min_cpus_per_npu:
            raise RuntimeError(
                "The number of CPUs is insufficient. Each NPU requires at "
                f"least {min_cpus_per_npu} CPUs."
            )
        main = cpu_pool[2:-2] if reserve_irq_cpus else cpu_pool[:-2]
        assign_cpu_roles(self, npu, main, [cpu_pool[-2]], [cpu_pool[-1]])


def print_plan(self) -> None:
    """Print the worker/UCM split for every supported device."""
    print_ucm_plan(self)


def bind_threads(self) -> None:
    """Bind UCM tasks and preserve Ascend 950 memory placement."""
    if self._is_ascend_950():
        bind_ucm_threads(self)
        main_pid = str(psutil.Process().pid)
        current_npu = self.device_info.running_npu_list[self.rank_id]
        self.bind_memory(main_pid, current_npu)
        return
    bind_ucm_threads(self)
