# UCM patch for vllm-ascend 0.22.1:
# Reuse common binding behavior, with 0.22.1's IRQ reservation rules.
from ucm.integration.vllm.patch.cpu_binding_affinity_patch import (
    assign_cpu_roles,
    bind_threads,
    print_plan,
)


def allocate(self) -> None:
    self.assign_ucm = {}

    reserve_irq_cpus = self._reserve_irq_cpus()
    min_cpus_per_npu = self._min_cpus_per_npu()

    for npu, pool in self.npu_cpu_pool.items():
        if len(pool) < min_cpus_per_npu:
            raise RuntimeError(
                "The number of CPUs is insufficient. Each NPU requires at "
                f"least {min_cpus_per_npu} CPUs."
            )

        if reserve_irq_cpus:
            main = pool[2:-2]
        else:
            main = pool[:-2]

        assign_cpu_roles(self, npu, main, [pool[-2]], [pool[-1]])
