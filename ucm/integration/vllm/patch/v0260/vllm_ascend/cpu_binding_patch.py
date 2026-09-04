"""Install the UCM CPU-affinity patch for vLLM-Ascend 0.26.0 and later.

The fixed implementation wraps the upstream ``CpuAlloc`` methods (allocate /
print_plan / bind_threads) instead of re-implementing allocation rules, so it
is version-independent: from v0.26.0 onward this entry point can be reused as
is, no per-version cpu_binding patch is needed.
"""

from ucm.integration.vllm.patch.cpu_binding_affinity_patch import (
    allocate_fixed as allocate,
)
from ucm.integration.vllm.patch.cpu_binding_affinity_patch import (
    bind_threads_fixed as bind_threads,
)
from ucm.integration.vllm.patch.cpu_binding_affinity_patch import (
    install_cpu_binding_patch,
)
from ucm.integration.vllm.patch.cpu_binding_affinity_patch import (
    print_plan_fixed as print_plan,
)
from ucm.integration.vllm.patch.utils import when_imported


@when_imported("vllm_ascend.cpu_binding")
def patch_cpu_binding(mod):
    install_cpu_binding_patch(
        mod,
        allocate_func=allocate,
        bind_threads_func=bind_threads,
        print_plan_func=print_plan,
    )
