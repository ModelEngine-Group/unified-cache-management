"""Install the UCM CPU-affinity patch for vLLM-Ascend 0.25.1."""

from ucm.integration.vllm.patch.cpu_binding_affinity_patch import (
    install_cpu_binding_patch,
)
from ucm.integration.vllm.patch.utils import when_imported


@when_imported("vllm_ascend.cpu_binding")
def patch_cpu_binding(mod):
    from ucm.integration.vllm.patch.v0251.vllm_ascend.cpu_binding.bind_threads import (
        allocate,
        bind_threads,
        print_plan,
    )

    install_cpu_binding_patch(
        mod,
        allocate_func=allocate,
        bind_threads_func=bind_threads,
        print_plan_func=print_plan,
    )
