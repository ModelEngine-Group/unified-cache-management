# UCM patch for vllm-ascend 0.22.1:
# Install the shared UCM CPU affinity patch with 0.22.1 allocation rules.
from ucm.integration.vllm.patch.cpu_binding_affinity_patch import (
    install_cpu_binding_patch,
)
from ucm.integration.vllm.patch.utils import when_imported


@when_imported("vllm_ascend.cpu_binding")
def patch_cpu_binding(mod):
    from ucm.integration.vllm.patch.v0221.vllm_ascend.cpu_binding.bind_threads import (
        allocate,
    )

    install_cpu_binding_patch(mod, allocate_func=allocate)
