# UCM patch for vllm-ascend 0.19.1:
# Install the shared UCM CPU affinity patch.
from ucm.integration.vllm.patch.cpu_binding_affinity_patch import (
    install_cpu_binding_patch,
)
from ucm.integration.vllm.patch.utils import when_imported


@when_imported("vllm_ascend.cpu_binding")
def patch_cpu_binding(mod):
    install_cpu_binding_patch(mod)
