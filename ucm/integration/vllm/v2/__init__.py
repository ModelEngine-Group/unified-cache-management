"""Second-generation UCM vLLM connector components.

This package is intentionally isolated from the production connector while the
v2 contract is validated.  Importing it does not alter connector registration.
"""

from .layout import (
    UCMKVCacheLayout,
    UCMKVCacheSpec,
    parse_kv_cache_config,
)
from .test_stores import (
    FileTestStore,
    MemoryTestStore,
    TorchTensorByteAccess,
    UCMByteAccess,
)
from .ucm_connector import UCMConnector, UCMConnectorMetadata, UCMRuntimeContext
from .ucm_kv_cache import UCMTransferBuilder
from .ucm_proxy import (
    ParallelProxy,
    PipelineProxy,
    UCMProxy,
    UCMProxyAdapter,
)
from .ucm_scheduler import UCMDispatcher

__all__ = [
    "UCMConnector",
    "UCMConnectorMetadata",
    "UCMDispatcher",
    "UCMKVCacheLayout",
    "UCMKVCacheSpec",
    "UCMTransferBuilder",
    "UCMProxy",
    "UCMProxyAdapter",
    "UCMByteAccess",
    "UCMRuntimeContext",
    "FileTestStore",
    "MemoryTestStore",
    "PipelineProxy",
    "ParallelProxy",
    "TorchTensorByteAccess",
    "parse_kv_cache_config",
]
