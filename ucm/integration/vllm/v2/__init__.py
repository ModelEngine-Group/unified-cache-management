"""Second-generation UCM vLLM connector components.

This package is intentionally isolated from the production connector while the
v2 contract is validated.  Importing it does not alter connector registration.
"""

from .connector_scheduler import (
    UCMConnectorMetadata,
    UCMConnectorScheduler,
)
from .connector_worker import (
    UCMConnectorWorker,
    UCMTransferBuilder,
)
from .layout import (
    RestorePolicy,
    UCMKVCacheLayout,
    UCMKVCacheSpec,
    parse_kv_cache_config,
)
from .runtime import ParallelRank, ParallelTopology, UCMRuntimeContext
from .test_stores import (
    FileTestStore,
    MemoryTestStore,
    TorchTensorByteAccess,
    UCMByteAccess,
)
from .ucm_connector import UCMConnector
from .ucm_proxy import (
    ParallelProxy,
    PipelineProxy,
    UCMProxy,
    UCMProxyAdapter,
)

__all__ = [
    "RestorePolicy",
    "ParallelRank",
    "ParallelTopology",
    "UCMConnector",
    "UCMConnectorScheduler",
    "UCMConnectorWorker",
    "UCMConnectorMetadata",
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
