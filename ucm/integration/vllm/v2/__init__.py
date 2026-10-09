"""Second-generation UCM vLLM connector components.

This package is intentionally isolated from the production connector while the
v2 contract is validated.  Importing it does not alter connector registration.
"""

from .ucm_connector import UCMConnector, UCMConnectorMetadata, UCMRuntimeContext
from .layout import (
    UCMKVCacheLayout,
    UCMKVCacheSpec,
    parse_kv_cache_config,
)
from .ucm_kv_cache import UCMTransferBuilder
from .ucm_proxy import (
    SimpleFileUCMProxy,
    TorchTensorByteAccess,
    UCMByteAccess,
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
    "SimpleFileUCMProxy",
    "TorchTensorByteAccess",
    "parse_kv_cache_config",
]
