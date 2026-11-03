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
from .kv_cache_layout import UCMKVCacheLayout
from .kv_cache_spec import RestorePolicy, UCMKVCacheSpec, parse_kv_cache_config
from .runtime import ParallelRank, ParallelTopology, UCMRuntimeContext
from .store_client import AddressedToken
from .ucm_connector import UCMConnector

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
    "AddressedToken",
    "UCMRuntimeContext",
    "parse_kv_cache_config",
]
