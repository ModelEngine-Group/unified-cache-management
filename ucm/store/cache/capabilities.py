"""Query the installed Cache core, not the caller's build-time environment."""

import ctypes
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def cache_core_version() -> int:
    """Return 1 for Cache/cc, 2 for Cache2, or 0 for an older unversioned wheel."""
    library = Path(__file__).with_name("libcachestore.so")
    if not library.exists():
        return 0
    handle = ctypes.CDLL(str(library))
    query = getattr(handle, "UcmCacheCoreVersion", None)
    if query is None:
        return 0
    query.argtypes = []
    query.restype = ctypes.c_uint32
    return query()


def uses_partitioned_cache() -> bool:
    return cache_core_version() >= 2
