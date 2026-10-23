"""Query the installed Cache core, not the caller's build-time environment."""

import ctypes
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def uses_partitioned_cache() -> bool:
    """Old wheels and platforms retaining cache/cc keep their SHM behavior."""
    library = Path(__file__).with_name("libcachestore.so")
    if not library.exists():
        return False
    handle = ctypes.CDLL(str(library))
    query = getattr(handle, "UcmCacheCoreVersion", None)
    if query is None:
        return False
    query.argtypes = []
    query.restype = ctypes.c_uint32
    return query() >= 2
