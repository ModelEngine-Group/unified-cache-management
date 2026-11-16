"""Patch SGLang request and radix insert lifecycles for the V2 linker."""

from contextvars import ContextVar
from functools import wraps


def install_request_hooks(linker_type):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.mem_cache.utils import get_storage_hash_str

    if getattr(UnifiedRadixCache.insert, "_ucm_v2_hook", False):
        return

    active_insert = ContextVar("ucm_v2_request_insert", default=None)
    checkpoint = UnifiedRadixCache.checkpoint_into_tree
    insert = UnifiedRadixCache.insert
    try:
        from sglang.srt.managers.scheduler_components.request_retirer import (
            RequestRetirer,
        )
    except ModuleNotFoundError as error:
        if error.name not in {
            "sglang.srt.managers.scheduler_components",
            "sglang.srt.managers.scheduler_components.request_retirer",
        }:
            raise
        RequestRetirer = None

    def linker_for(cache):
        linker = getattr(getattr(cache, "linker", None), "cache_linker", None)
        return linker if isinstance(linker, linker_type) else None

    @wraps(checkpoint)
    def checkpoint_request(cache, req, **kwargs):
        linker = linker_for(cache)
        if linker is None:
            return checkpoint(cache, req, **kwargs)
        token = active_insert.set((cache, linker, req.rid))
        try:
            return checkpoint(cache, req, **kwargs)
        finally:
            active_insert.reset(token)

    @wraps(insert)
    def insert_pages(cache, params):
        active = active_insert.get()
        if active is not None and active[0] is cache and len(params.key):
            _, linker, rid = active
            keys = get_storage_hash_str(params.key, page_size=cache.page_size)
            linker.associate_pages(rid, keys)
        return insert(cache, params)

    def wrap_release(release, get_cache):
        @wraps(release)
        def release_request(owner, req, **kwargs):
            try:
                return release(owner, req, **kwargs)
            finally:
                linker = linker_for(get_cache(owner))
                if linker is not None:
                    linker.finish_request(req.rid)

        return release_request

    if RequestRetirer is None:
        # v0.5.22 calls on_release after checkpointing and freeing the KV row.
        UnifiedRadixCache.on_release = wrap_release(
            UnifiedRadixCache.on_release, lambda cache: cache
        )
    else:
        RequestRetirer.release = wrap_release(
            RequestRetirer.release, lambda retirer: retirer.tree_cache
        )

    abandon = getattr(UnifiedRadixCache, "_abandon_attempt", None)
    if abandon is not None:
        @wraps(abandon)
        def abandon_request(cache, request):
            try:
                return abandon(cache, request)
            finally:
                linker = linker_for(cache)
                if linker is not None:
                    linker.finish_request(request.rid)

        UnifiedRadixCache._abandon_attempt = abandon_request

    insert_pages._ucm_v2_hook = True
    UnifiedRadixCache.checkpoint_into_tree = checkpoint_request
    UnifiedRadixCache.insert = insert_pages
