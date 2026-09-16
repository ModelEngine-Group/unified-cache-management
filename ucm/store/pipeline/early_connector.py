"""Early-only pipeline assembled from native Store subclasses."""

import numpy as np

from ucm.integration.vllm.early_lookup import current_lookup

from .connector import UcmPipelineStore, UcmPipelineStoreBuilder, ucmpipelinestore

_EARLY_PIPELINE = "Early(Cache|Posix)"


class _ReservedLoad:
    """Observe native I/O without duplicating conversion, queues or resources."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.reservations = ()
        self.observer = None
        self.receipts = {}

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    def Load(self, ids, indexes, addrs):
        reservations = self.reservations
        receipt = self.observer.begin(ids, indexes) if self.observer else None
        try:
            if not reservations and receipt is None:
                return self.pipeline.Load(ids, indexes, addrs)
            task = ucmpipelinestore.EarlyLoad(self.pipeline, ids, indexes, addrs, reservations)
            if receipt is not None:
                self.receipts[task] = receipt
                self.observer.submitted(receipt, task)
            return task
        except Exception:
            if receipt is not None:
                self.observer.complete(receipt, None, False)
            elif self.observer:
                self.observer.unscoped_failure()
            raise

    def Wait(self, task):
        receipt = self.receipts.pop(task, None)
        try:
            result = self.pipeline.Wait(task)
        except Exception:
            if receipt is not None:
                self.observer.complete(receipt, task, False)
            elif self.observer:
                self.observer.unscoped_failure()
            raise
        if receipt is not None:
            self.observer.complete(receipt, task, True)
        return result

    def Dump(self, ids, indexes, addrs, *args):
        if self.observer:
            self.observer.check_task("dump", len(indexes))
        try:
            return self.pipeline.Dump(ids, indexes, addrs, *args)
        except Exception:
            if self.observer:
                self.observer.unscoped_failure()
            raise


class _EarlyStack:
    """Adapt factory names while the native builder configures the same pipeline."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def Stack(self, name, path, config):
        if name not in ("Cache", "Posix"):
            raise ValueError(f"unsupported Early store: {name}")
        self.pipeline.Stack(f"Early{name}", path, config)


def _build_early_pipeline(config, pipeline):
    builder = UcmPipelineStoreBuilder.get("Cache|Posix")
    if builder is None:
        raise ValueError("unknown store pipeline: Cache|Posix")
    builder(dict(config, store_pipeline="Cache|Posix"), _EarlyStack(pipeline))


UcmPipelineStoreBuilder.register(_EARLY_PIPELINE, _build_early_pipeline)


class EarlyPipelineStore(UcmPipelineStore):
    def __init__(self, config: dict[str, object]) -> None:
        pipeline = config.get("store_pipeline")
        if pipeline != "Cache|Posix":
            raise ValueError(
                f"Early supports only store_pipeline='Cache|Posix', got {pipeline!r}"
            )

        required = (
            "EarlyReservePrefix", "EarlyPrefetchReserved", "EarlyReleaseReservation",
            "EarlyLoad",
            "EarlyBatch",
            "EarlyDomainInfo",
        )
        if not all(callable(getattr(ucmpipelinestore, name, None)) for name in required):
            raise RuntimeError(
                "Early pipeline bindings are missing; rebuild UCM for this source"
            )
        super().__init__(dict(config, store_pipeline=_EARLY_PIPELINE))
        self._native_store = self.store_
        self.store_ = _ReservedLoad(self._native_store)
        # The internal builder key changes selection only, not the user's config.
        self.config = config

    def lookup_on_prefix(self, block_ids: list[bytes]) -> int:
        context = current_lookup.get()
        if context is not None and context.connector.store is self:
            if context.hbm_only:
                # Recompute the native APC suffix. Still let native token and
                # dump-metadata generation run, without querying external KV.
                return -1
            job = context.job
            if job is not None and job.state == "READY":
                # Native rank-consistency filtering may shorten this prefix.
                # Never claim data beyond the protected all-rank result.
                count = min(max(job.hit - context.hbm_hit_hashes, 0), len(block_ids))
                return count - 1
        return super().lookup_on_prefix(block_ids)

    def reserve_prefix(self, block_ids: list[bytes]) -> dict:
        return ucmpipelinestore.EarlyReservePrefix(
            self._native_store, np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        )

    def domain_info(self):
        return dict(ucmpipelinestore.EarlyDomainInfo(self._native_store))

    def set_consumer(self, observer):
        self.store_.observer = observer

    def prefetch_reserved(self, handle: int) -> dict:
        return ucmpipelinestore.EarlyPrefetchReserved(self._native_store, handle)

    def release_reservation(self, handle: int) -> None:
        self.store_.reservations = tuple(
            active for active in self.store_.reservations if active != handle
        )
        ucmpipelinestore.EarlyReleaseReservation(self._native_store, handle)

    def set_load_reservations(self, handles) -> None:
        self.store_.reservations = tuple(handles)
