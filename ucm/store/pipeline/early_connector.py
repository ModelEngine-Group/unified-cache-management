"""Early-only pipeline assembled from native Store subclasses."""

import copy
from pathlib import Path

import numpy as np

from ucm.integration.vllm.early_lookup import current_lookup
from ucm.store.ucmstore_v1 import UcmKVStoreBaseV1

from .connector import UcmPipelineStore, _preload_metrics, ucmpipelinestore


class EarlyPipelineStore(UcmPipelineStore):
    def __init__(self, config: dict[str, object]) -> None:
        pipeline = config.get("store_pipeline")
        if pipeline != "Cache|Posix":
            raise ValueError(
                f"Early supports only store_pipeline='Cache|Posix', got {pipeline!r}"
            )

        UcmKVStoreBaseV1.__init__(self, config)
        self.store_ = ucmpipelinestore.PipelineStore()
        store_dir = Path(__file__).resolve().parent.parent
        posix_config = copy.deepcopy(config)
        if config.get("device_id", -1) >= 0:
            posix_config["tensor_size"] = config["shard_size"]
        _preload_metrics(store_dir)
        self.store_.Stack(
            "EarlyPosix",
            str(store_dir / "posix/libposixstore.so"),
            posix_config,
        )
        self.store_.Stack(
            "EarlyCache",
            str(store_dir / "cache/libcachestore.so"),
            config,
        )

    def lookup_on_prefix(self, block_ids: list[bytes]) -> int:
        context = current_lookup.get()
        if context is not None and context.connector.store is self:
            job = context.job
            if job is not None and job.state == "READY":
                # Native rank-consistency filtering may shorten this prefix.
                # Never claim data beyond the protected all-rank result.
                count = min(max(job.hit - context.hbm_hit_hashes, 0), len(block_ids))
                return count - 1
        return super().lookup_on_prefix(block_ids)

    def reserve_prefix(self, block_ids: list[bytes]) -> dict:
        return self.store_.ReservePrefix(
            np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        )

    def prefetch_reserved(self, handle: int) -> dict:
        return self.store_.PrefetchReserved(handle)

    def release_reservation(self, handle: int) -> None:
        self.store_.ReleaseReservation(handle)
