"""Thin Early extensions of the native direct and layerwise connectors."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .early_lookup import LookupContext, current_lookup
from .early_prefetch import EarlyScheduler, EarlyWorker
from .ucm_connector import (
    KVConnectorRole,
    UCMConnectorMetadata,
    UCMDirectConnector,
    UCMLayerWiseConnector,
    logger,
)


@dataclass
class EarlyConnectorMetadata(UCMConnectorMetadata):
    early_keys: dict[str, str] = field(default_factory=dict)


class _EarlyConnectorMixin:
    _defer_scheduler_store = True

    def _lookup_context(self):
        context = current_lookup.get()
        return context if context is not None and context.connector is self else None

    @property
    def enable_record_traces(self):
        # Early records once per instance. Suppress the native inline trace
        # only while that same connector consumes the prepared result.
        return self._early_record_traces and self._lookup_context() is None

    @enable_record_traces.setter
    def enable_record_traces(self, enabled):
        # Retain the value assigned by native Config initialization.
        self._early_record_traces = enabled

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self.connector_configs = copy.deepcopy(self.connector_configs)
        connector = self.connector_configs[0]
        name = connector["ucm_connector_name"]
        module_path = connector.get("ucm_connector_module_path")
        if name != "UcmPipelineStore" or module_path not in (
            None,
            "ucm.store.pipeline.connector",
        ):
            raise ValueError("Early requires the native UcmPipelineStore")
        connector["ucm_connector_name"] = "EarlyPipelineStore"
        connector["ucm_connector_module_path"] = "ucm.store.pipeline.early_connector"
        if role == KVConnectorRole.SCHEDULER:
            self.store = self._create_store(None)
        self._early = (
            EarlyScheduler(self) if role == KVConnectorRole.SCHEDULER else None
        )
        logger.info(
            "UCM_EARLY enabled role=%s dp=%s tp=%s share_identity=%s "
            "policy=stan_hrrn lookup_workers=4 data_workers=1 work_unit=shard "
            "connector=%s source=%s",
            role,
            vllm_config.parallel_config.data_parallel_index,
            self.tp_size,
            self.unique_id,
            type(self).__name__,
            __file__,
        )

    def on_new_request(self, request):
        super().on_new_request(request)
        self._early.submit(request)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        assert num_computed_tokens % self.block_size == 0
        job = self._early.probe(request)
        if job is not None and job.state == "PENDING":
            return None, False
        context = LookupContext(
            self,
            request,
            job,
            num_computed_tokens // self.block_size * self.cp_world_size,
        )
        token = current_lookup.set(context)
        try:
            return super().get_num_new_matched_tokens(request, num_computed_tokens)
        finally:
            current_lookup.reset(token)

    def _bind_request_block_hasher(self):
        super()._bind_request_block_hasher()
        if self._role == KVConnectorRole.SCHEDULER:
            self._native_request_block_hasher = self.request_block_hasher
            self.request_block_hasher = self._early_request_block_hasher

    def _early_request_block_hasher(self, request):
        context = self._lookup_context()
        if context is None or request is not context.request:
            return self._native_request_block_hasher(request)
        job = context.job
        blocks = job.blocks if job is not None else None
        if blocks is None:
            blocks = self._native_request_block_hasher(request)
            if job is not None:
                job.blocks = blocks
        self._early.trace_request(
            context.request, blocks, enabled=self._early_record_traces
        )
        return blocks

    def _prefetch_direct_hit_key_hotness(self, hbm_hit_block_ids, all_hit_block_ids):
        context = self._lookup_context()
        if context is not None and context.job is not None:
            if context.job.state == "READY":
                return
        return super()._prefetch_direct_hit_key_hotness(
            hbm_hit_block_ids, all_hit_block_ids
        )

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        for request_id in metadata.preempted_req_ids:
            self._early.finish(request_id, cancelled=True)
        return EarlyConnectorMetadata(
            **vars(metadata), early_keys=self._early.scheduled(metadata.request_meta)
        )

    def request_finished(self, request, block_ids):
        self._early.finish(request, cancelled="ABORT" in str(request.status.name))
        return super().request_finished(request, block_ids)

    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        self._early = EarlyWorker(self)

    def bind_connector_metadata(self, connector_metadata):
        super().bind_connector_metadata(connector_metadata)
        self._early.bind(connector_metadata)

    def handle_preemptions(self, kv_connector_metadata):
        super().handle_preemptions(kv_connector_metadata)
        request_ids = (
            kv_connector_metadata
            if isinstance(kv_connector_metadata, set)
            else getattr(kv_connector_metadata, "preempted_req_ids", set())
        )
        if request_ids:
            self._early.finished(request_ids)

    def start_load_kv(self, forward_context, **kwargs):
        super().start_load_kv(forward_context, **kwargs)
        if not self.use_layerwise:
            self._early.loaded({
                request_id for request_id, meta in
                self._get_connector_metadata().request_meta.items()
                if not meta.load_async
            })

    def wait_for_layer_load(self, layer_name):
        super().wait_for_layer_load(layer_name)
        if self.use_layerwise and self.has_connector_metadata():
            if self._failure_req_ids:
                self._early.loaded(self._failure_req_ids)
            if self.layer_name_to_id[layer_name] == self.layer_ids[-1]:
                self._early.loaded(self._get_connector_metadata().request_meta)

    def get_finished(self, finished_req_ids):
        result = super().get_finished(finished_req_ids)
        if result[1]:
            # Native polling has waited/reaped these tasks before releasing
            # their Early reservations, including failed async transfers.
            self._early.loaded(result[1])
        self._early.finished(finished_req_ids)
        return result

    def shutdown(self):
        try:
            if self._early is not None:
                early, self._early = self._early, None
                early.close()
        finally:
            super().shutdown()


class EarlyDirectConnector(_EarlyConnectorMixin, UCMDirectConnector):
    pass


class EarlyLayerWiseConnector(_EarlyConnectorMixin, UCMLayerWiseConnector):
    pass
