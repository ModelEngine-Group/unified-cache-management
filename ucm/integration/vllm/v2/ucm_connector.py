"""vLLM callbacks with base metadata and separate Scheduler/Worker state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)

from .connector_scheduler import (
    UCMConnectorMetadata,
    UCMConnectorScheduler,
)
from .connector_worker import UCMConnectorWorker, UCMWorkerMetadata
from .runtime import UCMRuntimeContext

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


class UCMConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if (
            len(kv_cache_config.kv_cache_groups) > 1
            and vllm_config.kv_transfer_config.kv_load_failure_policy != "fail"
        ):
            raise ValueError(
                "UCM v2 multi-group KV cache requires kv_load_failure_policy='fail'"
            )
        self.context = UCMRuntimeContext.from_vllm_config(
            vllm_config, role, kv_cache_config
        )
        graph_mode = vllm_config.compilation_config.cudagraph_mode
        has_full_graphs = (
            not vllm_config.model_config.enforce_eager
            and graph_mode is not None
            and graph_mode.has_full_cudagraphs()
        )
        speculative = vllm_config.speculative_config
        recompute_tokens = 1
        if has_full_graphs:
            recompute_tokens = 2 + (
                speculative.num_speculative_tokens if speculative is not None else 0
            )
        self.connector_scheduler = (
            UCMConnectorScheduler(self.context, recompute_tokens=recompute_tokens)
            if role == KVConnectorRole.SCHEDULER
            else None
        )
        self.connector_worker = (
            UCMConnectorWorker(
                self.context,
                kv_cache_config,
                num_hidden_layers=vllm_config.model_config.hf_text_config.num_hidden_layers,
            )
            if role == KVConnectorRole.WORKER
            else None
        )

    @property
    def requires_kv_delivery(self):
        return False

    def shutdown(self):
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()
        else:
            self.connector_worker.shutdown()

    def on_new_request(self, request: Request) -> None:
        self.connector_scheduler.on_new_request(request)

    def has_pending_push_work(self) -> bool:
        return (
            self.connector_scheduler is not None
            and self.connector_scheduler.has_pending_push_work()
        )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> UCMConnectorMetadata:
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        self.connector_scheduler.update_connector_output(connector_output)

    def set_xfer_handshake_metadata(self, metadata):
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def get_handshake_metadata(self):
        return self.connector_worker.get_handshake_metadata()

    def register_kv_caches(self, kv_caches):
        self.connector_worker.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata):
        self.connector_worker.begin_step(connector_metadata)
        super().bind_connector_metadata(connector_metadata)

    def start_load_kv(self, forward_context, **kwargs):
        if self.has_connector_metadata():
            self.connector_worker.start_load_kv(
                self._get_connector_metadata(),
                forward_context,
                **kwargs,
            )

    def wait_for_layer_load(self, layer_name):
        if self.has_connector_metadata():
            self.connector_worker.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if self.has_connector_metadata():
            self.connector_worker.save_kv_layer(
                self._get_connector_metadata(),
                layer_name,
                kv_layer,
                attn_metadata,
                **kwargs,
            )

    def wait_for_save(self):
        if self.has_connector_metadata():
            self.connector_worker.wait_for_save(self._get_connector_metadata())

    def get_transfer_results(self, finished_req_ids):
        self.connector_worker.poll_finished()
        return super().get_transfer_results(finished_req_ids)

    def build_connector_worker_meta(self):
        return self.connector_worker.build_connector_worker_meta()

    def get_block_ids_with_load_errors(self):
        return self.connector_worker.get_block_ids_with_load_errors()

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.connector_scheduler.update_state_after_alloc(request, num_external_tokens)

    def request_finished_all_groups(self, request, block_ids):
        self.connector_scheduler.request_finished(request.request_id)
        return False, None


__all__ = [
    "UCMConnector",
    "UCMConnectorMetadata",
    "UCMWorkerMetadata",
    "UCMRuntimeContext",
]
