"""Opt-in port of the GLM experiment's sparse MLA layer-load synchronization.

Wait before the sparse indexer reads restored KV data, and deduplicate layer
waits within a connector batch. Keep the original forward and cache-save path.
"""

from functools import wraps


def install():
    """Register the experiment-compatible hooks without importing vLLM early."""
    from wrapt import when_imported

    @when_imported("ucm.integration.vllm.ucm_connector")
    def patch_ucm(module):
        cls = module.UCMLayerWiseConnector
        # Retain the experiment's marker so an existing runtime hook and this
        # packaged hook can coexist without wrapping the same methods twice.
        if getattr(cls, "_glm52_sync_fix", False):
            return
        original_start = cls.start_load_kv
        original_wait = cls.wait_for_layer_load

        @wraps(original_start)
        def start(self, *args, **kwargs):
            self._glm52_waited_layers = set()
            return original_start(self, *args, **kwargs)

        @wraps(original_wait)
        def wait(self, layer_name):
            if not self._connector_metadata or not self.need_load:
                return
            layer_id = self.layer_name_to_id[layer_name]
            if layer_id in self._glm52_waited_layers:
                return
            result = original_wait(self, layer_name)
            self._glm52_waited_layers.add(layer_id)
            return result

        cls.start_load_kv = start
        cls.wait_for_layer_load = wait
        cls._glm52_sync_fix = True
        print(
            "GLM51_SYNC_FIX: per-batch layer waits deduplicated (GLM52/UCM08 port)",
            flush=True,
        )

    @when_imported("vllm.model_executor.layers.mla")
    def patch_mla(module):
        cls = module.MultiHeadLatentAttentionWrapper
        original = cls.forward
        if getattr(original, "_glm52_sync_fix", False):
            return

        @wraps(original)
        def forward(self, *args, **kwargs):
            if self.is_sparse and self.indexer is not None:
                from vllm.distributed.kv_transfer import (
                    get_kv_transfer_group,
                    has_kv_transfer_group,
                    is_v1_kv_transfer_group,
                )

                if has_kv_transfer_group() and is_v1_kv_transfer_group():
                    from vllm.forward_context import get_forward_context

                    from ucm.integration.vllm.ucm_connector import UCMLayerWiseConnector

                    connector = get_kv_transfer_group()
                    if (
                        isinstance(
                            getattr(connector, "connector", None), UCMLayerWiseConnector
                        )
                        and connector.has_connector_metadata()
                        and get_forward_context().attn_metadata is not None
                    ):
                        connector.wait_for_layer_load(self.mla_attn.layer_name)
            return original(self, *args, **kwargs)

        forward._glm52_sync_fix = True
        cls.forward = forward
        print(
            "GLM51_SYNC_FIX: wait before independent DSA indexer; original save retained",
            flush=True,
        )
