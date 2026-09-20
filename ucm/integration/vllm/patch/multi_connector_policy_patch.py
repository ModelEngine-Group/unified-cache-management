"""Propagate the top-level kv_load_failure_policy into each MultiConnector child.

vLLM's ``MultiConnector._get_connector_classes_and_configs`` rebuilds each
child ``VllmConfig`` from only the child config dict::

    temp_config = copy.copy(vllm_config)
    temp_config.kv_transfer_config = KVTransferConfig(**ktc, engine_id=engine_id)

So fields set only on the outer (top-level) KV transfer config are never
inherited by UCM child connectors. In particular ``kv_load_failure_policy``
falls back to its upstream default "fail" even when the user requested
"recompute" on the top-level config.

UCM's hybrid/multi-group connectors (``UCMFAWAConnector`` /
``UCMHybridLinearAttentionConnector``) do not implement the recompute path, so
the mismatch would silently go unnoticed. This patch forwards the top-level
policy into each child config (using ``setdefault`` so an explicit per-child
setting still wins), letting the HMA/HLA validation reject the unsupported
combination up front.
"""

from copy import copy
from typing import Any

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _propagate_load_failure_policy_to_children(
    vllm_config: "Any", ktcs: list[dict]
) -> list[dict]:
    """Return the child configs with the top-level policy forwarded to each."""
    top_policy = getattr(
        vllm_config.kv_transfer_config, "kv_load_failure_policy", "fail"
    )
    forwarded: list[dict] = []
    for ktc in ktcs:
        child_ktc = dict(ktc)
        child_ktc.setdefault("kv_load_failure_policy", top_policy)
        forwarded.append(child_ktc)
    return forwarded


def patch_multi_connector_policy(mod: Any) -> None:
    if getattr(mod, "_ucm_multi_connector_policy_patch", False):
        return

    original = getattr(mod.MultiConnector, "_get_connector_classes_and_configs", None)
    if original is None:
        # Module layout mismatch between vLLM versions; leave untouched.
        return

    @classmethod
    def wrapped(cls, vllm_config):
        ktcs = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "connectors"
        )
        assert ktcs is not None
        from vllm.config.kv_transfer import KVTransferConfig
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

        ret = []
        for ktc in _propagate_load_failure_policy_to_children(vllm_config, ktcs):
            temp_config = copy(vllm_config)
            engine_id = ktc.get("engine_id", vllm_config.kv_transfer_config.engine_id)
            temp_config.kv_transfer_config = KVTransferConfig(
                **ktc, engine_id=engine_id
            )
            ret.append(
                (
                    KVConnectorFactory.get_connector_class(
                        temp_config.kv_transfer_config
                    ),
                    temp_config,
                )
            )
        return ret

    mod.MultiConnector._get_connector_classes_and_configs = wrapped
    mod._ucm_multi_connector_policy_patch = True
    logger.debug(
        "Patched MultiConnector to propagate the top-level "
        "kv_load_failure_policy to each child connector"
    )


when_imported("vllm.distributed.kv_transfer.kv_connector.v1.multi_connector")(
    patch_multi_connector_policy
)
