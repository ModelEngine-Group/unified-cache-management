from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.forward_context import get_forward_context


def wait_for_kv_layer_from_connector(layer_name: str) -> None:
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()
    if (
        hasattr(connector, "has_connector_metadata")
        and not connector.has_connector_metadata()
    ):
        return

    attn_metadata = get_forward_context().attn_metadata
    if attn_metadata is None:
        return
    assert isinstance(attn_metadata, dict)
    connector.wait_for_layer_load(layer_name)


def maybe_save_kv_layer_to_connector(
    layer_name: str,
    kv_cache_layer: "list[object]",
) -> None:
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()
    if (
        hasattr(connector, "has_connector_metadata")
        and not connector.has_connector_metadata()
    ):
        return

    attn_metadata = get_forward_context().attn_metadata
    if attn_metadata is None:
        return
    assert isinstance(attn_metadata, dict)
    connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata[layer_name])
