import torch
import torch_npu
from vllm_ascend.compilation.acl_graph import get_graph_params


def update_attn_params(
    update_stream, forward_context, runtime_shape, kv_transfer_config=None
):
    graph_params = get_graph_params()
    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
            forward_context.attn_metadata,
            graph_params.attn_params[runtime_shape],
            graph_params.handles[runtime_shape],
            graph_params.events[runtime_shape],
        ):
            (
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                num_heads,
                scale,
                block_table,
                seq_lens,
                output,
            ) = param
            seq_lens = forward_context.attn_metadata[key].seq_lens

            workspace = torch_npu._npu_paged_attention_get_workspace(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=num_kv_heads,
                num_heads=num_heads,
                scale_value=scale,
                block_table=block_table,
                context_lens=seq_lens,
                out=output,
            )
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=num_kv_heads,
                num_heads=num_heads,
                scale_value=scale,
                block_table=block_table,
                context_lens=seq_lens,
                out=output,
                workspace=workspace,
            )
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)
