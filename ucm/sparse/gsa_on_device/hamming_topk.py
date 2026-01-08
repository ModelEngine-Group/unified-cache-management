import torch

if hasattr(torch, "cuda") and torch.cuda.is_available():
    from ucm.sparse.gsa_on_device.ham_dist import hamming


@torch.compile()
def update_seq_lens(seq_lens, topk_token, block_size):
    drop_block_num = (
        (seq_lens - topk_token).clip(min=0) + block_size - 1
    ) // block_size
    seq_lens = seq_lens - drop_block_num * block_size
    return seq_lens


def cuda_hamming_topk(
    q_hash,
    k_hash,
    block_table,
    seq_lens,
    topk_token,
    sink_token,
    recent_token,
    is_mla,
):
    q_hash = q_hash.view(torch.int32)
    k_hash = k_hash.view(torch.int32)
    # assert k_hash.shape[1] == 1
    # assert k_hash.shape[-1] == 18 and q_hash.shape[-1] == 18
    block_size = k_hash.shape[1]
    assert topk_token % block_size == 0
    assert recent_token > 0 and topk_token > (sink_token + recent_token)
    max_seqlen = block_size * block_table.shape[1]
    output = hamming.hamming_score(
        k_hash,
        q_hash,
        block_table,
        seq_lens,
        max_seqlen,
        sink_token,
        recent_token,
    )

    k_blocks = topk_token // block_size
    B, Hk, S = output.shape
    num_blocks = S // block_size

    # block_output: [B, Hk, num_blocks]
    block_output = output.view(B, Hk, num_blocks, block_size).amin(dim=-1)

    if is_mla:
        block_score = block_output[:, 0, :]
        ind = torch.topk(block_score, k=k_blocks, dim=-1, largest=False).indices
        ind = ind.sort(dim=-1).values
        return torch.gather(block_table, dim=-1, index=ind)

    block_score = block_output.amin(dim=1)  # [B, num_blocks]
    ind = torch.topk(block_score, k=k_blocks, dim=-1, largest=False).indices
    ind = ind.sort(dim=-1).values
    return torch.gather(block_table, dim=-1, index=ind)


def fake_hamming_topk(
    q_hash,
    k_hash,
    block_table,
    seq_lens,
    topk_token,
    sink_token,
    recent_token,
):
    q_hash = q_hash.view(torch.int32)
    k_hash = k_hash.view(torch.int32)
    assert k_hash.shape[1] == 1
    assert k_hash.shape[-1] == 18 and q_hash.shape[-1] == 18
    block_size = k_hash.shape[1]
    assert topk_token % block_size == 0
    assert recent_token > 0 and topk_token > (sink_token + recent_token)
    max_seqlen = block_size * block_table.shape[1]

    new_block_table = block_table[:, : topk_token // block_size]
    return new_block_table
