import torch

from ucm.sparse.kvcomp.ham_dist import hamming


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
):
    q_hash = q_hash.view(torch.int32)
    k_hash = k_hash.view(torch.int32)
    assert k_hash.shape[1] == 1
    # assert k_hash.shape[-1] == 18 and q_hash.shape[-1] == 18
    block_size = k_hash.shape[2]
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

    block_output = torch.min(
        output.view(output.shape[0], output.shape[-1] // block_size, block_size), dim=-1
    )[0]

    ind = torch.topk(block_output, k=(topk_token // block_size), dim=-1, largest=False)[
        1
    ]
    ind = torch.sort(ind, dim=-1, descending=False)[0]

    new_block_table = torch.gather(block_table, dim=-1, index=ind)
    return new_block_table


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
    block_size = k_hash.shape[2]
    assert topk_token % block_size == 0
    assert recent_token > 0 and topk_token > (sink_token + recent_token)
    max_seqlen = block_size * block_table.shape[1]

    new_block_table = block_table[:, : topk_token // block_size]
    return new_block_table
