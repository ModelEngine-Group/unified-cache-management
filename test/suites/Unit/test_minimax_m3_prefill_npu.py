"""Device regression for the #16121 backport; requires Ascend and its Triton."""

import importlib.util
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

pytestmark = [pytest.mark.stage(0), pytest.mark.platform("npu")]


@pytest.fixture(scope="module")
def backport():
    if version("vllm-ascend").split("+", 1)[0] != "0.26.0rc1":
        pytest.skip("backport targets vllm-ascend 0.26.0rc1")
    if not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    path = (
        Path(__file__).resolve().parents[3]
        / "ucm/integration/vllm/patch/v0260/vllm_ascend/minimax_m3_prefill.py"
    )
    spec = importlib.util.spec_from_file_location("m3_prefill_npu_backport", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize(
    "query_lengths,prefix_lengths,init_blocks,local_blocks",
    [
        ([1, 2, 3], [0, 127, 65512], 4, 8),
        ([127, 128, 129], [0, 127, 128], 4, 8),
        ([2048, 2049], [0, 128], 4, 8),
        ([0, 3, 0, 1, 0], [0, 256, 0, 127, 0], 0, 0),
        ([3, 2], [0, 1], 32, 64),
    ],
)
def test_prefill_score_prepare_topk_on_npu(
    backport, query_lengths, prefix_lengths, init_blocks, local_blocks
):
    device = "npu"
    heads, head_dim, topk = 2, 128, 32
    offsets = [0]
    for length in query_lengths:
        offsets.append(offsets[-1] + length)
    sequence_lengths = [q + p for q, p in zip(query_lengths, prefix_lengths)]
    max_seq_len = max(sequence_lengths)
    max_query_len = max(query_lengths)
    block_count = (max_seq_len + 127) // 128
    q = torch.randn(offsets[-1], heads, head_dim, dtype=torch.bfloat16, device=device)
    cache = torch.randn(block_count, 128, head_dim, dtype=torch.bfloat16, device=device)
    block_table = torch.arange(block_count, dtype=torch.int32, device=device)
    block_table = block_table.repeat(len(query_lengths), 1)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(sequence_lengths, dtype=torch.int32, device=device)
    prefix = torch.tensor(prefix_lengths, dtype=torch.int32, device=device)
    score = backport.minimax_m3_index_score(
        q,
        cache,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix,
        max_query_len,
        max_seq_len,
        heads,
    )
    torch.npu.synchronize()
    expected = score.cpu().clone()
    valid_counts = []
    for batch, length in enumerate(query_lengths):
        for query in range(length):
            token = offsets[batch] + query
            valid = (prefix_lengths[batch] + query + 128) // 128
            valid_counts.append(valid)
            assert torch.isfinite(expected[:, token, :valid]).all()
            assert torch.isneginf(expected[:, token, valid:]).all()
            expected[:, token, : min(init_blocks, valid)] = 1e30
            expected[:, token, max(valid - local_blocks, 0) : valid] = 1e29

    output = torch.full(
        (heads, offsets[-1] + 3, topk), -99, dtype=torch.int32, device=device
    )
    indices = backport.minimax_m3_index_topk(
        score,
        cu_seqlens,
        prefix,
        max_query_len,
        topk,
        init_blocks,
        local_blocks,
        out=output,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(score.cpu(), expected, rtol=0, atol=0)
    assert indices.data_ptr() == output.data_ptr()
    assert (output[:, offsets[-1] :].cpu() == -99).all()
    indices = indices.cpu().long()
    counts = torch.tensor(valid_counts)[None, :, None]
    valid_slots = torch.arange(topk)[None, None, :] < counts.clamp(max=topk)
    valid_slots = valid_slots.expand(heads, -1, -1)
    assert ((indices >= 0) == valid_slots).all()
    assert ((indices < counts) | ~valid_slots).all()
    selected_scores = torch.gather(expected, -1, indices.clamp(min=0))
    selected_scores.masked_fill_(~valid_slots, float("-inf"))
    selected_count = min(topk, expected.shape[-1])
    expected_scores = torch.full_like(selected_scores, float("-inf"))
    expected_scores[..., :selected_count] = torch.topk(
        expected, selected_count, dim=-1
    ).values
    torch.testing.assert_close(selected_scores, expected_scores, rtol=0, atol=0)
