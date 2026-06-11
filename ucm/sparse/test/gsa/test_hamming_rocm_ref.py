"""Cross-checked pass/fail test for the Hamming-distance scoring kernel.

The original test_cuda_hamming_{mla,gqa}.py drivers only print/benchmark the
kernel output. This test instead computes an independent CPU reference for the
paged (block-mode) Hamming score and asserts the GPU kernel matches it, giving
a deterministic correctness gate that is valid on both NVIDIA and ROCm. Run it
with one GPU visible, e.g. HIP_VISIBLE_DEVICES=0 / CUDA_VISIBLE_DEVICES=0.
"""

import os
import sys

import torch

try:
    from ucm.sparse.gsa_on_device.csrc.cuda.ham_dist import hamming
except ModuleNotFoundError:
    # Allow running against the freshly built extension without an editable
    # install: point HAMMING_DIR at the directory holding hamming*.so.
    sys.path.insert(0, os.environ.get("HAMMING_DIR", os.getcwd()))
    import hamming


def num_chunk_for(hd: int) -> int:
    return hd // 32


def popcount32(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int64) & 0xFFFFFFFF
    count = torch.zeros_like(x)
    for i in range(32):
        count += (x >> i) & 1
    return count


def reference_block_score(
    key_i32, query_i32, block_table, seqlen, max_seqlen, sink, recent, reduce_kvhead
):
    # key:   (num_blocks, block_size, num_kv_head, num_chunk) int32
    # query: (b, 1, num_head, num_chunk) int32
    #
    # Exact integer reference (popcount of key XOR query, summed over the GQA
    # group and over chunks). The kernel stores the score in fp16, so the
    # comparison below allows for fp16 rounding of these large integer sums.
    num_blocks, block_size, num_kv_head, num_chunk = key_i32.shape
    b = query_i32.shape[0]
    num_head = query_i32.shape[2]
    kv_group = num_head // num_kv_head

    key = key_i32.cpu()
    query = query_i32.cpu()
    bt = block_table.cpu()
    sl = seqlen.cpu()

    if reduce_kvhead:
        out = torch.full((b, 1, max_seqlen), float("inf"), dtype=torch.float32)
    else:
        out = torch.full((b, num_kv_head, max_seqlen), float("inf"), dtype=torch.float32)

    for bi in range(b):
        actual = int(sl[bi].item())
        for pos in range(max_seqlen):
            is_inf = pos >= actual
            is_sink_or_recent = (pos < sink) or (
                (pos >= actual - recent) and (pos < actual)
            )
            if is_inf:
                continue
            block_slot = pos // block_size
            offset = pos % block_size
            phys = int(bt[bi, block_slot].item())
            # The kernel reads the key block as (num_kv_head, block_size,
            # num_chunk): base = phys*(num_kv_head*block_size*num_chunk),
            # element (kv*block_size + offset)*num_chunk + chunk. Index a flat
            # view of the key tensor the same way so the reference matches the
            # kernel's layout regardless of the host tensor's nominal shape.
            block = key[phys].reshape(-1)
            per_kv = []
            for kv in range(num_kv_head):
                base = (kv * block_size + offset) * num_chunk
                k_chunks = block[base : base + num_chunk]
                s = 0
                for g in range(kv_group):
                    head = kv * kv_group + g
                    q_chunks = query[bi, 0, head, :]
                    s += int(popcount32(k_chunks ^ q_chunks).sum().item())
                per_kv.append(s)
            if reduce_kvhead:
                val = 0.0 if is_sink_or_recent else float(min(per_kv))
                out[bi, 0, pos] = val
            else:
                for kv in range(num_kv_head):
                    val = 0.0 if is_sink_or_recent else float(per_kv[kv])
                    out[bi, kv, pos] = val
    return out


def build_inputs(b, h, hk, hd, block_size, seqlen_list, seed=42):
    torch.manual_seed(seed)
    max_seqlen = max(seqlen_list)
    seqlen = torch.tensor(seqlen_list, dtype=torch.int32).cuda()
    num_blocks_per_seq = (seqlen + block_size - 1) // block_size
    num_blocks = int(num_blocks_per_seq.sum().item()) + 1
    max_num_block_per_seq = (max_seqlen + block_size - 1) // block_size
    max_seqlen = int(max_num_block_per_seq * block_size)

    block_table = torch.zeros((b, max_num_block_per_seq), dtype=torch.int32)
    start = 1
    for i, n in enumerate(num_blocks_per_seq):
        block_table[i, :n] = torch.arange(start, start + n, dtype=torch.int32)
        start += int(n)
    block_table = block_table.cuda()

    key = torch.randn(num_blocks, block_size, hk, hd // 32).to(torch.float32)
    query = torch.randn(b, 1, h, hd // 32).to(torch.float32)
    key = key.view(torch.int32).cuda()
    query = query.view(torch.int32).cuda()
    return key, query, block_table, seqlen, max_seqlen


def run_case(name, b, h, hk, hd, block_size, seqlen_list, sink, recent, reduce_kvhead):
    key, query, block_table, seqlen, max_seqlen = build_inputs(
        b, h, hk, hd, block_size, seqlen_list
    )
    out = hamming.hamming_score(
        key, query, block_table, seqlen, max_seqlen, sink, recent, reduce_kvhead
    )
    out_cpu = out.detach().float().cpu()
    ref = reference_block_score(
        key, query, block_table, seqlen, max_seqlen, sink, recent, reduce_kvhead
    )

    finite = torch.isfinite(ref)
    mismatch_inf = torch.isinf(ref) ^ torch.isinf(out_cpu)
    if mismatch_inf.any():
        print(f"[{name}] FAIL: inf mask mismatch at {int(mismatch_inf.sum())} positions")
        return False
    # The kernel sums the per-chunk integer popcounts into an fp16 accumulator,
    # so each finite score carries fp16 rounding (relative ~2^-10) accumulated
    # over the chunks. Compare with a relative tolerance plus a small absolute
    # floor; an order-of-magnitude error (a wrong popcount/index) is far outside
    # this band and would still fail.
    if finite.any():
        diff = (out_cpu[finite] - ref[finite]).abs()
        tol = (2 ** -10) * ref[finite].abs() * num_chunk_for(hd) + 1.0
        max_abs = diff.max().item()
        max_rel = (diff / (ref[finite].abs() + 1.0)).max().item()
        ok = bool((diff <= tol).all())
    else:
        max_abs = max_rel = 0.0
        ok = True
    print(
        f"[{name}] {'PASS' if ok else 'FAIL'}: max_abs_err={max_abs:.1f} "
        f"max_rel_err={max_rel:.2e} shape={tuple(out_cpu.shape)} "
        f"reduce_kvhead={reduce_kvhead}"
    )
    return ok


def main():
    torch.cuda.set_device(0)
    results = []
    # MLA-style: many heads, single kv head, no kv reduction.
    results.append(
        run_case("mla", b=2, h=128, hk=1, hd=576, block_size=64,
                 seqlen_list=[513, 320], sink=1, recent=1, reduce_kvhead=False)
    )
    # GQA-style: grouped heads, multiple kv heads, kv reduction (min over kv).
    results.append(
        run_case("gqa", b=3, h=32, hk=8, hd=128, block_size=128,
                 seqlen_list=[640, 512, 384], sink=1, recent=1, reduce_kvhead=True)
    )
    # Determinism: two runs must be bit-identical.
    key, query, bt, sl, ms = build_inputs(2, 128, 1, 576, 64, [513, 320])
    o1 = hamming.hamming_score(key, query, bt, sl, ms, 1, 1, False).detach().cpu()
    o2 = hamming.hamming_score(key, query, bt, sl, ms, 1, 1, False).detach().cpu()
    det = torch.equal(o1, o2)
    print(f"[determinism] {'PASS' if det else 'FAIL'}: two-run bit-identical={det}")
    results.append(det)

    if all(results):
        print("ALL HAMMING TESTS PASSED")
        return 0
    print("HAMMING TESTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
