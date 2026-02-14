import hamming
import torch

torch.cuda.set_device(1)
torch.manual_seed(42)


def op_hamming(key, query, block_table, seqlen, max_seqlen, sink, recent):
    output = hamming.hamming_score(
        key, query, block_table, seqlen, max_seqlen, sink, recent, True
    )
    return output


def block_topk(output, block_table, block_size, topk_tokens=4096):
    block_output = torch.min(
        output.view(output.shape[0], output.shape[-1] // block_size, block_size), dim=-1
    )[0]
    # print(f"block output shape: {block_output.shape}, dtype: {block_output.dtype}")
    k = topk_tokens // block_size

    ind = torch.topk(block_output, k=k, dim=-1, largest=False)[1]

    ind = torch.sort(ind, dim=-1, descending=False)[0]
    topk_block_table = torch.gather(block_table, dim=-1, index=ind)
    return topk_block_table, block_output


def time_cuda_event(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return out, ms


def summarize(times_ms, name):
    avg = sum(times_ms) / len(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    ts = sorted(times_ms)
    p50 = ts[len(ts) // 2]
    p90 = ts[int(len(ts) * 0.9) - 1] if len(ts) >= 10 else ts[-1]
    print(
        f"[{name}] {len(times_ms)} runs: avg={avg:.3f} ms, p50={p50:.3f} ms, p90={p90:.3f} ms, min={mn:.3f} ms, max={mx:.3f} ms"
    )


print(f"=========================data=========================")
b = 100  # batch_size
sq = 1  # seq_len_q
hd = 128  # head_dim
h = 32  # num_head
hk = 8  # num_kv_head
gqa = h // hk
block_size = 128
sink = 1
recent = 1
seqlen_list = [32769] * 100
max_seqlen = max(seqlen_list)

seqlen = torch.tensor(seqlen_list, dtype=torch.int32).cuda()
print(f"seqlen: {seqlen}")

num_blocks_per_seq = (seqlen + block_size - 1) // block_size
num_blocks = num_blocks_per_seq.sum().item()
print(f"num_blocks:{num_blocks}")

max_num_block_per_seq = (max_seqlen + block_size - 1) // block_size
max_seqlen = max_num_block_per_seq * block_size
print(f"max_seqlen: {max_seqlen}")

block_table = torch.zeros((b, max_num_block_per_seq), dtype=torch.int32)
start = 1
for i, n in enumerate(num_blocks_per_seq):
    block_table[i, :n] = torch.arange(start, start + n, dtype=torch.int32)
    start += n
block_table = block_table.cuda()
print(f"block_table: {block_table}")

key = torch.randn(num_blocks, block_size, hk, hd // 32).to(torch.float32)
query = torch.randn(b, sq, h, hd // 32).to(torch.float32)
key = key.view(torch.int32).cuda()
query = query.view(torch.int32).cuda()
print(f"key.shape: {key.shape}, key.dtype: {key.dtype}")
print(f"query.shape: {query.shape}, query.dtype: {query.dtype}")

topk_tokens = 4096
warmup = 2
exec_time = 5
# ------------------------- warmup -------------------------
print(f"\n=========================warmup ({warmup})=========================")
out = None
for _ in range(warmup):
    out = op_hamming(key, query, block_table, seqlen, max_seqlen, sink, recent)
    _ = block_topk(out, block_table, block_size)

torch.cuda.synchronize()
print("warmup done.")

# ------------------------- timed runs -------------------------
print(f"\n=========================timed runs ({exec_time})=========================")
t_hamming = []
t_topk = []
t_total = []

out_last = None
topk_last = None

for _ in range(exec_time):
    # 1) hamming kernel time
    out, ms1 = time_cuda_event(
        lambda: op_hamming(key, query, block_table, seqlen, max_seqlen, sink, recent)
    )
    out_last = out
    t_hamming.append(ms1)

    # 2) block_topk time (post-process)
    (topk_tbl, blk_score), ms2 = time_cuda_event(
        lambda: block_topk(out, block_table, block_size)
    )
    topk_last = topk_tbl
    t_topk.append(ms2)

    # 3) end-to-end time
    _, ms3 = time_cuda_event(
        lambda: block_topk(
            op_hamming(key, query, block_table, seqlen, max_seqlen, sink, recent),
            block_table,
            block_size,
        )
    )
    t_total.append(ms3)

summarize(t_hamming, "op_hamming")
summarize(t_topk, "block_topk")
summarize(t_total, "op_hamming + block_topk")

# ------------------------- print one sample output (avoid polluting timing) -------------------------
print("\n=========================sample output=========================")
print(f"output shape: {tuple(out_last.shape)}, dtype: {out_last.dtype}")
print(f"topk_block_table shape: {tuple(topk_last.shape)}, dtype: {topk_last.dtype}")
print(f"topk_block_table[0, :8]: {topk_last[0, :8].detach().cpu()}")
