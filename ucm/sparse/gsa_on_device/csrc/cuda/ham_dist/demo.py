import hamming
import torch

torch.cuda.set_device(0)
torch.manual_seed(42)

print(f"=========================data=========================")
b = 2  # batch_size
sq = 1  # seq_len_q
hd = 576  # head_dim
h = 128  # num_head
hk = 1  # num_kv_head
gqa = h // hk
block_size = 64
sink = 1
recent = 1
seqlen_list = [32769, 24575]
max_seqlen = max(seqlen_list)

seqlen = torch.tensor(seqlen_list, dtype=torch.int32).cuda()
print(f"seqlen: {seqlen}")

num_blocks_per_seq = (seqlen + block_size - 1) // block_size
num_blocks = num_blocks_per_seq.sum().item() + 1
# print(f'num_blocks:{num_blocks}')

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

key = torch.randn(num_blocks, hk, block_size, hd // 32).to(torch.float32)
query = torch.randn(b, sq, h, hd // 32).to(torch.float32)
key = key.view(torch.int32).cuda()
query = query.view(torch.int32).cuda()
print(f"key.shape: {key.shape}, key.dtype: {key.dtype}")
print(f"query.shape: {query.shape}, query.dtype: {query.dtype}")

print(f"=========================op_hamming=========================")

output = hamming.hamming_score(
    key, query, block_table, seqlen, max_seqlen, sink, recent
)

print(f"output shape: {output.shape}, dtype: {output.dtype}")
print(f"output: {output}")

print(f"=========================block topk=========================")

block_output = torch.min(
    output.view(output.shape[0], output.shape[-1] // block_size, block_size), dim=-1
)[0]
print(f"block output shape: {block_output.shape}, dtype: {block_output.dtype}")
print(f"block output: {block_output}")

k = 2048 // block_size
ind = torch.topk(block_output, k=k, dim=-1, largest=False)[1]
print(f"topk ind: {ind}")
ind = torch.sort(ind, dim=-1, descending=False)[0]
topk_block_table = torch.gather(block_table, dim=-1, index=ind)
print(f"topk block_table: {topk_block_table}")
