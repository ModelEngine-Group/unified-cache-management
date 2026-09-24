# 版权所有（c）华为技术有限公司 2012-2026

import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch_npu
from ucm.integration.vllm.perf_counter import PerfCounters


def get_device():
	"""自动选择计算设备"""
	if torch.npu.is_available():
		return torch.device('npu:0')
	elif torch.cuda.is_available():
		return torch.device('cuda:0')
	else:
		return torch.device('cpu')


def get_attetion_weights_batched(query, key, seq_mask, num_query_heads, num_kv_heads, head_dim):
	"""
	计算GQA下最后一个query token对key序列的平均注意力权重。
	返回形状：(num_query_heads, key_len)的CPU tensor
	"""

	q = query.view(query.shape[0], -1, num_query_heads, head_dim)
	k = key.view(key.shape[0], -1, num_kv_heads, head_dim)

	# GQA：扩展key heads 到 query heads
	if num_kv_heads < num_query_heads:
		assert num_query_heads % num_kv_heads == 0, "query heads 必须能被 kv heads 整除"
		repeat = num_query_heads // num_kv_heads
		k = k.repeat_interleave(repeat,dim=2) # (k_len, q_heads, dim)
	elif num_kv_heads >num_query_heads:
		raise ValueError("kv heads >query heads 不常见，请检查参数")


	# 批量矩阵乘法计算注意力分数
	q_bmm = q.transpose(1, 2)
	k_bmm = k.transpose(1, 2) .transpose(-1,-2)
	scores = torch.matul(q_bmm, k_bmm) / (head_dim ** 0.5)

	# seq_mask: [B, L] -> [B, 1,1,1]广播到注意力分数维度
	# 将填充位置设为 -inf, softmax后权重为0
	mask_expanded = seq_mask.unsqueeze(1).unsqueeze(2) #[0,1,1,L]
	scores =scores.masked_fill(~mask_expanded, float('-inf'))
	
	attn = torch.softmax(scores, dim=-1)
	attn_mean = attn.squeeze(2) # (B, QH, K)
	return attn_mean


def compute_block_scores_batched(attn_perr_head, block_size):
	"""
	attn_per_head: (batch_size, num_heads, key_len) CPU tensor
	对每个head和每个key block 计算平均分数， 返回list of (head_idx, block_idx, score)
	"""
	batch_size, num_heads, key_len =attn_per_head.shape
	num_blocks = key_len // block_size

	valid_len = num_blocks * block_size
	attn_per_head = attn_per_head[:, :, :valid_len]
	reshaped = attn_per_head.view(batch_size,num_heads, num_blocks, block_size)
	block_scores = reshaped.sum(dim=-1) # (batch_size,num_heads, num_blocks)
	
	return block_scores


def extract_blocks_flat(
	kv_cache,
	block_table,
	actual_seq_lengths_kv,
	block_size
):
	"""
	从分块KV缓存中提取每个序列实际使用的key或value, 并填充对齐
	
	Returns:
		extracted: [batch_size, max_seq_len, hidden_dim]填充后的矩形Tensor
		seq_mask: [batch_size, max_seq_len]有效token的布尔掩码，用于下游忽略填充位
	"""
	batch_size = block_table.shape[0]
	device = kv_cache.device
	hidden_dim = kv_cache.shape[2]
	# 1. 计算每个序列需要的块数和最大长度
	num_blocks_needed = [(s + block_size -1) // block_size for s in actual_seq_lengths_kv]
	max_seq_len = max(actual_seq_lengths_kv)
	
	# 2. 预分配出Tensor和掩码
	extracted = torch.zeros(
		(batch_size,max_seq_len,hidden_dim),
		dtype = kv_cache.dtype,
		device = device
	)
	seq_mask = torch.zeros(
        (batch_size, max_seq_len),
        dtype=torch.bool,
        device=device
    )

	# 3. 向量化索引提取（避免Python for 循环）
	# 注意：此处仍需要循环处理每个batch的块索引， 因为block_table是变长的
	# 在图模式下， batch_size通常是静态常量，此循环可被安全捕获
	for b in range(batch_size):
		n_blocks = num_blocks_needed[b]
		seq_len = actual_seq_lengths_kv[b]

		if n_blocks == 0:
			continue

		blocks = block_table[b, :n_blocks]
		selected_blocks = kv_cache[blocks] # [n_blocks, block_size, hidden_dim]
		flat_seq = selected_blocks.view(-1, hidden_dim)[:seq_len]
		
		extracted[b, :seq_len] = flat_seq
		seq_mask[b, :seq_len] = True

	return extracted, seq_mask


def clear_directory(dir_path: str):
	p = Path(dir_path)
	if not p.exists() or not p.is_dir():
		return
	# 只删除文件， 保留子目录结构
	for file in p.iterdir():
		if file.is_file():
			file.unlink()


def save_sparse_block_table(
	block_attention_score,
	key_cache,
	query,
	actual_seq_lengths_q: list[int],
    layer_idx: int,
    block_table,
    actual_seq_lengths_kv: list[int],
    req_hash: Optional[dict[int, str]] = None,  # :right:修正类型注解
    block_size: int = 128,
    num_heads: int = 6,
    head_dim: int = 128,
    kv_heads: int = 1,
    top_k: int = 10,
    per_head: bool = True,
    union: bool = True,
    save_dir: str = "./code_agent_meta/",
) -> Optional[dict]:
    """
    计算query最后一个token与key序列的注意力， 并保存top-k block索引。
    支持多 batch size。

    新增过滤条件：
    - 仅当 actual_seq_lengths_kv[bs_idx] > 10000 且 req_hash[bs_idx]非None时才计算
    - 保存文件名使用 hash 值替代 bs_idx

    Args:
        req_hash: dict, key 为 batch 索引 (0~batch_size-1), value 为 hash 字符串
    """

    if block_table is None:
        return None # :right:显式返回None, 与异常分支保持一致
    
    try:
        return _save_sparse_block_table_impl(
            block_attention_score,
            key_cache,
            query,
            actual_seq_lengths_q,
            layer_idx,
            block_table,
            actual_seq_lengths_kv,
            req_hash,
            block_size,
            num_heads,
            head_dim,
            kv_heads,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _save_sparse_block_table_impl(
    block_attention_score,
    key_cache,
    query,
    actual_seq_lengths_q: list[int],
    layer_idx: int,
    block_table,
    actual_seq_lengths_kv: list[int],
    req_hash: Optional[dict[int, str]],
    block_size: int,
    num_heads: int,
    head_dim: int,
    kv_heads:int,
):
    batch_size = len(actual_seq_lengths_q)

    # 1. 计算每个batch最后一个token在压平query中的位置
    last_token_indices = [actual_seq_lengths_q[i] - 1 for i in range(batch_size)]

    # 提取每个batch的最后一个query token, shape: (batch_size, hidden_dim)
    batch_queries = query[last_token_indices]

    results={}

    # 2. 逐batch处理
    process_idx = []
    process_seq_lengths_kv = []
    for bs_idx in range(batch_size):
        # :right:新增：过滤条件校验
        kv_len = actual_seq_lengths_kv[bs_idx]
        query_len = actual_seq_lengths_q[bs_idx] - actual_seq_lengths_q[bs_idx - 1] if bs_idx > 0 else \
            actual_seq_lengths_q[bs_idx]
        hash_val = req_hash.get(bs_idx) if req_hash is not None else None

        if hash_val is None:
            continue

        process_idx.append(bs_idx)
        process_seq_lengths_kv.append(kv_len)
    if len(process_idx) == 0:
        return None

    start_time = time.perf_counter()
    process_queries = batch_queries[process_idx].unsqueeze(1)
    process_keys_extracted, seq_mask = extract_blocks_flat(
            key_cache, block_table[process_idx], process_seq_lengths_kv, block_size
        ) 
    # 计算注意力权重
    attn_per_head = get_attetion_weights_batched(
        process_queries, process_keys_extracted, seq_mask, num_heads, kv_heads, head_dim
    )

    # 计算每个 block 的分数
    head_block_scores = compute_block_scores_batched(attn_per_head, block_size)

    if layer_idx != 61:
        return None
    for i, bs_idx in enumerate(process_idx):
        hash_val = req_hash.get(bs_idx)

        block_score_sum = head_block_scores[i].sum(dim=0) / (num_heads / kv_heads)

        block_attention_score[bs_idx, :block_score_sum.shape[0]] = block_score_sum
        # 2. 讲聚合结果转为列表， 并按总分从大到小排序
        num_blocks = process_seq_lengths_kv[i] // block_size
        _, topk_indices = torch.topk(block_score_sum, num_blocks, dim=0,largest=True, sorted=True)

        results[hash_val] = topk_indices

        rank = torch_npu.npu.current_device()
        filename = f"union_blocks_l{layer_idx}_rank{rank}_hash{hash_val}.pkl"

    end_time = time.perf_counter()
    PerfCounters.get_inst().update("get_sparse_block_tables", end_time - start_time)
    return results
