import math

import torch

DEFAULT_BLOCK_SIZE = 128
MIN_TOPK_LEN = 32
MAX_TOPK_LEN = 48
MAX_BS = 256
SEG_PREFILL_THRESHOLD = 8400
CUDA_TOPK = False
PTOPK_PREFETCH_ENABLE = False
VLLM_CUDA_MEM_ALIGN_KV_CACHE = False
LOCAL_WINDOW_SZ = MIN_TOPK_LEN - 1


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def get_type_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def align_to_256bytes(extent: int, dtype: torch.dtype) -> int:
    dtype_szie = get_type_size(dtype)
    eles_per_256bytes = 256 // dtype_szie
    return round_up(extent, eles_per_256bytes)


def compute_topk_len(raw_seq_len):
    topk_len = int(raw_seq_len * 0.3)
    if topk_len < MIN_TOPK_LEN:
        topk_len = min(MIN_TOPK_LEN, raw_seq_len)
    elif topk_len > MAX_TOPK_LEN:
        topk_len = MAX_TOPK_LEN
    return topk_len
