import math

import torch

DEFAULT_BLOCK_SIZE = 128
MIN_TOPK_LEN = 32
MAX_TOPK_LEN = 48
MAX_BS = 256
SEG_PREFILL_THRESHOLD = 8400
CUDA_TOPK = True #True for KVComp
PTOPK_PREFETCH_ENABLE = False
VLLM_CUDA_MEM_ALIGN_KV_CACHE = False
INIT_WINDOW_SZ = 1
NUM_PREFETCH_BLOCKS = 1
NUM_GSA_BLOCKS = 1

#KVComp utils
ENABLE_KVCOMP = True        #should be True for KVComp inside GSA
KVCOMP_PRESERVE_BLOCKS = 32 #number of blocks to preserve in KVComp for attention computation
RECENT_WINDOW_SZ = 4


class GSAConfig:
    def __init__(self):
        self.block_size = DEFAULT_BLOCK_SIZE
        self.init_windows_size = INIT_WINDOW_SZ
        self.recent_window_size = RECENT_WINDOW_SZ
        self.kvcomp_preserve_blocks = KVCOMP_PRESERVE_BLOCKS
        self.num_prefetch_blocks = NUM_PREFETCH_BLOCKS
        self.min_topk_len = MIN_TOPK_LEN
        self.max_topk_len = MAX_TOPK_LEN

    def set_config(self, block_szie):
        self.block_size = block_szie
        self.min_topk_len = math.ceil(MIN_TOPK_LEN * DEFAULT_BLOCK_SIZE / block_szie)
        self.max_topk_len = math.ceil(MAX_TOPK_LEN * DEFAULT_BLOCK_SIZE / block_szie)
        self.num_prefetch_blocks = math.ceil(
            NUM_PREFETCH_BLOCKS * DEFAULT_BLOCK_SIZE / block_szie
        )
        self.init_windows_size = math.ceil(
            INIT_WINDOW_SZ * DEFAULT_BLOCK_SIZE / block_szie
        )
        self.recent_window_size = math.ceil(
            RECENT_WINDOW_SZ * DEFAULT_BLOCK_SIZE / block_szie
        )
        self.kvcomp_preserve_blocks = math.ceil(
            KVCOMP_PRESERVE_BLOCKS * DEFAULT_BLOCK_SIZE / block_szie
        )
        self.num_gsa_blocks = math.ceil(
            NUM_GSA_BLOCKS * DEFAULT_BLOCK_SIZE / block_szie
        )

    # (NOTE) raw_seq_len is the number of blocks (not tokens) in the sequence)
    def compute_topk_len(self, raw_seq_len):
        if ENABLE_KVCOMP:
            topk_len = min(self.kvcomp_preserve_blocks, raw_seq_len)
        else:
            topk_len = math.ceil(raw_seq_len * 0.3)
        # topk_len = max(1, topk_len)
        if topk_len < self.min_topk_len:
            topk_len = min(self.min_topk_len, raw_seq_len)
        elif topk_len > self.max_topk_len:
            topk_len = self.max_topk_len
        return topk_len


gsa_config = GSAConfig()


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def get_type_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def align_to_256bytes(extent: int, dtype: torch.dtype) -> int:
    dtype_szie = get_type_size(dtype)
    eles_per_256bytes = 256 // dtype_szie
    return round_up(extent, eles_per_256bytes)
