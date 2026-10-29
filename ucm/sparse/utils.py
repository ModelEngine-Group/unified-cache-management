import ctypes
import math
import os
import sys
from multiprocessing import shared_memory

import torch

from ucm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_BLOCK_SIZE = 128


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from the environment, falling back to ``default``."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


###### Legacy GSA (ucm/sparse/gsa) shared defaults.
# Restored to their pre-GSA-v2 values: the old gsa chain
# (ucm/sparse/gsa/prefetch/prefetch_engine.py) still imports these.
MIN_TOPK_LEN = 32
MAX_TOPK_LEN = 48
MAX_BS = 256
SEG_PREFILL_THRESHOLD = 8400
CUDA_TOPK = False
VLLM_CUDA_MEM_ALIGN_KV_CACHE = False
INIT_WINDOW_SZ = 1
NUM_PREFETCH_BLOCKS = 1
NUM_GSA_BLOCKS = 1

###### GSA algorithm switch
# GSA / GSAPD are opt-in: they are only initialized when this switch is enabled
# (e.g. via env var UCM_SPARSE_GSA_ENABLE=1), so non-GSA deployments are not
# affected. Other sparse methods (ESA / KVStar / GSAOnDevice / Blend) are driven
# by their own kv_transfer_config and do not consume this switch.
IS_OPEN_GSA = _env_bool("UCM_SPARSE_GSA_ENABLE", False)
# Topk-prefetch switch consumed by GSA v2 and the vLLM integration patches.
# Configurable via env var, defaulting to True for GSA v2.
PTOPK_PREFETCH_ENABLE = _env_bool("UCM_GSA_PTOPK_PREFETCH_ENABLE", True)

######  GSA v2 threshold for offload
INDEXER_OFFLOAD_TOKENS = 8192
INDEXER_OFFLOAD_BLOCKS = INDEXER_OFFLOAD_TOKENS // DEFAULT_BLOCK_SIZE
INDEXER_PREFETCH_TOKENS = 32768

###### GSA offload params
REMAIN_TOKENS = 3072
SPA_TOKENS_LEN = 5120
PREFETCH_SPACE = 0
WARMUP_TOPK_LEN = 16
GSA_SPARSE_LEN = SPA_TOKENS_LEN + REMAIN_TOKENS
KVCACHE_BUFFER_SIZE = 16
STREAM_NUM = 1
PREFETCH_THREAD_NUM = 16
DEFAULT_TOPK_LEN = 2048
IS_SHARED_MEMCPY = True
PREFETCH_TYPE = 0
MAX_INDEXER_BLOCK_NUM = 13600
MAX_SLOTS_BLOCK_NUM = 12000
INIT_LOCAL_WINDOWS = 5
LOCAL_WINDOWS_LEN = 1
IS_INDEXER_OFFLOAD = False
INDEXER_SPARSE_RATE = 0.2
INDEXER_PREFETCH_RATIO = 1.5

###### vllm version
VLLM_ASCEND_VERSION = "0.18"
IS_PERFORMANCE_LOG = True


class DSAInputStruct:
    def __init__(self, shape, device, dtype):
        self.cpu = torch.zeros(
            shape, dtype=dtype, device="cpu", pin_memory=True
        ).contiguous()
        self.cpu.fill_(-1)
        self.npu = torch.zeros_like(self.cpu, device=device)
        if dtype != torch.bfloat16:
            self.np = self.cpu.numpy()

    def copy_to_npu(self, n):
        if n == 0:
            return
        if n == -1:
            self.npu.copy_(self.cpu, non_blocking=True)
        else:
            self.npu[:n].copy_(self.cpu[:n], non_blocking=True)

    def copy_to_cpu(self, n):
        if n == 0:
            return
        if n == -1:
            self.cpu.copy_(self.npu)
        else:
            self.cpu[:n].copy_(self.npu[:n])


class TopkIndices:
    def __init__(self, shape, device, dtype):
        self.layer_num = shape[0]
        self.cpu = torch.zeros(
            shape, dtype=dtype, device="cpu", pin_memory=True
        ).contiguous()
        self.cpu.fill_(-1)
        self.npu = torch.zeros_like(self.cpu, device=device)
        if dtype != torch.bfloat16:
            self.np = self.cpu.numpy()
        self.slice = list(self.npu.unbind(dim=0))

    def copy_to_npu(self, n):
        if n == 0:
            return
        if n == -1:
            self.npu.copy_(self.cpu, non_blocking=True)
        else:
            self.npu[:, :n].copy_(self.cpu[:, :n], non_blocking=True)

    def copy_to_cpu(self, n):
        if n == 0:
            return
        if n == -1:
            self.cpu.copy_(self.npu)
        else:
            self.cpu[:, :n].copy_(self.npu[:, :n])

    def copy_to_npu_slice(self, n):
        if n == 0:
            return
        self.npu[:, :n].copy_(self.cpu[:, :n])
        self.slice = list(self.npu.unbind(dim=0))


class GSAConfig:
    """Legacy GSA configuration, kept for the old ``ucm/sparse/gsa`` chain.

    Sizes scale with ``block_size`` relative to ``DEFAULT_BLOCK_SIZE`` so the
    configuration keeps working when the KV cache block size differs from 128.
    """

    def __init__(self):
        self.block_size = DEFAULT_BLOCK_SIZE
        self.init_windows_size = INIT_WINDOW_SZ
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
        self.num_gsa_blocks = math.ceil(
            NUM_GSA_BLOCKS * DEFAULT_BLOCK_SIZE / block_szie
        )

    def compute_topk_len(self, raw_seq_len):
        topk_len = math.ceil(raw_seq_len * 0.3)
        if topk_len < self.min_topk_len:
            topk_len = min(self.min_topk_len, raw_seq_len)
        elif topk_len > self.max_topk_len:
            topk_len = self.max_topk_len
        return topk_len


gsa_config = GSAConfig()


def init_indexer_layer_map(vllm_config):
    topk_layer_info = []
    topk_layer_map = []
    indexer_types = None
    indexr_num = -1
    layer_num = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    hf_config = vllm_config.model_config.hf_config
    hf_text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    config_candidates = (hf_config, hf_text_config)
    for config in config_candidates:
        if config is None:
            continue
        indexer_types = getattr(config, "indexer_types", None)
        if indexer_types is not None:
            break

    if indexer_types is None:
        topk_layer_info = list(range(layer_num))
        topk_layer_map = list(range(layer_num))
        indexr_num = layer_num
        return topk_layer_info, topk_layer_map, indexr_num
    else:
        for index, itype in enumerate(indexer_types):
            if itype == "full":
                indexr_num += 1
                topk_layer_map.append(indexr_num)
                topk_layer_info.append(index)
            else:
                last_id = topk_layer_info[-1]
                topk_layer_info.append(last_id)
                topk_layer_map.append(indexr_num)
        indexr_num += 1
        return topk_layer_info, topk_layer_map, indexr_num


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def get_type_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def align_to_256bytes(extent: int, dtype: torch.dtype) -> int:
    dtype_szie = get_type_size(dtype)
    eles_per_256bytes = 256 // dtype_szie
    return round_up(extent, eles_per_256bytes)


def rm_shared_memory(dp_rank):
    """Clean up the shared memory segments created by GSA for a given DP rank."""
    shm_names = [
        f"gsa_glm52_slot_dp{dp_rank}",
        f"indexer_glm52_slot_dp{dp_rank}",
        f"gsa_glm52_slots_num_dp{dp_rank}",
        f"prefetch_glm52_status_dp{dp_rank}",
        f"gsa_kv_pack_control_dp{dp_rank}",
    ]
    for name in shm_names:
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.unlink()
            shm.close()
            logger.info("Successfully cleaned shared memory: %s", name)
        except FileNotFoundError:
            logger.info(
                "Shared memory '%s' does not exist or was already cleaned.", name
            )
        except PermissionError:
            logger.warning(
                "Permission denied while cleaning shared memory '%s'; on Linux run "
                "as the current user, on Windows try as administrator.",
                name,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to clean shared memory '%s': %s", name, e)


def bind_thread_memory_policy(node_id: int):
    """使用 libnuma 设置当前线程后续分配内存的 NUMA 策略"""
    libnuma = ctypes.CDLL("libnuma.so.1")

    libnuma.numa_allocate_nodemask.restype = ctypes.c_void_p
    libnuma.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libnuma.numa_set_membind.argtypes = [ctypes.c_void_p]
    libnuma.numa_bitmask_free.argtypes = [ctypes.c_void_p]

    mask = libnuma.numa_allocate_nodemask()
    libnuma.numa_bitmask_setbit(mask, node_id)
    libnuma.numa_set_membind(mask)
    libnuma.numa_bitmask_free(mask)

    print(f"✅ 当前线程内存分配策略已成功绑定至 NUMA Node {node_id}")
