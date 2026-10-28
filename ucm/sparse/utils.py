import ctypes
import math
import os
import sys
from multiprocessing import shared_memory

import torch

DEFAULT_BLOCK_SIZE = 128
######  GSA threshold for offload
SEG_PREFILL_THRESHOLD = 32768
INDEXER_OFFLOAD_TOKENS = 8192
INDEXER_OFFLOAD_BLOCKS = INDEXER_OFFLOAD_TOKENS // DEFAULT_BLOCK_SIZE
INDEXER_PRWEFETCH_TOKENS = 32768

###### GSA algorithm switch
PTOPK_PREFETCH_ENABLE = True
IS_OPEN_GSA = True

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
    def __init__(self):
        self.block_size = DEFAULT_BLOCK_SIZE
        self.init_windows_size = LOCAL_WINDOWS_LEN
        self.num_prefetch_blocks = 2048
        self.min_topk_len = 32
        self.max_topk_len = 48

    def set_config(self, block_szie):
        self.block_size = block_szie
        self.num_prefetch_blocks = 2048

    def compute_topk_len(self):
        return REMAIN_TOKENS


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
    SHM_NAME = "gsa_glm52_slot_dp" + str(dp_rank)
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        shm.unlink()
        print(f"✅ 成功清理: {SHM_NAME}")
    except FileNotFoundError:
        print(f"ℹ️ 共享内存 '{SHM_NAME}' 不存在或已被清理。")
    except PermissionError:
        print("❌ 权限不足，请在 Linux 使用当前用户，Windows 尝试管理员运行。")
    except Exception as e:
        print(f"❌ 清理失败: {e}")

    SHM_NAME = "indexer_glm52_slot_dp" + str(dp_rank)
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        shm.unlink()
        print(f"✅ 成功清理: {SHM_NAME}")
    except FileNotFoundError:
        print(f"ℹ️ 共享内存 '{SHM_NAME}' 不存在或已被清理。")
    except PermissionError:
        print("❌ 权限不足，请在 Linux 使用当前用户，Windows 尝试管理员运行。")
    except Exception as e:
        print(f"❌ 清理失败: {e}")

    SHM_NAME = "gsa_glm52_slots_num_dp" + str(dp_rank)
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        shm.unlink()
        print(f"✅ 成功清理: {SHM_NAME}")
    except FileNotFoundError:
        print(f"ℹ️ 共享内存 '{SHM_NAME}' 不存在或已被清理。")
    except PermissionError:
        print("❌ 权限不足，请在 Linux 使用当前用户，Windows 尝试管理员运行。")
    except Exception as e:
        print(f"❌ 清理失败: {e}")

    SHM_NAME = "prefetch_glm52_status_dp" + str(dp_rank)
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        shm.unlink()
        print(f"✅ 成功清理: {SHM_NAME}")
    except FileNotFoundError:
        print(f"ℹ️ 共享内存 '{SHM_NAME}' 不存在或已被清理。")
    except PermissionError:
        print("❌ 权限不足，请在 Linux 使用当前用户，Windows 尝试管理员运行。")
    except Exception as e:
        print(f"❌ 清理失败: {e}")
    SHM_NAME = "gsa_kv_pack_control_dp" + str(dp_rank)
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        shm.unlink()
        print(f"✅ 成功清理: {SHM_NAME}")
    except FileNotFoundError:
        print(f"ℹ️ 共享内存 '{SHM_NAME}' 不存在或已被清理。")
    except PermissionError:
        print("❌ 权限不足，请在 Linux 使用当前用户，Windows 尝试管理员运行。")
    except Exception as e:
        print(f"❌ 清理失败: {e}")


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
