# -*- coding: utf-8 -*-
"""gsav2 UT 套件的轻量假对象（fake objects）。

设计原则：
- FakeVllmConfig 只提供被测代码真实访问到的字段/方法，参数全部微型化以控制内存；
- 所有 fake 均为纯 Python，可在 CPU 与 NPU 环境无差别运行；
- C++ 扩展 mock（FakeEngineC / FakeUCMPrefetch）记录调用并支持可编程返回值。
"""
import types
from types import SimpleNamespace

import torch


# ---------------------------------------------------------------------------
# vllm 配置树
# ---------------------------------------------------------------------------
class FakeParallelConfig:
    def __init__(self, rank=0, dp_rank=0, tp_size=1):
        self.rank = rank
        self.data_parallel_rank = dp_rank
        self.tensor_parallel_size = tp_size


class FakeCacheConfig:
    def __init__(self, block_size=128):
        self.block_size = block_size


class FakeHfTextConfig:
    def __init__(self):
        self.kv_lora_rank = 16
        self.qk_rope_head_dim = 16


class FakeHfConfig:
    """init_indexer_layer_map 会读取 indexer_types 字段"""

    def __init__(self, indexer_types=None):
        self.indexer_types = indexer_types


class FakeModelConfig:
    def __init__(self):
        self.use_mla = True
        self.max_model_len = 2048
        self.dtype = torch.float16
        self.hf_text_config = FakeHfTextConfig()
        self.hf_config = FakeHfConfig(indexer_types=None)

    def get_num_layers(self, parallel_config):
        return 2

    def get_head_size(self):
        return 16

    def get_num_kv_heads(self, parallel_config):
        return 1


class FakeSchedulerConfig:
    def __init__(self):
        self.max_num_seqs = 4
        self.max_num_batched_tokens = 256


class FakeDeviceConfig:
    def __init__(self, device_type="cpu"):
        self.device_type = device_type


class FakeKVTransferConfig:
    """默认为 PD 分离 decode 节点：仅 consumer"""

    def __init__(self, consumer=True, producer=False):
        self.is_kv_consumer = consumer
        self.is_kv_producer = producer


class FakeSpeculativeConfig:
    def __init__(self, num_spec_tokens=3):
        self.num_speculative_tokens = num_spec_tokens


class FakeCompilationConfig:
    def __init__(self):
        self.cudagraph_capture_sizes = [2, 4]


class FakeVllmConfig:
    """微型 VllmConfig：layer_num=2, max_model_len=2048, max_bs=4, block=128"""

    def __init__(
        self,
        rank=0,
        dp_rank=0,
        tp_size=1,
        block_size=128,
        consumer=True,
        producer=False,
        speculative=None,
        device_type="cpu",
        additional_config=None,
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        indexer_types=None,
        dtype=torch.float16,
    ):
        self.parallel_config = FakeParallelConfig(rank, dp_rank, tp_size)
        self.cache_config = FakeCacheConfig(block_size)
        self.model_config = FakeModelConfig()
        self.model_config.max_model_len = max_model_len
        self.model_config.dtype = dtype
        self.model_config.hf_config = FakeHfConfig(indexer_types)
        self.scheduler_config = FakeSchedulerConfig()
        self.scheduler_config.max_num_seqs = max_num_seqs
        self.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
        self.device_config = FakeDeviceConfig(device_type)
        self.kv_transfer_config = FakeKVTransferConfig(consumer, producer)
        self.speculative_config = speculative
        self.compilation_config = FakeCompilationConfig()
        self.additional_config = {} if additional_config is None else additional_config


# ---------------------------------------------------------------------------
# vllm 运行时对象
# ---------------------------------------------------------------------------
class _FakeRequestStatus:
    PREEMPTED = "preempted"
    RUNNING = "running"


class FakeRequest:
    """调度侧 Request（estimate_num_slots_sparsed / allocate_slots 用）"""

    def __init__(
        self,
        request_id=1,
        num_output_tokens=0,
        num_prompt_tokens=10,
        prompt_token_ids=None,
        status=None,
    ):
        self.request_id = request_id
        self.num_output_tokens = num_output_tokens
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_token_ids = (
            list(range(num_prompt_tokens))
            if prompt_token_ids is None
            else prompt_token_ids
        )
        self.status = _FakeRequestStatus.RUNNING if status is None else status
        self.cache_salt = None


class FakeAddReqState:
    """GSAMetaData.get_model_input / add_req_new 使用的请求状态"""

    def __init__(
        self,
        req_id=1,
        block_ids=None,
        prompt_token_ids=None,
        output_token_ids=None,
        num_computed_tokens=0,
    ):
        self.req_id = req_id
        self.block_ids = [[1, 2, 3]] if block_ids is None else block_ids
        self.prompt_token_ids = (
            list(range(200)) if prompt_token_ids is None else prompt_token_ids
        )
        self.output_token_ids = [] if output_token_ids is None else output_token_ids
        self.num_computed_tokens = num_computed_tokens


class FakeSchedulerOutput:
    def __init__(
        self,
        num_scheduled_tokens=None,
        cached_req_ids=None,
        resumed_req_ids=None,
        new_reqs=None,
        preempted_req_ids=None,
    ):
        self.num_scheduled_tokens = num_scheduled_tokens or {}
        self.scheduled_cached_reqs = SimpleNamespace(
            req_ids=cached_req_ids or [], resumed_req_ids=resumed_req_ids or []
        )
        self.scheduled_new_reqs = new_reqs or []
        self.preempted_req_ids = preempted_req_ids or []


class FakeInputBatch:
    def __init__(self, req_id_to_index=None, req_ids=None):
        self.req_id_to_index = req_id_to_index or {}
        self.req_ids = list(req_id_to_index.keys()) if req_ids is None else req_ids


# ---------------------------------------------------------------------------
# C++ 扩展 mock
# ---------------------------------------------------------------------------
class FakeEngineC:
    """gsa_prefetch.GSAPrefetchEngineC 的 mock。

    - 所有调用记录到 self.calls
    - get_prefetch_status / get_indexer_status 支持脚本化序列：
      prefetch_status=[True, False] 表示第一次返回 True，之后返回 False
    - 未显式定义的方法经 __getattr__ 兜底：记录调用并返回 None
    """

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls = []
        self.prefetch_status = [False]
        self.indexer_status = [False]
        self.store_v1_ptr = 0
        self.self_ptr = 0x1234
        self.topk_indices_result = []

    # -- 状态查询（脚本化） --
    @staticmethod
    def _scripted(seq):
        if len(seq) > 1:
            return seq.pop(0)
        return seq[0] if seq else False

    def get_prefetch_status(self):
        self.calls.append(("get_prefetch_status", (), {}))
        return self._scripted(self.prefetch_status)

    def get_indexer_status(self):
        self.calls.append(("get_indexer_status", (), {}))
        return self._scripted(self.indexer_status)

    def get_store_v1_ptr(self):
        self.calls.append(("get_store_v1_ptr", (), {}))
        return self.store_v1_ptr

    def get_self_ptr(self):
        self.calls.append(("get_self_ptr", (), {}))
        return self.self_ptr

    def get_topk_indices(self, layer_id, req_id):
        self.calls.append(("get_topk_indices", (layer_id, req_id), {}))
        return self.topk_indices_result

    def called(self, name):
        return [c for c in self.calls if c[0] == name]

    def __getattr__(self, name):
        # 仅在常规属性查找失败时触发；双下划线保留原语义
        if name.startswith("__"):
            raise AttributeError(name)

        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None

        return _record


class FakeUCMPrefetch:
    """gsa_prefetch.UCMKVPrefetch 的 mock：ucm_lookup 支持脚本化重试序列"""

    def __init__(self):
        self.calls = []
        self.lookup_results = None  # e.g. [[False]*4, [True]*4]

    def store_init(self, ptr, layer_num):
        self.calls.append(("store_init", ptr, layer_num))

    def ucm_lookup(self, ids):
        self.calls.append(("ucm_lookup", len(ids)))
        if self.lookup_results is not None and len(self.lookup_results) > 1:
            return self.lookup_results.pop(0)
        if self.lookup_results:
            return self.lookup_results[0]
        return [True] * len(ids)

    def ucm_load_ssd(self, ids):
        self.calls.append(("ucm_load_ssd", len(ids)))
        return True


# ---------------------------------------------------------------------------
# 共享内存类的进程内替身（GSA 用例中替换真实 shm，真 shm 由独立文件覆盖）
# ---------------------------------------------------------------------------
class FakeSharedBlockTable:
    def __init__(self, name, is_create=False):
        self.name = name
        self.is_creator = is_create
        self.written = None
        self.read_result = None  # None → 走 "if add_blocks is not None" 的 False 侧
        self.read_lens = None

    def write_add_block(self, add_blocks):
        self.written = add_blocks

    def read_add_block(self, add_block_lens):
        self.read_lens = add_block_lens
        return self.read_result

    def close(self):
        pass

    def unlink(self):
        pass


class FakeSharedTensorVersion:
    def __init__(self, name, rank=0, tp_size=16, is_create_version=False):
        self.name = name
        self.rank = rank
        self.tp_size = tp_size
        self.is_creator = is_create_version
        self.status_calls = []
        self.version_calls = []
        self.version_same = True

    def update_prefetch_status(self, is_done, version):
        self.status_calls.append((is_done, version))

    def update_version(self, version):
        self.version_calls.append(version)

    def is_version_same(self, version):
        return self.version_same

    def close(self):
        pass

    def unlink(self):
        pass


class FakeSharedTensorManager:
    def __init__(self, name, is_create=False, i_num=0, k_num=0):
        self.name = name
        self.is_creator = is_create
        self.init_nums = (i_num, k_num)
        self.update_calls = []
        self.read_nums = (100, 100)

    def update_num(self, i_num, k_num):
        self.update_calls.append((i_num, k_num))

    def read_num(self):
        return self.read_nums


# ---------------------------------------------------------------------------
# vllm_ascend.CpuAlloc 假实现（bind_cpus 三条路径）
# ---------------------------------------------------------------------------
class FakeCpuAlloc:
    # 类级配置，测试按需覆盖
    cpu_pool = {0: [3]}
    cpu_node = {3: 1}
    assign_main = {0: list(range(30))}

    def __init__(self, local_rank):
        self.local_rank = local_rank
        self.device_info = SimpleNamespace(running_npu_list=[0])
        self.npu_cpu_pool = FakeCpuAlloc.cpu_pool
        self.cpu_node = FakeCpuAlloc.cpu_node
        self.assign_main = FakeCpuAlloc.assign_main

    def build_cpu_pools(self):
        pass

    def allocate(self):
        pass


# ---------------------------------------------------------------------------
# kv transfer group / connector 假对象（store 注入路径）
# ---------------------------------------------------------------------------
class FakeStore:
    def __init__(self, ptr=4096):
        self._ptr = ptr

    def cc_store(self):
        return self._ptr


def make_plain_connector(
    with_store=True, store_ptr=4096, with_inner=False, inner_with_store=True
):
    """普通 connector：可选 store / 可选 .connector 二级"""
    obj = SimpleNamespace()
    if with_store:
        obj.store = FakeStore(store_ptr)
    if with_inner:
        obj.connector = make_plain_connector(
            with_store=inner_with_store, store_ptr=store_ptr
        )
    return obj


def make_multi_connector(variant="direct_store"):
    """AscendMultiConnector：名字匹配 _find_store_connector 的类型判断。

    使用独立类而非修改 SimpleNamespace 的 __name__（那会污染全局类名，
    导致同进程中其他 SimpleNamespace 假对象也被误判为 multi connector）。
    """
    if variant == "no_connectors":
        return AscendMultiConnector()  # 无 _connectors 属性
    inner = make_plain_connector(
        with_store=False, with_inner=True, inner_with_store=(variant == "inner_store")
    )
    return AscendMultiConnector(connectors=[inner])


def fake_kv_group_with(connector):
    group = SimpleNamespace()
    group.connector = connector  # group.connector → connector
    return group


class AscendMultiConnector:
    """vllm_ascend PD 分离多 connector 的假实现。

    用独立类型而非 SimpleNamespace + 改 __name__：后者会污染全局类名，
    使同进程中所有 SimpleNamespace 实例都被 `_find_store_connector`
    误判为 AscendMultiConnector。
    """

    def __init__(self, connectors=None):
        if connectors is not None:
            self._connectors = connectors
