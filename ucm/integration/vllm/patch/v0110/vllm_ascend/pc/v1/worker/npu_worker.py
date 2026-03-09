import copy
from typing import Optional, Union

import torch
import torch.nn as nn
import torch_npu
import vllm.envs as envs_vllm
import vllm_ascend.envs as envs_ascend
from torch._dynamo.variables import TorchInGraphFunctionVariable
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
from torch_npu.profiler import dynamic_profile as dp
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger
from vllm.lora.request import LoRARequest
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, GiB_bytes
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.worker_base import WorkerBase
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.cpu_binding import bind_cpus
from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.spec_decode.interface import SpecDcodeType


class NPUWorker(WorkerBase):
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[Union[ModelRunnerOutput, AsyncModelRunnerOutput]]:
        if envs_ascend.MSMONITOR_USE_DAEMON:
            dp.step()

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        if forward_pass and not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group())
            )

        output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput)):
            return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.distributed_executor_backend != ("external_launcher")
            and not get_pp_group().is_last_rank
        )

        get_pp_group().send_tensor_dict(output.tensors, all_gather_group=get_tp_group())

        kv_connector_output = output.kv_connector_output
        if not kv_connector_output:
            return None

        # In case of PP with kv transfer, we need to pass through the
        # kv_connector_output
        if kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output
