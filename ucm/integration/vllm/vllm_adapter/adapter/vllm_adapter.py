import importlib

import vllm.v1.outputs as default_outputs

from ucm.integration.vllm.vllm_adapter.adapter.utils import (
    patch_dataclass_fields,
)
from ucm.integration.vllm.vllm_adapter.v1.outputs import ModelRunnerOutput

patch_dataclass_fields(default_outputs.ModelRunnerOutput, ModelRunnerOutput)

from vllm.v1.core.sched import scheduler

from ucm.integration.vllm.vllm_adapter.v1.core.sched.scheduler import Scheduler

scheduler.Scheduler.update_from_output = Scheduler.update_from_output
scheduler.Scheduler.__init__ = Scheduler.__init__
scheduler.Scheduler.schedule = Scheduler.schedule
scheduler.Scheduler.add_request = Scheduler.add_request
scheduler.Scheduler._free_request = Scheduler._free_request

from vllm.v1.worker import gpu_model_runner

from ucm.integration.vllm.vllm_adapter.v1.worker.gpu_model_runner import (
    _prepare_inputs,
    _update_states,
    execute_model,
    maybe_wait_for_kv_save,
)

gpu_model_runner.GPUModelRunner.execute_model = execute_model
gpu_model_runner.GPUModelRunner.maybe_wait_for_kv_save = maybe_wait_for_kv_save
gpu_model_runner.GPUModelRunner._update_states = _update_states
gpu_model_runner.GPUModelRunner._prepare_inputs = _prepare_inputs

import vllm.distributed.kv_transfer.kv_connector.utils as utils_mod

from ucm.integration.vllm.vllm_adapter.distributed.kv_transfer.kv_connector.utils import (
    KVOutputAggregator,
)

utils_mod.KVOutputAggregator = KVOutputAggregator

from vllm.v1.executor import multiproc_executor

from ucm.integration.vllm.vllm_adapter.v1.executor.multiproc_executor import (
    MultiprocExecutor,
)

multiproc_executor.MultiprocExecutor._init_executor = MultiprocExecutor._init_executor
multiproc_executor.MultiprocExecutor.execute_model = MultiprocExecutor.execute_model

from vllm.v1.worker import gpu_worker

from ucm.integration.vllm.vllm_adapter.v1.worker.gpu_worker import (
    Worker,
    init_worker_distributed_environment,
)

gpu_worker.Worker.execute_model = Worker.execute_model
gpu_worker.init_worker_distributed_environment = init_worker_distributed_environment

from vllm.attention import layer

from ucm.integration.vllm.vllm_adapter.attention.layer import (
    unified_attention,
    unified_attention_with_output,
)

layer.unified_attention = unified_attention
layer.unified_attention_with_output = unified_attention_with_output

from vllm.v1.core import kv_cache_manager

from ucm.integration.vllm.vllm_adapter.v1.core.kv_cache_manager import (
    KVCacheManager,
)

kv_cache_manager.KVCacheManager.allocate_slots = KVCacheManager.allocate_slots

from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable

from ucm.integration.vllm.vllm_adapter.v1.worker.block_table import (
    reset_row,
    reset_row_multi,
)

BlockTable.reset_row = reset_row
MultiGroupBlockTable.reset_row = reset_row_multi
