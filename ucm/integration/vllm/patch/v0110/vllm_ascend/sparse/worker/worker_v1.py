import torch
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.v1.worker.worker_base import WorkerBase
from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel

from ucm.sparse.state import ensure_ucm_sparse_initialized


class NPUWorker(WorkerBase):
    def _init_worker_distributed_environment(self) -> None:
        """Initialize the distributed environment."""
        init_distributed_environment(
            self.parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            "hccl",
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
        )
        init_ascend_model_parallel(self.parallel_config)
        ensure_kv_transfer_initialized(self.vllm_config)
        ensure_ucm_sparse_initialized(self.vllm_config)
