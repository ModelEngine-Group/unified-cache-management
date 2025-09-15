# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import os
import pickle
import signal
import sys
import threading
import time
import traceback
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from threading import Thread
from typing import Optional, Union

import vllm.envs as envs
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.executor.multiproc_worker_utils import set_multiprocessing_worker_envs
from vllm.logger import init_logger
from vllm.utils import get_distributed_init_method, get_open_port
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.executor.multiproc_executor import UnreadyWorkerProcHandle, WorkerProc
from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


class MultiprocExecutor(Executor):

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pp_parallel_size = self.parallel_config.pipeline_parallel_size
        assert self.world_size == tensor_parallel_size * pp_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
            f"_parallel_size ({pp_parallel_size}). "
        )

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port()
        )

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size, self.world_size, max_chunk_bytes=max_chunk_bytes
        )
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            for rank in range(self.world_size):
                unready_workers.append(
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                    )
                )

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()

            self.start_worker_monitor()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                self._ensure_worker_termination([w.proc for w in unready_workers])

        # For pipeline parallel, we use a thread pool for asynchronous
        # execute_model.
        if self.max_concurrent_batches > 1:
            # Note: must use only 1 IO thread to keep dequeue sequence
            # from the response queue
            # _async_aggregate_workers_output also assumes a single IO thread
            self.io_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mp_exec_io"
            )

        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.kv_output_aggregator = KVOutputAggregator(self.parallel_config.world_size)

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        non_block = self.max_concurrent_batches > 1

        if not self.has_connector:
            # get output only from a single worker (output_rank)
            (output,) = self.collective_rpc(
                "execute_model",
                args=(scheduler_output,),
                unique_reply_rank=self.output_rank,
                non_block=non_block,
                timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            )
            return output

        # get output from all workers
        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        )

        # aggregate all workers output to a single output
        if non_block:
            return self.kv_output_aggregator.async_aggregate(outputs, self.output_rank)
        return self.kv_output_aggregator.aggregate(outputs, self.output_rank)
