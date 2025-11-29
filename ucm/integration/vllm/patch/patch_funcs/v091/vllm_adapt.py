#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import CancelledError, Future
from typing import Optional, cast

from ucm.logger import init_logger

logger = init_logger(__name__)


def _apply_adapt_patch() -> None:
    """
    Apply the PC adaptation patch for vLLM 0.9.1.
    This function contains all the necessary monkey patches for version 0.9.1.
    """
    try:
        _patch_kv_connector_utils()
        _patch_kv_connector_v1_base()
        _patch_block_pool()
        _patch_single_type_kv_cache_manager()
        _patch_multiproc_executor()
        _patch_outputs()
        _patch_request()
        _patch_gpu_input_batch()
        _patch_gpu_model_runner()
        _patch_gpu_worker()
        _patch_scheduler()

    except Exception as e:
        logger.error(f"Failed to apply vLLM 0.9.1 PC adaptation patch: {e}")
        raise


def _patch_kv_connector_utils():
    """
    Apply monkey patch to vllm.distributed.kv_transfer.kv_connector.utils
    Adds KVOutputAggregator class to the module.
    """
    try:
        from collections import defaultdict
        from collections.abc import Sequence
        from concurrent.futures import CancelledError, Future
        from typing import Optional, cast

        import torch
        import vllm.envs as envs
        from vllm import _custom_ops as ops
        from vllm.config import VllmConfig, get_current_vllm_config
        from vllm.distributed.kv_transfer.kv_connector import utils
        from vllm.logger import init_logger
        from vllm.v1.outputs import ModelRunnerOutput

        class KVOutputAggregator:
            """Utility class to aggregate the output of all workers into a single
            output corresponding to Rank 0 for scheduler."""

            def __init__(self, world_size: int):
                # Complete transfer tracker. Used by to track finished requests
                # [req_id -> n_finished_workers]
                self._recv_remaining_count = defaultdict[str, int](lambda: world_size)
                self._send_remaining_count = defaultdict[str, int](lambda: world_size)
                self._dump_remaining_count = defaultdict[str, int](lambda: world_size)

            def aggregate(
                self, outputs: list[ModelRunnerOutput], output_rank: int = 0
            ) -> ModelRunnerOutput:
                # aggregate finished_sending, finished_recving from all workers

                def update_finished_set(
                    req_ids: Optional[set[str]],
                    remaining_count_dict: dict[str, int],
                    finished_set: set[str],
                ) -> None:
                    for req_id in req_ids or ():
                        new_count = remaining_count_dict[req_id] - 1
                        if new_count == 0:
                            finished_set.add(req_id)
                            del remaining_count_dict[req_id]
                        else:
                            remaining_count_dict[req_id] = new_count

                def update_finished_list(
                    req_ids: Optional[dict[str, list[str]]],
                    remaining_count_dict: dict[str, int],
                    finished_list: dict[str, list[str]],
                ) -> None:
                    for req_id, succeed_dump_blocks in (req_ids or {}).items():
                        if req_id not in finished_list:
                            finished_list[req_id] = []
                        for blk_id in succeed_dump_blocks:
                            new_count = remaining_count_dict[blk_id] - 1
                            if new_count == 0:
                                finished_list[req_id].append(blk_id)
                                del remaining_count_dict[blk_id]
                            else:
                                remaining_count_dict[blk_id] = new_count

                finished_sending = set[str]()
                finished_recving = set[str]()
                invalid_block_ids = set[int]()
                finished_dumping: dict[str, list[str]] = {}
                for output in outputs:
                    update_finished_set(
                        output.finished_sending,
                        self._send_remaining_count,
                        finished_sending,
                    )
                    update_finished_set(
                        output.finished_recving,
                        self._recv_remaining_count,
                        finished_recving,
                    )
                    update_finished_list(
                        output.finished_dumping,
                        self._dump_remaining_count,
                        finished_dumping,
                    )
                    if output.invalid_block_ids:
                        invalid_block_ids |= output.invalid_block_ids

                # select output of the worker specified by output_rank
                output = outputs[output_rank]

                # set the aggregated finished_sending / finished_recving
                # if output.finished_sending/recving is not empty, but the other ranks
                # still have unfinished send/recv, we want to set the aggregated
                # finished_sending/recving to None until all ranks have finished
                # send/recv
                output.finished_sending = finished_sending if finished_sending else None
                output.finished_recving = finished_recving if finished_recving else None
                output.finished_dumping = finished_dumping if finished_dumping else None
                output.invalid_block_ids = invalid_block_ids or None

                return output

            def async_aggregate(
                self,
                output_futures: Sequence[Future[ModelRunnerOutput]],
                output_rank: int = 0,
            ) -> Future[ModelRunnerOutput]:
                """Takes a list of futures and returns a single future which resolves
                to the respective list of outputs."""
                result_future: Future[ModelRunnerOutput] = Future()

                outputs: list[Optional[ModelRunnerOutput]] = [None] * len(
                    output_futures
                )

                def make_callback(idx):

                    def callback(fut):
                        if result_future.done():
                            return

                        try:
                            outputs[idx] = fut.result()
                        except CancelledError:
                            result_future.cancel()
                        except Exception as e:
                            result_future.set_exception(e)

                        # this check assumes io_thread_pool uses a single thread
                        if all(outputs):
                            result_future.set_result(
                                self.aggregate(
                                    cast(list[ModelRunnerOutput], outputs), output_rank
                                )
                            )

                    return callback

                for i, output_future in enumerate(output_futures):
                    output_future.add_done_callback(make_callback(i))

                return result_future

        utils.KVOutputAggregator = KVOutputAggregator

    except Exception as e:
        logger.error(f"Failed to patch kv_connector utils: {e}")
        raise


def _patch_kv_connector_v1_base():
    """
    Apply monkey patch to vllm.distributed.kv_transfer.kv_connector.v1.base
    Adds get_block_ids_with_load_errors method to KVConnectorBase_V1 class.
    """
    try:
        import enum
        from abc import ABC, abstractmethod
        from typing import TYPE_CHECKING, Any, Optional

        import torch
        from vllm.logger import init_logger
        from vllm.v1.core.sched.output import SchedulerOutput

        if TYPE_CHECKING:
            from vllm.attention.backends.abstract import AttentionMetadata
            from vllm.config import VllmConfig
            from vllm.forward_context import ForwardContext
            from vllm.v1.core.kv_cache_manager import KVCacheBlocks
            from vllm.v1.request import Request

        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

        def get_block_ids_with_load_errors(self) -> Optional[set[int]]:
            """
            Get the set of block IDs that failed to load.
            Returns:
                Optional[set[int]]: A set of block IDs that encountered load errors.
                Returns None if no errors occurred during load.
            """
            return None

        KVConnectorBase_V1.get_block_ids_with_load_errors = (
            get_block_ids_with_load_errors
        )

    except Exception as e:
        logger.error(f"Failed to patch kv_connector v1 base: {e}")
        raise


def _patch_block_pool() -> None:
    """Patch BlockPool.cache_full_blocks to fix num_cached_blocks comparison."""
    try:
        from typing import Callable

        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
        from vllm.v1.request import Request

        def cache_full_blocks(
            self,
            request: Request,
            blocks: list[KVCacheBlock],
            block_hashes: list[BlockHash],
            num_cached_blocks: int,
            num_full_blocks: int,
            block_size: int,
            kv_cache_group_id: int,
            hash_fn: Callable,
        ) -> None:
            """Cache a list of full blocks for prefix caching.
            This function takes a list of blocks that will have their block hash
            metadata to be updated and cached. Given a request, it computes the
            block hashes for the blocks starting from `num_cached_blocks` to
            `num_full_blocks`, updating the metadata for each block
            and caching them in the `cached_block_hash_to_block`.

            Args:
                request: The request to cache the blocks.
                blocks: All blocks in the request.
                block_hashes: Block hashes of the blocks in the request. Note that
                this list may be shorter than the blocks list. In this case the
                missed block hash will be computed in this function.
                num_cached_blocks: The number of blocks that are already cached.
                num_full_blocks: The number of blocks that are full and should
                    be cached after this function.
                block_size: Number of tokens in each block.
                kv_cache_group_id: The id of the KV cache group.
                hash_fn: The hash function to use for block hashes.
            """
            if num_cached_blocks >= num_full_blocks:
                return
            new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
            assert len(block_hashes) >= num_cached_blocks
            new_block_hashes = block_hashes[num_cached_blocks:]

            # Update the new blocks with the block hashes through the chain.
            if num_cached_blocks == 0:
                prev_block_hash_value = None
            else:
                prev_block = blocks[num_cached_blocks - 1]
                assert prev_block.block_hash is not None
                prev_block_hash_value = prev_block.block_hash.get_hash_value()

            parent_block_hash = prev_block_hash_value
            new_hashes: Optional[list[int]] = (
                [] if self.enable_kv_cache_events else None
            )
            for i, blk in enumerate(new_full_blocks):
                assert blk.block_hash is None

                if i < len(new_block_hashes):
                    # The block hash may already be computed in
                    # "get_computed_blocks" if the tokens are not generated by
                    # this request (either the prompt tokens or the previously
                    # generated tokens with preemption), or by other
                    # single_type_managers with the same block_size.
                    # In this case we simply reuse the block hash.
                    block_hash = new_block_hashes[i]
                else:
                    # Otherwise compute the block hash and cache it in the request
                    # in case it will be preempted in the future.
                    blk_idx = num_cached_blocks + i
                    start_token_idx = blk_idx * block_size
                    end_token_idx = (blk_idx + 1) * block_size
                    block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
                    assert len(block_tokens) == block_size, (
                        f"Expected {block_size} tokens, got "
                        f"{len(block_tokens)} at {blk_idx}th block for request "
                        f"{request.request_id}({request})"
                    )

                    # Generate extra keys for multi-modal inputs. Note that since
                    # we reach to this branch only when the block is completed with
                    # generated tokens, we only need to consider the last mm input.
                    extra_keys, _ = generate_block_hash_extra_keys(
                        request, start_token_idx, end_token_idx, -1
                    )

                    # Compute the hash of the current block.
                    block_hash = hash_block_tokens(
                        hash_fn, prev_block_hash_value, block_tokens, extra_keys
                    )
                    block_hashes.append(block_hash)

                # Update and added the full block to the cache.
                block_hash_with_group_id = BlockHashWithGroupId(
                    block_hash, kv_cache_group_id
                )
                blk.block_hash = block_hash_with_group_id
                self.cached_block_hash_to_block[block_hash_with_group_id][
                    blk.block_id
                ] = blk
                if new_hashes is not None:
                    new_hashes.append(block_hash.hash_value)
                prev_block_hash_value = block_hash.hash_value

            if self.enable_kv_cache_events:
                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=new_hashes,
                        parent_block_hash=parent_block_hash,
                        token_ids=request.all_token_ids[
                            num_cached_blocks
                            * block_size : num_full_blocks
                            * block_size
                        ],
                        block_size=block_size,
                        lora_id=(
                            request.lora_request.id if request.lora_request else None
                        ),
                    )
                )

        BlockPool.cache_full_blocks = cache_full_blocks
    except ImportError:
        logger.warning("Could not patch BlockPool.cache_full_blocks - module not found")


def _patch_single_type_kv_cache_manager() -> None:
    """Patch SingleTypeKVCacheManager to add cache_blocks method."""
    try:
        from vllm.v1.core.kv_cache_utils import BlockHash
        from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
        from vllm.v1.request import Request

        def cache_blocks(
            self, request: Request, block_hashes: list[BlockHash], num_tokens: int
        ) -> None:
            """
            Cache the blocks for the request.

            Args:
                request: The request.
                block_hashes: The block hashes of the request.
                num_tokens: The total number of tokens that need to be cached
                    (including tokens that are already cached).
            """
            num_cached_blocks = self.num_cached_block[request.request_id]
            num_full_blocks = num_tokens // self.block_size
            if num_cached_blocks >= num_full_blocks:
                return

            self.block_pool.cache_full_blocks(
                request=request,
                blocks=self.req_to_blocks[request.request_id],
                block_hashes=block_hashes,
                num_cached_blocks=num_cached_blocks,
                num_full_blocks=num_full_blocks,
                block_size=self.block_size,
                kv_cache_group_id=self.kv_cache_group_id,
                hash_fn=self.caching_hash_fn,
            )

            self.num_cached_block[request.request_id] = num_full_blocks

        SingleTypeKVCacheManager.cache_blocks = cache_blocks
    except ImportError:
        logger.warning(
            "Could not patch SingleTypeKVCacheManager.cache_blocks - module not found"
        )


def _patch_multiproc_executor():
    """
    Apply monkey patch to vllm.v1.executor.multiproc_executor
    Adds KVOutputAggregator and modifies execute_model method.
    """
    try:
        import threading
        import weakref
        from concurrent.futures import Future, ThreadPoolExecutor
        from typing import TYPE_CHECKING, Any, Optional, Union

        import vllm.envs as envs
        from vllm.distributed.device_communicators.shm_broadcast import (
            Handle,
            MessageQueue,
        )
        from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
        from vllm.executor.multiproc_worker_utils import (
            _add_prefix,
            set_multiprocessing_worker_envs,
        )
        from vllm.utils import (
            get_distributed_init_method,
            get_mp_context,
            get_open_port,
        )
        from vllm.v1.executor.abstract import Executor, FailureCallback
        from vllm.v1.executor.multiproc_executor import (
            EXECUTE_MODEL_TIMEOUT_S,
            MultiprocExecutor,
            UnreadyWorkerProcHandle,
            WorkerProc,
        )
        from vllm.v1.outputs import (
            EMPTY_MODEL_RUNNER_OUTPUT,
            LogprobsTensors,
            ModelRunnerOutput,
        )

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
            self.kv_output_aggregator = KVOutputAggregator(
                self.parallel_config.world_size
            )

        MultiprocExecutor._init_executor = _init_executor

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
                    timeout=EXECUTE_MODEL_TIMEOUT_S,
                )
                return output

            # get output from all workers
            outputs = self.collective_rpc(
                "execute_model",
                args=(scheduler_output,),
                non_block=non_block,
                timeout=EXECUTE_MODEL_TIMEOUT_S,
            )

            # aggregate all workers output to a single output
            if non_block:
                return self.kv_output_aggregator.async_aggregate(
                    outputs, self.output_rank
                )
            return self.kv_output_aggregator.aggregate(outputs, self.output_rank)

        MultiprocExecutor.execute_model = execute_model

    except Exception as e:
        logger.error(f"Failed to patch multiproc executor: {e}")
        raise


def _patch_outputs():
    """
    Apply monkey patch to vllm.v1.outputs
    Adds finished_dumping and invalid_block_ids fields to ModelRunnerOutput.
    """
    try:
        from dataclasses import dataclass
        from typing import NamedTuple, Optional

        import torch
        from vllm.v1 import outputs
        from vllm.v1.outputs import LogprobsLists, LogprobsTensors

        @dataclass
        class ModelRunnerOutput:
            # [num_reqs]
            req_ids: list[str]
            # req_id -> index
            req_id_to_index: dict[str, int]

            # num_reqs x num_generated_tokens
            # num_generated_tokens is the number of tokens
            # generated in the current step. It can be different for
            # each request due to speculative/jump decoding.
            sampled_token_ids: list[list[int]]

            # num_reqs x num_spec_tokens
            spec_token_ids: Optional[list[list[int]]]

            # [num_reqs, max_num_logprobs + 1]
            # [num_reqs, max_num_logprobs + 1]
            # [num_reqs]
            logprobs: Optional[LogprobsLists]

            # req_id -> (token_ids, logprobs, ranks)
            # [prompt_len, num_prompt_logprobs]
            # [prompt_len, num_prompt_logprobs]
            # [prompt_len]
            prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]]

            # [req_ids]
            finished_sending: Optional[set[str]] = None
            finished_recving: Optional[set[str]] = None
            finished_dumping: Optional[dict[str, list[str]]] = None

            # IDs of externally computed KV blocks that failed to load.
            # Requests referencing these blocks should be rescheduled to recompute them.
            invalid_block_ids: Optional[set[int]] = None

        ModelRunnerOutput.__module__ = outputs.__name__
        ModelRunnerOutput.__qualname__ = "ModelRunnerOutput"

        outputs.ModelRunnerOutput = ModelRunnerOutput

        EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            finished_sending=None,
            finished_recving=None,
        )
        outputs.EMPTY_MODEL_RUNNER_OUTPUT = EMPTY_MODEL_RUNNER_OUTPUT

    except Exception as e:
        logger.error(f"Failed to patch outputs: {e}")
        raise


def _patch_request():
    """
    Apply monkey patch to vllm.v1.request
    Adds succeed_dumped_blocks field to Request class.
    """
    try:
        import enum
        from typing import TYPE_CHECKING, Any, Optional, Union

        from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
        from vllm.sampling_params import SamplingParams
        from vllm.utils import is_list_of
        from vllm.v1.engine import (
            EngineCoreEvent,
            EngineCoreEventType,
            EngineCoreRequest,
            FinishReason,
        )
        from vllm.v1.structured_output.request import StructuredOutputRequest
        from vllm.v1.utils import ConstantList

        if TYPE_CHECKING:
            from vllm.lora.request import LoRARequest

        from vllm.v1.request import Request, RequestStatus

        def __init__(
            self,
            request_id: str,
            prompt_token_ids: list[int],
            multi_modal_inputs: Optional[list[MultiModalKwargs]],
            multi_modal_hashes: Optional[list[str]],
            multi_modal_placeholders: Optional[list[PlaceholderRange]],
            sampling_params: SamplingParams,
            eos_token_id: Optional[int],
            client_index: int = 0,
            lora_request: Optional["LoRARequest"] = None,
            structured_output_request: Optional["StructuredOutputRequest"] = None,
            cache_salt: Optional[str] = None,
        ) -> None:
            self.request_id = request_id
            self.client_index = client_index
            self.sampling_params = sampling_params
            # Because of LoRA, the eos token id can be different for each request.
            self.eos_token_id = eos_token_id
            self.lora_request = lora_request
            self.structured_output_request = structured_output_request

            self.status = (
                RequestStatus.WAITING_FOR_FSM
                if sampling_params.guided_decoding is not None
                else RequestStatus.WAITING
            )
            self.events: list[EngineCoreEvent] = []
            self.stop_reason: Union[int, str, None] = None
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens

            self.prompt_token_ids = prompt_token_ids
            self.num_prompt_tokens = len(self.prompt_token_ids)
            self._output_token_ids: list[int] = []
            self._all_token_ids: list[int] = self.prompt_token_ids.copy()
            self.spec_token_ids: list[int] = []
            self.num_computed_tokens = 0
            self.cache_salt: Optional[str] = cache_salt

            # Multi-modal related
            self.mm_positions = multi_modal_placeholders or []
            self.mm_inputs = multi_modal_inputs or []
            self.mm_hashes: list[str] = multi_modal_hashes or []
            self.num_encoder_inputs = len(self.mm_inputs)
            self.has_encoder_inputs = self.num_encoder_inputs > 0

            # P/D: Connector-specific KV transfer parameters.
            kv_params = (
                None
                if sampling_params.extra_args is None
                else sampling_params.extra_args.get("kv_transfer_params")
            )
            self.kv_transfer_params: Optional[dict[str, Any]] = kv_params

            # Sanity check
            assert len(self.mm_inputs) == len(self.mm_positions)
            if self.mm_hashes:
                assert len(self.mm_inputs) == len(self.mm_hashes)

            # Read-only views
            # Prevent directly appending to these lists since
            # they should also be updated simultaneously.
            self.output_token_ids = ConstantList(self._output_token_ids)
            self.all_token_ids = ConstantList(self._all_token_ids)

            # State
            # The number of tokens with prefix cache hits.
            self.num_cached_tokens = -1
            self.succeed_dumped_blocks: list[str] = []

        Request.__init__ = __init__

    except Exception as e:
        logger.error(f"Failed to patch request: {e}")
        raise


def _patch_gpu_input_batch():
    """
    Apply monkey patch to vllm.v1.worker.gpu_input_batch
    Adds generator offset tracking fields and logic.
    """
    try:
        from dataclasses import dataclass
        from typing import Optional, cast

        import numpy as np
        import torch
        from vllm.lora.request import LoRARequest
        from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
        from vllm.sampling_params import SamplingParams, SamplingType
        from vllm.utils import swap_dict_values
        from vllm.v1.outputs import LogprobsTensors
        from vllm.v1.sample.metadata import SamplingMetadata
        from vllm.v1.utils import copy_slice
        from vllm.v1.worker.block_table import MultiGroupBlockTable
        from vllm.v1.worker.gpu_input_batch import (
            _SAMPLING_EPS,
            CachedRequestState,
            InputBatch,
        )

        def __post_init__(self):
            self.num_prompt_tokens = len(self.prompt_token_ids)
            # 'last_generator_offset' and 'last_gelen_last_output_token_ids' are
            # used to allow safe rollback in case a sampled token turns out to be
            # invalid (e.g., due to KV load errors).
            self.last_generator_offset = 0 if self.generator else None
            self.len_last_output_token_ids = len(self.output_token_ids)

        CachedRequestState.__post_init__ = __post_init__

        def __init__(
            self,
            max_num_reqs: int,
            max_model_len: int,
            max_num_batched_tokens: int,
            device: torch.device,
            pin_memory: bool,
            vocab_size: int,
            block_sizes: list[int],  # The block_size of each kv cache group
        ):
            self.max_num_reqs = max_num_reqs
            self.max_model_len = max_model_len
            self.max_num_batched_tokens = max_num_batched_tokens
            self.device = device
            self.pin_memory = pin_memory
            self.vocab_size = vocab_size

            self._req_ids: list[Optional[str]] = []
            self.req_id_to_index: dict[str, int] = {}

            # TODO(woosuk): This buffer could be too large if max_model_len is big.
            # Find a way to reduce the CPU memory usage.
            # This buffer is not directly transferred to the GPU, so it does not
            # need to be pinned.
            self.token_ids_cpu_tensor = torch.zeros(
                (max_num_reqs, max_model_len),
                device="cpu",
                dtype=torch.int32,
                pin_memory=False,
            )
            self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
            self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
            self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)
            self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
            self.num_computed_tokens_cpu_tensor = torch.zeros(
                (max_num_reqs,),
                device="cpu",
                dtype=torch.int32,
                pin_memory=pin_memory,
            )
            self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()

            # Block table.
            self.block_table = MultiGroupBlockTable(
                max_num_reqs=max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                pin_memory=pin_memory,
                device=device,
                block_sizes=block_sizes,
            )

            # Sampling-related.
            self.temperature = torch.empty(
                (max_num_reqs,), dtype=torch.float32, device=device
            )
            self.temperature_cpu_tensor = torch.empty(
                (max_num_reqs,),
                dtype=torch.float32,
                device="cpu",
                pin_memory=pin_memory,
            )
            self.temperature_cpu = self.temperature_cpu_tensor.numpy()
            self.greedy_reqs: set[str] = set()
            self.random_reqs: set[str] = set()

            self.top_p = torch.empty(
                (max_num_reqs,), dtype=torch.float32, device=device
            )
            self.top_p_cpu_tensor = torch.empty(
                (max_num_reqs,),
                dtype=torch.float32,
                device="cpu",
                pin_memory=pin_memory,
            )
            self.top_p_cpu = self.top_p_cpu_tensor.numpy()
            self.top_p_reqs: set[str] = set()

            self.top_k = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
            self.top_k_cpu_tensor = torch.empty(
                (max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
            self.top_k_cpu = self.top_k_cpu_tensor.numpy()
            self.top_k_reqs: set[str] = set()

            self.min_p = torch.empty(
                (max_num_reqs,), dtype=torch.float32, device=device
            )
            self.min_p_cpu_tensor = torch.empty(
                (max_num_reqs,),
                dtype=torch.float32,
                device="cpu",
                pin_memory=pin_memory,
            )
            self.min_p_cpu = self.min_p_cpu_tensor.numpy()
            self.min_p_reqs: set[str] = set()

            # Frequency penalty related data structures
            self.frequency_penalties = torch.empty(
                (max_num_reqs,), dtype=torch.float, device=device
            )
            self.frequency_penalties_cpu_tensor = torch.empty(
                (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
            )
            self.frequency_penalties_cpu = self.frequency_penalties_cpu_tensor.numpy()
            self.frequency_penalties_reqs: set[str] = set()

            # Presence penalty related data structures
            self.presence_penalties = torch.empty(
                (max_num_reqs,), dtype=torch.float, device=device
            )
            self.presence_penalties_cpu_tensor = torch.empty(
                (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
            )
            self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy()
            self.presence_penalties_reqs: set[str] = set()

            # Repetition penalty related data structures
            self.repetition_penalties = torch.empty(
                (max_num_reqs,), dtype=torch.float, device=device
            )
            self.repetition_penalties_cpu_tensor = torch.empty(
                (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
            )
            self.repetition_penalties_cpu = self.repetition_penalties_cpu_tensor.numpy()
            self.repetition_penalties_reqs: set[str] = set()

            # req_index -> (min_tokens, stop_token_ids)
            self.min_tokens: dict[int, tuple[int, set[int]]] = {}

            # lora related
            self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int32)
            self.lora_id_to_request_ids: dict[int, set[str]] = {}
            self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

            # req_index -> generator
            # NOTE(woosuk): The indices of the requests that do not have their own
            # generator should not be included in the dictionary.
            self.generators: dict[int, torch.Generator] = {}
            self.generators_last_offset: dict[int, int] = {}

            self.num_logprobs: dict[str, int] = {}
            # NOTE(rob): num_prompt_logprobs only includes reqs
            # that are currently in the prefill phase.
            self.num_prompt_logprobs: dict[str, int] = {}

            # To accumulate prompt logprobs tensor chunks across prefill steps.
            self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}

            self.logit_bias: list[Optional[dict[int, float]]] = [None] * max_num_reqs
            self.has_allowed_token_ids: set[str] = set()
            # NOTE(lufang): In the mask tensor, if the corresponding token allowed,
            # the value is False. Since we use masked_fill_ to set -inf.
            self.allowed_token_ids_mask: Optional[torch.Tensor] = None
            self.allowed_token_ids_mask_cpu_tensor: Optional[torch.Tensor] = None

            # req_index -> bad_words_token_ids
            self.bad_words_token_ids: dict[int, list[list[int]]] = {}

            self.req_output_token_ids: list[Optional[list[int]]] = []

            # This is updated each time the batch constituents change.
            self.sampling_metadata = self._make_sampling_metadata()

        InputBatch.__init__ = __init__

        def add_request(
            self,
            request: "CachedRequestState",
            req_index: Optional[int] = None,
        ) -> None:
            if req_index is None:
                req_index = self.num_reqs
            assert req_index < self.max_num_reqs

            req_id = request.req_id
            if req_index == len(self._req_ids):
                self._req_ids.append(req_id)
                self.req_output_token_ids.append(request.output_token_ids)
            else:
                self._req_ids[req_index] = req_id
                self.req_output_token_ids[req_index] = request.output_token_ids

            self.req_id_to_index[req_id] = req_index

            # Copy the prompt token ids and output token ids.
            num_prompt_tokens = len(request.prompt_token_ids)
            self.num_prompt_tokens[req_index] = num_prompt_tokens
            self.token_ids_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids
            start_idx = num_prompt_tokens
            end_idx = start_idx + len(request.output_token_ids)
            self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids
            # Number of token ids in token_ids_cpu.
            # NOTE(woosuk): This may include spec decode tokens.
            self.num_tokens[req_index] = request.num_tokens
            # Number of tokens without spec decode tokens.
            self.num_tokens_no_spec[req_index] = request.num_tokens

            self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
            self.block_table.add_row(request.block_ids, req_index)

            sampling_params = request.sampling_params
            if sampling_params.sampling_type == SamplingType.GREEDY:
                # Avoid later division by zero.
                self.temperature_cpu[req_index] = -1.0
                self.greedy_reqs.add(req_id)
            else:
                self.temperature_cpu[req_index] = sampling_params.temperature
                self.random_reqs.add(req_id)

            self.top_p_cpu[req_index] = sampling_params.top_p
            if sampling_params.top_p < 1:
                self.top_p_reqs.add(req_id)
            top_k = sampling_params.top_k
            if 0 < top_k < self.vocab_size:
                self.top_k_reqs.add(req_id)
            else:
                top_k = self.vocab_size
            self.top_k_cpu[req_index] = top_k
            self.min_p_cpu[req_index] = sampling_params.min_p
            self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
            if sampling_params.min_p > _SAMPLING_EPS:
                self.min_p_reqs.add(req_id)
            if sampling_params.frequency_penalty != 0.0:
                self.frequency_penalties_reqs.add(req_id)
            self.presence_penalties_cpu[req_index] = sampling_params.presence_penalty
            if sampling_params.presence_penalty != 0.0:
                self.presence_penalties_reqs.add(req_id)
            self.repetition_penalties_cpu[req_index] = (
                sampling_params.repetition_penalty
            )
            if sampling_params.repetition_penalty != 1.0:
                self.repetition_penalties_reqs.add(req_id)
            if sampling_params.min_tokens:
                self.min_tokens[req_index] = (
                    sampling_params.min_tokens,
                    sampling_params.all_stop_token_ids,
                )

            # NOTE(woosuk): self.generators should not include the requests that
            # do not have their own generator.
            if request.generator is not None:
                self.generators[req_index] = request.generator
                assert request.last_generator_offset is not None
                self.generators_last_offset[req_index] = request.last_generator_offset

            if sampling_params.logprobs is not None:
                self.num_logprobs[req_id] = sampling_params.logprobs
            if sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = sampling_params.prompt_logprobs
            if sampling_params.logit_bias is not None:
                self.logit_bias[req_index] = sampling_params.logit_bias

            if sampling_params.allowed_token_ids:
                self.has_allowed_token_ids.add(req_id)
                if self.allowed_token_ids_mask_cpu_tensor is None:
                    # Lazy allocation for this tensor, which can be large.
                    # False means we don't fill with -inf.
                    self.allowed_token_ids_mask = torch.zeros(
                        self.max_num_reqs,
                        self.vocab_size,
                        dtype=torch.bool,
                        device=self.device,
                    )
                    self.allowed_token_ids_mask_cpu_tensor = torch.zeros(
                        self.max_num_reqs,
                        self.vocab_size,
                        dtype=torch.bool,
                        device="cpu",
                    )
                self.allowed_token_ids_mask_cpu_tensor[req_index] = True
                # False means we don't fill with -inf.
                self.allowed_token_ids_mask_cpu_tensor[req_index][
                    sampling_params.allowed_token_ids
                ] = False

            if sampling_params.bad_words_token_ids:
                self.bad_words_token_ids[req_index] = (
                    sampling_params.bad_words_token_ids
                )

            # Add request lora ID
            if request.lora_request:
                lora_id = request.lora_request.lora_int_id
                if lora_id not in self.lora_id_to_request_ids:
                    self.lora_id_to_request_ids[lora_id] = set()

                self.request_lora_mapping[req_index] = lora_id
                self.lora_id_to_request_ids[lora_id].add(request.req_id)
                self.lora_id_to_lora_request[lora_id] = request.lora_request
            else:
                # No LoRA
                self.request_lora_mapping[req_index] = 0

        InputBatch.add_request = add_request

    except Exception as e:
        logger.error(f"Failed to patch gpu input batch: {e}")
        raise


def _patch_gpu_model_runner():
    """
    Apply monkey patch to vllm.v1.worker.gpu_model_runner
    Adds KV cache error handling and generator rollback logic.
    """
    try:
        import copy
        from typing import TYPE_CHECKING, Optional, Union

        import torch
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
        from vllm.sampling_params import SamplingType
        from vllm.sequence import IntermediateTensors
        from vllm.utils import async_tensor_h2d
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
        from vllm.v1.spec_decode.eagle import EagleProposer
        from vllm.v1.spec_decode.medusa import MedusaProposer
        from vllm.v1.spec_decode.ngram_proposer import NgramProposer
        from vllm.v1.worker.gpu_input_batch import CachedRequestState

        if TYPE_CHECKING:
            from vllm.v1.core.sched.output import SchedulerOutput

        # Import the target module
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
            """Update the cached states and the persistent batch with the scheduler
            output.

            The updated states are used by the `_prepare_inputs` function to create
            the input GPU tensors for the model.

            The SamplingMetadata is updated and copied to the GPU if there is a
            new/resumed/paused/finished request in the batch.
            """
            # Remove finished requests from the cached states.
            for req_id in scheduler_output.finished_req_ids:
                self.requests.pop(req_id, None)
                self.encoder_cache.pop(req_id, None)
            # Remove the finished requests from the persistent batch.
            # NOTE(woosuk): There could be an edge case where finished_req_ids and
            # scheduled_req_ids overlap. This happens when a request is aborted and
            # then resubmitted with the same ID. In this case, we treat them as two
            # distinct requests - clearing the cached states for the first request
            # and handling the second as a new request.
            removed_req_indices: list[int] = []
            for req_id in scheduler_output.finished_req_ids:
                req_index = self.input_batch.remove_request(req_id)
                if req_index is not None:
                    removed_req_indices.append(req_index)

            # Free the cached encoder outputs.
            for req_id, input_id in scheduler_output.free_encoder_input_ids:
                encoder_outputs = self.encoder_cache.get(req_id)
                if encoder_outputs is not None:
                    encoder_outputs.pop(input_id, None)
                    if not encoder_outputs:
                        self.encoder_cache.pop(req_id, None)

            # Remove the unscheduled requests from the persistent batch.
            # NOTE(woosuk): The unscheduled requests are either preempted requests
            # or running requests that are not scheduled in this step. We remove
            # them from the persistent batch but keep their cached states since
            # they will be scheduled again sometime in the future.
            scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
            cached_req_ids = self.input_batch.req_id_to_index.keys()
            unscheduled_req_ids = cached_req_ids - scheduled_req_ids
            # NOTE(woosuk): The persistent batch optimization assumes that
            # consecutive batches contain mostly the same requests. If batches
            # have low request overlap (e.g., alternating between two distinct
            # sets of requests), this optimization becomes very inefficient.
            for req_id in unscheduled_req_ids:
                req_index = self.input_batch.remove_request(req_id)
                assert req_index is not None
                removed_req_indices.append(req_index)

            req_ids_to_add: list[str] = []
            # Add new requests to the cached states.
            for new_req_data in scheduler_output.scheduled_new_reqs:
                req_id = new_req_data.req_id
                sampling_params = new_req_data.sampling_params
                if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(sampling_params.seed)
                else:
                    generator = None

                self.requests[req_id] = CachedRequestState(
                    req_id=req_id,
                    prompt_token_ids=new_req_data.prompt_token_ids,
                    mm_inputs=new_req_data.mm_inputs,
                    mm_positions=new_req_data.mm_positions,
                    sampling_params=sampling_params,
                    generator=generator,
                    block_ids=new_req_data.block_ids,
                    num_computed_tokens=new_req_data.num_computed_tokens,
                    output_token_ids=[],
                    lora_request=new_req_data.lora_request,
                )

                # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
                if self.uses_mrope:
                    image_grid_thw = []
                    video_grid_thw = []
                    second_per_grid_ts = []
                    audio_feature_lengths = []
                    use_audio_in_video = False
                    for mm_input in self.requests[req_id].mm_inputs:
                        if mm_input.get("image_grid_thw") is not None:
                            image_grid_thw.extend(mm_input["image_grid_thw"].tolist())
                        if mm_input.get("video_grid_thw") is not None:
                            video_grid_thw.extend(mm_input["video_grid_thw"].tolist())
                        if mm_input.get("second_per_grid_ts") is not None:
                            second_per_grid_ts.extend(mm_input["second_per_grid_ts"])
                        if mm_input.get("audio_feature_lengths") is not None:
                            audio_feature_lengths.extend(
                                mm_input["audio_feature_lengths"]
                            )
                        if mm_input.get("use_audio_in_video") is True:
                            use_audio_in_video = True

                    hf_config = self.model_config.hf_config

                    (
                        self.requests[req_id].mrope_positions,
                        self.requests[req_id].mrope_position_delta,
                    ) = MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )

                req_ids_to_add.append(req_id)

            # Update the states of the running/resumed requests.
            for req_data in scheduler_output.scheduled_cached_reqs:
                req_id = req_data.req_id
                req_state = self.requests[req_id]
                num_computed_tokens = req_data.num_computed_tokens

                # Update the cached states.
                if num_computed_tokens <= req_state.num_computed_tokens:
                    # The request was rescheduled after a KV load failure. Clear
                    # the last sampled tokens and rewind the generator state
                    len_output_token_ids = len(req_state.output_token_ids)
                    del req_state.output_token_ids[
                        req_state.len_last_output_token_ids :
                    ]
                    if req_state.generator:
                        req_state.generator.set_offset(req_state.last_generator_offset)
                    req_index = self.input_batch.req_id_to_index.get(req_id)
                    if req_index is not None:
                        len_last_sampled = (
                            len_output_token_ids - req_state.len_last_output_token_ids
                        )
                        end_idx = (
                            self.input_batch.num_tokens_no_spec[req_index]
                            - len_last_sampled
                        )
                        self.input_batch.num_tokens[req_index] = end_idx
                        self.input_batch.num_tokens_no_spec[req_index] = end_idx

                num_computed_tokens = req_data.num_computed_tokens
                req_state.num_computed_tokens = num_computed_tokens
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec decode tokens.
                num_new_tokens = (
                    num_computed_tokens
                    + len(req_data.new_token_ids)
                    - req_state.num_tokens
                )
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(req_data.new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        req_data.new_token_ids[-num_new_tokens:]
                    )
                req_state.len_last_output_token_ids = len(req_state.output_token_ids)
                if req_state.generator:
                    req_state.last_generator_offset = req_state.generator.get_offset()

                # Update the block IDs.
                if not req_data.resumed_from_preemption:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_block_ids in zip(  # type: ignore[call-overload]
                        req_state.block_ids, req_data.new_block_ids, strict=True
                    ):
                        block_ids.extend(new_block_ids)
                else:
                    # The request is resumed from preemption.
                    # Replace the existing block IDs with the new ones.
                    req_state.block_ids = req_data.new_block_ids

                req_index = self.input_batch.req_id_to_index.get(req_id)
                if req_index is None:
                    # The request is not in the persistent batch.
                    # The request was either preempted and resumed later, or was not
                    # scheduled in the previous step and needs to be added again.
                    req_ids_to_add.append(req_id)
                    continue

                if req_state.generator:
                    assert req_state.last_generator_offset is not None
                    self.input_batch.generators_last_offset[req_index] = (
                        req_state.last_generator_offset
                    )

                # Update the persistent batch.
                self.input_batch.num_computed_tokens_cpu[req_index] = (
                    num_computed_tokens
                )
                self.input_batch.block_table.append_row(
                    req_data.new_block_ids, req_index
                )
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(req_data.new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = req_data.new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index
                # Add spec_token_ids to token_ids_cpu.
                spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                    req_id, ()
                )
                if spec_token_ids:
                    start_index = end_token_index
                    end_token_index += len(spec_token_ids)
                    self.input_batch.token_ids_cpu[
                        req_index, start_index:end_token_index
                    ] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
                self.input_batch.num_tokens[req_index] = end_token_index

            # Check if the batch has changed. If not, we can skip copying the
            # sampling metadata from CPU to GPU.
            batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

            # Add the new or resumed requests to the persistent batch.
            # The smaller empty indices are filled first.
            removed_req_indices.sort(reverse=True)
            for req_id in req_ids_to_add:
                req_state = self.requests[req_id]
                if removed_req_indices:
                    # Fill the empty index.
                    req_index = removed_req_indices.pop()
                else:
                    # Append to the end.
                    req_index = None
                self.input_batch.add_request(req_state, req_index)

            # Condense the batched states if there are empty indices.
            if removed_req_indices:
                self.input_batch.condense(removed_req_indices)

            batch_reordered = self._may_reorder_batch(scheduler_output)

            if batch_changed or batch_reordered:
                self.input_batch.refresh_sampling_metadata()

        GPUModelRunner._update_states = _update_states

        @torch.inference_mode()
        def execute_model(
            self,
            scheduler_output: "SchedulerOutput",
            intermediate_tensors: Optional[IntermediateTensors] = None,
        ) -> Union[ModelRunnerOutput, IntermediateTensors]:

            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT

                return self.kv_connector_no_forward(scheduler_output)

            # Prepare the decoder inputs.
            attn_metadata, logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output
            )
            num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            if (
                self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]
            ):
                # Use piecewise CUDA graphs.
                # Add padding to the batch size.
                num_input_tokens = self.vllm_config.pad_for_cudagraph(
                    num_scheduled_tokens
                )
            else:
                # Eager mode.
                # Pad tokens to multiple of tensor_parallel_size when
                # enabled collective fusion for SP
                tp_size = self.vllm_config.parallel_config.tensor_parallel_size
                if (
                    self.vllm_config.compilation_config.pass_config.enable_sequence_parallelism
                    and tp_size > 1
                ):
                    from vllm.utils import round_up

                    num_input_tokens = round_up(num_scheduled_tokens, tp_size)
                else:
                    num_input_tokens = num_scheduled_tokens

            # Padding for DP
            num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
            num_input_tokens += num_pad

            # _prepare_inputs may reorder the batch, so we must gather multi
            # modal outputs after that to ensure the correct order
            if self.is_multimodal_model:
                # Run the multimodal encoder if any.
                self._execute_mm_encoder(scheduler_output)
                mm_embeds = self._gather_mm_embeddings(scheduler_output)
            else:
                mm_embeds = []

            if self.is_multimodal_model and get_pp_group().is_first_rank:
                # NOTE(woosuk): To unify token ids and soft tokens (vision
                # embeddings), we always use embeddings (rather than token ids)
                # as input to the multimodal model, even when the input is text.
                input_ids = self.input_ids[:num_scheduled_tokens]
                if mm_embeds:
                    inputs_embeds = self.model.get_input_embeddings(
                        input_ids, mm_embeds
                    )
                else:
                    inputs_embeds = self.model.get_input_embeddings(input_ids)
                # TODO(woosuk): Avoid the copy. Optimize.
                self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
                inputs_embeds = self.inputs_embeds[:num_input_tokens]
                input_ids = None
            else:
                # For text-only models, we use token ids as input.
                # While it is possible to use embeddings as input just like the
                # multimodal models, it is not desirable for performance since
                # then the embedding layer is not included in the CUDA graph.
                input_ids = self.input_ids[:num_input_tokens]
                inputs_embeds = None
            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_input_tokens]
            else:
                positions = self.positions[:num_input_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_input_tokens, intermediate_tensors, True
                )

            # Run the decoder.
            # Use persistent buffers for CUDA graphs.
            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
            ):
                self.maybe_setup_kv_connector(scheduler_output)

                model_output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )

                finished_dumping = self.maybe_wait_for_kv_save()
                finished_sending, finished_recving = self.get_finished_kv_transfers(
                    scheduler_output
                )
                invalid_block_ids = self.get_block_ids_with_load_errors()

            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
            # Broadcast PP output for external_launcher (torchrun)
            # to make sure we are synced across pp ranks
            # TODO: Support overlapping mirco-batches
            # https://github.com/vllm-project/vllm/issues/18019
            broadcast_pp_output = (
                self.parallel_config.distributed_executor_backend == "external_launcher"
                and len(get_pp_group().ranks) > 0
            )
            if not get_pp_group().is_last_rank:
                # For mid-pipeline stages, return the hidden states.
                if not broadcast_pp_output:
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group()
                )
                logits = None
            else:
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states, None)
            if broadcast_pp_output:
                model_output_broadcast_data = (
                    {
                        "logits": logits.contiguous(),
                    }
                    if logits is not None
                    else {}
                )
                model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                self.apply_grammar_bitmask(scheduler_output, logits)

            # Sample the next token and get logprobs if needed.
            sampling_metadata = self.input_batch.sampling_metadata
            if spec_decode_metadata is None:
                sampler_output = self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
            else:
                # When indexing with a tensor (bonus_logits_indices), PyTorch
                # creates a new tensor with separate storage from the original
                # logits tensor. This means any in-place operations on bonus_logits
                # won't affect the original logits tensor.
                assert logits is not None
                bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=sampling_metadata,
                )
                bonus_token_ids = sampler_output.sampled_token_ids

                # Just like `bonus_logits`, `target_logits` is a new tensor with
                # separate storage from the original `logits` tensor. Therefore,
                # it is safe to update `target_logits` in place.
                target_logits = logits[spec_decode_metadata.target_logits_indices]
                output_token_ids = self.rejection_sampler(
                    spec_decode_metadata,
                    None,  # draft_probs
                    target_logits,
                    bonus_token_ids,
                    sampling_metadata,
                )
                sampler_output.sampled_token_ids = output_token_ids

            # TODO(woosuk): The following loop can be slow since it iterates over
            # the requests one by one. Optimize.
            discard_sampled_tokens_req_indices = []
            for i, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                seq_len = (
                    req_state.num_computed_tokens
                    + scheduler_output.num_scheduled_tokens[req_id]
                )
                if seq_len < req_state.num_tokens:
                    # Ignore the sampled token for partial prefills.
                    # Rewind the generator state as if the token was not sampled.
                    # This relies on cuda-specific torch-internal impl details
                    generator = self.input_batch.generators.get(i)
                    if generator is not None:
                        generator.set_offset(
                            self.input_batch.generators_last_offset.get(i)
                        )
                    # Record the index of the request that should not be sampled,
                    # so that we could clear the sampled tokens before returning.
                    discard_sampled_tokens_req_indices.append(i)

            # NOTE: GPU -> CPU Sync happens here.
            # Move as many CPU operations as possible before this sync point.
            logprobs_tensors = sampler_output.logprobs_tensors
            logprobs_lists = (
                logprobs_tensors.tolists() if logprobs_tensors is not None else None
            )

            # Compute prompt logprobs if needed.
            prompt_logprobs_dict = self._get_prompt_logprobs_dict(
                hidden_states[:num_scheduled_tokens],
                scheduler_output,
            )

            # Get the valid generated tokens.
            sampled_token_ids = sampler_output.sampled_token_ids
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = sampled_token_ids.tolist()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )
            # Mask out the sampled tokens that should not be sampled.
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

            if not self.speculative_config:
                # Speculative decoding is not enabled.
                spec_token_ids = None
            elif self.speculative_config.method == "ngram":
                assert isinstance(self.drafter, NgramProposer)
                spec_token_ids = self.generate_draft_token_ids(
                    valid_sampled_token_ids, sampling_metadata
                )
            elif self.speculative_config.method == "medusa":
                assert isinstance(self.drafter, MedusaProposer)
                if max_gen_len == 1:
                    hidden_states = sample_hidden_states
                else:
                    indices = []
                    offset = 0
                    for num_draft, tokens in zip(
                        spec_decode_metadata.num_draft_tokens, valid_sampled_token_ids
                    ):
                        indices.append(offset + len(tokens) - 1)
                        offset += num_draft + 1

                    indices = torch.tensor(indices, device=sample_hidden_states.device)
                    hidden_states = sample_hidden_states[indices]

                spec_token_ids = self.drafter.propose(
                    target_hidden_states=hidden_states,
                    sampling_metadata=sampling_metadata,
                )
            elif self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                # TODO(woosuk): Refactor the loop.
                next_token_ids: list[int] = []
                for i, token_ids in enumerate(valid_sampled_token_ids):
                    if token_ids:
                        # Common case.
                        next_token_id = token_ids[-1]
                    else:
                        # Partial prefill (rare case).
                        # Get the next token id from the request state.
                        req_id = self.input_batch.req_ids[i]
                        req_state = self.requests[req_id]
                        seq_len = (
                            req_state.num_computed_tokens
                            + scheduler_output.num_scheduled_tokens[req_id]
                        )
                        next_token_id = req_state.get_token_id(seq_len)
                    next_token_ids.append(next_token_id)
                next_token_ids = torch.tensor(
                    next_token_ids, dtype=torch.int32, device=self.device
                )
                # At this moment, we assume all eagle layers belong to the same KV
                # cache group, thus using the same attention metadata.
                eagle_attn_metadata = attn_metadata[self.drafter.attn_layer_names[0]]

                # NOTE: deepseek_mtp uses MLA which does not have `block_table`
                if hasattr(eagle_attn_metadata, "block_table"):
                    block_table = eagle_attn_metadata.block_table
                else:
                    block_table = None

                if spec_decode_metadata is None:
                    # input_ids can be None for multimodal models.
                    target_token_ids = self.input_ids[:num_scheduled_tokens]
                    target_positions = positions[:num_scheduled_tokens]
                    if self.use_aux_hidden_state_outputs:
                        target_hidden_states = torch.cat(
                            [h[:num_scheduled_tokens] for h in aux_hidden_states],
                            dim=-1,
                        )
                    else:
                        target_hidden_states = hidden_states[:num_scheduled_tokens]
                    target_slot_mapping = eagle_attn_metadata.slot_mapping
                    cu_num_tokens = eagle_attn_metadata.query_start_loc
                else:
                    # TODO(woosuk): Refactor this.
                    num_draft_tokens = spec_decode_metadata.num_draft_tokens
                    num_rejected_tokens = [
                        n + 1 - len(valid_sampled_token_ids[i]) if n > 0 else 0
                        for i, n in enumerate(num_draft_tokens)
                    ]
                    num_rejected_tokens_tensor = async_tensor_h2d(
                        num_rejected_tokens,
                        dtype=torch.int32,
                        target_device=self.device,
                        pin_memory=True,
                    )
                    num_tokens = num_scheduled_tokens - sum(num_rejected_tokens)
                    cu_num_tokens, token_indices = self.drafter.prepare_inputs(
                        eagle_attn_metadata.query_start_loc,
                        num_rejected_tokens_tensor,
                        num_tokens,
                    )
                    target_token_ids = self.input_ids[token_indices]
                    target_positions = positions[token_indices]
                    if self.use_aux_hidden_state_outputs:
                        target_hidden_states = torch.cat(
                            [h[token_indices] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[token_indices]
                    target_slot_mapping = eagle_attn_metadata.slot_mapping[
                        token_indices
                    ]
                draft_token_ids = self.drafter.propose(
                    target_token_ids=target_token_ids,
                    target_positions=target_positions,
                    target_hidden_states=target_hidden_states,
                    target_slot_mapping=target_slot_mapping,
                    next_token_ids=next_token_ids,
                    cu_num_tokens=cu_num_tokens,
                    block_table=block_table,
                    sampling_metadata=sampling_metadata,
                )
                spec_token_ids = draft_token_ids.tolist()

            # Clear KVConnector state after all KVs are generated.
            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

            return ModelRunnerOutput(
                req_ids=self.input_batch.req_ids,
                req_id_to_index=self.input_batch.req_id_to_index,
                sampled_token_ids=valid_sampled_token_ids,
                spec_token_ids=spec_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                finished_dumping=finished_dumping,
                invalid_block_ids=invalid_block_ids,
            )

        GPUModelRunner.execute_model = execute_model

        def kv_connector_no_forward(
            self, scheduler_output: "SchedulerOutput"
        ) -> ModelRunnerOutput:
            # KV send/recv even if no work to do.
            with set_forward_context(None, self.vllm_config):
                self.maybe_setup_kv_connector(scheduler_output)
                finished_sending, finished_recving = self.get_finished_kv_transfers(
                    scheduler_output
                )
                invalid_block_ids = self.get_block_ids_with_load_errors()
                get_kv_transfer_group().clear_connector_metadata()

            if not finished_sending and not finished_recving and not invalid_block_ids:
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.finished_sending = finished_sending
            output.finished_recving = finished_recving
            output.invalid_block_ids = invalid_block_ids
            return output

        GPUModelRunner.kv_connector_no_forward = kv_connector_no_forward

        @staticmethod
        def maybe_wait_for_kv_save() -> Optional[dict[str, list[str]]]:
            if has_kv_transfer_group():
                return get_kv_transfer_group().wait_for_save()

        GPUModelRunner.maybe_wait_for_kv_save = maybe_wait_for_kv_save

        def get_block_ids_with_load_errors(self) -> Optional[set[int]]:
            if has_kv_transfer_group():
                return get_kv_transfer_group().get_block_ids_with_load_errors()
            return None

        GPUModelRunner.get_block_ids_with_load_errors = get_block_ids_with_load_errors

    except Exception as e:
        logger.error(f"Failed to patch gpu model runner: {e}")
        raise


def _patch_gpu_worker():
    """
    Apply monkey patch to vllm.v1.worker.gpu_worker
    Adds KV transfer handling in execute_model method.
    """
    try:
        import copy
        import gc
        import os
        from typing import TYPE_CHECKING, Optional

        import torch
        import torch.distributed
        import torch.nn as nn
        import vllm.envs as envs
        from vllm.config import VllmConfig
        from vllm.device_allocator.cumem import CuMemAllocator
        from vllm.distributed import (
            ensure_model_parallel_initialized,
            init_distributed_environment,
            set_custom_all_reduce,
        )
        from vllm.distributed.kv_transfer import (
            ensure_kv_transfer_initialized,
            has_kv_transfer_group,
        )
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        from vllm.logger import init_logger
        from vllm.lora.request import LoRARequest
        from vllm.model_executor import set_random_seed
        from vllm.platforms import current_platform
        from vllm.sequence import IntermediateTensors
        from vllm.utils import GiB_bytes, MemorySnapshot, memory_profiling
        from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
        from vllm.v1.utils import report_usage_stats
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm.v1.worker.gpu_worker import Worker
        from vllm.v1.worker.worker_base import WorkerBase

        @torch.inference_mode()
        def execute_model(
            self,
            scheduler_output: "SchedulerOutput",
        ) -> Optional[ModelRunnerOutput]:
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = IntermediateTensors(
                    get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group())
                )

            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            parallel_config = self.vllm_config.parallel_config
            if (
                parallel_config.distributed_executor_backend != "external_launcher"
                and not get_pp_group().is_last_rank
            ):
                assert isinstance(output, IntermediateTensors)
                get_pp_group().send_tensor_dict(
                    output.tensors, all_gather_group=get_tp_group()
                )
                if not has_kv_transfer_group():
                    return None

                # In case of PP with kv transfer, we need to pass through the
                # finished_sending and finished_recving buffers.
                new_output = EMPTY_MODEL_RUNNER_OUTPUT
                if (
                    output.finished_sending
                    or output.finished_recving
                    or output.finished_dumping
                    or output.invalid_block_ids
                ):
                    new_output = copy.copy(new_output)
                    new_output.finished_sending = output.finished_sending
                    new_output.finished_recving = output.finished_recving
                    new_output.finished_dumping = output.finished_dumping
                    new_output.invalid_block_ids = output.invalid_block_ids
                output = new_output

            assert isinstance(output, ModelRunnerOutput)
            return output

        Worker.execute_model = execute_model
    except Exception as e:
        logger.error(f"Failed to patch gpu worker: {e}")
        raise


def _patch_scheduler() -> None:
    """Patch Scheduler to add num_output_tokens field."""
    try:
        from collections.abc import Iterable
        from typing import Optional

        from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.core.sched.utils import check_stop
        from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
        from vllm.v1.outputs import ModelRunnerOutput
        from vllm.v1.request import Request, RequestStatus
        from vllm.v1.spec_decode.metrics import SpecDecodingStats

        def update_from_output(
            self,
            scheduler_output: SchedulerOutput,
            model_runner_output: ModelRunnerOutput,
        ) -> dict[int, EngineCoreOutputs]:

            sampled_token_ids = model_runner_output.sampled_token_ids
            spec_token_ids = model_runner_output.spec_token_ids
            logprobs = model_runner_output.logprobs
            prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            invalid_block_ids = model_runner_output.invalid_block_ids

            new_running: list[Request] = []
            outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
            spec_decoding_stats: Optional[SpecDecodingStats] = None

            recovered_req_ids = None
            if invalid_block_ids:
                # These blocks contain externally computed tokens that failed to
                # load. Identify affected requests and adjust their computed token
                # count to trigger recomputation of the invalid blocks.
                recovered_req_ids = self._handle_invalid_blocks(invalid_block_ids)

            # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
            # loop can be a performance bottleneck. We should do our best to avoid
            # expensive operations inside the loop.
            for request in self.running:
                req_id = request.request_id
                if recovered_req_ids and req_id in recovered_req_ids:
                    # Skip requests that were recovered from KV load failure
                    new_running.append(request)
                    continue
                num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
                if num_tokens_scheduled == 0:
                    # The request was not scheduled in this step.
                    new_running.append(request)
                    continue

                req_index = model_runner_output.req_id_to_index[req_id]
                generated_token_ids = sampled_token_ids[req_index]

                scheduled_spec_token_ids = (
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id)
                )
                if scheduled_spec_token_ids:
                    # num_computed_tokens represents the number of tokens
                    # processed in the current step, considering scheduled
                    # tokens and rejections. If some tokens are rejected,
                    # num_computed_tokens is decreased by the number of rejected
                    # tokens, where is given by:
                    # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                    num_tokens_rejected = (
                        len(scheduled_spec_token_ids) + 1 - len(generated_token_ids)
                    )
                    request.num_computed_tokens -= num_tokens_rejected
                    spec_decoding_stats = self.make_spec_decoding_stats(
                        spec_decoding_stats,
                        num_draft_tokens=len(scheduled_spec_token_ids),
                        num_accepted_tokens=len(generated_token_ids) - 1,
                    )

                cached_encoder_input_ids = (
                    self.encoder_cache_manager.get_cached_input_ids(request)
                )
                # OPTIMIZATION: Avoid list(set) if the set is empty.
                if cached_encoder_input_ids:
                    for input_id in list(cached_encoder_input_ids):
                        mm_positions = request.mm_positions[input_id]
                        start_pos = mm_positions.offset
                        num_tokens = mm_positions.length
                        if start_pos + num_tokens <= request.num_computed_tokens:
                            # The encoder output is already processed and stored
                            # in the decoder's KV cache.
                            self.encoder_cache_manager.free_encoder_input(
                                request, input_id
                            )

                stopped = False
                new_logprobs = None
                new_token_ids = generated_token_ids
                kv_transfer_params = None
                if model_runner_output.finished_dumping is not None:
                    request.succeed_dumped_blocks.extend(
                        model_runner_output.finished_dumping.get(req_id, [])
                    )
                    is_prefill = request.num_output_tokens == 0
                    if is_prefill:
                        self.connector.connector.commit(
                            model_runner_output.finished_dumping.get(req_id, []), True
                        )

                # Append generated tokens and check for stop. Note that if
                # a request is still being prefilled, we expect the model runner
                # to return empty token ids for the request.
                for num_new, output_token_id in enumerate(new_token_ids, 1):
                    request.append_output_token_ids(output_token_id)

                    # Check for stop and update request state.
                    # This must be called before we make the EngineCoreOutput.
                    stopped = check_stop(request, self.max_model_len)
                    if stopped:
                        kv_transfer_params = self._free_request(request)
                        del new_token_ids[num_new:]  # Trim new tokens if needed.
                        break

                # Extract sample logprobs if needed.
                if request.sampling_params.logprobs is not None and logprobs:
                    # NOTE: once we support N tokens per step (spec decode),
                    # the outer lists can be of length > 1.
                    new_logprobs = logprobs.slice(req_index, req_index + 1)

                if new_token_ids and self.structured_output_manager.should_advance(
                    request
                ):
                    # NOTE: structured_output_request
                    # should not be None if use_structured_output, we have
                    # check above, so safe to ignore type warning
                    request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                        req_id, new_token_ids
                    )

                # Add newly generated spec token ids to the request.
                if spec_token_ids is not None:
                    if self.structured_output_manager.should_advance(request):
                        metadata = request.structured_output_request
                        # Needs to happen after new_token_ids are accepted.
                        request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                            spec_token_ids[req_index]
                        )
                    else:
                        request.spec_token_ids = spec_token_ids[req_index]

                # Get prompt logprobs for this request.
                prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
                if new_token_ids or kv_transfer_params:

                    # Add EngineCoreOutput for this Request.
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=req_id,
                            new_token_ids=new_token_ids,
                            finish_reason=request.get_finished_reason(),
                            new_logprobs=new_logprobs,
                            new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                            stop_reason=request.stop_reason,
                            events=request.take_events(),
                            kv_transfer_params=kv_transfer_params,
                            num_cached_tokens=request.num_cached_tokens,
                        )
                    )

                else:
                    # Invariant: EngineCore returns no partial prefill outputs.
                    assert not prompt_logprobs_tensors

                if not stopped:
                    new_running.append(request)

            # self.running = new_running

            # KV Connector: update state for finished KV Transfers.
            self._update_from_kv_xfer_finished(model_runner_output)

            # Return the cached request data to the queue so they can be reused.
            for req_data in scheduler_output.scheduled_cached_reqs:
                # NOTE(rob): since we free stopped reqs above, adding stopped reqs
                # to _cached_reqs_data will cause a memory leak.
                if req_data.req_id not in self.finished_req_ids:
                    self._cached_reqs_data[req_data.req_id].append(req_data)

            self.running = new_running

            # Create EngineCoreOutputs for all clients that have requests with
            # outputs in this step.
            engine_core_outputs = {
                client_index: EngineCoreOutputs(outputs=outs)
                for client_index, outs in outputs.items()
            }

            finished_req_ids = self.finished_req_ids_dict
            if finished_req_ids:
                # Include ids of requests that finished since last outputs
                # were sent.
                for client_index, finished_set in finished_req_ids.items():
                    # Set finished request set in EngineCoreOutputs for this client.
                    if (eco := engine_core_outputs.get(client_index)) is not None:
                        eco.finished_requests = finished_set
                    else:
                        engine_core_outputs[client_index] = EngineCoreOutputs(
                            finished_requests=finished_set
                        )
                finished_req_ids.clear()

            if engine_core_outputs:
                # Return stats to only one of the front-ends.
                next(iter(engine_core_outputs.values())).scheduler_stats = (
                    self.make_stats(spec_decoding_stats)
                )

            return engine_core_outputs

        Scheduler.update_from_output = update_from_output

        def _update_requests_with_invalid_blocks(
            self, requests: Iterable[Request], invalid_block_ids: set[int]
        ) -> tuple[set[Request], int, set[int]]:
            affected_requests: set[Request] = set()
            num_tokens_to_reschedule = 0
            # If a block is invalid and shared by multiple requests in the batch,
            # all requests must be rescheduled, but only the first will recompute
            # it. This set tracks blocks already marked for recomputation.
            marked_invalid_block_ids: set[int] = set()
            for request in requests:
                is_affected = False
                marked_invalid_block = False
                req_id = request.request_id
                req_block_ids = self.kv_cache_manager.get_block_ids(req_id)[0]
                # We iterate only over blocks that may contain externally computed
                # tokens
                if request.num_cached_tokens > 0:
                    req_num_computed_blocks = (
                        request.num_cached_tokens + self.block_size - 1
                    ) // self.block_size
                else:
                    req_num_computed_blocks = len(req_block_ids)

                for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):

                    if block_id not in invalid_block_ids:
                        continue

                    is_affected = True

                    if block_id in marked_invalid_block_ids:
                        # This invalid block is shared with a previous request
                        # and was already marked for recomputation.
                        # This means this request can still consider this block
                        # as computed when rescheduled.
                        continue

                    marked_invalid_block_ids.add(block_id)

                    if marked_invalid_block:
                        # This request has already marked an invalid block for
                        # recomputation and updated its num_computed_tokens.
                        continue

                    marked_invalid_block = True
                    num_tokens_to_reschedule += request.num_computed_tokens
                    request.num_computed_tokens = idx * self.block_size
                    num_tokens_to_reschedule -= request.num_computed_tokens

                if is_affected:
                    if not marked_invalid_block:
                        # All invalid blocks of this request are shared with
                        # previous requests and will be recomputed by them.
                        # Revert to considering only cached tokens as computed.
                        num_tokens_to_reschedule += (
                            request.num_computed_tokens - request.num_cached_tokens
                        )
                        request.num_computed_tokens = request.num_cached_tokens

                    affected_requests.add(request)

            return (
                affected_requests,
                num_tokens_to_reschedule,
                marked_invalid_block_ids,
            )

        Scheduler._update_requests_with_invalid_blocks = (
            _update_requests_with_invalid_blocks
        )

        def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
            total_requests_to_reschedule = 0
            total_tokens_to_reschedule = 0

            # --- Handle async KV loads (WAITING_FOR_REMOTE_KVS) ---
            async_load_reqs = (
                req
                for req in self.waiting
                if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            )
            (affected_requests, num_tokens_to_reschedule, marked_invalid_block_ids) = (
                self._update_requests_with_invalid_blocks(
                    async_load_reqs, invalid_block_ids
                )
            )

            total_requests_to_reschedule += len(affected_requests)
            total_tokens_to_reschedule += num_tokens_to_reschedule

            for request in affected_requests:
                if request.num_computed_tokens:
                    # Cache any valid computed tokens.
                    self.kv_cache_manager.cache_blocks(
                        request, request.num_computed_tokens
                    )
                else:
                    # No valid computed tokens, release allocated blocks.
                    # There may be a local cache hit on retry.
                    self.kv_cache_manager.free(request)

                request.status = RequestStatus.WAITING

            # Remove async loaded invalid blocks already handled,
            # as they cannot be shared with running requests.
            invalid_block_ids.difference_update(marked_invalid_block_ids)

            # --- Handle sync KV loads (running requests) ---
            affected_requests, num_tokens_to_reschedule, _ = (
                self._update_requests_with_invalid_blocks(
                    self.running, invalid_block_ids
                )
            )

            total_requests_to_reschedule += len(affected_requests)
            total_tokens_to_reschedule += num_tokens_to_reschedule

            if total_requests_to_reschedule:
                logger.info(
                    "Recovered from KV load failure: "
                    "%d request(s) rescheduled (%d tokens affected).",
                    total_requests_to_reschedule,
                    total_tokens_to_reschedule,
                )

            # Return the IDs of affected running requests to skip in
            # update_from_output.
            return {r.request_id for r in affected_requests}

        Scheduler._handle_invalid_blocks = _handle_invalid_blocks

    except ImportError:
        logger.warning("Could not patch Scheduler - module not found")
