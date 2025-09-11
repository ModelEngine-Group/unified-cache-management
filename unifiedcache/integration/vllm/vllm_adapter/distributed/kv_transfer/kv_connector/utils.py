from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import CancelledError, Future
from typing import Optional, cast

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm_adapter.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)



class KVOutputAggregator:
    """Utility class to aggregate the output of all workers into a single 
    output corresponding to Rank 0 for scheduler."""

    def __init__(self, world_size: int):
        # Complete transfer tracker. Used by to track finished requests
        # [req_id -> n_finished_workers]
        self._recv_remaining_count = defaultdict[str, int](lambda: world_size)
        self._send_remaining_count = defaultdict[str, int](lambda: world_size)
        self._dump_remaining_count = defaultdict[str, int](lambda: world_size)

    def aggregate(self,
                  outputs: list[ModelRunnerOutput],
                  output_rank: int = 0) -> ModelRunnerOutput:
        # aggregate finished_sending, finished_recving from all workers

        def update_finished_set(req_ids: Optional[set[str]],
                                remaining_count_dict: dict[str, int],
                                finished_set: set[str]) -> None:
            for req_id in req_ids or ():
                new_count = remaining_count_dict[req_id] - 1
                if new_count == 0:
                    finished_set.add(req_id)
                    del remaining_count_dict[req_id]
                else:
                    remaining_count_dict[req_id] = new_count
        
        def update_finished_list(req_ids: Optional[dict[str, list[str]]],
                                 remaining_count_dict: dict[str, int],
                                 finished_list: dict[str, list[str]]) -> None:
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
        finished_dumping: dict[str, list[str]] = {}
        for output in outputs:
            update_finished_set(output.finished_sending,
                                self._send_remaining_count, finished_sending)
            update_finished_set(output.finished_recving,
                                self._recv_remaining_count, finished_recving)
            update_finished_list(output.finished_dumping,
                                self._dump_remaining_count, finished_dumping)

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

        return output

    def async_aggregate(self,
                        output_futures: Sequence[Future[ModelRunnerOutput]],
                        output_rank: int = 0) -> Future[ModelRunnerOutput]:
        """Takes a list of futures and returns a single future which resolves
        to the respective list of outputs."""
        result_future: Future[ModelRunnerOutput] = Future()

        outputs: list[Optional[ModelRunnerOutput]] = [None
                                                      ] * len(output_futures)

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
                        self.aggregate(cast(list[ModelRunnerOutput], outputs),
                                       output_rank))

            return callback

        for i, output_future in enumerate(output_futures):
            output_future.add_done_callback(make_callback(i))

        return result_future
