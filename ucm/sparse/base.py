"""
UcmSparseBase Class provides interfaces for general sparse attention algorithm implementation in vLLM.

The class provides the following primitives:
    Scheduler-side: runs in the scheduler, binds metadata, which
    is used by the worker-side to retrieval/load KV cache.
        estimate_num_slots_sparsed() - get the number of required slots.
        update_state_after_alloc() - update UcmSparse state after
            temporary buffer alloc by the CacheManager.
        request_finished_in_scheduler() - called when a request is finished, with
            the computed kv cache blocks for the request.
            Returns metadata for the next step.

    Worker-side: runs in each worker, retrieval/load KV cache.
        execute_begin() - hook at the beginning of "ModelRunner->execute_model".
        execute_finished() - hook at the end of "ModelRunner->execute_model".
        attention_begin() - hook at the beginning of "unified_attention".
        attention_finished() - hook at the end of "unified_attention".
        request_finished_in_worker() - release the resources, like block features.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request
    from vllm.config import VllmConfig

import torch
from vllm.forward_context import ForwardContext

INVALID_SLOT = -1


class UcmSparseRole(enum.Enum):
    # sparser running in the scheduler process
    SCHEDULER = 0

    # sparser running in the worker process
    WORKER = 1


class UcmSparseMetadata(ABC):  # noqa: B024
    """
    Abstract Metadata used to communicate between the
    Scheduler UcmSparse instance and Worker UcmSparse instance.
    """

    pass


class UcmSparseCpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU. Inferred by vLLM."""

    def __init__(
        self,
        *size: Union[int, torch.SymInt],
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool = True,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size, dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.gpu = self.cpu.to(device)
        self.np: np.ndarray
        self.n = 0

        if with_numpy:
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call UcmSparseCpuGpuBuffer with with_numpy=False"
                )
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: Optional[int] = None) -> None:
        # TODO: replace with esa_copy
        if n is None:
            n = self.n
        if n <= 0:
            return
        self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    def copy_to_cpu(self, n: Optional[int] = None) -> None:
        # TODO: replace with esa_copy
        """NOTE: Because this method is non-blocking, explicit synchronization
        is needed to ensure the data is copied to CPU."""
        if n is None:
            n = self.n
        if n <= 0:
            return
        self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)

    def append_numpy(self, data: List[Any]) -> None:
        size = len(data)
        assert (
            self.np is not None
        ), "append_numpy meed to be initialized by with_numpy=True."
        assert self.n + size < self.cpu.shape[0], "append_numpy data out of range."
        self.np[self.n : self.n + size] = data
        self.n += size

    def clear(self) -> None:
        self.n = 0

    @property
    def size(self) -> int:
        return self.n

    @property
    def valid_np(self) -> np.ndarray:
        return self.np[: self.n]

    @property
    def valid_cpu(self) -> torch.Tensor:
        return self.cpu[: self.n]

    @property
    def valid_gpu(self) -> torch.Tensor:
        return self.gpu[: self.n]


class UcmSparseBase(ABC):
    """
    An general interface for impl sparse attention algorithm in vLLM
    """

    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        self._sparse_metadata: Optional[UcmSparseMetadata] = None
        self._vllm_config = vllm_config
        self._role = role

    @property
    def role(self) -> UcmSparseRole:
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_sparse_metadata(self, sparse_metadata: UcmSparseMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self._sparse_metadata = sparse_metadata

    def clear_sparse_metadata(self) -> None:
        """Clear the sparse metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self._sparse_metadata = None

    def _get_sparse_metadata(self) -> UcmSparseMetadata:
        """Get the sparse metadata.

        This function should only be called inside the UCMSparse.

        Returns:
            SparseMetadata: the UCM sparse metadata.
        """

        # Should only be called while set to valid metadata.
        assert self._sparse_metadata is not None
        return self._sparse_metadata

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        pass

    def execute_begin(self, scheduler_output: SchedulerOutput):
        """
        This is called at the beginning of "ModelRunner->execute_model" function.
        """
        pass

    def execute_finished(self, logits_indices: torch.Tensor) -> torch.Tensor:
        """
        This is called at the end of "ModelRunner->execute_model" function.
        """
        return logits_indices

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        output: Optional[torch.Tensor] = None,
        phase: Optional[str] = None,
        k_hash: Optional[torch.Tensor] = None,
        decode_ql_nope: Optional[torch.Tensor] = None,
        decode_q_pe: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This is called at the beginning of "unified_attention".
        Sparse attention algorithm can modify forward_context.attn_metadata if necessary.
        (UC_TODO: modify dataclass is not allowed in python?)
        """
        return query, key, value, output

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        phase: Optional[str] = None,
    ) -> None:
        """
        This is called at the end of "unified_attention".
        """
        pass

    def update_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        This is called at the beginning of "ModelRunner->execute_model" function.
        Update the cached states with the scheduler output.
        """
        pass

    def ffn_begin(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is called at the beginning of ffn in each DecodeLayer.
        """
        return hidden_states, residual

    def ffn_finished(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is called at the end of ffn in each DecodeLayer.
        """
        return hidden_states, residual

    def layer_begin(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This is called at the beginning of DecodeLayer.
        """
        return positions, hidden_states, residual

    def layer_finished(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This is called at the end of DecodeLayer.
        """
        return positions, hidden_states, residual

    def request_finished_in_worker(self, request_id: Union[int, str]):
        """
        This function releases the resources of finished requests at worker-side.
        """
        pass

    # ==============================
    # Scheduler-side methods
    # ==============================

    @abstractmethod
    def request_begin(self, request_id: Union[int, str], prompt_token_ids: List[int]):
        """
        This is called at the beginning of "Scheduler->add_request" function.
        """
        pass

    def request_finished_in_scheduler(self, request_id: Union[int, str]):
        """
        This is called inside "Scheduler->finish_requests" function.
        Generate the metadata required by UcmSparse instance at worker-side.
        """
        pass

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        """
        This is called by "Scheduler->schedule" function to estimate the number of required blocks.
        """
        pass

    def update_state_after_alloc(self, request: Request, num_blocks: int):
        """
        Update UcmSparse state after block allocation.
        """
        pass

    def build_sparse_meta(
        self, scheduler_output, requests, input_batch, attn_metadata
    ) -> UcmSparseMetadata:
        """
        Build the sparse metadata for this step.
        """
        pass

    def allocate_slots(self, kv_cache_manager, request, num_slots_sparsed):
        pass
