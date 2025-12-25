"""
UCM Sparse State Management with Single Agent Pattern

This module provides global state management for UCM sparse using a single agent,
similar to KV connector pattern. It allows the scheduler and worker to access
the same UCM sparse agent across different processes.
"""

from typing import TYPE_CHECKING, Optional

import torch

from ucm.logger import init_logger
from ucm.sparse.base import UcmSparseBase, UcmSparseRole
from ucm.sparse.factory import UcmSparseFactory
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Global UCM sparse agent instance
_UCM_SPARSE_AGENT: Optional[UcmSparseBase] = None


def ensure_ucm_sparse_initialized(
    vllm_config: "VllmConfig", role: UcmSparseRole = UcmSparseRole.WORKER
) -> None:
    """
    Initialize UCM sparse agent for the given role.

    Args:
        vllm_config: vLLM configuration
        role: UCM sparse role (SCHEDULER or WORKER)
    """
    global _UCM_SPARSE_AGENT

    if vllm_config.kv_transfer_config is None:
        return

    # Check if UCM sparse is enabled
    ucm_config = Config(vllm_config.kv_transfer_config)
    ucm_sparse_config = ucm_config.get_config().get("ucm_sparse_method")
    if not ucm_sparse_config:
        return

    sparse_method_name = ucm_sparse_config

    if _UCM_SPARSE_AGENT is None:
        logger.info("Initializing UCM sparse agent with method: %s", sparse_method_name)
        _UCM_SPARSE_AGENT = UcmSparseFactory.create_sparse_method(
            vllm_config, role=role
        )
    else:
        # Update role if needed (for debugging/logging purposes)
        logger.debug(
            "UCM sparse agent already initialized, current role: %s",
            _UCM_SPARSE_AGENT._role,
        )


def get_ucm_sparse() -> UcmSparseBase:
    """Get the current UCM sparse agent instance."""
    global _UCM_SPARSE_AGENT

    if _UCM_SPARSE_AGENT is None:
        raise RuntimeError("UCM sparse agent is not initialized")

    return _UCM_SPARSE_AGENT


def has_ucm_sparse() -> bool:
    """Check if UCM sparse agent is available."""
    global _UCM_SPARSE_AGENT
    return _UCM_SPARSE_AGENT is not None


def maybe_execute_sparse_layer_begin(
    positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor
):
    if not has_ucm_sparse():
        return positions, hidden_states, residual
    ucm_spare = get_ucm_sparse()
    return ucm_spare.layer_begin(positions, hidden_states, residual)


def maybe_execute_sparse_layer_finished(
    positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor
):
    if not has_ucm_sparse():
        return positions, hidden_states, residual
    ucm_spare = get_ucm_sparse()
    return ucm_spare.layer_finished(positions, hidden_states, residual)


def maybe_execute_sparse_ffn_begin(hidden_states: torch.Tensor, residual: torch.Tensor):
    if not has_ucm_sparse():
        return hidden_states, residual
    ucm_spare = get_ucm_sparse()
    return ucm_spare.ffn_begin(hidden_states, residual)


def maybe_execute_sparse_ffn_finished(
    hidden_states: torch.Tensor, residual: torch.Tensor
):
    if not has_ucm_sparse():
        return hidden_states, residual
    ucm_spare = get_ucm_sparse()
    return ucm_spare.ffn_finished(hidden_states, residual)
