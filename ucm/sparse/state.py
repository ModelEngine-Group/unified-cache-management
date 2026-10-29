"""
UCM Sparse State Management with Single Agent Pattern

This module provides global state management for UCM sparse using a single agent,
similar to KV connector pattern. It allows the scheduler and worker to access
the same UCM sparse agent across different processes.
"""

from typing import TYPE_CHECKING, Optional

import torch
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext

from ucm.logger import init_logger
from ucm.sparse.base import UcmSparseBase, UcmSparseRole
from ucm.sparse.factory import UcmSparseFactory
from ucm.sparse.utils import IS_OPEN_GSA
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

    The sparse method is selected from the user's ``kv_transfer_config``
    (``ucm_sparse_config``), restoring the config-driven behavior so that
    ESA / KVStar / GSAOnDevice / Blend can still be selected by their own
    configurations. GSA / GSAPD additionally require the explicit
    ``IS_OPEN_GSA`` switch (see ``ucm.sparse.utils``).

    Args:
        vllm_config: vLLM configuration
        role: UCM sparse role (SCHEDULER or WORKER)
    """
    global _UCM_SPARSE_AGENT

    if vllm_config.kv_transfer_config is None:
        return

    # Read the user-configured sparse method (config-driven).
    ucm_config = Config(vllm_config.kv_transfer_config)
    ucm_sparse_config = ucm_config.get_config().get("ucm_sparse_config")
    if not ucm_sparse_config:
        return

    sparse_method_name = next(iter(ucm_sparse_config))

    # GSA / GSAPD are opt-in: require the explicit IS_OPEN_GSA switch so that
    # non-GSA deployments are unaffected.
    if sparse_method_name in ("GSA", "GSAPD") and not IS_OPEN_GSA:
        logger.info(
            "Sparse method %s selected but GSA is disabled; "
            "set UCM_SPARSE_GSA_ENABLE=1 to enable it. Skip initialization.",
            sparse_method_name,
        )
        return

    if _UCM_SPARSE_AGENT is None:
        logger.info("Initializing UCM sparse agent with method: %s", sparse_method_name)
        _UCM_SPARSE_AGENT = UcmSparseFactory.create_sparse_method(
            vllm_config,
            ucm_sparse_config,
            role=role,
        )
    else:
        # Update role if needed (for debugging/logging purposes)
        logger.debug(
            "UCM sparse agent already initialized, current role: %s",
            _UCM_SPARSE_AGENT._role,
        )


def ensure_ucm_sparse_shutdown(role: UcmSparseRole = UcmSparseRole.WORKER) -> None:
    if not has_ucm_sparse():
        return
    ucm_spare = get_ucm_sparse()
    ucm_spare.shutdown()


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


def maybe_execute_sparse_attention_begin(
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
):
    if not has_ucm_sparse():
        return query, key, value, output

    ucm_sparse = get_ucm_sparse()

    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return query, key, value, output

    return ucm_sparse.attention_begin(
        query,
        key,
        value,
        layer_name,
        forward_context,
        output,
        phase,
        k_hash,
        decode_ql_nope,
        decode_q_pe,
    )


def maybe_execute_sparse_attention_finished(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    layer_name: str,
    forward_context: ForwardContext,
    phase: Optional[str] = None,
):
    if not has_ucm_sparse():
        return

    ucm_sparse = get_ucm_sparse()

    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return

    ucm_sparse.attention_finished(
        query, key, value, attn_output, layer_name, forward_context, phase
    )
