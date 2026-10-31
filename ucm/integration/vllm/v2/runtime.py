"""Shared runtime context, parallel topology and storage identity contract."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorRole,
)

from ucm.logger import init_logger
from ucm.utils import Config

from .layout import (
    RestorePolicy,
    UCMKVCacheSpec,
    parse_kv_cache_config,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


logger = init_logger(__name__)


# Development format revision; compatibility with old records is not maintained.
UCM_CHUNK_ID_VERSION = 6


class ParallelRank(NamedTuple):
    """This worker's coordinates inside the engine's parallel groups."""

    tp_rank: int = 0
    pp_rank: int = 0
    pcp_rank: int = 0


@dataclass(frozen=True)
class ParallelTopology:
    tp_size: int = 1
    pp_size: int = 1
    pcp_size: int = 1
    dcp_size: int = 1
    cp_kv_cache_interleave_size: int = 1
    pp_partition: str = ""

    @classmethod
    def from_config(cls, config: VllmConfig) -> ParallelTopology:
        parallel_config = config.parallel_config
        # Engine configuration validation owns dimensions and legal TP/PCP/DCP
        # combinations. Only project the already-resolved configuration here.
        return cls(
            tp_size=parallel_config.tensor_parallel_size,
            pp_size=parallel_config.pipeline_parallel_size,
            pcp_size=parallel_config.prefill_context_parallel_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            cp_kv_cache_interleave_size=parallel_config.cp_kv_cache_interleave_size,
            pp_partition=os.getenv("VLLM_PP_LAYER_PARTITION", ""),
        )

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size * self.pcp_size


@dataclass(frozen=True)
class UCMStorageHandshake(KVConnectorHandshakeMetadata):
    rank: ParallelRank
    fingerprint: str


def make_common_ucm_chunk_ids(chunk_hashes: Iterable[bytes]) -> tuple[bytes, ...]:
    return tuple(value[:15] for value in chunk_hashes)


def make_ucm_chunk_ids(
    common_ucm_chunk_ids: Sequence[bytes],
    restore_policy: RestorePolicy = RestorePolicy.FA,
) -> tuple[bytes, ...]:
    """A common chunk hash plus one fixed restore-policy tag, shared by all ranks."""
    tag = restore_policy.tag
    return tuple(value + tag for value in common_ucm_chunk_ids)


def _load_launch_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Resolve kv_connector_extra_config through the shared ucm.utils.Config.

    The UCM_CONFIG_FILE yaml and terminal-input forms resolve exactly like
    every other UCM connector, including the v2 Store/Proxy configuration.
    """
    return dict(Config(vllm_config.kv_transfer_config).get_config())


def _worker_rank() -> int:
    from vllm.distributed.parallel_state import get_world_group

    return get_world_group().rank


def _worker_parallel_ranks(topology: ParallelTopology) -> ParallelRank:
    """This worker's (tp_rank, pp_rank, pcp_rank), read from vLLM's groups."""
    from vllm.distributed.parallel_state import (
        get_pcp_group,
        get_pp_group,
        get_tp_group,
    )

    pcp_rank = get_pcp_group().rank_in_group if topology.pcp_size > 1 else 0
    return ParallelRank(
        tp_rank=get_tp_group().rank_in_group,
        pp_rank=get_pp_group().rank_in_group,
        pcp_rank=pcp_rank,
    )


def _dump_raw_kv_cache_config(kv_cache_config: KVCacheConfig, rank: int | None) -> None:
    """Write the raw KVCacheConfig vLLM handed over to a JSON file.

    Enabled by ``UCM_V2_DUMP_CONFIG=<path>``; a ``%d`` in the path receives
    the worker rank so tensor-parallel shards can be captured side by side.
    """

    path_template = os.environ.get("UCM_V2_DUMP_CONFIG", "")
    if not path_template:
        return
    path = Path(
        path_template % rank
        if "%d" in path_template or "%s" in path_template
        else path_template
    )

    # Native configuration, group specs and tensor descriptors are dataclasses.
    # JSON's leaf fallback is needed for dtype and enum values only.
    payload = asdict(kv_cache_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, indent=2, ensure_ascii=False, default=str)
    print(
        f"[ucm-v2] raw KVCacheConfig dumped to {path} "
        f"(num_blocks={payload['num_blocks']}, "
        f"tensors={len(kv_cache_config.kv_cache_tensors)}, "
        f"groups={len(kv_cache_config.kv_cache_groups)})",
        flush=True,
    )


@dataclass(frozen=True)
class UCMRuntimeContext:
    """Stable configuration, process identity and logical KV contract.

    Constructed independently for each role; owns no requests, physical KV
    views, raw engine configurations or transfer lifetimes.
    """

    role: KVConnectorRole
    device_type: str
    engine_id: str | None
    global_rank: int | None
    rank: ParallelRank | None
    parallel_topology: ParallelTopology
    launch_config: dict[str, Any]
    use_layerwise: bool
    spec: UCMKVCacheSpec
    fingerprint: str

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> UCMRuntimeContext:
        launch_config = _load_launch_config(vllm_config)
        # vLLM's resolved scheduler granularity, not the raw CacheConfig
        # page size: multiple groups hash the LCM of group block sizes (the
        # resolve call is monkey-patched by vllm-ascend inside its process).
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

        scheduler_block_size, _ = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        chunk_size = launch_config.get("chunk_size")
        if chunk_size is not None:
            chunk_size = int(chunk_size)
        worker_rank = None if role == KVConnectorRole.SCHEDULER else _worker_rank()
        _dump_raw_kv_cache_config(kv_cache_config, worker_rank)
        parallel_topology = ParallelTopology.from_config(vllm_config)
        rank = (
            ParallelRank(*_worker_parallel_ranks(parallel_topology))
            if role == KVConnectorRole.WORKER
            else None
        )
        device_type = vllm_config.device_config.device_type
        text_config = vllm_config.model_config.hf_text_config
        spec: UCMKVCacheSpec = parse_kv_cache_config(
            kv_cache_config,
            scheduler_block_size=scheduler_block_size,
            hf_text_config=text_config,
            chunk_size=chunk_size,
            dcp_size=parallel_topology.dcp_size,
            device_type=device_type,
            # Longcat has two attention modules per decoder layer.
            num_attn_module=(
                2
                if vllm_config.model_config.hf_config.model_type == "longcat_flash"
                else 1
            ),
        )
        model_dtype = str(vllm_config.model_config.dtype).rsplit(".", 1)[-1]
        speculative_config = vllm_config.speculative_config
        additional_config = vllm_config.additional_config or {}
        use_layerwise = bool(launch_config.get("use_layerwise", False))
        fingerprint_schema = {
            "device_type": device_type,
            "model_dtype": model_dtype,
            "chunk_size": spec.chunk_size,
            "parallel_topology": asdict(parallel_topology),
            "format_version": UCM_CHUNK_ID_VERSION,
            "use_layerwise": use_layerwise,
            "model_identity": {
                "model": vllm_config.model_config.model.rstrip("/"),
                "revision": vllm_config.model_config.revision,
                "code_revision": vllm_config.model_config.code_revision,
            },
            "speculative": (
                {
                    "method": speculative_config.method,
                    "num_speculative_tokens": speculative_config.num_speculative_tokens,
                }
                if speculative_config is not None
                else None
            ),
            "sparse": {
                "enable_sparse_sfa_c8": bool(
                    additional_config.get("enable_sparse_sfa_c8", False)
                ),
                "enable_sparse_li_c8": bool(
                    additional_config.get("enable_sparse_li_c8", False)
                ),
            },
            "model": vllm_config.model_config.hf_text_config.to_dict(),
            "kv_dtype": vllm_config.cache_config.cache_dtype,
            "mamba_dtype": vllm_config.cache_config.mamba_cache_dtype,
            "mamba_ssm_dtype": vllm_config.cache_config.mamba_ssm_cache_dtype,
            "kv_cache_layout": kv_cache_config.kv_cache_layout,
            # vLLM decides which groups are stored; a change makes new records.
            "stored_groups": [
                (group.group_id, group.block_size)
                for group in spec.groups
                if group.is_persistent
            ],
            "ascend_gqa_head_first": (
                envs.VLLM_KV_CACHE_LAYOUT in ("LBHNC", "HND")
                if device_type == "npu"
                else None
            ),
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_schema, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        logger.info(
            "UCM fingerprint=%s model=%s device=%s dtype=%s chunk_size=%d "
            "topology=%s layerwise=%s",
            fingerprint,
            vllm_config.model_config.model,
            device_type,
            model_dtype,
            spec.chunk_size,
            parallel_topology,
            use_layerwise,
        )
        return cls(
            role=role,
            device_type=device_type,
            engine_id=vllm_config.instance_id,
            global_rank=worker_rank,
            rank=rank,
            parallel_topology=parallel_topology,
            launch_config=launch_config,
            use_layerwise=use_layerwise,
            spec=spec,
            fingerprint=fingerprint,
        )
