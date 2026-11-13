"""UCM-backed vLLM encoder-cache connector.

The connector submits loads before waiting and drains saves at the end of a step.
An encoder-cache item is a variable-length ``[N, D]`` tensor, while UCM stores
fixed-size blocks, so each item is represented by one or more fixed-row chunks.
Scheduler state carries
only the chunk IDs required by a selected external hit; workers reconstruct
the tensor from padded chunk storage or dump a newly computed tensor.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
)
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.integration.vllm.cache_config import (
    cache_rank_count,
    configure_cache_capacity,
    configure_cache_numa,
)
from ucm.integration.vllm.device import (
    create_device,
    get_current_device_id,
    get_ucm_worker_torch_device,
)
from ucm.integration.vllm.request_hasher import RequestHasher
from ucm.integration.vllm.ucm_connector import (
    _check_shm_capacity,
    _get_store_gc_block_size,
    _get_store_io_sizes,
    _scheduler_read_unique_id,
    _worker_generate_unique_id,
)
from ucm.logger import init_logger
from ucm.store.cache.capabilities import uses_partitioned_cache
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

_EPD_LOOKUP_INTERVAL_S = 0.01


class EncoderCacheLayoutError(ValueError):
    """Raised when scheduler and worker cannot share one fixed EC layout."""


class UCMEncoderCacheError(RuntimeError):
    """Runtime failure tied to a specific encoder-cache identifier."""

    def __init__(self, identifier: str, detail: str) -> None:
        super().__init__(f"{detail}: {identifier}")
        self.identifier = identifier


@dataclass(frozen=True, slots=True)
class EncoderCacheLayout:
    """EncoderCache tensor shape: [N,D]"""

    # Embedding width per row in encoder cache, D.
    width: int
    dtype: torch.dtype
    rows_per_chunk: int
    chunk_bytes: int


@dataclass(slots=True)
class RequestState:
    processed_feature_count: int = 0
    identifier_states: dict[str, IdentifierState] = field(default_factory=dict)
    # Store hits in ordinary mode; external load selections in EPD mode.
    hit_identifiers: set[str] = field(default_factory=set)
    # Request-local positive lookup snapshots, not retained or loaded EC items.
    store_visible_identifiers: set[str] = field(default_factory=set)
    store_wait_started_at: float | None = None
    next_lookup_at: float = 0.0


@dataclass(slots=True)
class IdentifierState:
    num_embeds: int
    chunk_ids: tuple[bytes, ...]
    request_ref_count: int = 0


@dataclass(frozen=True, slots=True)
class ECLoadSpec:
    num_embeds: int
    chunk_ids: tuple[bytes, ...]


@dataclass(slots=True)
class PendingECSave:
    identifier: str
    task: Task
    tensor: torch.Tensor
    tail: torch.Tensor | None
    event_handle: int


@dataclass
class UCMECConnectorMetadata(ECConnectorMetadata):
    loads: dict[str, ECLoadSpec] = field(default_factory=dict)
    save_identifiers: tuple[str, ...] = ()


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _configured_encoder_width(model: Any) -> int:
    # Preserve the old CPU connector helper's rule without importing its
    # removed private symbol. Do not substitute the LM hidden_size directly.
    vision = getattr(getattr(model, "hf_config", None), "vision_config", None)
    layers = getattr(vision, "deepstack_visual_indexes", None)
    output_width = getattr(vision, "out_hidden_size", None)
    if layers and output_width is not None:
        return _positive_int("vision out_hidden_size", output_width) * (1 + len(layers))
    return _positive_int(
        "encoder input embedding width", model.get_inputs_embeds_size()
    )


def resolve_encoder_cache_width(vllm_config: Any, encoder_config: dict) -> int:
    """Validate the fixed layout against measured widths when available.

    Measurements can be absent on consumers and on workers initialized before
    the producer's profiling RPC. Fixed-layout deployments must use compatible
    model configs (or an explicit width) on both sides. A measurement must not
    silently change the scheduler's layout after workers opened their Store.
    Validate actual tensors at the existing save boundary as well.
    """
    explicit = encoder_config.get("encoder_cache_hidden_dim")
    if explicit is not None:
        configured = _positive_int("encoder_cache_hidden_dim", explicit)
    else:
        configured = _configured_encoder_width(vllm_config.model_config)

    transfer = getattr(vllm_config, "ec_transfer_config", None)
    measured = getattr(transfer, "mm_encoder_output_widths", None)
    if measured is not None and not isinstance(measured, dict):
        raise ValueError("mm_encoder_output_widths must be a mapping.")
    if measured:
        widths = {
            _positive_int(f"encoder output width for {modality}", width)
            for modality, width in measured.items()
        }
        if len(widths) != 1:
            raise ValueError(
                "UCM EC requires one fixed encoder output width across modalities; "
                f"got {measured!r}."
            )
        width = widths.pop()
        if configured != width:
            raise ValueError(
                f"Configured/inferred EC width={configured} disagrees with measured "
                f"encoder output width={width}. Set encoder_cache_hidden_dim={width} "
                "consistently on E and PD before starting their workers."
            )
        return width
    return configured


def resolve_encoder_cache_layout(
    vllm_config: VllmConfig,
    encoder_config: dict[str, Any],
) -> EncoderCacheLayout:
    model_config = vllm_config.model_config
    try:
        width = resolve_encoder_cache_width(vllm_config, encoder_config)
    except ValueError as exc:
        raise EncoderCacheLayoutError(str(exc)) from exc

    dtype = model_config.dtype
    element_size = int(dtype.itemsize)

    rows_per_chunk = encoder_config["chunk_size"]
    if (
        not isinstance(rows_per_chunk, int)
        or isinstance(rows_per_chunk, bool)
        or rows_per_chunk <= 0
    ):
        raise EncoderCacheLayoutError(
            f"chunk_size must be a positive number of rows, got {rows_per_chunk!r}."
        )
    chunk_bytes = rows_per_chunk * width * element_size

    return EncoderCacheLayout(
        width=width,
        dtype=dtype,
        rows_per_chunk=rows_per_chunk,
        chunk_bytes=chunk_bytes,
    )


def build_ec_hash_meta(
    vllm_config: VllmConfig,
    encoder_config: dict[str, Any],
) -> str:
    """Build EC identity metadata without KV-specific deployment settings."""
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    explicit = encoder_config.get("cache_namespace")
    if explicit is not None:
        namespace_source: str | tuple = str(explicit)
    else:
        mm_config = model_config.multimodal_config
        mm_hash = mm_config.compute_hash() if mm_config is not None else None
        namespace_source = (
            model_config.model,
            model_config.revision,
            model_config.code_revision,
            model_config.tokenizer_revision,
            getattr(hf_config, "_commit_hash", None),
            tuple(getattr(hf_config, "architectures", None) or ()),
            mm_hash,
        )
    vision_config = getattr(hf_config, "vision_config", None)
    deepstack_indexes = tuple(
        getattr(vision_config, "deepstack_visual_indexes", None) or ()
    )
    # Equal-width deepstack outputs can still contain different layer features.
    # Keep this semantic salt even when the deployment supplies a namespace.
    return json.dumps(
        ("ucm:ec:v2", namespace_source, deepstack_indexes),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def make_chunk_ids(
    *,
    identifier: str,
    num_embeds: int,
    layout: EncoderCacheLayout,
    hasher: RequestHasher,
) -> tuple[bytes, ...]:
    if not identifier:
        raise EncoderCacheLayoutError("Encoder-cache identifier must not be empty.")
    if num_embeds <= 0:
        raise EncoderCacheLayoutError(
            f"Encoder-cache item {identifier!r} must have at least one row, "
            f"got {num_embeds}."
        )
    num_chunks = (num_embeds + layout.rows_per_chunk - 1) // layout.rows_per_chunk
    return tuple(
        hasher(
            (
                layout.width,
                str(layout.dtype),
                layout.rows_per_chunk,
                identifier,
                num_embeds,
                chunk_index,
            )
        )
        for chunk_index in range(num_chunks)
    )


def create_ucm_ec_store(
    *,
    vllm_config: VllmConfig,
    role: ECConnectorRole,
    layout: EncoderCacheLayout,
    ec_config: dict[str, Any],
    device_id: int,
) -> UcmKVStoreBaseV1:
    name = ec_config["ucm_connector_name"]
    store_config = dict(ec_config["ucm_connector_config"])
    encoder_config = ec_config["encoder_cache_config"]
    if "use_gdr" in store_config:
        logger.warning("UCM EC ignores configured use_gdr; GDR is disabled.")

    if "storage_backends" in store_config:
        store_config["storage_backends"] = [
            path for path in store_config["storage_backends"].split(":")
        ]

    store_config["share_buffer_enable"] = True
    configure_cache_capacity(store_config, vllm_config.parallel_config)
    # Unversioned wheels retain their original shared-buffer default.
    store_config.setdefault("cache_buffer_capacity_gb", 128)
    _check_shm_capacity(int(store_config["cache_buffer_capacity_gb"]))

    tensor_size_list = [layout.chunk_bytes]
    store_shard_size, store_block_size = _get_store_io_sizes(
        layout.chunk_bytes,
        layout.chunk_bytes,
    )
    gc_block_size = _get_store_gc_block_size(
        str(store_config.get("store_pipeline", "")),
        tensor_size_list,
        store_shard_size,
        store_block_size,
    )

    parallel_config = vllm_config.parallel_config
    dp_rank = parallel_config.data_parallel_rank
    # Store sharing/isolation domain: same id ⇒ shared shm + namespace.
    # Scheduler and workers must agree on one id, but instance_id is minted
    # per VllmConfig construction and is not guaranteed to match across
    # processes, so reuse the KV connector's handshake: workers broadcast a
    # uuid and publish it under /dev/shm, the scheduler reads it back.
    # dp{rank} keeps DP domains isolated when one engine serves several.
    unique_id = encoder_config.get("store_unique_id")
    if unique_id is None:
        # Dedicated file slot: KV and EC may run in one engine, and their
        # worker-side broadcasts mint different uuids, so sharing the KV
        # slot would let one scheduler read the other connector's id.
        if role == ECConnectorRole.WORKER:
            unique_id = _worker_generate_unique_id(prefix="ucm_ec")
        else:
            unique_id = _scheduler_read_unique_id(prefix="ucm_ec")
        unique_id = f"{unique_id}.ec.dp{dp_rank}"

    gc_owner = role == ECConnectorRole.SCHEDULER and dp_rank == 0

    store_config.update(
        {
            "unique_id": unique_id,
            "device_id": device_id,
            # TP size so ranks sharing one buffer get staggered traversal
            # order; each rank still loads the full EC (single-writer stays).
            "local_rank_size": parallel_config.tensor_parallel_size,
            "use_gdr": False,
            "posix_gc_enable": gc_owner,
        }
    )
    if uses_partitioned_cache():
        local_count = cache_rank_count(parallel_config)
        store_config["local_rank_size"] = local_count
    if role == ECConnectorRole.WORKER:
        if uses_partitioned_cache():
            configure_cache_numa(
                store_config,
                create_device(),
                device_id,
                parallel_config,
                device_id % local_count,
            )
        store_config.update(
            {
                "tensor_size_list": tensor_size_list,
                "shard_size": store_shard_size,
                "block_size": store_block_size,
            }
        )
    elif gc_owner:
        store_config["block_size"] = gc_block_size

    logger.info(
        "Creating UCM EC store %s: unique_id=%s, device_id=%s, "
        "chunk_bytes=%s, store_block_size=%s, gc_block_size=%s",
        name,
        unique_id,
        device_id,
        layout.chunk_bytes,
        store_block_size,
        gc_block_size,
    )
    return UcmConnectorFactoryV1.create_connector(name, store_config)


class UCMECConnector(ECConnectorBase):
    """Fixed-chunk EC transfers completed within each model execution step."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: ECConnectorRole,
    ) -> None:
        super().__init__(vllm_config, role)
        mm_config = vllm_config.model_config.multimodal_config
        # Pruning models (e.g. Qwen Efficient Video Sampling) append
        # mrope-position channels to a variable-count encoder output and
        # require a post-encoder recompute step; EC caches a fixed [N, D]
        # tensor and can't carry that channel/step, so reject up front.
        # TODO: support pruning if the position channels can be split and
        #       recomputed on the cached tensor.
        if mm_config is not None and mm_config.is_multimodal_pruning_enabled():
            raise EncoderCacheLayoutError("UCM EC does not support multimodal pruning.")

        ec_config = Config.load_ec_config(vllm_config.ec_transfer_config)
        encoder_config = ec_config["encoder_cache_config"]
        self.layout = resolve_encoder_cache_layout(vllm_config, encoder_config)
        self.enable_epd_disaggregate = encoder_config.get(
            "enable_epd_disaggregate", False
        )
        self.epd_timeout_ms = encoder_config.get("epd_timeout_ms", 30000)
        self._metadata_resolver = None
        if self.enable_epd_disaggregate and self.is_producer:
            from vllm.distributed.ec_transfer.ec_connector.utils import (
                PlaceholderMetadataResolver,
            )

            self._metadata_resolver = PlaceholderMetadataResolver(
                vllm_config.model_config
            )
        # These device-facing stages enqueue prerequisite waits on their copy
        # stream or transfer worker. Dram and Delegator wait inside Dump;
        # other stages may ignore the event entirely. Reject before opening
        # the Store rather than introducing a blocking save fallback.
        pipeline = ec_config["ucm_connector_config"].get("store_pipeline", "")
        if self.is_producer and (
            ec_config["ucm_connector_name"] != "UcmPipelineStore"
            or pipeline.split("|")[0] not in {"Cache", "Mooncake", "YuanRong"}
        ):
            raise ValueError(
                "UCM EC asynchronous saves require UcmPipelineStore with a "
                "Cache, Mooncake, or YuanRong first stage that queues "
                "prerequisite event waits."
            )
        self.enable_event_sync = encoder_config.get("enable_event_sync", True)
        self._block_hasher = RequestHasher(
            meta=build_ec_hash_meta(vllm_config, encoder_config)
        )

        self.device_id = (
            -1 if role == ECConnectorRole.SCHEDULER else get_current_device_id()
        )
        self.device = (
            None
            if role == ECConnectorRole.SCHEDULER
            else get_ucm_worker_torch_device(self.device_id)
        )
        self.store: UcmKVStoreBaseV1 | None = create_ucm_ec_store(
            vllm_config=vllm_config,
            role=role,
            layout=self.layout,
            ec_config=ec_config,
            device_id=self.device_id,
        )

        self.req_to_state: dict[str, RequestState] = {}
        self.identifier_to_blocks: dict[str, IdentifierState] = {}
        self.step_verified_hits: set[str] = set()
        self.pending_loads: dict[str, ECLoadSpec] = {}
        # Finalized load submissions, not worker completion or a pinned HBM cache.
        self._local_encoder_ids: set[str] = set()
        self._pending_saves: list[PendingECSave] = []
        self._step_saved_ids: set[str] = set()
        self._save_device = create_device() if role == ECConnectorRole.WORKER else None

        self.is_save_rank = False
        if role == ECConnectorRole.WORKER:
            self.is_save_rank = (
                get_pp_group().is_first_rank
                and get_tensor_model_parallel_rank() == 0
                and get_pcp_group().rank_in_group == 0
            )

        logger.info(
            "Initialized UCM EC connector: role=%s, producer=%s, consumer=%s, "
            "width=%s, dtype=%s, rows_per_chunk=%s, chunk_bytes=%s, "
            "hash_namespace=%s, save_rank=%s",
            role.name,
            self.is_producer,
            self.is_consumer,
            self.layout.width,
            self.layout.dtype,
            self.layout.rows_per_chunk,
            self.layout.chunk_bytes,
            self._block_hasher.seed.hex(),
            self.is_save_rank,
        )

    def ensure_cache_available(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> bool:
        self._register_encoder_inputs(request)
        if not (self.enable_epd_disaggregate and self.is_consumer):
            return True
        if not request.mm_features:
            return True

        state = self.req_to_state[request.request_id]
        pending_ids: dict[str, None] = {}
        local_occurrences = 0
        for feature in request.mm_features:
            if (
                feature.mm_position.offset + feature.mm_position.length
                <= num_computed_tokens
            ):
                continue
            identifier = feature.identifier
            if identifier in self._local_encoder_ids:
                local_occurrences += 1
            elif identifier not in state.store_visible_identifiers:
                pending_ids[identifier] = None
        if local_occurrences:
            logger.debug(
                "EC local cache reuse: request=%s, occurrences=%d",
                request.request_id,
                local_occurrences,
            )
        # KV coverage may remove a previous miss, even after its wait expired.
        # Only actual full lookup hits enter the positive snapshot set, so a
        # rolled-back prefix checks previously skipped items again.
        if not pending_ids:
            state.store_wait_started_at = None
            state.next_lookup_at = 0.0
            return True
        now = time.monotonic()
        if (
            state.store_wait_started_at is not None
            and now - state.store_wait_started_at >= self.epd_timeout_ms / 1000
        ):
            raise TimeoutError(
                f"Timed out waiting for encoder cache visibility for "
                f"request {request.request_id!r}."
            )
        if now < state.next_lookup_at:
            return False
        if self.store is None:
            raise RuntimeError("UCM EC store is closed.")

        # A query snapshot only: actual reads still happen after native allocation.
        # Returning False lets the native scheduler try other requests.
        for identifier in pending_ids:
            item = state.identifier_states[identifier]
            last = self.store.lookup_on_prefix(list(item.chunk_ids))
            if last != len(item.chunk_ids) - 1:
                if state.store_wait_started_at is None:
                    state.store_wait_started_at = now
                state.next_lookup_at = now + _EPD_LOOKUP_INTERVAL_S
                return False
            state.store_visible_identifiers.add(identifier)
        state.store_wait_started_at = None
        state.next_lookup_at = 0.0
        return True

    def _register_encoder_inputs(self, request: Request) -> bool:
        req_id = request.request_id
        request_state = self.req_to_state.get(req_id)
        if request_state is None:
            request_state = RequestState()
        num_features = len(request.mm_features)
        if num_features < request_state.processed_feature_count:
            raise EncoderCacheLayoutError(
                f"Encoder features cannot be removed from request {req_id!r}."
            )
        # Streaming updates append features; previously validated items are stable.
        if num_features == request_state.processed_feature_count:
            return True

        registered_states = request_state.identifier_states
        new_states: dict[str, IdentifierState] = {}
        for index in range(request_state.processed_feature_count, num_features):
            feature = request.mm_features[index]
            identifier = feature.identifier
            # Rows in the encoder cache tensor stored under `identifier`
            # (i.e. its first dimension).
            num_embeds = int(request.get_num_encoder_embeds(index))

            # Both previous batches and this batch refer to the same state type.
            state = registered_states.get(identifier) or new_states.get(identifier)
            seen_in_request = state is not None
            if state is None:
                state = self.identifier_to_blocks.get(identifier)

            if state is None:
                state = IdentifierState(
                    num_embeds=num_embeds,
                    chunk_ids=make_chunk_ids(
                        identifier=identifier,
                        num_embeds=num_embeds,
                        layout=self.layout,
                        hasher=self._block_hasher,
                    ),
                )
            elif state.num_embeds != num_embeds:
                scope = (
                    f"within request {req_id!r}"
                    if seen_in_request
                    else "across active requests"
                )
                raise EncoderCacheLayoutError(
                    f"Identifier {identifier!r} has inconsistent row counts "
                    f"{scope}: {state.num_embeds} and {num_embeds}."
                )

            if not seen_in_request:
                new_states[identifier] = state

        # Commit only after the entire appended batch has passed validation.
        for identifier, candidate in new_states.items():
            state = self.identifier_to_blocks.setdefault(identifier, candidate)
            state.request_ref_count += 1
            registered_states[identifier] = state
        request_state.processed_feature_count = num_features
        self.req_to_state[req_id] = request_state
        return True

    def has_cache_item(self, identifier: str) -> bool:
        if not self.is_consumer:
            return False
        state = self.identifier_to_blocks.get(identifier)
        if state is None:
            return False
        if self.store is None:
            raise RuntimeError("UCM EC store is closed.")

        # Ordinary mode requires a full Store hit; EPD is external-only.
        if self.enable_epd_disaggregate:
            # EPD is external-only. Never fall back to encoding metadata-only input.
            hit = True
        else:
            last = self.store.lookup_on_prefix(list(state.chunk_ids))
            hit = last == len(state.chunk_ids) - 1
        if hit:
            self.step_verified_hits.add(identifier)
        return hit

    def update_state_after_alloc(self, request: Request, index: int) -> None:
        identifier = request.mm_features[index].identifier
        if not self.is_consumer or identifier not in self.step_verified_hits:
            return
        # Accumulate the hit here (O(1) per hit) for the request-finished
        # summary instead of scanning every scheduled request. This must
        # precede the pending-load dedup: a shared identifier counts for every
        # request that references it, not just the one that queued the load.
        request_state = self.req_to_state.get(request.request_id)
        if request_state is not None:
            request_state.hit_identifiers.add(identifier)
        if identifier in self.pending_loads:
            return
        state = self.identifier_to_blocks.get(identifier)
        if state is None:
            raise UCMEncoderCacheError(
                identifier,
                "Missing scheduler state for verified encoder cache item",
            )
        self.pending_loads[identifier] = ECLoadSpec(
            num_embeds=state.num_embeds,
            chunk_ids=state.chunk_ids,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> UCMECConnectorMetadata:
        # A surviving request may share a pending item through the local cache
        # manager without an allocation callback of its own.
        needed_identifiers: set[str] = set()
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            if num_tokens > 0 and (state := self.req_to_state.get(req_id)) is not None:
                needed_identifiers.update(state.identifier_states)
        # Do not recreate entries after the worker applies final evictions.
        # vLLM excludes items reallocated in this step from this list.
        freed_identifiers = set(scheduler_output.free_encoder_mm_hashes)
        metadata = UCMECConnectorMetadata(
            loads={
                identifier: spec
                for identifier, spec in self.pending_loads.items()
                if identifier in needed_identifiers
                and identifier not in freed_identifiers
            }
        )
        if self.enable_epd_disaggregate:
            if self.is_consumer:
                # Request-reference release is not physical eviction. Only commit
                # final loads; cancelled/filtered allocations never enter this set.
                self._local_encoder_ids.difference_update(freed_identifiers)
                self._local_encoder_ids.update(metadata.loads)
            if self.is_producer:
                metadata.save_identifiers = tuple(
                    sorted(needed_identifiers - freed_identifiers)
                )
        self.pending_loads = {}
        self.step_verified_hits.clear()
        return metadata

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        transfer_params = None
        if self.enable_epd_disaggregate and self.is_producer:
            from vllm.distributed.ec_transfer.ec_connector.utils import (
                collect_ec_item_metadata,
            )

            transfer_params = collect_ec_item_metadata(
                request.mm_features, self._metadata_resolver
            )
            # Older 0.30 helpers return only metadata; preserve native values
            # while supplying occurrence information for the current proxy.
            for identifier, item in transfer_params.items():
                indices = [
                    index
                    for index, feature in enumerate(request.mm_features)
                    if feature.identifier == identifier
                ]
                item.setdefault("modality", request.mm_features[indices[0]].modality)
                item.setdefault("item_indices", indices)
            transfer_params = transfer_params or None

        request_state = self.req_to_state.pop(request.request_id, None)
        if request_state is None:
            return False, transfer_params
        if self.is_consumer:
            message = (
                "EC request external load selections: req=%s, mm_items=%d, "
                "selected_total=%d"
                if self.enable_epd_disaggregate
                else "EC request cache hits: req=%s, mm_items=%d, hits_total=%d"
            )
            logger.info(
                message,
                request.request_id,
                len(request_state.identifier_states),
                len(request_state.hit_identifiers),
            )

        for identifier in request_state.identifier_states:
            state = self.identifier_to_blocks.get(identifier)
            if state is None:
                continue
            state.request_ref_count -= 1
            if state.request_ref_count <= 0:
                del self.identifier_to_blocks[identifier]
        return False, transfer_params

    def _get_load_event_handle(self) -> int:
        if self._save_device is None:
            raise RuntimeError("Worker transfer device is not initialized.")
        if not self.enable_event_sync:
            self._save_device.synchronize()
            return 0
        event_handle = self._save_device.get_event_handle()
        if event_handle == 0:
            logger.warning("Failed to record EC load event; synchronizing before load.")
            self._save_device.synchronize()
        return event_handle

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        del kwargs
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMECConnectorMetadata):
            raise TypeError(
                "UCMECConnector expected UCMECConnectorMetadata, got "
                f"{type(metadata).__name__}."
            )
        if self.device is None:
            raise RuntimeError("Worker accelerator device is not initialized.")
        if self.store is None:
            raise RuntimeError("UCM EC store is closed.")

        rows = self.layout.rows_per_chunk
        width = self.layout.width
        pending: list[tuple[str, ECLoadSpec, torch.Tensor, Task, int]] = []
        wait_error: tuple[str, Exception] | None = None
        skipped_items = 0
        submit_start = time.perf_counter()
        try:
            for identifier, spec in metadata.loads.items():
                if identifier in encoder_cache:
                    skipped_items += 1
                    continue
                expected_chunks = (spec.num_embeds + rows - 1) // rows
                if spec.num_embeds <= 0 or len(spec.chunk_ids) != expected_chunks:
                    raise EncoderCacheLayoutError(
                        f"Invalid load metadata for {identifier!r}: "
                        f"num_embeds={spec.num_embeds}, chunks={len(spec.chunk_ids)}, "
                        f"expected_chunks={expected_chunks}."
                    )

                storage = torch.empty(
                    (expected_chunks, rows, width),
                    dtype=self.layout.dtype,
                    device=self.device,
                )
                dst_addrs = self._chunk_addresses(storage, expected_chunks)
                # Fence this destination before handing its raw addresses to UCM.
                event_handle = self._get_load_event_handle()
                try:
                    task = self.store.load_data(
                        list(spec.chunk_ids),
                        [0] * expected_chunks,
                        dst_addrs,
                        prerequisite_handle=event_handle,
                    )
                except Exception as exc:
                    if event_handle:
                        self._save_device.destroy_event_handle(event_handle)
                    raise UCMEncoderCacheError(
                        identifier, "Failed to load encoder cache item"
                    ) from exc
                pending.append((identifier, spec, storage, task, event_handle))
        finally:
            submit_ms = (time.perf_counter() - submit_start) * 1e3
            # Retain destinations and drain every submitted task even if a later
            # allocation/submission or an earlier wait fails.
            wait_start = time.perf_counter()
            loaded_items = failed_items = 0
            total_chunks = 0
            total_bytes = 0
            for identifier, spec, storage, task, event_handle in pending:
                item_start = time.perf_counter()
                try:
                    self.store.wait(task)
                    encoder_cache[identifier] = storage.view(-1, width)[
                        : spec.num_embeds
                    ]
                    loaded_items += 1
                    item_chunks = len(spec.chunk_ids)
                    item_bytes = item_chunks * self.layout.chunk_bytes
                    total_chunks += item_chunks
                    total_bytes += item_bytes
                    logger.debug(
                        "EC load item %s: shape=[%d, %d], chunks=%d, bytes=%d, "
                        "wait_ms=%.2f",
                        identifier,
                        spec.num_embeds,
                        width,
                        item_chunks,
                        item_bytes,
                        (time.perf_counter() - item_start) * 1e3,
                    )
                except Exception as exc:
                    encoder_cache.pop(identifier, None)
                    failed_items += 1
                    logger.exception("Failed to load EC item %s", identifier)
                    if wait_error is None:
                        wait_error = (identifier, exc)
                finally:
                    if event_handle:
                        self._save_device.destroy_event_handle(event_handle)
            if pending:
                logger.debug(
                    "EC loads: submitted=%d, loaded=%d, failed=%d, skipped=%d, "
                    "chunks=%d, bytes=%d, submit_ms=%.2f, wait_ms=%.2f",
                    len(pending),
                    loaded_items,
                    failed_items,
                    skipped_items,
                    total_chunks,
                    total_bytes,
                    submit_ms,
                    (time.perf_counter() - wait_start) * 1e3,
                )

        if wait_error is not None:
            identifier, cause = wait_error
            raise UCMEncoderCacheError(
                identifier, "Failed to load encoder cache item"
            ) from cause

    def _chunk_addresses(self, tensor: torch.Tensor, num_chunks: int) -> np.ndarray:
        # EC tensors are contiguous; no per-chunk Tensor views are needed.
        # Use logical chunk bytes, not the Store's aligned physical shard size.
        offsets = np.arange(num_chunks, dtype=np.uint64)
        offsets *= np.uint64(self.layout.chunk_bytes)
        offsets += np.uint64(tensor.data_ptr())
        return offsets.reshape(-1, 1)

    def _validate_encoder_tensor(
        self,
        identifier: str,
        tensor: torch.Tensor,
    ) -> None:
        if tensor.ndim != 2:
            raise EncoderCacheLayoutError(
                f"Encoder cache {identifier!r} must be 2-D, got shape "
                f"{tuple(tensor.shape)}."
            )
        if int(tensor.shape[0]) <= 0:
            raise EncoderCacheLayoutError(
                f"Encoder cache {identifier!r} must contain at least one row."
            )
        if int(tensor.shape[1]) != self.layout.width:
            raise EncoderCacheLayoutError(
                f"Encoder cache {identifier!r} has width {tensor.shape[1]}, "
                f"expected {self.layout.width}."
            )
        if tensor.dtype != self.layout.dtype:
            raise EncoderCacheLayoutError(
                f"Encoder cache {identifier!r} has dtype {tensor.dtype}, "
                f"expected {self.layout.dtype}."
            )
        if not tensor.is_contiguous():
            raise EncoderCacheLayoutError(
                f"Encoder cache {identifier!r} must be contiguous."
            )

    def start_save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if not (self.enable_epd_disaggregate and self.is_producer):
            return
        self._step_saved_ids.clear()
        if not self.is_save_rank or encoder_cache is None:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMECConnectorMetadata):
            raise TypeError(
                "UCMECConnector expected UCMECConnectorMetadata, got "
                f"{type(metadata).__name__}."
            )
        for identifier in metadata.save_identifiers:
            if identifier in encoder_cache:
                # Cached outputs do not trigger the runner's new-output save hook.
                self.save_caches(encoder_cache, identifier)
        logger.debug(
            "EC cached saves: candidates=%d, submitted=%d",
            len(metadata.save_identifiers),
            len(self._step_saved_ids),
        )

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if not self.is_producer or not self.is_save_rank:
            return
        if self.enable_epd_disaggregate and mm_hash in self._step_saved_ids:
            return
        if self.store is None:
            raise RuntimeError("UCM EC store is closed.")

        tensor = encoder_cache[mm_hash]
        save_start = time.perf_counter()
        self._validate_encoder_tensor(mm_hash, tensor)
        num_embeds = int(tensor.shape[0])
        rows = self.layout.rows_per_chunk
        width = self.layout.width
        chunk_ids = make_chunk_ids(
            identifier=mm_hash,
            num_embeds=num_embeds,
            layout=self.layout,
            hasher=self._block_hasher,
        )

        num_full_chunks, tail_rows = divmod(num_embeds, rows)
        event_handle = 0
        try:
            src_addrs = self._chunk_addresses(tensor, len(chunk_ids))
            tail: torch.Tensor | None = None
            if tail_rows:
                tail = torch.zeros(
                    (rows, width),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                tail[:tail_rows].copy_(tensor[num_full_chunks * rows :])
                src_addrs[-1, 0] = np.uint64(tail.data_ptr())

            # Record after both encoder compute and tail padding. The Store's
            # copy stream must wait for these writes before reading raw pointers.
            event_handle = self._save_device.get_event_handle()
            if event_handle == 0:
                raise RuntimeError("Failed to record EC save prerequisite event.")
            task = self.store.dump_data(
                list(chunk_ids),
                [0] * len(chunk_ids),
                src_addrs,
                prerequisite_handle=event_handle,
            )
            self._pending_saves.append(
                PendingECSave(mm_hash, task, tensor, tail, event_handle)
            )
            if self.enable_epd_disaggregate:
                self._step_saved_ids.add(mm_hash)
            logger.debug(
                "EC dump item %s: shape=[%d, %d], chunks=%d, tail_rows=%d, "
                "bytes=%d, submit_ms=%.2f",
                mm_hash,
                num_embeds,
                width,
                len(chunk_ids),
                tail_rows,
                len(chunk_ids) * self.layout.chunk_bytes,
                (time.perf_counter() - save_start) * 1e3,
            )
        except Exception:
            if event_handle:
                self._save_device.destroy_event_handle(event_handle)
            logger.exception("Failed to dump EC item %s", mm_hash)

    def _wait_pending_saves(self) -> None:
        if not self._pending_saves:
            return
        if self.store is None:
            raise RuntimeError("UCM EC store is closed with pending saves.")
        pending, self._pending_saves = self._pending_saves, []
        for save in pending:
            try:
                self.store.wait(save.task)
            except Exception:
                # Saving is best effort; one failure must not skip later waits.
                logger.exception("Failed to dump EC item %s", save.identifier)
            finally:
                if save.event_handle:
                    self._save_device.destroy_event_handle(save.event_handle)

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        del finished_req_ids
        try:
            self._wait_pending_saves()
        finally:
            self._step_saved_ids.clear()
        return None, None

    def shutdown(self) -> None:
        self._wait_pending_saves()
        self.pending_loads.clear()
        self.step_verified_hits.clear()
        self._local_encoder_ids.clear()
        self._step_saved_ids.clear()
        self.req_to_state.clear()
        self.identifier_to_blocks.clear()
        self.clear_connector_metadata()

        store = self.store
        self.store = None
        if store is not None:
            close = getattr(store, "close", None)
            if callable(close):
                close()


__all__ = [
    "ECLoadSpec",
    "EncoderCacheLayout",
    "EncoderCacheLayoutError",
    "IdentifierState",
    "UCMECConnector",
    "UCMECConnectorMetadata",
    "UCMEncoderCacheError",
    "create_ucm_ec_store",
    "make_chunk_ids",
    "build_ec_hash_meta",
    "resolve_encoder_cache_layout",
]
