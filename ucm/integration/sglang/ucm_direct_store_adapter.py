"""UCM Store V1 adapter used by SGLang's L1 external-cache linker."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UcmPoolMetadata:
    name: str
    pool_name: str
    source_pool: str
    store_group: str
    page_size: int
    layer_schemas: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    device_buffers: tuple[tuple[int, int], ...]
    capacity_pages: int = 0


@dataclass(frozen=True)
class UcmLayerTransfer:
    store_group: str
    source_pool: str
    pools: tuple[str, ...]
    layers: tuple[int, ...]
    shard_size: int
    tensor_size_list: tuple[int, ...]
    page_keys: tuple[str, ...]
    # I/O uses a contiguous (layer, page, fragment) uint64 array.
    # Lookup does not materialize pointers.
    addresses: np.ndarray | None
    fragments: tuple[tuple[int, int], ...]
    hit_policy: str = "all_pages"


@dataclass(frozen=True)
class _StoreTask:
    store: UcmKVStoreBaseV1
    task: Task


@dataclass(frozen=True)
class _PreparedPipelineTransfer:
    """SGLang-local fast path for the stock PipelineStore.

    IDs are encoded once per transfer. Native Load/Dump copy the submission
    descriptors before returning; address arrays and events are retained by
    the linker until tasks are reaped. Other Stores use their public methods.
    """

    store: Any
    ids: np.ndarray
    page_count: int
    task_type: type[Task]

    def load_data(self, shard_index: int, addresses: np.ndarray) -> Task:
        indexes = np.full(self.page_count, shard_index, dtype=np.uint64)
        return self.task_type(self.store.Load(self.ids, indexes, addresses))

    def dump_data(
        self, shard_index: int, addresses: np.ndarray, prerequisite_handle: int = 0
    ) -> Task:
        indexes = np.full(self.page_count, shard_index, dtype=np.uint64)
        return self.task_type(
            self.store.Dump(self.ids, indexes, addresses, prerequisite_handle)
        )


@dataclass(frozen=True)
class _PreparedStoreTransfer:
    transfer: UcmLayerTransfer
    store: UcmKVStoreBaseV1
    block_ids: list[bytes]
    shards_by_layer: dict[int, tuple[int, ...]]
    prepared_io: _PreparedPipelineTransfer | None = None


@dataclass(frozen=True)
class _TransferLayout:
    signature: tuple[object, ...]
    store: UcmKVStoreBaseV1
    tensor_count: int
    shards_by_layer: dict[int, tuple[int, ...]]


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def _pool_names_key(members: Iterable[UcmPoolMetadata]) -> tuple[str, ...]:
    pool_order = {"kv": 0, "indexer": 1, "swa": 2, "mamba": 3}
    return tuple(
        sorted(
            {_safe_name(pool.pool_name).strip("._-").lower() for pool in members},
            key=lambda name: (pool_order.get(name, 4), name),
        )
    )


class UcmDirectStoreAdapter:
    """Map SGLang device-address rows to UCM Store V1 operations.

    One Store is created during device-pool registration for each linker-selected
    ``store_group``. A Store block holds every fixed fragment for one page across
    the group's full logical-layer range.
    """

    def __init__(
        self,
        *,
        connector_name: str,
        module_path: str | None,
        store_config: dict[str, Any],
        identity: dict[str, Any],
    ) -> None:
        self._connector_name = connector_name
        self._module_path = module_path
        self._identity = dict(identity)
        self._block_identity_prefix = self._identity_suffix()
        self._store_config = copy.deepcopy(store_config)
        # Cache's D2H stream waits on this CUDA event before reading KV data.
        # Other connectors/pipelines have not established that contract.
        self.supports_dump_prerequisite = (
            connector_name == "UcmPipelineStore"
            and self._store_config.get("store_pipeline") == "Cache|Posix"
        )
        prefetch_layers_config = self._store_config.pop(
            "sglang_load_prefetch_layers", 2
        )
        try:
            self.load_prefetch_layers = min(8, max(0, int(prefetch_layers_config)))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid sglang_load_prefetch_layers=%r; fallback to 2.",
                prefetch_layers_config,
            )
            self.load_prefetch_layers = 2
        self._stores: dict[str, UcmKVStoreBaseV1] = {}
        self._lookup_store_signatures: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._schemas: dict[str, tuple[tuple[int, int], ...]] = {}
        self._transfer_layouts: dict[str, _TransferLayout] = {}
        self._pool_metadata: dict[str, UcmPoolMetadata] = {}
        self._backend_paths_by_group: dict[str, list[str]] = {}
        self._gc_capacity_by_group: dict[str, int] = {}
        self._store_lock = threading.Lock()

    @classmethod
    def from_config(
        cls, config: dict[str, Any], identity: dict[str, Any]
    ) -> "UcmDirectStoreAdapter":
        config_file = config.get("config_file")
        if not config_file:
            raise ValueError(
                "SGLang UCM linker requires linker_extra_config.config_file "
                "or UNIFIEDCACHE_CONFIG_FILE."
            )
        path = Path(config_file)
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        if not isinstance(document, dict):
            raise ValueError("UCM configuration YAML root must be a mapping.")
        connectors = document.get("ucm_connectors")
        if not isinstance(connectors, list) or len(connectors) != 1:
            raise ValueError(
                "SGLang UCM direct linker requires exactly one ucm_connectors entry."
            )
        connector = connectors[0]
        if not isinstance(connector, dict):
            raise ValueError("UCM connector entry must be a mapping.")
        name = connector.get("ucm_connector_name")
        store_config = connector.get("ucm_connector_config")
        if not isinstance(name, str) or not name:
            raise ValueError("UCM connector entry is missing ucm_connector_name.")
        if not isinstance(store_config, dict):
            raise ValueError("UCM connector entry is missing ucm_connector_config.")
        return cls(
            connector_name=name,
            module_path=connector.get("ucm_connector_module_path"),
            store_config=store_config,
            identity=identity,
        )

    def _estimate_gc_capacities(
        self, by_group: dict[str, list[UcmPoolMetadata]]
    ) -> dict[str, int]:
        """Split the configured total GiB by logical pages and block bytes."""
        configured = self._store_config.get("posix_capacity_gb", 0)
        total = int(configured or 0)
        if total < 0:
            raise ValueError("posix_capacity_gb must be non-negative")
        if not total or not by_group:
            return {}

        weights: dict[str, int] = {}
        for group, members in sorted(by_group.items()):
            pages = members[0].capacity_pages
            block_bytes = sum(
                int(size)
                for member in members
                for _, schema in member.layer_schemas
                for size, _ in schema
            )
            weights[group] = pages * block_bytes
        weight_total = sum(weights.values())
        if weight_total <= 0:
            raise ValueError(
                "Cannot split posix_capacity_gb: Store groups have no positive "
                "estimated pool capacity. Check pool capacity_pages and layer_schemas."
            )
        capacities = {
            group: total * weight // weight_total for group, weight in weights.items()
        }
        leftover = total - sum(capacities.values())
        order = sorted(weights, key=lambda g: (-(total * weights[g] % weight_total), g))
        for group in order[:leftover]:
            capacities[group] += 1
        for group in weights:
            logger.info(
                "SGLang UCM GC capacity group=%s estimated_bytes=%d " "capacity_gb=%d",
                group,
                weights[group],
                capacities[group],
            )
        return capacities

    def register_device_pools(self, pools: list[UcmPoolMetadata]) -> None:
        for pool in pools:
            previous = self._pool_metadata.get(pool.name)
            if previous is not None and previous != pool:
                raise ValueError(
                    f"Conflicting UCM metadata for device pool {pool.name}."
                )
            self._pool_metadata[pool.name] = pool
        by_group: dict[str, list[UcmPoolMetadata]] = {}
        for pool in self._pool_metadata.values():
            by_group.setdefault(pool.store_group, []).append(pool)
        self._gc_capacity_by_group = self._estimate_gc_capacities(by_group)
        self._backend_paths_by_group = self._prepare_backend_paths(by_group)
        for group, members in by_group.items():
            try:
                ordered_members = sorted(members, key=lambda item: item.name)
                schemas_by_member = {
                    member.name: dict(member.layer_schemas) for member in members
                }
                layer_sets = {tuple(schemas) for schemas in schemas_by_member.values()}
                if len(layer_sets) != 1:
                    raise ValueError("inconsistent logical layers")
                layers = next(iter(layer_sets), ())
                if not layers:
                    raise ValueError("no logical layers")
                page_sizes = {member.page_size for member in members}
                if len(page_sizes) != 1:
                    raise ValueError(f"inconsistent page sizes: {sorted(page_sizes)}")
                schema = tuple(
                    (int(size), 0)
                    for layer in layers
                    for member in ordered_members
                    for size, _ in schemas_by_member[member.name][layer]
                )
                device_buffers = tuple(
                    dict.fromkeys(
                        buffer for member in members for buffer in member.device_buffers
                    )
                )
                if not device_buffers:
                    raise ValueError("no device buffers")
                logger.info(
                    "UCM register pool group=%s source=%s pools=%s page_size=%d "
                    "tensors=%d page_block_bytes=%d buffers=%d "
                    "buffer_bytes=%d.",
                    group,
                    members[0].source_pool,
                    [member.pool_name for member in members],
                    next(iter(page_sizes)),
                    len(schema),
                    sum(size for size, _ in schema),
                    len(device_buffers),
                    sum(size for _, size in device_buffers),
                )
                layer_tensor_size_list = tuple(
                    size
                    for member in ordered_members
                    for size, _ in schemas_by_member[member.name][layers[0]]
                )
                layer_shard_size = sum(layer_tensor_size_list)
                self._ensure_store_schema(
                    group,
                    schema,
                    device_buffers,
                    shard_size=layer_shard_size,
                    tensor_size_list=layer_tensor_size_list,
                )
            except Exception:
                logger.exception(
                    "Failed to register UCM pool group=%s source=%s pools=%s.",
                    group,
                    members[0].source_pool,
                    [member.pool_name for member in members],
                )
                raise

    def _identity_suffix(self) -> str:
        model = str(self._identity.get("model") or "model")
        model_slug = _safe_name(model).strip("._-")[:48] or "model"
        model_fingerprint = hashlib.sha256(model.encode("utf-8")).hexdigest()[:16]
        parts = [
            f"{model_slug}-{model_fingerprint}",
            f"pp{int(self._identity.get('pp_rank', 0))}",
            f"cp{int(self._identity.get('attn_cp_rank', 0))}",
        ]
        if not self._identity.get("rank_replicated", False):
            parts.append(f"tp{int(self._identity.get('tp_rank', 0))}")
        return "_".join(parts)

    def _store_unique_id(
        self,
        store_group: str,
        *,
        schema: tuple[tuple[int, int], ...],
        shard_size: int,
        tensor_size_list: tuple[int, ...],
        block_size: int,
    ) -> str:
        """Return a bounded, deterministic ID for Cache/Fake shm objects.

        The C++ backends prepend ``uc_shm_cache_`` or ``uc_shm_fake_`` to this
        value.  Hash the complete sharing identity and layout instead of
        passing the long layer-expanded ``store_group`` directly, while
        preserving the full group in persistent block IDs.
        """
        identity = {
            "model": str(self._identity.get("model") or "model"),
            "pp_rank": int(self._identity.get("pp_rank", 0)),
            "pp_size": int(self._identity.get("pp_size", 1)),
            "cp_rank": int(self._identity.get("attn_cp_rank", 0)),
            "cp_size": int(self._identity.get("attn_cp_size", 1)),
            "tp_size": int(self._identity.get("tp_size", 1)),
            "rank_replicated": bool(self._identity.get("rank_replicated", False)),
        }
        # Replicated TP ranks intentionally share one shm object.  Non-
        # replicated ranks require separate namespaces, matching _identity_suffix.
        if not identity["rank_replicated"]:
            identity["tp_rank"] = int(self._identity.get("tp_rank", 0))
        layout = {
            "store_group": str(store_group),
            "schema": [[int(size), int(offset)] for size, offset in schema],
            "shard_size": int(shard_size),
            "tensor_size_list": [int(size) for size in tensor_size_list],
            "block_size": int(block_size),
        }
        canonical = json.dumps(
            {"identity": identity, "layout": layout},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _schema(transfer: UcmLayerTransfer) -> tuple[tuple[int, int], ...]:
        raw_schema = tuple(
            (int(size), int(offset)) for size, offset in transfer.fragments
        )
        # Tensor sizes are validated centrally when creating the Store.
        # Offsets are discarded below, so check them before normalization.
        if any(offset < 0 for _, offset in raw_schema):
            raise ValueError(
                f"UCM transfer {transfer.store_group} has an invalid fragment schema."
            )
        # Range offsets select each source layer in SGLang's packed pool. The
        # UCM block represents one full page, so its fixed tensor layout is the
        # ordered fragments for every layer in the page.
        return tuple((size, 0) for size, _ in raw_schema)

    def _prepare_backend_paths(
        self, by_group: dict[str, list[UcmPoolMetadata]]
    ) -> dict[str, list[str]]:
        configured = self._store_config.get("storage_backends")
        if isinstance(configured, str):
            roots = [item for item in configured.split(":") if item]
        elif isinstance(configured, (list, tuple)):
            roots = [str(item) for item in configured]
        else:
            raise ValueError("UCM connector config requires storage_backends.")
        if not roots:
            raise ValueError("UCM connector config has no storage_backends.")
        groups_by_pools: dict[tuple[str, ...], list[str]] = {}
        for group, members in by_group.items():
            pool_names = _pool_names_key(members)
            if any(not name for name in pool_names):
                raise ValueError(
                    f"UCM store group {group} has an empty physical pool name."
                )
            groups_by_pools.setdefault(pool_names, []).append(group)

        backends: dict[str, list[str]] = {}
        for pool_names, groups in groups_by_pools.items():
            base_name = "_".join(pool_names)
            # A single Store for this physical pool set keeps the simple name.
            if len(groups) == 1:
                backends[groups[0]] = [str(Path(root) / base_name) for root in roots]
                continue

            groups_by_layers: dict[tuple[int, ...], list[str]] = {}
            for group in groups:
                layers = tuple(
                    sorted(
                        {
                            layer
                            for pool in by_group[group]
                            for layer, _ in pool.layer_schemas
                        }
                    )
                )
                groups_by_layers.setdefault(layers, []).append(group)
            for layers, matching_groups in groups_by_layers.items():
                directory_name = base_name
                if layers:
                    contiguous = layers == tuple(range(layers[0], layers[-1] + 1))
                    label = (
                        f"{layers[0]}-{layers[-1]}"
                        if contiguous
                        else "_".join(map(str, layers))
                    )
                    directory_name = f"{directory_name}_layers_{label}"
                for group in matching_groups:
                    name = directory_name
                    # Same pool set and layer range require a layout fingerprint.
                    if len(matching_groups) > 1:
                        fingerprint = hashlib.sha256(group.encode("utf-8")).hexdigest()[
                            :10
                        ]
                        name = f"{name}_layout_{fingerprint}"
                    backends[group] = [str(Path(root) / name) for root in roots]
        return backends

    def _ensure_store_schema(
        self,
        store_group: str,
        schema: tuple[tuple[int, int], ...],
        device_buffers: tuple[tuple[int, int], ...] = (),
        *,
        shard_size: int,
        tensor_size_list: tuple[int, ...],
    ) -> UcmKVStoreBaseV1:
        with self._store_lock:
            existing_schema = self._schemas.get(store_group)
            if existing_schema is not None and existing_schema != schema:
                raise ValueError(
                    "One UCM store group must have a fixed page/layer schema: "
                    f"group={store_group}."
                )
            existing = self._stores.get(store_group)
            if existing is not None:
                return existing

            if not schema or any(size <= 0 for size, _ in schema):
                raise ValueError(
                    f"UCM store group {store_group} has no valid fragments."
                )
            self._schemas[store_group] = schema

            tensor_size_list = [int(size) for size in tensor_size_list]
            payload_size = sum(size for size, _ in schema)
            if not tensor_size_list or sum(tensor_size_list) != shard_size:
                raise ValueError(
                    "Store tensor_size_list must describe exactly one shard: "
                    f"group={store_group} tensor_size_list={tensor_size_list} "
                    f"shard_size={shard_size}."
                )
            if shard_size <= 0 or payload_size % shard_size:
                raise ValueError(
                    "Store block_size must be an integer number of shards: "
                    f"group={store_group} shard_size={shard_size} "
                    f"block_size={payload_size}."
                )
            store_config = copy.deepcopy(self._store_config)
            if store_group in self._gc_capacity_by_group:
                store_config["posix_capacity_gb"] = self._gc_capacity_by_group[
                    store_group
                ]
            store_config["storage_backends"] = list(
                self._backend_paths_by_group[store_group]
            )
            for backend in store_config["storage_backends"]:
                Path(backend).mkdir(parents=True, exist_ok=True)
            # ``unique_id`` is used in Cache/Fake POSIX shared-memory names.
            # Keep it bounded and include the actual layout so incompatible
            # Stores cannot accidentally attach to the same shared buffer.
            store_config["unique_id"] = self._store_unique_id(
                store_group,
                schema=schema,
                shard_size=shard_size,
                tensor_size_list=tuple(tensor_size_list),
                block_size=payload_size,
            )
            store_config["tensor_layout"] = "sglang_l1"
            store_config["tensor_size_list"] = tensor_size_list
            # CacheStore treats a scalar tensor_size as higher priority than
            # tensor_size_list. Do not collapse a composite DSA payload such
            # as [indexer_bytes, kv_bytes] into one contiguous tensor here.
            # Pipeline backends that need a scalar set it on their private
            # downstream configuration (for example, Posix uses shard_size).
            store_config["shard_size"] = shard_size
            store_config["block_size"] = payload_size
            store_config["device_id"] = int(self._identity.get("device_id", 0))
            # Nominate DP0 / attention-DP0 / TP0 as the GC contender.
            # Posix's directory lease coordinates contenders across PP/CP
            # groups and other processes sharing the backend.
            gc_enabled = store_config.get("posix_gc_enable", True)
            if not isinstance(gc_enabled, bool):
                raise ValueError("posix_gc_enable must be a boolean")
            gc_owner = (
                int(self._identity.get("dp_rank", 0)) == 0
                and int(self._identity.get("attn_dp_rank", 0)) == 0
                and int(self._identity.get("tp_rank", 0)) == 0
            )
            store_config["posix_gc_enable"] = gc_enabled and gc_owner
            # For now, only the GC owner tracks Posix lookup hotness. Other
            # lookup owners may access different files or storage backends.
            store_config["posix_hotness_enable"] = (
                store_config["posix_gc_enable"]
                and int(store_config.get("posix_capacity_gb", 0) or 0) > 0
            )
            store_config["gpu_kv_buffer_addrs"] = [
                address for address, _ in device_buffers
            ]
            store_config["gpu_kv_buffer_sizes"] = [size for _, size in device_buffers]
            store_config["local_rank_size"] = (
                int(self._identity.get("tp_size", 1))
                if self._identity.get("rank_replicated", False)
                else 1
            )
            logger.info(
                "Creating SGLang UCM Store group=%s fragments=%d "
                "page_shard_bytes=%d page_block_bytes=%d layers=%d buffers=%d.",
                store_group,
                len(tensor_size_list),
                shard_size,
                payload_size,
                payload_size // shard_size,
                len(device_buffers),
            )
            logger.info(
                "SGLang UCM Posix GC group=%s enabled=%s capacity_gb=%s "
                "dp_rank=%d attn_dp_rank=%d tp_rank=%d",
                store_group,
                store_config["posix_gc_enable"],
                store_config.get("posix_capacity_gb", 0),
                int(self._identity.get("dp_rank", 0)),
                int(self._identity.get("attn_dp_rank", 0)),
                int(self._identity.get("tp_rank", 0)),
            )
            store = UcmConnectorFactoryV1.create_connector(
                self._connector_name, store_config, self._module_path
            )
            self._stores[store_group] = store
            self._lookup_store_signatures[store_group] = (
                store_config["unique_id"],
                tuple(store_config["storage_backends"]),
            )
            return store

    def _block_ids(
        self, transfer: UcmLayerTransfer, page_keys: Iterable[str] | None = None
    ) -> list[bytes]:
        keys = transfer.page_keys if page_keys is None else page_keys
        prefix = f"{self._block_identity_prefix}|{transfer.store_group}|"
        return [hashlib.md5(f"{prefix}{key}".encode("utf-8")).digest() for key in keys]

    @staticmethod
    def _is_trailing(transfer: UcmLayerTransfer) -> bool:
        return str(transfer.hit_policy) == "trailing_pages"

    @staticmethod
    def _apply_trailing_hit_policy(
        candidates: list[int], exists: list[bool], transfer: UcmLayerTransfer
    ) -> list[int]:
        prefix = [0]
        for present in exists:
            prefix.append(prefix[-1] + int(bool(present)))
        trailing = max(1, len(transfer.page_keys))
        return [
            end
            for end in candidates
            if prefix[end] - prefix[max(0, end - trailing)]
            == end - max(0, end - trailing)
        ]

    def lookup(
        self,
        transfers: list[UcmLayerTransfer],
        page_keys: list[str],
        *,
        resume_from_pages: int = 0,
    ) -> list[int]:
        if not transfers or not page_keys:
            return []
        candidates = list(range(1, len(page_keys) + 1))
        for transfer in transfers:
            try:
                store = self._transfer_layout(transfer).store
                keys = page_keys[: candidates[-1]]
                # ALL_PAGES rechecks the last jointly hit page, then probes
                # the tail.
                # TRAILING_PAGES can have sparse valid boundaries and must
                # still inspect its complete candidate range.
                if self._is_trailing(transfer):
                    exists = [
                        bool(value)
                        for value in store.lookup(self._block_ids(transfer, keys))
                    ]
                    if len(exists) != len(keys):
                        raise RuntimeError(f"expected={len(keys)} actual={len(exists)}")
                    candidates = self._apply_trailing_hit_policy(
                        candidates, exists, transfer
                    )
                else:
                    # Resume assumes the earlier hit pages remain present.
                    # If Posix GC evicts an earlier page while the last hit
                    # page survives, this check can overestimate the restorable
                    # prefix and a later load can fail at the layer wait.
                    # SGLang 0.5.20/0.5.21 have no request-level load-failure path.
                    # A miss on the checked boundary still triggers a full
                    # prefix lookup below; earlier holes are not detected here.
                    start = (
                        resume_from_pages - 1
                        if 0 < resume_from_pages <= len(keys)
                        else 0
                    )
                    query_keys = keys[start:]
                    # ALL_PAGES needs only the contiguous hit length. Native
                    # stores can stop at the first miss without returning a
                    # bool for every page.
                    found = (
                        store.lookup_on_prefix(self._block_ids(transfer, query_keys))
                        + 1
                    )
                    if not 0 <= found <= len(query_keys):
                        raise RuntimeError(
                            f"invalid prefix hit count={found} pages={len(query_keys)}"
                        )
                    if start and found == 0:
                        # The previous boundary vanished; rebuild from page 0.
                        found = (
                            store.lookup_on_prefix(self._block_ids(transfer, keys)) + 1
                        )
                        if not 0 <= found <= len(keys):
                            raise RuntimeError(
                                f"invalid prefix hit count={found} pages={len(keys)}"
                            )
                        start = 0
                    boundary = start + found
                    candidates = [end for end in candidates if end <= boundary]
                if not candidates:
                    return []
            except Exception:
                logger.exception(
                    "UCM lookup failed: group=%s source=%s layers=%s pools=%s "
                    "pages=%d.",
                    transfer.store_group,
                    transfer.source_pool,
                    transfer.layers,
                    transfer.pools,
                    len(page_keys),
                )
                raise
        return candidates

    def _transfer_layout(self, transfer: UcmLayerTransfer) -> _TransferLayout:
        signature = (
            transfer.layers,
            transfer.fragments,
            transfer.tensor_size_list,
            transfer.shard_size,
        )
        with self._store_lock:
            cached = self._transfer_layouts.get(transfer.store_group)
        if cached is not None:
            if cached.signature != signature:
                raise ValueError(
                    f"UCM group {transfer.store_group} changed its layout."
                )
            return cached

        schema = self._schema(transfer)
        tensor_count = len(transfer.tensor_size_list)
        if not tensor_count or len(schema) != len(transfer.layers) * tensor_count:
            raise ValueError("layer count/address schema mismatch")
        if transfer.shard_size != sum(transfer.tensor_size_list):
            raise ValueError("tensor sizes do not match shard size")
        shard_indices: dict[int, list[int]] = {}
        for index, layer in enumerate(transfer.layers):
            start = index * tensor_count
            sizes = tuple(size for size, _ in schema[start : start + tensor_count])
            if sizes != transfer.tensor_size_list:
                raise ValueError(
                    f"UCM group {transfer.store_group} layer={layer} "
                    "does not match the Store shard schema."
                )
            shard_indices.setdefault(layer, []).append(index)
        layout = _TransferLayout(
            signature=signature,
            store=self._ensure_store_schema(
                transfer.store_group,
                schema,
                shard_size=transfer.shard_size,
                tensor_size_list=transfer.tensor_size_list,
            ),
            tensor_count=tensor_count,
            shards_by_layer={
                layer: tuple(indices) for layer, indices in shard_indices.items()
            },
        )
        with self._store_lock:
            cached = self._transfer_layouts.setdefault(transfer.store_group, layout)
        if cached.signature != signature:
            raise ValueError(f"UCM group {transfer.store_group} changed its layout.")
        return cached

    def _prepare_pipeline_io(
        self, store: UcmKVStoreBaseV1, block_ids: list[bytes]
    ) -> _PreparedPipelineTransfer | None:
        if self._connector_name != "UcmPipelineStore" or self._module_path:
            return None
        # Do not bypass public methods of custom Stores or subclasses.
        from ucm.store.pipeline.connector import (
            UcmPipelineStore,
            UcmPipelineStoreTransTask,
        )

        if type(store) is not UcmPipelineStore:
            return None
        return _PreparedPipelineTransfer(
            store.store_,
            np.frombuffer(b"".join(block_ids), dtype=np.uint8),
            len(block_ids),
            UcmPipelineStoreTransTask,
        )

    def _prepare_transfers(
        self, transfers: list[UcmLayerTransfer], *, is_offload: bool
    ) -> list[_PreparedStoreTransfer]:
        prepared: list[_PreparedStoreTransfer] = []
        for transfer in transfers:
            try:
                layout = self._transfer_layout(transfer)
                addresses = transfer.addresses
                if (
                    not isinstance(addresses, np.ndarray)
                    or addresses.shape
                    != (
                        len(transfer.layers),
                        len(transfer.page_keys),
                        layout.tensor_count,
                    )
                    or addresses.dtype != np.uint64
                    or not addresses.flags.c_contiguous
                ):
                    raise ValueError("address/schema mismatch")
                block_ids = self._block_ids(transfer)
                prepared.append(
                    _PreparedStoreTransfer(
                        transfer=transfer,
                        store=layout.store,
                        block_ids=block_ids,
                        shards_by_layer=layout.shards_by_layer,
                        prepared_io=self._prepare_pipeline_io(layout.store, block_ids),
                    )
                )
            except Exception:
                logger.exception(
                    "UCM %s prepare failed: group=%s source=%s layers=%s pools=%s "
                    "pages=%d.",
                    "offload" if is_offload else "load",
                    transfer.store_group,
                    transfer.source_pool,
                    transfer.layers,
                    transfer.pools,
                    len(transfer.page_keys),
                )
                raise
        return prepared

    def prepare_load(
        self, transfers: list[UcmLayerTransfer]
    ) -> dict[int, list[_PreparedStoreTransfer]]:
        by_layer: dict[int, list[_PreparedStoreTransfer]] = {}
        for item in self._prepare_transfers(transfers, is_offload=False):
            for layer in item.shards_by_layer:
                by_layer.setdefault(layer, []).append(item)
        return by_layer

    def _submit_prepared(
        self,
        prepared: list[_PreparedStoreTransfer],
        *,
        is_offload: bool,
        logical_layer: int | None = None,
        prerequisite_handle: int = 0,
    ) -> list[_StoreTask]:
        tasks: list[_StoreTask] = []
        for item in prepared:
            transfer = item.transfer
            try:
                target = (
                    item.prepared_io if item.prepared_io is not None else item.store
                )
                operation = target.dump_data if is_offload else target.load_data
                shard_indices = (
                    range(len(transfer.layers))
                    if logical_layer is None
                    else item.shards_by_layer.get(logical_layer, ())
                )
                for shard_index in shard_indices:
                    # The Store index is local to this bundle; transfer.layers
                    # contains the corresponding SGLang logical layer.
                    addresses = transfer.addresses[shard_index]
                    args = (
                        (shard_index, addresses)
                        if item.prepared_io is not None
                        else (
                            item.block_ids,
                            [shard_index] * len(item.block_ids),
                            addresses,
                        )
                    )
                    task = (
                        operation(*args, prerequisite_handle=prerequisite_handle)
                        if is_offload and prerequisite_handle
                        else operation(*args)
                    )
                    tasks.append(_StoreTask(item.store, task))
            except Exception:
                # A later shard may fail to submit after earlier shards have
                # started. Reap those handles before their device buffers can
                # be reused by the caller.
                try:
                    self.wait(tasks)
                except Exception:
                    logger.exception(
                        "UCM partially submitted transfer failed while draining"
                    )
                logger.exception(
                    "UCM %s submit failed: group=%s source=%s layers=%s pools=%s "
                    "pages=%d.",
                    "offload" if is_offload else "load",
                    transfer.store_group,
                    transfer.source_pool,
                    transfer.layers,
                    transfer.pools,
                    len(transfer.page_keys),
                )
                raise
        return tasks

    def submit_prepared_load_layer(
        self, prepared: dict[int, list[_PreparedStoreTransfer]], logical_layer: int
    ) -> list[_StoreTask]:
        return self._submit_prepared(
            prepared.get(logical_layer, []),
            is_offload=False,
            logical_layer=logical_layer,
        )

    def submit_offload(
        self, transfers: list[UcmLayerTransfer], prerequisite_handle: int = 0
    ) -> list[_StoreTask]:
        return self._submit_prepared(
            self._prepare_transfers(transfers, is_offload=True),
            is_offload=True,
            prerequisite_handle=prerequisite_handle,
        )

    def wait(self, tasks: list[_StoreTask]) -> None:
        # Wait consumes task handles; do not Check them again after this call.
        first_error: Exception | None = None
        for item in tasks:
            try:
                item.store.wait(item.task)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def check(self, task: _StoreTask) -> bool:
        return bool(task.store.check(task.task))

    def close(self) -> None:
        with self._store_lock:
            stores = tuple(self._stores.values())
            self._stores.clear()
            self._transfer_layouts.clear()
            self._lookup_store_signatures.clear()
        for store in stores:
            close = getattr(store, "close", None)
            if callable(close):
                close()
