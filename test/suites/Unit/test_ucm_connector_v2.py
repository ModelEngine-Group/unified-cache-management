import enum
import importlib.util
import inspect
import json
import math
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# The repository's top-level ``ucm`` package applies production patches and
# therefore requires the full runtime dependency set.  These component tests
# load only the isolated v2 package.
for package_name, package_path in (
    ("ucm", REPO_ROOT / "ucm"),
    ("ucm.integration", REPO_ROOT / "ucm" / "integration"),
    ("ucm.integration.vllm", REPO_ROOT / "ucm" / "integration" / "vllm"),
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules.setdefault(package_name, package)

# vLLM is not installed in the lightweight unit-test environment.  Provide the
# small SPI surface needed to import the public connector facade.
for package_name in (
    "vllm",
    "vllm.distributed",
    "vllm.distributed.kv_transfer",
    "vllm.distributed.kv_transfer.kv_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1",
    "vllm.v1",
):
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules.setdefault(package_name, package)

# The connector reuses ucm.utils.Config for launch-config parsing; that
# module's logging backend is a compiled extension beside
# ucm/shared/infra, absent in this environment.  Stub the backend so the
# real Config code runs; its logs simply go nowhere.
ucm_shared = types.ModuleType("ucm.shared")
ucm_shared.__path__ = [str(REPO_ROOT / "ucm" / "shared")]
sys.modules.setdefault("ucm.shared", ucm_shared)
ucm_infra = types.ModuleType("ucm.shared.infra")
ucmlogger_stub = types.ModuleType("ucm.shared.infra.ucmlogger")


class _StubLevel(enum.IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


ucmlogger_stub.Level = _StubLevel
ucmlogger_stub.setup = lambda *args, **kwargs: None
ucmlogger_stub.flush = lambda: None
ucmlogger_stub.log = lambda *args, **kwargs: None
ucmlogger_stub.log_rate_limit = lambda *args, **kwargs: None
ucm_infra.ucmlogger = ucmlogger_stub
sys.modules.setdefault("ucm.shared.infra", ucm_infra)
sys.modules.setdefault("ucm.shared.infra.ucmlogger", ucmlogger_stub)

vllm_base = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")


class KVConnectorRole(enum.Enum):
    SCHEDULER = "scheduler"
    WORKER = "worker"


class KVConnectorBaseV1:
    def __init__(self, vllm_config, role, kv_cache_config):
        self._vllm_config = vllm_config
        self._role = role
        self._kv_cache_config = kv_cache_config
        self._connector_metadata = None

    def bind_connector_metadata(self, metadata):
        self._connector_metadata = metadata

    def clear_connector_metadata(self):
        self._connector_metadata = None

    def has_connector_metadata(self):
        return self._connector_metadata is not None

    def _get_connector_metadata(self):
        if self._connector_metadata is None:
            raise AssertionError("connector metadata is not bound")
        return self._connector_metadata


class KVConnectorMetadata:
    pass


class KVConnectorWorkerMetadata:
    def aggregate(self, other):
        raise NotImplementedError


class SupportsHMA:
    pass


class KVConnectorHandshakeMetadata:
    pass


vllm_base.KVConnectorBase_V1 = KVConnectorBaseV1
vllm_base.KVConnectorHandshakeMetadata = KVConnectorHandshakeMetadata
vllm_base.KVConnectorMetadata = KVConnectorMetadata
vllm_base.KVConnectorRole = KVConnectorRole
vllm_base.KVConnectorWorkerMetadata = KVConnectorWorkerMetadata
vllm_base.SupportsHMA = SupportsHMA
vllm_base.KVConnectorTransferResults = SimpleNamespace
sys.modules["vllm"].envs = SimpleNamespace(VLLM_KV_CACHE_LAYOUT="NHD")
sys.modules[vllm_base.__name__] = vllm_base
vllm_multi_connector = types.ModuleType(
    "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
)


class MultiConnector:
    """vLLM 0.30 MultiConnector handshake, before UCM's patch."""

    def __init__(self, connectors):
        self._connectors = list(connectors)

    def get_handshake_metadata(self):
        for connector in self._connectors:
            metadata = connector.get_handshake_metadata()
            if metadata is not None:
                return metadata
        return None

    def set_xfer_handshake_metadata(self, metadata):
        for connector in self._connectors:
            connector.set_xfer_handshake_metadata(metadata)

    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        for connector in self._connectors:
            connector.set_xfer_handshake_metadata_pp_aware(metadata)


vllm_multi_connector.MultiConnector = MultiConnector
sys.modules[vllm_multi_connector.__name__] = vllm_multi_connector
# UCM's patch helpers import wrapt for operator proxies only.
if importlib.util.find_spec("wrapt") is None:
    wrapt_stub = types.ModuleType("wrapt")
    wrapt_stub.ObjectProxy = object
    sys.modules["wrapt"] = wrapt_stub

# Single-process worker: rank 0 in every parallel group. Tests patch
# ucm_connector._worker_parallel_ranks for other ranks.
vllm_parallel_state = types.ModuleType("vllm.distributed.parallel_state")
_rank0_group = SimpleNamespace(rank=0, rank_in_group=0, world_size=1)
vllm_parallel_state.get_world_group = lambda: _rank0_group
vllm_parallel_state.get_tp_group = lambda: _rank0_group
vllm_parallel_state.get_pp_group = lambda: _rank0_group
vllm_parallel_state.get_pcp_group = lambda: _rank0_group
sys.modules.setdefault(vllm_parallel_state.__name__, vllm_parallel_state)

vllm_model_utils = types.ModuleType("vllm.model_executor.models.utils")


def extract_layer_index(layer_name, num_attn_module=1):
    subnames = layer_name.split(".")
    int_vals = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    # Like vLLM: several attention modules per layer number their modules.
    if num_attn_module > 1 and "attn" in layer_name and len(int_vals) == 2:
        return int_vals[0] * num_attn_module + int_vals[1]
    return int_vals[-1] if int_vals else 0


vllm_model_utils.extract_layer_index = extract_layer_index
vllm_model_executor = types.ModuleType("vllm.model_executor")
vllm_model_executor.__path__ = []
vllm_models = types.ModuleType("vllm.model_executor.models")
vllm_models.__path__ = []
sys.modules.setdefault("vllm.model_executor", vllm_model_executor)
sys.modules.setdefault("vllm.model_executor.models", vllm_models)
sys.modules.setdefault(vllm_model_utils.__name__, vllm_model_utils)


vllm_kv_cache_interface = types.ModuleType("vllm.v1.kv_cache_interface")


class KVCacheSpecKind(str, enum.Enum):
    FULL_ATTENTION = "full_attention"
    MLA_ATTENTION = "mla_attention"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_MLA = "sliding_window_mla"
    SINK_FULL_ATTENTION = "sink_full_attention"
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
    ENCODER_ONLY_ATTENTION = "encoder_only_attention"
    CROSS_ATTENTION = "cross_attention"
    MAMBA = "mamba"
    UNKNOWN = "unknown"


class _KVCacheSpec:
    prefix_cacheable = True


class _AttentionSpec(_KVCacheSpec):
    max_tp_shards = None


class _FullAttentionSpec(_AttentionSpec):
    pass


class _MLAAttentionSpec(_FullAttentionSpec):
    pass


class _SlidingWindowSpec(_AttentionSpec):
    pass


class _SlidingWindowMLASpec(_SlidingWindowSpec):
    pass


class _MambaSpec(_KVCacheSpec):
    pass


def get_kv_cache_spec_kind(spec):
    if getattr(spec, "spec_kind", None) is not None:
        return spec.spec_kind
    if isinstance(spec, _SlidingWindowMLASpec):
        return KVCacheSpecKind.SLIDING_WINDOW_MLA
    if isinstance(spec, _MLAAttentionSpec):
        return KVCacheSpecKind.MLA_ATTENTION
    if isinstance(spec, _FullAttentionSpec):
        return KVCacheSpecKind.FULL_ATTENTION
    if isinstance(spec, _SlidingWindowSpec):
        return KVCacheSpecKind.SLIDING_WINDOW
    if isinstance(spec, _MambaSpec):
        return KVCacheSpecKind.MAMBA
    return KVCacheSpecKind.UNKNOWN


@dataclass
class KVCacheTensor:
    size: int
    layers: list[str]
    layer_stride: int
    block_stride: int
    offset: int = 0
    host_resident: bool = False


@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]
    kv_cache_spec: _KVCacheSpec
    is_eagle_group: bool = False
    host_resident: bool = False
    enable_kv_transfer: bool = True


@dataclass
class KVCacheConfig:
    num_blocks: int
    kv_cache_tensors: list[KVCacheTensor]
    kv_cache_groups: list[KVCacheGroupSpec]
    prefix_cache_retention_interval: int | None = None
    kv_cache_layout: str | None = None

    @property
    def prefix_cacheable_group_ids(self):
        return prefix_cacheable_group_ids(self.kv_cache_groups)


for name, value in (
    ("KVCacheSpecKind", KVCacheSpecKind),
    ("KVCacheTensor", KVCacheTensor),
    ("KVCacheGroupSpec", KVCacheGroupSpec),
    ("KVCacheConfig", KVCacheConfig),
    ("KVCacheSpec", _KVCacheSpec),
    ("AttentionSpec", _AttentionSpec),
    ("FullAttentionSpec", _FullAttentionSpec),
    ("MLAAttentionSpec", _MLAAttentionSpec),
    ("SlidingWindowSpec", _SlidingWindowSpec),
    ("SlidingWindowMLASpec", _SlidingWindowMLASpec),
    ("MambaSpec", _MambaSpec),
    ("get_kv_cache_spec_kind", get_kv_cache_spec_kind),
):
    setattr(vllm_kv_cache_interface, name, value)
vllm_kv_cache_interface.CircularBufferSpec = type(
    "CircularBufferSpec", (_KVCacheSpec,), {"prefix_cacheable": False}
)


@dataclass
class KpoolTailSpec(_SlidingWindowSpec):
    """GLM kpool scratch. vLLM derives it from SlidingWindowSpec."""

    block_size: int
    sliding_window: int
    tokens_per_state: int = 1
    prefix_cacheable = False


vllm_kv_cache_interface.KpoolTailSpec = KpoolTailSpec


@dataclass
class UniformTypeKVCacheSpecs:
    kv_cache_specs: dict[str, object]

    @property
    def block_size(self):
        return next(iter(self.kv_cache_specs.values())).block_size

    @property
    def prefix_cacheable(self):
        return all(spec.prefix_cacheable for spec in self.kv_cache_specs.values())


vllm_kv_cache_interface.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs
vllm_kv_cache_interface.HiddenStateCacheSpec = type(
    "HiddenStateCacheSpec", (_MLAAttentionSpec,), {}
)
sys.modules[vllm_kv_cache_interface.__name__] = vllm_kv_cache_interface


# vllm.v1.kv_cache_layout: physical stride permutation of [L, B, H, N, C].
class KVCacheLayout(enum.Enum):
    LBHNC = (0, 1, 2, 3, 4)
    LBNHC = (0, 1, 3, 2, 4)
    LHBNC = (0, 2, 1, 3, 4)
    BLHNC = (1, 0, 2, 3, 4)
    BLNHC = (1, 0, 3, 2, 4)
    BHLNC = (1, 2, 0, 3, 4)

    @property
    def is_block_compact(self):
        return set(self.value[:2]) == {0, 1}

    @property
    def is_block_outermost(self):
        return self.value[0] == 1


vllm_kv_cache_layout = types.ModuleType("vllm.v1.kv_cache_layout")
vllm_kv_cache_layout.KVCacheLayout = KVCacheLayout
sys.modules.setdefault(vllm_kv_cache_layout.__name__, vllm_kv_cache_layout)

# Geometry entry points used by the connector. Attention spans DCP ranks;
# recurrent state remains a whole per-rank snapshot, as in the native helpers.
vllm_v1_core = types.ModuleType("vllm.v1.core")
vllm_v1_core.__path__ = []
sys.modules.setdefault("vllm.v1.core", vllm_v1_core)
vllm_kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")


def resolve_dcp_kv_block_size(spec, dcp_world_size):
    specs = (
        tuple(spec.kv_cache_specs.values())
        if isinstance(spec, UniformTypeKVCacheSpecs)
        else (spec,)
    )
    return (
        spec.block_size * dcp_world_size
        if all(isinstance(s, _AttentionSpec) for s in specs)
        else spec.block_size
    )


def dcp_world_size_for_kv_cache_spec(spec, dcp_world_size):
    if dcp_world_size <= 1:
        return 1
    inner = spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        inner = next(iter(spec.kv_cache_specs.values()))
    return dcp_world_size if isinstance(inner, _FullAttentionSpec) else 1


def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups
    if len(groups) <= 1:
        block_size = (
            groups[0].kv_cache_spec.block_size
            if groups
            else vllm_config.cache_config.block_size
        )
        return block_size * dcp, block_size * dcp
    scheduler = math.lcm(
        *(resolve_dcp_kv_block_size(g.kv_cache_spec, dcp) for g in groups)
    )
    return scheduler, scheduler


vllm_kv_cache_utils.resolve_dcp_kv_block_size = resolve_dcp_kv_block_size
vllm_kv_cache_utils.dcp_world_size_for_kv_cache_spec = dcp_world_size_for_kv_cache_spec
vllm_kv_cache_utils.resolve_kv_cache_block_sizes = resolve_kv_cache_block_sizes
sys.modules.setdefault(vllm_kv_cache_utils.__name__, vllm_kv_cache_utils)

from ucm.integration.vllm.request_hasher import RequestHasher  # noqa: E402
from ucm.integration.vllm.v2.connector_scheduler import (  # noqa: E402
    RequestState,
    UCMConnectorMetadata,
    UCMConnectorScheduler,
)
from ucm.integration.vllm.v2.connector_worker import (  # noqa: E402
    UCMConnectorWorker,
    UCMTransferBuilder,
    UCMWorkerMetadata,
)
from ucm.integration.vllm.v2.layout import RestorePolicy  # noqa: E402
from ucm.integration.vllm.v2.layout import (  # noqa: E402
    UCMKVCacheLayout,
    parse_kv_cache_config,
)
from ucm.integration.vllm.v2.runtime import (  # noqa: E402
    make_common_ucm_chunk_ids,
    make_ucm_chunk_ids,
)
from ucm.integration.vllm.v2.test_stores import (
    FileTestStore,
    MemoryTestStore,
    record_segments,
)
from ucm.integration.vllm.v2.ucm_connector import UCMConnector  # noqa: E402
from ucm.integration.vllm.v2.ucm_proxy import (  # noqa: E402
    PipelineProxy,
    UCMProxyAdapter,
    UCMProxyError,
)


@dataclass
class FullAttentionSpec(_FullAttentionSpec):
    block_size: int
    tokens_per_state: int = 1
    sliding_window: int | None = None


@dataclass
class UnsupportedSpec(_KVCacheSpec):
    """A spec of any kind, for checking which kinds the parser accepts."""

    block_size: int
    spec_kind: KVCacheSpecKind = KVCacheSpecKind.UNKNOWN
    prefix_cacheable: bool = True
    tokens_per_state: int = 1


@dataclass
class MLAAttentionSpec(_MLAAttentionSpec):
    block_size: int
    tokens_per_state: int = 1
    sliding_window: int | None = None


@dataclass
class MambaSpec(_MambaSpec):
    block_size: int
    mamba_cache_mode: str = "align"
    tp_replicated: bool = False
    shapes: tuple[tuple[int, ...], ...] | None = None
    dtypes: tuple[object, ...] | None = None
    page_size_bytes: int | None = None


@dataclass
class AscendSlidingWindowMLASpec(_SlidingWindowMLASpec):
    block_size: int
    compress_ratio: int
    sliding_window: int
    tokens_per_state: int = 1

    @property
    def storage_block_size(self):
        return self.block_size


@dataclass
class AscendSFAIndexerCacheSpec(FullAttentionSpec):
    """Ascend SFA indexer cache; DCP ranks replicate its kernel rows."""

    sfa_dcp_replicated_indexer_size: int = 1


for package_name in ("vllm_ascend", "vllm_ascend.core"):
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules.setdefault(package_name, package)
vllm_ascend_kv_cache_interface = types.ModuleType("vllm_ascend.core.kv_cache_interface")
vllm_ascend_kv_cache_interface.AscendSlidingWindowMLASpec = AscendSlidingWindowMLASpec
vllm_ascend_kv_cache_interface.AscendSFAIndexerCacheSpec = AscendSFAIndexerCacheSpec
vllm_ascend_kv_cache_interface.AscendIndexerKPoolTailSpec = type(
    "AscendIndexerKPoolTailSpec", (_KVCacheSpec,), {"prefix_cacheable": False}
)
sys.modules.setdefault(
    vllm_ascend_kv_cache_interface.__name__, vllm_ascend_kv_cache_interface
)


def group(names, specs, *, enable_kv_transfer=True):
    if not isinstance(specs, dict):
        spec = specs
    else:
        spec = UniformTypeKVCacheSpecs(specs)
    return SimpleNamespace(
        layer_names=list(names),
        kv_cache_spec=spec,
        is_eagle_group=False,
        enable_kv_transfer=enable_kv_transfer,
    )


def prefix_cacheable_group_ids(groups):
    """vLLM's KVCacheConfig.prefix_cacheable_group_ids for these groups."""
    return tuple(
        group_id
        for group_id, group in enumerate(groups)
        if getattr(group, "enable_kv_transfer", True)
        and group.kv_cache_spec.prefix_cacheable
    )


def config(*groups, num_blocks=8, tensors=(), kv_cache_layout="LBHNC"):
    return SimpleNamespace(
        kv_cache_groups=list(groups),
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_layout=kv_cache_layout,
        prefix_cacheable_group_ids=prefix_cacheable_group_ids(groups),
    )


class TextConfig(SimpleNamespace):
    """hf_text_config stand-in: attributes plus PretrainedConfig.to_dict."""

    def __init__(self, model_type="llama", num_hidden_layers=4, **fields):
        super().__init__(
            model_type=model_type, num_hidden_layers=num_hidden_layers, **fields
        )

    def to_dict(self):
        return dict(vars(self))


# Dense-model text config: no transient alignment and no compressor tails.
DENSE_TEXT_CONFIG = TextConfig()


def vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="org/model",
            revision=None,
            code_revision=None,
            dtype="torch.bfloat16",
            hf_config=TextConfig(),
            hf_text_config=TextConfig(),
        ),
        cache_config=SimpleNamespace(
            block_size=4,
            cache_dtype="auto",
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "v2_storage_path": str(Path(tempfile.gettempdir()) / "ucm-v2-test")
            }
        ),
        device_config=SimpleNamespace(device_type="cpu"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
            data_parallel_size=1,
            world_size=2,
            rank=0,
        ),
        speculative_config=None,
        additional_config={},
        instance_id="engine",
        use_v2_model_runner=False,
    )


def plan_token_starts(token_start, token_end, num_blocks):
    """First token of each of a plan's UCM chunks in [token_start, token_end)."""
    chunk_size = (token_end - token_start) // num_blocks
    return np.arange(num_blocks, dtype=np.uint64) * np.uint64(chunk_size) + np.uint64(
        token_start
    )


def make_layout(
    raw_config,
    caches,
    *,
    scheduler_block_size=8,
    device_type="cpu",
    num_attn_module=1,
    **kwargs,
):
    """Parse a config and register its caches."""
    parsed = parse_spec(
        raw_config,
        scheduler_block_size=scheduler_block_size,
        device_type=device_type,
        num_attn_module=num_attn_module,
        **kwargs,
    )
    return make_cache_layout(
        parsed, caches, kv_cache_config=raw_config, num_hidden_layers=64
    )


def parse_spec(kv_cache_config, *, hf_text_config=DENSE_TEXT_CONFIG, **kwargs):
    """parse_kv_cache_config for a dense model unless the test passes its own."""
    return parse_kv_cache_config(
        kv_cache_config, hf_text_config=hf_text_config, **kwargs
    )


def make_cache_layout(*args, **kwargs):
    from ucm.integration.vllm.v2.runtime import ParallelRank, ParallelTopology

    layout = UCMKVCacheLayout(*args, **kwargs)
    layout.build_store_layouts(
        ParallelTopology(),
        ParallelRank(),
        ((0, layout.export_layout_metadata(1), None),),
    )
    return layout


def logical_id(byte, restore_policy=RestorePolicy.FA):
    """Single-rank local UCM chunk id of a block whose common id is ``byte``
    repeated."""
    from ucm.integration.vllm.v2.runtime import make_ucm_chunk_ids

    return make_ucm_chunk_ids((bytes([byte]) * 15,), restore_policy)[0]


class FakeRequest:
    def __init__(self, request_id, token_count):
        self.request_id = request_id
        self.all_token_ids = list(range(token_count))


def replica_pair(info):
    return info.index, info.count


def make_connector_scheduler(
    parsed,
    proxy=None,
    hasher=None,
    *,
    recompute_tokens=1,
    parallel=None,
    load_threshold_tokens=0,
):
    """Real scheduler with explicit synthetic seed; handshake tested separately."""
    from ucm.integration.vllm.v2.runtime import ParallelTopology, UCMRuntimeContext

    context = UCMRuntimeContext(
        role=KVConnectorRole.SCHEDULER,
        device_type=parsed.device_type,
        engine_id="component-test",
        global_rank=None,
        rank=None,
        parallel_topology=parallel or ParallelTopology(),
        launch_config={"load_tokens_threshold": load_threshold_tokens},
        use_layerwise=False,
        spec=parsed,
        fingerprint="component-test",
    )
    hasher = hasher or RequestHasher("component-test")
    adapter = UCMProxyAdapter(PipelineProxy(proxy or RecordingTestStore()))
    with mock.patch(
        "ucm.integration.vllm.v2.connector_scheduler.create_proxy", return_value=adapter
    ):
        scheduler = UCMConnectorScheduler(context)
    scheduler._recompute_tokens = recompute_tokens
    scheduler._chunk_hasher = hasher.make_request_block_hasher(
        parsed.chunk_size, b"seed"
    )
    return scheduler


def ucm_ids(common_ucm_chunk_ids, restore_policy=RestorePolicy.FA):
    """UCM object ids for one restore policy, shared by all ranks."""
    return make_ucm_chunk_ids(common_ucm_chunk_ids, restore_policy)


def request_ucm_ids(scheduler, request):
    """Single-rank local UCM chunk ids of a request per restore policy, plan
    order."""
    common_ucm_chunk_ids = make_common_ucm_chunk_ids(scheduler._chunk_hasher(request))
    return tuple(
        ucm_ids(common_ucm_chunk_ids, policy) for policy in scheduler.groups_by_policy
    )


def seed_request_state(
    scheduler,
    request,
    external_hit_tokens=0,
    common_ucm_chunk_ids=(),
    hbm_hit_tokens=0,
):
    """Seed scheduler-owned state without querying the synthetic store."""
    state = RequestState(
        hbm_hit_tokens=hbm_hit_tokens,
        external_hit_tokens=external_hit_tokens,
        num_token_ids=len(request.all_token_ids),
        token_processed=hbm_hit_tokens + external_hit_tokens,
        common_ucm_chunk_ids=tuple(common_ucm_chunk_ids),
        need_load=external_hit_tokens > 0,
    )
    scheduler.requests[str(request.request_id)] = state
    return state


class RecordingTestStore:
    backend = None

    def __init__(self):
        self.present = set()
        self.lookup_calls = []
        self.load_calls = []
        self.dump_calls = []

    def lookup(self, keys):
        self.lookup_calls.append(tuple(keys))
        return [key in self.present for key in keys]

    def lookup_on_prefix(self, keys):
        self.lookup_calls.append(tuple(keys))
        for index, key in enumerate(keys):
            if key not in self.present:
                return index - 1
        return len(keys) - 1

    def lookup_on_reverse(self, keys):
        self.lookup_calls.append(tuple(keys))
        for index in range(len(keys) - 1, -1, -1):
            if keys[index] in self.present:
                return index
        return -1

    def load(self, block_ids, offsets, ptrs, sizes):
        self.load_calls.append((block_ids, offsets, ptrs, sizes))

    def commit(self, success, staging_ids, object_ids):
        if success:
            self.present.update(object_ids)

    def dump(self, block_ids, offsets, ptrs, sizes, metadata=(), record_nbytes=None):
        self.dump_calls.append((block_ids, offsets, ptrs, sizes, metadata))


def _flat_layout_records(layout, operation, metadata, *args, **kwargs):
    """Flatten the native matrices only for existing byte-address assertions."""
    transfers = getattr(UCMTransferBuilder(layout), f"build_{operation}_transfers")(
        metadata, *args, **kwargs
    )
    block_ids = tuple(
        key
        for t in transfers
        for key in t.ucm_chunk_ids
        for _ in range(t.ptrs.shape[1])
    )
    fields = {
        field: (
            np.concatenate([getattr(t, attr).reshape(-1) for t in transfers])
            if transfers
            else np.empty(0, dtype=np.uint64)
        )
        for field, attr in (
            ("offsets", "ucm_chunk_offsets"),
            ("ptrs", "ptrs"),
            ("sizes", "sizes"),
        )
    }
    return SimpleNamespace(block_ids=block_ids, **fields)


def make_transfer(keys, ptrs, sizes, offsets, metadata=()):
    from ucm.integration.vllm.v2.ucm_proxy import UCMProxyTransfer

    indices = np.arange(len(keys), dtype=np.int64)
    lengths = np.max(offsets + sizes, axis=1, initial=0)
    metadata = metadata or tuple(
        json.dumps({"type": "FA", "chunk_index": int(i)}) for i in indices
    )
    return UCMProxyTransfer(keys, indices, ptrs, sizes, offsets, metadata, lengths)


def _submit_segments(adapter, operation, keys, offsets, ptrs, sizes):
    """Feed the existing byte fixtures through the native [K,S] adapter."""

    grouped = {}
    for key, offset, ptr, size in zip(keys, offsets, ptrs, sizes, strict=True):
        grouped.setdefault(key, []).append((offset, ptr, size))
    for key, segments in grouped.items():
        offsets, ptrs, sizes = np.asarray(segments, dtype=np.uint64).T
        adapter.submit(
            operation,
            make_transfer((key,), ptrs[None, :], sizes[None, :], offsets[None, :]),
        )


def compile_access(
    group_layout, *, token_offsets=0, token_counts=None, tokens_per_block=None
):
    """Fixed token ranges of every segment, for address assertions.

    Uses the layout's own token-to-byte conversion and resolve_ptrs.
    GroupStoreLayout.build applies the same whole-block and state-alignment
    rules in production. The block span defaults to the first layer's
    native block size.
    """
    tokens_per_block = tokens_per_block or int(
        group_layout.layer_specs[0].kv_cache_spec.block_size
    )
    token_starts = np.atleast_1d(np.asarray(token_offsets, dtype=np.int64))
    token_lengths = (
        tokens_per_block - token_starts
        if token_counts is None
        else np.atleast_1d(np.asarray(token_counts, dtype=np.int64))
    )
    token_starts, token_lengths = np.broadcast_arrays(token_starts, token_lengths)
    if (
        (token_starts < 0)
        | (token_lengths <= 0)
        | (token_starts + token_lengths > tokens_per_block)
    ).any():
        raise ValueError("Token range must lie inside one physical block")
    is_partial = ((token_starts != 0) | (token_lengths != tokens_per_block)).any()
    if is_partial and not group_layout.supports_partial_tokens:
        raise ValueError(
            f"KV cache group {group_layout.group_id} requires whole-block access"
        )
    for tokens in (token_starts, token_lengths):
        if (
            np.multiply.outer(tokens.astype(np.uint64), group_layout.states_per_block)
            % np.uint64(tokens_per_block)
        ).any():
            raise ValueError("Token ranges must align with stored states")
    return SimpleNamespace(
        layout=group_layout,
        tokens_per_block=tokens_per_block,
        block_byte_offsets=group_layout.tokens_to_bytes(
            token_starts.astype(np.uint64), tokens_per_block
        ),
        segment_bytes=group_layout.tokens_to_bytes(
            token_lengths.astype(np.uint64), tokens_per_block
        ),
    )


def _resolved_access(access, blocks, *, token_offsets=None, segment_mask=None):
    """[block, segment] pointers and sizes of a compiled access."""
    group_layout = access.layout
    columns = slice(None) if segment_mask is None else segment_mask
    blocks = np.asarray(blocks, dtype=np.uint64)
    blocks_per_window = len(access.segment_bytes)
    num_windows = len(blocks) // blocks_per_window
    fixed_offsets = access.block_byte_offsets[:, columns]
    ptrs = (
        group_layout.resolve_ptrs(blocks, segment_mask=segment_mask).reshape(
            num_windows, blocks_per_window, fixed_offsets.shape[1]
        )
        + fixed_offsets
    )
    if token_offsets is not None:
        ptrs = ptrs + group_layout.tokens_to_bytes(
            np.asarray(token_offsets, dtype=np.uint64),
            access.tokens_per_block,
            segment_mask,
        ).reshape(num_windows, blocks_per_window, fixed_offsets.shape[1])
    ptrs = ptrs.reshape(len(blocks), fixed_offsets.shape[1])
    sizes = np.tile(access.segment_bytes[:, columns], (num_windows, 1))
    return ptrs, sizes


def _flat_record_access(record, blocks, key_count, **kwargs):
    token_starts = np.zeros(key_count, dtype=np.uint64)
    offsets, ptrs, sizes = record.resolve_matrices(blocks, token_starts, **kwargs)
    return offsets.reshape(-1), ptrs.reshape(-1), sizes.reshape(-1), ptrs.shape[1]


def _scheduler_metadata(
    scheduler,
    scheduled_tokens,
    *,
    preempted_req_ids=(),
    finished_req_ids=(),
    state_offloads=None,
    block_tables=None,
):
    """Use the scheduler-output entry point for existing dispatch fixtures.

    state_offloads maps request id -> (group_id, vllm_block_id, boundary)
    hand-offs, the engine's boundary_state_offloads.
    """
    block_state = SimpleNamespace(
        req_ids=tuple(r for r in scheduled_tokens if r in scheduler.requests),
        get_block_ids=lambda r: (block_tables or {}).get(
            r, tuple([] for _ in scheduler.spec.groups)
        ),
        boundary_state_offloads=state_offloads or {},
    )
    output = SimpleNamespace(
        kv_connector_block_state=block_state,
        num_scheduled_tokens=scheduled_tokens,
        total_num_scheduled_tokens=sum(scheduled_tokens.values()),
        preempted_req_ids=set(preempted_req_ids),
        finished_req_ids=set(finished_req_ids),
    )
    return scheduler.build_connector_meta(output)


class ByteMemory:
    def __init__(self):
        self.values = {}

    def write(self, ptr, data):
        ptr = int(ptr)
        for index, value in enumerate(data):
            self.values[ptr + index] = value

    def read(self, ptr, size):
        ptr, size = int(ptr), int(size)
        return bytes(self.values.get(ptr + index, 0) for index in range(size))


class MemoryFixtureStore(MemoryTestStore):
    def __init__(self, memory):
        super().__init__(MemoryByteAccess(memory))


class MemoryByteAccess:
    def __init__(self, memory):
        self.memory = memory
        self.register_calls = []
        self.synchronize_calls = 0

    def register_tensors(self, kv_caches):
        self.register_calls.append(kv_caches)

    def synchronize(self):
        self.synchronize_calls += 1

    def read(self, ptr, size):
        return self.memory.read(ptr, size)

    def write(self, ptr, payload):
        self.memory.write(ptr, payload)


class FakeTensor:
    """Pointer/shape/stride-only tensor: the API layout views read.

    A 3-D [blocks, states, channels] view is the single-head BHNC view
    [blocks, 1, states, channels].
    """

    def __init__(self, ptr, shape, strides, element_size=1, storage_base_ptr=None):
        shape, strides = tuple(shape), tuple(strides)
        # One head reads the same as BHNC or BNHC.
        self._single_head = len(shape) == 3
        if self._single_head:
            shape = (shape[0], 1, *shape[1:])
            strides = (strides[0], shape[2] * strides[1], *strides[1:])
        self._ptr = ptr
        self.shape = shape
        self._strides = strides
        self._element_size = element_size
        self._storage_base_ptr = ptr if storage_base_ptr is None else storage_base_ptr

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return math.prod(self.shape)

    def data_ptr(self):
        return self._ptr

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self._storage_base_ptr)

    def stride(self, index=None):
        return self._strides if index is None else self._strides[index]

    def element_size(self):
        return self._element_size

    def permute(self, *dims):
        if self._single_head and dims == (0, 2, 1, 3):
            return self
        return FakeTensor(
            self._ptr,
            tuple(self.shape[d] for d in dims),
            tuple(self._strides[d] for d in dims),
            self._element_size,
            self._storage_base_ptr,
        )

    def unbind(self, dim):
        assert dim == 0
        return tuple(
            FakeTensor(
                self._ptr + i * self._strides[0] * self._element_size,
                self.shape[1:],
                self._strides[1:],
                self._element_size,
                self._storage_base_ptr,
            )
            for i in range(self.shape[0])
        )


def bind_worker(worker, metadata):
    from dataclasses import replace

    from ucm.integration.vllm.v2.connector_worker import UCMWorkerMetadata

    if not hasattr(worker, "_test_scheduler"):
        scheduler = make_connector_scheduler(worker.spec)
        scheduler._proxy = worker._proxy
        worker._failed_dump_attempts = set()
        worker._test_scheduler = scheduler
        worker._parallel_rank = worker.transfer_builder.rank
        if not hasattr(worker, "_worker_metadata"):
            worker._worker_metadata = UCMWorkerMetadata()
    scheduler = worker._test_scheduler
    prepared = replace(
        metadata,
        requests={
            rid: scheduler._attach_save_jobs(request)
            for rid, request in metadata.requests.items()
        },
    )
    worker.bind_connector_metadata(prepared)


def finish_worker_save(worker):
    worker.wait_for_save()
    if not hasattr(worker, "_test_scheduler"):
        return
    scheduler = worker._test_scheduler
    result = worker.build_connector_worker_meta()
    worker._test_last_result = result
    if result is not None:
        scheduler.update_connector_output(
            SimpleNamespace(kv_connector_worker_meta=result)
        )
    for rid in worker._get_connector_metadata().requests:
        scheduler.request_finished(rid)


class KVCacheSpecTest(unittest.TestCase):
    def test_default_chunk_is_lcm_of_stored_group_spans(self):
        fa_groups = (
            group(["model.layers.0.attn"], FullAttentionSpec(8)),
            group(["model.layers.1.attn"], FullAttentionSpec(16)),
        )
        attention_only = parse_spec(config(*fa_groups), scheduler_block_size=16)
        self.assertEqual(attention_only.chunk_size, 16)
        self.assertEqual(attention_only.blocks_per_chunk, (2, 1))
        with_state = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(8)),
                group(["model.layers.1.mixer"], MambaSpec(16)),
            ),
            scheduler_block_size=16,
        )
        self.assertEqual(with_state.chunk_size, 16)
        self.assertEqual(with_state.blocks_per_chunk, (2, 1))

    def test_spec_is_frozen(self):
        from dataclasses import FrozenInstanceError

        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(8))),
            scheduler_block_size=8,
        )
        with self.assertRaises(FrozenInstanceError):
            parsed.chunk_size = 16
        with self.assertRaises(FrozenInstanceError):
            parsed.groups[0].block_size = 2

    def test_direct_custom_chunk_keeps_base_hash_size(self):
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
            chunk_size=512,
        )

        self.assertEqual(parsed.chunk_size, 512)
        self.assertEqual(parsed.blocks_per_chunk[0], 4)

    def test_hybrid_classifies_attention_and_state(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            scheduler_block_size=128,
        )

        self.assertEqual(
            tuple(g.group_id for g in parsed.groups if g.is_attention), (0,)
        )
        self.assertEqual(tuple(g.group_id for g in parsed.state_groups), (1,))

    def test_hybrid_rejects_non_align_mamba(self):
        with self.assertRaisesRegex(ValueError, "mamba_cache_mode='align'"):
            parse_spec(
                config(
                    group(["model.layers.0.attn"], FullAttentionSpec(128)),
                    group(
                        ["model.layers.1.mixer"],
                        MambaSpec(128, mamba_cache_mode="none"),
                    ),
                ),
                scheduler_block_size=128,
            )

    def test_hybrid_rejects_misaligned_mamba_block(self):
        with self.assertRaisesRegex(ValueError, "must divide chunk_size"):
            parse_spec(
                config(
                    group(["model.layers.0.attn"], FullAttentionSpec(128)),
                    group(["model.layers.1.mixer"], MambaSpec(256)),
                ),
                scheduler_block_size=128,
                chunk_size=128,
            )

    def test_parse_rejects_fa_slices_that_do_not_tile_the_block(self):
        # A UCM chunk smaller than an FA block must divide it.
        with self.assertRaisesRegex(ValueError, "must divide its block span"):
            parse_spec(
                config(
                    group(["model.layers.0.attn"], FullAttentionSpec(512)),
                    group(["model.layers.1.attn"], FullAttentionSpec(768)),
                ),
                scheduler_block_size=1536,
                chunk_size=384,
            )

    def test_parse_rejects_fa_slices_that_split_a_compressed_row(self):
        # A c128 block stores one KV row per 128 tokens; a slice must hold
        # whole rows.
        with self.assertRaisesRegex(ValueError, "tokens_per_state=128"):
            parse_spec(
                config(
                    group(
                        ["model.layers.0.attn"],
                        MLAAttentionSpec(16384, tokens_per_state=128),
                    )
                ),
                scheduler_block_size=16384,
                chunk_size=64,
            )

    def test_parse_logs_scheduler_granularity_without_guessing_replay_rules(self):
        window = AscendSlidingWindowMLASpec(128, 1, 128)
        window.prefix_replay_tokens = 128
        with self.assertLogs(
            "ucm.integration.vllm.v2.layout.kv_cache", level="INFO"
        ) as logs:
            parse_spec(
                config(
                    group(["model.layers.0.attn"], MLAAttentionSpec(1024)),
                    group(["model.layers.1.swa_cache"], window),
                ),
                scheduler_block_size=1024,
                chunk_size=512,
            )
        self.assertIn("scheduler_block_size=1024 chunk_size=512", "".join(logs.output))
        self.assertNotIn("bounded replay", "".join(logs.output))

    def test_groups_vllm_does_not_store_are_not_persistent(self):
        # A group with transfer off or a non-prefix-cacheable spec is not
        # listed in prefix_cacheable_group_ids, so UCM does not store it.
        for enabled, cacheable in ((False, True), (True, False)):
            with self.subTest(enabled=enabled, cacheable=cacheable):
                parsed = parse_spec(
                    config(
                        group(["model.layers.0.attn"], FullAttentionSpec(8)),
                        group(
                            ["model.layers.1.scratch"],
                            UnsupportedSpec(7, prefix_cacheable=cacheable),
                            enable_kv_transfer=enabled,
                        ),
                    ),
                    scheduler_block_size=8,
                )
                scratch = parsed.groups[1]
                self.assertFalse(scratch.is_persistent)
                self.assertEqual(scratch.kinds, frozenset())
                self.assertEqual(parsed.blocks_per_chunk[1], 0)

    def test_unsupported_restore_semantics_are_rejected(self):
        for kind in (
            KVCacheSpecKind.UNKNOWN,
            KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION,
            KVCacheSpecKind.ENCODER_ONLY_ATTENTION,
            KVCacheSpecKind.CROSS_ATTENTION,
        ):
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(TypeError, "Unsupported"),
            ):
                parse_spec(
                    config(
                        group(
                            ["model.layers.0.attn"], UnsupportedSpec(8, spec_kind=kind)
                        )
                    ),
                    scheduler_block_size=8,
                )

    def test_sink_attention_is_a_supported_kind(self):
        parsed = parse_spec(
            config(
                group(
                    ["model.layers.0.attn"],
                    UnsupportedSpec(8, spec_kind=KVCacheSpecKind.SINK_FULL_ATTENTION),
                )
            ),
            scheduler_block_size=8,
        )
        self.assertEqual(tuple(g.group_id for g in parsed.fa_groups), (0,))

    def test_parse_rejects_misaligned_sliding_groups(self):
        # A sliding tail keeps a static window shape: the boundary must
        # land on the group's block grid, so its token block must divide
        # the cache block.
        with self.assertRaisesRegex(ValueError, "must divide"):
            parse_spec(
                config(
                    group(["model.layers.0.attn"], FullAttentionSpec(128)),
                    group(
                        ["model.layers.1.swa_cache"],
                        AscendSlidingWindowMLASpec(192, 1, 512),
                    ),
                ),
                scheduler_block_size=128,
                chunk_size=128,
            )

    def test_full_attention_with_window_metadata_is_not_sliding(self):
        parsed = parse_spec(
            config(
                group(
                    ["model.layers.0.attn"],
                    FullAttentionSpec(128, sliding_window=4096),
                )
            ),
            scheduler_block_size=128,
        )

        self.assertEqual(
            parsed.groups[0].kinds,
            frozenset((KVCacheSpecKind.FULL_ATTENTION,)),
        )
        self.assertTrue(parsed.groups[0].is_attention)
        self.assertFalse(parsed.groups[0].is_sliding_window)

    def test_dsv4_scheduler_and_worker_share_model_compression_ratios(self):
        c4_name = "model.layers.2.attn"
        c128_name = "model.layers.3.attn"
        fa_specs = {
            c4_name: MLAAttentionSpec(256, tokens_per_state=4),
            c128_name: MLAAttentionSpec(256, tokens_per_state=128),
        }
        windows = (
            group(
                ["model.layers.2.attn.swa_cache"],
                AscendSlidingWindowMLASpec(64, 1, 128),
            ),
            group(
                ["model.layers.2.attn.compressor.state_cache"],
                AscendSlidingWindowMLASpec(4, 1, 8),
            ),
            group(
                ["model.layers.3.attn.compressor.state_cache"],
                AscendSlidingWindowMLASpec(8, 1, 128),
            ),
        )
        runtime_config = vllm_config()
        runtime_config.cache_config.block_size = 256
        runtime_config.model_config.hf_text_config = TextConfig(
            compress_ratios=[0, 0, 4, 128],
            num_hidden_layers=4,
        )
        worker = UCMConnector(
            runtime_config,
            KVConnectorRole.WORKER,
            config(group(fa_specs, fa_specs), *windows),
        )
        self.assertEqual(
            [g.tail_tokens for g in worker.connector_worker.spec.wa_groups], [128, 4, 0]
        )
        for representative in fa_specs.values():
            with self.subTest(representative=representative.tokens_per_state):
                scheduler = UCMConnector(
                    runtime_config,
                    KVConnectorRole.SCHEDULER,
                    config(group(fa_specs, representative), *windows),
                )
                self.assertEqual(
                    [
                        g.tail_tokens
                        for g in scheduler.connector_scheduler.spec.wa_groups
                    ],
                    [128, 4, 0],
                )


class StateLoadTargetTest(unittest.TestCase):
    """Every device restores both the running and checkpoint State slots."""

    def _state_load_blocks(self, device_type):
        from ucm.integration.vllm.v2.layout import (
            UCMKVCacheGroupSpec,
            UCMKVCacheSpec,
        )

        spec = UCMKVCacheSpec(
            groups=(
                UCMKVCacheGroupSpec(
                    0,
                    128,
                    frozenset({KVCacheSpecKind.FULL_ATTENTION}),
                ),
                UCMKVCacheGroupSpec(1, 128, frozenset({KVCacheSpecKind.MAMBA})),
            ),
            chunk_size=128,
            device_type=device_type,
        )
        scheduler = make_connector_scheduler(spec)
        request = FakeRequest("r", 1024)
        fa_keys = tuple(bytes([index]) * 15 for index in range(8))
        state = seed_request_state(
            scheduler,
            request,
            external_hit_tokens=256,
            common_ucm_chunk_ids=fa_keys,
        )
        # Restore boundary 256 lives in block 11; this step's running block
        # for tokens [256, 384) is block 12.
        block_tables = ([1, 2, 3, 4], [0, 11, 12])

        meta = scheduler._build_request_plan("r", state, block_tables, 128)
        state_plans = [
            p for p in meta.load_plans if p.restore_policy is RestorePolicy.STATE
        ]
        for plan in state_plans:
            self.assertEqual(plan.common_ucm_chunk_ids, (fa_keys[1],))
        return [plan.vllm_block_ids_by_group[0].tolist() for plan in state_plans]

    def test_cuda_restores_running_and_checkpoint_blocks(self):
        self.assertEqual(self._state_load_blocks("cuda"), [[12], [11]])

    def test_ascend_restores_running_and_checkpoint_blocks(self):
        self.assertEqual(self._state_load_blocks("npu"), [[12], [11]])


class RequestAwareKeyTest(unittest.TestCase):
    """v2 keys fold vLLM's block-hash extra keys, not only token IDs."""

    def _fa_keys(self, request):
        from ucm.integration.vllm.v2.layout import (
            UCMKVCacheGroupSpec,
            UCMKVCacheSpec,
        )

        spec = UCMKVCacheSpec(
            groups=(
                UCMKVCacheGroupSpec(
                    0,
                    128,
                    frozenset({KVCacheSpecKind.FULL_ATTENTION}),
                ),
            ),
            chunk_size=128,
            device_type="cuda",
        )
        (keys,) = request_ucm_ids(make_connector_scheduler(spec), request)
        return keys

    def test_same_tokens_with_different_images_get_different_keys(self):
        def extra_keys(request, start, end, start_mm_idx):
            # The image placeholder tokens fill the second block only.
            if start == 128:
                return ((request.image_id, 0),), start_mm_idx + 1
            return None, start_mm_idx

        image_a = FakeRequest("a", 384)
        image_a.image_id = "img-a"
        image_b = FakeRequest("b", 384)
        image_b.image_id = "img-b"
        with mock.patch(
            "ucm.integration.vllm.request_hasher.generate_block_hash_extra_keys",
            extra_keys,
        ):
            keys_a = self._fa_keys(image_a)
            keys_b = self._fa_keys(image_b)

        self.assertEqual(keys_a[0], keys_b[0])
        self.assertNotEqual(keys_a[1], keys_b[1])
        self.assertNotEqual(keys_a[2], keys_b[2])


class DumpMetadataTest(unittest.TestCase):
    """Each dumped UCM chunk carries JSON with its attention type and block index."""

    def test_chunk_index_counts_ucm_chunks_from_the_request_start(self):
        from ucm.integration.vllm.v2.connector_scheduler import UCMGroupDispatchPlan
        from ucm.integration.vllm.v2.connector_worker import dump_metadata

        keys = tuple(bytes([index]) * 16 for index in range(3))
        fa = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            keys,
            plan_token_starts(256, 640, len(keys)),
            (),
            group_ids=(),
        )
        wa = UCMGroupDispatchPlan(
            RestorePolicy.WA,
            keys[:1],
            plan_token_starts(512, 640, len(keys[:1])),
            (),
            group_ids=(),
        )
        state = UCMGroupDispatchPlan(
            RestorePolicy.STATE,
            keys[:1],
            plan_token_starts(896, 1024, len(keys[:1])),
            (),
            group_ids=(),
        )

        self.assertEqual(
            [json.loads(item) for item in dump_metadata(fa, 128)],
            [
                {"type": "FA", "chunk_index": 2},
                {"type": "FA", "chunk_index": 3},
                {"type": "FA", "chunk_index": 4},
            ],
        )
        self.assertEqual(
            json.loads(dump_metadata(wa, 128)[0]), {"type": "WA", "chunk_index": 4}
        )
        self.assertEqual(
            json.loads(dump_metadata(state, 128)[0]),
            {"type": "State", "chunk_index": 7},
        )

    def test_adapter_passes_metadata_to_dump_only(self):

        proxy = RecordingTestStore()
        adapter = UCMProxyAdapter(PipelineProxy(proxy))
        keys = (b"a" * 16, b"b" * 16)
        grid = np.zeros((2, 1), dtype=np.uint64)
        metadata = (
            '{"type":"FA","chunk_index":0}',
            '{"type":"FA","chunk_index":1}',
        )

        adapter.submit("dump", make_transfer(keys, grid, grid, grid, metadata))
        adapter.submit("load", make_transfer(keys, grid, grid, grid))

        self.assertIs(proxy.dump_calls[-1][4], metadata)
        self.assertEqual(len(proxy.load_calls[-1]), 4)


class MtpLayerIndexTest(unittest.TestCase):
    def test_longcat_layer_ids_match_native_attention_module_numbering(self):
        names = ("model.layers.2.attn.0", "model.layers.2.attn.1")
        layout = make_layout(
            config(group(names, FullAttentionSpec(8))),
            {
                name: FakeTensor(1000 + i * 1000, (8, 1, 8, 1), (8, 8, 1, 1))
                for i, name in enumerate(names)
            },
            num_attn_module=2,
        )
        self.assertEqual(layout.layer_id_by_name, {names[0]: 4, names[1]: 5})

    def test_native_numbering(self):
        from ucm.integration.vllm.v2.layout.kv_cache import _layer_index

        for name, device, expected in (
            ("model.layers.0.attn", "npu", 0),
            ("mtp.0.attn", "npu", 43),
            ("model.mtp.1.indexer", "npu", 44),
            ("mtp.43.attn", "npu", 43),
            ("model.layers.43.attn", "cpu", 43),
            ("model.layers.44.attn", "cuda", 44),
        ):
            with self.subTest(name=name, device=device):
                self.assertEqual(_layer_index(name, device, 43), expected)

    def test_mtp_identity_is_shared_by_selection_ratio_and_tail(self):
        raw = config(
            group(["model.layers.0.attn"], MLAAttentionSpec(512, tokens_per_state=128)),
            group(["mtp.0.attn"], MLAAttentionSpec(512, tokens_per_state=128)),
            group(["mtp.0.swa_cache"], AscendSlidingWindowMLASpec(4, 1, 4)),
            group(
                ["mtp.0.compressor.state_cache"], AscendSlidingWindowMLASpec(4, 1, 128)
            ),
        )
        cfg = vllm_config()
        cfg.device_config.device_type = "npu"
        # mtp.0 is model layer 4 on Ascend; its compressor tail is 128 - 128.
        cfg.model_config.hf_text_config = TextConfig(
            num_hidden_layers=4, compress_ratios=[128, 0, 0, 0, 128]
        )
        with tempfile.TemporaryDirectory() as path:
            cfg.kv_transfer_config.kv_connector_extra_config["v2_storage_path"] = path
            parsed = UCMConnector(
                cfg, KVConnectorRole.SCHEDULER, raw
            ).connector_scheduler.spec
        self.assertEqual(parsed.groups[1].block_token_span, 512)
        self.assertEqual(parsed.groups[3].tail_tokens, 0)
        names = [name for g in raw.kv_cache_groups for name in g.layer_names]
        layout = make_cache_layout(
            parsed,
            {
                name: FakeTensor(0x1000 * (i + 1), (8, 4, 1), (4, 1, 1))
                for i, name in enumerate(names)
            },
            kv_cache_config=raw,
            num_hidden_layers=4,
        )
        self.assertEqual(layout.layer_names_by_id[0], {"model.layers.0.attn"})
        self.assertEqual(set(layout.layer_names_by_id[4]), set(names[1:]))
        # Group-level columns: model layer 0 belongs to group 0 only, so
        # group 1 answers nothing for it; layer 4 names live in group 1.
        self.assertFalse(layout.group_layouts[1].segment_mask(layer_ids=[0]).any())
        group1 = layout.group_layouts[1]
        whole = _resolved_access(
            compile_access(group1),
            [1],
            segment_mask=group1.segment_mask(layer_ids=[4]),
        )
        self.assertTrue(whole[0].size)


class ConnectorSPIContractTest(unittest.TestCase):
    def test_metadata_implements_vllm_contracts(self):
        self.assertTrue(issubclass(UCMConnectorMetadata, KVConnectorMetadata))
        self.assertTrue(issubclass(UCMWorkerMetadata, KVConnectorWorkerMetadata))

    def test_public_override_parameter_names_match_vllm(self):
        expected = {
            "__init__": ("self", "vllm_config", "role", "kv_cache_config"),
            "register_kv_caches": ("self", "kv_caches"),
            "start_load_kv": ("self", "forward_context", "kwargs"),
            "wait_for_layer_load": ("self", "layer_name"),
            "save_kv_layer": (
                "self",
                "layer_name",
                "kv_layer",
                "attn_metadata",
                "kwargs",
            ),
            "update_state_after_alloc": (
                "self",
                "request",
                "blocks",
                "num_external_tokens",
            ),
            "build_connector_meta": ("self", "scheduler_output"),
            "update_connector_output": ("self", "connector_output"),
            "handle_preemptions": ("self", "kv_connector_metadata"),
        }
        for method_name, parameter_names in expected.items():
            with self.subTest(method=method_name):
                method = getattr(UCMConnector, method_name)
                self.assertEqual(
                    tuple(inspect.signature(method).parameters), parameter_names
                )


class HashAndLookupTest(unittest.TestCase):
    def setUp(self):
        self.hasher = RequestHasher("component-test")

    def test_hash_golden_vector(self):
        self.assertEqual(
            self.hasher((b"parent", (1, 2, 3, 4))).hex(),
            "17386ae868f2a973b81ca989cc10b587",
        )

    def test_ucm_chunk_id_layout_is_stable(self):
        for policy, tag in (
            (RestorePolicy.FA, b"\x00"),
            (RestorePolicy.WA, b"\x01"),
            (RestorePolicy.STATE, b"\x02"),
        ):
            self.assertEqual(
                make_ucm_chunk_ids((b"h" * 15,), policy), (b"h" * 15 + tag,)
            )

    def test_global_ids_share_fifteen_hash_bytes_across_restore_policies(self):
        common = make_common_ucm_chunk_ids((b"a" * 16, b"b" * 16))
        self.assertEqual(common, (b"a" * 15, b"b" * 15))
        for policy in RestorePolicy:
            ids = make_ucm_chunk_ids(common, policy)
            self.assertEqual(tuple(key[:15] for key in ids), common)

    def test_state_tag_stays_fixed_when_other_policies_are_absent(self):
        for native in (
            config(group(["model.layers.0.mixer"], MambaSpec(4))),
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(4)),
                group(["model.layers.1.mixer"], MambaSpec(4)),
            ),
        ):
            parsed = parse_spec(native, scheduler_block_size=4, device_type="cpu")
            scheduler = make_connector_scheduler(parsed, proxy=MemoryTestStore())
            try:
                request = FakeRequest("r", 12)
                common = make_common_ucm_chunk_ids(scheduler._chunk_hasher(request))
                policies = scheduler.groups_by_policy
                keys = dict(
                    zip(policies, request_ucm_ids(scheduler, request), strict=True)
                )
                self.assertEqual(
                    keys[RestorePolicy.STATE], tuple(key + b"\x02" for key in common)
                )
            finally:
                scheduler.shutdown()

    def test_writer_rows_split_chunks_once_and_evenly(self):
        from ucm.integration.vllm.v2.runtime import ParallelRank, ParallelTopology

        layer_name = "model.layers.0.attn"
        native = config(group([layer_name], MLAAttentionSpec(4)))
        parsed = parse_spec(native, scheduler_block_size=4, device_type="cpu")
        layout = UCMKVCacheLayout(
            parsed,
            {layer_name: FakeTensor(1000, (8, 4, 2), (8, 2, 1))},
            kv_cache_config=native,
            num_hidden_layers=1,
        )
        positions = np.arange(3, 1003, dtype=np.uint64)
        for count in (1, 2, 8):
            topology = ParallelTopology(tp_size=count)
            pp_metadata = ((0, layout.export_layout_metadata(count), None),)
            stores = []
            for tp_rank in range(count):
                layout.build_store_layouts(topology, ParallelRank(tp_rank), pp_metadata)
                stores.append(layout.store_layouts[0])
            rows = [store.writer_rows(positions) for store in stores]
            self.assertEqual(sorted(np.concatenate(rows).tolist()), list(range(1000)))
            self.assertLessEqual(max(map(len, rows)) - min(map(len, rows)), 1)
            for index, selected in enumerate(rows):
                self.assertTrue(np.all(positions[selected] % count == index))
        # Slicing a plan never renumbers the writer's absolute position.
        self.assertEqual(stores[1].writer_rows(np.array([9, 16, 23])).tolist(), [0])
        # Periodic State checkpoints can still concentrate on one writer.
        self.assertEqual(
            stores[7].writer_rows(np.array([7, 15, 23])).tolist(), [0, 1, 2]
        )

    def test_restore_policies_share_one_hash_and_differ_only_by_tag(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            scheduler_block_size=128,
        )
        coordinator = make_connector_scheduler(
            parsed, RecordingTestStore(), self.hasher
        )
        request = FakeRequest("r", 512)
        values = self.hasher.make_request_block_hasher(128, b"seed")(request)
        fa_keys, state_keys = request_ucm_ids(coordinator, request)

        self.assertEqual(fa_keys[0][:15], values[0][:15])
        self.assertEqual(state_keys[0][:15], values[0][:15])
        self.assertEqual(fa_keys[0][15:].hex(), "00")
        self.assertEqual(state_keys[0][15:].hex(), "02")
        self.assertNotEqual(fa_keys[0], state_keys[0])

    def test_custom_chunk_hashes_at_the_chunk(self):
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
            chunk_size=512,
        )
        proxy = RecordingTestStore()
        coordinator = make_connector_scheduler(
            parsed, proxy, self.hasher, recompute_tokens=0
        )
        request = FakeRequest("r", 1024)
        (keys,) = request_ucm_ids(coordinator, request)
        proxy.present.update(keys)

        result = coordinator.lookup(request, 0)

        self.assertEqual(ucm_ids(result.common_ucm_chunk_ids), keys)
        self.assertEqual(result.external_hit_tokens, 1024)

    def test_direct_prefix_stops_at_first_miss(self):
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
        )
        proxy = RecordingTestStore()
        coordinator = make_connector_scheduler(
            parsed, proxy, self.hasher, recompute_tokens=0
        )
        request = FakeRequest("r", 512)
        (keys,) = request_ucm_ids(coordinator, request)
        proxy.present.update((keys[0], keys[2], keys[3]))

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 128)

    def test_direct_miss_never_moves_an_unaligned_hbm_boundary_backwards(self):
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
            chunk_size=512,
        )
        proxy = RecordingTestStore()
        coordinator = make_connector_scheduler(
            parsed, proxy, RequestHasher("component-test")
        )

        result = coordinator.lookup(FakeRequest("unaligned", 1024), 128)

        self.assertEqual(result.external_hit_tokens, 0)

    def test_direct_full_hit_leaves_the_recompute_margin_block(self):
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
        )
        proxy = RecordingTestStore()
        coordinator = make_connector_scheduler(parsed, proxy, self.hasher)
        request = FakeRequest("r", 512)
        (keys,) = request_ucm_ids(coordinator, request)
        proxy.present.update(keys)

        result = coordinator.lookup(request, 0)

        # The scan ends at the last complete block below the recompute
        # margin (v1 semantics): 384 restored, the last 128 recomputed.
        self.assertEqual(result.external_hit_tokens, 384)

    def test_state_only_model_restores_at_latest_state(self):
        # A pure Mamba model has no FA prefix; any full block can be the
        # restore boundary.
        parsed = parse_spec(
            config(group(["model.layers.0.mixer"], MambaSpec(128))),
            scheduler_block_size=128,
        )
        proxy = RecordingTestStore()
        coordinator = make_connector_scheduler(
            parsed, proxy, self.hasher, recompute_tokens=0
        )
        request = FakeRequest("r", 1024)
        (state_keys,) = request_ucm_ids(coordinator, request)
        proxy.present.update((state_keys[1], state_keys[4]))

        self.assertEqual(coordinator.lookup(request, 0).external_hit_tokens, 640)
        self.assertEqual(
            coordinator.lookup(FakeRequest("short", 512), 0).external_hit_tokens, 256
        )

    def test_hybrid_reverse_selects_latest_common_state(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            scheduler_block_size=128,
        )
        proxy = RecordingTestStore()
        coordinator = make_connector_scheduler(
            parsed, proxy, self.hasher, recompute_tokens=0
        )
        request = FakeRequest("r", 1024)
        fa_keys, state_keys = request_ucm_ids(coordinator, request)
        proxy.present.update(fa_keys)
        proxy.present.add(state_keys[3])

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 512)

    def test_hybrid_full_hit_leaves_one_complete_state_alignment_block(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            scheduler_block_size=128,
        )
        proxy = RecordingTestStore()
        coordinator = make_connector_scheduler(parsed, proxy, self.hasher)
        request = FakeRequest("r", 1024)
        fa_keys, state_keys = request_ucm_ids(coordinator, request)
        proxy.present.update((*fa_keys, *state_keys))

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 896)

    def test_hybrid_miss_still_returns_state_keys_for_later_dump(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            scheduler_block_size=128,
        )
        coordinator = make_connector_scheduler(
            parsed, RecordingTestStore(), self.hasher, recompute_tokens=0
        )

        result = coordinator.lookup(FakeRequest("r", 1024), 0)

        self.assertEqual(result.external_hit_tokens, 0)
        self.assertEqual(len(result.common_ucm_chunk_ids), 8)

    def test_dsv4_requires_fa_prefix_and_latest_wa_boundary(self):
        parsed = parse_spec(
            config(
                group(
                    ["model.layers.2.attn"],
                    MLAAttentionSpec(512, tokens_per_state=4),
                ),
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 128),
                ),
                num_blocks=4,
            ),
            scheduler_block_size=512,
        )
        proxy = RecordingTestStore()
        coordinator = make_connector_scheduler(
            parsed, proxy, self.hasher, recompute_tokens=0
        )
        request = FakeRequest("r", 1536)
        fa_keys, wa_keys = request_ucm_ids(coordinator, request)
        proxy.present.update(fa_keys)
        proxy.present.add(wa_keys[1])

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 1024)
        self.assertEqual(request_ucm_ids(coordinator, request), (fa_keys, wa_keys))

    def test_dsv4_full_hit_leaves_one_complete_canonical_block(self):
        parsed = parse_spec(
            config(
                group(
                    ["model.layers.2.attn"],
                    MLAAttentionSpec(512, tokens_per_state=4),
                ),
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 128),
                ),
                num_blocks=4,
            ),
            scheduler_block_size=512,
        )
        proxy = RecordingTestStore()
        coordinator = make_connector_scheduler(parsed, proxy, self.hasher)
        request = FakeRequest("r", 1536)
        fa_keys, wa_keys = request_ucm_ids(coordinator, request)
        proxy.present.update((*fa_keys, *wa_keys))

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 1024)


class ShardingHandshakeTest(unittest.TestCase):
    def _workers(self, *, tp=2, pp=1, stage_ids=None):
        from ucm.integration.vllm.v2.runtime import ParallelRank, ParallelTopology

        topology = ParallelTopology(tp_size=tp, pp_size=pp)
        specs = (MLAAttentionSpec(4), MambaSpec(4))
        parsed = parse_spec(
            config(
                *(
                    group([f"model.layers.{i}.attn"], native)
                    for i, native in enumerate(specs)
                )
            ),
            scheduler_block_size=4,
            device_type="cpu",
        )
        layouts = []
        for stage in range(pp):
            ids = (0, 1) if stage_ids is None else stage_ids[stage]
            native = config(
                *(
                    group([f"model.layers.{stage*2+i}.attn"] if i in ids else [], value)
                    for i, value in enumerate(specs)
                )
            )
            caches = {
                name: FakeTensor(
                    1000 + i * 1000,
                    (8, 4, 2) if i == 0 else (8, 1, 1, 8),
                    (8, 2, 1) if i == 0 else (8, 8, 8, 1),
                )
                for i, g in enumerate(native.kv_cache_groups)
                for name in g.layer_names
            }
            layouts.append(
                UCMKVCacheLayout(
                    parsed, caches, kv_cache_config=native, num_hidden_layers=4
                )
            )
        templates = tuple(
            (stage, layout.export_layout_metadata(tp), None)
            for stage, layout in enumerate(layouts)
        )
        workers = {}
        for stage, layout in enumerate(layouts):
            for tp_rank in range(tp):
                rank = ParallelRank(tp_rank, stage, 0)
                layout.build_store_layouts(topology, rank, templates)
                worker = SimpleNamespace(
                    spec=parsed,
                    layout=layout,
                    _parallel_rank=rank,
                    fingerprint="test",
                    transfer_builder=object(),
                )
                workers[stage, tp_rank] = UCMConnectorWorker.get_handshake_metadata(
                    worker
                )
        return parsed, topology, workers

    def test_complete_handshake_initializes_policy_lookup(self):
        for pp in (1, 2):
            spec, topology, workers = self._workers(pp=pp)
            scheduler = make_connector_scheduler(
                spec, proxy=MemoryTestStore(), parallel=topology
            )
            scheduler.context = SimpleNamespace(fingerprint="test")
            try:
                scheduler.set_xfer_handshake_metadata(workers)
                self.assertEqual(
                    list(scheduler.groups_by_policy),
                    [RestorePolicy.FA, RestorePolicy.STATE],
                )
            finally:
                scheduler.shutdown()


class MultiConnectorHandshakePatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Patches the stub MultiConnector, which is already imported.
        import ucm.integration.vllm.patch.v0300.vllm.multi_connector_handshake_patch  # noqa: F401

    class Sub:
        def __init__(self, metadata):
            self.metadata = metadata
            self.received = []

        def get_handshake_metadata(self):
            return self.metadata

        def set_xfer_handshake_metadata(self, metadata):
            self.received.append(metadata)

        def set_xfer_handshake_metadata_pp_aware(self, metadata):
            self.received.append(metadata)

    def test_each_sub_connector_gets_only_its_own_metadata(self):
        from ucm.integration.vllm.patch.v0300.vllm.multi_connector_handshake_patch import (
            MultiConnectorHandshakeMetadata,
        )

        # Mooncake first, UCM second: vLLM would send Mooncake's only.
        mooncake, ucm = self.Sub("mooncake"), self.Sub("ucm")
        worker = MultiConnector([mooncake, ucm])
        self.assertEqual(
            worker.get_handshake_metadata(),
            MultiConnectorHandshakeMetadata(("mooncake", "ucm")),
        )
        scheduler = MultiConnector([self.Sub(None), self.Sub(None)])
        scheduler.set_xfer_handshake_metadata_pp_aware(
            {
                (0, 0): MultiConnectorHandshakeMetadata(("m0", "u0")),
                (0, 1): MultiConnectorHandshakeMetadata((None, "u1")),
            }
        )
        self.assertEqual(
            [sub.received for sub in scheduler._connectors],
            [[{(0, 0): "m0"}], [{(0, 0): "u0", (0, 1): "u1"}]],
        )

    def test_no_metadata_and_unpatched_workers_keep_vllm_behavior(self):
        self.assertIsNone(
            MultiConnector([self.Sub(None), self.Sub(None)]).get_handshake_metadata()
        )
        scheduler = MultiConnector([self.Sub(None), self.Sub(None)])
        scheduler.set_xfer_handshake_metadata({0: "plain"})
        self.assertEqual(
            [sub.received for sub in scheduler._connectors],
            [[{0: "plain"}], [{0: "plain"}]],
        )


class StateBytePageTest(unittest.TestCase):
    def test_dense_permutation_keeps_storage_offset_stride_and_aliasing(self):
        import torch

        from ucm.integration.vllm.v2.layout.view import _as_byte_page

        backing = torch.arange(300, dtype=torch.float32).to(torch.bfloat16)
        original = torch.as_strided(
            backing, (4, 2, 3), (64, 3, 1), storage_offset=5
        ).transpose(1, 2)
        byte_page = _as_byte_page(original)
        self.assertEqual(tuple(byte_page.shape), (4, 1, 1, 12))
        self.assertEqual(byte_page.stride(0), 128)
        self.assertEqual(byte_page.data_ptr(), original.data_ptr())
        for block in range(4):
            expected = backing[5 + 64 * block : 11 + 64 * block].view(torch.uint8)
            self.assertTrue(torch.equal(byte_page[block, 0, 0], expected))
        byte_page[0, 0, 0].zero_()
        self.assertEqual(backing[5:11].tolist(), [0] * 6)

    def test_npu_state_binding_uses_typed_component_payloads(self):
        import torch

        names = ("model.layers.0.attn", "model.layers.1.state")
        backing = torch.zeros(300, dtype=torch.float32)
        component = torch.as_strided(
            backing, (4, 2, 3), (64, 3, 1), storage_offset=5
        ).transpose(1, 2)
        raw = config(
            group([names[0]], FullAttentionSpec(8)),
            group([names[1]], MambaSpec(8)),
            num_blocks=4,
            tensors=(
                SimpleNamespace(
                    layers=(names[1],), offset=0, layer_stride=1024, block_stride=128
                ),
            ),
        )
        layout = make_layout(
            raw,
            {
                names[0]: FakeTensor(1000, (4, 8, 1, 1), (8, 1, 1, 1)),
                names[1]: (component,),
            },
            device_type="npu",
        )
        _, ptrs, sizes = layout.store_layouts[1].resolve_matrices(
            np.array([2], dtype=np.uint64), np.array([0], dtype=np.uint64)
        )
        self.assertEqual(ptrs.tolist(), [[component.data_ptr() + 512]])
        self.assertEqual(sizes.tolist(), [[24]])

    def test_non_dense_state_payload_is_rejected(self):
        import torch

        from ucm.integration.vllm.v2.layout.view import _as_byte_page

        with self.assertRaisesRegex(ValueError, "dense"):
            _as_byte_page(torch.zeros((4, 8, 6))[:, :, ::2])


class ParallelShardTest(unittest.TestCase):
    def test_dcp_scales_only_dcp_sharded_block_spans(self):
        spec = parse_spec(
            config(
                group(["model.layers.0.attn"], MLAAttentionSpec(4)),
                group(["model.layers.1.mixer"], MambaSpec(4)),
            ),
            scheduler_block_size=8,
            dcp_size=2,
        )
        self.assertEqual([g.block_size for g in spec.groups], [4, 4])
        self.assertEqual([g.block_token_span for g in spec.groups], [8, 4])
        self.assertEqual(spec.chunk_size, 8)
        self.assertEqual(spec.blocks_per_chunk, (1, 1))

    def test_parser_uses_native_attention_and_state_dcp_geometry(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], MLAAttentionSpec(4)),
                group(["model.layers.1.attn"], FullAttentionSpec(4)),
                group(["model.layers.2.mixer"], MambaSpec(8)),
            ),
            scheduler_block_size=8,
            dcp_size=2,
        )
        self.assertEqual([g.dcp_sharded for g in parsed.groups], [True, True, False])
        # Supported native specs have no synthetic dcp_sharded attribute.
        self.assertFalse(hasattr(MambaSpec(8), "dcp_sharded"))

    def test_specs_without_vllm_031_sharding_fields(self):
        # The supported engine has no max_tp_shards; only MLA is known to
        # be replicated across TP. State is not token-sharded by DCP.
        with mock.patch.object(_AttentionSpec, "max_tp_shards", None):
            parsed = parse_spec(
                config(
                    group(["model.layers.0.attn"], MLAAttentionSpec(4)),
                    group(["model.layers.1.attn"], FullAttentionSpec(4)),
                    group(["model.layers.2.mixer"], MambaSpec(8)),
                ),
                scheduler_block_size=8,
                dcp_size=2,
            )
        self.assertEqual(
            [g.dcp_sharded for g in parsed.groups],
            [True, True, False],
        )

    def test_group_with_different_tp_splits_stores_complete_group_per_rank(self):
        from ucm.integration.vllm.v2.runtime import ParallelRank, ParallelTopology

        two, three = FullAttentionSpec(4), FullAttentionSpec(4)
        two.max_tp_shards, three.max_tp_shards = 2, 3
        cases = (
            {"model.layers.0.attn": two, "model.layers.1.attn": three},
            {
                "model.layers.0.attn": MLAAttentionSpec(4),
                "model.layers.1.attn": FullAttentionSpec(4),
            },
        )
        for specs in cases:
            with self.subTest(
                specs=tuple(type(value).__name__ for value in specs.values())
            ):
                native = config(group(list(specs), specs))
                parsed = parse_spec(native, scheduler_block_size=4, device_type="cpu")
                caches = {
                    name: FakeTensor(1000 + index * 1000, (8, 4, 2), (8, 2, 1))
                    for index, name in enumerate(specs)
                }
                layout = UCMKVCacheLayout(
                    parsed, caches, kv_cache_config=native, num_hidden_layers=4
                )
                layout.build_store_layouts(
                    ParallelTopology(tp_size=6),
                    ParallelRank(4),
                    ((0, layout.export_layout_metadata(6), None),),
                )
                store = layout.store_layouts[0]
                self.assertEqual(store.tp_shards, 6)
                self.assertEqual(store.replica_count, 1)
                self.assertEqual(store.whole_chunk_nbytes, 96)
                self.assertEqual(store.chunk_segment_bytes.tolist(), [[8, 8]])
                self.assertEqual(store.chunk_segment_offsets.tolist(), [[64, 72]])

    def test_parser_reads_mamba_tp_replication_from_specs(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
                group(["model.layers.2.conv"], MambaSpec(128, tp_replicated=True)),
            ),
            scheduler_block_size=128,
        )
        from ucm.integration.vllm.v2.layout.kv_cache import _spec_tp_shards

        self.assertEqual(
            [
                _spec_tp_shards(MambaSpec(128), 8),
                _spec_tp_shards(MambaSpec(128, tp_replicated=True), 8),
            ],
            [8, 1],
        )
        self.assertEqual(
            [g.dcp_sharded for g in parsed.groups],
            [False, False, False],
        )

    def test_non_persistent_groups_have_no_kind(self):
        # KpoolTailSpec is a SlidingWindowSpec, but the pool is not stored,
        # so it is not a WA group and GLM can run with DCP.

        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], MLAAttentionSpec(4)),
                group(["model.layers.0.indexer.kpool_tail"], KpoolTailSpec(4, 4)),
            ),
            hf_text_config=TextConfig(model_type="glm5_next", index_kpool=4),
            scheduler_block_size=8,
            dcp_size=2,
        )
        kpool_group = parsed.groups[1]
        self.assertFalse(kpool_group.is_persistent)
        self.assertEqual(kpool_group.kinds, frozenset())
        self.assertEqual(parsed.wa_groups, ())
        self.assertEqual(parsed.chunk_size, 8)
        self.assertEqual(parsed.blocks_per_chunk[1], 0)

    def test_empty_pp_sliding_group_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "empty persistent SWA"):
            parse_spec(
                config(
                    group(["model.layers.0.attn"], MLAAttentionSpec(4)),
                    group(
                        [], {"model.layers.3.attn": AscendSlidingWindowMLASpec(4, 1, 8)}
                    ),
                ),
                scheduler_block_size=4,
            )

    def test_topologies_do_not_share_a_fingerprint(self):
        from ucm.integration.vllm.v2.runtime import ParallelTopology, UCMRuntimeContext

        variants = [
            ParallelTopology(tp_size=2),
            ParallelTopology(pp_size=2),
            ParallelTopology(pcp_size=2),
            ParallelTopology(tp_size=2, dcp_size=2),
            ParallelTopology(tp_size=2, dcp_size=2, cp_kv_cache_interleave_size=128),
            ParallelTopology(pp_size=2, pp_partition="1,3"),
        ]
        native = config(group(["model.layers.0.attn"], FullAttentionSpec(4)))
        fingerprints = []
        for topology in variants:
            with mock.patch.object(
                ParallelTopology, "from_config", return_value=topology
            ):
                context = UCMRuntimeContext.from_vllm_config(
                    vllm_config(), KVConnectorRole.SCHEDULER, native
                )
            fingerprints.append(context.fingerprint)
        self.assertEqual(len(set(fingerprints)), len(variants))

    def test_cp_scales_blocks_by_dcp_only(self):
        from ucm.integration.vllm.v2.runtime import ParallelTopology

        topology = ParallelTopology(tp_size=4, pp_size=2, pcp_size=2, dcp_size=2)
        self.assertEqual(topology.world_size, 16)
        native = config(group(["model.layers.0.attn"], FullAttentionSpec(4)))
        scaled = parse_spec(native, scheduler_block_size=8, dcp_size=topology.dcp_size)
        self.assertEqual(scaled.groups[0].block_token_span, 8)
        self.assertEqual(scaled.chunk_size, 8)
        self.assertEqual(scaled.blocks_per_chunk, (1,))

    def test_dcp_rejects_fa_slices(self):
        with self.assertRaisesRegex(ValueError, "whole distributed"):
            parse_spec(
                config(group(["model.layers.0.attn"], FullAttentionSpec(8))),
                scheduler_block_size=16,
                chunk_size=4,
                dcp_size=2,
            )

    def test_topology_projects_resolved_engine_config(self):
        from ucm.integration.vllm.v2.runtime import ParallelTopology

        parallel_config = SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=8,
            cp_kv_cache_interleave_size=1,
        )
        topology = ParallelTopology.from_config(
            SimpleNamespace(parallel_config=parallel_config)
        )
        self.assertEqual(
            (topology.tp_size, topology.pp_size, topology.pcp_size, topology.dcp_size),
            (4, 2, 2, 8),
        )
        self.assertEqual(topology.cp_kv_cache_interleave_size, 1)
        self.assertEqual(topology.world_size, 16)

    def test_ascend_dcp_indexer_replication_folds_adjacent_kernel_rows(self):

        native_spec = AscendSFAIndexerCacheSpec(4, sfa_dcp_replicated_indexer_size=2)
        raw_config = config(
            group(["model.layers.0.indexer"], native_spec), num_blocks=3
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=8,
            dcp_size=2,
            device_type="npu",
        )
        layout = make_cache_layout(
            parsed,
            {"model.layers.0.indexer": FakeTensor(1000, (6, 4, 1, 2), (8, 2, 2, 1))},
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        # DCP doubles the 4-token native block to 8 tokens.
        access = compile_access(
            layout.group_layouts[0],
            token_offsets=np.array([0]),
            token_counts=np.array([8]),
            tokens_per_block=8,
        )
        ptrs, sizes = _resolved_access(access, np.array([2], dtype=np.uint64))
        self.assertEqual(ptrs.reshape(-1).tolist(), [1032])
        self.assertEqual(sizes.reshape(-1).tolist(), [16])

    def test_pp_empty_group_preserves_global_route_without_allocating_view(self):
        raw_config = config(
            group([], FullAttentionSpec(4)),
            group(["model.layers.2.attn"], FullAttentionSpec(4)),
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=4,
        )
        layout = UCMKVCacheLayout(
            parsed,
            {"model.layers.2.attn": FakeTensor(1000, (8, 4, 2), (8, 2, 1))},
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        self.assertEqual(tuple(layout.group_layouts), (1,))
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (b"p" * 15,),
            plan_token_starts(0, 4, 1),
            (np.array([1], dtype=np.uint64), np.array([2], dtype=np.uint64)),
            group_ids=(0, 1),
        )
        meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
        )
        from ucm.integration.vllm.v2.layout.group import GroupLayoutMetadata
        from ucm.integration.vllm.v2.runtime import ParallelRank, ParallelTopology

        templates = (
            (0, layout.export_layout_metadata(1), None),
            (
                1,
                {
                    0: GroupLayoutMetadata((8,), (4,), True, 1),
                    1: GroupLayoutMetadata((), (), True, 1),
                },
                None,
            ),
        )
        layout.build_store_layouts(
            ParallelTopology(pp_size=2), ParallelRank(), templates
        )
        (transfer,) = UCMTransferBuilder(layout).build_dump_transfers(meta)
        self.assertEqual(transfer.ptrs.tolist(), [[1016]])
        self.assertEqual(transfer.ucm_chunk_offsets.tolist(), [[8]])

    def test_pp_empty_sliding_group_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "empty persistent SWA"):
            parse_spec(
                config(
                    group(["model.layers.0.attn"], FullAttentionSpec(4)),
                    group(
                        [],
                        {
                            "model.layers.9.swa_cache": AscendSlidingWindowMLASpec(
                                4, 1, 8
                            )
                        },
                    ),
                ),
                scheduler_block_size=4,
            )

    def test_pp_empty_state_spec_can_be_parsed(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(4)),
                group([], {"model.layers.1.state": MambaSpec(4)}),
            ),
            scheduler_block_size=4,
        )
        self.assertEqual(len(parsed.state_groups), 1)
        self.assertEqual(parsed.blocks_per_chunk[parsed.state_groups[0].group_id], 1)


class LayerwiseLifecycleTest(unittest.TestCase):
    def _worker(self, directory, *, state=False, deferred=False):
        memory = ByteMemory()
        backend = FileTestStore(directory, MemoryByteAccess(memory))
        if deferred:

            class DeferredTestStore:
                def __init__(self):
                    self.events = []
                    self.fail = None

                def register_tensors(self, caches):
                    backend.register_tensors(caches)

                def load(self, *args):
                    self.events.append(("load", args[0]))
                    return ("load", args)

                def dump(self, *args):
                    self.events.append(("dump", args[0]))
                    return ("dump", args)

                def wait(self, task):
                    operation, args = task
                    self.events.append(("wait", operation))
                    if self.fail == operation:
                        raise OSError("deferred failure")
                    getattr(backend, operation)(*args)

                def commit(self, success, staging_ids, object_ids):
                    self.events.append(("commit" if success else "discard", object_ids))
                    backend.commit(success, staging_ids, object_ids)

            proxy = DeferredTestStore()
        else:
            proxy = backend
        names = ("model.layers.0.attn", "model.layers.1.attn")
        groups = [group(names, FullAttentionSpec(4))]
        caches = {
            name: FakeTensor(0x1000 + i * 0x1000, (8, 4, 2), (8, 2, 1))
            for i, name in enumerate(names)
        }
        if state:
            groups.append(group(["model.layers.2.state"], MambaSpec(4)))
            caches["model.layers.2.state"] = FakeTensor(
                0x3000, (8, 1, 1, 8), (8, 8, 8, 1)
            )
        cfg = vllm_config()
        cfg.parallel_config.tensor_parallel_size = 1
        cfg.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
        with mock.patch(
            "ucm.integration.vllm.v2.factory.FileTestStore",
            return_value=proxy,
        ):
            worker = UCMConnector(
                cfg, KVConnectorRole.WORKER, config(*groups)
            ).connector_worker
        worker.register_kv_caches(caches)
        return worker, backend, proxy, memory, names

    def _meta(self, *, load=False, state=False, token_start=0):
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        restore_policies = (
            (RestorePolicy.FA, RestorePolicy.STATE) if state else (RestorePolicy.FA,)
        )
        plans = tuple(
            UCMGroupDispatchPlan(
                restore_policy,
                (bytes([i + 1]) * 15,),
                plan_token_starts(token_start, token_start + 4, 1),
                (np.array([2 if load else 1], dtype=np.uint64),),
                group_ids=(i,),
            )
            for i, restore_policy in enumerate(restore_policies)
        )
        return UCMConnectorMetadata(
            requests={
                "r": RequestDispatchMeta(
                    "r",
                    load_plans=plans if load else (),
                    dump_plans=() if load else plans,
                )
            }
        )

    def test_layerwise_roundtrip_deferred_wait_commit_and_step_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            worker, backend, proxy, memory, names = self._worker(
                directory, deferred=True
            )
            memory.write(0x1008, b"abcdefgh")
            memory.write(0x2008, b"ijklmnop")
            bind_worker(worker, self._meta())
            worker.start_load_kv(None)
            worker.save_kv_layer(names[0], None, None)
            worker.save_kv_layer(names[0], None, None)
            worker.save_kv_layer(names[1], None, None)
            self.assertEqual([e[0] for e in proxy.events], ["dump", "dump"])
            self.assertEqual(backend.lookup((logical_id(1),)), (False,))
            finish_worker_save(worker)
            self.assertEqual(
                [e[0] for e in proxy.events], ["dump", "dump", "wait", "wait", "commit"]
            )
            self.assertEqual(
                backend._path(logical_id(1)).read_bytes(), b"abcdefghijklmnop"
            )
            count = len(proxy.events)
            finish_worker_save(worker)
            self.assertEqual(len(proxy.events), count)
            # New step, same worker; only each selected layer becomes ready at wait.
            bind_worker(worker, self._meta(load=True))
            worker.start_load_kv(None)
            self.assertEqual(memory.read(0x1010, 8), bytes(8))
            worker.wait_for_layer_load(names[0])
            self.assertEqual(memory.read(0x1010, 8), b"abcdefgh")
            self.assertEqual(memory.read(0x2010, 8), bytes(8))
            worker.wait_for_layer_load(names[0])
            worker.wait_for_layer_load(names[1])
            self.assertEqual(memory.read(0x2010, 8), b"ijklmnop")
            finish_worker_save(worker)

    def test_no_forward_facade_finalizes_state_save_once(self):
        with tempfile.TemporaryDirectory() as directory:
            worker, backend, _, _, _ = self._worker(directory, state=True)
            metadata = self._meta(state=True)
            metadata.no_forward = True
            bind_worker(worker, metadata)
            facade = UCMConnector.__new__(UCMConnector)
            facade.connector_worker = worker
            result = object()
            with mock.patch.object(
                KVConnectorBaseV1,
                "get_transfer_results",
                return_value=result,
                create=True,
            ):
                self.assertIs(facade.get_transfer_results(set()), result)
                self.assertTrue(worker._step_save_finalized)
                self.assertFalse(
                    backend._path(logical_id(2, RestorePolicy.STATE)).exists()
                )
            finish_worker_save(worker)
            self.assertTrue(backend._path(logical_id(2, RestorePolicy.STATE)).exists())
            with mock.patch.object(worker._proxy, "commit") as commit:
                worker.finalize_no_forward()
                commit.assert_not_called()
            worker.clear_connector_metadata()

    def test_no_hook_state_is_saved_at_end_and_loaded_before_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            worker, backend, proxy, memory, names = self._worker(directory, state=True)
            for ptr, payload in (
                (0x1008, b"abcdefgh"),
                (0x2008, b"ijklmnop"),
                (0x3008, b"state123"),
            ):
                memory.write(ptr, payload)
            bind_worker(worker, self._meta(state=True))
            worker.start_load_kv(None)
            worker.save_kv_layer(names[0], None, None)
            # Missing attention hook is also covered at end of forward.
            finish_worker_save(worker)
            self.assertEqual(
                backend.lookup((logical_id(1), logical_id(2, RestorePolicy.STATE))),
                (True, True),
            )
            bind_worker(worker, self._meta(load=True, state=True))
            worker.start_load_kv(None)
            self.assertEqual(memory.read(0x3010, 8), b"state123")
            finish_worker_save(worker)
            self.assertEqual(memory.read(0x2010, 8), b"ijklmnop")

    def test_roundtrip_all_physical_orders_and_partial_blocks(self):
        for order in ("LBNHC", "LBHNC", "LHBNC", "BLNHC", "BLHNC", "BHLNC"):
            # Half blocks need token-contiguous NHC; multi-head HNC is
            # head-major, so only whole blocks. A half block is the first
            # or the second slice of the vLLM block.
            cases = ((4, 0), (4, 4), (8, 0)) if order.endswith("NHC") else ((8, 0),)
            for unit, token_start in cases:
                with (
                    self.subTest(order=order, unit=unit, token_start=token_start),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    worker, backend, _, memory, names = self._worker(directory)
                    worker.layout, strides = LayerViewSegmentsTest()._layout(
                        order, unit=unit
                    )
                    worker.spec = worker.layout.spec
                    worker.transfer_builder = UCMTransferBuilder(worker.layout)

                    def address(layer, block, head, token, content):
                        return 1000 + sum(
                            i * strides[a]
                            for a, i in zip(
                                "LBHNC", (layer, block, head, token, content)
                            )
                        )

                    for layer in range(2):
                        for head in range(2):
                            for token in range(8):
                                for content in range(2):
                                    value = (
                                        1 + layer * 64 + head * 24 + token * 2 + content
                                    )
                                    memory.write(
                                        address(layer, 1, head, token, content),
                                        bytes([value]),
                                    )
                                    memory.write(
                                        address(layer, 2, head, token, content), b"\xff"
                                    )
                    bind_worker(worker, self._meta(token_start=token_start))
                    worker.start_load_kv(None)
                    for name in reversed(names):
                        worker.save_kv_layer(name, None, None)
                    finish_worker_save(worker)
                    bind_worker(worker, self._meta(load=True, token_start=token_start))
                    worker.start_load_kv(None)
                    for name in names:
                        worker.wait_for_layer_load(name)
                    finish_worker_save(worker)
                    for layer in range(2):
                        for head in range(2):
                            for token in range(8):
                                for content in range(2):
                                    expected = (
                                        memory.read(
                                            address(layer, 1, head, token, content), 1
                                        )
                                        if token_start <= token < token_start + unit
                                        else b"\xff"
                                    )
                                    self.assertEqual(
                                        memory.read(
                                            address(layer, 2, head, token, content), 1
                                        ),
                                        expected,
                                    )

    def test_model_check_layerwise_oracle_uses_payloads_not_bulk_padding(self):
        import importlib.util

        common_path = REPO_ROOT / "toolkit/ucm_toolkit/tools/model_check/common.py"
        spec = importlib.util.spec_from_file_location(
            "layerwise_check_common", common_path
        )
        common = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules, {"torch": types.ModuleType("torch"), spec.name: common}
        ):
            spec.loader.exec_module(common)
        name = "model.layers.0.attn"
        layout = SimpleNamespace(layer_id_by_name={name: 0})

        def build(metadata, layer_name=None):

            return (
                make_transfer(
                    (b"a" * 15,),
                    np.array([[100]], dtype=np.uint64),
                    np.array([[8 if layer_name else 16]], dtype=np.uint64),
                    np.array([[4]], dtype=np.uint64),
                ),
            )

        worker = SimpleNamespace(
            layout=layout,
            use_layerwise=True,
            transfer_builder=SimpleNamespace(build_dump_transfers=build),
        )
        batch = common._v2_batch(worker, SimpleNamespace(requests={}), "dump")
        self.assertEqual(batch.sizes, (8,))
        worker.use_layerwise = False
        self.assertEqual(
            common._v2_batch(worker, SimpleNamespace(requests={}), "dump").sizes, (16,)
        )
        self.assertEqual(
            len(common._v2_segment_payload(b"a" * 15, np.uint64(4), np.uint64(8))), 8
        )

    def test_failed_write_drains_other_tasks_without_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            worker, backend, proxy, memory, names = self._worker(
                directory, deferred=True
            )
            bind_worker(worker, self._meta())
            worker.start_load_kv(None)
            for name in names:
                worker.save_kv_layer(name, None, None)
            proxy.fail = "dump"
            finish_worker_save(worker)
            self.assertEqual(sum(e == ("wait", "dump") for e in proxy.events), 2)
            self.assertFalse(any(e[0] == "commit" for e in proxy.events))
            self.assertFalse(worker._dump_tasks)
            self.assertEqual(backend.lookup((logical_id(1),)), (False,))

    def test_failed_load_reports_blocks_and_suppresses_save(self):
        with tempfile.TemporaryDirectory() as directory:
            worker, backend, proxy, memory, names = self._worker(
                directory, deferred=True
            )
            metadata = self._meta(load=True)
            from dataclasses import replace

            metadata.requests["r"] = replace(
                metadata.requests["r"], dump_plans=self._meta().requests["r"].dump_plans
            )
            bind_worker(worker, metadata)
            worker.start_load_kv(None)
            proxy.fail = "load"
            worker.wait_for_layer_load(names[0])
            worker.save_kv_layer(names[0], None, None)
            finish_worker_save(worker)
            self.assertEqual(worker.get_block_ids_with_load_errors(), {2})
            self.assertEqual(worker._test_last_result.load_failed_reqs, {"r"})
            self.assertFalse(any(e[0] in ("dump", "commit") for e in proxy.events))


class ProxyAdapterTest(unittest.TestCase):
    def test_simple_file_proxy_publishes_and_loads_complete_record(self):
        key = b"f" * 16
        memory = ByteMemory()
        access = MemoryByteAccess(memory)
        source = bytes(range(12))
        memory.write(0x1000, source[:5])
        memory.write(0x2000, source[5:])
        registered = {"layer": object()}

        with tempfile.TemporaryDirectory() as directory:
            proxy = FileTestStore(directory, access)
            adapter = UCMProxyAdapter(PipelineProxy(proxy))
            adapter.register_tensors(registered)
            _submit_segments(adapter, "dump", (key, key), (0, 5), (4096, 8192), (5, 7))

            self.assertEqual(
                adapter.lookup((key,), np.arange(len((key,)), dtype=np.int64)), (False,)
            )
            adapter.commit(True, (key,), (key,), np.arange(len((key,)), dtype=np.int64))
            self.assertEqual(
                adapter.lookup(
                    (key, b"m" * 16), np.arange(len((key, b"m" * 16)), dtype=np.int64)
                ),
                (True, False),
            )
            self.assertEqual(
                (Path(directory) / f"{key.hex()}.ucm").read_bytes(), source
            )

            _submit_segments(
                adapter, "load", (key, key), (0, 5), (12288, 16384), (5, 7)
            )

        self.assertEqual(access.register_calls, [registered])
        self.assertEqual(memory.read(0x3000, 5), source[:5])
        self.assertEqual(memory.read(0x4000, 7), source[5:])
        self.assertEqual(access.synchronize_calls, 2)

    def test_partial_dumps_stay_hidden_until_selected_keys_commit(self):
        key, other = b"g" * 16, b"h" * 16
        memory = ByteMemory()
        memory.write(0x1000, b"abcd")
        memory.write(0x2000, b"efgh")
        with tempfile.TemporaryDirectory() as directory:
            proxy = FileTestStore(directory, MemoryByteAccess(memory))
            adapter = UCMProxyAdapter(PipelineProxy(proxy))
            # Later layer first; repeated writes must preserve earlier ranges.
            _submit_segments(adapter, "dump", (key,), (4,), (8192,), (4,))
            _submit_segments(adapter, "dump", (other,), (0,), (8192,), (4,))
            self.assertEqual(
                adapter.lookup(
                    (key, other), np.arange(len((key, other)), dtype=np.int64)
                ),
                (False, False),
            )
            _submit_segments(adapter, "dump", (key,), (0,), (4096,), (4,))
            adapter.commit(True, (key,), (key,), np.arange(len((key,)), dtype=np.int64))
            self.assertEqual(
                adapter.lookup(
                    (key, other), np.arange(len((key, other)), dtype=np.int64)
                ),
                (True, False),
            )
            self.assertEqual(proxy._path(key).read_bytes(), b"abcdefgh")
            adapter.commit(
                True, (key,), (key,), np.arange(len((key,)), dtype=np.int64)
            )  # Idempotent retry.
            _submit_segments(adapter, "load", (key,), (0,), (12288,), (8,))
            self.assertEqual(memory.read(0x3000, 8), b"abcdefgh")
            # A replacement remains hidden behind the previous complete block.
            _submit_segments(adapter, "dump", (key,), (0,), (8192,), (4,))
            self.assertEqual(proxy._path(key).read_bytes(), b"abcdefgh")
            adapter.commit(
                True,
                (key, other),
                (key, other),
                np.arange(len((key, other)), dtype=np.int64),
            )
            self.assertEqual(proxy._path(key).read_bytes(), b"abcdefgh")
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_bulk_save_waits_without_publishing_and_handles_terminal_failure(self):
        connector = UCMConnectorWorker.__new__(UCMConnectorWorker)
        connector.select_dump_rows = lambda group_id, indices: np.arange(len(indices))
        connector.use_layerwise = False
        connector._step_save_finalized = False
        connector._failed_load_reqs = set()
        connector._failed_dump_attempts = set()
        connector.spec = SimpleNamespace(fa_groups=(), wa_groups=(), state_groups=())
        connector.has_connector_metadata = lambda: True
        connector._get_connector_metadata = lambda: UCMConnectorMetadata()
        batches = (
            SimpleNamespace(ucm_chunk_ids=(b"a" * 16,), chunk_indices=np.array([0])),
            SimpleNamespace(ucm_chunk_ids=(b"b" * 16,), chunk_indices=np.array([1])),
        )
        connector.layout = SimpleNamespace(spec=connector.spec, group_layouts={})
        connector.transfer_builder = SimpleNamespace(
            build_dump_transfers=lambda _, **kwargs: batches,
        )
        connector._dump_tasks = []
        connector._proxy = mock.Mock()
        connector._proxy.enqueue.side_effect = ["task-a", "task-b"]
        connector.wait_for_save()
        self.assertEqual(
            connector._proxy.mock_calls,
            [
                mock.call.enqueue("dump", batches[0]),
                mock.call.wait("task-a"),
                mock.call.enqueue("dump", batches[1]),
                mock.call.wait("task-b"),
            ],
        )
        connector._proxy.reset_mock()
        connector._step_save_finalized = False
        connector._proxy.enqueue.side_effect = ["task-a", UCMProxyError("write failed")]
        connector.wait_for_save()
        connector._proxy.commit.assert_not_called()

    def test_commit_waits_and_normalizes_failure(self):
        proxy = mock.Mock()
        proxy.wait = mock.Mock()
        proxy.commit.return_value = "commit-task"
        adapter = UCMProxyAdapter(PipelineProxy(proxy))
        adapter.commit(
            True,
            (b"a" * 16,),
            (b"a" * 16,),
            np.arange(len((b"a" * 16,)), dtype=np.int64),
        )
        proxy.wait.assert_called_once_with("commit-task")
        proxy.commit.side_effect = OSError("rename failed")
        with self.assertRaisesRegex(UCMProxyError, "Proxy commit failed"):
            adapter.commit(
                True,
                (b"a" * 16,),
                (b"a" * 16,),
                np.arange(len((b"a" * 16,)), dtype=np.int64),
            )

    def test_normalizes_proxy_error(self):
        class BrokenTestStore(RecordingTestStore):
            def lookup(self, keys):
                raise OSError("backend detail")

        with self.assertRaises(UCMProxyError) as caught:
            UCMProxyAdapter(PipelineProxy(BrokenTestStore())).lookup(
                [b"x" * 16], np.arange(len([b"x" * 16]), dtype=np.int64)
            )
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_file_proxy_prefix_and_reverse_scans(self):
        keys = [bytes([index]) * 16 for index in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            proxy = FileTestStore(directory)
            for key in (keys[0], keys[1], keys[3]):
                (Path(directory) / f"{key.hex()}{proxy._SUFFIX}").write_bytes(b"x")
            adapter = UCMProxyAdapter(PipelineProxy(proxy))

            self.assertEqual(
                adapter.lookup_on_prefix(keys, np.arange(len(keys), dtype=np.int64)), 1
            )
            self.assertEqual(
                adapter.lookup_on_reverse(keys, np.arange(len(keys), dtype=np.int64)), 3
            )
            self.assertEqual(
                adapter.lookup_on_prefix(
                    keys[2:], np.arange(len(keys[2:]), dtype=np.int64)
                ),
                -1,
            )
            self.assertEqual(
                adapter.lookup_on_reverse(
                    (keys[4],), np.arange(len((keys[4],)), dtype=np.int64)
                ),
                -1,
            )

    def test_scan_rejects_out_of_range_results(self):
        class BadScanProxy(PipelineProxy):
            def lookup_on_prefix(self, keys, chunk_indices):
                return len(keys)

        with self.assertRaisesRegex(ValueError, "lookup_on_prefix"):
            UCMProxyAdapter(BadScanProxy(RecordingTestStore())).lookup_on_prefix(
                (b"x" * 16,), np.arange(len((b"x" * 16,)), dtype=np.int64)
            )

    def test_waits_for_async_load_and_dump_tasks(self):
        class AsyncTestStore(RecordingTestStore):
            def __init__(self):
                super().__init__()
                self.wait_calls = []

            def load(self, block_ids, offsets, ptrs, sizes):
                super().load(block_ids, offsets, ptrs, sizes)
                return ("load-task", len(block_ids))

            def dump(
                self, block_ids, offsets, ptrs, sizes, metadata=(), record_nbytes=None
            ):
                super().dump(block_ids, offsets, ptrs, sizes, metadata, record_nbytes)
                return ("dump-task", len(block_ids))

            def wait(self, task):
                self.wait_calls.append(task)

        proxy = AsyncTestStore()
        adapter = UCMProxyAdapter(PipelineProxy(proxy))
        key = b"x" * 16

        _submit_segments(adapter, "load", (key,), (0,), (4096,), (32,))
        _submit_segments(adapter, "dump", (key,), (0,), (8192,), (32,))

        self.assertEqual(
            proxy.wait_calls,
            [("load-task", 1), ("dump-task", 1)],
        )

    def test_rejects_async_task_without_wait_contract(self):
        class IncompleteAsyncTestStore(RecordingTestStore):
            def load(self, block_ids, offsets, ptrs, sizes):
                return object()

        with self.assertRaisesRegex(RuntimeError, "does not provide wait"):
            _submit_segments(
                UCMProxyAdapter(PipelineProxy(IncompleteAsyncTestStore())),
                "load",
                (b"x" * 16,),
                (0,),
                (4096,),
                (32,),
            )

    def test_worker_load_failure_reports_request_and_block_ids_once(self):
        class BrokenLoadTestStore(RecordingTestStore):
            def load(self, block_ids, offsets, ptrs, sizes):
                raise OSError("device copy failed")

        with mock.patch(
            "ucm.integration.vllm.v2.factory.FileTestStore",
            return_value=BrokenLoadTestStore(),
        ):
            connector = UCMConnector(
                vllm_config(),
                KVConnectorRole.WORKER,
                config(group(["model.layers.0.attn"], FullAttentionSpec(4))),
            )
        connector.register_kv_caches(
            {"model.layers.0.attn": FakeTensor(0x1000, (8, 4, 2), (8, 2, 1))}
        )
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (b"x" * 15,),
            plan_token_starts(0, 4, 1),
            (
                np.array(
                    [
                        5,
                    ],
                    dtype=np.uint64,
                ),
            ),
            group_ids=(0,),
        )
        connector.bind_connector_metadata(
            UCMConnectorMetadata(
                requests={"failed": RequestDispatchMeta("failed", (plan,))}
            )
        )

        connector.start_load_kv(None)

        self.assertEqual(
            connector.build_connector_worker_meta().load_failed_reqs, {"failed"}
        )
        self.assertIsNone(connector.build_connector_worker_meta())
        self.assertEqual(connector.get_block_ids_with_load_errors(), {5})
        self.assertEqual(connector.get_block_ids_with_load_errors(), set())

    def test_worker_load_failure_is_isolated_to_its_request(self):
        class SelectiveLoadTestStore(RecordingTestStore):
            def load(self, block_ids, offsets, ptrs, sizes):
                if logical_id(ord("x")) in block_ids:
                    raise OSError("one request failed")
                super().load(block_ids, offsets, ptrs, sizes)

        proxy = SelectiveLoadTestStore()
        with mock.patch(
            "ucm.integration.vllm.v2.factory.FileTestStore",
            return_value=proxy,
        ):
            connector = UCMConnector(
                vllm_config(),
                KVConnectorRole.WORKER,
                config(group(["model.layers.0.attn"], FullAttentionSpec(4))),
            )
        connector.register_kv_caches(
            {"model.layers.0.attn": FakeTensor(0x1000, (8, 4, 2), (8, 2, 1))}
        )
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        def request_meta(request_id, key, block_id):
            plan = UCMGroupDispatchPlan(
                RestorePolicy.FA,
                (key,),
                plan_token_starts(0, 4, 1),
                (
                    np.array(
                        [
                            block_id,
                        ],
                        dtype=np.uint64,
                    ),
                ),
                group_ids=(0,),
            )
            return RequestDispatchMeta(request_id, (plan,))

        connector.bind_connector_metadata(
            UCMConnectorMetadata(
                requests={
                    "failed": request_meta("failed", b"x" * 15, 5),
                    "loaded": request_meta("loaded", b"y" * 15, 6),
                }
            )
        )

        connector.start_load_kv(None)

        self.assertEqual(
            connector.build_connector_worker_meta().load_failed_reqs, {"failed"}
        )
        self.assertEqual(connector.get_block_ids_with_load_errors(), {5})
        self.assertEqual(len(proxy.load_calls), 1)


class ConnectorSchedulerLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
        )
        self.scheduler = make_connector_scheduler(self.parsed)

    @staticmethod
    def scheduler_output(*, tables=None, scheduled=None, preempted=(), finished=()):
        """vLLM 0.30 SchedulerOutput: full per-request block tables each step."""
        tables = tables or {}
        scheduled = scheduled or {}
        return SimpleNamespace(
            kv_connector_block_state=SimpleNamespace(
                req_ids=tuple(tables),
                get_block_ids=tables.__getitem__,
                boundary_state_offloads={},
            ),
            num_scheduled_tokens=scheduled,
            total_num_scheduled_tokens=sum(scheduled.values()),
            preempted_req_ids=set(preempted),
            finished_req_ids=set(finished),
        )

    def add_state(self, request_id="r"):
        request = FakeRequest(request_id, 512)
        keys = tuple(bytes([index]) * 15 for index in range(4))
        return seed_request_state(self.scheduler, request, common_ucm_chunk_ids=keys)

    def test_plans_use_current_engine_block_tables_each_step(self):
        state = self.add_state()
        for index, table in enumerate(([3], [3, 4], [7, 6, 8])):
            metadata = self.scheduler.build_connector_meta(
                self.scheduler_output(tables={"r": (table,)}, scheduled={"r": 128})
            )
            plan = metadata.requests["r"].dump_plans[0]
            self.assertEqual(plan.vllm_block_ids_by_group[0].tolist(), [table[index]])
        self.assertNotIn("vllm_block_tables", vars(state))

    def test_preempt_and_finish_remove_request_snapshots(self):
        self.add_state("preempted")
        self.add_state("finished")

        metadata = self.scheduler.build_connector_meta(
            self.scheduler_output(
                preempted=("preempted",),
                finished=("finished",),
            )
        )

        self.assertEqual(metadata.preempted_req_ids, {"preempted"})
        self.assertEqual(metadata.finished_req_ids, {"finished"})
        self.assertEqual(self.scheduler.requests, {})

    def test_direct_custom_chunk_completes_once_across_steps(self):
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
            chunk_size=512,
        )
        scheduler = make_connector_scheduler(parsed)
        request = FakeRequest("chunked", 1024)
        keys = (b"a" * 15, b"b" * 15)
        seed_request_state(scheduler, request, common_ucm_chunk_ids=keys)
        block_tables = ([1, 2, 3, 4, 5, 6, 7, 8],)

        first = _scheduler_metadata(
            scheduler,
            {"chunked": 128},
            block_tables={"chunked": block_tables},
        )
        second = _scheduler_metadata(
            scheduler,
            {"chunked": 384},
            block_tables={"chunked": block_tables},
        )
        third = _scheduler_metadata(
            scheduler,
            {"chunked": 512},
            block_tables={"chunked": block_tables},
        )

        self.assertEqual(first.requests["chunked"].dump_plans, ())
        second_plan = second.requests["chunked"].dump_plans[0]
        self.assertEqual(
            (
                second_plan.common_ucm_chunk_ids,
                second_plan.token_starts.tolist(),
            ),
            ((keys[0],), [0]),
        )
        self.assertEqual(
            second_plan.vllm_block_ids_by_group[0].tolist(),
            [1, 2, 3, 4],
        )
        third_plan = third.requests["chunked"].dump_plans[0]
        self.assertEqual(
            (
                third_plan.common_ucm_chunk_ids,
                third_plan.token_starts.tolist(),
            ),
            ((keys[1],), [512]),
        )
        self.assertEqual(third_plan.vllm_block_ids_by_group[0].tolist(), [5, 6, 7, 8])

    def test_external_load_plan_is_consumed_after_first_scheduled_step(self):
        state = self.add_state()
        state.external_hit_tokens = 128
        state.token_processed = 128
        state.need_load = True
        block_tables = ([3, 4, 5, 6],)

        first = _scheduler_metadata(
            self.scheduler,
            {"r": 128},
            block_tables={"r": block_tables},
        )
        second = _scheduler_metadata(
            self.scheduler,
            {"r": 128},
            block_tables={"r": block_tables},
        )

        self.assertEqual(len(first.requests["r"].load_plans), 1)
        self.assertEqual(first.requests["r"].load_plans[0].token_starts.tolist(), [0])
        self.assertEqual(second.requests["r"].load_plans, ())

    def test_state_dumps_follow_engine_boundary_offloads(self):
        # The engine hands off exact State checkpoints (boundary_state_offloads);
        # a step straddling a boundary still dumps that boundary's block, and
        # offers off the UCM grid or inside the matched prefix are dropped.
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            scheduler_block_size=128,
        )
        scheduler = make_connector_scheduler(parsed)
        request = FakeRequest("r", 2048)
        fa_keys = tuple(bytes([index]) * 15 for index in range(16))
        seed_request_state(
            scheduler,
            request,
            hbm_hit_tokens=128,
            common_ucm_chunk_ids=fa_keys,
        )
        block_tables = (list(range(16)), list(range(16)))

        def state_plans(scheduled, offloads):
            metadata = _scheduler_metadata(
                scheduler,
                {"r": scheduled},
                state_offloads={"r": offloads},
                block_tables={"r": block_tables},
            )
            return [
                plan
                for plan in metadata.requests["r"].dump_plans
                if plan.restore_policy is RestorePolicy.STATE
            ]

        # [128, 1000): 896 is the newest boundary; 128 is the matched prefix.
        (plan,) = state_plans(872, [(1, 0, 128), (1, 6, 896)])
        self.assertEqual(plan.common_ucm_chunk_ids, (fa_keys[6],))
        self.assertEqual(plan.token_starts.tolist(), [768])
        self.assertEqual(plan.vllm_block_ids_by_group[0].tolist(), [6])

        # [1000, 1100) straddles 1024; a mid-block offer is not a boundary.
        (plan,) = state_plans(100, [(1, 7, 1024), (1, 9, 1000)])
        self.assertEqual(plan.common_ucm_chunk_ids, (fa_keys[7],))
        self.assertEqual(plan.vllm_block_ids_by_group[0].tolist(), [7])

        # [1100, 1124) completes no new boundary.
        self.assertEqual(state_plans(24, []), [])

    def test_dsv4_wa_dispatch_uses_only_final_boundary_and_real_tails(self):
        parsed = parse_spec(
            config(
                group(
                    ["model.layers.2.attn"],
                    MLAAttentionSpec(512, tokens_per_state=4),
                ),
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 128),
                ),
                group(
                    ["model.layers.2.compressor.state_cache"],
                    AscendSlidingWindowMLASpec(8, 1, 8),
                ),
                group(
                    ["model.layers.3.compressor.state_cache"],
                    AscendSlidingWindowMLASpec(32, 1, 128),
                ),
            ),
            scheduler_block_size=512,
            hf_text_config=TextConfig(compress_ratios=[0, 0, 4, 128]),
        )
        scheduler = make_connector_scheduler(parsed)
        request = FakeRequest("dsv4", 1024)
        fa_keys = (b"a" * 15, b"b" * 15)
        seed_request_state(scheduler, request, 1024, fa_keys)
        block_tables = (
            [1, 2],
            list(range(8)),
            list(range(128)),
            list(range(32)),
        )

        metadata = _scheduler_metadata(
            scheduler,
            {"dsv4": 1},
            block_tables={"dsv4": block_tables},
        )
        fa_plan = next(
            plan
            for plan in metadata.requests["dsv4"].load_plans
            if plan.restore_policy is RestorePolicy.FA
        )
        wa_plan = next(
            plan
            for plan in metadata.requests["dsv4"].load_plans
            if plan.restore_policy is RestorePolicy.WA
        )

        self.assertEqual(len(fa_plan.vllm_block_ids_by_group), 1)
        self.assertEqual(wa_plan.common_ucm_chunk_ids, (fa_keys[1],))
        # Windows carry only block ids; the static shapes live in the spec
        # (swa: one whole 128-token block; C4 compressor: the [4, 8)
        # half-block sub-span) and the worker derives them from templates.
        self.assertEqual(
            tuple(blocks.tolist() for blocks in wa_plan.vllm_block_ids_by_group),
            ([7], [127]),
        )

    def test_first_wa_chunk_already_contains_a_full_tail(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(
                    ["model.layers.1.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 512),
                ),
            ),
            scheduler_block_size=128,
            chunk_size=512,
        )
        scheduler = make_connector_scheduler(parsed)
        request = FakeRequest("r", 1024)
        fa_keys = tuple(bytes([index]) * 15 for index in range(2))
        seed_request_state(scheduler, request, common_ucm_chunk_ids=fa_keys)
        block_tables = (
            list(range(8)),
            list(range(8)),
        )

        # Before the first complete chunk neither FA nor WA has a record.
        early = _scheduler_metadata(
            scheduler,
            {"r": 384},
            block_tables={"r": block_tables},
        )
        self.assertFalse(
            any(
                plan.restore_policy is RestorePolicy.FA
                for plan in early.requests["r"].dump_plans
            )
        )
        self.assertFalse(
            any(
                plan.restore_policy is RestorePolicy.WA
                for plan in early.requests["r"].dump_plans
            )
        )

        # Completing the first chunk produces the full WA window [0, 512).
        later = _scheduler_metadata(
            scheduler,
            {"r": 256},
            block_tables={"r": block_tables},
        )
        wa_plan = next(
            plan
            for plan in later.requests["r"].dump_plans
            if plan.restore_policy is RestorePolicy.WA
        )
        self.assertEqual(wa_plan.common_ucm_chunk_ids, (fa_keys[0],))
        self.assertEqual(wa_plan.vllm_block_ids_by_group[0].tolist(), [0, 1, 2, 3])

    def test_wa_window_not_dividing_block_keeps_partial_head(self):
        # ceil, not HMA's floor: a 96-token window over 64-token blocks
        # spans two blocks -- the head block keeps its trailing 32 tokens
        # at a 32-token in-block offset, the newest block stays whole,
        # and the two IO spans are memory-adjacent.
        raw_config = config(
            group(["model.layers.0.attn"], FullAttentionSpec(128)),
            group(
                ["model.layers.1.swa_cache"],
                AscendSlidingWindowMLASpec(64, 1, 96),
            ),
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=128,
        )
        scheduler = make_connector_scheduler(parsed)
        request = FakeRequest("r", 512)
        fa_keys = tuple(bytes([index]) * 15 for index in range(4))
        seed_request_state(scheduler, request, common_ucm_chunk_ids=fa_keys)
        block_tables = (
            list(range(4)),
            list(range(8)),
        )

        metadata = _scheduler_metadata(
            scheduler,
            {"r": 384},
            block_tables={"r": block_tables},
        )
        wa_plan = next(
            plan
            for plan in metadata.requests["r"].dump_plans
            if plan.restore_policy is RestorePolicy.WA
        )
        # Window [288, 384): the head half of block 4 plus all of block 5.
        self.assertEqual(wa_plan.common_ucm_chunk_ids, (fa_keys[2],))
        self.assertEqual(wa_plan.vllm_block_ids_by_group[0].tolist(), [4, 5])

        layout = make_cache_layout(
            parsed,
            {
                "model.layers.0.attn": FakeTensor(
                    0x1000,
                    (8, 128, 1, 512),
                    (65536, 512, 512, 1),
                    element_size=2,
                ),
                "model.layers.1.swa_cache": FakeTensor(
                    0x3000,
                    (8, 64, 1, 512),
                    (32768, 512, 512, 1),
                    element_size=2,
                ),
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        batch = _flat_layout_records(
            layout, "dump", metadata, "model.layers.1.swa_cache"
        )
        self.assertEqual(batch.sizes.tolist(), [32768, 65536])
        self.assertEqual(batch.offsets.tolist(), [0, 32768])
        self.assertEqual(
            batch.ptrs.tolist(),
            [0x3000 + 4 * 65536 + 32768, 0x3000 + 5 * 65536],
        )

    def test_dsv4_fa_windows_encode_n_to_one_keys_flat(self):
        parsed = parse_spec(
            config(
                group(
                    ["model.layers.2.attn"],
                    MLAAttentionSpec(512, tokens_per_state=4),
                ),
                group(
                    ["model.layers.3.attn"],
                    MLAAttentionSpec(16384, tokens_per_state=128),
                ),
            ),
            scheduler_block_size=16384,
            chunk_size=512,
        )
        scheduler = make_connector_scheduler(parsed)
        request = FakeRequest("r", 2048)
        c4_keys = tuple(bytes([index]) * 15 for index in range(4))
        seed_request_state(scheduler, request, common_ucm_chunk_ids=c4_keys)
        block_tables = ([1, 2, 3, 4], [5, 6])

        metadata = _scheduler_metadata(
            scheduler,
            {"r": 2048},
            block_tables={"r": block_tables},
        )
        plan = metadata.requests["r"].dump_plans[0]

        # Group 0 (token block 512 == unit): one whole block per key.
        self.assertEqual(
            plan.vllm_block_ids_by_group[0].tolist(),
            [1, 2, 3, 4],
        )
        # Group 1 (token block 16384): four keys share one page -- the
        # same id four times, flat; each key's 512-token sub-span head is
        # derived on the worker from the plan's token range.
        self.assertEqual(
            plan.vllm_block_ids_by_group[1].tolist(),
            [5, 5, 5, 5],
        )

    def test_large_prefill_plan_stays_flat_and_compact(self):
        # 8192 keys in one step (unit 128 => a 1M-token prefill): the
        # flat encoding keeps the pickled metadata well under a megabyte.
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
        )
        scheduler = make_connector_scheduler(parsed)
        keys = tuple(bytes([index % 256]) * 15 for index in range(8192))
        seed_request_state(
            scheduler,
            FakeRequest("r", 8192 * 128),
            common_ucm_chunk_ids=keys,
        )
        block_tables = (list(range(8192)),)

        metadata = _scheduler_metadata(
            scheduler,
            {"r": 8192 * 128},
            block_tables={"r": block_tables},
        )
        plan = metadata.requests["r"].dump_plans[0]

        self.assertEqual(len(plan.common_ucm_chunk_ids), 8192)
        self.assertEqual(plan.vllm_block_ids_by_group[0].tolist(), list(range(8192)))

        import pickle

        self.assertLess(len(pickle.dumps(metadata)), 1024 * 1024)


class RaggedLayoutTest(unittest.TestCase):
    def test_cpu_attention_derives_storage_axis_from_tokens_per_state(self):
        # vLLM 0.29 dropped spec.storage_block_size; the storage axis is
        # derived as block_size // tokens_per_state (DSV4 C4A: 256/4 = 64).
        raw_config = config(
            group(
                ["model.layers.0.attn"],
                FullAttentionSpec(256, tokens_per_state=4),
            )
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=256,
            device_type="cpu",
        )
        layout = make_cache_layout(
            parsed,
            {
                "model.layers.0.attn": FakeTensor(
                    0x1000,
                    (8, 64, 8),
                    (512, 8, 1),
                )
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )

        component = layout.group_layouts[0]
        self.assertEqual(parsed.groups[0].block_token_span, 256)
        # One stored state per 4 tokens: the block spans 64 states.
        self.assertEqual(
            (
                int(component.states_per_block[0]),
                int(component.payload_bytes[0] // component.states_per_block[0]),
            ),
            (64, 8),
        )

    def test_combined_mamba_raw_page_uses_one_io_region(self):
        raw_config = config(
            group(["model.layers.0.attn"], FullAttentionSpec(4)),
            group(
                ["model.layers.1.mixer"],
                MambaSpec(
                    4,
                    shapes=((2, 2), (2,)),
                    dtypes=(
                        SimpleNamespace(itemsize=2),
                        SimpleNamespace(itemsize=4),
                    ),
                    page_size_bytes=64,
                ),
            ),
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=4,
            device_type="cpu",
        )
        layout = make_cache_layout(
            parsed,
            {
                "model.layers.0.attn": FakeTensor(0x800, (8, 4, 1), (4, 1, 1)),
                "model.layers.1.mixer": FakeTensor(
                    0x1000,
                    (8, 1, 1, 16),
                    (64, 16, 16, 1),
                ),
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )

        # The combined byte page stays a single component: whole-block
        # state IO is a byte copy, so the conv/SSM split the spec
        # describes adds no addressing information.  The view carries
        # the content (16B); the page padding (64-byte stride) stays
        # outside the record.
        group_layout = layout.group_layouts[1]
        self.assertEqual(len(group_layout.layer_names), 1)
        self.assertEqual(
            (
                int(group_layout.base_ptrs[0]),
                int(group_layout.block_strides[0]),
                int(group_layout.payload_bytes[0]),
                int(
                    layout.store_layouts[group_layout.group_id].block_segment_offsets[0]
                ),
            ),
            (0x1000, 64, 16, 0),
        )

        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"m" * 15
        plan = UCMGroupDispatchPlan(
            RestorePolicy.STATE,
            (key,),
            plan_token_starts(0, 4, 1),
            (
                np.array(
                    [
                        3,
                    ],
                    dtype=np.uint64,
                ),
            ),
            group_ids=(1,),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = _flat_layout_records(layout, "load", metadata)
        self.assertEqual(
            batch.ptrs.tolist(),
            [
                0x1000 + 3 * 64,
            ],
        )
        self.assertEqual(
            batch.sizes.tolist(),
            [
                16,
            ],
        )
        self.assertEqual(
            batch.offsets.tolist(),
            [
                0,
            ],
        )

    def test_explicit_components_and_4d_ascend_view_are_supported(self):
        raw_config = config(
            group(
                ["model.layers.0.attn", "model.layers.1.attn"],
                {
                    "model.layers.0.attn": FullAttentionSpec(4),
                    "model.layers.1.attn": FullAttentionSpec(4),
                },
            )
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=4,
        )
        components = (
            FakeTensor(0x1000, (8, 4, 3), (12, 3, 1)),
            FakeTensor(0x2000, (8, 4, 3), (12, 3, 1)),
        )
        ascend_4d = FakeTensor(0x3000, (8, 4, 2, 3), (24, 6, 3, 1))

        layout = make_cache_layout(
            parsed,
            {
                "model.layers.0.attn": components,
                "model.layers.1.attn": ascend_4d,
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )

        self.assertEqual(
            tuple(
                zip(
                    layout.group_layouts[0].layer_names,
                    layout.group_layouts[0].base_ptrs.tolist(),
                )
            ),
            (
                ("model.layers.0.attn", 0x1000),
                ("model.layers.0.attn", 0x2000),
                ("model.layers.1.attn", 0x3000),
            ),
        )
        # One column per component, straight from its view: layer 0's two
        # tensors, then layer 1's single 4-D view (trailing dims permuted,
        # 24-byte payload == row stride).
        group_layout = layout.group_layouts[0]
        self.assertEqual(
            tuple(
                (int(base_ptr), int(payload))
                for base_ptr, payload in zip(
                    group_layout.base_ptrs, group_layout.payload_bytes
                )
            ),
            ((0x1000, 12), (0x2000, 12), (0x3000, 24)),
        )

    def test_unknown_5d_axis_order_fails_fast(self):
        raw_config = config(group(["model.layers.0.attn"], FullAttentionSpec(4)))
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=4,
            device_type="cpu",
        )

        with self.assertRaisesRegex(ValueError, "normalized BHNC view"):
            make_cache_layout(
                parsed,
                {
                    "model.layers.0.attn": FakeTensor(
                        0x1000,
                        (2, 8, 4, 2, 3),
                        (192, 24, 6, 3, 1),
                    )
                },
                kv_cache_config=raw_config,
                num_hidden_layers=4,
            )

    def test_partial_range_keeps_block_wise_entries(self):
        raw_config = config(
            group(
                ["model.layers.0.swa_cache"],
                AscendSlidingWindowMLASpec(128, 1, 192),
            )
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=128,
            chunk_size=256,
        )
        layout = make_cache_layout(
            parsed,
            {
                "model.layers.0.swa_cache": FakeTensor(
                    0x1000,
                    (8, 128, 1),
                    (128, 1, 1),
                )
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )

        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"p" * 15
        # A WA tail shorter than two blocks: the [64, 128) half-block head
        # of block 2 plus the whole block 3.  Only the ids travel; the
        # static tail pattern places the sub-spans.
        plan = UCMGroupDispatchPlan(
            RestorePolicy.WA,
            (key,),
            plan_token_starts(0, 192, 1),
            (np.array([2, 3], dtype=np.uint64),),
            group_ids=(0,),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = _flat_layout_records(layout, "load", metadata)

        # The half-block head and the whole trailing block stay separate
        # entries even though they are contiguous in memory: record
        # positions are a function of the logical enumeration, never of
        # physical adjacency.  Segments pack back to back -- the proxy
        # requires a gapless record.
        self.assertEqual(batch.block_ids, ucm_ids((key, key), RestorePolicy.WA))
        self.assertEqual(batch.ptrs.tolist(), [0x1000 + 2 * 128 + 64, 0x1000 + 3 * 128])
        self.assertEqual(batch.sizes.tolist(), [64, 128])
        self.assertEqual(batch.offsets.tolist(), [0, 64])

    def test_partial_records_roundtrip_through_real_file_proxy(self):
        # Exercise the real Proxy's complete-record contract. A memory stub
        # that silently zero-fills holes can hide incorrect file offsets.
        raw_config = config(
            group(
                ["model.layers.0.swa_cache"],
                AscendSlidingWindowMLASpec(128, 1, 192),
            )
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=128,
            chunk_size=256,
        )
        layout = make_cache_layout(
            parsed,
            {"model.layers.0.swa_cache": FakeTensor(0x1000, (8, 128, 1), (128, 1, 1))},
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"p" * 15

        def batch(phase, block_ids):
            plan = UCMGroupDispatchPlan(
                RestorePolicy.WA,
                (key,),
                plan_token_starts(0, 192, 1),
                (np.array(block_ids, dtype=np.uint64),),
                group_ids=(0,),
            )
            metadata = UCMConnectorMetadata(
                requests={
                    "r": RequestDispatchMeta("r", **{phase + "_plans": (plan,)}),
                }
            )
            return _flat_layout_records(layout, phase, metadata)

        memory = ByteMemory()
        memory.write(0x1000, b"\xef" * 1024)
        memory.write(0x1000 + 2 * 128, bytes(range(128)))
        memory.write(0x1000 + 3 * 128, bytes(range(128, 256)))
        # The tail record: block 2's [64, 128) half-block head plus the
        # whole block 3.
        expected = bytes(range(64, 256))
        dumped = batch("dump", (2, 3))
        loaded = batch("load", (7, 5))
        with tempfile.TemporaryDirectory() as directory:
            proxy = FileTestStore(directory, MemoryByteAccess(memory))
            proxy.dump(dumped.block_ids, dumped.offsets, dumped.ptrs, dumped.sizes)
            proxy.commit(True, dumped.block_ids, dumped.block_ids)
            self.assertEqual(
                (
                    Path(directory)
                    / (ucm_ids((key,), RestorePolicy.WA)[0].hex() + ".ucm")
                ).read_bytes(),
                expected,
            )
            proxy.load(loaded.block_ids, loaded.offsets, loaded.ptrs, loaded.sizes)
        self.assertEqual(
            memory.read(0x1000 + 7 * 128, 128), b"\xef" * 64 + expected[:64]
        )
        self.assertEqual(memory.read(0x1000 + 5 * 128, 128), expected[64:])

    def test_dump_and_load_survive_different_physical_block_layouts(self):
        # Dump and load address different vLLM blocks (the source request is
        # freed before the target allocates), so their physical adjacency
        # differs.  Record positions must stay a function of the logical
        # dispatch plan alone: entry merging may only join ranges that are
        # contiguous in both the record and memory, never reorder or drop
        # them based on physical addresses.
        raw_config = config(group(["model.layers.0.attn"], FullAttentionSpec(8)))
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=8,
        )
        layout = make_cache_layout(
            parsed,
            {
                "model.layers.0.attn": FakeTensor(
                    0x1000,
                    (8, 8, 6),
                    (48, 6, 1),
                )
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        memory = ByteMemory()
        source_first = bytes(range(48))
        source_second = bytes(value + 100 for value in range(48))
        memory.write(0x1000 + 3 * 48, source_first)
        memory.write(0x1000 + 4 * 48, source_second)

        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )
        from ucm.integration.vllm.v2.ucm_proxy import UCMProxyAdapter

        key_a, key_b = b"d" * 15, b"e" * 15
        adapter = UCMProxyAdapter(PipelineProxy(MemoryFixtureStore(memory)))
        dump_plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (key_a, key_b),
            plan_token_starts(0, 16, 2),
            (np.array([3, 4], dtype=np.uint64),),
            group_ids=(0,),
        )
        dump_meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", dump_plans=(dump_plan,))}
        )
        dump_batch = _flat_layout_records(layout, "dump", dump_meta)
        # Blocks 3 and 4 are enumerated separately even though adjacent.
        self.assertEqual(dump_batch.sizes.tolist(), [48, 48])

        # The load lands on scattered blocks; each key's record keeps its
        # own byte position.
        load_plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (key_a, key_b),
            plan_token_starts(0, 16, 2),
            (np.array([6, 2], dtype=np.uint64),),
            group_ids=(0,),
        )
        load_meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(load_plan,))}
        )
        load_batch = _flat_layout_records(layout, "load", load_meta)
        self.assertEqual(load_batch.sizes.tolist(), [48, 48])
        self.assertEqual(load_batch.offsets.tolist(), [0, 0])
        self.assertEqual(load_batch.ptrs.tolist(), [0x1000 + 6 * 48, 0x1000 + 2 * 48])

        _submit_segments(
            adapter,
            "dump",
            dump_batch.block_ids,
            dump_batch.offsets,
            dump_batch.ptrs,
            dump_batch.sizes,
        )
        ucm_chunk_ids = tuple(dict.fromkeys(dump_batch.block_ids))
        adapter.commit(
            True,
            ucm_chunk_ids,
            ucm_chunk_ids,
            np.arange(len(ucm_chunk_ids), dtype=np.int64),
        )
        _submit_segments(
            adapter,
            "load",
            load_batch.block_ids,
            load_batch.offsets,
            load_batch.ptrs,
            load_batch.sizes,
        )
        self.assertEqual(memory.read(0x1000 + 6 * 48, 48), source_first)
        self.assertEqual(memory.read(0x1000 + 2 * 48, 48), source_second)

    def test_kimi_mla_six_kernel_rows_coalesce_per_component(self):
        raw_config = config(
            group(["model.layers.3.self_attn.attn"], FullAttentionSpec(768)),
            num_blocks=2,
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=768,
        )
        caches = {
            "model.layers.3.self_attn.attn": (
                FakeTensor(
                    0x1000,
                    (12, 128, 1, 512),
                    (65536, 512, 512, 1),
                    element_size=2,
                ),
                FakeTensor(
                    0x200000,
                    (12, 128, 1, 64),
                    (8192, 64, 64, 1),
                    element_size=2,
                ),
            )
        }
        layout = make_cache_layout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        key = b"k" * 15
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (key,),
            plan_token_starts(0, 768, 1),
            (
                np.array(
                    [
                        1,
                    ],
                    dtype=np.uint64,
                ),
            ),
            group_ids=(0,),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )

        batch = _flat_layout_records(layout, "load", metadata)

        self.assertEqual(batch.sizes.tolist(), [786432, 98304])
        self.assertEqual(batch.offsets.tolist(), [0, 786432])
        self.assertEqual(batch.ptrs.tolist(), [0x1000 + 786432, 0x200000 + 98304])

    def test_kimi_state_components_copy_payload_not_shared_page_stride(self):
        raw_config = config(
            group(["model.layers.3.attn"], FullAttentionSpec(768)),
            group(
                ["model.layers.0.self_attn"],
                MambaSpec(
                    768,
                    shapes=((3, 4608), (12, 128, 128)),
                ),
            ),
            num_blocks=2,
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=768,
        )
        import torch

        num_blocks = 2
        conv_bytes = 27648
        ssm_bytes = 786432
        backing = torch.zeros(num_blocks * (conv_bytes + ssm_bytes), dtype=torch.uint8)
        root = backing.data_ptr()
        caches = {
            "model.layers.3.attn": FakeTensor(
                0x1000,
                (12, 128, 1, 576),
                (73728, 576, 576, 1),
                element_size=2,
            ),
            "model.layers.0.self_attn": (
                torch.as_strided(
                    backing.view(torch.bfloat16),
                    (2, 3, 4608),
                    (conv_bytes // 2, 4608, 1),
                ),
                torch.as_strided(
                    backing.view(torch.float32),
                    (2, 12, 128, 128),
                    (ssm_bytes // 4, 16384, 128, 1),
                    num_blocks * conv_bytes // 4,
                ),
            ),
        }
        layout = make_cache_layout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        key = b"s" * 15
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            RestorePolicy.STATE,
            (key,),
            plan_token_starts(1536, 2304, 1),
            (
                np.array(
                    [
                        1,
                    ],
                    dtype=np.uint64,
                ),
            ),
            group_ids=(1,),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )

        batch = _flat_layout_records(layout, "load", metadata)

        self.assertEqual(
            batch.ptrs.tolist(),
            [
                root + conv_bytes,
                root + num_blocks * conv_bytes + ssm_bytes,
            ],
        )
        self.assertEqual(batch.sizes.tolist(), [conv_bytes, ssm_bytes])
        self.assertEqual(batch.offsets.tolist(), [0, conv_bytes])

    def test_component_major_layout_spans_multiple_physical_blocks(self):
        raw_config = config(
            group(
                ["model.layers.0.attn", "model.layers.1.attn"],
                {
                    "model.layers.0.attn": FullAttentionSpec(128),
                    "model.layers.1.attn": FullAttentionSpec(128),
                },
            )
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=128,
            chunk_size=256,
        )
        caches = {
            "model.layers.0.attn": (
                FakeTensor(0x1000, (8, 128, 4), (512, 4, 1)),
                FakeTensor(0x3000, (8, 128, 2), (256, 2, 1)),
            ),
            "model.layers.1.attn": (FakeTensor(0x5000, (8, 128, 3), (384, 3, 1)),),
        }
        layout = make_cache_layout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        proxy = RecordingTestStore()
        coordinator = make_connector_scheduler(
            parsed, proxy, RequestHasher("component-test"), recompute_tokens=0
        )
        request = FakeRequest("r", 256)
        scheduler = coordinator
        scheduler.lookup(request, 0)
        state = scheduler.requests["r"]
        block_tables = ([2, 5],)
        state.external_hit_tokens = 256
        state.need_load = True
        metadata = _scheduler_metadata(
            scheduler,
            {"r": 1},
            block_tables={"r": block_tables},
        )

        batch = _flat_layout_records(layout, "load", metadata)

        # Records are block-major: block 0's views, then block 1's -- each
        # block's layer pages land at the per-block anchor (K/V of layer 0
        # then layer 1), so one ucm chunk is the byte-concatenation of its
        # two vllm blocks.
        self.assertEqual(batch.offsets.tolist(), [0, 512, 768, 1152, 1664, 1920])
        self.assertEqual(batch.sizes.tolist(), [512, 256, 384, 512, 256, 384])
        self.assertEqual(
            batch.ptrs.tolist(), [0x1400, 0x3200, 0x5300, 0x1A00, 0x3500, 0x5780]
        )
        # A layerwise (filtered) batch must address the very same record
        # coordinates: a layerwise load lands exactly where the full record
        # was dumped.
        layer_batch = _flat_layout_records(
            layout, "load", metadata, "model.layers.1.attn"
        )
        self.assertEqual(layer_batch.offsets.tolist(), [768, 1920])
        self.assertEqual(layer_batch.sizes.tolist(), [384, 384])
        self.assertEqual(layer_batch.ptrs.tolist(), [0x5300, 0x5780])
        full_positions = {
            (ptr, size): offset
            for ptr, size, offset in zip(batch.ptrs, batch.sizes, batch.offsets)
        }
        for ptr, size, offset in zip(
            layer_batch.ptrs, layer_batch.sizes, layer_batch.offsets
        ):
            self.assertEqual(full_positions[(ptr, size)], offset)

    def test_interleaved_declarations_take_one_span_per_block(self):
        # Block First (DSV4 0.29): one descriptor's layer pages tile each
        # block slot (layer_stride < block_stride).  Whole batches collapse
        # to one descriptor-sized span per block -- paddings riding inside
        # -- while layered and sub-block queries stay per-view exact and
        # address the very same record coordinates.
        num_blocks = 4
        page = 4096  # 64 states x 64B, dense
        layer_stride = page
        block_stride = 2 * page  # both layers' pages tile one slot
        declarations = (
            SimpleNamespace(
                size=num_blocks * block_stride,
                layers=("model.layers.0.attn", "model.layers.1.attn"),
                offset=0,
                layer_stride=layer_stride,
                block_stride=block_stride,
            ),
        )
        raw_config = config(
            group(
                ["model.layers.0.attn", "model.layers.1.attn"],
                {
                    "model.layers.0.attn": FullAttentionSpec(256, 4),
                    "model.layers.1.attn": FullAttentionSpec(256, 4),
                },
            ),
            num_blocks=4,
            tensors=declarations,
            kv_cache_layout="BLHNC",
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=256,
            device_type="cpu",
        )
        caches = {
            "model.layers.0.attn": FakeTensor(
                0x1000, (num_blocks, 64, 64), (block_stride, 64, 1)
            ),
            "model.layers.1.attn": FakeTensor(
                0x1000 + page, (num_blocks, 64, 64), (block_stride, 64, 1)
            ),
        }
        layout = make_cache_layout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        group_layout = layout.group_layouts[0]
        span = group_layout.block_first
        self.assertIsNotNone(span)
        self.assertEqual(
            (span.base_ptr, span.block_stride, span.block_size_bytes),
            (0x1000, block_stride, 2 * layer_stride),
        )
        # HBM can include padding; the file stores compact component payloads.
        self.assertEqual(
            layout.store_layouts[group_layout.group_id].whole_block_nbytes,
            2 * page,
        )

        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"b" * 15
        plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (key,),
            plan_token_starts(0, 256, 1),
            (
                np.array(
                    [
                        2,
                    ],
                    dtype=np.uint64,
                ),
            ),
            group_ids=(0,),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = _flat_layout_records(layout, "load", metadata)
        # Whole batch: one span covering both layers' pages of block 2.
        self.assertEqual(
            batch.ptrs.tolist(),
            [
                0x1000 + 2 * block_stride,
                0x1000 + page + 2 * block_stride,
            ],
        )
        self.assertEqual(
            batch.sizes.tolist(),
            [
                page,
                page,
            ],
        )
        self.assertEqual(
            batch.offsets.tolist(),
            [0, page],
        )

        # Layered batch: layer 1's page only, at its in-slot record offset
        # (one page in) -- the same bytes the whole-batch span covered.
        layer_batch = _flat_layout_records(
            layout, "load", metadata, "model.layers.1.attn"
        )
        self.assertEqual(
            layer_batch.ptrs.tolist(),
            [
                0x1000 + page + 2 * block_stride,
            ],
        )
        self.assertEqual(
            layer_batch.sizes.tolist(),
            [
                page,
            ],
        )
        self.assertEqual(
            layer_batch.offsets.tolist(),
            [
                page,
            ],
        )

        # Whole spans and compiled access templates expose the same bytes:
        # the span covers the group, while the grid has one column per segment.
        span_ptrs = group_layout.resolve_block_first_ptrs((2,))
        self.assertEqual(
            (span_ptrs.tolist(), group_layout.block_first.block_size_bytes),
            ([0x1000 + 2 * block_stride], 2 * page),
        )
        grid_ptrs, grid_sizes = _resolved_access(
            compile_access(group_layout, token_counts=256), [2]
        )
        self.assertEqual(grid_ptrs[:, 1].tolist(), [0x1000 + page + 2 * block_stride])
        self.assertEqual(grid_sizes[:, 1].tolist(), [page])

        # Sub-block token windows never use the descriptor span: they walk
        # the layer's own view and stay payload-exact (128 tokens at the
        # 4:1 compression = 32 states x 64B = 2048B).
        window_ptrs, window_sizes = _resolved_access(
            compile_access(group_layout, token_counts=128),
            [2],
        )
        self.assertEqual(window_ptrs[:, 0].tolist(), [0x1000 + 2 * block_stride])
        self.assertEqual(window_sizes[:, 0].tolist(), [2048])

    def test_tiled_group_spans_join_two_descriptors(self):
        # The DSV4 shape: one group, several descriptors tiling the block
        # slot back to back, with their layers interleaved in layer order.
        # The whole group is one span per block; each layer's record
        # anchor is its page slot inside its own descriptor's tile --
        # not the layer-order running sum (which would interleave two
        # tiles' strides and land layered queries at wrong offsets).
        num_blocks = 4
        slot = 12288
        declarations = (
            SimpleNamespace(
                layers=("model.layers.0.attn", "model.layers.1.attn"),
                offset=0,
                layer_stride=4096,
                block_stride=slot,
            ),
            SimpleNamespace(
                layers=("model.layers.0.indexer.k", "model.layers.1.indexer.k"),
                offset=2 * 4096,  # right after the first tile
                layer_stride=2048,
                block_stride=slot,
            ),
        )
        raw_config = config(
            group(
                [
                    "model.layers.0.attn",
                    "model.layers.1.attn",
                    "model.layers.0.indexer.k",
                    "model.layers.1.indexer.k",
                ],
                {
                    "model.layers.0.attn": FullAttentionSpec(256, 4),
                    "model.layers.1.attn": FullAttentionSpec(256, 4),
                    "model.layers.0.indexer.k": FullAttentionSpec(256, 8),
                    "model.layers.1.indexer.k": FullAttentionSpec(256, 8),
                },
            ),
            num_blocks=num_blocks,
            tensors=declarations,
            kv_cache_layout="BLHNC",
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=256,
            device_type="cpu",
        )
        caches = {
            "model.layers.0.attn": FakeTensor(
                0x1000, (num_blocks, 64, 64), (slot, 64, 1)
            ),
            "model.layers.1.attn": FakeTensor(
                0x1000 + 4096, (num_blocks, 64, 64), (slot, 64, 1)
            ),
            "model.layers.0.indexer.k": FakeTensor(
                0x1000 + 8192, (num_blocks, 32, 64), (slot, 64, 1)
            ),
            "model.layers.1.indexer.k": FakeTensor(
                0x1000 + 8192 + 2048, (num_blocks, 32, 64), (slot, 64, 1)
            ),
        }
        layout = make_cache_layout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        group_layout = layout.group_layouts[0]

        # One group-level span: both tiles joined, slot-sized.
        span = group_layout.block_first
        self.assertIsNotNone(span)
        self.assertEqual(
            (span.base_ptr, span.block_stride, span.block_size_bytes),
            (0x1000, slot, slot),
        )
        self.assertEqual(
            layout.store_layouts[group_layout.group_id].whole_block_nbytes, slot
        )

        # Each segment keeps its offset inside the span for layerwise IO.
        self.assertEqual(
            layout.store_layouts[group_layout.group_id].block_segment_offsets.tolist(),
            [0, 4096, 6144, 10240],
        )
        from ucm.integration.vllm.v2.layout.store_layout import GroupStoreLayout
        from ucm.integration.vllm.v2.runtime import ParallelRank, ParallelTopology

        layerwise_store = GroupStoreLayout.build(
            parsed,
            group_layout.group_id,
            group_layout,
            ((0, layout.export_layout_metadata(1), None),),
            ParallelTopology(),
            ParallelRank(),
        )
        self.assertEqual(
            layerwise_store.whole_block_nbytes, sum(group_layout.payload_bytes)
        )
        self.assertEqual(
            layerwise_store.block_segment_offsets.tolist(), [0, 4096, 6144, 10240]
        )
        # Layerwise positions come from the physical view, not storage slots.
        offsets = dict(
            zip(
                group_layout.layer_names,
                (group_layout.base_ptrs - np.uint64(span.base_ptr)).tolist(),
            )
        )
        self.assertEqual(
            offsets,
            {
                "model.layers.0.attn": 0,
                "model.layers.0.indexer.k": 8192,
                "model.layers.1.attn": 4096,
                "model.layers.1.indexer.k": 10240,
            },
        )

        # A whole batch is one span per block; the layered batch for
        # layer 1's attention lands at its tile-slot record offset.
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"c" * 15
        plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (key,),
            plan_token_starts(0, 256, 1),
            (
                np.array(
                    [
                        3,
                    ],
                    dtype=np.uint64,
                ),
            ),
            group_ids=(0,),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = _flat_layout_records(layout, "load", metadata)
        self.assertEqual(
            batch.ptrs.tolist(),
            [
                0x1000 + 3 * slot,
                0x1000 + 8192 + 3 * slot,
                0x1000 + 4096 + 3 * slot,
                0x1000 + 10240 + 3 * slot,
            ],
        )
        self.assertEqual(
            batch.sizes.tolist(),
            [
                4096,
                2048,
                4096,
                2048,
            ],
        )
        layer_batch = _flat_layout_records(
            layout, "load", metadata, "model.layers.1.attn"
        )
        self.assertEqual(
            layer_batch.ptrs.tolist(),
            [
                0x1000 + 4096 + 3 * slot,
            ],
        )
        self.assertEqual(
            layer_batch.sizes.tolist(),
            [
                4096,
            ],
        )
        self.assertEqual(
            layer_batch.offsets.tolist(),
            [
                6144,
            ],
        )

    def test_layer_contiguous_views_compile_to_one_entry_per_layer(self):
        # GLM-style placement: layer 1 sits after *all* of layer 0's
        # blocks.  Each layer keeps its own whole-block entry read
        # straight off its view; nothing merges by probing pointers
        # (0.26's per-view allocations and layer-contiguous layouts are
        # never adjacent inside a block by design).
        raw_config = config(
            group(
                ["model.layers.0.attn", "model.layers.1.attn"],
                {
                    "model.layers.0.attn": FullAttentionSpec(4),
                    "model.layers.1.attn": FullAttentionSpec(4),
                },
            )
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=4,
            chunk_size=8,
        )
        layout = make_cache_layout(
            parsed,
            {
                "model.layers.0.attn": FakeTensor(0x1000, (8, 4, 3), (12, 3, 1)),
                "model.layers.1.attn": FakeTensor(
                    0x1000 + 8 * 12, (8, 4, 3), (12, 3, 1)
                ),
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        group_layout = layout.group_layouts[0]
        self.assertEqual(
            tuple(
                (int(base_ptr), int(stride), int(payload))
                for base_ptr, stride, payload in zip(
                    group_layout.base_ptrs,
                    group_layout.block_strides,
                    group_layout.payload_bytes,
                )
            ),
            ((0x1000, 12, 12), (0x1000 + 8 * 12, 12, 12)),
        )
        self.assertEqual(
            layout.store_layouts[group_layout.group_id].whole_block_nbytes, 24
        )
        self.assertEqual(
            group_layout.layer_names,
            ("model.layers.0.attn", "model.layers.1.attn"),
        )
        self.assertEqual(
            layout.store_layouts[group_layout.group_id].block_segment_offsets.tolist(),
            [0, 12],
        )

        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"r" * 15
        plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (key,),
            plan_token_starts(0, 8, 1),
            (np.array([2, 5], dtype=np.uint64),),
            group_ids=(0,),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = _flat_layout_records(layout, "load", metadata)
        # Block-major record: block 0's views, then block 1's -- adjacent
        # physical blocks never merge into one entry; cross-block
        # adjacency is an allocation accident, not a record fact.
        self.assertEqual(batch.sizes.tolist(), [12, 12, 12, 12])
        self.assertEqual(batch.offsets.tolist(), [0, 12, 24, 36])
        self.assertEqual(
            batch.ptrs.tolist(),
            [
                0x1000 + 2 * 12,
                0x1000 + 8 * 12 + 2 * 12,
                0x1000 + 5 * 12,
                0x1000 + 8 * 12 + 5 * 12,
            ],
        )

        # The compiled access grid answers the same spans without
        # records: rows are blocks, columns views.
        grid_ptrs, grid_sizes = _resolved_access(compile_access(group_layout), [2, 5])
        self.assertEqual(
            grid_ptrs.tolist(),
            [
                [0x1000 + 2 * 12, 0x1000 + 8 * 12 + 2 * 12],
                [0x1000 + 5 * 12, 0x1000 + 8 * 12 + 5 * 12],
            ],
        )
        self.assertEqual(grid_sizes.tolist(), [[12, 12], [12, 12]])
        mask = group_layout.segment_mask(layer_names=("model.layers.1.attn",))
        self.assertEqual(
            grid_ptrs[:, mask].tolist(),
            [[0x1000 + 8 * 12 + 2 * 12], [0x1000 + 8 * 12 + 5 * 12]],
        )

    def test_layerwise_batches_of_every_layer_tile_the_full_record(self):
        # Whatever the layout, unioning each layer's filtered batch must
        # reproduce the full batch exactly -- same entries, same record
        # coordinates.  This is the layerwise-consumer contract.
        raw_config = config(
            group(
                ["model.layers.0.attn", "model.layers.1.attn"],
                {
                    "model.layers.0.attn": FullAttentionSpec(128),
                    "model.layers.1.attn": FullAttentionSpec(128),
                },
            )
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=128,
            chunk_size=256,
        )
        caches = {
            "model.layers.0.attn": (
                FakeTensor(0x1000, (8, 128, 4), (512, 4, 1)),
                FakeTensor(0x3000, (8, 128, 2), (256, 2, 1)),
            ),
            "model.layers.1.attn": (FakeTensor(0x5000, (8, 128, 3), (384, 3, 1)),),
        }
        layout = make_cache_layout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"t" * 15
        plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (key,),
            plan_token_starts(0, 256, 1),
            (np.array([2, 5], dtype=np.uint64),),
            group_ids=(0,),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        full = _flat_layout_records(layout, "load", metadata)
        union: dict[tuple[int, int, int], None] = {}
        for layer_name in ("model.layers.0.attn", "model.layers.1.attn"):
            layer_batch = _flat_layout_records(layout, "load", metadata, layer_name)
            self.assertTrue(layer_batch.block_ids)
            for ptr, size, offset in zip(
                layer_batch.ptrs, layer_batch.sizes, layer_batch.offsets
            ):
                self.assertNotIn((ptr, size, offset), union)
                union[(ptr, size, offset)] = None
        self.assertEqual(
            sorted(union),
            sorted(
                (ptr, size, offset)
                for ptr, size, offset in zip(full.ptrs, full.sizes, full.offsets)
            ),
        )

    def test_glm_indexer_k_and_scale_keep_separate_runtime_addressing(self):
        raw_config = config(
            group(["model.layers.0.indexer"], FullAttentionSpec(128)), num_blocks=4
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=128,
        )
        num_blocks = 4
        k_base = 0x1000
        scale_base = k_base + num_blocks * 16384
        layout = make_cache_layout(
            parsed,
            {
                "model.layers.0.indexer": (
                    FakeTensor(k_base, (4, 128, 1, 128), (16384, 128, 128, 1)),
                    FakeTensor(
                        scale_base,
                        (4, 128, 1, 1),
                        (128, 1, 1, 1),
                        element_size=2,
                    ),
                )
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        key = b"g" * 15
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (key,),
            plan_token_starts(0, 128, 1),
            (
                np.array(
                    [
                        2,
                    ],
                    dtype=np.uint64,
                ),
            ),
            group_ids=(0,),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )

        batch = _flat_layout_records(layout, "load", metadata)

        self.assertEqual(
            batch.ptrs.tolist(), [k_base + 2 * 16384, scale_base + 2 * 256]
        )
        self.assertEqual(batch.sizes.tolist(), [16384, 256])
        self.assertEqual(batch.offsets.tolist(), [0, 16384])

    def test_model_layer_selection_spans_cache_names_and_groups(self):
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        packed_names = [
            "model.layers.1.attn",
            "model.layers.2.attn",
            "model.layers.2.indexer",
            "model.layers.3.attn",
        ]
        aux_names = ["model.layers.1.aux", "model.layers.2.aux"]
        raw_config = config(
            group(packed_names, FullAttentionSpec(4)),
            group(aux_names, FullAttentionSpec(4)),
            tensors=(
                SimpleNamespace(
                    layers=packed_names,
                    offset=0,
                    layer_stride=8,
                    block_stride=64,
                ),
            ),
            kv_cache_layout="BLHNC",
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=4,
            device_type="cpu",
        )
        caches = {
            name: FakeTensor(0x1000 + index * 8, (8, 4, 1), (64, 1, 1))
            for index, name in enumerate(packed_names)
        }
        caches.update(
            {
                name: FakeTensor(0x4000 + index * 0x1000, (8, 4, 1), (4, 1, 1))
                for index, name in enumerate(aux_names)
            }
        )
        layout = make_cache_layout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        layer_id = layout.layer_id_by_name["model.layers.2.attn"]
        self.assertEqual(layer_id, 2)
        self.assertEqual(
            layout.layer_names_by_id[layer_id],
            frozenset(
                (
                    "model.layers.2.attn",
                    "model.layers.2.indexer",
                    "model.layers.2.aux",
                )
            ),
        )

        def metadata(block0, block1):
            # One whole-block key over both groups: the packed declared
            # group answers at its slot anchors, the aux layer-first
            # group at its payload slots.
            plan = UCMGroupDispatchPlan(
                RestorePolicy.FA,
                (b"record_________",),
                plan_token_starts(0, 4, 1),
                (
                    np.array([block0], dtype=np.uint64),
                    np.array([block1], dtype=np.uint64),
                ),
                group_ids=(0, 1),
            )
            return UCMConnectorMetadata(
                requests={
                    "r": RequestDispatchMeta(
                        "r", load_plans=(plan,), dump_plans=(plan,)
                    ),
                }
            )

        # Whole Block First IO includes padding; sub-span IO (FA sharing a
        # block, WA tails) is covered by its own tests.  Model-layer
        # selection must preserve the record coordinates either way.
        offsets = (4, 8, 20)
        with (
            self.subTest(window=(0, 4)),
            tempfile.TemporaryDirectory() as path,
        ):
            source_meta = metadata(1, 2)
            target_meta = metadata(3, 4)
            source = _flat_layout_records(layout, "dump", source_meta)
            target = _flat_layout_records(layout, "load", target_meta)
            source_layer = _flat_layout_records(
                layout, "dump", source_meta, layer_id=layer_id
            )
            target_layer = _flat_layout_records(
                layout, "load", target_meta, layer_id=layer_id
            )
            self.assertEqual(source_layer.offsets.tolist(), list(offsets))
            self.assertEqual(target_layer.offsets.tolist(), list(offsets))
            exact = _flat_layout_records(
                layout, "load", target_meta, "model.layers.2.attn"
            )
            self.assertEqual(exact.offsets.tolist(), [offsets[0]])
            window = _resolved_access(
                compile_access(layout.group_layouts[0], token_counts=4),
                [3],
                segment_mask=layout.group_layouts[0].segment_mask(layer_ids=[2]),
            )
            self.assertEqual(
                tuple(
                    zip(window[0].reshape(-1).tolist(), window[1].reshape(-1).tolist())
                ),
                tuple(zip(target_layer.ptrs[:2], target_layer.sizes[:2])),
            )
            with self.assertRaisesRegex(ValueError, "either"):
                _flat_layout_records(
                    layout, "load", target_meta, "model.layers.2.attn", layer_id=2
                )
            with self.assertRaises(KeyError):
                _flat_layout_records(layout, "load", target_meta, layer_id=99)

            memory = ByteMemory()
            for ptr, size in zip(target.ptrs, target.sizes):
                memory.write(ptr, b"\xef" * size)
            for offset, ptr, size in zip(source.offsets, source.ptrs, source.sizes):
                memory.write(ptr, bytes((int(offset) + i) % 251 for i in range(size)))
            proxy = FileTestStore(Path(path), byte_access=MemoryByteAccess(memory))
            proxy.dump(source.block_ids, source.offsets, source.ptrs, source.sizes)
            proxy.commit(True, source.block_ids, source.block_ids)
            proxy.load(
                target_layer.block_ids,
                target_layer.offsets,
                target_layer.ptrs,
                target_layer.sizes,
            )
            for offset, ptr, size in zip(target.offsets, target.ptrs, target.sizes):
                expected = bytearray(b"\xef" * size)
                for selected_offset, selected_size in zip(
                    target_layer.offsets, target_layer.sizes
                ):
                    if offset <= selected_offset < offset + size:
                        begin = selected_offset - offset
                        expected[begin : begin + selected_size] = bytes(
                            (int(selected_offset) + i) % 251
                            for i in range(selected_size)
                        )
                self.assertEqual(memory.read(ptr, size), bytes(expected))

    def test_dsv4_canonical_subrange_selects_intersecting_large_page(self):
        raw_config = config(
            group(
                ["model.layers.2.attn"],
                MLAAttentionSpec(512, tokens_per_state=4),
            ),
            group(
                ["model.layers.3.attn"],
                MLAAttentionSpec(16384, tokens_per_state=128),
            ),
            group(
                ["model.layers.0.swa_cache"],
                AscendSlidingWindowMLASpec(128, 1, 128),
            ),
            num_blocks=4,
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=16384,
            chunk_size=512,
        )
        caches = {
            "model.layers.2.attn": FakeTensor(
                0x1000,
                (4, 128, 1, 512),
                (65536, 512, 512, 1),
                element_size=2,
            ),
            # One physical page covers 16384 source tokens; a 512-token key
            # maps to four physical tokens inside that page.
            "model.layers.3.attn": FakeTensor(
                0x5000,
                (4, 128, 1, 512),
                (65536, 512, 512, 1),
                element_size=2,
            ),
            "model.layers.0.swa_cache": FakeTensor(
                0x3000,
                (4, 128, 1, 512),
                (65536, 512, 512, 1),
                element_size=2,
            ),
        }
        layout = make_cache_layout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        key = b"k" * 15
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (key,),
            plan_token_starts(512, 1024, 1),
            # One window per FA group in route order: group 0 is 1:1
            # (one whole block); group 1 is N:1 -- the key is one
            # 512-token sub-span inside a 16384-token page.
            (np.array([1], dtype=np.uint64), np.array([2], dtype=np.uint64)),
            group_ids=(0, 1),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )

        batch = _flat_layout_records(layout, "load", metadata)

        # Group 0's whole block, then group 1's sub-span: 512-token page
        # window selected by token start inside the large page.
        self.assertEqual(batch.sizes[-1], 4096)
        self.assertEqual(batch.ptrs[-1], 0x5000 + 2 * 131072 + 4 * 1024)

    def test_hybrid_state_loads_complete_checkpoint_into_both_slots(self):
        raw_config = config(
            group(["model.layers.0.attn"], FullAttentionSpec(128)),
            group(["model.layers.1.mixer"], MambaSpec(128)),
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=128,
        )
        import torch

        state_page = torch.zeros((8, 64, 4), dtype=torch.uint8)
        caches = {
            "model.layers.0.attn": FakeTensor(0x1000, (8, 128, 4), (512, 4, 1)),
            "model.layers.1.mixer": state_page,
        }
        layout = make_cache_layout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        scheduler = make_connector_scheduler(parsed)
        request = FakeRequest("r", 640)
        common_ids = tuple(bytes([index]) * 15 for index in range(5))
        seed_request_state(scheduler, request, 512, common_ids)
        block_tables = ([1, 2, 3, 4, 5], [0, 1, 2, 6, 7])

        metadata = _scheduler_metadata(
            scheduler,
            {"r": 1},
            block_tables={"r": block_tables},
        )
        batch = _flat_layout_records(layout, "load", metadata)

        state_segments = [
            (ptr, size)
            for key, ptr, size in zip(batch.block_ids, batch.ptrs, batch.sizes)
            if key in ucm_ids(common_ids, RestorePolicy.STATE)
        ]
        self.assertEqual(
            state_segments,
            [
                (state_page.data_ptr() + 7 * 256, 256),
                (state_page.data_ptr() + 6 * 256, 256),
            ],
        )

    def test_synchronous_proxy_dump_load_to_different_blocks_is_byte_exact(self):
        raw_config = config(group(["model.layers.0.attn"], FullAttentionSpec(4)))
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=4,
        )
        caches = {
            "model.layers.0.attn": (
                FakeTensor(0x1000, (8, 4, 2), (8, 2, 1)),
                FakeTensor(0x2000, (8, 4, 3), (12, 3, 1)),
            )
        }
        layout = make_cache_layout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        key = b"z" * 15
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        dump_plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (key,),
            plan_token_starts(0, 4, 1),
            (
                np.array(
                    [
                        1,
                    ],
                    dtype=np.uint64,
                ),
            ),
            group_ids=(0,),
        )
        load_plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (key,),
            plan_token_starts(0, 4, 1),
            (
                np.array(
                    [
                        6,
                    ],
                    dtype=np.uint64,
                ),
            ),
            group_ids=(0,),
        )
        dump_meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", dump_plans=(dump_plan,))}
        )
        load_meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(load_plan,))}
        )
        memory = ByteMemory()
        source_k = bytes((index * 7 + 3) % 256 for index in range(8))
        source_v = bytes((index * 11 + 5) % 256 for index in range(12))
        memory.write(0x1000 + 8, source_k)
        memory.write(0x2000 + 12, source_v)
        proxy = MemoryFixtureStore(memory)
        adapter = UCMProxyAdapter(PipelineProxy(proxy))

        dump_batch = _flat_layout_records(layout, "dump", dump_meta)
        _submit_segments(
            adapter,
            "dump",
            dump_batch.block_ids,
            dump_batch.offsets,
            dump_batch.ptrs,
            dump_batch.sizes,
        )
        ucm_chunk_ids = tuple(dict.fromkeys(dump_batch.block_ids))
        adapter.commit(
            True,
            ucm_chunk_ids,
            ucm_chunk_ids,
            np.arange(len(ucm_chunk_ids), dtype=np.int64),
        )
        self.assertEqual(
            adapter.lookup(ucm_ids((key,)), np.arange(1, dtype=np.int64)), (True,)
        )
        load_batch = _flat_layout_records(layout, "load", load_meta)
        _submit_segments(
            adapter,
            "load",
            load_batch.block_ids,
            load_batch.offsets,
            load_batch.ptrs,
            load_batch.sizes,
        )

        self.assertEqual(memory.read(0x1000 + 6 * 8, 8), source_k)
        self.assertEqual(memory.read(0x2000 + 6 * 12, 12), source_v)

        class DeferredByteStore(MemoryFixtureStore):
            def dump(
                self, block_ids, offsets, ptrs, sizes, metadata=(), record_nbytes=None
            ):
                return (
                    "dump",
                    (block_ids, offsets, ptrs, sizes, metadata, record_nbytes),
                )

            def load(self, block_ids, offsets, ptrs, sizes):
                return ("load", (block_ids, offsets, ptrs, sizes))

            def wait(self, task):
                operation, arguments = task
                if operation == "dump":
                    MemoryFixtureStore.dump(self, *arguments)
                else:
                    MemoryFixtureStore.load(self, *arguments)

        deferred_proxy = DeferredByteStore(memory)
        deferred_adapter = UCMProxyAdapter(PipelineProxy(deferred_proxy))
        _submit_segments(
            deferred_adapter,
            "dump",
            dump_batch.block_ids,
            dump_batch.offsets,
            dump_batch.ptrs,
            dump_batch.sizes,
        )
        ucm_chunk_ids = tuple(dict.fromkeys(dump_batch.block_ids))
        deferred_adapter.commit(
            True,
            ucm_chunk_ids,
            ucm_chunk_ids,
            np.arange(len(ucm_chunk_ids), dtype=np.int64),
        )
        deferred_load_plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (key,),
            plan_token_starts(0, 4, 1),
            (
                np.array(
                    [
                        7,
                    ],
                    dtype=np.uint64,
                ),
            ),
            group_ids=(0,),
        )
        deferred_load_meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(deferred_load_plan,))}
        )
        deferred_load_batch = _flat_layout_records(layout, "load", deferred_load_meta)
        _submit_segments(
            deferred_adapter,
            "load",
            deferred_load_batch.block_ids,
            deferred_load_batch.offsets,
            deferred_load_batch.ptrs,
            deferred_load_batch.sizes,
        )

        self.assertEqual(memory.read(0x1000 + 7 * 8, 8), source_k)
        self.assertEqual(memory.read(0x2000 + 7 * 12, 12), source_v)


class DeclaredLayoutModelTest(unittest.TestCase):
    """The declared mode must mirror vLLM's own packed-tensor placements."""

    def _layout(self, tensors=None):
        # Two dense per-layer tensors, 12-byte blocks, 8 blocks each; the
        # default declared placement packs them 96 bytes apart in one backing.
        declarations = (
            SimpleNamespace(
                size=192,
                layers=("model.layers.0.attn", "model.layers.1.attn"),
                offset=0,
                layer_stride=96,
                block_stride=12,
            ),
        )
        raw = config(
            group(
                ["model.layers.0.attn", "model.layers.1.attn"],
                {
                    "model.layers.0.attn": FullAttentionSpec(4),
                    "model.layers.1.attn": FullAttentionSpec(4),
                },
            ),
            tensors=declarations if tensors is None else tensors,
        )
        parsed = parse_spec(raw, scheduler_block_size=4, device_type="cpu")
        caches = {
            "model.layers.0.attn": FakeTensor(0x1000, (8, 4, 3), (12, 3, 1)),
            "model.layers.1.attn": FakeTensor(0x1060, (8, 4, 3), (12, 3, 1)),
        }
        return make_cache_layout(
            parsed, caches, kv_cache_config=raw, num_hidden_layers=4
        )

    def test_declarations_only_feed_the_block_first_special_case(self):
        # Layer-contiguous placements (layer_stride >= block_stride is
        # false here: 96 > 12 means layer-outermost) do not tile block
        # slots, so declarations change nothing: with or without them the
        # entries and the records are identical.
        declared = self._layout()
        undeclared = self._layout(tensors=())

        for group_id, declared_layout in declared.group_layouts.items():
            undeclared_layout = undeclared.group_layouts[group_id]
            self.assertEqual(declared_layout.layer_names, undeclared_layout.layer_names)
            for column in (
                "base_ptrs",
                "block_strides",
                "payload_bytes",
            ):
                self.assertEqual(
                    getattr(declared_layout, column).tolist(),
                    getattr(undeclared_layout, column).tolist(),
                )
            self.assertEqual(
                declared.store_layouts[group_id].whole_block_nbytes,
                undeclared.store_layouts[group_id].whole_block_nbytes,
            )
            np.testing.assert_array_equal(
                declared.store_layouts[group_id].block_segment_offsets,
                undeclared.store_layouts[group_id].block_segment_offsets,
            )
            self.assertIsNone(declared_layout.block_first)

    def test_disagreeing_block_stride_is_rejected(self):
        bad = (
            SimpleNamespace(
                size=192,
                layers=("model.layers.0.attn", "model.layers.1.attn"),
                offset=0,
                layer_stride=96,
                block_stride=13,
            ),
        )
        with self.assertRaisesRegex(ValueError, "block stride"):
            self._layout(tensors=bad)


class RawConfigDumpTest(unittest.TestCase):
    """UCM_V2_DUMP_CONFIG serializes the raw KVCacheConfig vLLM handed over."""

    def _connector(self, tmp, path_template):
        from ucm.integration.vllm.v2.ucm_connector import UCMConnector

        kv_cache_config = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=192,
                    layers=["model.layers.0.attn", "model.layers.1.attn"],
                    offset=0,
                    layer_stride=96,
                    block_stride=12,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["model.layers.0.attn", "model.layers.1.attn"],
                    kv_cache_spec=FullAttentionSpec(12),
                ),
            ],
            prefix_cache_retention_interval=0,
            kv_cache_layout="LBHNC",
        )
        os.environ["UCM_V2_DUMP_CONFIG"] = str(tmp / path_template)
        try:
            return UCMConnector(
                vllm_config(),
                KVConnectorRole.WORKER,
                kv_cache_config,
            )
        finally:
            os.environ.pop("UCM_V2_DUMP_CONFIG", None)

    def test_dumps_raw_config_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._connector(tmp, "raw_config.json")
            payload = json.loads((tmp / "raw_config.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["num_blocks"], 4)
            self.assertEqual(len(payload["kv_cache_tensors"]), 1)
            tensor = payload["kv_cache_tensors"][0]
            self.assertEqual(tensor["block_stride"], 12)
            self.assertEqual(
                tensor["layers"], ["model.layers.0.attn", "model.layers.1.attn"]
            )
            group = payload["kv_cache_groups"][0]
            self.assertEqual(
                group["layer_names"], ["model.layers.0.attn", "model.layers.1.attn"]
            )
            self.assertEqual(payload["kv_cache_layout"], "LBHNC")
            self.assertIn("block_size", group["kv_cache_spec"])

    def test_percent_d_receives_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._connector(tmp, "raw_config_rank%d.json")
            self.assertTrue((tmp / "raw_config_rank0.json").exists())


class BlockAccessTest(unittest.TestCase):
    def setUp(self):
        names = ["model.layers.0.attn", "model.layers.1.attn"]
        raw_config = config(group(names, FullAttentionSpec(8)), num_blocks=4)
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=8,
        )
        self.layout = make_cache_layout(
            parsed,
            {
                names[0]: FakeTensor(1000, (4, 8, 2), (16, 2, 1)),
                names[1]: FakeTensor(2000, (4, 8, 3), (24, 3, 1)),
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        ).group_layouts[0]

    def test_fixed_partial_window_repeats_across_physical_blocks(self):
        access = compile_access(self.layout, token_offsets=[4, 0], token_counts=[4, 8])
        ptrs, sizes = _resolved_access(access, [1, 2, 0, 3])
        np.testing.assert_array_equal(
            ptrs, [[1024, 2036], [1032, 2048], [1008, 2012], [1048, 2072]]
        )
        np.testing.assert_array_equal(sizes, [[8, 12], [16, 24]] * 2)

    def test_partial_ranges_and_layer_selection(self):
        mask = self.layout.segment_mask(layer_ids=[1])
        for start, block, expected_ptr in ((0, 2, 2048), (4, 2, 2060), (2, 0, 2006)):
            access = compile_access(self.layout, token_offsets=start, token_counts=4)
            ptrs, sizes = _resolved_access(access, [block], segment_mask=mask)
            np.testing.assert_array_equal(ptrs, [[expected_ptr]])
            np.testing.assert_array_equal(sizes, [[12]])

    def test_dynamic_subblocks_and_layer_selection(self):
        access = compile_access(self.layout, token_counts=4)
        mask = self.layout.segment_mask(layer_ids=[1])
        ptrs, sizes = _resolved_access(
            access, [2, 2], token_offsets=np.array([0, 4]), segment_mask=mask
        )
        np.testing.assert_array_equal(ptrs, [[2048], [2060]])
        np.testing.assert_array_equal(sizes, [[12], [12]])
        # Reuse the compiled template without mutating its fixed offsets.
        ptrs, _ = _resolved_access(
            access, [0], token_offsets=np.array([2]), segment_mask=mask
        )
        np.testing.assert_array_equal(ptrs, [[2006]])

    def test_resolved_arrays_do_not_alias_templates_or_other_calls(self):
        access = compile_access(self.layout, token_offsets=[4, 0], token_counts=[4, 8])
        first = _resolved_access(access, [1, 2, 0, 3])
        second = _resolved_access(access, [1, 2, 0, 3])
        expected = tuple(array.copy() for array in second)
        for array in first:
            array[:] = 0
        for actual, wanted in zip(second, expected):
            np.testing.assert_array_equal(actual, wanted)
        for actual, wanted in zip(_resolved_access(access, [1, 2, 0, 3]), expected):
            np.testing.assert_array_equal(actual, wanted)

    def test_ranges_with_empty_blocks_or_selection(self):
        access = compile_access(self.layout, token_counts=4)
        ptrs, sizes = _resolved_access(access, [])
        self.assertEqual(ptrs.shape, (0, 2))
        self.assertEqual(sizes.shape, (0, 2))
        ptrs, sizes = _resolved_access(
            access, [2], segment_mask=np.array([False, False])
        )
        self.assertEqual(ptrs.shape, (1, 0))
        self.assertEqual(sizes.shape, (1, 0))

    def test_dynamic_ranges_with_empty_blocks_or_selection(self):
        access = compile_access(self.layout, token_counts=4)
        ptrs, sizes = _resolved_access(
            access, [], token_offsets=np.array([], dtype=np.int64)
        )
        self.assertEqual(ptrs.shape, (0, 2))
        self.assertEqual(sizes.shape, (0, 2))
        ptrs, sizes = _resolved_access(
            access,
            [2],
            token_offsets=np.array([4]),
            segment_mask=np.array([False, False]),
        )
        self.assertEqual(ptrs.shape, (1, 0))
        self.assertEqual(sizes.shape, (1, 0))

    def test_invalid_ranges_and_incomplete_windows_are_rejected(self):
        with self.assertRaises(ValueError):
            compile_access(self.layout, token_offsets=6, token_counts=4)
        window = compile_access(self.layout, token_counts=[4, 8])
        with self.assertRaises(ValueError):
            _resolved_access(window, [0])


class LayerSegmentMergingTest(unittest.TestCase):
    def _attention(self, parts, *, names=None, chunk_size=8):
        names = names or ("model.layers.0.attn",)
        caches = dict(zip(names, parts))
        return make_layout(
            config(group(names, FullAttentionSpec(8)), num_blocks=3),
            caches,
            device_type="npu",
            scheduler_block_size=4,
            chunk_size=chunk_size,
        )

    def _packed_views(self):
        import torch

        raw = torch.arange(96, dtype=torch.uint8)
        key = torch.as_strided(raw, (3, 8, 1, 2), (32, 2, 2, 1))
        value = torch.as_strided(raw, (3, 8, 1, 1), (32, 1, 1, 1), storage_offset=16)
        full = torch.as_strided(raw, (3, 8, 1, 3), (32, 3, 3, 1))
        return raw, key, value, full

    def test_packed_kv_merges_and_preserves_block_stride_and_padding(self):
        raw, key, value, _ = self._packed_views()
        layout = self._attention(((key, value),))
        hbm = layout.group_layouts[0]
        self.assertFalse(hbm.supports_partial_tokens)
        self.assertEqual(hbm.payload_bytes.tolist(), [24])
        self.assertEqual(hbm.block_strides.tolist(), [32])
        offsets, ptrs, sizes = layout.store_layouts[0].resolve_matrices(
            np.array([2, 0], dtype=np.uint64), np.array([0, 8], dtype=np.uint64)
        )
        self.assertEqual(offsets.tolist(), [[0], [0]])
        self.assertEqual(ptrs.tolist(), [[raw.data_ptr() + 64], [raw.data_ptr()]])
        self.assertEqual(sizes.tolist(), [[24], [24]])
        with self.assertRaisesRegex(ValueError, "whole-block access"):
            self._attention(((key, value),), chunk_size=4)

    def test_a5_joint_alias_is_stored_once_in_any_component_position(self):
        import torch

        raw = torch.arange(128, dtype=torch.uint8)
        # Four four-byte KV cells and four fp32 scales share 32 bytes.
        key = torch.as_strided(raw, (3, 4, 1, 4), (40, 4, 4, 1))
        scale = torch.as_strided(
            raw.view(torch.float32),
            (3, 4, 1, 1),
            (10, 1, 1, 1),
            storage_offset=4,
        )
        full = torch.as_strided(raw, (3, 4, 1, 8), (40, 8, 8, 1))
        name = "model.layers.0.attn"
        for parts in ((key, scale, full), (full, key, scale), (key, full, scale)):
            with self.subTest(first=parts[0].shape):
                layout = make_layout(
                    config(group([name], FullAttentionSpec(4)), num_blocks=3),
                    {name: parts},
                    device_type="npu",
                    scheduler_block_size=4,
                )
                hbm = layout.group_layouts[0]
                self.assertEqual(hbm.payload_bytes.tolist(), [32])
                self.assertEqual(
                    hbm.resolve_ptrs([2]).tolist(), [[raw.data_ptr() + 80]]
                )
                self.assertEqual(layout.store_layouts[0].whole_chunk_nbytes, 32)
                self.assertFalse(hbm.supports_partial_tokens)

    def test_merged_file_roundtrip_ignores_alias_and_both_page_paddings(self):
        import torch

        source, key, value, full = self._packed_views()
        source_layout = self._attention(((key, value, full),))
        target = torch.full((144,), 255, dtype=torch.uint8)
        target_key = torch.as_strided(target, (3, 8, 1, 2), (48, 2, 2, 1))
        target_value = torch.as_strided(
            target, (3, 8, 1, 1), (48, 1, 1, 1), storage_offset=16
        )
        target_layout = self._attention(((target_key, target_value),))
        starts = np.array([0], dtype=np.uint64)
        dumped = source_layout.store_layouts[0].resolve_matrices(
            np.array([2], dtype=np.uint64), starts
        )
        loaded = target_layout.store_layouts[0].resolve_matrices(
            np.array([1], dtype=np.uint64), starts
        )
        chunk_ids = (logical_id(97),)
        with tempfile.TemporaryDirectory() as directory:
            store = FileTestStore(directory)
            store.register_tensors(
                {"source": (key, value, full), "target": (target_key, target_value)}
            )
            store.dump(chunk_ids, *dumped)
            store.commit(True, chunk_ids, chunk_ids)
            self.assertEqual(
                (Path(directory) / (chunk_ids[0].hex() + ".ucm")).read_bytes(),
                bytes(source[64:88].tolist()),
            )
            store.load(chunk_ids, *loaded)
        expected = torch.full_like(target, 255)
        expected[48:72] = source[64:88]
        self.assertTrue(torch.equal(target, expected))

    def test_conv_and_ssm_with_different_dtypes_merge_without_page_padding(self):
        import torch

        raw = torch.zeros(192, dtype=torch.uint8)
        conv = torch.as_strided(
            raw.view(torch.bfloat16), (3, 2, 3), (32, 3, 1), storage_offset=2
        )
        ssm = torch.as_strided(
            raw.view(torch.float32), (3, 2, 2), (16, 2, 1), storage_offset=4
        )
        name = "model.layers.0.state"
        layout = make_layout(
            config(group([name], MambaSpec(8)), num_blocks=3),
            {name: (conv, ssm)},
            device_type="npu",
        )
        hbm = layout.group_layouts[0]
        self.assertEqual(hbm.payload_bytes.tolist(), [28])
        self.assertEqual(hbm.block_strides.tolist(), [64])
        self.assertEqual(hbm.resolve_ptrs([2]).tolist(), [[raw.data_ptr() + 132]])
        self.assertFalse(hbm.supports_partial_tokens)

    def test_layer_names_of_one_layer_stay_separate(self):
        raw, key, value, _ = self._packed_views()
        names = ("model.layers.0.attn", "model.layers.0.indexer")
        hbm = self._attention((key, value), names=names).group_layouts[0]
        self.assertEqual(hbm.base_ptrs.tolist(), [raw.data_ptr(), raw.data_ptr() + 16])
        self.assertEqual(hbm.layer_ids.tolist(), [0, 0])
        self.assertEqual(
            hbm.segment_mask(layer_names=(names[0],)).tolist(), [True, False]
        )
        self.assertEqual(
            hbm.segment_mask(layer_names=(names[1],)).tolist(), [False, True]
        )
        self.assertEqual(hbm.segment_mask(layer_ids=(0,)).tolist(), [True, True])
        self.assertTrue(hbm.supports_partial_tokens)
        layout = self._attention((key, value), names=names, chunk_size=4)
        self.assertEqual(layout.store_layouts[0].chunk_segment_bytes.tolist(), [[8, 4]])

    def test_adjacent_different_layers_and_groups_stay_separate(self):
        raw, key, value, _ = self._packed_views()
        names = ("model.layers.0.attn", "model.layers.1.attn")
        hbm = self._attention((key, value), names=names).group_layouts[0]
        self.assertEqual(hbm.payload_bytes.tolist(), [16, 8])
        self.assertTrue(hbm.supports_partial_tokens)
        names = ("model.layers.0.attn", "model.layers.0.indexer")
        layout = make_layout(
            config(
                *(group([name], FullAttentionSpec(8)) for name in names), num_blocks=3
            ),
            dict(zip(names, (key, value))),
            device_type="npu",
        )
        self.assertEqual(layout.group_layouts[0].payload_bytes.tolist(), [16])
        self.assertEqual(layout.group_layouts[1].payload_bytes.tolist(), [8])
        self.assertEqual(
            layout.group_layouts[1].resolve_ptrs([1]).tolist(), [[raw.data_ptr() + 48]]
        )

    def test_aliases_under_different_layer_names_are_not_deduplicated(self):
        raw, key, _, _ = self._packed_views()
        names = ("model.layers.0.attn", "model.layers.0.indexer")
        hbm = self._attention((key, key), names=names).group_layouts[0]
        self.assertEqual(hbm.base_ptrs.tolist(), [raw.data_ptr(), raw.data_ptr()])
        self.assertEqual(hbm.payload_bytes.tolist(), [16, 16])
        self.assertEqual(hbm.layer_names, names)
        self.assertTrue(hbm.supports_partial_tokens)

    def test_adjacent_base_addresses_with_different_strides_do_not_merge(self):
        import torch

        raw = torch.zeros(128, dtype=torch.uint8)
        key = torch.as_strided(raw, (3, 8, 1, 2), (32, 2, 2, 1))
        value = torch.as_strided(raw, (3, 8, 1, 1), (48, 1, 1, 1), storage_offset=16)
        hbm = self._attention(((key, value),)).group_layouts[0]
        self.assertEqual(hbm.payload_bytes.tolist(), [16, 8])
        self.assertEqual(
            hbm.resolve_ptrs([2]).tolist(),
            [[raw.data_ptr() + 64, raw.data_ptr() + 112]],
        )
        self.assertTrue(hbm.supports_partial_tokens)

    def test_separate_allocations_keep_logical_order_even_when_adjacent(self):
        for key_base, value_base in ((1000, 1008), (2000, 1000)):
            key = FakeTensor(key_base, (3, 8, 1, 1), (32, 1, 1, 1))
            value = FakeTensor(value_base, (3, 8, 1, 1), (32, 1, 1, 1))
            hbm = self._attention(((key, value),)).group_layouts[0]
            self.assertEqual(hbm.base_ptrs.tolist(), [key_base, value_base])
            self.assertEqual(hbm.payload_bytes.tolist(), [8, 8])

    def test_partial_overlap_keeps_only_the_unique_tail(self):
        import torch

        raw = torch.arange(96, dtype=torch.uint8)
        first = torch.as_strided(raw, (3, 8, 1, 2), (32, 2, 2, 1))
        second = torch.as_strided(raw, (3, 8, 1, 2), (32, 2, 2, 1), storage_offset=8)
        hbm = self._attention(((first, second),)).group_layouts[0]
        self.assertEqual(hbm.payload_bytes.tolist(), [24])
        self.assertEqual(hbm.base_ptrs.tolist(), [raw.data_ptr()])
        self.assertFalse(hbm.supports_partial_tokens)

    def test_multirow_head_fragments_merge_at_the_layer_level(self):
        name = "model.layers.0.attn"
        cache = FakeTensor(1000, (6, 2, 4, 2), (8, 48, 2, 1))
        layout = make_layout(
            config(group([name], FullAttentionSpec(8)), num_blocks=3),
            {name: cache},
            device_type="cpu",
        )
        hbm = layout.group_layouts[0]
        layer_view = hbm.layer_views[name]
        self.assertEqual(len(layer_view.components[0].segments), 4)
        self.assertEqual(len(layer_view.segments), 2)
        self.assertFalse(layer_view.supports_partial_tokens)
        self.assertEqual(hbm.base_ptrs.tolist(), [1000, 1048])
        self.assertEqual(hbm.payload_bytes.tolist(), [16, 16])
        self.assertEqual(hbm.resolve_ptrs([2]).tolist(), [[1032, 1080]])


class LayerViewSegmentsTest(unittest.TestCase):
    def test_npu_split_components_use_view_strides_not_whole_page_descriptor(self):
        name = "model.layers.0.attn"
        raw = config(
            group([name], FullAttentionSpec(8)),
            tensors=(
                SimpleNamespace(
                    layers=(name,), offset=0, layer_stride=512, block_stride=64
                ),
            ),
        )
        layout = make_layout(
            raw,
            {
                name: (
                    FakeTensor(1000, (8, 8, 1, 2), (16, 2, 2, 1)),
                    FakeTensor(2000, (8, 8, 1, 1), (8, 1, 1, 1)),
                )
            },
            device_type="npu",
        )
        self.assertEqual(
            layout.group_layouts[0].resolve_ptrs([3]).tolist(), [[1048, 2024]]
        )
        self.assertIsNone(layout.group_layouts[0].block_first)
        self.assertEqual(layout.store_layouts[0].whole_chunk_nbytes, 24)

    def test_cpu_native_descriptor_stride_is_still_validated(self):
        name = "model.layers.0.attn"
        raw = config(
            group([name], FullAttentionSpec(8)),
            tensors=(
                SimpleNamespace(
                    layers=(name,), offset=0, layer_stride=512, block_stride=64
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            make_layout(raw, {name: FakeTensor(1000, (8, 1, 8, 2), (16, 16, 2, 1))})

    def test_separate_backings_do_not_merge_from_declarations_alone(self):
        names = ["model.layers.0.attn", "model.layers.1.attn"]
        raw_config = config(
            group(names, FullAttentionSpec(8)),
            num_blocks=3,
            tensors=tuple(
                SimpleNamespace(
                    layers=(name,), offset=i * 16, layer_stride=16, block_stride=32
                )
                for i, name in enumerate(names)
            ),
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=8,
            device_type="cpu",
        )
        layout = make_cache_layout(
            parsed,
            {
                names[0]: FakeTensor(1000, (3, 8, 1), (32, 1, 1)),
                names[1]: FakeTensor(5000, (3, 8, 1), (32, 1, 1)),
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        self.assertIsNone(layout.group_layouts[0].block_first)
        record = layout.store_layouts[0]
        offsets, ptrs, sizes, entries = _flat_record_access(record, np.array([2]), 1)
        self.assertEqual(
            (offsets.tolist(), ptrs.tolist(), sizes.tolist(), entries),
            ([0, 8], [1064, 5064], [8, 8], 2),
        )

    def test_multi_head_wa_partial_tail_is_rejected(self):
        # A 12-token tail starts mid-block; head-major HNC bytes cannot be
        # sliced by token, so the layout refuses the WA group at registration.
        name = "model.layers.1.swa_cache"
        raw_config = config(
            group(["model.layers.0.attn"], FullAttentionSpec(8)),
            group([name], AscendSlidingWindowMLASpec(8, 1, 12)),
            num_blocks=3,
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=8,
            chunk_size=16,
            device_type="cpu",
        )
        with self.assertRaisesRegex(ValueError, "group 1 requires whole-block access"):
            make_cache_layout(
                parsed,
                {
                    "model.layers.0.attn": FakeTensor(100, (3, 8, 2), (16, 2, 1)),
                    name: FakeTensor(1000, (3, 2, 8, 2), (32, 16, 2, 1)),
                },
                kv_cache_config=raw_config,
                num_hidden_layers=4,
            )

    def test_equal_axis_sizes_use_backend_semantics(self):
        name = "model.layers.0.attn"
        for device, strides, expected in (
            ("cpu", (16, 4, 1, 1), None),
            ("npu", (16, 4, 1, 1), [[1016]]),
        ):
            raw_config = config(group([name], FullAttentionSpec(4)), num_blocks=3)
            parsed = parse_spec(
                raw_config,
                scheduler_block_size=4,
                device_type=device,
            )
            gl = make_cache_layout(
                parsed,
                {
                    name: FakeTensor(996, (3, 4, 4, 1), strides),
                },
                kv_cache_config=raw_config,
                num_hidden_layers=4,
            ).group_layouts[0]
            if expected is None:
                # The same strides read as multi-head HNC on CPU.
                with self.assertRaisesRegex(ValueError, "whole-block access"):
                    compile_access(gl, token_offsets=1, token_counts=2)
                continue
            ptrs, sizes = _resolved_access(
                compile_access(gl, token_offsets=1, token_counts=2), [1]
            )
            self.assertEqual(ptrs.tolist(), expected)
            self.assertEqual(int(sizes.sum()), 8)

    def test_multirow_head_separated_range_is_explicitly_rejected(self):
        name = "model.layers.0.attn"
        raw_config = config(group([name], FullAttentionSpec(8)), num_blocks=3)
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=8,
            device_type="cpu",
        )
        layout = make_cache_layout(
            parsed,
            {
                name: FakeTensor(1000, (6, 2, 4, 2), (16, 8, 2, 1)),
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        with self.assertRaisesRegex(ValueError, "whole-block access"):
            compile_access(layout.group_layouts[0], token_offsets=0, token_counts=4)

    def _layout(self, order, *, unit=None, declared=False):
        # Physical backing axes, independently turned into logical BHNC views.
        dims = {"L": 2, "B": 3, "H": 2, "N": 8, "C": 2}
        strides = {}
        stride = 1
        for axis in reversed(order):
            strides[axis] = stride
            stride *= dims[axis]
        names = ["model.layers.0.attn", "model.layers.1.attn"]
        tensors = ()
        if declared:
            tensors = (
                SimpleNamespace(
                    layers=tuple(names),
                    offset=0,
                    layer_stride=strides["L"],
                    block_stride=strides["B"],
                ),
            )
        raw_config = config(
            group(names, FullAttentionSpec(8)),
            num_blocks=3,
            tensors=tensors,
            kv_cache_layout=order,
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=8,
            chunk_size=unit,
            device_type="cpu",
        )
        layout = make_cache_layout(
            parsed,
            {
                name: FakeTensor(
                    1000 + layer * strides["L"],
                    (3, 2, 8, 2),
                    tuple(strides[a] for a in "BHNC"),
                )
                for layer, name in enumerate(names)
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        return layout, strides

    def test_layout_permutations_select_exact_logical_elements(self):
        for order in ("LBNHC", "LBHNC", "BLNHC", "BLHNC", "LHBNC", "BHLNC"):
            with self.subTest(order=order):
                # Token-contiguous NHC reads half blocks; multi-head HNC
                # bytes are head-major, so only whole blocks are addressable.
                partial = order.endswith("NHC")
                count = 4 if partial else 8
                starts = (0, 4) if partial else (0,)
                layout, strides = self._layout(order)
                group_layout = layout.group_layouts[0]
                if not partial:
                    with self.assertRaisesRegex(ValueError, "whole-block access"):
                        compile_access(group_layout, token_counts=4)
                for start in starts:
                    access = compile_access(
                        group_layout, token_offsets=start, token_counts=count
                    )
                    ptrs, sizes = _resolved_access(
                        access,
                        [2],
                        segment_mask=group_layout.segment_mask(layer_ids=[1]),
                    )
                    actual = [
                        address
                        for ptr, size in zip(ptrs[0], sizes[0])
                        for address in range(int(ptr), int(ptr + size))
                    ]
                    expected = sorted(
                        1000
                        + strides["L"]
                        + 2 * strides["B"]
                        + h * strides["H"]
                        + n * strides["N"]
                        + c
                        for h in range(2)
                        for n in range(start, start + count)
                        for c in range(2)
                    )
                    self.assertEqual(actual, expected)
                lv = group_layout.layer_views["model.layers.1.attn"]
                self.assertEqual(len(lv.components), 1)
                compact = order[-3:] in ("NHC", "HNC")
                self.assertEqual(len(lv.segments), 1 if compact else 2)

    def test_ucm_offsets_and_layerwise_selection_share_one_mapping(self):
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        for order in ("LBHNC", "LHBNC", "BHLNC"):
            with self.subTest(order=order):
                # Multi-head HNC: one UCM chunk per whole vLLM block.
                layout, strides = self._layout(order, unit=8)
                plan = UCMGroupDispatchPlan(
                    RestorePolicy.FA,
                    (b"a" * 15, b"b" * 15),
                    plan_token_starts(0, 16, 2),
                    (np.array([1, 2], dtype=np.uint64),),
                    group_ids=(0,),
                )
                meta = UCMConnectorMetadata(
                    requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
                )
                batch = _flat_layout_records(layout, "load", meta)
                selected = _flat_layout_records(layout, "load", meta, layer_id=1)
                if order == "LBHNC":
                    # Compact HNC page: one 32-byte segment per layer.
                    self.assertEqual(batch.offsets.tolist(), [0, 32] * 2)
                    self.assertEqual(batch.sizes.tolist(), [32] * 4)
                    self.assertEqual(selected.offsets.tolist(), [32] * 2)
                    rows = [1, 3]
                else:
                    # Head-separated: one 16-byte segment per layer and head.
                    self.assertEqual(batch.offsets.tolist(), [0, 16, 32, 48] * 2)
                    self.assertEqual(batch.sizes.tolist(), [16] * 8)
                    self.assertEqual(selected.offsets.tolist(), [32, 48] * 2)
                    rows = [2, 3, 6, 7]
                self.assertEqual(selected.ptrs.tolist(), batch.ptrs[rows].tolist())
                # Independent memory oracle checks byte content, not roundtrip alone.
                memory = ByteMemory()
                for address in range(1000, 1192):
                    memory.write(address, bytes([address % 251]))
                dump = _flat_layout_records(
                    layout,
                    "dump",
                    UCMConnectorMetadata(
                        requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
                    ),
                )
                with tempfile.TemporaryDirectory() as tmp:
                    proxy = FileTestStore(tmp, MemoryByteAccess(memory))
                    proxy.dump(dump.block_ids, dump.offsets, dump.ptrs, dump.sizes)
                    proxy.commit(True, dump.block_ids, dump.block_ids)
                    for k, block in zip(ucm_ids(plan.common_ucm_chunk_ids), (1, 2)):
                        expected = bytes(
                            (
                                1000
                                + l * strides["L"]
                                + block * strides["B"]
                                + h * strides["H"]
                                + n * strides["N"]
                                + c
                            )
                            % 251
                            for l in range(2)
                            for h in range(2)
                            for n in range(8)
                            for c in range(2)
                        )
                        self.assertEqual(
                            (Path(tmp) / (k.hex() + ".ucm")).read_bytes(), expected
                        )

    def test_compact_declared_heads_keep_bulk_fast_path(self):
        layout, _ = self._layout("BLHNC", unit=8, declared=True)
        group_layout = layout.group_layouts[0]
        self.assertIsNotNone(group_layout.block_first)
        access = compile_access(group_layout)
        ptrs, sizes = _resolved_access(
            access, [1], segment_mask=group_layout.segment_mask(layer_ids=[1])
        )
        # One compact [H, N, C] page per layer.
        self.assertEqual(ptrs.tolist(), [[1096]])
        self.assertEqual(sizes.tolist(), [[32]])
        # Each segment keeps its offset inside the span for layerwise IO.
        self.assertEqual(
            layout.store_layouts[group_layout.group_id].block_segment_offsets.tolist(),
            [0, 32],
        )
        # BHLNC interleaves other layers between heads: must not copy a
        # presumed contiguous layer page via the declaration fast path.
        fragmented, _ = self._layout("BHLNC", unit=8, declared=True)
        self.assertIsNone(fragmented.group_layouts[0].block_first)

    def test_real_kimi_kernel_rows_and_shared_components(self):
        name = "model.layers.0.attn"
        for device, blocks, tensors, expected_ptrs in (
            (
                "cpu",
                30,
                FakeTensor(1000, (2880, 1, 64, 576), (36864, 576, 576, 1), 2),
                [1000 + 2 * 7077888 + 512 * 1152],
            ),
            (
                "npu",
                27,
                (
                    FakeTensor(1000, (1296, 128, 1, 512), (65536, 512, 512, 1), 2),
                    FakeTensor(90000000, (1296, 128, 1, 64), (8192, 64, 64, 1), 2),
                ),
                [1000 + 2 * 6291456 + 512 * 1024, 90000000 + 2 * 786432 + 512 * 128],
            ),
        ):
            with self.subTest(device=device):
                raw_config = config(
                    group([name], FullAttentionSpec(6144)), num_blocks=blocks
                )
                parsed = parse_spec(
                    raw_config,
                    scheduler_block_size=6144,
                    device_type=device,
                )
                gl = make_cache_layout(
                    parsed,
                    {name: tensors},
                    kv_cache_config=raw_config,
                    num_hidden_layers=4,
                ).group_layouts[0]
                ptrs, sizes = _resolved_access(
                    compile_access(gl, token_offsets=512, token_counts=512), [2]
                )
                self.assertEqual(ptrs.tolist(), [expected_ptrs])
                self.assertEqual(int(sizes.sum()), 512 * 1152)

    def test_documented_fa_subblocks_preserve_layer_offsets(self):
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        names = ["model.layers.0.attn", "model.layers.1.attn"]
        raw_config = config(group(names, FullAttentionSpec(16384)), num_blocks=3)
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=16384,
            chunk_size=512,
            device_type="npu",
        )
        layout = make_cache_layout(
            parsed,
            {
                name: FakeTensor(base, (3, 16384, 2, 2), (65536, 4, 2, 1))
                for name, base in zip(names, (1000000, 2000000))
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (b"a" * 15, b"b" * 15),
            plan_token_starts(0, 1024, 2),
            (np.array([2, 2], dtype=np.uint64),),
            group_ids=(0,),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = _flat_layout_records(layout, "load", metadata)
        self.assertEqual(batch.ptrs.tolist(), [1131072, 2131072, 1133120, 2133120])
        self.assertEqual(batch.sizes.tolist(), [2048] * 4)
        self.assertEqual(batch.offsets.tolist(), [0, 2048, 0, 2048])
        selected = _flat_layout_records(layout, "load", metadata, layer_id=1)
        self.assertEqual(selected.ptrs.tolist(), [2131072, 2133120])
        self.assertEqual(selected.offsets.tolist(), [2048, 2048])
        # A single-group batch now avoids concatenate; its buffers must still
        # be independent from another dispatch and from compiled templates.
        expected = [array.copy() for array in (batch.offsets, batch.ptrs, batch.sizes)]
        retained = _flat_layout_records(layout, "load", metadata)
        for array in (batch.offsets, batch.ptrs, batch.sizes):
            array[:] = 0
        later = _flat_layout_records(layout, "load", metadata)
        for other in (retained, later):
            for actual, wanted in zip(
                (other.offsets, other.ptrs, other.sizes), expected
            ):
                np.testing.assert_array_equal(actual, wanted)

    def test_fa_16384_tokens_to_32_ucm_keys(self):
        name = "model.layers.0.attn"
        raw_config = config(group([name], FullAttentionSpec(16384)), num_blocks=3)
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=16384,
            chunk_size=512,
            device_type="npu",
        )
        layout = make_cache_layout(
            parsed,
            {name: FakeTensor(1000, (3, 16384, 2, 2), (65536, 4, 2, 1))},
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        access = compile_access(layout.group_layouts[0], token_counts=512)
        ptrs, sizes = _resolved_access(
            access, [2] * 32, token_offsets=np.arange(32) * 512
        )
        np.testing.assert_array_equal(
            ptrs[:, 0], 1000 + 2 * 65536 + np.arange(32) * 2048
        )
        np.testing.assert_array_equal(sizes, np.full((32, 1), 2048))

    def test_fa_slices_need_token_contiguous_layout(self):
        # Multi-head HNC bytes are head-major: a half block is not one
        # contiguous range per head, so slices are rejected at registration.
        with self.assertRaisesRegex(ValueError, "not token-contiguous"):
            self._layout("LBHNC", unit=4)


class MatrixTransferTest(unittest.TestCase):
    def _metadata(self, ucm_chunk_ids=(b"a" * 15, b"b" * 15)):
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            ucm_chunk_ids,
            plan_token_starts(0, 16, len(ucm_chunk_ids)),
            (np.array([1, 2], dtype=np.uint64),),
            group_ids=(0,),
        )
        return UCMConnectorMetadata(
            requests={
                "r": RequestDispatchMeta("r", load_plans=(plan,), dump_plans=(plan,))
            }
        )

    def test_matrix_bytes_and_layerwise_offsets(self):
        layout, _ = LayerViewSegmentsTest()._layout("LBNHC")
        meta = self._metadata()
        (batch,) = UCMTransferBuilder(layout).build_dump_transfers(meta)
        self.assertEqual(batch.ucm_chunk_ids, ucm_ids((b"a" * 15, b"b" * 15)))
        self.assertEqual(batch.ptrs.shape, (2, 2))
        self.assertEqual(batch.ucm_chunk_offsets.tolist(), [[0, 32]] * 2)
        self.assertEqual(batch.sizes.tolist(), [[32] * 2] * 2)
        self.assertEqual(batch.sizes.strides[0], 0)
        self.assertFalse(batch.sizes.flags.writeable)
        (selected,) = UCMTransferBuilder(layout).build_load_transfers(meta, layer_id=1)
        np.testing.assert_array_equal(selected.ptrs, batch.ptrs[:, 1:])
        self.assertEqual(selected.ucm_chunk_offsets.tolist(), [[32]] * 2)
        legacy = _flat_layout_records(layout, "dump", meta)
        memory = ByteMemory()
        for address in range(1000, 1192):
            memory.write(address, bytes([address % 251]))
        with tempfile.TemporaryDirectory() as directory:
            proxy = FileTestStore(directory, MemoryByteAccess(memory))
            adapter = UCMProxyAdapter(PipelineProxy(proxy))
            adapter.submit("dump", batch)
            adapter.commit(
                True,
                batch.ucm_chunk_ids,
                batch.ucm_chunk_ids,
                np.arange(len(batch.ucm_chunk_ids), dtype=np.int64),
            )
            expected = []
            for key in batch.ucm_chunk_ids:
                pieces = [
                    (int(o), memory.read(int(p), int(n)))
                    for k, o, p, n in zip(
                        legacy.block_ids, legacy.offsets, legacy.ptrs, legacy.sizes
                    )
                    if k == key
                ]
                expected.append(b"".join(payload for _, payload in sorted(pieces)))
                self.assertEqual(
                    (Path(directory) / (key.hex() + ".ucm")).read_bytes(), expected[-1]
                )
            for address in range(1000, 1192):
                memory.write(address, b"\0")
            adapter.submit("load", batch)
            for i in range(2):
                self.assertEqual(
                    b"".join(
                        memory.read(int(p), int(n))
                        for p, n in zip(batch.ptrs[i], batch.sizes[i])
                    ),
                    expected[i],
                )

    def test_multiple_groups_match_legacy_record_positions(self):
        from ucm.integration.vllm.v2.connector_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        names = ["model.layers.0.attn", "model.layers.1.attn"]
        raw_config = config(
            group([names[0]], FullAttentionSpec(8)),
            group([names[1]], FullAttentionSpec(16)),
            num_blocks=4,
        )
        spec = parse_spec(raw_config, scheduler_block_size=16, chunk_size=16)
        layout = make_cache_layout(
            spec,
            {
                names[0]: FakeTensor(1000, (4, 8, 1), (8, 1, 1)),
                names[1]: FakeTensor(2000, (4, 16, 1), (16, 1, 1)),
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        # One 16-token UCM chunk: two blocks of group 0, one of group 1.
        plan = UCMGroupDispatchPlan(
            RestorePolicy.FA,
            (b"a" * 15,),
            plan_token_starts(0, 16, 1),
            (np.array([0, 1]), np.array([2])),
            group_ids=(0, 1),
        )
        meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        (batch,) = UCMTransferBuilder(layout).build_load_transfers(meta)
        self.assertEqual(batch.ptrs.tolist(), [[1000, 1008, 2032]])
        self.assertEqual(batch.ucm_chunk_offsets.tolist(), [[0, 8, 16]])
        self.assertEqual(batch.sizes.tolist(), [[8, 8, 16]])
        (selected,) = UCMTransferBuilder(layout).build_load_transfers(meta, layer_id=1)
        self.assertEqual(selected.ucm_chunk_offsets.tolist(), [[16]])

    def test_shared_keys_across_requests_keep_separate_destinations(self):
        from dataclasses import replace

        layout, _ = LayerViewSegmentsTest()._layout("LBNHC")
        metadata = self._metadata()
        first = metadata.requests["r"]
        plan = first.load_plans[0]
        other_plan = replace(
            plan, vllm_block_ids_by_group=(np.array([0, 0], dtype=np.uint64),)
        )
        metadata.requests["other"] = replace(
            first,
            request_id="other",
            load_plans=(other_plan,),
            dump_plans=(other_plan,),
        )
        builder = UCMTransferBuilder(layout)
        for build in (builder.build_load_transfers, builder.build_dump_transfers):
            for layer_id in (None, 1):
                batches = build(metadata, layer_id=layer_id)
                self.assertEqual(len(batches), 2)
                self.assertEqual(
                    [b.ucm_chunk_ids for b in batches],
                    [ucm_ids(plan.common_ucm_chunk_ids)] * 2,
                )
                self.assertFalse(np.array_equal(batches[0].ptrs, batches[1].ptrs))

    def test_templates_and_retained_transfers_are_independent(self):
        layout, _ = LayerViewSegmentsTest()._layout("LBNHC")
        meta = self._metadata()
        (first,) = UCMTransferBuilder(layout).build_load_transfers(meta)
        (retained,) = UCMTransferBuilder(layout).build_load_transfers(meta)
        expected = retained.ptrs.copy()
        first.ptrs[:] = 1
        # Even a caller deliberately changing a broadcast view's backing
        # cannot change the compiled layout or another retained submission.
        first.sizes.base[:] = 1
        (later,) = UCMTransferBuilder(layout).build_load_transfers(meta)
        np.testing.assert_array_equal(retained.ptrs, expected)
        np.testing.assert_array_equal(later.ptrs, expected)
        self.assertEqual(later.sizes.tolist(), [[32] * 2] * 2)

    def test_submit_passes_through_strided_and_broadcast_matrices(self):

        class RecordingTransferStore(RecordingTestStore):
            def load(self, *args):
                self.received = args

            dump = load

        keys = (b"a" * 16, b"b" * 16)
        ptrs = np.array([[10, 11], [20, 21]], dtype=np.uint64)[:, 1:]
        sizes = np.broadcast_to(np.array([5], dtype=np.uint64), (2, 1))
        offsets = np.broadcast_to(np.array([3], dtype=np.uint64), (2, 1))
        transfer = make_transfer(keys, ptrs, sizes, offsets)
        proxy = RecordingTransferStore()
        adapter = UCMProxyAdapter(PipelineProxy(proxy))
        for operation in ("load", "dump"):
            adapter.submit(operation, transfer)
            for actual, original in zip(proxy.received, (keys, offsets, ptrs, sizes)):
                self.assertIs(actual, original)

    def test_single_column_strided_rows_keep_key_and_segment_order(self):
        keys = (b"a" * 16, b"b" * 16)
        ptrs = np.array([[10, 11], [20, 21]], dtype=np.uint64)[:, 1:]
        offsets = np.broadcast_to(np.array([3], dtype=np.uint64), (2, 1))
        sizes = np.broadcast_to(np.array([5], dtype=np.uint64), (2, 1))
        records = [
            (key, list(segments))
            for key, segments in record_segments(keys, offsets, ptrs, sizes)
        ]
        self.assertEqual(records, [(keys[0], [(3, 11, 5)]), (keys[1], [(3, 21, 5)])])

    def test_matrix_buffers_live_until_async_wait(self):
        layout, _ = LayerViewSegmentsTest()._layout("LBNHC")
        (batch,) = UCMTransferBuilder(layout).build_load_transfers(self._metadata())

        class Deferred(RecordingTestStore):
            def load(self, keys, offsets, ptrs, sizes):
                self.refs = (keys, offsets, ptrs, sizes)
                return object()

            def wait(self, task):
                self.checked = self.refs[2].shape == (2, 2) and self.refs[3][0, 0] == 32

        proxy = Deferred()
        UCMProxyAdapter(PipelineProxy(proxy)).submit("load", batch)
        self.assertTrue(proxy.checked)


if __name__ == "__main__":
    unittest.main()
