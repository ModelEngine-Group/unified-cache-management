import enum
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


vllm_base.KVConnectorBase_V1 = KVConnectorBaseV1
vllm_base.KVConnectorMetadata = KVConnectorMetadata
vllm_base.KVConnectorRole = KVConnectorRole
vllm_base.KVConnectorWorkerMetadata = KVConnectorWorkerMetadata
vllm_base.SupportsHMA = SupportsHMA
vllm_base.KVConnectorTransferResults = SimpleNamespace
sys.modules["vllm"].envs = SimpleNamespace(VLLM_KV_CACHE_LAYOUT="NHD")
sys.modules[vllm_base.__name__] = vllm_base

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
    dcp_sharded = False
    prefix_cacheable = True


class _AttentionSpec(_KVCacheSpec):
    dcp_sharded = True
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


for name, value in (
    ("KVCacheSpecKind", KVCacheSpecKind),
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

# The connector resolves its scheduler granularity through vLLM's
# resolve_kv_cache_block_sizes; mirror its dcp=1 semantics (single group =
# cache_config.block_size, multiple groups = LCM of group block sizes).
vllm_v1_core = types.ModuleType("vllm.v1.core")
vllm_v1_core.__path__ = []
sys.modules.setdefault("vllm.v1.core", vllm_v1_core)
vllm_kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")


def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
    dcp = int(
        getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1) or 1
    )
    groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    if len(groups) <= 1:
        block_size = (
            groups[0].kv_cache_spec.block_size
            if groups
            else vllm_config.cache_config.block_size
        )
        size = int(block_size) * dcp
        return size, size
    scheduler = math.lcm(*(int(g.kv_cache_spec.block_size) for g in groups))
    return scheduler, scheduler


vllm_kv_cache_utils.resolve_kv_cache_block_sizes = resolve_kv_cache_block_sizes
sys.modules.setdefault(vllm_kv_cache_utils.__name__, vllm_kv_cache_utils)

from ucm.integration.vllm.v2.layout import (  # noqa: E402
    UCMKVCacheLayout,
    parse_kv_cache_config,
)
from ucm.integration.vllm.v2.test_stores import (
    FileTestStore,
    MemoryTestStore,
    record_segments,
)
from ucm.integration.vllm.v2.ucm_block_id import (  # noqa: E402
    make_common_ucm_block_ids,
    make_local_ucm_block_ids,
)
from ucm.integration.vllm.v2.ucm_connector import (  # noqa: E402
    UCMConnector,
    UCMWorkerMetadata,
)
from ucm.integration.vllm.v2.ucm_kv_cache import UCMTransferBuilder  # noqa: E402
from ucm.integration.vllm.v2.ucm_proxy import (  # noqa: E402
    PipelineProxy,
    UCMProxyAdapter,
    UCMProxyError,
)
from ucm.integration.vllm.v2.ucm_scheduler import (  # noqa: E402
    AttentionType,
    RequestState,
    UCMConnectorMetadata,
    UCMDispatcher,
    make_request_hasher,
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
    return UCMKVCacheLayout(
        parsed, caches, kv_cache_config=raw_config, num_hidden_layers=64
    )


def parse_spec(kv_cache_config, *, hf_text_config=DENSE_TEXT_CONFIG, **kwargs):
    """parse_kv_cache_config for a dense model unless the test passes its own."""
    return parse_kv_cache_config(
        kv_cache_config, hf_text_config=hf_text_config, **kwargs
    )


def logical_id(byte, attention_type=AttentionType.FA):
    """Single-rank local UCM block id of a block whose common id is ``byte``
    repeated."""
    from ucm.integration.vllm.v2.ucm_block_id import make_local_ucm_block_ids

    return make_local_ucm_block_ids((bytes([byte]) * 12,), attention_type)[0]


class FakeRequest:
    def __init__(self, request_id, token_count):
        self.request_id = request_id
        self.all_token_ids = list(range(token_count))


def make_dispatcher(
    parsed, proxy=None, hasher=None, *, recompute_tokens=1, parallel=None, **kwargs
):
    """A UCMDispatcher whose lookup probes ``proxy`` (a recording Store by default)."""
    from ucm.integration.vllm.v2.parallel import ParallelLayout
    from ucm.integration.vllm.v2.ucm_lookup import UCMLookup
    from ucm.integration.vllm.v2.ucm_sharding import UCMSharding

    lookup = UCMLookup(
        parsed,
        UCMProxyAdapter(PipelineProxy(proxy or RecordingTestStore())),
        UCMSharding(parsed, parallel or ParallelLayout()),
        recompute_tokens=recompute_tokens,
    )
    return UCMDispatcher(
        parsed,
        lookup,
        hasher or make_request_hasher(vllm_config(), 0),
        b"seed",
        **kwargs,
    )


def ucm_ids(common_ucm_block_ids, attention_type=AttentionType.FA):
    """Single-rank local UCM block ids (record class 0, shard 0)."""
    return make_local_ucm_block_ids(common_ucm_block_ids, attention_type)


def request_ucm_ids(dispatcher, request):
    """Single-rank local UCM block ids of a request per attention type, plan
    order."""
    common_ucm_block_ids = make_common_ucm_block_ids(dispatcher._block_hasher(request))
    return tuple(
        ucm_ids(common_ucm_block_ids, attention_type)
        for attention_type, _ in dispatcher._groups_by_attention_type
    )


def seed_request_state(
    dispatcher,
    request,
    external_hit_tokens=0,
    common_ucm_block_ids=(),
    hbm_hit_tokens=0,
):
    """Record a synthetic lookup result the way UCMDispatcher.lookup would."""
    state = RequestState(
        hbm_hit_tokens=hbm_hit_tokens,
        external_hit_tokens=external_hit_tokens,
        num_token_ids=len(request.all_token_ids),
        token_processed=hbm_hit_tokens + external_hit_tokens,
        common_ucm_block_ids=tuple(common_ucm_block_ids),
        vllm_block_tables=tuple([] for _ in dispatcher.spec.groups),
        load_pending=external_hit_tokens > 0,
    )
    dispatcher.requests[str(request.request_id)] = state
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
        for key in t.ucm_block_ids
        for _ in range(t.ptrs.shape[1])
    )
    fields = {
        field: (
            np.concatenate([getattr(t, attr).reshape(-1) for t in transfers])
            if transfers
            else np.empty(0, dtype=np.uint64)
        )
        for field, attr in (
            ("offsets", "ucm_block_offsets"),
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
        json.dumps({"type": "FA", "block_index": int(i)}) for i in indices
    )
    return UCMProxyTransfer(keys, indices, ptrs, sizes, offsets, metadata, lengths)


def _submit_segments(adapter, operation, keys, offsets, ptrs, sizes):
    """Feed the existing byte fixtures through the native [K,S] adapter."""
    from ucm.integration.vllm.v2.ucm_proxy import UCMProxyTransfer

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
    group_layout, *, token_offsets=0, token_counts=None, token_block_size=None
):
    """Fixed token ranges of every segment, for address assertions.

    Uses the layout's own token-to-byte conversion and resolve_ptrs.
    GroupStoreLayout.build applies the same whole-block and state-alignment
    rules in production. The block span defaults to the first layer's
    native block size.
    """
    token_block_size = token_block_size or int(
        group_layout.layers[0].kv_cache_spec.block_size
    )
    token_starts = np.atleast_1d(np.asarray(token_offsets, dtype=np.int64))
    token_lengths = (
        token_block_size - token_starts
        if token_counts is None
        else np.atleast_1d(np.asarray(token_counts, dtype=np.int64))
    )
    token_starts, token_lengths = np.broadcast_arrays(token_starts, token_lengths)
    if (
        (token_starts < 0)
        | (token_lengths <= 0)
        | (token_starts + token_lengths > token_block_size)
    ).any():
        raise ValueError("Token range must lie inside one physical block")
    is_partial = ((token_starts != 0) | (token_lengths != token_block_size)).any()
    if is_partial and not group_layout.supports_partial_tokens:
        raise ValueError(
            f"KV cache group {group_layout.group_id} requires whole-block access"
        )
    for tokens in (token_starts, token_lengths):
        if (
            np.multiply.outer(tokens.astype(np.uint64), group_layout.states_per_block)
            % np.uint64(token_block_size)
        ).any():
            raise ValueError("Token ranges must align with stored states")
    return SimpleNamespace(
        layout=group_layout,
        token_block_size=token_block_size,
        block_byte_offsets=group_layout.tokens_to_bytes(
            token_starts.astype(np.uint64), token_block_size
        ),
        segment_bytes=group_layout.tokens_to_bytes(
            token_lengths.astype(np.uint64), token_block_size
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
            access.token_block_size,
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
    dispatcher,
    scheduled_tokens,
    *,
    preempted_req_ids=(),
    finished_req_ids=(),
    state_offloads=None,
):
    """Use the scheduler-output entry point for existing dispatch fixtures.

    state_offloads maps request id -> (group_id, vllm_block_id, boundary)
    hand-offs, the engine's boundary_state_offloads.
    """
    block_state = SimpleNamespace(
        req_ids=tuple(r for r in scheduled_tokens if r in dispatcher.requests),
        get_block_ids=lambda r: dispatcher.requests[r].vllm_block_tables,
        boundary_state_offloads=state_offloads or {},
    )
    output = SimpleNamespace(
        kv_connector_block_state=block_state,
        num_scheduled_tokens=scheduled_tokens,
        total_num_scheduled_tokens=sum(scheduled_tokens.values()),
        preempted_req_ids=set(preempted_req_ids),
        finished_req_ids=set(finished_req_ids),
    )
    return dispatcher.build_from_scheduler_output(output)


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

    def __init__(self, ptr, shape, strides, element_size=1):
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

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return math.prod(self.shape)

    def data_ptr(self):
        return self._ptr

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
        )

    def unbind(self, dim):
        assert dim == 0
        return tuple(
            FakeTensor(
                self._ptr + i * self._strides[0] * self._element_size,
                self.shape[1:],
                self._strides[1:],
                self._element_size,
            )
            for i in range(self.shape[0])
        )


class KVCacheSpecTest(unittest.TestCase):
    def test_direct_custom_chunk_keeps_base_hash_size(self):
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
            ucm_cache_block_size=512,
        )

        self.assertEqual(parsed.ucm_cache_block_size, 512)
        self.assertEqual(parsed.groups[0].tail_blocks, 4)

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
        with self.assertRaisesRegex(ValueError, "must divide UCM block size"):
            parse_spec(
                config(
                    group(["model.layers.0.attn"], FullAttentionSpec(128)),
                    group(["model.layers.1.mixer"], MambaSpec(256)),
                ),
                scheduler_block_size=128,
            )

    def test_parse_rejects_fa_slices_that_do_not_tile_the_block(self):
        # A UCM block smaller than an FA block must divide it.
        with self.assertRaisesRegex(ValueError, "must divide its block span"):
            parse_spec(
                config(
                    group(["model.layers.0.attn"], FullAttentionSpec(512)),
                    group(["model.layers.1.attn"], FullAttentionSpec(768)),
                ),
                scheduler_block_size=1536,
                ucm_cache_block_size=384,
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
                ucm_cache_block_size=64,
            )

    def test_parse_warns_that_bounded_replay_drops_finer_hits(self):
        window = AscendSlidingWindowMLASpec(128, 1, 128)
        window.prefix_replay_tokens = 128
        with self.assertLogs(
            "ucm.integration.vllm.v2.layout.kv_cache", level="WARNING"
        ) as logs:
            parse_spec(
                config(
                    group(["model.layers.0.attn"], MLAAttentionSpec(1024)),
                    group(["model.layers.1.swa_cache"], window),
                ),
                scheduler_block_size=1024,
                ucm_cache_block_size=512,
            )
        self.assertIn("bounded replay", "".join(logs.output))

    def test_groups_vllm_does_not_store_are_transient(self):
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
                self.assertTrue(scratch.is_transient)
                self.assertEqual(scratch.kinds, frozenset())
                self.assertEqual(scratch.tail_blocks, 0)

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
        self.assertEqual([g.tail_tokens for g in worker.spec.wa_groups], [128, 4, 0])
        for representative in fa_specs.values():
            with self.subTest(representative=representative.tokens_per_state):
                scheduler = UCMConnector(
                    runtime_config,
                    KVConnectorRole.SCHEDULER,
                    config(group(fa_specs, representative), *windows),
                )
                self.assertEqual(
                    [g.tail_tokens for g in scheduler.spec.wa_groups],
                    [128, 4, 0],
                )


class StateLoadTargetTest(unittest.TestCase):
    """State load slots follow the runner's precopy/load order, not the device."""

    def _state_load_blocks(self, need_load_to_running_state):
        from ucm.integration.vllm.v2.layout import (
            UCMKVCacheGroupInfo,
            UCMKVCacheSpec,
        )

        spec = UCMKVCacheSpec(
            groups=(
                UCMKVCacheGroupInfo(
                    0, 128, frozenset({KVCacheSpecKind.FULL_ATTENTION}), tail_blocks=1
                ),
                UCMKVCacheGroupInfo(
                    1, 128, frozenset({KVCacheSpecKind.MAMBA}), tail_blocks=1
                ),
            ),
            scheduler_block_size=128,
            ucm_cache_block_size=128,
            device_type="npu",
        )
        dispatcher = make_dispatcher(
            spec, need_load_to_running_state=need_load_to_running_state
        )
        request = FakeRequest("r", 1024)
        fa_keys = tuple(bytes([index]) * 16 for index in range(8))
        state_keys = tuple(bytes([index + 32]) * 16 for index in range(8))
        state = seed_request_state(
            dispatcher,
            request,
            external_hit_tokens=256,
            common_ucm_block_ids=fa_keys,
        )
        # Restore boundary 256 lives in block 11; this step's running block
        # for tokens [256, 384) is block 12.
        state.vllm_block_tables = ([1, 2, 3, 4], [0, 11, 12, 0])

        meta = dispatcher._request_meta("r", state, 128)
        state_plans = [
            p for p in meta.load_plans if p.attention_type is AttentionType.STATE
        ]
        for plan in state_plans:
            self.assertEqual(plan.common_ucm_block_ids, (fa_keys[1],))
        return [plan.vllm_block_ids_by_group[0].tolist() for plan in state_plans]

    def test_copy_after_load_restores_only_the_boundary_block(self):
        # Ascend V1 runner: its A->B copy runs after the connector load.
        self.assertEqual(self._state_load_blocks(False), [[11]])

    def test_precopy_before_load_restores_running_and_boundary_blocks(self):
        # CUDA V1 and both V2 runners copy A->B before the load.
        self.assertEqual(self._state_load_blocks(True), [[12], [11]])


class RequestAwareKeyTest(unittest.TestCase):
    """v2 keys fold vLLM's block-hash extra keys, not only token IDs."""

    def _fa_keys(self, request):
        from ucm.integration.vllm.v2.layout import (
            UCMKVCacheGroupInfo,
            UCMKVCacheSpec,
        )

        spec = UCMKVCacheSpec(
            groups=(
                UCMKVCacheGroupInfo(
                    0, 128, frozenset({KVCacheSpecKind.FULL_ATTENTION}), tail_blocks=1
                ),
            ),
            scheduler_block_size=128,
            ucm_cache_block_size=128,
            device_type="cuda",
        )
        (keys,) = request_ucm_ids(make_dispatcher(spec), request)
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
    """Each dumped UCM block carries JSON with its attention type and block index."""

    def test_block_index_counts_ucm_blocks_from_the_request_start(self):
        from ucm.integration.vllm.v2.ucm_kv_cache import dump_metadata
        from ucm.integration.vllm.v2.ucm_scheduler import UCMGroupDispatchPlan

        keys = tuple(bytes([index]) * 16 for index in range(3))
        fa = UCMGroupDispatchPlan(AttentionType.FA, keys, 256, 640, ())
        wa = UCMGroupDispatchPlan(AttentionType.WA, keys[:1], 512, 640, ())
        state = UCMGroupDispatchPlan(AttentionType.STATE, keys[:1], 896, 1024, ())

        self.assertEqual(
            [json.loads(item) for item in dump_metadata(fa, 128)],
            [
                {"type": "FA", "block_index": 2},
                {"type": "FA", "block_index": 3},
                {"type": "FA", "block_index": 4},
            ],
        )
        self.assertEqual(
            json.loads(dump_metadata(wa, 128)[0]), {"type": "WA", "block_index": 4}
        )
        self.assertEqual(
            json.loads(dump_metadata(state, 128)[0]),
            {"type": "State", "block_index": 7},
        )

    def test_adapter_passes_metadata_to_dump_only(self):
        from ucm.integration.vllm.v2.ucm_proxy import UCMProxyTransfer

        proxy = RecordingTestStore()
        adapter = UCMProxyAdapter(PipelineProxy(proxy))
        keys = (b"a" * 16, b"b" * 16)
        grid = np.zeros((2, 1), dtype=np.uint64)
        metadata = (
            '{"type":"FA","block_index":0}',
            '{"type":"FA","block_index":1}',
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
            parsed = UCMConnector(cfg, KVConnectorRole.SCHEDULER, raw).spec
        self.assertEqual(parsed.groups[1].token_block_size, 512)
        self.assertEqual(parsed.groups[3].tail_tokens, 0)
        names = [name for g in raw.kv_cache_groups for name in g.layer_names]
        layout = UCMKVCacheLayout(
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
        self.hasher = make_request_hasher(vllm_config(), 1)

    def test_hash_golden_vector(self):
        self.assertEqual(
            self.hasher((b"parent", (1, 2, 3, 4))).hex(),
            "e60666fdcd359c82d9a369fc06bb3270",
        )

    def test_ucm_block_id_layout_is_stable(self):
        # Golden encoding: common id (block hash[:12]) + type(2)
        # record_class(4) pp(5) head(6) token(7) reserved(8), big-endian.
        from ucm.integration.vllm.v2.ucm_block_id import ShardKey

        def suffix(attention_type, record_class=0, shard=ShardKey()):
            (ucm_block_id,) = make_local_ucm_block_ids(
                make_common_ucm_block_ids((b"h" * 16,)),
                attention_type,
                record_class,
                shard,
            )
            self.assertEqual(ucm_block_id[:12], b"h" * 12)
            return ucm_block_id[12:].hex()

        self.assertEqual(suffix(AttentionType.FA), "00000000")
        self.assertEqual(suffix(AttentionType.WA), "40000000")
        self.assertEqual(suffix(AttentionType.STATE), "80000000")
        self.assertEqual(suffix(AttentionType.FA, 1), "04000000")
        self.assertEqual(suffix(AttentionType.FA, 0, ShardKey(pp_key=1)), "00200000")
        self.assertEqual(suffix(AttentionType.FA, 0, ShardKey(head_key=1)), "00008000")
        self.assertEqual(suffix(AttentionType.FA, 0, ShardKey(token_key=1)), "00000100")
        self.assertEqual(
            suffix(
                AttentionType.STATE, 15, ShardKey(head_key=63, token_key=127, pp_key=31)
            ),
            "bfffff00",
        )

    def test_local_ids_share_the_common_id_across_classes_and_shards(self):
        from ucm.integration.vllm.v2.ucm_block_id import ShardKey

        common_ids = make_common_ucm_block_ids((b"a" * 16, b"b" * 16))
        self.assertEqual(common_ids, (b"a" * 12, b"b" * 12))
        logical = make_local_ucm_block_ids(common_ids, AttentionType.WA)
        variants = (
            make_local_ucm_block_ids(
                common_ids, AttentionType.WA, 0, ShardKey(head_key=1)
            ),
            make_local_ucm_block_ids(common_ids, AttentionType.WA, 1),
            make_local_ucm_block_ids(common_ids, AttentionType.FA),
        )
        for local_ids in (logical, *variants):
            self.assertEqual(tuple(local_id[:12] for local_id in local_ids), common_ids)
        for local_ids in variants:
            self.assertNotEqual(local_ids, logical)
        # Local ids are not common ids: they cannot be tagged again.
        with self.assertRaisesRegex(ValueError, "12 bytes"):
            make_local_ucm_block_ids(logical, AttentionType.WA)

    def test_unencodable_topology_fails_at_startup(self):
        from ucm.integration.vllm.v2.ucm_block_id import (
            ShardKey,
            validate_shard_key_encoding,
        )

        validate_shard_key_encoding(15, ShardKey(head_key=63, token_key=127, pp_key=31))
        for record_class, shard in (
            (16, ShardKey()),
            (0, ShardKey(head_key=64)),
            (0, ShardKey(token_key=128)),
            (0, ShardKey(pp_key=32)),
        ):
            with self.assertRaises(ValueError):
                validate_shard_key_encoding(record_class, shard)

    def test_writer_rows_split_blocks_once_and_evenly(self):
        import hashlib

        from ucm.integration.vllm.v2.ucm_block_id import writer_rows

        common_ids = make_common_ucm_block_ids(
            hashlib.md5(str(i).encode()).digest() for i in range(1000)
        )
        rows = [writer_rows(common_ids, index, 4) for index in range(4)]
        self.assertEqual(sorted(np.concatenate(rows).tolist()), list(range(1000)))
        for part in rows:
            self.assertGreater(len(part), 180)
        self.assertEqual(writer_rows(common_ids, 0, 1).tolist(), list(range(1000)))

    def test_attention_types_share_one_hash_and_differ_only_by_tag(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            scheduler_block_size=128,
        )
        coordinator = make_dispatcher(parsed, RecordingTestStore(), self.hasher)
        request = FakeRequest("r", 512)
        values = self.hasher.make_request_block_hasher(128, b"seed")(request)
        fa_keys, state_keys = request_ucm_ids(coordinator, request)

        self.assertEqual(fa_keys[0][:12], values[0][:12])
        self.assertEqual(state_keys[0][:12], values[0][:12])
        self.assertEqual(fa_keys[0][12:].hex(), "00000000")
        self.assertEqual(state_keys[0][12:].hex(), "80000000")
        self.assertNotEqual(fa_keys[0], state_keys[0])

    def test_custom_cache_block_hashes_at_the_cache_block(self):
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
            ucm_cache_block_size=512,
        )
        proxy = RecordingTestStore()
        coordinator = make_dispatcher(parsed, proxy, self.hasher, recompute_tokens=0)
        request = FakeRequest("r", 1024)
        (keys,) = request_ucm_ids(coordinator, request)
        proxy.present.update(keys)

        result = coordinator.lookup(request, 0)

        self.assertEqual(ucm_ids(result.common_ucm_block_ids), keys)
        self.assertEqual(result.external_hit_tokens, 1024)

    def test_direct_prefix_stops_at_first_miss(self):
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
        )
        proxy = RecordingTestStore()
        coordinator = make_dispatcher(parsed, proxy, self.hasher, recompute_tokens=0)
        request = FakeRequest("r", 512)
        (keys,) = request_ucm_ids(coordinator, request)
        proxy.present.update((keys[0], keys[2], keys[3]))

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 128)

    def test_direct_miss_never_moves_an_unaligned_hbm_boundary_backwards(self):
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
            ucm_cache_block_size=512,
        )
        proxy = RecordingTestStore()
        coordinator = make_dispatcher(
            parsed, proxy, make_request_hasher(vllm_config(), 0)
        )

        result = coordinator.lookup(FakeRequest("unaligned", 1024), 128)

        self.assertEqual(result.external_hit_tokens, 0)

    def test_direct_full_hit_leaves_the_recompute_margin_block(self):
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
        )
        proxy = RecordingTestStore()
        coordinator = make_dispatcher(parsed, proxy, self.hasher)
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
        coordinator = make_dispatcher(parsed, proxy, self.hasher, recompute_tokens=0)
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
        coordinator = make_dispatcher(parsed, proxy, self.hasher, recompute_tokens=0)
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
        coordinator = make_dispatcher(parsed, proxy, self.hasher)
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
        coordinator = make_dispatcher(
            parsed, RecordingTestStore(), self.hasher, recompute_tokens=0
        )

        result = coordinator.lookup(FakeRequest("r", 1024), 0)

        self.assertEqual(result.external_hit_tokens, 0)
        self.assertEqual(len(result.common_ucm_block_ids), 8)

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
        coordinator = make_dispatcher(parsed, proxy, self.hasher, recompute_tokens=0)
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
        coordinator = make_dispatcher(parsed, proxy, self.hasher)
        request = FakeRequest("r", 1536)
        fa_keys, wa_keys = request_ucm_ids(coordinator, request)
        proxy.present.update((*fa_keys, *wa_keys))

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 1024)


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
    @staticmethod
    def _spec(*groups, ucm_block_size=4):
        from ucm.integration.vllm.v2.layout import UCMKVCacheGroupInfo, UCMKVCacheSpec

        return UCMKVCacheSpec(
            tuple(
                UCMKVCacheGroupInfo(
                    group_id,
                    4,
                    frozenset((kind,)),
                    tail_blocks=1,
                    tp_shards=tp_shards,
                    dcp_sharded=dcp_sharded,
                )
                for group_id, (kind, tp_shards, dcp_sharded) in enumerate(groups)
            ),
            4,
            ucm_block_size,
            "cpu",
        )

    MLA = (KVCacheSpecKind.MLA_ATTENTION, 1, True)
    GQA = (KVCacheSpecKind.FULL_ATTENTION, None, True)
    STATE = (KVCacheSpecKind.MAMBA, None, False)
    REPLICATED_STATE = (KVCacheSpecKind.MAMBA, 1, False)

    def _layout(self, spec, **parallel):
        from ucm.integration.vllm.v2.parallel import ParallelLayout
        from ucm.integration.vllm.v2.ucm_sharding import UCMSharding

        layout = ParallelLayout(**parallel)
        sharding = UCMSharding(spec, layout)
        ranks = [
            (tp, pp, pcp)
            for pp in range(layout.pp)
            for pcp in range(layout.pcp)
            for tp in range(layout.tp)
        ]
        return sharding, ranks

    def test_replicas_follow_tp_dcp_pcp_and_state_splits(self):
        from ucm.integration.vllm.v2.ucm_block_id import ShardKey

        cases = (
            # MLA: one KV copy on every TP rank.
            ((self.MLA,), dict(tp=4), AttentionType.FA, [ShardKey()], 4),
            # GQA heads split across TP: no replicas.
            ((self.GQA,), dict(tp=4), AttentionType.FA, 4, 1),
            # PCP alone splits compute only: PCP ranks are replicas.
            ((self.GQA,), dict(tp=2, pcp=2), AttentionType.FA, 2, 2),
            # DCP inside TP splits MLA tokens.
            ((self.MLA,), dict(tp=2, dcp=2), AttentionType.FA, 2, 1),
            # DCP spanning PCP splits tokens across PCP ranks.
            ((self.MLA,), dict(tp=2, pcp=2, dcp=2), AttentionType.FA, 2, 2),
            # GLM-5.3 style: MLA replicated, KDA/Mamba state split by TP.
            ((self.MLA, self.STATE), dict(tp=2), AttentionType.STATE, 2, 1),
            # Replicated Mamba state: every TP rank holds the same state.
            ((self.REPLICATED_STATE,), dict(tp=2), AttentionType.STATE, 1, 2),
            # PP stages hold different layers.
            ((self.MLA,), dict(tp=2, pp=2), AttentionType.FA, 2, 2),
        )
        for groups, parallel, attention_type, shards, replicas in cases:
            with self.subTest(groups=groups, parallel=parallel):
                sharding, ranks = self._layout(self._spec(*groups), **parallel)
                (record_class,) = sharding.record_classes(attention_type)
                found = sharding.all_shard_keys(record_class)
                if isinstance(shards, int):
                    self.assertEqual(len(found), shards)
                else:
                    self.assertEqual(list(found), shards)
                slots = {
                    sharding.replica_index_and_count(record_class, rank)
                    for rank in ranks
                }
                self.assertEqual(slots, {(i, replicas) for i in range(replicas)})

    def test_head_shards_group_consecutive_tp_ranks_like_vllm(self):
        # vLLM replicates a KV head on consecutive TP ranks: QKVParallelLinear
        # loads head tp_rank // (tp / num_kv_heads). Two head shards on TP4:
        # ranks 0, 1 hold head 0 and ranks 2, 3 hold head 1.
        gqa_two_heads = (KVCacheSpecKind.FULL_ATTENTION, 2, True)
        sharding, ranks = self._layout(self._spec(gqa_two_heads), tp=4)
        (record_class,) = sharding.record_classes(AttentionType.FA)
        self.assertEqual(
            [sharding.shard_key(record_class, rank).head_key for rank in ranks],
            [0, 0, 1, 1],
        )
        self.assertEqual(
            [sharding.replica_index_and_count(record_class, rank) for rank in ranks],
            [(0, 2), (1, 2), (0, 2), (1, 2)],
        )

    def test_groups_splitting_differently_are_separate_records(self):
        # Under DCP the MLA cache splits tokens across DCP ranks, while a
        # draft running at DCP=1 (dcp_sharded=False) keeps every token.
        draft = (KVCacheSpecKind.FULL_ATTENTION, None, False)
        sharding, ranks = self._layout(self._spec(self.MLA, draft), tp=2, dcp=2)
        main, second = sharding.record_classes(AttentionType.FA)
        self.assertEqual((main.group_positions, second.group_positions), ((0,), (1,)))
        self.assertEqual(len(sharding.all_shard_keys(main)), 2)
        self.assertFalse(second.is_token_sharded)

    def test_dcp_scales_only_dcp_sharded_block_spans(self):
        # vLLM reports per-rank block sizes; only dcp_sharded caches span
        # dcp x block_size tokens (resolve_dcp_kv_block_size).
        from ucm.integration.vllm.v2.parallel import ParallelLayout

        spec = self._spec(
            self.MLA, (KVCacheSpecKind.FULL_ATTENTION, None, False), ucm_block_size=8
        )
        ParallelLayout(tp=2, dcp=2).apply_context_parallel(spec)
        self.assertEqual([g.token_block_size for g in spec.groups], [8, 4])
        self.assertEqual(spec.ucm_cache_block_size, 8)
        self.assertEqual([g.tail_blocks for g in spec.groups], [1, 2])

    def test_parser_reads_dcp_flags_from_specs(self):
        draft = FullAttentionSpec(4)
        draft.dcp_sharded = False
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], MLAAttentionSpec(4)),
                group(["draft.layers.0.attn"], draft),
            ),
            scheduler_block_size=4,
        )
        self.assertEqual([g.dcp_sharded for g in parsed.groups], [True, False])

    def test_parser_reads_mamba_tp_replication_from_specs(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
                group(["model.layers.2.conv"], MambaSpec(128, tp_replicated=True)),
            ),
            scheduler_block_size=128,
        )
        self.assertEqual(
            [(g.tp_shards, g.dcp_sharded) for g in parsed.groups],
            [(None, True), (None, False), (1, False)],
        )

    def test_transient_groups_have_no_kind(self):
        # KpoolTailSpec is a SlidingWindowSpec, but the pool is not stored,
        # so it is not a WA group and GLM can run with DCP.
        from ucm.integration.vllm.v2.parallel import ParallelLayout

        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], MLAAttentionSpec(4)),
                group(["model.layers.0.indexer.kpool_tail"], KpoolTailSpec(4, 4)),
            ),
            hf_text_config=TextConfig(model_type="glm5_next", index_kpool=4),
            scheduler_block_size=8,
        )
        kpool_group = parsed.groups[1]
        self.assertTrue(kpool_group.is_transient)
        self.assertEqual(kpool_group.kinds, frozenset())
        self.assertEqual(parsed.wa_groups, ())
        ParallelLayout(tp=2, dcp=2).apply_context_parallel(parsed)
        self.assertEqual(parsed.ucm_cache_block_size, 8)
        self.assertEqual(kpool_group.tail_blocks, 0)

    def test_empty_pp_sliding_group_reads_tail_from_all_stages(self):
        # vLLM keeps the map of all stages in an empty PP group.
        window = AscendSlidingWindowMLASpec(4, compress_ratio=1, sliding_window=8)
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], MLAAttentionSpec(4)),
                group([], {"model.layers.3.attn": window}),
            ),
            scheduler_block_size=4,
        )
        self.assertEqual(parsed.groups[1].tail_tokens, 8)

    def test_unencodable_topology_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "head shards"):
            self._layout(self._spec(self.GQA), tp=128)

    def test_lookup_requires_every_shard_and_replicas_read_one_record(self):
        from ucm.integration.vllm.v2.ucm_block_id import ShardKey
        from ucm.integration.vllm.v2.ucm_lookup import UCMLookup

        spec = self._spec(self.GQA)
        sharding, _ = self._layout(spec, tp=2)
        common_ids = (b"p" * 12,)
        ids = [
            make_local_ucm_block_ids(
                common_ids, AttentionType.FA, 0, ShardKey(head_key=tp)
            )
            for tp in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            memory = ByteMemory()
            memory.write(100, b"rankzero")
            memory.write(200, b"rank-one")
            store = FileTestStore(directory, MemoryByteAccess(memory))
            lookup = UCMLookup(spec, UCMProxyAdapter(PipelineProxy(store)), sharding)
            self.assertNotEqual(ids[0], ids[1])
            for tp in range(2):
                store.dump(ids[tp], (0,), (100 + tp * 100,), (8,))
            store.commit(ids[0])
            # Only shard 0 is committed: the block does not hit yet.
            self.assertEqual(
                lookup._prefix_hit_end_block(AttentionType.FA, common_ids, 0, 1), 0
            )
            self.assertEqual(
                lookup._common_boundary_hit_end_block(
                    [AttentionType.FA], common_ids, 0, 1
                ),
                0,
            )
            store.commit(ids[1])
            self.assertEqual(
                lookup._prefix_hit_end_block(AttentionType.FA, common_ids, 0, 1), 1
            )
            for tp in range(2):
                store.load(ids[tp], (0,), (300 + 100 * tp,), (8,))
            self.assertEqual(memory.read(300, 8), b"rankzero")
            self.assertEqual(memory.read(400, 8), b"rank-one")

    def test_prefix_hit_is_the_shortest_prefix_of_all_shards(self):
        from ucm.integration.vllm.v2.ucm_block_id import ShardKey
        from ucm.integration.vllm.v2.ucm_lookup import UCMLookup

        spec = self._spec(self.GQA)
        sharding, _ = self._layout(spec, tp=2)
        common_ids = tuple(bytes([index]) * 12 for index in range(4))
        shard0, shard1 = (
            make_local_ucm_block_ids(
                common_ids, AttentionType.FA, 0, ShardKey(head_key=head_key)
            )
            for head_key in range(2)
        )
        store = RecordingTestStore()
        lookup = UCMLookup(spec, UCMProxyAdapter(PipelineProxy(store)), sharding)
        # Shard 0 hits blocks 0-2; shard 1 hits 0, 1 and 3. Block 3 alone does
        # not extend the prefix.
        store.present.update(shard0[:3] + shard1[:2] + shard1[3:])
        self.assertEqual(
            lookup._prefix_hit_end_block(AttentionType.FA, common_ids, 0, 4), 2
        )
        # Shard 1 only scans up to shard 0's prefix.
        self.assertEqual([len(keys) for keys in store.lookup_calls], [4, 3])

        # A shard with no hit ends the scan.
        store.present.clear()
        store.present.update(shard1)
        store.lookup_calls.clear()
        self.assertEqual(
            lookup._prefix_hit_end_block(AttentionType.FA, common_ids, 0, 4), 0
        )
        self.assertEqual(len(store.lookup_calls), 1)

    def test_boundary_hit_is_the_latest_block_every_shard_has(self):
        from ucm.integration.vllm.v2.ucm_block_id import ShardKey
        from ucm.integration.vllm.v2.ucm_lookup import UCMLookup

        spec = self._spec(self.STATE)
        sharding, _ = self._layout(spec, tp=2)
        common_ids = tuple(bytes([index]) * 12 for index in range(4))
        shard0, shard1 = (
            make_local_ucm_block_ids(
                common_ids, AttentionType.STATE, 0, ShardKey(head_key=head_key)
            )
            for head_key in range(2)
        )
        store = RecordingTestStore()
        lookup = UCMLookup(
            spec, UCMProxyAdapter(PipelineProxy(store)), sharding, recompute_tokens=0
        )
        # Shard 0 has states at blocks 1 and 3, shard 1 at blocks 1 and 2.
        # Both have block 1, so the restore ends after it.
        store.present.update((shard0[1], shard0[3], shard1[1], shard1[2]))
        self.assertEqual(lookup.hit_tokens(common_ids, 16, 0), 8)
        # Every scan is a reverse scan below the current restore end.
        self.assertEqual([len(keys) for keys in store.lookup_calls], [4, 4, 3, 2, 2, 2])

    def test_empty_pp_shard_marker_is_not_visible_before_commit(self):
        from ucm.integration.vllm.v2.ucm_block_id import ShardKey
        from ucm.integration.vllm.v2.ucm_lookup import UCMLookup

        spec = self._spec(self.MLA)
        sharding, _ = self._layout(spec, pp=2)
        common_ids = (b"s" * 12,)
        ids = [
            make_local_ucm_block_ids(
                common_ids, AttentionType.FA, 0, ShardKey(pp_key=pp)
            )
            for pp in range(2)
        ]
        empty = np.empty((1, 0), dtype=np.uint64)
        with tempfile.TemporaryDirectory() as directory:
            store = FileTestStore(directory, MemoryByteAccess(ByteMemory()))
            lookup = UCMLookup(spec, UCMProxyAdapter(PipelineProxy(store)), sharding)
            store.dump(ids[0], empty, empty, empty)
            store.dump(ids[1], np.array([[0]]), np.array([[10]]), np.array([[4]]))
            store.commit(ids[1])
            self.assertEqual(
                lookup._prefix_hit_end_block(AttentionType.FA, common_ids, 0, 1), 0
            )
            store.commit(ids[0])
            self.assertEqual(
                lookup._prefix_hit_end_block(AttentionType.FA, common_ids, 0, 1), 1
            )
            self.assertEqual(store._path(ids[0][0]).stat().st_size, 0)

    def test_replicas_share_ids_and_write_each_block_once(self):
        import hashlib

        from ucm.integration.vllm.v2.parallel import ParallelLayout
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )
        from ucm.integration.vllm.v2.ucm_sharding import UCMSharding

        raw_config = config(group(["model.layers.0.attn"], MLAAttentionSpec(4)))
        parsed = parse_spec(raw_config, scheduler_block_size=4, device_type="cpu")
        layout = UCMKVCacheLayout(
            parsed,
            {"model.layers.0.attn": FakeTensor(0x1000, (8, 4, 2), (8, 2, 1))},
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        sharding = UCMSharding(parsed, ParallelLayout(tp=2))
        common_ids = make_common_ucm_block_ids(
            hashlib.md5(bytes([i])).digest() for i in range(8)
        )
        plan = UCMGroupDispatchPlan(
            AttentionType.FA, common_ids, 0, 32, (np.arange(8, dtype=np.uint64),)
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", (plan,), (plan,))}
        )
        loads, dumps = [], []
        for tp in range(2):
            worker = UCMConnector.__new__(UCMConnector)
            (record_class,) = sharding.record_classes(AttentionType.FA)
            worker._replica_index_and_count_by_class = {
                record_class: sharding.replica_index_and_count(record_class, (tp, 0, 0))
            }
            builder = UCMTransferBuilder(layout, sharding, (tp, 0, 0))
            (load,) = builder.build_load_transfers(metadata)
            (dump,) = builder.build_dump_transfers(
                metadata, row_selector=worker._owned_rows
            )
            loads.append(load.ucm_block_ids)
            dumps.append(dict(zip(dump.ucm_block_ids, dump.block_indices.tolist())))
        # Replicas read the same full record set...
        self.assertEqual(loads[0], loads[1])
        self.assertEqual(len(loads[0]), 8)
        # ...and split the writes: every block written by exactly one replica.
        self.assertFalse(dumps[0].keys() & dumps[1].keys())
        self.assertEqual(dumps[0].keys() | dumps[1].keys(), set(loads[0]))
        for written in dumps:
            for ucm_block_id, block_index in written.items():
                self.assertEqual(loads[0][block_index], ucm_block_id)

    def test_topologies_do_not_share_a_namespace(self):
        from ucm.integration.vllm.v2.parallel import ParallelLayout

        variants = [
            ParallelLayout(tp=2),
            ParallelLayout(pp=2),
            ParallelLayout(pcp=2),
            ParallelLayout(tp=2, dcp=2),
            ParallelLayout(tp=2, dcp=2, interleave=128),
            ParallelLayout(pp=2, pp_partition="1,3"),
        ]
        self.assertEqual(len({p.namespace for p in variants}), len(variants))

    def test_cp_scales_blocks_by_dcp_only(self):
        from ucm.integration.vllm.v2.parallel import ParallelLayout

        def spec(scheduler_block_size=4):
            return parse_spec(
                config(group(["model.layers.0.attn"], FullAttentionSpec(4))),
                scheduler_block_size=scheduler_block_size,
            )

        # PCP splits prefill compute only: PCP ranks hold whole blocks.
        pcp_only = spec()
        ParallelLayout(tp=4, pcp=2).apply_context_parallel(pcp_only)
        self.assertEqual(pcp_only, spec())
        parallel = ParallelLayout(tp=4, pp=2, pcp=2, dcp=2)
        self.assertEqual(parallel.world_size, 16)
        # vLLM's scheduler block already counts DCP.
        scaled = spec(8)
        parallel.apply_context_parallel(scaled)
        self.assertEqual(scaled.groups[0].token_block_size, 8)
        self.assertEqual(scaled.ucm_cache_block_size, 8)
        self.assertEqual(scaled.groups[0].tail_blocks, 1)
        # A UCM block that cannot hold a whole DCP block is rejected and
        # the spec is left untouched.
        rejected = spec()
        with self.assertRaisesRegex(ValueError, "whole distributed"):
            parallel.apply_context_parallel(rejected)
        self.assertEqual(rejected, spec())

    def test_dcp_rejects_fa_slices(self):
        from ucm.integration.vllm.v2.parallel import ParallelLayout

        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(8))),
            scheduler_block_size=16,
            ucm_cache_block_size=4,
        )
        with self.assertRaisesRegex(ValueError, "whole distributed"):
            ParallelLayout(tp=2, dcp=2).apply_context_parallel(parsed)

    def test_dcp_sizes_follow_vllm_rules(self):
        from ucm.integration.vllm.v2.parallel import ParallelLayout

        def layout(tp, pcp, dcp):
            parallel_config = SimpleNamespace(
                tensor_parallel_size=tp,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=pcp,
                decode_context_parallel_size=dcp,
                cp_kv_cache_interleave_size=1,
            )
            return ParallelLayout.from_config(
                SimpleNamespace(parallel_config=parallel_config)
            )

        # Without PCP, DCP is a TP subgroup; with PCP it is off or spans PCP.
        for tp, pcp, dcp in ((4, 1, 2), (4, 2, 1), (4, 2, 2), (4, 2, 8)):
            with self.subTest(tp=tp, pcp=pcp, dcp=dcp):
                self.assertEqual(layout(tp, pcp, dcp).dcp, dcp)
        for tp, pcp, dcp in ((4, 1, 3), (4, 2, 4)):
            with self.subTest(tp=tp, pcp=pcp, dcp=dcp):
                with self.assertRaisesRegex(ValueError, "Unsupported DCP"):
                    layout(tp, pcp, dcp)

    def test_ascend_dcp_indexer_replication_folds_adjacent_kernel_rows(self):
        from ucm.integration.vllm.v2.parallel import ParallelLayout

        native_spec = AscendSFAIndexerCacheSpec(4, sfa_dcp_replicated_indexer_size=2)
        raw_config = config(
            group(["model.layers.0.indexer"], native_spec), num_blocks=3
        )
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=8,
            device_type="npu",
        )
        ParallelLayout(tp=2, dcp=2).apply_context_parallel(parsed)
        layout = UCMKVCacheLayout(
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
            token_block_size=8,
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
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (b"p" * 12,),
            0,
            4,
            (np.array([1], dtype=np.uint64), np.array([2], dtype=np.uint64)),
        )
        meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
        )
        (transfer,) = UCMTransferBuilder(layout).build_dump_transfers(meta)
        self.assertEqual(transfer.ptrs.tolist(), [[1016]])
        self.assertEqual(transfer.ucm_block_offsets.tolist(), [[0]])

    def test_pp_empty_sliding_group_uses_retained_global_layer_specs(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(4)),
                group(
                    [],
                    {"model.layers.9.swa_cache": AscendSlidingWindowMLASpec(4, 1, 8)},
                ),
            ),
            scheduler_block_size=4,
        )
        self.assertEqual(parsed.wa_groups[0].tail_tokens, 8)
        self.assertEqual(parsed.wa_groups[0].tail_blocks, 2)

    def test_pp_empty_state_spec_can_be_parsed(self):
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(4)),
                group([], {"model.layers.1.state": MambaSpec(4)}),
            ),
            scheduler_block_size=4,
        )
        self.assertEqual(len(parsed.state_groups), 1)
        self.assertEqual(parsed.state_groups[0].tail_blocks, 1)


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

                def commit(self, keys):
                    self.events.append(("commit", keys))
                    backend.commit(keys)

                def abort(self, keys):
                    self.events.append(("abort", keys))
                    backend.abort(keys)

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
        cfg.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
        with mock.patch(
            "ucm.integration.vllm.v2.factory.FileTestStore",
            return_value=proxy,
        ):
            worker = UCMConnector(cfg, KVConnectorRole.WORKER, config(*groups))
        worker.register_kv_caches(caches)
        return worker, backend, proxy, memory, names

    def _meta(self, *, load=False, state=False, token_start=0):
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        attention_types = (
            (AttentionType.FA, AttentionType.STATE) if state else (AttentionType.FA,)
        )
        plans = tuple(
            UCMGroupDispatchPlan(
                attention_type,
                (bytes([i + 1]) * 12,),
                token_start,
                token_start + 4,
                (np.array([2 if load else 1], dtype=np.uint64),),
            )
            for i, attention_type in enumerate(attention_types)
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
            worker.bind_connector_metadata(self._meta())
            worker.start_load_kv(None)
            worker.save_kv_layer(names[0], None, None)
            worker.save_kv_layer(names[0], None, None)
            worker.save_kv_layer(names[1], None, None)
            self.assertEqual([e[0] for e in proxy.events], ["dump", "dump"])
            self.assertEqual(backend.lookup((logical_id(1),)), (False,))
            worker.wait_for_save()
            self.assertEqual(
                [e[0] for e in proxy.events], ["dump", "dump", "wait", "wait", "commit"]
            )
            self.assertEqual(
                backend._path(logical_id(1)).read_bytes(), b"abcdefghijklmnop"
            )
            count = len(proxy.events)
            worker.wait_for_save()
            self.assertEqual(len(proxy.events), count)
            # New step, same worker; only each selected layer becomes ready at wait.
            worker.bind_connector_metadata(self._meta(load=True))
            worker.start_load_kv(None)
            self.assertEqual(memory.read(0x1010, 8), bytes(8))
            worker.wait_for_layer_load(names[0])
            self.assertEqual(memory.read(0x1010, 8), b"abcdefgh")
            self.assertEqual(memory.read(0x2010, 8), bytes(8))
            worker.wait_for_layer_load(names[0])
            worker.wait_for_layer_load(names[1])
            self.assertEqual(memory.read(0x2010, 8), b"ijklmnop")
            worker.wait_for_save()

    def test_no_hook_state_is_saved_at_end_and_loaded_before_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            worker, backend, proxy, memory, names = self._worker(directory, state=True)
            for ptr, payload in (
                (0x1008, b"abcdefgh"),
                (0x2008, b"ijklmnop"),
                (0x3008, b"state123"),
            ):
                memory.write(ptr, payload)
            worker.bind_connector_metadata(self._meta(state=True))
            worker.start_load_kv(None)
            worker.save_kv_layer(names[0], None, None)
            # Missing attention hook is also covered at end of forward.
            worker.wait_for_save()
            self.assertEqual(
                backend.lookup((logical_id(1), logical_id(2, AttentionType.STATE))),
                (True, True),
            )
            worker.bind_connector_metadata(self._meta(load=True, state=True))
            worker.start_load_kv(None)
            self.assertEqual(memory.read(0x3010, 8), b"state123")
            worker.wait_for_save()
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
                    worker.bind_connector_metadata(self._meta(token_start=token_start))
                    worker.start_load_kv(None)
                    for name in reversed(names):
                        worker.save_kv_layer(name, None, None)
                    worker.wait_for_save()
                    worker.bind_connector_metadata(
                        self._meta(load=True, token_start=token_start)
                    )
                    worker.start_load_kv(None)
                    for name in names:
                        worker.wait_for_layer_load(name)
                    worker.wait_for_save()
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
            from ucm.integration.vllm.v2.ucm_proxy import UCMProxyTransfer

            return (
                make_transfer(
                    (b"a" * 16,),
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
            len(common._v2_segment_payload(b"a" * 16, np.uint64(4), np.uint64(8))), 8
        )

    def test_failed_write_drains_other_tasks_without_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            worker, backend, proxy, memory, names = self._worker(
                directory, deferred=True
            )
            worker.bind_connector_metadata(self._meta())
            worker.start_load_kv(None)
            for name in names:
                worker.save_kv_layer(name, None, None)
            proxy.fail = "dump"
            with self.assertRaisesRegex(UCMProxyError, "dump failed"):
                worker.wait_for_save()
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
            worker.bind_connector_metadata(metadata)
            worker.start_load_kv(None)
            proxy.fail = "load"
            worker.wait_for_layer_load(names[0])
            worker.save_kv_layer(names[0], None, None)
            worker.wait_for_save()
            self.assertEqual(worker.get_block_ids_with_load_errors(), {2})
            self.assertEqual(
                worker.build_connector_worker_meta().load_failed_reqs, {"r"}
            )
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
            adapter.commit((key,), np.arange(len((key,)), dtype=np.int64))
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
            adapter.commit((key,), np.arange(len((key,)), dtype=np.int64))
            self.assertEqual(
                adapter.lookup(
                    (key, other), np.arange(len((key, other)), dtype=np.int64)
                ),
                (True, False),
            )
            self.assertEqual(proxy._path(key).read_bytes(), b"abcdefgh")
            adapter.commit(
                (key,), np.arange(len((key,)), dtype=np.int64)
            )  # Idempotent retry.
            _submit_segments(adapter, "load", (key,), (0,), (12288,), (8,))
            self.assertEqual(memory.read(0x3000, 8), b"abcdefgh")
            # A replacement remains hidden behind the previous complete block.
            _submit_segments(adapter, "dump", (key,), (0,), (8192,), (4,))
            self.assertEqual(proxy._path(key).read_bytes(), b"abcdefgh")
            adapter.commit((key, other), np.arange(len((key, other)), dtype=np.int64))
            self.assertEqual(proxy._path(key).read_bytes(), b"efgh")
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_wait_for_save_commits_after_all_dumps_and_not_on_dump_failure(self):
        connector = UCMConnector.__new__(UCMConnector)
        connector.use_layerwise = False
        connector._save_complete = False
        connector._failed_load_reqs = set()
        connector.spec = SimpleNamespace(fa_groups=(), wa_groups=(), state_groups=())
        connector.has_connector_metadata = lambda: True
        connector._get_connector_metadata = lambda: UCMConnectorMetadata()
        batches = (
            SimpleNamespace(ucm_block_ids=(b"a" * 16,), block_indices=np.array([0])),
            SimpleNamespace(ucm_block_ids=(b"b" * 16,), block_indices=np.array([1])),
        )
        connector.layout = SimpleNamespace(spec=connector.spec, group_layouts={})
        connector.transfer_builder = SimpleNamespace(
            build_dump_transfers=lambda _, **kwargs: batches,
            build_empty_markers=lambda *_: (),
            dump_targets=lambda *_: (),
        )
        connector._proxy = mock.Mock()
        connector.wait_for_save()
        self.assertEqual(
            connector._proxy.mock_calls,
            [
                mock.call.submit("dump", batches[0]),
                mock.call.submit("dump", batches[1]),
                mock.call.commit(batches[0].ucm_block_ids, batches[0].block_indices),
                mock.call.commit(batches[1].ucm_block_ids, batches[1].block_indices),
            ],
        )
        connector._proxy.reset_mock()
        connector._save_complete = False
        connector._proxy.submit.side_effect = [None, UCMProxyError("write failed")]
        with self.assertRaises(UCMProxyError):
            connector.wait_for_save()
        connector._proxy.commit.assert_not_called()

    def test_commit_waits_and_normalizes_failure(self):
        proxy = mock.Mock()
        proxy.wait = mock.Mock()
        proxy.commit.return_value = "commit-task"
        adapter = UCMProxyAdapter(PipelineProxy(proxy))
        adapter.commit((b"a" * 16,), np.arange(len((b"a" * 16,)), dtype=np.int64))
        proxy.wait.assert_called_once_with("commit-task")
        proxy.commit.side_effect = OSError("rename failed")
        with self.assertRaisesRegex(UCMProxyError, "Proxy commit failed"):
            adapter.commit((b"a" * 16,), np.arange(len((b"a" * 16,)), dtype=np.int64))

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
            def lookup_on_prefix(self, keys, block_indices):
                return len(keys)

        with self.assertRaisesRegex(UCMProxyError, "lookup_on_prefix"):
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

        with self.assertRaisesRegex(UCMProxyError, "does not provide wait"):
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
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (b"x" * 12,),
            0,
            4,
            (
                np.array(
                    [
                        5,
                    ],
                    dtype=np.uint64,
                ),
            ),
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
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        def request_meta(request_id, key, block_id):
            plan = UCMGroupDispatchPlan(
                AttentionType.FA,
                (key,),
                0,
                4,
                (
                    np.array(
                        [
                            block_id,
                        ],
                        dtype=np.uint64,
                    ),
                ),
            )
            return RequestDispatchMeta(request_id, (plan,))

        connector.bind_connector_metadata(
            UCMConnectorMetadata(
                requests={
                    "failed": request_meta("failed", b"x" * 12, 5),
                    "loaded": request_meta("loaded", b"y" * 12, 6),
                }
            )
        )

        connector.start_load_kv(None)

        self.assertEqual(
            connector.build_connector_worker_meta().load_failed_reqs, {"failed"}
        )
        self.assertEqual(connector.get_block_ids_with_load_errors(), {5})
        self.assertEqual(len(proxy.load_calls), 1)


class DispatcherLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
        )
        self.dispatcher = make_dispatcher(self.parsed)

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
        keys = tuple(bytes([index]) * 16 for index in range(4))
        return seed_request_state(self.dispatcher, request, common_ucm_block_ids=keys)

    def test_block_state_tables_replace_previous_tables_each_step(self):
        state = self.add_state()
        state.vllm_block_tables = ([1, 2, 3],)
        for table in ([3], [3, 4], [7, 6, 8]):
            self.dispatcher.build_from_scheduler_output(
                self.scheduler_output(tables={"r": (table,)}, scheduled={"r": 128})
            )
            self.assertEqual(state.vllm_block_tables, (table,))

    def test_preempt_and_finish_remove_request_snapshots(self):
        self.add_state("preempted")
        self.add_state("finished")

        metadata = self.dispatcher.build_from_scheduler_output(
            self.scheduler_output(
                preempted=("preempted",),
                finished=("finished",),
            )
        )

        self.assertEqual(metadata.preempted_req_ids, {"preempted"})
        self.assertEqual(metadata.finished_req_ids, {"finished"})
        self.assertEqual(self.dispatcher.requests, {})

    def test_direct_custom_chunk_completes_once_across_steps(self):
        parsed = parse_spec(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
            ucm_cache_block_size=512,
        )
        dispatcher = make_dispatcher(parsed)
        request = FakeRequest("chunked", 1024)
        keys = (b"a" * 16, b"b" * 16)
        state = seed_request_state(dispatcher, request, common_ucm_block_ids=keys)
        state.vllm_block_tables = ([1, 2, 3, 4, 5, 6, 7, 8],)

        first = _scheduler_metadata(dispatcher, {"chunked": 128})
        second = _scheduler_metadata(dispatcher, {"chunked": 384})
        third = _scheduler_metadata(dispatcher, {"chunked": 512})

        self.assertEqual(first.requests["chunked"].dump_plans, ())
        second_plan = second.requests["chunked"].dump_plans[0]
        self.assertEqual(
            (
                second_plan.common_ucm_block_ids,
                second_plan.token_start,
                second_plan.token_end,
            ),
            ((keys[0],), 0, 512),
        )
        self.assertEqual(
            second_plan.vllm_block_ids_by_group[0].tolist(),
            [1, 2, 3, 4],
        )
        third_plan = third.requests["chunked"].dump_plans[0]
        self.assertEqual(
            (
                third_plan.common_ucm_block_ids,
                third_plan.token_start,
                third_plan.token_end,
            ),
            ((keys[1],), 512, 1024),
        )
        self.assertEqual(third_plan.vllm_block_ids_by_group[0].tolist(), [5, 6, 7, 8])

    def test_external_load_plan_is_consumed_after_first_scheduled_step(self):
        state = self.add_state()
        state.external_hit_tokens = 128
        state.token_processed = 128
        state.load_pending = True
        state.vllm_block_tables = ([3, 4, 5, 6],)

        first = _scheduler_metadata(self.dispatcher, {"r": 128})
        second = _scheduler_metadata(self.dispatcher, {"r": 128})

        self.assertEqual(len(first.requests["r"].load_plans), 1)
        self.assertEqual(first.requests["r"].load_plans[0].token_start, 0)
        self.assertEqual(first.requests["r"].load_plans[0].token_end, 128)
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
        dispatcher = make_dispatcher(parsed)
        request = FakeRequest("r", 2048)
        fa_keys = tuple(bytes([index]) * 16 for index in range(16))
        state_keys = tuple(bytes([index + 32]) * 16 for index in range(16))
        state = seed_request_state(
            dispatcher,
            request,
            hbm_hit_tokens=128,
            common_ucm_block_ids=fa_keys,
        )
        state.vllm_block_tables = (list(range(16)), list(range(16)))

        def state_plans(scheduled, offloads):
            metadata = _scheduler_metadata(
                dispatcher, {"r": scheduled}, state_offloads={"r": offloads}
            )
            return [
                plan
                for plan in metadata.requests["r"].dump_plans
                if plan.attention_type is AttentionType.STATE
            ]

        # [128, 1000): 896 is the newest boundary; 128 is the matched prefix.
        (plan,) = state_plans(872, [(1, 0, 128), (1, 6, 896)])
        self.assertEqual(plan.common_ucm_block_ids, (fa_keys[6],))
        self.assertEqual((plan.token_start, plan.token_end), (768, 896))
        self.assertEqual(plan.vllm_block_ids_by_group[0].tolist(), [6])

        # [1000, 1100) straddles 1024; a mid-block offer is not a boundary.
        (plan,) = state_plans(100, [(1, 7, 1024), (1, 9, 1000)])
        self.assertEqual(plan.common_ucm_block_ids, (fa_keys[7],))
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
        dispatcher = make_dispatcher(parsed)
        request = FakeRequest("dsv4", 1024)
        fa_keys = (b"a" * 16, b"b" * 16)
        wa_keys = (b"c" * 16, b"d" * 16)
        state = seed_request_state(dispatcher, request, 1024, fa_keys)
        state.vllm_block_tables = (
            [1, 2],
            list(range(8)),
            list(range(128)),
            list(range(32)),
        )

        metadata = _scheduler_metadata(dispatcher, {"dsv4": 1})
        fa_plan = next(
            plan
            for plan in metadata.requests["dsv4"].load_plans
            if plan.attention_type is AttentionType.FA
        )
        wa_plan = next(
            plan
            for plan in metadata.requests["dsv4"].load_plans
            if plan.attention_type is AttentionType.WA
        )

        self.assertEqual(len(fa_plan.vllm_block_ids_by_group), 1)
        self.assertEqual(wa_plan.common_ucm_block_ids, (fa_keys[1],))
        # Windows carry only block ids; the static shapes live in the spec
        # (swa: one whole 128-token block; C4 compressor: the [4, 8)
        # half-block sub-span) and the worker derives them from templates.
        self.assertEqual(
            tuple(blocks.tolist() for blocks in wa_plan.vllm_block_ids_by_group),
            ([7], [127]),
        )

    def test_wa_dump_skips_incomplete_tail_windows(self):
        # A tail window longer than the cache block clamps at the request
        # head on early boundaries; those partial windows are the
        # request's own prefix (restored with it), so nothing is dumped
        # until the boundary outgrows the tail.
        parsed = parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(
                    ["model.layers.1.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 512),
                ),
            ),
            scheduler_block_size=128,
        )
        dispatcher = make_dispatcher(parsed)
        request = FakeRequest("r", 1024)
        fa_keys = tuple(bytes([index]) * 16 for index in range(8))
        wa_keys = tuple(bytes([index + 32]) * 16 for index in range(8))
        seed_request_state(dispatcher, request, common_ucm_block_ids=fa_keys)
        dispatcher.requests["r"].vllm_block_tables = (
            list(range(8)),
            list(range(8)),
        )

        # [0, 384): every completed boundary (128..384) is shorter than
        # the 512-token tail -- no WA record.
        early = _scheduler_metadata(dispatcher, {"r": 384})
        self.assertTrue(
            any(
                plan.attention_type is AttentionType.FA
                for plan in early.requests["r"].dump_plans
            )
        )
        self.assertFalse(
            any(
                plan.attention_type is AttentionType.WA
                for plan in early.requests["r"].dump_plans
            )
        )

        # [384, 640): the newest boundary 640 outgrows the tail -- the
        # full window [128, 640) is one WA record at that boundary.
        later = _scheduler_metadata(dispatcher, {"r": 256})
        wa_plan = next(
            plan
            for plan in later.requests["r"].dump_plans
            if plan.attention_type is AttentionType.WA
        )
        self.assertEqual(wa_plan.common_ucm_block_ids, (fa_keys[4],))
        self.assertEqual(wa_plan.vllm_block_ids_by_group[0].tolist(), [1, 2, 3, 4])

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
        dispatcher = make_dispatcher(parsed)
        request = FakeRequest("r", 512)
        fa_keys = tuple(bytes([index]) * 12 for index in range(4))
        wa_keys = tuple(bytes([index + 32]) * 12 for index in range(4))
        seed_request_state(dispatcher, request, common_ucm_block_ids=fa_keys)
        dispatcher.requests["r"].vllm_block_tables = (
            list(range(4)),
            list(range(8)),
        )

        metadata = _scheduler_metadata(dispatcher, {"r": 384})
        wa_plan = next(
            plan
            for plan in metadata.requests["r"].dump_plans
            if plan.attention_type is AttentionType.WA
        )
        # Window [288, 384): the head half of block 4 plus all of block 5.
        self.assertEqual(wa_plan.common_ucm_block_ids, (fa_keys[2],))
        self.assertEqual(wa_plan.vllm_block_ids_by_group[0].tolist(), [4, 5])

        layout = UCMKVCacheLayout(
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
            ucm_cache_block_size=512,
        )
        dispatcher = make_dispatcher(parsed)
        request = FakeRequest("r", 2048)
        c4_keys = tuple(bytes([index]) * 16 for index in range(4))
        c128_keys = tuple(bytes([index + 16]) * 16 for index in range(4))
        seed_request_state(dispatcher, request, common_ucm_block_ids=c4_keys)
        dispatcher.requests["r"].vllm_block_tables = ([1, 2, 3, 4], [5, 6])

        metadata = _scheduler_metadata(dispatcher, {"r": 2048})
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
        dispatcher = make_dispatcher(parsed)
        keys = tuple(bytes([index % 256]) * 16 for index in range(8192))
        state = seed_request_state(
            dispatcher,
            FakeRequest("r", 8192 * 128),
            common_ucm_block_ids=keys,
        )
        state.vllm_block_tables = (list(range(8192)),)

        metadata = _scheduler_metadata(dispatcher, {"r": 8192 * 128})
        plan = metadata.requests["r"].dump_plans[0]

        self.assertEqual(len(plan.common_ucm_block_ids), 8192)
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
        layout = UCMKVCacheLayout(
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
        self.assertEqual(parsed.groups[0].token_block_size, 256)
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
        layout = UCMKVCacheLayout(
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
                int(layout.store_layouts[group_layout.group_id].group_block_offsets[0]),
            ),
            (0x1000, 64, 16, 0),
        )

        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"m" * 12
        plan = UCMGroupDispatchPlan(
            AttentionType.STATE,
            (key,),
            0,
            4,
            (
                np.array(
                    [
                        3,
                    ],
                    dtype=np.uint64,
                ),
            ),
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

        layout = UCMKVCacheLayout(
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
            UCMKVCacheLayout(
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
        )
        layout = UCMKVCacheLayout(
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

        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"p" * 12
        # A WA tail shorter than two blocks: the [64, 128) half-block head
        # of block 2 plus the whole block 3.  Only the ids travel; the
        # static tail pattern places the sub-spans.
        plan = UCMGroupDispatchPlan(
            AttentionType.WA,
            (key,),
            0,
            192,
            (np.array([2, 3], dtype=np.uint64),),
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
        self.assertEqual(batch.block_ids, ucm_ids((key, key), AttentionType.WA))
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
        )
        layout = UCMKVCacheLayout(
            parsed,
            {"model.layers.0.swa_cache": FakeTensor(0x1000, (8, 128, 1), (128, 1, 1))},
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"p" * 12

        def batch(phase, block_ids):
            plan = UCMGroupDispatchPlan(
                AttentionType.WA,
                (key,),
                0,
                192,
                (np.array(block_ids, dtype=np.uint64),),
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
            proxy.commit(dumped.block_ids)
            self.assertEqual(
                (
                    Path(directory)
                    / (ucm_ids((key,), AttentionType.WA)[0].hex() + ".ucm")
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
        layout = UCMKVCacheLayout(
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

        from ucm.integration.vllm.v2.ucm_proxy import UCMProxyAdapter
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key_a, key_b = b"d" * 12, b"e" * 12
        adapter = UCMProxyAdapter(PipelineProxy(MemoryFixtureStore(memory)))
        dump_plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (key_a, key_b),
            0,
            16,
            (np.array([3, 4], dtype=np.uint64),),
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
            AttentionType.FA,
            (key_a, key_b),
            0,
            16,
            (np.array([6, 2], dtype=np.uint64),),
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
        ucm_block_ids = tuple(dict.fromkeys(dump_batch.block_ids))
        adapter.commit(ucm_block_ids, np.arange(len(ucm_block_ids), dtype=np.int64))
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
        layout = UCMKVCacheLayout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        key = b"k" * 12
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (key,),
            0,
            768,
            (
                np.array(
                    [
                        1,
                    ],
                    dtype=np.uint64,
                ),
            ),
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
        layout = UCMKVCacheLayout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        key = b"s" * 12
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            AttentionType.STATE,
            (key,),
            1536,
            2304,
            (
                np.array(
                    [
                        1,
                    ],
                    dtype=np.uint64,
                ),
            ),
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
            ucm_cache_block_size=256,
        )
        caches = {
            "model.layers.0.attn": (
                FakeTensor(0x1000, (8, 128, 4), (512, 4, 1)),
                FakeTensor(0x3000, (8, 128, 2), (256, 2, 1)),
            ),
            "model.layers.1.attn": (FakeTensor(0x5000, (8, 128, 3), (384, 3, 1)),),
        }
        layout = UCMKVCacheLayout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        proxy = RecordingTestStore()
        coordinator = make_dispatcher(
            parsed, proxy, make_request_hasher(vllm_config(), 0), recompute_tokens=0
        )
        request = FakeRequest("r", 256)
        dispatcher = coordinator
        result = dispatcher.lookup(request, 0)
        state = dispatcher.requests["r"]
        state.vllm_block_tables = ([2, 5],)
        state.external_hit_tokens = 256
        state.load_pending = True
        metadata = _scheduler_metadata(dispatcher, {"r": 1})

        batch = _flat_layout_records(layout, "load", metadata)

        # Records are block-major: block 0's views, then block 1's -- each
        # block's layer pages land at the per-block anchor (K/V of layer 0
        # then layer 1), so one ucm block is the byte-concatenation of its
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
        layout = UCMKVCacheLayout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        group_layout = layout.group_layouts[0]
        span = group_layout.block_first
        self.assertIsNotNone(span)
        self.assertEqual(
            (span.base_ptr, span.block_stride, span.block_size_bytes),
            (0x1000, block_stride, 2 * layer_stride),
        )
        # The record is slot-sized: both layers' page slots, paddings and all.
        self.assertEqual(
            layout.store_layouts[group_layout.group_id].whole_block_bytes,
            2 * layer_stride,
        )

        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"b" * 12
        plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (key,),
            0,
            256,
            (
                np.array(
                    [
                        2,
                    ],
                    dtype=np.uint64,
                ),
            ),
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
            ],
        )
        self.assertEqual(
            batch.sizes.tolist(),
            [
                2 * layer_stride,
            ],
        )
        self.assertEqual(
            batch.offsets.tolist(),
            [
                0,
            ],
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
        layout = UCMKVCacheLayout(
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
            layout.store_layouts[group_layout.group_id].whole_block_bytes, slot
        )

        # Each segment keeps its offset inside the span for layerwise IO.
        self.assertEqual(
            layout.store_layouts[group_layout.group_id].group_block_offsets.tolist(),
            [0, 8192, 4096, 10240],
        )
        from ucm.integration.vllm.v2.layout.store_layout import GroupStoreLayout

        layerwise_store = GroupStoreLayout.build(
            parsed.groups[group_layout.group_id],
            group_layout,
            parsed.ucm_cache_block_size,
            use_layerwise=True,
        )
        self.assertFalse(layerwise_store.merge_whole_blocks)
        self.assertEqual(
            layerwise_store.group_block_offsets.tolist(), [0, 4096, 6144, 10240]
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
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"c" * 12
        plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (key,),
            0,
            256,
            (
                np.array(
                    [
                        3,
                    ],
                    dtype=np.uint64,
                ),
            ),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = _flat_layout_records(layout, "load", metadata)
        self.assertEqual(
            batch.ptrs.tolist(),
            [
                0x1000 + 3 * slot,
            ],
        )
        self.assertEqual(
            batch.sizes.tolist(),
            [
                slot,
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
                4096,
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
            ucm_cache_block_size=8,
        )
        layout = UCMKVCacheLayout(
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
            layout.store_layouts[group_layout.group_id].whole_block_bytes, 24
        )
        self.assertEqual(
            group_layout.layer_names,
            ("model.layers.0.attn", "model.layers.1.attn"),
        )
        self.assertEqual(
            layout.store_layouts[group_layout.group_id].group_block_offsets.tolist(),
            [0, 12],
        )

        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"r" * 12
        plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (key,),
            0,
            8,
            (np.array([2, 5], dtype=np.uint64),),
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
            ucm_cache_block_size=256,
        )
        caches = {
            "model.layers.0.attn": (
                FakeTensor(0x1000, (8, 128, 4), (512, 4, 1)),
                FakeTensor(0x3000, (8, 128, 2), (256, 2, 1)),
            ),
            "model.layers.1.attn": (FakeTensor(0x5000, (8, 128, 3), (384, 3, 1)),),
        }
        layout = UCMKVCacheLayout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"t" * 12
        plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (key,),
            0,
            256,
            (np.array([2, 5], dtype=np.uint64),),
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
        layout = UCMKVCacheLayout(
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
        key = b"g" * 12
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (key,),
            0,
            128,
            (
                np.array(
                    [
                        2,
                    ],
                    dtype=np.uint64,
                ),
            ),
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
        from ucm.integration.vllm.v2.ucm_scheduler import (
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
        layout = UCMKVCacheLayout(
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
                AttentionType.FA,
                (b"record______",),
                0,
                4,
                (
                    np.array([block0], dtype=np.uint64),
                    np.array([block1], dtype=np.uint64),
                ),
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
        offsets = (8, 16, 36)
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
            proxy.commit(source.block_ids)
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
            ucm_cache_block_size=512,
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
        layout = UCMKVCacheLayout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        key = b"k" * 12
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (key,),
            512,
            1024,
            # One window per FA group in route order: group 0 is 1:1
            # (one whole block); group 1 is N:1 -- the key is one
            # 512-token sub-span inside a 16384-token page.
            (np.array([1], dtype=np.uint64), np.array([2], dtype=np.uint64)),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )

        batch = _flat_layout_records(layout, "load", metadata)

        # Group 0's whole block, then group 1's sub-span: 512-token page
        # window selected by token start inside the large page.
        self.assertEqual(batch.sizes[-1], 4096)
        self.assertEqual(batch.ptrs[-1], 0x5000 + 2 * 131072 + 4 * 1024)

    def test_hybrid_state_uses_one_complete_checkpoint_block(self):
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
        layout = UCMKVCacheLayout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        # Ascend V1 copies State after the load: only the boundary block loads.
        dispatcher = make_dispatcher(parsed, need_load_to_running_state=False)
        request = FakeRequest("r", 512)
        common_ids = tuple(bytes([index]) * 12 for index in range(4))
        state = seed_request_state(dispatcher, request, 512, common_ids)
        state.vllm_block_tables = ([1, 2, 3, 4], [0, 1, 2, 6])

        metadata = _scheduler_metadata(dispatcher, {"r": 1})
        batch = _flat_layout_records(layout, "load", metadata)

        state_segments = [
            (ptr, size)
            for key, ptr, size in zip(batch.block_ids, batch.ptrs, batch.sizes)
            if key in ucm_ids(common_ids, AttentionType.STATE)
        ]
        self.assertEqual(state_segments, [(state_page.data_ptr() + 6 * 256, 256)])

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
        layout = UCMKVCacheLayout(
            parsed, caches, kv_cache_config=raw_config, num_hidden_layers=4
        )
        key = b"z" * 12
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        dump_plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (key,),
            0,
            4,
            (
                np.array(
                    [
                        1,
                    ],
                    dtype=np.uint64,
                ),
            ),
        )
        load_plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (key,),
            0,
            4,
            (
                np.array(
                    [
                        6,
                    ],
                    dtype=np.uint64,
                ),
            ),
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
        ucm_block_ids = tuple(dict.fromkeys(dump_batch.block_ids))
        adapter.commit(ucm_block_ids, np.arange(len(ucm_block_ids), dtype=np.int64))
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
        ucm_block_ids = tuple(dict.fromkeys(dump_batch.block_ids))
        deferred_adapter.commit(
            ucm_block_ids, np.arange(len(ucm_block_ids), dtype=np.int64)
        )
        deferred_load_plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (key,),
            0,
            4,
            (
                np.array(
                    [
                        7,
                    ],
                    dtype=np.uint64,
                ),
            ),
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
        return UCMKVCacheLayout(
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
                declared.store_layouts[group_id].whole_block_bytes,
                undeclared.store_layouts[group_id].whole_block_bytes,
            )
            np.testing.assert_array_equal(
                declared.store_layouts[group_id].group_block_offsets,
                undeclared.store_layouts[group_id].group_block_offsets,
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

        kv_cache_config = SimpleNamespace(
            num_blocks=4,
            kv_cache_tensors=(
                SimpleNamespace(
                    size=192,
                    layers=("model.layers.0.attn", "model.layers.1.attn"),
                    offset=0,
                    layer_stride=96,
                    block_stride=12,
                ),
            ),
            kv_cache_groups=(
                SimpleNamespace(
                    layer_names=["model.layers.0.attn", "model.layers.1.attn"],
                    is_eagle_group=False,
                    kv_cache_spec=FullAttentionSpec(12),
                ),
            ),
            prefix_cache_retention_interval=0,
            kv_cache_layout="LBHNC",
            prefix_cacheable_group_ids=(0,),
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
            self.assertEqual(group["layer_names"], group["layer_names"])
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
        self.layout = UCMKVCacheLayout(
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
        self.assertEqual(layout.store_layouts[0].store_bytes, 24)

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
        layout = UCMKVCacheLayout(
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
            device_type="cpu",
        )
        with self.assertRaisesRegex(ValueError, "group 1 requires whole-block access"):
            UCMKVCacheLayout(
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
            gl = UCMKVCacheLayout(
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
        layout = UCMKVCacheLayout(
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
            ucm_cache_block_size=unit,
            device_type="cpu",
        )
        layout = UCMKVCacheLayout(
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
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        for order in ("LBHNC", "LHBNC", "BHLNC"):
            with self.subTest(order=order):
                # Multi-head HNC: one UCM block per whole vLLM block.
                layout, strides = self._layout(order, unit=8)
                plan = UCMGroupDispatchPlan(
                    AttentionType.FA,
                    (b"a" * 12, b"b" * 12),
                    0,
                    16,
                    (np.array([1, 2], dtype=np.uint64),),
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
                    proxy.commit(dump.block_ids)
                    for k, block in zip(ucm_ids(plan.common_ucm_block_ids), (1, 2)):
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
            layout.store_layouts[group_layout.group_id].group_block_offsets.tolist(),
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
                gl = UCMKVCacheLayout(
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
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        names = ["model.layers.0.attn", "model.layers.1.attn"]
        raw_config = config(group(names, FullAttentionSpec(16384)), num_blocks=3)
        parsed = parse_spec(
            raw_config,
            scheduler_block_size=16384,
            ucm_cache_block_size=512,
            device_type="npu",
        )
        layout = UCMKVCacheLayout(
            parsed,
            {
                name: FakeTensor(base, (3, 16384, 2, 2), (65536, 4, 2, 1))
                for name, base in zip(names, (1000000, 2000000))
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (b"a" * 12, b"b" * 12),
            0,
            1024,
            (np.array([2, 2], dtype=np.uint64),),
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
            ucm_cache_block_size=512,
            device_type="npu",
        )
        layout = UCMKVCacheLayout(
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
    def _metadata(self, ucm_block_ids=(b"a" * 12, b"b" * 12)):
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            AttentionType.FA, ucm_block_ids, 0, 16, (np.array([1, 2], dtype=np.uint64),)
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
        self.assertEqual(batch.ucm_block_ids, ucm_ids((b"a" * 12, b"b" * 12)))
        self.assertEqual(batch.ptrs.shape, (2, 2))
        self.assertEqual(batch.ucm_block_offsets.tolist(), [[0, 32]] * 2)
        self.assertEqual(batch.sizes.tolist(), [[32] * 2] * 2)
        self.assertEqual(batch.sizes.strides[0], 0)
        self.assertFalse(batch.sizes.flags.writeable)
        (selected,) = UCMTransferBuilder(layout).build_load_transfers(meta, layer_id=1)
        np.testing.assert_array_equal(selected.ptrs, batch.ptrs[:, 1:])
        self.assertEqual(selected.ucm_block_offsets.tolist(), [[32]] * 2)
        legacy = _flat_layout_records(layout, "dump", meta)
        memory = ByteMemory()
        for address in range(1000, 1192):
            memory.write(address, bytes([address % 251]))
        with tempfile.TemporaryDirectory() as directory:
            proxy = FileTestStore(directory, MemoryByteAccess(memory))
            adapter = UCMProxyAdapter(PipelineProxy(proxy))
            adapter.submit("dump", batch)
            adapter.commit(
                batch.ucm_block_ids, np.arange(len(batch.ucm_block_ids), dtype=np.int64)
            )
            expected = []
            for key in batch.ucm_block_ids:
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
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        names = ["model.layers.0.attn", "model.layers.1.attn"]
        raw_config = config(
            group([names[0]], FullAttentionSpec(8)),
            group([names[1]], FullAttentionSpec(16)),
            num_blocks=4,
        )
        spec = parse_spec(
            raw_config,
            scheduler_block_size=16,
        )
        layout = UCMKVCacheLayout(
            spec,
            {
                names[0]: FakeTensor(1000, (4, 8, 1), (8, 1, 1)),
                names[1]: FakeTensor(2000, (4, 16, 1), (16, 1, 1)),
            },
            kv_cache_config=raw_config,
            num_hidden_layers=4,
        )
        # One 16-token UCM block: two blocks of group 0, one of group 1.
        plan = UCMGroupDispatchPlan(
            AttentionType.FA,
            (b"a" * 12,),
            0,
            16,
            (np.array([0, 1]), np.array([2])),
        )
        meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        (batch,) = UCMTransferBuilder(layout).build_load_transfers(meta)
        self.assertEqual(batch.ptrs.tolist(), [[1000, 1008, 2032]])
        self.assertEqual(batch.ucm_block_offsets.tolist(), [[0, 8, 16]])
        self.assertEqual(batch.sizes.tolist(), [[8, 8, 16]])
        (selected,) = UCMTransferBuilder(layout).build_load_transfers(meta, layer_id=1)
        self.assertEqual(selected.ucm_block_offsets.tolist(), [[16]])

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
                    [b.ucm_block_ids for b in batches],
                    [ucm_ids(plan.common_ucm_block_ids)] * 2,
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
        from ucm.integration.vllm.v2.ucm_proxy import UCMProxyTransfer

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
