import enum
import inspect
import json
import math
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass, replace
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


@dataclass
class KVConnectorTransferResults:
    pass


class KVConnectorMetadata:
    pass


class KVConnectorWorkerMetadata:
    def aggregate(self, other):
        raise NotImplementedError


class SupportsHMA:
    pass


vllm_base.KVConnectorTransferResults = KVConnectorTransferResults
vllm_base.KVConnectorBase_V1 = KVConnectorBaseV1
vllm_base.KVConnectorMetadata = KVConnectorMetadata
vllm_base.KVConnectorRole = KVConnectorRole
vllm_base.KVConnectorWorkerMetadata = KVConnectorWorkerMetadata
vllm_base.SupportsHMA = SupportsHMA
sys.modules[vllm_base.__name__] = vllm_base

vllm_model_utils = types.ModuleType("vllm.model_executor.models.utils")


def extract_layer_index(layer_name, num_attn_module=1):
    values = [int(part) for part in layer_name.split(".") if part.isdecimal()]
    if num_attn_module == 1 or "attn" not in layer_name:
        assert len(values) == 1
        return values[0]
    assert 1 <= len(values) <= 2
    return values[0] * num_attn_module + values[1] if len(values) == 2 else values[0]


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
    MAMBA = "mamba"
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
    SINK_FULL_ATTENTION = "sink_full_attention"
    ENCODER_ONLY_ATTENTION = "encoder_only_attention"
    CROSS_ATTENTION = "cross_attention"
    UNKNOWN = "unknown"


@dataclass
class FullAttentionSpec:
    block_size: int
    tokens_per_state: int = 1
    sliding_window: int | None = None
    prefix_cacheable: bool = True


@dataclass
class MLAAttentionSpec(FullAttentionSpec):
    pass


@dataclass
class HiddenStateCacheSpec(MLAAttentionSpec):
    pass


@dataclass
class SlidingWindowMLASpec(FullAttentionSpec):
    pass


@dataclass
class AscendSlidingWindowMLASpec(SlidingWindowMLASpec):
    @property
    def storage_block_size(self):
        return self.block_size // self.tokens_per_state


@dataclass
class AscendSFAIndexerCacheSpec(MLAAttentionSpec):
    sfa_dcp_replicated_indexer_size: int = 1


@dataclass
class MambaSpec:
    block_size: int
    mamba_cache_mode: str = "align"
    prefix_cacheable: bool = True


@dataclass
class UnsupportedSpec(FullAttentionSpec):
    kind: KVCacheSpecKind = KVCacheSpecKind.UNKNOWN


@dataclass
class UniformTypeKVCacheSpecs:
    kv_cache_specs: dict[str, object]

    @property
    def block_size(self):
        return next(iter(self.kv_cache_specs.values())).block_size

    @property
    def prefix_cacheable(self):
        return all(spec.prefix_cacheable for spec in self.kv_cache_specs.values())


def get_kv_cache_spec_kind(spec):
    if isinstance(spec, UnsupportedSpec):
        return spec.kind
    if isinstance(spec, MambaSpec):
        return KVCacheSpecKind.MAMBA
    if isinstance(spec, SlidingWindowMLASpec):
        return KVCacheSpecKind.SLIDING_WINDOW_MLA
    if isinstance(spec, MLAAttentionSpec):
        return KVCacheSpecKind.MLA_ATTENTION
    if isinstance(spec, FullAttentionSpec):
        return KVCacheSpecKind.FULL_ATTENTION
    return KVCacheSpecKind.UNKNOWN


for name in (
    "KVCacheSpecKind",
    "UniformTypeKVCacheSpecs",
    "HiddenStateCacheSpec",
    "MLAAttentionSpec",
    "SlidingWindowMLASpec",
    "get_kv_cache_spec_kind",
):
    setattr(vllm_kv_cache_interface, name, globals()[name])
sys.modules[vllm_kv_cache_interface.__name__] = vllm_kv_cache_interface

vllm_envs = types.ModuleType("vllm.envs")
vllm_envs.VLLM_KV_CACHE_LAYOUT = "LBNHC"
sys.modules[vllm_envs.__name__] = vllm_envs
sys.modules["vllm"].envs = vllm_envs

parallel_state = types.ModuleType("vllm.distributed.parallel_state")
parallel_state.get_world_group = lambda: SimpleNamespace(rank=0)
sys.modules[parallel_state.__name__] = parallel_state

native_layout = types.ModuleType("vllm.v1.kv_cache_layout")


class KVCacheLayout(enum.Enum):
    LBNHC = "LBNHC"
    LBHNC = "LBHNC"
    BLNHC = "BLNHC"
    BLHNC = "BLHNC"

    @property
    def is_block_outermost(self):
        return self.name.startswith("B")

    @property
    def is_block_compact(self):
        return self.name in ("BLNHC", "BLHNC")


native_layout.KVCacheLayout = KVCacheLayout
sys.modules[native_layout.__name__] = native_layout

for name in ("vllm_ascend", "vllm_ascend.core"):
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules.setdefault(name, package)
ascend_specs = types.ModuleType("vllm_ascend.core.kv_cache_interface")
ascend_specs.AscendSFAIndexerCacheSpec = AscendSFAIndexerCacheSpec
ascend_specs.AscendSlidingWindowMLASpec = AscendSlidingWindowMLASpec
sys.modules[ascend_specs.__name__] = ascend_specs

vllm_v1_core = types.ModuleType("vllm.v1.core")
vllm_v1_core.__path__ = []
sys.modules.setdefault("vllm.v1.core", vllm_v1_core)
vllm_kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")


def resolve_kv_cache_block_sizes(raw, cfg):
    if len(raw.kv_cache_groups) == 1:
        size = (
            cfg.cache_config.block_size
            * cfg.parallel_config.decode_context_parallel_size
        )
    else:
        size = math.lcm(*(g.kv_cache_spec.block_size for g in raw.kv_cache_groups))
    return size, size


vllm_kv_cache_utils.resolve_kv_cache_block_sizes = resolve_kv_cache_block_sizes
sys.modules[vllm_kv_cache_utils.__name__] = vllm_kv_cache_utils

from ucm.integration.vllm.v2.layout import (  # noqa: E402
    UCMKVCacheLayout,
    parse_kv_cache_config,
)
from ucm.integration.vllm.v2.parallel import (  # noqa: E402
    AllShardLookup,
    ParallelLayout,
)
from ucm.integration.vllm.v2.ucm_connector import (  # noqa: E402
    UCMConnector,
    UCMWorkerMetadata,
)
from ucm.integration.vllm.v2.ucm_kv_cache import UCMTransferBuilder  # noqa: E402
from ucm.integration.vllm.v2.ucm_proxy import (  # noqa: E402
    SimpleFileUCMProxy,
    UCMProxyAdapter,
    UCMProxyError,
    UCMProxyTransfer,
)
from ucm.integration.vllm.v2.ucm_scheduler import (  # noqa: E402
    RequestDispatchMeta,
    RequestHasher,
    RequestState,
    UCMConnectorMetadata,
    UCMDispatcher,
    UCMGroupDispatchPlan,
)


def group(names, specs, *, enable_kv_transfer=True):
    spec = UniformTypeKVCacheSpecs(specs) if isinstance(specs, dict) else specs
    return SimpleNamespace(
        layer_names=list(names),
        kv_cache_spec=spec,
        is_eagle_group=False,
        enable_kv_transfer=enable_kv_transfer,
    )


def config(*groups, num_blocks=8, tensors=(), layout="LBNHC"):
    return SimpleNamespace(
        kv_cache_groups=list(groups),
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_layout=layout,
        prefix_cache_retention_interval=0,
        prefix_cacheable_group_ids=tuple(
            i
            for i, g in enumerate(groups)
            if g.enable_kv_transfer and g.kv_cache_spec.prefix_cacheable
        ),
    )


class ModelConfig(SimpleNamespace):
    def to_dict(self):
        return vars(self)


def vllm_config():
    hf = ModelConfig(num_hidden_layers=64, model_type="fixture", compress_ratios=[])
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="org/model",
            dtype="torch.bfloat16",
            hf_config=hf,
            hf_text_config=hf,
        ),
        cache_config=SimpleNamespace(
            block_size=4,
            cache_dtype="auto",
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "v2_storage_path": str(Path(tempfile.gettempdir()) / "ucm-v2-test"),
            }
        ),
        device_config=SimpleNamespace(device_type="cpu"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
            world_size=2,
        ),
        speculative_config=None,
        additional_config={},
        instance_id="fixture",
        use_v2_model_runner=False,
    )


class FakeTensor:
    """Explicit registered BHNC/BNHC geometry; no legacy-shape normalization."""

    def __init__(self, ptr, shape, strides, element_size=1):
        self._ptr, self.shape = ptr, shape
        self._strides, self._element_size = strides, element_size
        self.ndim = len(shape)

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return math.prod(self.shape)

    def stride(self, index=None):
        return self._strides if index is None else self._strides[index]

    def element_size(self):
        return self._element_size

    def permute(self, *axes):
        return FakeTensor(
            self._ptr,
            tuple(self.shape[i] for i in axes),
            tuple(self._strides[i] for i in axes),
            self._element_size,
        )


def make_layout(
    raw,
    caches,
    *,
    tokens_per_chunk=None,
    scheduler=8,
    device="cpu",
    layerwise=False,
    ratios=None,
    num_attn_module=1,
):
    spec = parse_kv_cache_config(
        raw,
        scheduler_block_size=scheduler,
        num_hidden_layers=64,
        tokens_per_chunk=tokens_per_chunk,
        device_type=device,
        compressor_tokens_per_state=ratios,
        num_attn_module=num_attn_module,
    )
    return UCMKVCacheLayout(
        spec,
        caches,
        kv_cache_config=raw,
        num_hidden_layers=64,
        use_layerwise=layerwise,
    )


class FakeRequest:
    def __init__(self, request_id, token_count):
        self.request_id = request_id
        self.all_token_ids = list(range(token_count))


class FakeProxy:
    def __init__(self):
        self.present = set()
        self.lookup_calls, self.load_calls, self.dump_calls = [], [], []

    def lookup(self, keys):
        self.lookup_calls.append(tuple(keys))
        return [key in self.present for key in keys]

    def lookup_on_prefix(self, keys):
        return next(
            (i - 1 for i, key in enumerate(keys) if key not in self.present),
            len(keys) - 1,
        )

    def lookup_on_reverse(self, keys):
        return next(
            (i for i in range(len(keys) - 1, -1, -1) if keys[i] in self.present), -1
        )

    def load(self, keys, offsets, ptrs, sizes):
        self.load_calls.append((keys, offsets, ptrs, sizes))

    def dump(self, keys, offsets, ptrs, sizes):
        self.dump_calls.append((keys, offsets, ptrs, sizes))


def make_dispatcher(parsed, proxy=None, hasher=None, **kwargs):
    return UCMDispatcher(
        parsed,
        UCMProxyAdapter(proxy or FakeProxy()),
        hasher or RequestHasher(vllm_config(), 0),
        b"seed",
        **kwargs,
    )


def seed_request_state(
    dispatcher, request, external_hit_tokens=0, group_ucm_block_ids=(), hbm_hit_tokens=0
):
    state = RequestState(
        hbm_hit_tokens=hbm_hit_tokens,
        external_hit_tokens=external_hit_tokens,
        num_token_ids=len(request.all_token_ids),
        token_processed=hbm_hit_tokens + external_hit_tokens,
        group_ucm_block_ids=group_ucm_block_ids,
        group_vllm_block_ids=tuple([] for _ in dispatcher.spec.groups),
        load_pending=external_hit_tokens > 0,
    )
    dispatcher.requests[str(request.request_id)] = state
    return state


def _scheduler_metadata(
    dispatcher,
    scheduled_tokens,
    *,
    offloads=None,
    preempted_req_ids=(),
    finished_req_ids=(),
):
    block_state = SimpleNamespace(
        req_ids=tuple(r for r in scheduled_tokens if r in dispatcher.requests),
        get_block_ids=lambda r: dispatcher.requests[r].group_vllm_block_ids,
        boundary_state_offloads=offloads or {},
    )
    return dispatcher.build_from_scheduler_output(
        SimpleNamespace(
            kv_connector_block_state=block_state,
            num_scheduled_tokens=scheduled_tokens,
            total_num_scheduled_tokens=sum(scheduled_tokens.values()),
            preempted_req_ids=set(preempted_req_ids),
            finished_req_ids=set(finished_req_ids),
        )
    )


def metadata(kind, blocks, *, keys=(b"a" * 16,), start=0, end=8, load=False):
    plan = UCMGroupDispatchPlan(
        kind,
        keys,
        np.linspace(start, end, len(keys), endpoint=False, dtype=np.uint64),
        tuple(np.asarray(b, dtype=np.uint64) for b in blocks),
    )
    return UCMConnectorMetadata(
        requests={
            "r": RequestDispatchMeta(
                "r",
                load_plans=(plan,) if load else (),
                dump_plans=() if load else (plan,),
            ),
        }
    )


def _submit_segments(adapter, operation, keys, offsets, ptrs, sizes):
    grouped = {}
    for key, offset, ptr, size in zip(keys, offsets, ptrs, sizes, strict=True):
        grouped.setdefault(key, []).append((offset, ptr, size))
    for key, segments in grouped.items():
        offsets, ptrs, sizes = np.asarray(segments, dtype=np.uint64).T
        adapter.submit(
            operation,
            UCMProxyTransfer((key,), ptrs[None, :], sizes[None, :], offsets[None, :]),
        )


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
        self.hasher = RequestHasher(vllm_config(), 1)

    def test_hash_golden_vector(self):
        self.assertEqual(
            self.hasher((b"parent", (1, 2, 3, 4))).hex(),
            "e60666fdcd359c82d9a369fc06bb3270",
        )

    def test_key_tag_bit_layout(self):
        from ucm.integration.vllm.v2.ucm_scheduler import _key_tag

        # type(2) group(4) tp_rank(4) pp_rank(4) reserved(2), big-endian.
        self.assertEqual(_key_tag("FA"), b"\x00\x00")
        self.assertEqual(_key_tag("WA"), b"\x40\x00")
        self.assertEqual(_key_tag("State"), b"\x80\x00")
        self.assertEqual(_key_tag("FA", tp_rank=1), b"\x00\x40")
        self.assertEqual(_key_tag("FA", pp_rank=1), b"\x00\x04")
        self.assertEqual(_key_tag("WA", tp_rank=15, pp_rank=15), b"\x43\xfc")

    def test_chain_keys_share_one_chain_and_differ_only_by_tag(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            num_hidden_layers=64,
            scheduler_block_size=128,
        )
        coordinator = UCMDispatcher(
            parsed, UCMProxyAdapter(FakeProxy()), self.hasher, b"seed"
        )
        request = FakeRequest("r", 512)
        values = coordinator._chain(request.all_token_ids, 128, b"seed")
        fa_keys, state_keys = coordinator._chain_keys(request.all_token_ids)

        self.assertEqual(fa_keys[0][:14], values[0][:14])
        self.assertEqual(state_keys[0][:14], values[0][:14])
        self.assertEqual(fa_keys[0][14:], b"\x00\x00")
        self.assertEqual(state_keys[0][14:], b"\x80\x00")
        self.assertNotEqual(fa_keys[0], state_keys[0])

    def test_custom_cache_block_chains_at_the_cache_block(self):
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            num_hidden_layers=64,
            scheduler_block_size=128,
            tokens_per_chunk=512,
        )
        proxy = FakeProxy()
        coordinator = UCMDispatcher(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed", recompute_tokens=0
        )
        request = FakeRequest("r", 1024)
        (keys,) = coordinator._chain_keys(request.all_token_ids)
        proxy.present.update(keys)

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.group_ucm_block_ids[0], keys)
        self.assertEqual(result.external_hit_tokens, 1024)

    def test_direct_prefix_stops_at_first_miss(self):
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            num_hidden_layers=64,
            scheduler_block_size=128,
        )
        proxy = FakeProxy()
        coordinator = UCMDispatcher(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed", recompute_tokens=0
        )
        request = FakeRequest("r", 512)
        (keys,) = coordinator._chain_keys(request.all_token_ids)
        proxy.present.update((keys[0], keys[2], keys[3]))

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 128)

    def test_direct_miss_never_moves_an_unaligned_hbm_boundary_backwards(self):
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            num_hidden_layers=64,
            scheduler_block_size=128,
            tokens_per_chunk=512,
        )
        proxy = FakeProxy()
        coordinator = UCMDispatcher(
            parsed,
            UCMProxyAdapter(proxy),
            RequestHasher(vllm_config(), 0),
            b"seed",
        )

        result = coordinator.lookup(FakeRequest("unaligned", 1024), 128)

        self.assertEqual(result.external_hit_tokens, 0)

    def test_direct_full_hit_leaves_the_recompute_margin_block(self):
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            num_hidden_layers=64,
            scheduler_block_size=128,
        )
        proxy = FakeProxy()
        coordinator = UCMDispatcher(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed"
        )
        request = FakeRequest("r", 512)
        (keys,) = coordinator._chain_keys(request.all_token_ids)
        proxy.present.update(keys)

        result = coordinator.lookup(request, 0)

        # The scan ends at the last complete block below the recompute
        # margin (v1 semantics): 384 restored, the last 128 recomputed.
        self.assertEqual(result.external_hit_tokens, 384)

    def test_hybrid_reverse_selects_latest_common_state(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            num_hidden_layers=64,
            scheduler_block_size=128,
        )
        proxy = FakeProxy()
        coordinator = UCMDispatcher(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed", recompute_tokens=0
        )
        request = FakeRequest("r", 1024)
        fa_keys, state_keys = coordinator._chain_keys(request.all_token_ids)
        proxy.present.update(fa_keys)
        proxy.present.add(state_keys[3])

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 512)

    def test_hybrid_full_hit_leaves_one_complete_state_alignment_block(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            num_hidden_layers=64,
            scheduler_block_size=128,
        )
        proxy = FakeProxy()
        coordinator = UCMDispatcher(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed"
        )
        request = FakeRequest("r", 1024)
        fa_keys, state_keys = coordinator._chain_keys(request.all_token_ids)
        proxy.present.update((*fa_keys, *state_keys))

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 896)

    def test_hybrid_miss_still_returns_state_keys_for_later_dump(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            num_hidden_layers=64,
            scheduler_block_size=128,
        )
        coordinator = UCMDispatcher(
            parsed,
            UCMProxyAdapter(FakeProxy()),
            self.hasher,
            b"seed",
            recompute_tokens=0,
        )

        result = coordinator.lookup(FakeRequest("r", 1024), 0)

        self.assertEqual(result.external_hit_tokens, 0)
        self.assertEqual(len(result.group_ucm_block_ids[0]), 8)
        self.assertEqual(len(result.group_ucm_block_ids[1]), 8)

    def test_dsv4_requires_fa_prefix_and_latest_wa_boundary(self):
        parsed = parse_kv_cache_config(
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
            num_hidden_layers=64,
            scheduler_block_size=16,
        )
        proxy = FakeProxy()
        coordinator = UCMDispatcher(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed", recompute_tokens=0
        )
        request = FakeRequest("r", 1536)
        fa_keys, wa_keys = coordinator._chain_keys(request.all_token_ids)
        proxy.present.update(fa_keys)
        proxy.present.add(wa_keys[1])

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 1024)
        self.assertEqual(result.group_ucm_block_ids, (fa_keys, wa_keys))

    def test_dsv4_full_hit_leaves_one_complete_canonical_block(self):
        parsed = parse_kv_cache_config(
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
            num_hidden_layers=64,
            scheduler_block_size=16,
        )
        proxy = FakeProxy()
        coordinator = UCMDispatcher(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed"
        )
        request = FakeRequest("r", 1536)
        fa_keys, wa_keys = coordinator._chain_keys(request.all_token_ids)
        proxy.present.update((*fa_keys, *wa_keys))

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 1024)


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
            proxy = SimpleFileUCMProxy(directory, access)
            adapter = UCMProxyAdapter(proxy)
            adapter.register_tensors(registered)
            _submit_segments(adapter, "dump", (key, key), (0, 5), (4096, 8192), (5, 7))

            self.assertEqual(adapter.lookup((key,)), (False,))
            adapter.commit((key,))
            self.assertEqual(adapter.lookup((key, b"m" * 16)), (True, False))
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
            proxy = SimpleFileUCMProxy(directory, MemoryByteAccess(memory))
            adapter = UCMProxyAdapter(proxy)
            # Later layer first; repeated writes must preserve earlier ranges.
            _submit_segments(adapter, "dump", (key,), (4,), (8192,), (4,))
            _submit_segments(adapter, "dump", (other,), (0,), (8192,), (4,))
            self.assertEqual(adapter.lookup((key, other)), (False, False))
            _submit_segments(adapter, "dump", (key,), (0,), (4096,), (4,))
            adapter.commit((key,))
            self.assertEqual(adapter.lookup((key, other)), (True, False))
            self.assertEqual(proxy._path(key).read_bytes(), b"abcdefgh")
            adapter.commit((key,))  # Idempotent retry.
            _submit_segments(adapter, "load", (key,), (0,), (12288,), (8,))
            self.assertEqual(memory.read(0x3000, 8), b"abcdefgh")
            # A replacement remains hidden behind the previous complete block.
            _submit_segments(adapter, "dump", (key,), (0,), (8192,), (4,))
            self.assertEqual(proxy._path(key).read_bytes(), b"abcdefgh")
            adapter.commit((key, other))
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
            SimpleNamespace(keys=(b"a" * 16,)),
            SimpleNamespace(keys=(b"b" * 16,)),
        )
        connector.layout = SimpleNamespace(spec=connector.spec, group_layouts={})
        connector.transfer_builder = SimpleNamespace(
            build_dump_transfers=lambda _: batches
        )
        connector._proxy = mock.Mock()
        connector.wait_for_save()
        self.assertEqual(
            connector._proxy.mock_calls,
            [
                mock.call.submit("dump", batches[0]),
                mock.call.submit("dump", batches[1]),
                mock.call.commit(batches[0].keys),
                mock.call.commit(batches[1].keys),
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
        adapter = UCMProxyAdapter(proxy)
        adapter.commit((b"a" * 16,))
        proxy.wait.assert_called_once_with("commit-task")
        proxy.commit.side_effect = OSError("rename failed")
        with self.assertRaisesRegex(UCMProxyError, "Proxy commit failed"):
            adapter.commit((b"a" * 16,))

    def test_normalizes_proxy_error(self):
        class BrokenProxy(FakeProxy):
            def lookup(self, keys):
                raise OSError("backend detail")

        with self.assertRaises(UCMProxyError) as caught:
            UCMProxyAdapter(BrokenProxy()).lookup([b"x" * 16])
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_file_proxy_prefix_and_reverse_scans(self):
        keys = [bytes([index]) * 16 for index in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            proxy = SimpleFileUCMProxy(directory)
            for key in (keys[0], keys[1], keys[3]):
                (Path(directory) / f"{key.hex()}{proxy._SUFFIX}").write_bytes(b"x")
            adapter = UCMProxyAdapter(proxy)

            self.assertEqual(adapter.lookup_on_prefix(keys), 1)
            self.assertEqual(adapter.lookup_on_reverse(keys), 3)
            self.assertEqual(adapter.lookup_on_prefix(keys[2:]), -1)
            self.assertEqual(adapter.lookup_on_reverse((keys[4],)), -1)

    def test_scan_rejects_out_of_range_results(self):
        class BadScanProxy(FakeProxy):
            def lookup_on_prefix(self, keys):
                return len(keys)

        with self.assertRaisesRegex(UCMProxyError, "lookup_on_prefix"):
            UCMProxyAdapter(BadScanProxy()).lookup_on_prefix((b"x" * 16,))

    def test_waits_for_async_load_and_dump_tasks(self):
        class AsyncProxy(FakeProxy):
            def __init__(self):
                super().__init__()
                self.wait_calls = []

            def load(self, block_ids, offsets, ptrs, sizes):
                super().load(block_ids, offsets, ptrs, sizes)
                return ("load-task", len(block_ids))

            def dump(self, block_ids, offsets, ptrs, sizes):
                super().dump(block_ids, offsets, ptrs, sizes)
                return ("dump-task", len(block_ids))

            def wait(self, task):
                self.wait_calls.append(task)

        proxy = AsyncProxy()
        adapter = UCMProxyAdapter(proxy)
        key = b"x" * 16

        _submit_segments(adapter, "load", (key,), (0,), (4096,), (32,))
        _submit_segments(adapter, "dump", (key,), (0,), (8192,), (32,))

        self.assertEqual(
            proxy.wait_calls,
            [("load-task", 1), ("dump-task", 1)],
        )

    def test_rejects_async_task_without_wait_contract(self):
        class IncompleteAsyncProxy(FakeProxy):
            def load(self, block_ids, offsets, ptrs, sizes):
                return object()

        with self.assertRaisesRegex(UCMProxyError, "does not provide wait"):
            _submit_segments(
                UCMProxyAdapter(IncompleteAsyncProxy()),
                "load",
                (b"x" * 16,),
                (0,),
                (4096,),
                (32,),
            )


class ChunkMetadataTest(unittest.TestCase):
    def test_group_block_counts_are_shared_and_recomputed_after_cp(self):
        raw = config(
            group(["model.layers.0.attn"], FullAttentionSpec(8)),
            group(["model.layers.1.attn"], FullAttentionSpec(16)),
            group(
                ["model.layers.2.scratch"], UnsupportedSpec(7), enable_kv_transfer=False
            ),
        )
        spec = parse_kv_cache_config(
            raw, scheduler_block_size=8, num_hidden_layers=64, device_type="cpu"
        )
        self.assertEqual((spec.tokens_per_chunk, spec.blocks_per_chunk), (8, (1, 1, 0)))
        scaled = ParallelLayout(dcp=2).apply_context_parallel(spec)
        self.assertEqual(
            (scaled.tokens_per_chunk, scaled.blocks_per_chunk), (32, (2, 1, 0))
        )
        self.assertEqual(scaled.groups[2].tokens_per_block, 7)
        with self.assertRaisesRegex(ValueError, "whole distributed"):
            ParallelLayout(dcp=2).apply_context_parallel(spec, 8)

    def test_fa_swa_state_counts_use_their_own_semantics(self):
        raw = config(
            group(["model.layers.0.attn"], FullAttentionSpec(8)),
            group(
                ["model.layers.1.swa_cache"], SlidingWindowMLASpec(8, sliding_window=12)
            ),
            group(["model.layers.2.state"], MambaSpec(8)),
        )
        spec = parse_kv_cache_config(
            raw, scheduler_block_size=8, tokens_per_chunk=32, num_hidden_layers=64
        )
        self.assertEqual(spec.blocks_per_chunk, (4, 2, 1))
        self.assertFalse(hasattr(spec.groups[0], "tail_blocks"))

    def test_disabled_native_groups_are_not_classified(self):
        for enabled, cacheable in ((False, True), (True, False)):
            raw = config(
                group(["model.layers.0.attn"], FullAttentionSpec(8)),
                group(
                    ["model.layers.1.scratch"],
                    UnsupportedSpec(7, prefix_cacheable=cacheable),
                    enable_kv_transfer=enabled,
                ),
            )
            spec = parse_kv_cache_config(
                raw, scheduler_block_size=8, num_hidden_layers=64
            )
            self.assertEqual(spec.blocks_per_chunk, (1, 0))
            self.assertEqual(spec.groups[1].kinds, frozenset())

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
                parse_kv_cache_config(
                    config(
                        group(["model.layers.0.attn"], UnsupportedSpec(8, kind=kind))
                    ),
                    scheduler_block_size=8,
                    num_hidden_layers=64,
                )

    def test_sink_attention_is_an_explicit_supported_kind(self):
        raw = config(
            group(
                ["model.layers.0.attn"],
                UnsupportedSpec(8, kind=KVCacheSpecKind.SINK_FULL_ATTENTION),
            )
        )
        spec = parse_kv_cache_config(raw, scheduler_block_size=8, num_hidden_layers=64)
        self.assertEqual(tuple(g.group_id for g in spec.fa_groups), (0,))

    def test_chunk_alignment_and_state_mode_are_checked(self):
        raw = config(
            group(["model.layers.0.attn"], FullAttentionSpec(8)),
            group(["model.layers.1.state"], MambaSpec(8)),
        )
        for size in (0, -8, 12):
            with (
                self.subTest(size=size),
                self.assertRaisesRegex(ValueError, "positive multiple"),
            ):
                parse_kv_cache_config(
                    raw,
                    scheduler_block_size=8,
                    tokens_per_chunk=size,
                    num_hidden_layers=64,
                )
        raw.kv_cache_groups[1].kv_cache_spec.mamba_cache_mode = "all"
        with self.assertRaisesRegex(ValueError, "align"):
            parse_kv_cache_config(raw, scheduler_block_size=8, num_hidden_layers=64)

    def test_empty_pp_group_keeps_metadata_and_route_position(self):
        raw = config(
            group([], {"model.layers.0.attn": FullAttentionSpec(8)}),
            group(["model.layers.1.attn"], FullAttentionSpec(8)),
        )
        layout = make_layout(
            raw, {"model.layers.1.attn": FakeTensor(1000, (8, 1, 8, 2), (16, 16, 2, 1))}
        )
        self.assertEqual(layout.spec.blocks_per_chunk, (1, 1))
        self.assertEqual(tuple(layout.group_layouts), (1,))
        (batch,) = UCMTransferBuilder(layout).build_dump_transfers(
            metadata("FA", ([7], [2]))
        )
        self.assertEqual(batch.ptrs.tolist(), [[1032]])
        self.assertEqual(batch.ucm_block_offsets.tolist(), [[0]])

    def test_legacy_compressor_tail_uses_model_ratio_only(self):
        raw = config(
            group(["model.layers.0.attn"], MLAAttentionSpec(8, tokens_per_state=4)),
            group(
                ["model.layers.0.compressor.state_cache"],
                SlidingWindowMLASpec(4, sliding_window=8),
            ),
        )
        spec = parse_kv_cache_config(
            raw,
            scheduler_block_size=8,
            num_hidden_layers=64,
            compressor_tokens_per_state={0: 4},
        )
        self.assertEqual(
            (spec.groups[1].tail_tokens, spec.blocks_per_chunk), (4, (1, 1))
        )

    def test_longcat_layer_ids_match_native_attention_module_numbering(self):
        names = ("model.layers.2.attn.0", "model.layers.2.attn.1")
        raw = config(group(names, FullAttentionSpec(8)))
        caches = {
            name: FakeTensor(1000 + i * 1000, (8, 1, 8, 1), (8, 8, 1, 1))
            for i, name in enumerate(names)
        }
        layout = make_layout(raw, caches, num_attn_module=2)
        self.assertEqual(layout.layer_id_by_name, {names[0]: 4, names[1]: 5})
        from ucm.integration.vllm.v2.layout.kv_cache import _layer_index

        self.assertEqual(_layer_index("mtp.0.attn", "npu", 64), 64)


class HBMAddressTest(unittest.TestCase):
    def test_group_resolves_arbitrary_block_order_without_chunk_shape(self):
        names = ("model.layers.0.attn", "model.layers.1.attn")
        layout = make_layout(
            config(group(names, FullAttentionSpec(8))),
            {
                names[0]: FakeTensor(1000, (8, 1, 8, 2), (32, 16, 2, 1)),
                names[1]: FakeTensor(2000, (8, 1, 8, 3), (48, 24, 3, 1)),
            },
        )
        group_layout = layout.group_layouts[0]
        self.assertEqual(
            group_layout.resolve_ptrs([3, 1, 3]).tolist(),
            [[1096, 2144], [1032, 2048], [1096, 2144]],
        )
        mask = group_layout.segment_mask(layer_ids=[1])
        self.assertEqual(
            group_layout.resolve_ptrs([2], segment_mask=mask).tolist(), [[2096]]
        )
        self.assertEqual(group_layout.resolve_ptrs([]).shape, (0, 2))
        self.assertFalse(hasattr(group_layout, "spec"))
        self.assertFalse(hasattr(group_layout, "tokens_per_chunk"))

    def test_npu_split_components_use_view_strides_not_whole_page_descriptor(self):
        name = "model.layers.0.attn"
        raw = config(
            group([name], FullAttentionSpec(8)),
            tensors=(
                SimpleNamespace(
                    layers=(name,),
                    offset=0,
                    layer_stride=512,
                    block_stride=64,
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
            device="npu",
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
                    layers=(name,),
                    offset=0,
                    layer_stride=512,
                    block_stride=64,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            make_layout(raw, {name: FakeTensor(1000, (8, 1, 8, 2), (16, 16, 2, 1))})

    def test_chunk_alignment_is_validated_at_initialization(self):
        name = "model.layers.0.attn"
        raw = config(group([name], MLAAttentionSpec(128, tokens_per_state=4)))
        caches = {name: FakeTensor(1000, (8, 1, 32, 16), (512, 512, 16, 1))}
        with self.assertRaisesRegex(ValueError, "stored states"):
            make_layout(raw, caches, scheduler=2, tokens_per_chunk=2)
        layout = make_layout(
            raw,
            caches,
            scheduler=128,
        ).group_layouts[0]
        self.assertEqual(
            layout.tokens_to_bytes(np.array([4], dtype=np.uint64), 128).tolist(),
            [[16]],
        )

    def test_compressed_entries_follow_native_tokens_per_state(self):
        name = "model.layers.0.attn"
        layout = make_layout(
            config(group([name], MLAAttentionSpec(16, tokens_per_state=4))),
            {name: FakeTensor(1000, (8, 1, 4, 3), (12, 12, 3, 1))},
            scheduler=4,
            tokens_per_chunk=4,
        )
        _, ptrs, sizes = layout.store_layouts[0].resolve_matrices(
            np.array([2, 2]), np.array([4, 8], dtype=np.uint64)
        )
        self.assertEqual(ptrs.tolist(), [[1027], [1030]])
        self.assertEqual(sizes.tolist(), [[3], [3]])

    def test_multirow_compact_heads_stay_one_segment(self):
        name = "model.layers.0.attn"
        layout = make_layout(
            config(group([name], FullAttentionSpec(8)), num_blocks=3),
            {name: FakeTensor(1000, (6, 2, 4, 2), (16, 8, 2, 1))},
        )
        physical = layout.group_layouts[0]
        self.assertEqual(physical.payload_bytes.tolist(), [32])
        self.assertEqual(physical.resolve_ptrs([2]).tolist(), [[1064]])
        self.assertFalse(physical.supports_partial_tokens)

    def test_group_partial_capability_cannot_be_bypassed_by_layer_selection(self):
        names = ("model.layers.0.attn", "model.layers.1.attn")
        raw = config(group(names, FullAttentionSpec(8)))
        with self.assertRaisesRegex(ValueError, "whole-block"):
            make_layout(
                raw,
                {
                    names[0]: FakeTensor(1000, (8, 1, 8, 2), (16, 16, 2, 1)),
                    names[1]: FakeTensor(2000, (8, 2, 8, 2), (32, 16, 2, 1)),
                },
                tokens_per_chunk=4,
                scheduler=4,
            )


class StoreChunkTest(unittest.TestCase):
    def _two_layers(self, *, tokens_per_chunk=4):
        names = ("model.layers.0.attn", "model.layers.1.attn")
        return make_layout(
            config(group(names, FullAttentionSpec(8)), num_blocks=4),
            {
                names[0]: FakeTensor(1000, (4, 1, 8, 2), (16, 16, 2, 1)),
                names[1]: FakeTensor(2000, (4, 1, 8, 3), (24, 24, 3, 1)),
            },
            tokens_per_chunk=tokens_per_chunk,
            scheduler=4,
        )

    def test_subblock_starts_follow_chunk_position_and_keep_layer_offsets(self):
        layout = self._two_layers()
        store = layout.store_layouts[0]
        offsets, ptrs, sizes = store.resolve_matrices(
            np.array([2, 2]), np.array([0, 4], dtype=np.uint64)
        )
        self.assertEqual(ptrs.tolist(), [[1032, 2048], [1040, 2060]])
        self.assertEqual(offsets.tolist(), [[0, 8], [0, 8]])
        self.assertEqual(sizes.tolist(), [[8, 12], [8, 12]])
        _, ptrs, _ = store.resolve_matrices(
            np.array([2, 3]), np.array([4, 16], dtype=np.uint64)
        )
        self.assertEqual(ptrs.tolist(), [[1040, 2060], [1048, 2072]])
        offsets, ptrs, sizes = store.resolve_matrices(
            np.array([2]),
            np.array([4], dtype=np.uint64),
            segment_mask=layout.group_layouts[0].segment_mask(layer_ids=[1]),
        )
        self.assertEqual(
            (offsets.tolist(), ptrs.tolist(), sizes.tolist()), ([[8]], [[2060]], [[12]])
        )

    def test_multiple_blocks_are_packed_per_chunk_in_table_order(self):
        layout = self._two_layers(tokens_per_chunk=16)
        offsets, ptrs, sizes = layout.store_layouts[0].resolve_matrices(
            np.array([3, 1, 0, 2]), np.array([0, 16], dtype=np.uint64)
        )
        self.assertEqual(
            ptrs.tolist(), [[1048, 2072, 1016, 2024], [1000, 2000, 1032, 2048]]
        )
        self.assertEqual(offsets.tolist(), [[0, 16, 40, 56]] * 2)
        self.assertEqual(sizes.tolist(), [[16, 24, 16, 24]] * 2)
        self.assertEqual(layout.store_layouts[0].store_bytes, 80)

    def test_swa_tail_starts_at_end_of_partial_first_block(self):
        names = ("model.layers.0.attn", "model.layers.1.swa_cache")
        layout = make_layout(
            config(
                group([names[0]], FullAttentionSpec(8)),
                group([names[1]], SlidingWindowMLASpec(8, sliding_window=12)),
            ),
            {
                names[0]: FakeTensor(100, (8, 1, 8, 1), (8, 8, 1, 1)),
                names[1]: FakeTensor(1000, (8, 1, 8, 2), (16, 16, 2, 1)),
            },
        )
        offsets, ptrs, sizes = layout.store_layouts[1].resolve_matrices(
            np.array([1, 2]), np.array([0], dtype=np.uint64)
        )
        self.assertEqual(
            (offsets.tolist(), ptrs.tolist(), sizes.tolist()),
            ([[0, 8]], [[1024, 1032]], [[8, 16]]),
        )
        self.assertEqual(layout.spec.blocks_per_chunk, (1, 2))

    def test_state_saves_one_complete_snapshot_when_chunk_spans_many_blocks(self):
        names = ("model.layers.0.attn", "model.layers.1.state")
        layout = make_layout(
            config(
                group([names[0]], FullAttentionSpec(8)), group([names[1]], MambaSpec(8))
            ),
            {
                names[0]: FakeTensor(100, (8, 1, 8, 1), (8, 8, 1, 1)),
                names[1]: FakeTensor(1000, (8, 1, 1, 12), (32, 12, 12, 1)),
            },
            tokens_per_chunk=32,
        )
        store = layout.store_layouts[1]
        offsets, ptrs, sizes = store.resolve_matrices(
            np.array([3]), np.array([0], dtype=np.uint64)
        )
        self.assertEqual(
            (offsets.tolist(), ptrs.tolist(), sizes.tolist()), ([[0]], [[1096]], [[12]])
        )
        self.assertEqual(layout.spec.blocks_per_chunk, (4, 1))

    def test_block_matrix_shape_matches_chunk_contract(self):
        store = self._two_layers().store_layouts[0]
        with self.assertRaises(ValueError):
            store.resolve_matrices(
                np.array([0, 1, 2]), np.array([0, 4], dtype=np.uint64)
            )

    def test_empty_block_batch_and_empty_selection_keep_matrix_shape(self):
        store = self._two_layers().store_layouts[0]
        for array in store.resolve_matrices(
            np.array([], dtype=np.uint64), np.array([], dtype=np.uint64)
        ):
            self.assertEqual(array.shape, (0, 2))
        for array in store.resolve_matrices(
            np.array([1]),
            np.array([0], dtype=np.uint64),
            segment_mask=np.array([False, False]),
        ):
            self.assertEqual(array.shape, (1, 0))

    def test_retained_transfers_cannot_mutate_templates_or_later_calls(self):
        store = self._two_layers().store_layouts[0]
        first = store.resolve_matrices(np.array([1]), np.array([0], dtype=np.uint64))
        second = store.resolve_matrices(np.array([1]), np.array([0], dtype=np.uint64))
        expected = tuple(a.copy() for a in second)
        first[1][:] = 0
        for a in (first[0], first[2]):
            a.base[:] = 0
        for actual, wanted in zip(second, expected):
            np.testing.assert_array_equal(actual, wanted)
        for actual, wanted in zip(
            store.resolve_matrices(np.array([1]), np.array([0], dtype=np.uint64)),
            expected,
        ):
            np.testing.assert_array_equal(actual, wanted)

    def test_zero_tail_has_no_placeholder_storage(self):
        names = ("model.layers.0.attn", "model.layers.1.compressor.state_cache")
        layout = make_layout(
            config(
                group([names[0]], MLAAttentionSpec(8, tokens_per_state=4)),
                group([names[1]], SlidingWindowMLASpec(4, sliding_window=4)),
            ),
            {
                names[0]: FakeTensor(100, (8, 1, 2, 1), (2, 2, 1, 1)),
                names[1]: FakeTensor(1000, (8, 1, 4, 1), (4, 4, 1, 1)),
            },
            ratios={1: 4},
        )
        self.assertEqual(layout.spec.blocks_per_chunk, (1, 0))
        self.assertEqual(layout.store_layouts[1].store_bytes, 0)

    def _block_first(self, layerwise):
        names = ("model.layers.0.attn", "model.layers.1.attn")
        descriptor = SimpleNamespace(
            layers=names, offset=0, layer_stride=16, block_stride=64
        )
        return make_layout(
            config(
                group(names, FullAttentionSpec(8)),
                tensors=(descriptor,),
                layout="BLNHC",
            ),
            {
                names[0]: FakeTensor(1000, (8, 1, 8, 1), (64, 8, 1, 1)),
                names[1]: FakeTensor(1016, (8, 1, 8, 1), (64, 8, 1, 1)),
            },
            layerwise=layerwise,
        )

    def test_block_first_bulk_preserves_group_padding_without_global_slack(self):
        layout = self._block_first(False)
        store = layout.store_layouts[0]
        self.assertEqual(store.store_bytes, 32)
        self.assertEqual(
            store.resolve_matrices(np.array([2]), np.array([0], dtype=np.uint64))[
                1
            ].tolist(),
            [[1128]],
        )
        self.assertEqual(
            store.resolve_matrices(np.array([2]), np.array([0], dtype=np.uint64))[
                2
            ].tolist(),
            [[32]],
        )
        offsets, ptrs, sizes = store.resolve_matrices(
            np.array([2]),
            np.array([0], dtype=np.uint64),
            segment_mask=layout.group_layouts[0].segment_mask(layer_ids=[1]),
        )
        self.assertEqual(
            (offsets.tolist(), ptrs.tolist(), sizes.tolist()), ([[16]], [[1144]], [[8]])
        )

    def test_block_first_layerwise_always_uses_compact_payload(self):
        layout = self._block_first(True)
        store = layout.store_layouts[0]
        self.assertEqual(store.store_bytes, 16)
        self.assertEqual(
            store.resolve_matrices(np.array([2]), np.array([0], dtype=np.uint64))[
                0
            ].tolist(),
            [[0, 8]],
        )
        self.assertEqual(
            store.resolve_matrices(np.array([2]), np.array([0], dtype=np.uint64))[
                2
            ].tolist(),
            [[8, 8]],
        )

    def test_multiple_descriptor_spans_include_state_page_padding(self):
        names = ("model.layers.0.state", "model.layers.1.state")
        raw = config(
            group(["model.layers.2.attn"], FullAttentionSpec(8)),
            group(names, MambaSpec(8)),
            tensors=(
                SimpleNamespace(
                    layers=(names[0],), offset=0, layer_stride=16, block_stride=64
                ),
                SimpleNamespace(
                    layers=(names[1],), offset=16, layer_stride=24, block_stride=64
                ),
            ),
            layout="BLNHC",
        )
        layout = make_layout(
            raw,
            {
                "model.layers.2.attn": FakeTensor(4000, (8, 1, 8, 1), (8, 8, 1, 1)),
                names[0]: FakeTensor(1000, (8, 1, 1, 12), (64, 12, 12, 1)),
                names[1]: FakeTensor(1016, (8, 1, 1, 20), (64, 20, 20, 1)),
            },
        )
        store = layout.store_layouts[1]
        self.assertEqual(store.store_bytes, 40)
        self.assertEqual(
            store.resolve_matrices(np.array([2]), np.array([0], dtype=np.uint64))[
                2
            ].tolist(),
            [[40]],
        )

    def test_split_heads_follow_physical_addresses_and_reject_partial_tokens(self):
        name = "model.layers.0.attn"
        raw = config(group([name], FullAttentionSpec(8)), num_blocks=3)
        tensor = FakeTensor(1000, (6, 2, 4, 2), (32, 192, 2, 1))
        layout = make_layout(raw, {name: tensor})
        store = layout.store_layouts[0]
        self.assertEqual(
            store.resolve_matrices(np.array([1]), np.array([0], dtype=np.uint64))[
                1
            ].tolist(),
            [[1064, 1096, 1256, 1288]],
        )
        self.assertEqual(
            store.resolve_matrices(np.array([1]), np.array([0], dtype=np.uint64))[
                2
            ].tolist(),
            [[8, 8, 8, 8]],
        )
        with self.assertRaisesRegex(ValueError, "whole-block"):
            make_layout(raw, {name: tensor}, tokens_per_chunk=4, scheduler=4)


class SchedulerChunkContractTest(unittest.TestCase):
    def test_fa_subchunks_repeat_physical_id_in_shared_chunk_order(self):
        raw = config(group(["model.layers.0.attn"], FullAttentionSpec(8)))
        spec = parse_kv_cache_config(
            raw,
            scheduler_block_size=4,
            tokens_per_chunk=4,
            num_hidden_layers=64,
            device_type="cpu",
        )
        dispatcher = make_dispatcher(spec)
        keys = dispatcher._chain_keys(tuple(range(16)))
        state = seed_request_state(
            dispatcher, FakeRequest("r", 16), group_ucm_block_ids=keys
        )
        state.group_vllm_block_ids = ([2, 3],)
        meta = _scheduler_metadata(dispatcher, {"r": 16})
        (plan,) = meta.requests["r"].dump_plans
        self.assertEqual(plan.group_block_ids[0].tolist(), [2, 2, 3, 3])
        self.assertEqual(plan.token_starts.tolist(), [0, 4, 8, 12])
        layout = UCMKVCacheLayout(
            spec,
            {"model.layers.0.attn": FakeTensor(1000, (8, 1, 8, 1), (8, 8, 1, 1))},
            kv_cache_config=raw,
            num_hidden_layers=64,
        )
        (transfer,) = UCMTransferBuilder(layout).build_dump_transfers(meta)
        self.assertEqual(transfer.ptrs.tolist(), [[1016], [1020], [1024], [1028]])

    def test_swa_short_boundary_is_skipped_and_complete_tail_gathers_two_blocks(self):
        raw = config(
            group(["model.layers.0.attn"], FullAttentionSpec(8)),
            group(
                ["model.layers.1.swa_cache"], SlidingWindowMLASpec(8, sliding_window=12)
            ),
        )
        spec = parse_kv_cache_config(raw, scheduler_block_size=8, num_hidden_layers=64)
        dispatcher = make_dispatcher(spec)
        state = seed_request_state(
            dispatcher,
            FakeRequest("r", 32),
            group_ucm_block_ids=dispatcher._chain_keys(tuple(range(32))),
        )
        state.group_vllm_block_ids = ([1, 2, 3, 4], [4, 5, 6, 7])
        first = _scheduler_metadata(dispatcher, {"r": 8}).requests["r"]
        self.assertEqual(tuple(p.hash_group for p in first.dump_plans), ("FA",))
        second = _scheduler_metadata(dispatcher, {"r": 8}).requests["r"]
        (wa,) = (p for p in second.dump_plans if p.hash_group == "WA")
        self.assertEqual(wa.group_block_ids[0].tolist(), [4, 5])

    def test_state_offloads_use_exact_handoff_and_initial_hit_filter(self):
        raw = config(
            group(["model.layers.0.attn"], FullAttentionSpec(8)),
            group(["model.layers.1.state"], MambaSpec(8)),
        )
        spec = parse_kv_cache_config(raw, scheduler_block_size=8, num_hidden_layers=64)
        dispatcher = make_dispatcher(spec)
        state = seed_request_state(
            dispatcher,
            FakeRequest("r", 64),
            external_hit_tokens=8,
            group_ucm_block_ids=dispatcher._chain_keys(tuple(range(64))),
        )
        state.load_pending = False
        state.token_processed = 32
        state.group_vllm_block_ids = ([1] * 8, [0] * 8)
        meta = _scheduler_metadata(
            dispatcher, {"r": 0}, offloads={"r": [(1, 3, 8), (1, 5, 16)]}
        )
        (plan,) = meta.requests["r"].dump_plans
        self.assertEqual(
            (
                plan.hash_group,
                plan.token_starts.tolist(),
                plan.group_block_ids[0].tolist(),
            ),
            ("State", [8], [5]),
        )
        self.assertTrue(meta.no_forward)

    def test_state_load_targets_follow_platform_copy_order(self):
        for device, targets in (("cpu", [6, 5]), ("npu", [5])):
            with self.subTest(device=device):
                raw = config(
                    group(["model.layers.0.attn"], FullAttentionSpec(8)),
                    group(["model.layers.1.state"], MambaSpec(8)),
                )
                spec = parse_kv_cache_config(
                    raw,
                    scheduler_block_size=8,
                    num_hidden_layers=64,
                    device_type=device,
                )
                dispatcher = make_dispatcher(spec)
                state = seed_request_state(
                    dispatcher,
                    FakeRequest("r", 32),
                    external_hit_tokens=8,
                    group_ucm_block_ids=dispatcher._chain_keys(tuple(range(32))),
                )
                state.group_vllm_block_ids = ([2, 3, 4, 7], [5, 6, 0, 0])
                meta = _scheduler_metadata(dispatcher, {"r": 8})
                self.assertEqual(
                    [
                        int(p.group_block_ids[0][0])
                        for p in meta.requests["r"].load_plans
                        if p.hash_group == "State"
                    ],
                    targets,
                )
                self.assertFalse(
                    _scheduler_metadata(dispatcher, {"r": 8}).requests["r"].load_plans
                )

    def test_preemption_and_finish_remove_request_snapshots(self):
        spec = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(8))),
            scheduler_block_size=8,
            num_hidden_layers=64,
        )
        dispatcher = make_dispatcher(spec)
        for rid in ("preempt", "finish"):
            seed_request_state(dispatcher, FakeRequest(rid, 8))
        _scheduler_metadata(
            dispatcher, {}, preempted_req_ids=("preempt",), finished_req_ids=("finish",)
        )
        self.assertEqual(dispatcher.requests, {})


class TransferMatrixTest(unittest.TestCase):
    def _layout(self):
        names = ("model.layers.0.attn", "model.layers.0.indexer")
        raw = config(*(group([name], FullAttentionSpec(8)) for name in names))
        return make_layout(
            raw,
            {
                names[0]: FakeTensor(1000, (8, 1, 8, 1), (8, 8, 1, 1)),
                names[1]: FakeTensor(2000, (8, 1, 8, 2), (16, 16, 2, 1)),
            },
        )

    def test_group_record_offsets_are_added_and_layer_id_selects_all_caches(self):
        layout = self._layout()
        builder = UCMTransferBuilder(layout)
        meta = metadata("FA", ([2], [3]))
        (transfer,) = builder.build_dump_transfers(meta, layer_id=0)
        self.assertEqual(transfer.ptrs.tolist(), [[1016, 2048]])
        self.assertEqual(transfer.ucm_block_offsets.tolist(), [[0, 8]])
        self.assertEqual(transfer.sizes.tolist(), [[8, 16]])
        (selected,) = builder.build_dump_transfers(
            meta, layer_name="model.layers.0.indexer"
        )
        self.assertEqual(selected.ucm_block_offsets.tolist(), [[8]])

    def test_shared_key_for_different_requests_keeps_distinct_targets(self):
        builder = UCMTransferBuilder(self._layout())
        first = metadata("FA", ([2], [3]), load=True).requests["r"]
        second = metadata("FA", ([4], [5]), load=True).requests["r"]
        batches = builder.build_load_transfers(
            UCMConnectorMetadata(requests={"a": first, "b": second})
        )
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].keys, batches[1].keys)
        self.assertEqual(batches[1].ptrs.tolist(), [[1032, 2080]])

    def test_real_file_roundtrip_uses_different_source_and_destination_blocks(self):
        layout = self._layout()
        memory = ByteMemory()
        memory.write(1008, b"abcdefgh")
        memory.write(2016, b"0123456789abcdef")
        memory.write(1032, b"!" * 8)
        memory.write(2064, b"!" * 16)
        builder = UCMTransferBuilder(layout)
        with tempfile.TemporaryDirectory() as path:
            adapter = UCMProxyAdapter(
                SimpleFileUCMProxy(path, MemoryByteAccess(memory))
            )
            for batch in builder.build_dump_transfers(metadata("FA", ([1], [1]))):
                adapter.submit("dump", batch)
                adapter.commit(batch.keys)
            for batch in builder.build_load_transfers(
                metadata("FA", ([4], [4]), load=True)
            ):
                adapter.submit("load", batch)
        self.assertEqual(memory.read(1032, 8), b"abcdefgh")
        self.assertEqual(memory.read(2064, 16), b"0123456789abcdef")

    def test_adapter_retains_async_buffers_and_accepts_broadcast_descriptors(self):
        proxy = SimpleNamespace(
            dump=mock.Mock(return_value="in-flight"), wait=mock.Mock()
        )
        adapter = UCMProxyAdapter(proxy)
        ptrs = np.arange(12, dtype=np.uint64).reshape(3, 4)[:, ::2]
        sizes = np.broadcast_to(np.array([2, 3], dtype=np.uint64), ptrs.shape)
        offsets = np.broadcast_to(np.array([0, 2], dtype=np.uint64), ptrs.shape)
        batch = UCMProxyTransfer(
            (b"a" * 16, b"b" * 16, b"c" * 16), ptrs, sizes, offsets
        )
        task = adapter.enqueue("dump", batch)
        self.assertIs(task.transfer.ptrs, ptrs)
        self.assertIs(proxy.dump.call_args.args[2], ptrs)
        adapter.wait(task)
        proxy.wait.assert_called_once_with("in-flight")


class WorkerLifecycleTest(unittest.TestCase):
    def _worker(self, directory, *, layerwise=True):
        memory = ByteMemory()
        backend = SimpleFileUCMProxy(directory, MemoryByteAccess(memory))
        cfg = vllm_config()
        cfg.cache_config.block_size = 8
        cfg.parallel_config.tensor_parallel_size = cfg.parallel_config.world_size = 1
        cfg.kv_transfer_config.kv_connector_extra_config.update(
            v2_storage_path=directory, use_layerwise=layerwise
        )
        names = ("model.layers.0.attn", "model.layers.1.attn")
        with mock.patch(
            "ucm.integration.vllm.v2.ucm_connector.SimpleFileUCMProxy",
            return_value=backend,
        ):
            worker = UCMConnector(
                cfg, KVConnectorRole.WORKER, config(group(names, FullAttentionSpec(8)))
            )
        worker.register_kv_caches(
            {
                names[0]: FakeTensor(1000, (8, 1, 8, 1), (8, 8, 1, 1)),
                names[1]: FakeTensor(2000, (8, 1, 8, 1), (8, 8, 1, 1)),
            }
        )
        return worker, backend, memory, names

    def test_layerwise_commit_occurs_after_all_layers_then_step_state_resets(self):
        with tempfile.TemporaryDirectory() as path:
            worker, backend, memory, names = self._worker(path)
            memory.write(1008, b"abcdefgh")
            memory.write(2008, b"01234567")
            worker.bind_connector_metadata(metadata("FA", ([1],)))
            worker.save_kv_layer(names[0], None, None)
            self.assertEqual(backend.lookup((b"a" * 16,)), (False,))
            worker.wait_for_save()
            self.assertEqual(backend.lookup((b"a" * 16,)), (True,))
            worker.bind_connector_metadata(metadata("FA", ([3],), load=True))
            worker.start_load_kv(None)
            worker.wait_for_layer_load(names[0])
            worker.wait_for_layer_load(names[1])
            self.assertEqual(memory.read(1024, 8), b"abcdefgh")
            self.assertEqual(memory.read(2024, 8), b"01234567")
            worker.wait_for_save()

    def test_missing_record_reports_failed_blocks_and_suppresses_save(self):
        with tempfile.TemporaryDirectory() as path:
            worker, backend, memory, names = self._worker(path, layerwise=False)
            meta = metadata("FA", ([3],), load=True)
            req = meta.requests["r"]
            meta.requests["r"] = replace(
                req, dump_plans=metadata("FA", ([3],)).requests["r"].dump_plans
            )
            worker.bind_connector_metadata(meta)
            worker.start_load_kv(None)
            self.assertEqual(
                worker.build_connector_worker_meta().load_failed_reqs, {"r"}
            )
            self.assertEqual(worker.get_block_ids_with_load_errors(), {3})
            worker.wait_for_save()
            self.assertEqual(backend.lookup((b"a" * 16,)), (False,))

    def test_chunk_launch_option_reaches_shared_spec(self):
        cfg = vllm_config()
        cfg.kv_transfer_config.kv_connector_extra_config["tokens_per_chunk"] = 16
        with tempfile.TemporaryDirectory() as directory:
            cfg.kv_transfer_config.kv_connector_extra_config["v2_storage_path"] = (
                directory
            )
            worker = UCMConnector(
                cfg,
                KVConnectorRole.SCHEDULER,
                config(group(["model.layers.0.attn"], FullAttentionSpec(4))),
            )
        self.assertEqual(
            (worker.spec.tokens_per_chunk, worker.spec.blocks_per_chunk), (16, (4,))
        )

    def test_failed_deferred_write_drains_remaining_tasks_without_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            worker, backend, memory, names = self._worker(directory)

            class DeferredProxy:
                def __init__(self):
                    self.waited = []
                    self.committed = []

                def dump(self, *args):
                    return args

                def wait(self, args):
                    self.waited.append(args)
                    if len(self.waited) == 1:
                        raise OSError("first deferred copy failed")
                    backend.dump(*args)

                def commit(self, keys):
                    self.committed.append(keys)
                    backend.commit(keys)

            proxy = DeferredProxy()
            worker._proxy = UCMProxyAdapter(proxy)
            worker.bind_connector_metadata(metadata("FA", ([1],)))
            with self.assertRaises(UCMProxyError):
                worker.wait_for_save()
            self.assertEqual(len(proxy.waited), 2)
            self.assertEqual(proxy.committed, [])
            self.assertEqual(worker._dump_tasks, [])
            self.assertEqual(backend.lookup((b"a" * 16,)), (False,))

    def test_raw_config_dump_preserves_native_descriptor_fields(self):
        from ucm.integration.vllm.v2.ucm_connector import _dump_raw_kv_cache_config

        raw = config(
            group(["model.layers.0.attn"], FullAttentionSpec(8)),
            tensors=(
                SimpleNamespace(
                    layers=("model.layers.0.attn",),
                    offset=0,
                    layer_stride=64,
                    block_stride=8,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "rank-%d.json"
            with mock.patch.dict(os.environ, {"UCM_V2_DUMP_CONFIG": str(target)}):
                _dump_raw_kv_cache_config(raw, 3)
            payload = json.loads((Path(directory) / "rank-3.json").read_text())
            self.assertEqual(payload["kv_cache_tensors"][0]["block_stride"], 8)
            self.assertEqual(payload["prefix_cache_retention_interval"], 0)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch for native dtype views")
class StateBytePageTest(unittest.TestCase):
    def test_dense_permutation_preserves_storage_offset_stride_and_aliasing(self):
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
            device="npu",
        )
        _, ptrs, sizes = layout.store_layouts[1].resolve_matrices(
            np.array([2]), np.array([0], dtype=np.uint64)
        )
        self.assertEqual(ptrs.tolist(), [[component.data_ptr() + 512]])
        self.assertEqual(sizes.tolist(), [[24]])

    def test_non_dense_state_payload_is_rejected(self):
        from ucm.integration.vllm.v2.layout.view import _as_byte_page

        with self.assertRaisesRegex(ValueError, "dense"):
            _as_byte_page(torch.zeros((4, 8, 6))[:, :, ::2])


class ParallelShardTest(unittest.TestCase):
    def test_lookup_requires_all_committed_shards_and_loads_own_bytes(self):
        from ucm.integration.vllm.v2.parallel import AllShardLookup

        key = b"p" * 16
        with tempfile.TemporaryDirectory() as directory:
            memory = ByteMemory()
            memory.write(100, b"rankzero")
            memory.write(200, b"rank-one")
            shards = [
                SimpleFileUCMProxy(Path(directory) / str(i), MemoryByteAccess(memory))
                for i in range(2)
            ]
            scheduler = AllShardLookup(shards[0], shards)
            for i in range(2):
                shards[i].dump(
                    (key,),
                    np.array([[0]], dtype=np.uint64),
                    np.array([[100 + i * 100]], dtype=np.uint64),
                    np.array([[8]], dtype=np.uint64),
                )
            shards[0].commit((key,))
            self.assertEqual(scheduler.lookup((key,)), (False,))
            self.assertEqual(scheduler.lookup_on_prefix((key,)), -1)
            self.assertEqual(scheduler.lookup_on_reverse((key,)), -1)
            shards[1].commit((key,))
            self.assertEqual(scheduler.lookup((key,)), (True,))
            for i in range(2):
                AllShardLookup(shards[i], shards).load(
                    (key,),
                    np.array([[0]], dtype=np.uint64),
                    np.array([[300 + 100 * i]], dtype=np.uint64),
                    np.array([[8]], dtype=np.uint64),
                )
            self.assertEqual(memory.read(300, 8), b"rankzero")
            self.assertEqual(memory.read(400, 8), b"rank-one")

    def test_empty_pp_shard_marker_is_not_visible_before_commit(self):
        from ucm.integration.vllm.v2.parallel import AllShardLookup

        key = b"s" * 16
        empty = np.empty((1, 0), dtype=np.uint64)
        with tempfile.TemporaryDirectory() as directory:
            shards = [
                SimpleFileUCMProxy(
                    Path(directory) / str(i), MemoryByteAccess(ByteMemory())
                )
                for i in range(2)
            ]
            shards[0].dump((key,), empty, empty, empty)
            shards[1].dump((key,), np.array([[0]]), np.array([[10]]), np.array([[4]]))
            shards[1].commit((key,))
            scheduler = AllShardLookup(shards[0], shards)
            self.assertEqual(scheduler.lookup((key,)), (False,))
            shards[0].commit((key,))
            self.assertEqual(scheduler.lookup((key,)), (True,))
            self.assertEqual(shards[0]._path(key).stat().st_size, 0)

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


class PhysicalOrderOracleTest(unittest.TestCase):
    def test_all_six_physical_orders_match_independent_element_addresses(self):
        # The oracle enumerates logical elements from physical axis strides;
        # it does not call UCM's range compiler to compute expected addresses.
        dims = {"L": 2, "B": 4, "H": 2, "N": 8, "C": 2}
        names = ("model.layers.0.attn", "model.layers.1.attn")
        for order in ("LBNHC", "LBHNC", "BLNHC", "BLHNC", "LHBNC", "BHLNC"):
            with self.subTest(order=order):
                strides, stride = {}, 1
                for axis in reversed(order):
                    strides[axis] = stride
                    stride *= dims[axis]
                raw = config(group(names, FullAttentionSpec(8)), num_blocks=4)
                caches = {
                    name: FakeTensor(
                        1000 + layer * strides["L"],
                        (4, 2, 8, 2),
                        tuple(strides[axis] for axis in "BHNC"),
                    )
                    for layer, name in enumerate(names)
                }
                layout = make_layout(raw, caches, layerwise=True)
                _, ptrs, sizes = layout.store_layouts[0].resolve_matrices(
                    np.array([2]), np.array([0], dtype=np.uint64)
                )
                actual = [
                    address
                    for ptr, size in zip(ptrs[0], sizes[0])
                    for address in range(int(ptr), int(ptr) + int(size))
                ]
                expected = []
                for layer in range(2):
                    expected.extend(
                        sorted(
                            1000
                            + layer * strides["L"]
                            + 2 * strides["B"]
                            + head * strides["H"]
                            + token * strides["N"]
                            + channel
                            for head in range(2)
                            for token in range(8)
                            for channel in range(2)
                        )
                    )
                self.assertEqual(actual, expected)
                if order.endswith("NHC"):
                    partial = make_layout(raw, caches, tokens_per_chunk=4, scheduler=4)
                    _, ptrs, sizes = partial.store_layouts[0].resolve_matrices(
                        np.array([2]), np.array([4], dtype=np.uint64)
                    )
                    actual = [
                        a
                        for ptr, size in zip(ptrs[0], sizes[0])
                        for a in range(int(ptr), int(ptr) + int(size))
                    ]
                    expected = []
                    for layer in range(2):
                        expected.extend(
                            sorted(
                                1000
                                + layer * strides["L"]
                                + 2 * strides["B"]
                                + head * strides["H"]
                                + token * strides["N"]
                                + channel
                                for head in range(2)
                                for token in range(4, 8)
                                for channel in range(2)
                            )
                        )
                    self.assertEqual(actual, expected)
                else:
                    with self.assertRaisesRegex(ValueError, "whole-block"):
                        make_layout(raw, caches, tokens_per_chunk=4, scheduler=4)

    def test_cp_global_token_span_keeps_local_physical_rows(self):
        name = "model.layers.0.attn"
        raw = config(group([name], FullAttentionSpec(8)))
        spec = parse_kv_cache_config(
            raw, scheduler_block_size=8, num_hidden_layers=64, device_type="cpu"
        )
        spec = ParallelLayout(dcp=2).apply_context_parallel(spec, 32)
        layout = UCMKVCacheLayout(
            spec,
            {name: FakeTensor(1000, (8, 1, 8, 1), (8, 8, 1, 1))},
            kv_cache_config=raw,
            num_hidden_layers=64,
        )
        self.assertEqual(layout.spec.blocks_per_chunk, (2,))
        _, ptrs, sizes = layout.store_layouts[0].resolve_matrices(
            np.array([1, 3]), np.array([0], dtype=np.uint64)
        )
        self.assertEqual((ptrs.tolist(), sizes.tolist()), ([[1008, 1024]], [[8, 8]]))

    def test_native_replicated_indexer_rows_use_one_logical_block_stride(self):
        name = "model.layers.0.indexer"
        raw = config(
            group(
                [name], AscendSFAIndexerCacheSpec(8, sfa_dcp_replicated_indexer_size=2)
            )
        )
        layout = make_layout(
            raw, {name: FakeTensor(1000, (16, 8, 1, 1), (8, 1, 1, 1))}, device="npu"
        )
        self.assertEqual(layout.group_layouts[0].resolve_ptrs([2]).tolist(), [[1032]])
        self.assertEqual(layout.store_layouts[0].store_bytes, 16)


if __name__ == "__main__":
    unittest.main()
