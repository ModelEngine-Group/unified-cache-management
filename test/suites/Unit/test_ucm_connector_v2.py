import enum
import inspect
import json
import math
import os
import sys
import tempfile
import types
import numpy as np
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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
sys.modules[vllm_base.__name__] = vllm_base

vllm_model_utils = types.ModuleType("vllm.model_executor.models.utils")


def extract_layer_index(layer_name, num_attn_module=1):
    subnames = layer_name.split(".")
    int_vals = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
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
    MAMBA = "mamba"
    UNKNOWN = "unknown"


class _KVCacheSpec:
    pass


class _AttentionSpec(_KVCacheSpec):
    pass


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
sys.modules[vllm_kv_cache_interface.__name__] = vllm_kv_cache_interface

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
        size = int(vllm_config.cache_config.block_size) * dcp
        return size, size
    scheduler = math.lcm(*(int(g.kv_cache_spec.block_size) for g in groups))
    return scheduler, scheduler


vllm_kv_cache_utils.resolve_kv_cache_block_sizes = resolve_kv_cache_block_sizes
sys.modules.setdefault(vllm_kv_cache_utils.__name__, vllm_kv_cache_utils)

from ucm.integration.vllm.v2.ucm_kv_cache import (  # noqa: E402
    UCMKVCacheLayout,
    parse_kv_cache_config,
)
from ucm.integration.vllm.v2.ucm_connector import (  # noqa: E402
    UCMConnector,
    UCMWorkerMetadata,
)
from ucm.integration.vllm.v2.ucm_proxy import (  # noqa: E402
    SimpleFileUCMProxy,
    UCMProxyAdapter,
    UCMProxyError,
)
from ucm.integration.vllm.v2.ucm_scheduler import (  # noqa: E402
    RequestHasher,
    RequestState,
    UCMConnectorMetadata,
    UCMDispatcher,
)


@dataclass
class FullAttentionSpec(_FullAttentionSpec):
    block_size: int
    tokens_per_state: int = 1
    sliding_window: int | None = None


@dataclass
class MLAAttentionSpec(_MLAAttentionSpec):
    block_size: int
    tokens_per_state: int = 1
    sliding_window: int | None = None


@dataclass
class AscendMLAAttentionSpec(MLAAttentionSpec):
    """Ascend 0.26 spelling: the compression ratio is compress_ratio."""

    compress_ratio: int = 1


@dataclass
class MambaSpec(_MambaSpec):
    block_size: int
    mamba_cache_mode: str = "align"
    shapes: tuple[tuple[int, ...], ...] | None = None
    dtypes: tuple[object, ...] | None = None
    page_size_bytes: int | None = None


@dataclass
class AscendSlidingWindowMLASpec(_SlidingWindowMLASpec):
    block_size: int
    compress_ratio: int
    sliding_window: int
    tokens_per_state: int = 1


@dataclass
class UniformTypeKVCacheSpecs:
    kv_cache_specs: dict[str, object]

    @property
    def block_size(self):
        return next(iter(self.kv_cache_specs.values())).block_size


def group(names, specs):
    if not isinstance(specs, dict):
        spec = specs
    else:
        spec = UniformTypeKVCacheSpecs(specs)
    return SimpleNamespace(layer_names=list(names), kv_cache_spec=spec)


def config(*groups, num_blocks=8, tensors=()):
    return SimpleNamespace(
        kv_cache_groups=list(groups),
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
    )


def captured_spec(value, layer_name=""):
    fields = {key: item for key, item in value.items() if key != "kv_cache_specs"}
    if "kv_cache_specs" in value:
        nested = {
            name: captured_spec(item, name)
            for name, item in value["kv_cache_specs"].items()
        }
        spec_type = type("UniformTypeKVCacheSpecs", (), {})
        result = spec_type()
        result.kv_cache_specs = nested
        result.block_size = value["block_size"]
        return result
    if "shapes" in value:
        name = "MambaSpec"
        base = _MambaSpec
    elif fields.get("sliding_window") is not None:
        name = "AscendSlidingWindowMLASpec"
        base = _SlidingWindowMLASpec
    else:
        name = "MLAAttentionSpec"
        base = _MLAAttentionSpec
    spec_type = type(name, (base,), {})
    result = spec_type()
    for key, item in fields.items():
        setattr(result, key, item)
    return result


def captured_config(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = payload[0]
    groups = []
    for item in payload["kv_cache_groups"]:
        groups.append(
            SimpleNamespace(
                layer_names=item["layer_names"],
                kv_cache_spec=captured_spec(item["kv_cache_spec"]),
            )
        )
    return SimpleNamespace(kv_cache_groups=groups)


def vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(model="org/model", dtype="torch.bfloat16"),
        cache_config=SimpleNamespace(block_size=4),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "v2_storage_path": str(Path(tempfile.gettempdir()) / "ucm-v2-test")
            }
        ),
        device_config=SimpleNamespace(device_type="cpu"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            rank=0,
        ),
        speculative_config=None,
        additional_config={},
    )


class FakeRequest:
    def __init__(self, request_id, token_count):
        self.request_id = request_id
        self.all_token_ids = list(range(token_count))


def make_dispatcher(parsed, proxy=None, hasher=None, **kwargs):
    """A UCMDispatcher for dispatch-only tests (lookup never runs)."""
    return UCMDispatcher(
        parsed,
        UCMProxyAdapter(proxy or FakeProxy()),
        hasher or RequestHasher(vllm_config(), 0),
        b"seed",
        **kwargs,
    )


def seed_request_state(
    dispatcher,
    request,
    external_hit_tokens=0,
    group_ucm_block_ids=(),
    hbm_hit_tokens=0,
):
    """Record a synthetic lookup result the way UCMDispatcher.lookup would."""
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


class FakeProxy:
    def __init__(self):
        self.present = set()
        self.lookup_calls = []
        self.load_calls = []
        self.dump_calls = []

    def lookup(self, keys):
        self.lookup_calls.append(tuple(keys))
        return [key in self.present for key in keys]

    def lookup_on_prefix(self, keys):
        for index, key in enumerate(keys):
            if key not in self.present:
                return index - 1
        return len(keys) - 1

    def lookup_on_reverse(self, keys):
        for index in range(len(keys) - 1, -1, -1):
            if keys[index] in self.present:
                return index
        return -1

    def load(self, block_ids, offsets, ptrs, sizes):
        self.load_calls.append((block_ids, offsets, ptrs, sizes))

    def dump(self, block_ids, offsets, ptrs, sizes):
        self.dump_calls.append((block_ids, offsets, ptrs, sizes))


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


class InMemoryByteProxy(FakeProxy):
    """Synchronous test-only Proxy; one dump call publishes complete records."""

    def __init__(self, memory):
        super().__init__()
        self.memory = memory
        self.records = {}

    def lookup(self, keys):
        return [key in self.records for key in keys]

    def lookup_on_prefix(self, keys):
        for index, key in enumerate(keys):
            if key not in self.records:
                return index - 1
        return len(keys) - 1

    def lookup_on_reverse(self, keys):
        for index in range(len(keys) - 1, -1, -1):
            if keys[index] in self.records:
                return index
        return -1

    def dump(self, block_ids, offsets, ptrs, sizes):
        pending = {}
        for key, offset, ptr, size in zip(block_ids, offsets, ptrs, sizes):
            # NumPy 1.x uint64 + Python int can promote to float64.
            offset, ptr, size = int(offset), int(ptr), int(size)
            record = pending.setdefault(key, bytearray())
            required = offset + size
            record.extend(b"\x00" * max(required - len(record), 0))
            record[offset:required] = self.memory.read(ptr, size)
        self.records.update({key: bytes(value) for key, value in pending.items()})

    def load(self, block_ids, offsets, ptrs, sizes):
        for key, offset, ptr, size in zip(block_ids, offsets, ptrs, sizes):
            offset, ptr, size = int(offset), int(ptr), int(size)
            self.memory.write(ptr, self.records[key][offset : offset + size])


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
    def __init__(self, ptr, shape, strides, element_size=1):
        self._ptr = ptr
        self.shape = shape
        self._strides = strides
        self._element_size = element_size

    def data_ptr(self):
        return self._ptr

    def stride(self, index):
        return self._strides[index]

    def element_size(self):
        return self._element_size


class KVCacheSpecTest(unittest.TestCase):
    def test_direct_custom_chunk_keeps_base_hash_size(self):
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
            ucm_cache_block_size=512,
        )

        self.assertEqual(parsed.ucm_cache_block_size, 512)
        self.assertEqual(parsed.layer_to_group["model.layers.0.attn"], 0)

    def test_hybrid_classifies_attention_and_state(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            scheduler_block_size=128,
        )

        self.assertEqual(tuple(g.group_id for g in parsed.attn_groups), (0,))
        self.assertEqual(tuple(g.group_id for g in parsed.state_groups), (1,))

    def test_hybrid_rejects_non_align_mamba(self):
        with self.assertRaisesRegex(ValueError, "mamba_cache_mode='align'"):
            parse_kv_cache_config(
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
        with self.assertRaisesRegex(ValueError, "Mamba align block size"):
            parse_kv_cache_config(
                config(
                    group(["model.layers.0.attn"], FullAttentionSpec(128)),
                    group(["model.layers.1.mixer"], MambaSpec(256)),
                ),
                scheduler_block_size=128,
            )

    def test_hybrid_rejects_misaligned_attention_block(self):
        with self.assertRaisesRegex(ValueError, "every KV group block size"):
            parse_kv_cache_config(
                config(
                    group(["model.layers.0.attn"], FullAttentionSpec(256)),
                    group(["model.layers.1.mixer"], MambaSpec(128)),
                ),
                scheduler_block_size=128,
            )

    def test_dsv4_derives_canonical_size_once(self):
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.2.attn"],
                    AscendMLAAttentionSpec(128, compress_ratio=4),
                ),
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 128),
                ),
                group(
                    ["model.layers.2.compressor.state_cache"],
                    AscendSlidingWindowMLASpec(128, 4, 128),
                ),
            ),
            scheduler_block_size=16,
        )

        self.assertEqual(parsed.ucm_cache_block_size, 512)
        self.assertEqual(parsed.groups[2].token_block_size, 512)
        self.assertEqual(
            parsed.groups[2].kinds,
            frozenset((KVCacheSpecKind.SLIDING_WINDOW_MLA,)),
        )
        self.assertTrue(parsed.groups[2].is_attention)
        self.assertTrue(parsed.groups[2].is_sliding_window)

    def test_parse_rejects_non_divisible_fa_token_blocks(self):
        # Flat plan windows need a uniform per-key block count; an FA group
        # whose token block neither divides nor is divided by the cache
        # block would make keys straddle blocks. Fail at startup.
        with self.assertRaisesRegex(ValueError, "neither divides nor is a multiple"):
            parse_kv_cache_config(
                config(
                    group(["model.layers.0.attn"], FullAttentionSpec(512)),
                    group(["model.layers.1.attn"], FullAttentionSpec(768)),
                ),
                scheduler_block_size=256,
            )

    def test_parse_rejects_misaligned_sliding_groups(self):
        # A sliding tail keeps a static window shape: the boundary must
        # land on the group's block grid, so its token block must divide
        # the cache block.
        with self.assertRaisesRegex(ValueError, "must divide"):
            parse_kv_cache_config(
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
        parsed = parse_kv_cache_config(
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

    def test_hybrid_rejects_custom_chunk(self):
        with self.assertRaisesRegex(ValueError, "custom ucm_cache_block_size"):
            parse_kv_cache_config(
                config(
                    group(["model.layers.0.attn"], FullAttentionSpec(128)),
                    group(["model.layers.1.mixer"], MambaSpec(128)),
                ),
                scheduler_block_size=128,
                ucm_cache_block_size=512,
            )

    def test_real_ascend_runtime_captures_classify_groups(self):
        capture_dir = (
            REPO_ROOT.parent
            / "docs"
            / "kv-layout-ascend-20260905"
            / "vllm-ascend-026"
            / "results"
            / "runtime"
        )
        required = {
            "glm": capture_dir / "glm.runtime_kvcache_config.json",
            "kimi": capture_dir / "kimi.runtime_kvcache_config.json",
            "dsv4": capture_dir / "dsv4.runtime_kvcache_config.json",
        }
        if not all(path.exists() for path in required.values()):
            self.skipTest("workspace Ascend runtime captures are unavailable")

        glm = parse_kv_cache_config(
            captured_config(required["glm"]), scheduler_block_size=128
        )
        dsv4 = parse_kv_cache_config(
            captured_config(required["dsv4"]), scheduler_block_size=8
        )

        self.assertEqual(len(glm.groups), 1)
        self.assertEqual(len(dsv4.groups), 6)
        self.assertEqual(dsv4.ucm_cache_block_size, 512)
        self.assertEqual(tuple(group.group_id for group in dsv4.fa_groups), (0, 1))
        self.assertEqual(
            tuple(group.group_id for group in dsv4.wa_groups), (2, 3, 4, 5)
        )
        self.assertEqual(
            tuple(group.tail_tokens for group in dsv4.groups[2:]),
            (128, 128, 4, 0),
        )

        # This A-chain capture predates platform block-size alignment. It has
        # Attention=768, Mamba=1024 and mamba_cache_mode=none, so it must not
        # be accepted as a production Kimi align layout.
        with self.assertRaisesRegex(ValueError, "mamba_cache_mode='align'"):
            parse_kv_cache_config(
                captured_config(required["kimi"]), scheduler_block_size=768
            )

    def test_real_cpu_dsv4_capture_uses_fa_and_wa_groups(self):
        path = (
            REPO_ROOT.parent
            / "docs"
            / "kv-layout-ascend-20260905"
            / "vllm-029"
            / "results"
            / "runtime"
            / "dsv4_official_029_256.runtime_kvcache_config.json"
        )
        if not path.exists():
            self.skipTest("workspace vLLM 0.29 runtime capture is unavailable")

        parsed = parse_kv_cache_config(
            captured_config(path),
            scheduler_block_size=256,
            device_type="cpu",
        )

        self.assertEqual(parsed.ucm_cache_block_size, 256)
        # FA is the 256-token indexer+attention group; WA covers both SWA
        # groups and both compressor state groups.
        self.assertEqual(tuple(group.group_id for group in parsed.fa_groups), (2,))
        self.assertEqual(
            tuple(group.group_id for group in parsed.wa_groups), (0, 1, 3, 4)
        )
        self.assertEqual(
            tuple(group.tail_tokens for group in parsed.wa_groups), (128, 128, 4, 0)
        )
        # Replay the captured 43-layer worker config through the scheduler's
        # representative-spec conversion, using the captured per-layer ratios
        # as the model metadata. Tensor placement is irrelevant on this side.
        ratios = {
            layer.layer_index: layer.kv_cache_spec.tokens_per_state
            for fa_group in parsed.fa_groups
            for layer in fa_group.layers
        }
        scheduler_groups = []
        for raw_group in captured_config(path).kv_cache_groups:
            nested = raw_group.kv_cache_spec.kv_cache_specs
            scheduler_groups.append(
                group(raw_group.layer_names, next(iter(nested.values())))
            )
        scheduler_spec = parse_kv_cache_config(
            config(*scheduler_groups),
            scheduler_block_size=256,
            device_type="cpu",
            attention_tokens_per_state=ratios,
        )
        self.assertEqual(
            tuple(group.tail_tokens for group in scheduler_spec.wa_groups),
            (128, 128, 4, 0),
        )

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
        runtime_config.model_config.hf_text_config = SimpleNamespace(
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
                self.assertEqual(
                    [
                        layer.storage_block_size
                        for layer in scheduler.spec.groups[0].layers
                    ],
                    [64, 2],
                )


class MtpLayerIndexTest(unittest.TestCase):
    def test_native_numbering(self):
        from ucm.integration.vllm.v2.ucm_kv_cache import _layer_index

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
        with self.assertRaisesRegex(ValueError, "num_hidden_layers"):
            _layer_index("mtp.0.attn", "npu", None)

    def test_mtp_identity_is_shared_by_selection_ratio_and_tail(self):
        raw = config(
            group(
                ["model.layers.0.attn"], AscendMLAAttentionSpec(4, compress_ratio=128)
            ),
            group(["mtp.0.attn"], AscendMLAAttentionSpec(4, compress_ratio=128)),
            group(["mtp.0.swa_cache"], AscendSlidingWindowMLASpec(4, 1, 4)),
            group(
                ["mtp.0.compressor.state_cache"], AscendSlidingWindowMLASpec(4, 1, 128)
            ),
        )
        cfg = vllm_config()
        cfg.device_config.device_type = "npu"
        cfg.model_config.hf_text_config = SimpleNamespace(
            num_hidden_layers=4, compress_ratios=[4, 0, 0, 0]
        )
        with tempfile.TemporaryDirectory() as path:
            cfg.kv_transfer_config.kv_connector_extra_config["v2_storage_path"] = path
            parsed = UCMConnector(cfg, KVConnectorRole.SCHEDULER, raw).spec
        self.assertEqual(parsed.groups[1].token_block_size, 512)
        self.assertEqual(parsed.groups[3].tail_tokens, 0)
        names = [layer.layer_name for g in parsed.groups for layer in g.layers]
        layout = UCMKVCacheLayout(
            parsed,
            {
                name: FakeTensor(0x1000 * (i + 1), (8, 4, 1), (4, 1, 1))
                for i, name in enumerate(names)
            },
        )
        self.assertEqual(layout.layer_names_by_id[0], {"model.layers.0.attn"})
        self.assertEqual(set(layout.layer_names_by_id[4]), set(names[1:]))
        # Group-level columns: model layer 0 belongs to group 0 only, so
        # group 1 answers nothing for it; layer 4 names live in group 1.
        self.assertFalse(layout.group_layouts[1].segment_mask(layer_ids=[0]).any())
        group1 = layout.group_layouts[1]
        whole = group1.extract_segments(
            [1], [0], [group1.token_block_size], layer_ids=[4]
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
            scheduler_block_size=128,
            ucm_cache_block_size=512,
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
            scheduler_block_size=128,
            ucm_cache_block_size=512,
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
                    AscendMLAAttentionSpec(128, compress_ratio=4),
                ),
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 128),
                ),
                num_blocks=4,
            ),
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
                    AscendMLAAttentionSpec(128, compress_ratio=4),
                ),
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 128),
                ),
                num_blocks=4,
            ),
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


class LayerwiseLifecycleTest(unittest.TestCase):
    def _worker(self, directory, *, state=False, deferred=False):
        memory = ByteMemory()
        backend = SimpleFileUCMProxy(directory, MemoryByteAccess(memory))
        if deferred:

            class DeferredProxy:
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

            proxy = DeferredProxy()
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
            caches["model.layers.2.state"] = FakeTensor(0x3000, (8, 8), (8, 1))
        cfg = vllm_config()
        cfg.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
        with mock.patch(
            "ucm.integration.vllm.v2.ucm_connector.SimpleFileUCMProxy",
            return_value=proxy,
        ):
            worker = UCMConnector(cfg, KVConnectorRole.WORKER, config(*groups))
        worker.register_kv_caches(caches)
        return worker, backend, proxy, memory, names

    def _meta(self, *, load=False, state=False):
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        kinds = ("FA", "State") if state else ("FA",)
        plans = tuple(
            UCMGroupDispatchPlan(
                kind,
                (bytes([i + 1]) * 16,),
                0,
                4,
                (np.array([2 if load else 1], dtype=np.uint64),),
            )
            for i, kind in enumerate(kinds)
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
            self.assertEqual(backend.lookup((bytes([1]) * 16,)), (False,))
            worker.wait_for_save()
            self.assertEqual(
                [e[0] for e in proxy.events], ["dump", "dump", "wait", "wait", "commit"]
            )
            self.assertEqual(
                backend._path(bytes([1]) * 16).read_bytes(), b"abcdefghijklmnop"
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
                backend.lookup((bytes([1]) * 16, bytes([2]) * 16)), (True, True)
            )
            worker.bind_connector_metadata(self._meta(load=True, state=True))
            worker.start_load_kv(None)
            self.assertEqual(memory.read(0x3010, 8), b"state123")
            worker.wait_for_save()
            self.assertEqual(memory.read(0x2010, 8), b"ijklmnop")

    def test_roundtrip_all_physical_orders_and_partial_blocks(self):
        for order in ("LBNHC", "LBHNC", "LHBNC", "BLNHC", "BLHNC", "BHLNC"):
            for unit in (4, 8):
                with (
                    self.subTest(order=order, unit=unit),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    worker, backend, _, memory, names = self._worker(directory)
                    worker.layout, strides = LayerViewSegmentsTest()._layout(
                        order, unit=unit
                    )
                    worker.spec = worker.layout.spec

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
                    worker.bind_connector_metadata(self._meta())
                    worker.start_load_kv(None)
                    for name in reversed(names):
                        worker.save_kv_layer(name, None, None)
                    worker.wait_for_save()
                    worker.bind_connector_metadata(self._meta(load=True))
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
                                        if token < unit
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
            return SimpleNamespace(
                block_ids=(b"a" * 16,),
                offsets=(4,),
                ptrs=(100,),
                sizes=(8 if layer_name else 16,),
            )

        layout.build_dump_batches = build
        worker = SimpleNamespace(layout=layout, use_layerwise=True)
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
            self.assertEqual(backend.lookup((bytes([1]) * 16,)), (False,))

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
            proxy = SimpleFileUCMProxy(directory, access)
            adapter = UCMProxyAdapter(proxy)
            adapter.register_tensors(registered)
            adapter.dump(
                (key, key),
                (0, 5),
                (0x1000, 0x2000),
                (5, 7),
            )

            self.assertEqual(adapter.lookup((key,)), (False,))
            adapter.commit((key,))
            self.assertEqual(adapter.lookup((key, b"m" * 16)), (True, False))
            self.assertEqual(
                (Path(directory) / f"{key.hex()}.ucm").read_bytes(), source
            )

            adapter.load(
                (key, key),
                (0, 5),
                (0x3000, 0x4000),
                (5, 7),
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
            adapter.dump((key,), (4,), (0x2000,), (4,))
            adapter.dump((other,), (0,), (0x2000,), (4,))
            self.assertEqual(adapter.lookup((key, other)), (False, False))
            adapter.dump((key,), (0,), (0x1000,), (4,))
            adapter.commit((key,))
            self.assertEqual(adapter.lookup((key, other)), (True, False))
            self.assertEqual(proxy._path(key).read_bytes(), b"abcdefgh")
            adapter.commit((key,))  # Idempotent retry.
            adapter.load((key,), (0,), (0x3000,), (8,))
            self.assertEqual(memory.read(0x3000, 8), b"abcdefgh")
            # A replacement remains hidden behind the previous complete block.
            adapter.dump((key,), (0,), (0x2000,), (4,))
            self.assertEqual(proxy._path(key).read_bytes(), b"abcdefgh")
            adapter.commit((key, other))
            self.assertEqual(proxy._path(key).read_bytes(), b"efgh")
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_wait_for_save_commits_after_all_dumps_and_not_on_dump_failure(self):
        connector = UCMConnector.__new__(UCMConnector)
        connector.use_layerwise = False
        connector.has_connector_metadata = lambda: True
        connector._get_connector_metadata = lambda: UCMConnectorMetadata()
        batches = (
            SimpleNamespace(keys=(b"a" * 16,)),
            SimpleNamespace(keys=(b"b" * 16,)),
        )
        connector.layout = SimpleNamespace(build_dump_transfers=lambda _: batches)
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

    def test_rejects_misaligned_arrays_and_record_overflow(self):
        key = b"x" * 16
        adapter = UCMProxyAdapter(FakeProxy(), {key: 32})
        with self.assertRaisesRegex(ValueError, "identical lengths"):
            adapter.load([key], [], [1], [1])
        with self.assertRaisesRegex(ValueError, "exceeds record"):
            adapter.dump([key], [24], [1], [16])

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

        adapter.load((key,), (0,), (0x1000,), (32,))
        adapter.dump((key,), (0,), (0x2000,), (32,))

        self.assertEqual(
            proxy.wait_calls,
            [("load-task", 1), ("dump-task", 1)],
        )

    def test_rejects_async_task_without_wait_contract(self):
        class IncompleteAsyncProxy(FakeProxy):
            def load(self, block_ids, offsets, ptrs, sizes):
                return object()

        with self.assertRaisesRegex(UCMProxyError, "does not provide wait"):
            UCMProxyAdapter(IncompleteAsyncProxy()).load(
                (b"x" * 16,), (0,), (0x1000,), (32,)
            )

    def test_worker_load_failure_reports_request_and_block_ids_once(self):
        class BrokenLoadProxy(FakeProxy):
            def load(self, block_ids, offsets, ptrs, sizes):
                raise OSError("device copy failed")

        with mock.patch(
            "ucm.integration.vllm.v2.ucm_connector.SimpleFileUCMProxy",
            return_value=BrokenLoadProxy(),
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
            "FA",
            (b"x" * 16,),
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
        class SelectiveLoadProxy(FakeProxy):
            def load(self, block_ids, offsets, ptrs, sizes):
                if b"x" * 16 in block_ids:
                    raise OSError("one request failed")
                super().load(block_ids, offsets, ptrs, sizes)

        proxy = SelectiveLoadProxy()
        with mock.patch(
            "ucm.integration.vllm.v2.ucm_connector.SimpleFileUCMProxy",
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
                "FA",
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
                    "failed": request_meta("failed", b"x" * 16, 5),
                    "loaded": request_meta("loaded", b"y" * 16, 6),
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
        self.parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
        )
        self.dispatcher = make_dispatcher(self.parsed)

    @staticmethod
    def scheduler_output(
        *,
        new=(),
        cached_ids=(),
        cached_blocks=(),
        resumed=(),
        scheduled=None,
        preempted=(),
        finished=(),
    ):
        return SimpleNamespace(
            scheduled_new_reqs=list(new),
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=list(cached_ids),
                new_block_ids=list(cached_blocks),
                resumed_req_ids=set(resumed),
            ),
            num_scheduled_tokens=scheduled or {},
            preempted_req_ids=set(preempted),
            finished_req_ids=set(finished),
        )

    def add_state(self, request_id="r"):
        request = FakeRequest(request_id, 512)
        keys = tuple(bytes([index]) * 16 for index in range(4))
        return seed_request_state(self.dispatcher, request, group_ucm_block_ids=(keys,))

    def test_new_replaces_and_cached_appends_block_tables(self):
        state = self.add_state()
        self.dispatcher.build_from_scheduler_output(
            self.scheduler_output(
                new=(SimpleNamespace(req_id="r", block_ids=([3, 4],)),),
                scheduled={"r": 128},
            )
        )
        self.assertEqual(state.group_vllm_block_ids, ([3, 4],))

        self.dispatcher.build_from_scheduler_output(
            self.scheduler_output(
                cached_ids=("r",),
                cached_blocks=((([5],)),),
                scheduled={"r": 128},
            )
        )
        self.assertEqual(state.group_vllm_block_ids, ([3, 4, 5],))

    def test_resumed_request_replaces_stale_blocks(self):
        state = self.add_state()
        state.group_vllm_block_ids = ([1, 2, 3],)

        self.dispatcher.build_from_scheduler_output(
            self.scheduler_output(
                cached_ids=("r",),
                cached_blocks=((([7, 6],)),),
                resumed=("r",),
                scheduled={"r": 128},
            )
        )

        self.assertEqual(state.group_vllm_block_ids, ([7, 6],))

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
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
            ucm_cache_block_size=512,
        )
        dispatcher = make_dispatcher(parsed)
        request = FakeRequest("chunked", 1024)
        keys = (b"a" * 16, b"b" * 16)
        state = seed_request_state(dispatcher, request, group_ucm_block_ids=(keys,))
        state.group_vllm_block_ids = ([1, 2, 3, 4, 5, 6, 7, 8],)

        first = dispatcher.build_metadata({"chunked": 128})
        second = dispatcher.build_metadata({"chunked": 384})
        third = dispatcher.build_metadata({"chunked": 512})

        self.assertEqual(first.requests["chunked"].dump_plans, ())
        second_plan = second.requests["chunked"].dump_plans[0]
        self.assertEqual(
            (second_plan.keys, second_plan.token_start, second_plan.token_end),
            ((keys[0],), 0, 512),
        )
        self.assertEqual(
            second_plan.windows[0].tolist(),
            [1, 2, 3, 4],
        )
        third_plan = third.requests["chunked"].dump_plans[0]
        self.assertEqual(
            (third_plan.keys, third_plan.token_start, third_plan.token_end),
            ((keys[1],), 512, 1024),
        )
        self.assertEqual(third_plan.windows[0].tolist(), [5, 6, 7, 8])

    def test_external_load_plan_is_consumed_after_first_scheduled_step(self):
        state = self.add_state()
        state.external_hit_tokens = 128
        state.token_processed = 128
        state.load_pending = True
        state.group_vllm_block_ids = ([3, 4, 5, 6],)

        first = self.dispatcher.build_metadata({"r": 128})
        second = self.dispatcher.build_metadata({"r": 128})

        self.assertEqual(len(first.requests["r"].load_plans), 1)
        self.assertEqual(first.requests["r"].load_plans[0].token_start, 0)
        self.assertEqual(first.requests["r"].load_plans[0].token_end, 128)
        self.assertEqual(second.requests["r"].load_plans, ())

    def test_boundary_chain_dumps_straddled_boundaries(self):
        # A step may advance past a cache boundary without ending on it
        # (eagle-style multi-token steps, odd chunked-prefill sizes): the
        # boundary record must still be dumped -- the newest boundary the
        # step completed -- and a step completing no new boundary dumps
        # nothing.
        parsed = parse_kv_cache_config(
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
        seed_request_state(
            dispatcher, request, group_ucm_block_ids=(fa_keys, state_keys)
        )
        for request_id, state in dispatcher.requests.items():
            state.group_vllm_block_ids = (list(range(16)), list(range(16)))

        first = dispatcher.build_metadata({"r": 1000})
        state_plan = next(
            plan
            for plan in first.requests["r"].dump_plans
            if plan.hash_group == "State"
        )
        # [0, 1000) completes boundaries up to 896: the newest is 896's.
        self.assertEqual(state_plan.keys, (state_keys[6],))

        # [1000, 1100) straddles boundary 1024 without ending on it.
        second = dispatcher.build_metadata({"r": 100})
        state_plan = next(
            plan
            for plan in second.requests["r"].dump_plans
            if plan.hash_group == "State"
        )
        self.assertEqual(state_plan.keys, (state_keys[7],))
        self.assertEqual(state_plan.token_end, 1024)
        # The snapshot block is the boundary's (1024th token's) block,
        # not the step end's (1100th token's) block.
        self.assertEqual(state_plan.windows[0].tolist(), [7])

        # [1100, 1124) completes no new boundary: nothing more to record.
        third = dispatcher.build_metadata({"r": 24})
        self.assertFalse(
            any(plan.hash_group == "State" for plan in third.requests["r"].dump_plans)
        )

    def test_dsv4_wa_dispatch_uses_only_final_boundary_and_real_tails(self):
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.2.attn"],
                    AscendMLAAttentionSpec(128, compress_ratio=4),
                ),
                group(
                    ["model.layers.3.attn"],
                    AscendMLAAttentionSpec(128, compress_ratio=128),
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
            scheduler_block_size=8,
        )
        dispatcher = make_dispatcher(parsed)
        request = FakeRequest("dsv4", 1024)
        fa_keys = (b"a" * 16, b"b" * 16)
        wa_keys = (b"c" * 16, b"d" * 16)
        state = seed_request_state(dispatcher, request, 1024, (fa_keys, wa_keys))
        state.group_vllm_block_ids = (
            [1, 2],
            [3],
            list(range(8)),
            list(range(128)),
            list(range(32)),
        )

        metadata = dispatcher.build_metadata({"dsv4": 1})
        fa_plan = next(
            plan
            for plan in metadata.requests["dsv4"].load_plans
            if plan.hash_group == "FA"
        )
        wa_plan = next(
            plan
            for plan in metadata.requests["dsv4"].load_plans
            if plan.hash_group == "WA"
        )

        self.assertEqual(len(fa_plan.windows), 2)
        self.assertEqual(wa_plan.keys, (wa_keys[1],))
        # Windows carry only block ids; the static shapes live in the spec
        # (swa: one whole 128-token block; C4 compressor: the [4, 8)
        # half-block sub-span) and the worker derives them from templates.
        self.assertEqual(
            tuple(blocks.tolist() for blocks in wa_plan.windows),
            ([7], [127]),
        )

    def test_wa_dump_skips_incomplete_tail_windows(self):
        # A tail window longer than the cache block clamps at the request
        # head on early boundaries; those partial windows are the
        # request's own prefix (restored with it), so nothing is dumped
        # until the boundary outgrows the tail.
        parsed = parse_kv_cache_config(
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
        seed_request_state(dispatcher, request, group_ucm_block_ids=(fa_keys, wa_keys))
        dispatcher.requests["r"].group_vllm_block_ids = (
            list(range(8)),
            list(range(8)),
        )

        # [0, 384): every completed boundary (128..384) is shorter than
        # the 512-token tail -- no WA record.
        early = dispatcher.build_metadata({"r": 384})
        self.assertTrue(
            any(plan.hash_group == "FA" for plan in early.requests["r"].dump_plans)
        )
        self.assertFalse(
            any(plan.hash_group == "WA" for plan in early.requests["r"].dump_plans)
        )

        # [384, 640): the newest boundary 640 outgrows the tail -- the
        # full window [128, 640) is one WA record at that boundary.
        later = dispatcher.build_metadata({"r": 256})
        wa_plan = next(
            plan for plan in later.requests["r"].dump_plans if plan.hash_group == "WA"
        )
        self.assertEqual(wa_plan.keys, (wa_keys[4],))
        self.assertEqual(wa_plan.windows[0].tolist(), [1, 2, 3, 4])

    def test_wa_window_not_dividing_block_keeps_partial_head(self):
        # ceil, not HMA's floor: a 96-token window over 64-token blocks
        # spans two blocks -- the head block keeps its trailing 32 tokens
        # at a 32-token in-block offset, the newest block stays whole,
        # and the two IO spans are memory-adjacent.
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(
                    ["model.layers.1.swa_cache"],
                    AscendSlidingWindowMLASpec(64, 1, 96),
                ),
            ),
            scheduler_block_size=128,
        )
        dispatcher = make_dispatcher(parsed)
        request = FakeRequest("r", 512)
        fa_keys = tuple(bytes([index]) * 16 for index in range(4))
        wa_keys = tuple(bytes([index + 32]) * 16 for index in range(4))
        seed_request_state(dispatcher, request, group_ucm_block_ids=(fa_keys, wa_keys))
        dispatcher.requests["r"].group_vllm_block_ids = (
            list(range(4)),
            list(range(8)),
        )

        metadata = dispatcher.build_metadata({"r": 384})
        wa_plan = next(
            plan
            for plan in metadata.requests["r"].dump_plans
            if plan.hash_group == "WA"
        )
        # Window [288, 384): the head half of block 4 plus all of block 5.
        self.assertEqual(wa_plan.keys, (wa_keys[2],))
        self.assertEqual(wa_plan.windows[0].tolist(), [4, 5])

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
        )
        batch = layout.build_dump_batches(metadata, "model.layers.1.swa_cache")
        self.assertEqual(batch.sizes.tolist(), [32768, 65536])
        self.assertEqual(batch.offsets.tolist(), [0, 32768])
        self.assertEqual(
            batch.ptrs.tolist(),
            [0x3000 + 4 * 65536 + 32768, 0x3000 + 5 * 65536],
        )

    def test_dsv4_fa_windows_encode_n_to_one_keys_flat(self):
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.2.attn"],
                    AscendMLAAttentionSpec(128, compress_ratio=4),
                ),
                group(
                    ["model.layers.3.attn"],
                    AscendMLAAttentionSpec(128, compress_ratio=128),
                ),
            ),
            scheduler_block_size=16,
        )
        dispatcher = make_dispatcher(parsed)
        request = FakeRequest("r", 2048)
        c4_keys = tuple(bytes([index]) * 16 for index in range(4))
        c128_keys = tuple(bytes([index + 16]) * 16 for index in range(4))
        seed_request_state(
            dispatcher, request, group_ucm_block_ids=(c4_keys, c128_keys)
        )
        dispatcher.requests["r"].group_vllm_block_ids = ([1, 2, 3, 4], [5, 6])

        metadata = dispatcher.build_metadata({"r": 2048})
        plan = metadata.requests["r"].dump_plans[0]

        # Group 0 (token block 512 == unit): one whole block per key.
        self.assertEqual(
            plan.windows[0].tolist(),
            [1, 2, 3, 4],
        )
        # Group 1 (token block 16384): four keys share one page -- the
        # same id four times, flat; each key's 512-token sub-span head is
        # derived on the worker from the plan's token range.
        self.assertEqual(
            plan.windows[1].tolist(),
            [5, 5, 5, 5],
        )

    def test_large_prefill_plan_stays_flat_and_compact(self):
        # 8192 keys in one step (unit 128 => a 1M-token prefill): the
        # flat encoding keeps the pickled metadata well under a megabyte.
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
        )
        dispatcher = make_dispatcher(parsed)
        keys = tuple(bytes([index % 256]) * 16 for index in range(8192))
        state = seed_request_state(
            dispatcher,
            FakeRequest("r", 8192 * 128),
            group_ucm_block_ids=(keys,),
        )
        state.group_vllm_block_ids = (list(range(8192)),)

        metadata = dispatcher.build_metadata({"r": 8192 * 128})
        plan = metadata.requests["r"].dump_plans[0]

        self.assertEqual(len(plan.keys), 8192)
        self.assertEqual(plan.windows[0].tolist(), list(range(8192)))

        import pickle

        self.assertLess(len(pickle.dumps(metadata)), 1024 * 1024)


class RaggedLayoutTest(unittest.TestCase):
    def test_cpu_attention_derives_storage_axis_from_tokens_per_state(self):
        # vLLM 0.29 dropped spec.storage_block_size; the storage axis is
        # derived as block_size // tokens_per_state (DSV4 C4A: 256/4 = 64).
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.0.attn"],
                    FullAttentionSpec(256, tokens_per_state=4),
                )
            ),
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
        )

        component = layout.group_layouts[0]
        self.assertEqual(parsed.groups[0].token_block_size, 256)
        # One stored state per 4 tokens: the block spans 64 states.
        self.assertEqual(
            (int(component.states_per_block[0]), int(component.state_strides[0])),
            (64, 8),
        )

    def test_combined_mamba_raw_page_uses_one_io_region(self):
        parsed = parse_kv_cache_config(
            config(
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
            ),
            scheduler_block_size=4,
            device_type="cpu",
        )
        layout = UCMKVCacheLayout(
            parsed,
            {
                "model.layers.0.attn": FakeTensor(0x800, (8, 4, 1), (4, 1, 1)),
                "model.layers.1.mixer": FakeTensor(
                    0x1000,
                    (8, 1, 1, 64),
                    (64, 64, 64, 1),
                ),
            },
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
                    layout.record_layouts[group_layout.group_id].group_block_offsets[0]
                ),
            ),
            (0x1000, 64, 16, 0),
        )

        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"m" * 16
        plan = UCMGroupDispatchPlan(
            "State",
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
        batch = layout.build_load_batches(metadata)
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
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.0.attn", "model.layers.1.attn"],
                    {
                        "model.layers.0.attn": FullAttentionSpec(4),
                        "model.layers.1.attn": FullAttentionSpec(4),
                    },
                )
            ),
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
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(4))),
            scheduler_block_size=4,
        )

        with self.assertRaisesRegex(ValueError, "2-D, 3-D, or 4-D"):
            UCMKVCacheLayout(
                parsed,
                {
                    "model.layers.0.attn": FakeTensor(
                        0x1000,
                        (2, 8, 4, 2, 3),
                        (192, 24, 6, 3, 1),
                    )
                },
            )

    def test_partial_range_keeps_block_wise_entries(self):
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 192),
                )
            ),
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
        )

        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"p" * 16
        # A WA tail shorter than two blocks: the [64, 128) half-block head
        # of block 2 plus the whole block 3.  Only the ids travel; the
        # static tail pattern places the sub-spans.
        plan = UCMGroupDispatchPlan(
            "WA",
            (key,),
            0,
            192,
            (np.array([2, 3], dtype=np.uint64),),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = layout.build_load_batches(metadata)

        # The half-block head and the whole trailing block stay separate
        # entries even though they are contiguous in memory: record
        # positions are a function of the logical enumeration, never of
        # physical adjacency.  Segments pack back to back -- the proxy
        # requires a gapless record.
        self.assertEqual(batch.block_ids, (key, key))
        self.assertEqual(batch.ptrs.tolist(), [0x1000 + 2 * 128 + 64, 0x1000 + 3 * 128])
        self.assertEqual(batch.sizes.tolist(), [64, 128])
        self.assertEqual(batch.offsets.tolist(), [0, 64])

    def test_partial_records_roundtrip_through_real_file_proxy(self):
        # Exercise the real Proxy's complete-record contract. A memory stub
        # that silently zero-fills holes can hide incorrect file offsets.
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 192),
                )
            ),
            scheduler_block_size=128,
        )
        layout = UCMKVCacheLayout(
            parsed,
            {"model.layers.0.swa_cache": FakeTensor(0x1000, (8, 128, 1), (128, 1, 1))},
        )
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"p" * 16

        def batch(phase, block_ids):
            plan = UCMGroupDispatchPlan(
                "WA",
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
            return getattr(layout, "build_" + phase + "_batches")(metadata)

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
            proxy = SimpleFileUCMProxy(directory, MemoryByteAccess(memory))
            proxy.dump(dumped.block_ids, dumped.offsets, dumped.ptrs, dumped.sizes)
            proxy.commit(dumped.block_ids)
            self.assertEqual(
                (Path(directory) / (key.hex() + ".ucm")).read_bytes(), expected
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
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(8))),
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

        key_a, key_b = b"d" * 16, b"e" * 16
        adapter = UCMProxyAdapter(InMemoryByteProxy(memory))
        dump_plan = UCMGroupDispatchPlan(
            "FA",
            (key_a, key_b),
            0,
            16,
            (np.array([3, 4], dtype=np.uint64),),
        )
        dump_meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", dump_plans=(dump_plan,))}
        )
        dump_batch = layout.build_dump_batches(dump_meta)
        # Blocks 3 and 4 are enumerated separately even though adjacent.
        self.assertEqual(dump_batch.sizes.tolist(), [48, 48])

        # The load lands on scattered blocks; each key's record keeps its
        # own byte position.
        load_plan = UCMGroupDispatchPlan(
            "FA",
            (key_a, key_b),
            0,
            16,
            (np.array([6, 2], dtype=np.uint64),),
        )
        load_meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(load_plan,))}
        )
        load_batch = layout.build_load_batches(load_meta)
        self.assertEqual(load_batch.sizes.tolist(), [48, 48])
        self.assertEqual(load_batch.offsets.tolist(), [0, 0])
        self.assertEqual(load_batch.ptrs.tolist(), [0x1000 + 6 * 48, 0x1000 + 2 * 48])

        adapter.dump(
            dump_batch.block_ids,
            dump_batch.offsets,
            dump_batch.ptrs,
            dump_batch.sizes,
        )
        adapter.load(
            load_batch.block_ids,
            load_batch.offsets,
            load_batch.ptrs,
            load_batch.sizes,
        )
        self.assertEqual(memory.read(0x1000 + 6 * 48, 48), source_first)
        self.assertEqual(memory.read(0x1000 + 2 * 48, 48), source_second)

    def test_kimi_mla_six_kernel_rows_coalesce_per_component(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.3.self_attn.attn"], FullAttentionSpec(768)),
                num_blocks=2,
            ),
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
        layout = UCMKVCacheLayout(parsed, caches)
        key = b"k" * 16
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            "FA",
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

        batch = layout.build_load_batches(metadata)

        self.assertEqual(batch.sizes.tolist(), [786432, 98304])
        self.assertEqual(batch.offsets.tolist(), [0, 786432])
        self.assertEqual(batch.ptrs.tolist(), [0x1000 + 786432, 0x200000 + 98304])

    def test_kimi_state_components_copy_payload_not_shared_page_stride(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.3.attn"], FullAttentionSpec(768)),
                group(
                    ["model.layers.0.self_attn"],
                    MambaSpec(
                        768,
                        shapes=((3, 4608), (12, 128, 128)),
                    ),
                ),
                num_blocks=2,
            ),
            scheduler_block_size=768,
        )
        num_blocks = 2
        conv_bytes = 27648
        ssm_bytes = 786432
        caches = {
            "model.layers.3.attn": FakeTensor(
                0x1000,
                (12, 128, 1, 576),
                (73728, 576, 576, 1),
                element_size=2,
            ),
            "model.layers.0.self_attn": (
                FakeTensor(
                    0x5000,
                    (2, 3, 4608),
                    (conv_bytes // 2, 4608, 1),
                    element_size=2,
                ),
                FakeTensor(
                    0x5000 + num_blocks * conv_bytes,
                    (2, 12, 128, 128),
                    (ssm_bytes // 4, 16384, 128, 1),
                    element_size=4,
                ),
            ),
        }
        layout = UCMKVCacheLayout(parsed, caches)
        key = b"s" * 16
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            "State",
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

        batch = layout.build_load_batches(metadata)

        self.assertEqual(
            batch.ptrs.tolist(),
            [
                0x5000 + conv_bytes,
                0x5000 + num_blocks * conv_bytes + ssm_bytes,
            ],
        )
        self.assertEqual(batch.sizes.tolist(), [conv_bytes, ssm_bytes])
        self.assertEqual(batch.offsets.tolist(), [0, conv_bytes])

    def test_component_major_layout_spans_multiple_physical_blocks(self):
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.0.attn", "model.layers.1.attn"],
                    {
                        "model.layers.0.attn": FullAttentionSpec(128),
                        "model.layers.1.attn": FullAttentionSpec(128),
                    },
                )
            ),
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
        layout = UCMKVCacheLayout(parsed, caches)
        proxy = FakeProxy()
        coordinator = UCMDispatcher(
            parsed,
            UCMProxyAdapter(proxy),
            RequestHasher(vllm_config(), 0),
            b"seed",
            recompute_tokens=0,
        )
        request = FakeRequest("r", 256)
        dispatcher = coordinator
        result = dispatcher.lookup(request, 0)
        state = dispatcher.requests["r"]
        state.group_vllm_block_ids = ([2, 5],)
        state.external_hit_tokens = 256
        state.load_pending = True
        metadata = dispatcher.build_metadata({"r": 1})

        batch = layout.build_load_batches(metadata)

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
        layer_batch = layout.build_load_batches(metadata, "model.layers.1.attn")
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
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.0.attn", "model.layers.1.attn"],
                    {
                        "model.layers.0.attn": FullAttentionSpec(256, 4),
                        "model.layers.1.attn": FullAttentionSpec(256, 4),
                    },
                ),
                num_blocks=4,
                tensors=declarations,
            ),
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
        layout = UCMKVCacheLayout(parsed, caches)
        group_layout = layout.group_layouts[0]
        span = group_layout.block_first
        self.assertIsNotNone(span)
        self.assertEqual(
            (span.base_ptr, span.block_stride, span.block_size_bytes),
            (0x1000, block_stride, 2 * layer_stride),
        )
        # The record is slot-sized: both layers' page slots, paddings and all.
        self.assertEqual(
            layout.record_layouts[group_layout.group_id].whole_block_bytes,
            2 * layer_stride,
        )

        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"b" * 16
        plan = UCMGroupDispatchPlan(
            "FA",
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
        batch = layout.build_load_batches(metadata)
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
        layer_batch = layout.build_load_batches(metadata, "model.layers.1.attn")
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

        # The public extract_segments API mirrors both shapes: the whole
        # unfiltered window is the group span, layered queries the view
        # grid (block 2, layer 1's page).
        span_ptrs, span_sizes = group_layout.block_first_segments((2,))
        self.assertEqual(
            (span_ptrs.tolist(), span_sizes.tolist()),
            ([0x1000 + 2 * block_stride], [2 * page]),
        )
        grid_ptrs, grid_sizes = group_layout.extract_segments([2], [0], [256])
        self.assertEqual(grid_ptrs[:, 1].tolist(), [0x1000 + page + 2 * block_stride])
        self.assertEqual(grid_sizes[:, 1].tolist(), [page])

        # Sub-block token windows never use the descriptor span: they walk
        # the layer's own view and stay payload-exact (128 tokens at the
        # 4:1 compression = 32 states x 64B = 2048B).
        window_ptrs, window_sizes = group_layout.extract_segments([2], [0], [128])
        self.assertEqual(window_ptrs[:, 0].tolist(), [0x1000 + 2 * block_stride])
        self.assertEqual(window_sizes[:, 0].tolist(), [2048])

    def test_tiled_group_spans_chain_two_descriptors(self):
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
        parsed = parse_kv_cache_config(
            config(
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
            ),
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
        layout = UCMKVCacheLayout(parsed, caches)
        group_layout = layout.group_layouts[0]

        # One group-level span: both tiles chained, slot-sized.
        span = group_layout.block_first
        self.assertIsNotNone(span)
        self.assertEqual(
            (span.base_ptr, span.block_stride, span.block_size_bytes),
            (0x1000, slot, slot),
        )
        self.assertEqual(
            layout.record_layouts[group_layout.group_id].whole_block_bytes, slot
        )

        # Layered anchors follow the tile chain, not the layer order.
        offsets = dict(
            zip(
                group_layout.layer_names,
                layout.record_layouts[
                    group_layout.group_id
                ].group_block_offsets.tolist(),
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

        key = b"c" * 16
        plan = UCMGroupDispatchPlan(
            "FA",
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
        batch = layout.build_load_batches(metadata)
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
        layer_batch = layout.build_load_batches(metadata, "model.layers.1.attn")
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
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.0.attn", "model.layers.1.attn"],
                    {
                        "model.layers.0.attn": FullAttentionSpec(4),
                        "model.layers.1.attn": FullAttentionSpec(4),
                    },
                )
            ),
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
            layout.record_layouts[group_layout.group_id].whole_block_bytes, 24
        )
        self.assertEqual(
            group_layout.layer_names,
            ("model.layers.0.attn", "model.layers.1.attn"),
        )
        self.assertEqual(
            layout.record_layouts[group_layout.group_id].group_block_offsets.tolist(),
            [0, 12],
        )

        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"r" * 16
        plan = UCMGroupDispatchPlan(
            "FA",
            (key,),
            0,
            8,
            (np.array([2, 5], dtype=np.uint64),),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = layout.build_load_batches(metadata)
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

        # The extract_segments grid answers the same spans without
        # records: rows are blocks, columns views.
        whole = group_layout.token_block_size
        grid_ptrs, grid_sizes = group_layout.extract_segments(
            [2, 5], [0, 0], [whole, whole]
        )
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
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.0.attn", "model.layers.1.attn"],
                    {
                        "model.layers.0.attn": FullAttentionSpec(128),
                        "model.layers.1.attn": FullAttentionSpec(128),
                    },
                )
            ),
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
        layout = UCMKVCacheLayout(parsed, caches)
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        key = b"t" * 16
        plan = UCMGroupDispatchPlan(
            "FA",
            (key,),
            0,
            256,
            (np.array([2, 5], dtype=np.uint64),),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        full = layout.build_load_batches(metadata)
        union: dict[tuple[int, int, int], None] = {}
        for layer_name in ("model.layers.0.attn", "model.layers.1.attn"):
            layer_batch = layout.build_load_batches(metadata, layer_name)
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
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.0.indexer"], FullAttentionSpec(128)), num_blocks=4
            ),
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
        )
        key = b"g" * 16
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            "FA",
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

        batch = layout.build_load_batches(metadata)

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
        parsed = parse_kv_cache_config(
            config(
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
            ),
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
        layout = UCMKVCacheLayout(parsed, caches)
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
                "FA",
                (b"record",),
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
            source = layout.build_dump_batches(source_meta)
            target = layout.build_load_batches(target_meta)
            source_layer = layout.build_dump_batches(source_meta, layer_id=layer_id)
            target_layer = layout.build_load_batches(target_meta, layer_id=layer_id)
            self.assertEqual(source_layer.offsets.tolist(), list(offsets))
            self.assertEqual(target_layer.offsets.tolist(), list(offsets))
            exact = layout.build_load_batches(target_meta, "model.layers.2.attn")
            self.assertEqual(exact.offsets.tolist(), [offsets[0]])
            window = layout.group_layouts[0].extract_segments(
                [3], [0], [4], layer_ids=[2]
            )
            self.assertEqual(
                tuple(
                    zip(window[0].reshape(-1).tolist(), window[1].reshape(-1).tolist())
                ),
                tuple(zip(target_layer.ptrs[:2], target_layer.sizes[:2])),
            )
            with self.assertRaisesRegex(ValueError, "either"):
                layout.build_load_batches(
                    target_meta, "model.layers.2.attn", layer_id=2
                )
            with self.assertRaises(KeyError):
                layout.build_load_batches(target_meta, layer_id=99)

            memory = ByteMemory()
            for ptr, size in zip(target.ptrs, target.sizes):
                memory.write(ptr, b"\xef" * size)
            for offset, ptr, size in zip(source.offsets, source.ptrs, source.sizes):
                memory.write(ptr, bytes((int(offset) + i) % 251 for i in range(size)))
            proxy = SimpleFileUCMProxy(Path(path), byte_access=MemoryByteAccess(memory))
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
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.2.attn"],
                    AscendMLAAttentionSpec(128, compress_ratio=4),
                ),
                group(
                    ["model.layers.3.attn"],
                    AscendMLAAttentionSpec(128, compress_ratio=128),
                ),
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 128),
                ),
                num_blocks=4,
            ),
            scheduler_block_size=16,
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
        layout = UCMKVCacheLayout(parsed, caches)
        key = b"k" * 16
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            "FA",
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

        batch = layout.build_load_batches(metadata)

        # Group 0's whole block, then group 1's sub-span: 512-token page
        # window selected by token start inside the large page.
        self.assertEqual(batch.sizes[-1], 4096)
        self.assertEqual(batch.ptrs[-1], 0x5000 + 2 * 131072 + 4 * 1024)

    def test_hybrid_state_uses_one_complete_checkpoint_block(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            scheduler_block_size=128,
        )
        caches = {
            "model.layers.0.attn": FakeTensor(0x1000, (8, 128, 4), (512, 4, 1)),
            "model.layers.1.mixer": FakeTensor(0x5000, (8, 64, 4), (256, 4, 1)),
        }
        layout = UCMKVCacheLayout(parsed, caches)
        dispatcher = make_dispatcher(parsed)
        request = FakeRequest("r", 512)
        state = seed_request_state(
            dispatcher,
            request,
            512,
            ((b"a" * 16,) * 4, (b"s" * 16,) * 4),
        )
        state.group_vllm_block_ids = ([1, 2, 3, 4], [0, 1, 2, 6])

        metadata = dispatcher.build_metadata({"r": 1})
        batch = layout.build_load_batches(metadata)

        state_segments = [
            (ptr, size)
            for key, ptr, size in zip(batch.block_ids, batch.ptrs, batch.sizes)
            if key == b"s" * 16
        ]
        self.assertEqual(state_segments, [(0x5000 + 6 * 256, 256)])

    def test_synchronous_proxy_dump_load_to_different_blocks_is_byte_exact(self):
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(4))),
            scheduler_block_size=4,
        )
        caches = {
            "model.layers.0.attn": (
                FakeTensor(0x1000, (8, 4, 2), (8, 2, 1)),
                FakeTensor(0x2000, (8, 4, 3), (12, 3, 1)),
            )
        }
        layout = UCMKVCacheLayout(parsed, caches)
        key = b"z" * 16
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        dump_plan = UCMGroupDispatchPlan(
            "FA",
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
            "FA",
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
        proxy = InMemoryByteProxy(memory)
        adapter = UCMProxyAdapter(proxy)

        dump_batch = layout.build_dump_batches(dump_meta)
        adapter.dump(
            dump_batch.block_ids,
            dump_batch.offsets,
            dump_batch.ptrs,
            dump_batch.sizes,
        )
        self.assertEqual(adapter.lookup([key]), (True,))
        load_batch = layout.build_load_batches(load_meta)
        adapter.load(
            load_batch.block_ids,
            load_batch.offsets,
            load_batch.ptrs,
            load_batch.sizes,
        )

        self.assertEqual(memory.read(0x1000 + 6 * 8, 8), source_k)
        self.assertEqual(memory.read(0x2000 + 6 * 12, 12), source_v)

        class DeferredByteProxy(InMemoryByteProxy):
            def dump(self, block_ids, offsets, ptrs, sizes):
                return ("dump", (block_ids, offsets, ptrs, sizes))

            def load(self, block_ids, offsets, ptrs, sizes):
                return ("load", (block_ids, offsets, ptrs, sizes))

            def wait(self, task):
                operation, arguments = task
                if operation == "dump":
                    InMemoryByteProxy.dump(self, *arguments)
                else:
                    InMemoryByteProxy.load(self, *arguments)

        deferred_proxy = DeferredByteProxy(memory)
        deferred_adapter = UCMProxyAdapter(deferred_proxy)
        deferred_adapter.dump(
            dump_batch.block_ids,
            dump_batch.offsets,
            dump_batch.ptrs,
            dump_batch.sizes,
        )
        deferred_load_plan = UCMGroupDispatchPlan(
            "FA",
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
        deferred_load_batch = layout.build_load_batches(deferred_load_meta)
        deferred_adapter.load(
            deferred_load_batch.block_ids,
            deferred_load_batch.offsets,
            deferred_load_batch.ptrs,
            deferred_load_batch.sizes,
        )

        self.assertEqual(memory.read(0x1000 + 7 * 8, 8), source_k)
        self.assertEqual(memory.read(0x2000 + 7 * 12, 12), source_v)


class DeclaredLayoutModelTest(unittest.TestCase):
    """The declared mode must mirror vLLM's own packed-tensor placements."""

    def _fixture(self, tensors=None):
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
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.0.attn", "model.layers.1.attn"],
                    {
                        "model.layers.0.attn": FullAttentionSpec(4),
                        "model.layers.1.attn": FullAttentionSpec(4),
                    },
                ),
                tensors=declarations if tensors is None else tensors,
            ),
            scheduler_block_size=4,
            device_type="cpu",
        )
        caches = {
            "model.layers.0.attn": FakeTensor(0x1000, (8, 4, 3), (12, 3, 1)),
            "model.layers.1.attn": FakeTensor(0x1060, (8, 4, 3), (12, 3, 1)),
        }
        return parsed, caches

    def test_declarations_only_feed_the_block_first_special_case(self):
        # Layer-contiguous placements (layer_stride >= block_stride is
        # false here: 96 > 12 means layer-outermost) do not tile block
        # slots, so declarations change nothing: with or without them the
        # entries and the records are identical.
        declared = UCMKVCacheLayout(*self._fixture())
        undeclared = UCMKVCacheLayout(*self._fixture(tensors=()))

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
                declared.record_layouts[group_id].whole_block_bytes,
                undeclared.record_layouts[group_id].whole_block_bytes,
            )
            np.testing.assert_array_equal(
                declared.record_layouts[group_id].group_block_offsets,
                undeclared.record_layouts[group_id].group_block_offsets,
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
            UCMKVCacheLayout(*self._fixture(tensors=bad))


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
        parsed = parse_kv_cache_config(
            config(group(names, FullAttentionSpec(8)), num_blocks=4),
            scheduler_block_size=8,
        )
        self.layout = UCMKVCacheLayout(
            parsed,
            {
                names[0]: FakeTensor(1000, (4, 8, 2), (16, 2, 1)),
                names[1]: FakeTensor(2000, (4, 8, 3), (24, 3, 1)),
            },
        ).group_layouts[0]

    def test_fixed_partial_window_repeats_across_physical_blocks(self):
        access = self.layout.compile_access(token_offsets=[4, 0], token_counts=[4, 8])
        ptrs, sizes = access.resolve([1, 2, 0, 3])
        np.testing.assert_array_equal(
            ptrs, [[1024, 2036], [1032, 2048], [1008, 2012], [1048, 2072]]
        )
        np.testing.assert_array_equal(sizes, [[8, 12], [16, 24]] * 2)

    def test_dynamic_subblocks_and_layer_selection(self):
        access = self.layout.compile_access(token_counts=4, layer_ids=[1])
        ptrs, sizes = access.resolve([2, 2], token_offsets=np.array([0, 4]))
        np.testing.assert_array_equal(ptrs, [[2048], [2060]])
        np.testing.assert_array_equal(sizes, [[12], [12]])
        # Reuse the compiled template without mutating its fixed offsets.
        ptrs, _ = access.resolve([0], token_offsets=np.array([2]))
        np.testing.assert_array_equal(ptrs, [[2006]])

    def test_resolved_arrays_do_not_alias_templates_or_other_calls(self):
        access = self.layout.compile_access(token_offsets=[4, 0], token_counts=[4, 8])
        first = access.resolve([1, 2, 0, 3])
        second = access.resolve([1, 2, 0, 3])
        expected = tuple(array.copy() for array in second)
        for array in first:
            array[:] = 0
        for actual, wanted in zip(second, expected):
            np.testing.assert_array_equal(actual, wanted)
        for actual, wanted in zip(access.resolve([1, 2, 0, 3]), expected):
            np.testing.assert_array_equal(actual, wanted)

    def test_dynamic_ranges_with_empty_blocks_or_selection(self):
        access = self.layout.compile_access(token_counts=4)
        ptrs, sizes = access.resolve([], token_offsets=np.array([], dtype=np.int64))
        self.assertEqual(ptrs.shape, (0, 2))
        self.assertEqual(sizes.shape, (0, 2))
        ptrs, sizes = access.resolve(
            [2], token_offsets=np.array([4]), segment_mask=np.array([False, False])
        )
        self.assertEqual(ptrs.shape, (1, 0))
        self.assertEqual(sizes.shape, (1, 0))

    def test_invalid_ranges_and_incomplete_windows_are_rejected(self):
        with self.assertRaises(ValueError):
            self.layout.compile_access(token_offsets=6, token_counts=4)
        access = self.layout.compile_access(token_counts=4)
        with self.assertRaises(ValueError):
            access.resolve([0], token_offsets=np.array([6]))
        with self.assertRaises(ValueError):
            access.resolve([4])
        window = self.layout.compile_access(token_counts=[4, 8])
        with self.assertRaises(ValueError):
            window.resolve([0])


class LayerViewSegmentsTest(unittest.TestCase):
    def test_separate_backings_do_not_merge_from_declarations_alone(self):
        names = ["model.layers.0.attn", "model.layers.1.attn"]
        parsed = parse_kv_cache_config(
            config(
                group(names, FullAttentionSpec(8)),
                num_blocks=3,
                tensors=tuple(
                    SimpleNamespace(
                        layers=(name,), offset=i * 16, layer_stride=16, block_stride=32
                    )
                    for i, name in enumerate(names)
                ),
            ),
            scheduler_block_size=8,
            device_type="cpu",
        )
        layout = UCMKVCacheLayout(
            parsed,
            {
                names[0]: FakeTensor(1000, (3, 8, 1), (32, 1, 1)),
                names[1]: FakeTensor(5000, (3, 8, 1), (32, 1, 1)),
            },
        )
        self.assertIsNone(layout.group_layouts[0].block_first)
        record = layout.record_layouts[0]
        offsets, ptrs, sizes, entries = record.resolve(np.array([2]), 1)
        self.assertEqual(
            (offsets.tolist(), ptrs.tolist(), sizes.tolist(), entries),
            ([0, 8], [1064, 5064], [8, 8], 2),
        )

    def test_head_separated_wa_partial_head_then_whole_block(self):
        from ucm.integration.vllm.v2.record_layout import GroupRecordLayout

        name = "model.layers.1.swa_cache"
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(8)),
                group([name], AscendSlidingWindowMLASpec(8, 1, 12)),
                num_blocks=3,
            ),
            scheduler_block_size=8,
            device_type="cpu",
        )
        layout = UCMKVCacheLayout(
            parsed,
            {
                "model.layers.0.attn": FakeTensor(100, (3, 8, 2), (16, 2, 1)),
                name: FakeTensor(1000, (3, 2, 8, 2), (32, 16, 2, 1)),
            },
        )
        record = GroupRecordLayout.build(parsed.groups[1], layout.group_layouts[1], 8)
        offsets, ptrs, sizes, entries = record.resolve(np.array([1, 2]), 1)
        # Logical tokens 4..7 in block 1, then 0..7 in block 2, per head.
        self.assertEqual(ptrs.tolist(), [1040, 1056, 1064, 1080])
        self.assertEqual(sizes.tolist(), [8, 8, 16, 16])
        self.assertEqual(offsets.tolist(), [0, 8, 16, 32])
        self.assertEqual(entries, 4)

    def test_equal_axis_sizes_use_backend_semantics(self):
        name = "model.layers.0.attn"
        for device, strides, expected in (
            ("cpu", (16, 4, 1, 1), [[1013, 1017, 1021, 1025]]),
            ("npu", (16, 4, 1, 1), [[1016]]),
        ):
            parsed = parse_kv_cache_config(
                config(group([name], FullAttentionSpec(4)), num_blocks=3),
                scheduler_block_size=4,
                device_type=device,
            )
            gl = UCMKVCacheLayout(
                parsed,
                {
                    name: FakeTensor(996, (3, 4, 4, 1), strides),
                },
            ).group_layouts[0]
            ptrs, sizes = gl.compile_access(token_offsets=1, token_counts=2).resolve(
                [1]
            )
            self.assertEqual(ptrs.tolist(), expected)
            self.assertEqual(int(sizes.sum()), 8)

    def test_multirow_head_separated_range_is_explicitly_rejected(self):
        name = "model.layers.0.attn"
        parsed = parse_kv_cache_config(
            config(group([name], FullAttentionSpec(8)), num_blocks=3),
            scheduler_block_size=8,
            device_type="cpu",
        )
        with self.assertRaisesRegex(ValueError, "ragged range mapping"):
            UCMKVCacheLayout(
                parsed,
                {
                    name: FakeTensor(1000, (6, 2, 4, 2), (16, 8, 2, 1)),
                },
            )

    def _layout(self, order, *, unit=4, declared=False):
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
        parsed = parse_kv_cache_config(
            config(group(names, FullAttentionSpec(8)), num_blocks=3, tensors=tensors),
            scheduler_block_size=4,
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
        )
        return layout, strides

    def test_layout_permutations_select_exact_logical_elements(self):
        for order in ("LBNHC", "LBHNC", "BLNHC", "BLHNC", "LHBNC", "BHLNC"):
            with self.subTest(order=order):
                layout, strides = self._layout(order)
                group_layout = layout.group_layouts[0]
                access = group_layout.compile_access(token_counts=4, layer_ids=[1])
                ptrs, sizes = access.resolve([2, 2], token_offsets=np.array([0, 4]))
                for row, start in enumerate((0, 4)):
                    actual = [
                        address
                        for ptr, size in zip(ptrs[row], sizes[row])
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
                        for n in range(start, start + 4)
                        for c in range(2)
                    )
                    self.assertEqual(actual, expected)
                lv = group_layout.layer_views["model.layers.1.attn"]
                self.assertEqual(len(lv.components), 1)
                self.assertEqual(len(lv.segments), 1 if order.endswith("NHC") else 2)

    def test_ucm_offsets_and_layerwise_selection_share_one_mapping(self):
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        for order in ("LBHNC", "LHBNC", "BHLNC"):
            with self.subTest(order=order):
                layout, strides = self._layout(order)
                plan = UCMGroupDispatchPlan(
                    "FA",
                    (b"a" * 16, b"b" * 16),
                    0,
                    8,
                    (np.array([2, 2], dtype=np.uint64),),
                )
                meta = UCMConnectorMetadata(
                    requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
                )
                batch = layout.build_load_batches(meta)
                self.assertEqual(batch.offsets.tolist(), [0, 8, 16, 24] * 2)
                self.assertEqual(batch.sizes.tolist(), [8] * 8)
                selected = layout.build_load_batches(meta, layer_id=1)
                self.assertEqual(selected.offsets.tolist(), [16, 24] * 2)
                self.assertEqual(
                    selected.ptrs.tolist(), batch.ptrs[[2, 3, 6, 7]].tolist()
                )
                # Independent memory oracle checks byte content, not roundtrip alone.
                memory = ByteMemory()
                for address in range(1000, 1192):
                    memory.write(address, bytes([address % 251]))
                dump = layout.build_dump_batches(
                    UCMConnectorMetadata(
                        requests={"r": RequestDispatchMeta("r", dump_plans=(plan,))}
                    )
                )
                with tempfile.TemporaryDirectory() as tmp:
                    proxy = SimpleFileUCMProxy(tmp, MemoryByteAccess(memory))
                    proxy.dump(dump.block_ids, dump.offsets, dump.ptrs, dump.sizes)
                    proxy.commit(dump.block_ids)
                    for k, start in zip(plan.keys, (0, 4)):
                        expected = bytes(
                            (
                                1000
                                + l * strides["L"]
                                + 2 * strides["B"]
                                + h * strides["H"]
                                + n * strides["N"]
                                + c
                            )
                            % 251
                            for l in range(2)
                            for h in range(2)
                            for n in range(start, start + 4)
                            for c in range(2)
                        )
                        self.assertEqual(
                            (Path(tmp) / (k.hex() + ".ucm")).read_bytes(), expected
                        )

    def test_compact_declared_heads_keep_bulk_fast_path(self):
        layout, _ = self._layout("BLHNC", unit=8, declared=True)
        group_layout = layout.group_layouts[0]
        self.assertIsNotNone(group_layout.block_first)
        access = group_layout.compile_access(layer_ids=[1])
        ptrs, sizes = access.resolve([1])
        self.assertEqual(ptrs.tolist(), [[1096, 1112]])
        self.assertEqual(sizes.tolist(), [[16, 16]])
        self.assertEqual(
            layout.record_layouts[group_layout.group_id].group_block_offsets.tolist(),
            [0, 16, 32, 48],
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
                parsed = parse_kv_cache_config(
                    config(group([name], FullAttentionSpec(6144)), num_blocks=blocks),
                    scheduler_block_size=6144,
                    device_type=device,
                )
                gl = UCMKVCacheLayout(parsed, {name: tensors}).group_layouts[0]
                ptrs, sizes = gl.compile_access(token_counts=512).resolve(
                    [2], token_offsets=np.array([512])
                )
                self.assertEqual(ptrs.tolist(), [expected_ptrs])
                self.assertEqual(int(sizes.sum()), 512 * 1152)

    def test_documented_fa_subblocks_preserve_layer_offsets(self):
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        names = ["model.layers.0.attn", "model.layers.1.attn"]
        parsed = parse_kv_cache_config(
            config(group(names, FullAttentionSpec(16384)), num_blocks=3),
            scheduler_block_size=512,
            ucm_cache_block_size=512,
            device_type="npu",
        )
        layout = UCMKVCacheLayout(
            parsed,
            {
                name: FakeTensor(base, (3, 16384, 2, 2), (65536, 4, 2, 1))
                for name, base in zip(names, (1000000, 2000000))
            },
        )
        plan = UCMGroupDispatchPlan(
            "FA", (b"a" * 16, b"b" * 16), 0, 1024, (np.array([2, 2], dtype=np.uint64),)
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = layout.build_load_batches(metadata)
        self.assertEqual(batch.ptrs.tolist(), [1131072, 2131072, 1133120, 2133120])
        self.assertEqual(batch.sizes.tolist(), [2048] * 4)
        self.assertEqual(batch.offsets.tolist(), [0, 2048, 0, 2048])
        selected = layout.build_load_batches(metadata, layer_id=1)
        self.assertEqual(selected.ptrs.tolist(), [2131072, 2133120])
        self.assertEqual(selected.offsets.tolist(), [2048, 2048])
        # A single-group batch now avoids concatenate; its buffers must still
        # be independent from another dispatch and from compiled templates.
        expected = [array.copy() for array in (batch.offsets, batch.ptrs, batch.sizes)]
        retained = layout.build_load_batches(metadata)
        for array in (batch.offsets, batch.ptrs, batch.sizes):
            array[:] = 0
        later = layout.build_load_batches(metadata)
        for other in (retained, later):
            for actual, wanted in zip(
                (other.offsets, other.ptrs, other.sizes), expected
            ):
                np.testing.assert_array_equal(actual, wanted)

    def test_fa_16384_tokens_to_32_ucm_keys(self):
        name = "model.layers.0.attn"
        parsed = parse_kv_cache_config(
            config(group([name], FullAttentionSpec(16384)), num_blocks=3),
            scheduler_block_size=512,
            ucm_cache_block_size=512,
            device_type="npu",
        )
        layout = UCMKVCacheLayout(
            parsed, {name: FakeTensor(1000, (3, 16384, 2, 2), (65536, 4, 2, 1))}
        )
        access = layout.group_layouts[0].compile_access(token_counts=512)
        ptrs, sizes = access.resolve([2] * 32, token_offsets=np.arange(32) * 512)
        np.testing.assert_array_equal(
            ptrs[:, 0], 1000 + 2 * 65536 + np.arange(32) * 2048
        )
        np.testing.assert_array_equal(sizes, np.full((32, 1), 2048))


class MatrixTransferTest(unittest.TestCase):
    def _metadata(self, keys=(b"a" * 16, b"b" * 16)):
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            "FA", keys, 0, 8, (np.array([2, 2], dtype=np.uint64),)
        )
        return UCMConnectorMetadata(
            requests={
                "r": RequestDispatchMeta("r", load_plans=(plan,), dump_plans=(plan,))
            }
        )

    def test_matrix_bytes_and_layerwise_offsets(self):
        layout, _ = LayerViewSegmentsTest()._layout("LHBNC")
        meta = self._metadata()
        (batch,) = layout.build_dump_transfers(meta)
        self.assertEqual(batch.keys, (b"a" * 16, b"b" * 16))
        self.assertEqual(batch.ptrs.shape, (2, 4))
        self.assertEqual(batch.ucm_block_offsets.tolist(), [[0, 8, 16, 24]] * 2)
        self.assertEqual(batch.sizes.tolist(), [[8] * 4] * 2)
        self.assertEqual(batch.sizes.strides[0], 0)
        self.assertFalse(batch.sizes.flags.writeable)
        (selected,) = layout.build_load_transfers(meta, layer_id=1)
        np.testing.assert_array_equal(selected.ptrs, batch.ptrs[:, 2:])
        self.assertEqual(selected.ucm_block_offsets.tolist(), [[16, 24]] * 2)
        legacy = layout.build_dump_batches(meta)
        memory = ByteMemory()
        for address in range(1000, 1192):
            memory.write(address, bytes([address % 251]))
        with tempfile.TemporaryDirectory() as directory:
            proxy = SimpleFileUCMProxy(directory, MemoryByteAccess(memory))
            adapter = UCMProxyAdapter(proxy)
            adapter.submit("dump", batch)
            adapter.commit(batch.keys)
            expected = []
            for key in batch.keys:
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
        spec = parse_kv_cache_config(
            config(
                group([names[0]], FullAttentionSpec(8)),
                group([names[1]], FullAttentionSpec(16)),
                num_blocks=4,
            ),
            scheduler_block_size=8,
        )
        layout = UCMKVCacheLayout(
            spec,
            {
                names[0]: FakeTensor(1000, (4, 8, 1), (8, 1, 1)),
                names[1]: FakeTensor(2000, (4, 16, 1), (16, 1, 1)),
            },
        )
        plan = UCMGroupDispatchPlan(
            "FA", (b"a" * 16, b"b" * 16), 0, 16, (np.array([0, 1]), np.array([2, 2]))
        )
        meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        (batch,) = layout.build_load_transfers(meta)
        self.assertEqual(batch.ptrs.tolist(), [[1000, 2032], [1008, 2040]])
        self.assertEqual(batch.ucm_block_offsets.tolist(), [[0, 8], [0, 8]])
        self.assertEqual(batch.sizes.tolist(), [[8, 8], [8, 8]])
        (selected,) = layout.build_load_transfers(meta, layer_id=1)
        self.assertEqual(selected.ucm_block_offsets.tolist(), [[8], [8]])

    def test_shared_keys_across_requests_keep_separate_destinations(self):
        from dataclasses import replace

        layout, _ = LayerViewSegmentsTest()._layout("LHBNC")
        metadata = self._metadata()
        first = metadata.requests["r"]
        plan = first.load_plans[0]
        other_plan = replace(plan, windows=(np.array([1, 1], dtype=np.uint64),))
        metadata.requests["other"] = replace(
            first,
            request_id="other",
            load_plans=(other_plan,),
            dump_plans=(other_plan,),
        )
        for build in (layout.build_load_transfers, layout.build_dump_transfers):
            for layer_id in (None, 1):
                batches = build(metadata, layer_id=layer_id)
                self.assertEqual(len(batches), 2)
                self.assertEqual([b.keys for b in batches], [plan.keys, plan.keys])
                self.assertFalse(np.array_equal(batches[0].ptrs, batches[1].ptrs))

    def test_templates_and_retained_transfers_are_independent(self):
        layout, _ = LayerViewSegmentsTest()._layout("LBHNC")
        meta = self._metadata()
        (first,) = layout.build_load_transfers(meta)
        (retained,) = layout.build_load_transfers(meta)
        expected = retained.ptrs.copy()
        first.ptrs[:] = 1
        # Even a caller deliberately changing a broadcast view's backing
        # cannot change the compiled layout or another retained submission.
        first.sizes.base[:] = 1
        (later,) = layout.build_load_transfers(meta)
        np.testing.assert_array_equal(retained.ptrs, expected)
        np.testing.assert_array_equal(later.ptrs, expected)
        self.assertEqual(later.sizes.tolist(), [[8] * 4] * 2)

    def test_submit_passes_through_strided_and_broadcast_matrices(self):
        from ucm.integration.vllm.v2.ucm_proxy import UCMProxyTransfer

        class CaptureProxy(FakeProxy):
            def load(self, *args):
                self.received = args

            dump = load

        keys = (b"a" * 16, b"b" * 16)
        ptrs = np.array([[10, 11], [20, 21]], dtype=np.uint64)[:, 1:]
        sizes = np.broadcast_to(np.array([5], dtype=np.uint64), (2, 1))
        offsets = np.broadcast_to(np.array([3], dtype=np.uint64), (2, 1))
        transfer = UCMProxyTransfer(keys, ptrs, sizes, offsets)
        proxy = CaptureProxy()
        adapter = UCMProxyAdapter(proxy)
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
            for key, segments in SimpleFileUCMProxy._records(keys, offsets, ptrs, sizes)
        ]
        self.assertEqual(records, [(keys[0], [(3, 11, 5)]), (keys[1], [(3, 21, 5)])])

    def test_matrix_buffers_live_until_async_wait(self):
        layout, _ = LayerViewSegmentsTest()._layout("LBHNC")
        (batch,) = layout.build_load_transfers(self._metadata())

        class Deferred(FakeProxy):
            def load(self, keys, offsets, ptrs, sizes):
                self.refs = (keys, offsets, ptrs, sizes)
                return object()

            def wait(self, task):
                self.checked = self.refs[2].shape == (2, 4) and self.refs[3][0, 0] == 8

        proxy = Deferred()
        UCMProxyAdapter(proxy).submit("load", batch)
        self.assertTrue(proxy.checked)


if __name__ == "__main__":
    unittest.main()
