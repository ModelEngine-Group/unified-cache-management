import enum
import inspect
import json
import os
import sys
import tempfile
import types
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
    UCMConnectorMetadata,
    UCMDispatcher,
    UCMLookupResult,
    UCMLookupCoordinator,
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


def config(*groups, num_blocks=8):
    return SimpleNamespace(kv_cache_groups=list(groups), num_blocks=num_blocks)


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


class FakeProxy:
    def __init__(self):
        self.present = set()
        self.lookup_calls = []
        self.load_calls = []
        self.dump_calls = []

    def lookup(self, keys):
        self.lookup_calls.append(tuple(keys))
        return [key in self.present for key in keys]

    def load(self, block_ids, offsets, ptrs, sizes):
        self.load_calls.append((block_ids, offsets, ptrs, sizes))

    def dump(self, block_ids, offsets, ptrs, sizes):
        self.dump_calls.append((block_ids, offsets, ptrs, sizes))


class ByteMemory:
    def __init__(self):
        self.values = {}

    def write(self, ptr, data):
        for index, value in enumerate(data):
            self.values[ptr + index] = value

    def read(self, ptr, size):
        return bytes(self.values.get(ptr + index, 0) for index in range(size))


class InMemoryByteProxy(FakeProxy):
    """Synchronous test-only Proxy; one dump call publishes complete records."""

    def __init__(self, memory):
        super().__init__()
        self.memory = memory
        self.records = {}

    def lookup(self, keys):
        return [key in self.records for key in keys]

    def dump(self, block_ids, offsets, ptrs, sizes):
        pending = {}
        for key, offset, ptr, size in zip(block_ids, offsets, ptrs, sizes):
            record = pending.setdefault(key, bytearray())
            required = offset + size
            record.extend(b"\x00" * max(required - len(record), 0))
            record[offset:required] = self.memory.read(ptr, size)
        self.records.update({key: bytes(value) for key, value in pending.items()})

    def load(self, block_ids, offsets, ptrs, sizes):
        for key, offset, ptr, size in zip(block_ids, offsets, ptrs, sizes):
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
            chunk_size=512,
        )

        self.assertFalse(parsed.is_dsv4)
        self.assertEqual(parsed.groups[0].hash_block_size, 128)
        self.assertEqual(parsed.chunk_size, 512)
        self.assertEqual(parsed.layer_to_group["model.layers.0.attn"], 0)

    def test_hybrid_classifies_attention_and_state(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            scheduler_block_size=128,
        )

        self.assertFalse(parsed.is_dsv4)
        self.assertEqual(parsed.alignment_block_size, 128)
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
                group(["model.layers.2.attn"], MLAAttentionSpec(128, 4)),
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 4096),
                ),
                group(
                    ["model.layers.2.compressor.state_cache"],
                    AscendSlidingWindowMLASpec(128, 4, 4096),
                ),
            ),
            scheduler_block_size=16,
        )

        self.assertTrue(parsed.is_dsv4)
        self.assertEqual(parsed.chunk_size, 512)
        self.assertEqual(parsed.c4a_group.group_id, 0)
        self.assertEqual(parsed.groups[2].token_block_size, 512)
        self.assertEqual(
            parsed.groups[2].kinds,
            frozenset((KVCacheSpecKind.SLIDING_WINDOW_MLA,)),
        )
        self.assertTrue(parsed.groups[2].is_attention)
        self.assertTrue(parsed.groups[2].is_sliding_window)

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
        self.assertFalse(parsed.is_dsv4)

    def test_hybrid_rejects_custom_chunk(self):
        with self.assertRaisesRegex(ValueError, "custom chunk_size"):
            parse_kv_cache_config(
                config(
                    group(["model.layers.0.attn"], FullAttentionSpec(128)),
                    group(["model.layers.1.mixer"], MambaSpec(128)),
                ),
                scheduler_block_size=128,
                chunk_size=512,
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

        self.assertEqual((glm.is_dsv4, len(glm.groups)), (False, 1))
        self.assertEqual((dsv4.is_dsv4, len(dsv4.groups)), (True, 6))
        self.assertEqual(dsv4.c4a_group.group_id, 0)
        self.assertEqual(dsv4.chunk_size, 512)
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

        self.assertTrue(parsed.is_dsv4)
        self.assertEqual(parsed.chunk_size, 256)
        # FA is the 256-token indexer+attention group; WA covers both SWA
        # groups and both compressor state groups.
        self.assertEqual(tuple(group.group_id for group in parsed.fa_groups), (2,))
        self.assertEqual(
            tuple(group.group_id for group in parsed.wa_groups), (0, 1, 3, 4)
        )


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

    def test_custom_chunk_selects_h3_h7_from_base_chain(self):
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
            chunk_size=512,
        )
        proxy = FakeProxy()
        coordinator = UCMLookupCoordinator(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed", recompute_tokens=0
        )
        request = FakeRequest("r", 1024)
        base_chain = coordinator._chain(request.all_token_ids, 128, b"seed")
        proxy.present.update((base_chain[3], base_chain[7]))

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.group_ucm_block_ids[0], (base_chain[3], base_chain[7]))
        self.assertEqual(result.external_hit_tokens, 1024)

    def test_direct_prefix_stops_at_first_miss(self):
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
        )
        proxy = FakeProxy()
        coordinator = UCMLookupCoordinator(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed", recompute_tokens=0
        )
        request = FakeRequest("r", 512)
        keys = coordinator._attention_keys(request.all_token_ids, parsed.groups[0])
        proxy.present.update((keys[0], keys[2], keys[3]))

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.external_hit_tokens, 128)

    def test_direct_miss_never_moves_an_unaligned_hbm_boundary_backwards(self):
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
            chunk_size=512,
        )
        proxy = FakeProxy()
        coordinator = UCMLookupCoordinator(
            parsed,
            UCMProxyAdapter(proxy),
            RequestHasher(vllm_config(), 0),
            b"seed",
        )

        result = coordinator.lookup(FakeRequest("unaligned", 1024), 128)

        self.assertEqual(result.external_hit_tokens, 0)
        self.assertEqual(result.restore_end_tokens, 128)

    def test_direct_full_hit_loads_last_block_but_reports_one_less_token(self):
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
        )
        proxy = FakeProxy()
        coordinator = UCMLookupCoordinator(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed"
        )
        request = FakeRequest("r", 512)
        proxy.present.update(
            coordinator._attention_keys(request.all_token_ids, parsed.groups[0])
        )

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.restore_end_tokens, 512)
        self.assertEqual(result.external_hit_tokens, 511)

    def test_hybrid_reverse_selects_latest_common_state(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            scheduler_block_size=128,
        )
        proxy = FakeProxy()
        coordinator = UCMLookupCoordinator(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed", recompute_tokens=0
        )
        request = FakeRequest("r", 1024)
        attn_keys = coordinator._chain(
            request.all_token_ids, 128, coordinator._group_seed(0)
        )
        proxy.present.update(attn_keys)
        state_512 = self.hasher(
            (coordinator._group_seed(1), b"UCM_MAMBA_ALIGN_STATE", 512, attn_keys[3])
        )
        proxy.present.add(state_512)

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.restore_end_tokens, 512)
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
        coordinator = UCMLookupCoordinator(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed"
        )
        request = FakeRequest("r", 1024)
        attn_keys = coordinator._chain(
            request.all_token_ids, 128, coordinator._group_seed(0)
        )
        state_keys = tuple(
            self.hasher(
                (
                    coordinator._group_seed(1),
                    b"UCM_MAMBA_ALIGN_STATE",
                    boundary,
                    attn_keys[boundary // 128 - 1],
                )
            )
            for boundary in range(128, 1025, 128)
        )
        proxy.present.update((*attn_keys, *state_keys))

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.restore_end_tokens, 896)
        self.assertEqual(result.external_hit_tokens, 896)

    def test_hybrid_miss_still_returns_state_keys_for_later_dump(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.mixer"], MambaSpec(128)),
            ),
            scheduler_block_size=128,
        )
        coordinator = UCMLookupCoordinator(
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
                group(["model.layers.2.attn"], MLAAttentionSpec(128, 4)),
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 4096),
                ),
            ),
            scheduler_block_size=16,
        )
        proxy = FakeProxy()
        coordinator = UCMLookupCoordinator(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed", recompute_tokens=0
        )
        request = FakeRequest("r", 1536)
        fa_keys = coordinator._chain(
            request.all_token_ids, 512, self.hasher(b"FA_Block")
        )
        wa_keys = coordinator._chain(
            request.all_token_ids, 512, self.hasher(b"WA_Block")
        )
        proxy.present.update(fa_keys)
        proxy.present.add(wa_keys[1])

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.restore_end_tokens, 1024)
        self.assertEqual(result.group_ucm_block_ids, (fa_keys, wa_keys))

    def test_dsv4_full_hit_leaves_one_complete_canonical_block(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.2.attn"], MLAAttentionSpec(128, 4)),
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 4096),
                ),
            ),
            scheduler_block_size=16,
        )
        proxy = FakeProxy()
        coordinator = UCMLookupCoordinator(
            parsed, UCMProxyAdapter(proxy), self.hasher, b"seed"
        )
        request = FakeRequest("r", 1536)
        fa_keys = coordinator._chain(
            request.all_token_ids, 512, self.hasher(b"FA_Block")
        )
        wa_keys = coordinator._chain(
            request.all_token_ids, 512, self.hasher(b"WA_Block")
        )
        proxy.present.update((*fa_keys, *wa_keys))

        result = coordinator.lookup(request, 0)

        self.assertEqual(result.restore_end_tokens, 1024)
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
            adapter.dump(
                (key, key),
                (0, 5),
                (0x1000, 0x2000),
                (5, 7),
            )

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

    def test_simple_file_proxy_rejects_incomplete_or_overlapping_dump(self):
        key = b"g" * 16
        memory = ByteMemory()
        access = MemoryByteAccess(memory)
        with tempfile.TemporaryDirectory() as directory:
            adapter = UCMProxyAdapter(SimpleFileUCMProxy(directory, access))
            with self.assertRaisesRegex(UCMProxyError, "Proxy dump failed"):
                adapter.dump((key,), (4,), (0x1000,), (8,))
            with self.assertRaisesRegex(UCMProxyError, "Proxy dump failed"):
                adapter.dump((key, key), (0, 4), (0x1000, 0x2000), (8, 8))

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
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            0,
            (b"x" * 16,),
            0,
            0,
            4,
            (UCMGroupBlockIds(0, 0, (5,)),),
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
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        def request_meta(request_id, key, block_id):
            plan = UCMGroupDispatchPlan(
                0,
                (key,),
                0,
                0,
                4,
                (UCMGroupBlockIds(0, 0, (block_id,)),),
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
        self.dispatcher = UCMDispatcher(self.parsed)

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
        return self.dispatcher.record_lookup(request, 0, UCMLookupResult(0, 0, (keys,)))

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
            chunk_size=512,
        )
        dispatcher = UCMDispatcher(parsed)
        request = FakeRequest("chunked", 1024)
        keys = (b"a" * 16, b"b" * 16)
        state = dispatcher.record_lookup(request, 0, UCMLookupResult(0, 0, (keys,)))
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
        self.assertEqual(second_plan.vllm_blocks[0].block_ids, (1, 2, 3, 4))
        third_plan = third.requests["chunked"].dump_plans[0]
        self.assertEqual(
            (third_plan.keys, third_plan.token_start, third_plan.token_end),
            ((keys[1],), 512, 1024),
        )
        self.assertEqual(third_plan.vllm_blocks[0].block_ids, (5, 6, 7, 8))

    def test_external_load_plan_is_consumed_after_first_scheduled_step(self):
        state = self.add_state()
        state.external_hit_tokens = 128
        state.restore_end_tokens = 128
        state.token_processed = 128
        state.load_pending = True
        state.group_vllm_block_ids = ([3, 4, 5, 6],)

        first = self.dispatcher.build_metadata({"r": 128})
        second = self.dispatcher.build_metadata({"r": 128})

        self.assertEqual(len(first.requests["r"].load_plans), 1)
        self.assertEqual(first.requests["r"].load_plans[0].token_start, 0)
        self.assertEqual(first.requests["r"].load_plans[0].token_end, 128)
        self.assertEqual(second.requests["r"].load_plans, ())

    def test_dsv4_wa_dispatch_uses_only_final_boundary_and_real_tails(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.2.attn"], MLAAttentionSpec(128, 4)),
                group(["model.layers.3.attn"], MLAAttentionSpec(128, 128)),
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
        dispatcher = UCMDispatcher(parsed)
        request = FakeRequest("dsv4", 1024)
        fa_keys = (b"a" * 16, b"b" * 16)
        wa_keys = (b"c" * 16, b"d" * 16)
        state = dispatcher.record_lookup(
            request, 0, UCMLookupResult(1024, 1024, (fa_keys, wa_keys))
        )
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

        self.assertEqual(tuple(item.group_id for item in fa_plan.vllm_blocks), (0, 1))
        self.assertEqual(wa_plan.keys, (wa_keys[1],))
        self.assertEqual(
            tuple(
                (item.group_id, item.start_block_index, item.block_ids)
                for item in wa_plan.vllm_blocks
            ),
            ((2, 7, (7,)), (3, 127, (127,))),
        )


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
            num_blocks=8,
        )

        _, component, _, _, _ = layout.group_layouts[0].entries[0]
        self.assertEqual(parsed.groups[0].token_block_size, 256)
        self.assertEqual(component.states_per_row, 64)

    def test_combined_mamba_raw_page_uses_one_io_region(self):
        parsed = parse_kv_cache_config(
            config(
                group(
                    ["model.layers.0.mixer"],
                    MambaSpec(
                        4,
                        shapes=((2, 2), (2,)),
                        dtypes=(
                            SimpleNamespace(itemsize=2),
                            SimpleNamespace(itemsize=4),
                        ),
                        page_size_bytes=64,
                    ),
                )
            ),
            scheduler_block_size=4,
            device_type="cpu",
        )
        layout = UCMKVCacheLayout(
            parsed,
            {
                "model.layers.0.mixer": FakeTensor(
                    0x1000,
                    (8, 1, 1, 64),
                    (64, 64, 64, 1),
                )
            },
            num_blocks=8,
        )

        # The combined byte page stays a single component: whole-block
        # state IO is a byte copy, so the conv/SSM split the spec
        # describes adds no addressing information.  The entry carries
        # the content (16B); the page padding (64-byte stride) stays
        # outside the record.
        group_layout = layout.group_layouts[0]
        self.assertEqual(len(group_layout.entries), 1)
        _, component, offset, _, _ = group_layout.entries[0]
        self.assertEqual(
            (component.base_ptr, component.block_stride, component.payload_bytes),
            (0x1000, 64, 16),
        )

        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        key = b"m" * 16
        plan = UCMGroupDispatchPlan(0, (key,), 0, 0, 4, (UCMGroupBlockIds(0, 0, (3,)),))
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = layout.build_load_batches(metadata)
        self.assertEqual(batch.ptrs, (0x1000 + 3 * 64,))
        self.assertEqual(batch.sizes, (16,))
        self.assertEqual(batch.offsets, (0,))

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
            num_blocks=8,
        )

        self.assertEqual(
            tuple(
                (layer_name, component.base_ptr)
                for layer_name, component, _, _, _ in layout.group_layouts[0].entries
            ),
            (
                ("model.layers.0.attn", 0x1000),
                ("model.layers.0.attn", 0x2000),
                ("model.layers.1.attn", 0x3000),
            ),
        )
        # One entry per component, straight from its view: layer 0's two
        # tensors, then layer 1's single 4-D view (trailing dims permuted,
        # 24-byte payload == row stride).
        group_layout = layout.group_layouts[0]
        self.assertEqual(
            tuple(
                (component.base_ptr, component.payload_bytes)
                for _, component, _, _, _ in group_layout.entries
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
                num_blocks=8,
            )

    def test_partial_range_keeps_block_wise_entries(self):
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn"], FullAttentionSpec(128))),
            scheduler_block_size=128,
        )
        layout = UCMKVCacheLayout(
            parsed,
            {
                "model.layers.0.attn": FakeTensor(
                    0x1000,
                    (8, 128, 1),
                    (128, 1, 1),
                )
            },
            num_blocks=8,
        )

        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        key = b"p" * 16
        plan = UCMGroupDispatchPlan(
            0,
            (key,),
            0,
            64,
            192,
            (UCMGroupBlockIds(0, 0, (2, 3)),),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = layout.build_load_batches(metadata)

        # The tail of block 2 and the head of block 3 stay separate entries
        # even though they are contiguous in memory: record positions are a
        # function of the logical enumeration, never of physical adjacency.
        # Partial blocks still occupy full block-record slots (holes stay
        # zero): block 2's window sits at its in-block offset 64, block 3's
        # at slot 128.
        self.assertEqual(batch.block_ids, (key, key))
        self.assertEqual(batch.ptrs, (0x1000 + 2 * 128 + 64, 0x1000 + 3 * 128))
        self.assertEqual(batch.sizes, (64, 64))
        self.assertEqual(batch.offsets, (64, 128))

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
            num_blocks=8,
        )
        memory = ByteMemory()
        source_first = bytes(range(48))
        source_second = bytes(value + 100 for value in range(48))
        memory.write(0x1000 + 3 * 48, source_first)
        memory.write(0x1000 + 4 * 48, source_second)

        from ucm.integration.vllm.v2.ucm_proxy import UCMProxyAdapter
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        key = b"d" * 16
        adapter = UCMProxyAdapter(InMemoryByteProxy(memory))
        dump_plan = UCMGroupDispatchPlan(
            0, (key,), 0, 0, 16, (UCMGroupBlockIds(0, 0, (3, 4)),)
        )
        dump_meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", dump_plans=(dump_plan,))}
        )
        dump_batch = layout.build_dump_batches(dump_meta)
        # Blocks 3 and 4 are enumerated separately even though adjacent.
        self.assertEqual(dump_batch.sizes, (48, 48))

        # The load lands on scattered blocks; the record positions of both
        # logical blocks stay where the dump put them.
        load_plan = UCMGroupDispatchPlan(
            0, (key,), 0, 0, 16, (UCMGroupBlockIds(0, 0, (6, 2)),)
        )
        load_meta = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(load_plan,))}
        )
        load_batch = layout.build_load_batches(load_meta)
        self.assertEqual(load_batch.sizes, (48, 48))
        self.assertEqual(load_batch.offsets, (0, 48))
        self.assertEqual(load_batch.ptrs, (0x1000 + 6 * 48, 0x1000 + 2 * 48))

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
            config(group(["model.layers.3.self_attn.attn"], FullAttentionSpec(768))),
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
        layout = UCMKVCacheLayout(parsed, caches, num_blocks=2)
        key = b"k" * 16
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            0, (key,), 0, 0, 768, (UCMGroupBlockIds(0, 0, (1,)),)
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )

        batch = layout.build_load_batches(metadata)

        self.assertEqual(batch.sizes, (786432, 98304))
        self.assertEqual(batch.offsets, (0, 786432))
        self.assertEqual(batch.ptrs, (0x1000 + 786432, 0x200000 + 98304))

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
        layout = UCMKVCacheLayout(parsed, caches, num_blocks=num_blocks)
        key = b"s" * 16
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            0, (key,), 0, 1536, 2304, (UCMGroupBlockIds(1, 2, (1,)),)
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )

        batch = layout.build_load_batches(metadata)

        self.assertEqual(
            batch.ptrs,
            (
                0x5000 + conv_bytes,
                0x5000 + num_blocks * conv_bytes + ssm_bytes,
            ),
        )
        self.assertEqual(batch.sizes, (conv_bytes, ssm_bytes))
        self.assertEqual(batch.offsets, (0, conv_bytes))

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
            chunk_size=256,
        )
        caches = {
            "model.layers.0.attn": (
                FakeTensor(0x1000, (8, 128, 4), (512, 4, 1)),
                FakeTensor(0x3000, (8, 128, 2), (256, 2, 1)),
            ),
            "model.layers.1.attn": (FakeTensor(0x5000, (8, 128, 3), (384, 3, 1)),),
        }
        layout = UCMKVCacheLayout(parsed, caches, num_blocks=8)
        proxy = FakeProxy()
        coordinator = UCMLookupCoordinator(
            parsed,
            UCMProxyAdapter(proxy),
            RequestHasher(vllm_config(), 0),
            b"seed",
            recompute_tokens=0,
        )
        request = FakeRequest("r", 256)
        result = coordinator.lookup(request, 0)
        dispatcher = UCMDispatcher(parsed)
        state = dispatcher.record_lookup(request, 0, result)
        state.group_vllm_block_ids = ([2, 5],)
        state.external_hit_tokens = 256
        state.restore_end_tokens = 256
        state.load_pending = True
        metadata = dispatcher.build_metadata({"r": 1})

        batch = layout.build_load_batches(metadata)

        # Records are block-major: each plan block contributes its whole
        # record (1152 bytes: layer 0 K 512 + V 256 + layer 1 K 384 -- three
        # separate runs, since neither their strides nor their block-0
        # addresses chain) before the next block starts.
        self.assertEqual(batch.offsets, (0, 512, 768, 1152, 1664, 1920))
        self.assertEqual(batch.sizes, (512, 256, 384, 512, 256, 384))
        self.assertEqual(
            batch.ptrs,
            (0x1400, 0x3200, 0x5300, 0x1A00, 0x3500, 0x5780),
        )
        # A layerwise (filtered) batch must address the very same record
        # coordinates: a layerwise load lands exactly where the full record
        # was dumped.
        layer_batch = layout.build_load_batches(metadata, "model.layers.1.attn")
        self.assertEqual(layer_batch.offsets, (768, 1920))
        self.assertEqual(layer_batch.sizes, (384, 384))
        self.assertEqual(layer_batch.ptrs, (0x5300, 0x5780))
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
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.attn", "model.layers.1.attn"],
                         {"model.layers.0.attn": FullAttentionSpec(256, 4),
                          "model.layers.1.attn": FullAttentionSpec(256, 4)})),
            scheduler_block_size=256,
            device_type="cpu",
        )
        num_blocks = 4
        page = 4096  # 64 states x 64B, dense
        layer_stride = page
        block_stride = 2 * page  # both layers' pages tile one slot
        caches = {
            "model.layers.0.attn": FakeTensor(
                0x1000, (num_blocks, 64, 64), (block_stride, 64, 1)
            ),
            "model.layers.1.attn": FakeTensor(
                0x1000 + page, (num_blocks, 64, 64), (block_stride, 64, 1)
            ),
        }
        declarations = (
            SimpleNamespace(
                size=num_blocks * block_stride,
                layers=("model.layers.0.attn", "model.layers.1.attn"),
                offset=0,
                layer_stride=layer_stride,
                block_stride=block_stride,
            ),
        )
        layout = UCMKVCacheLayout(
            parsed, caches, num_blocks=num_blocks, kv_cache_tensors=declarations
        )
        group_layout = layout.group_layouts[0]
        self.assertEqual(len(group_layout.descriptor_spans), 1)
        span = group_layout.descriptor_spans[0]
        self.assertEqual(
            (span.base_ptr, span.block_stride, span.span_bytes),
            (0x1000, block_stride, 2 * layer_stride),
        )
        # The record is slot-sized: both layers' page slots, paddings and all.
        self.assertEqual(group_layout.record_size, 2 * layer_stride)

        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        key = b"b" * 16
        plan = UCMGroupDispatchPlan(
            0, (key,), 0, 0, 256, (UCMGroupBlockIds(0, 0, (2,)),)
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = layout.build_load_batches(metadata)
        # Whole batch: one span covering both layers' pages of block 2.
        self.assertEqual(batch.ptrs, (0x1000 + 2 * block_stride,))
        self.assertEqual(batch.sizes, (2 * layer_stride,))
        self.assertEqual(batch.offsets, (0,))

        # Layered batch: layer 1's page only, at its in-slot record offset
        # (one page in) -- the same bytes the whole-batch span covered.
        layer_batch = layout.build_load_batches(metadata, "model.layers.1.attn")
        self.assertEqual(layer_batch.ptrs, (0x1000 + page + 2 * block_stride,))
        self.assertEqual(layer_batch.sizes, (page,))
        self.assertEqual(layer_batch.offsets, (page,))

        # The public segments() API mirrors both shapes.
        self.assertEqual(
            group_layout.segments((2,)), ((0x1000 + 2 * block_stride, 2 * page),)
        )
        self.assertEqual(
            group_layout.segments((2,), layers=("model.layers.1.attn",)),
            ((0x1000 + page + 2 * block_stride, page),),
        )

        # Sub-block token ranges never use the descriptor span: they walk
        # the layer's own view and stay payload-exact (128 tokens at the
        # 4:1 compression = 32 states x 64B = 2048B).
        self.assertEqual(
            group_layout.segments(
                (2,), layers=("model.layers.0.attn",), token_range=(0, 128)
            ),
            ((0x1000 + 2 * block_stride, 2048),),
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
        )
        layout = UCMKVCacheLayout(
            parsed,
            {
                "model.layers.0.attn": FakeTensor(0x1000, (8, 4, 3), (12, 3, 1)),
                "model.layers.1.attn": FakeTensor(0x1000 + 8 * 12, (8, 4, 3), (12, 3, 1)),
            },
            num_blocks=8,
        )
        group_layout = layout.group_layouts[0]
        self.assertEqual(
            tuple(
                (component.base_ptr, component.block_stride, component.payload_bytes)
                for _, component, _, _, _ in group_layout.entries
            ),
            ((0x1000, 12, 12), (0x1000 + 8 * 12, 12, 12)),
        )
        self.assertEqual(group_layout.record_size, 24)

        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        key = b"r" * 16
        plan = UCMGroupDispatchPlan(
            0, (key,), 0, 0, 8, (UCMGroupBlockIds(0, 0, (2, 5)),)
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )
        batch = layout.build_load_batches(metadata)
        # Block-major records; adjacent physical blocks never merge into one
        # entry: cross-block adjacency is an allocation accident, not a
        # layout property.
        self.assertEqual(batch.sizes, (12, 12, 12, 12))
        self.assertEqual(batch.offsets, (0, 12, 24, 36))
        self.assertEqual(
            batch.ptrs,
            (
                0x1000 + 2 * 12,
                0x1000 + 8 * 12 + 2 * 12,
                0x1000 + 5 * 12,
                0x1000 + 8 * 12 + 5 * 12,
            ),
        )

        # The public segments() API answers the same spans without records.
        self.assertEqual(
            group_layout.segments((2, 5)),
            (
                (0x1000 + 2 * 12, 12),
                (0x1000 + 8 * 12 + 2 * 12, 12),
                (0x1000 + 5 * 12, 12),
                (0x1000 + 8 * 12 + 5 * 12, 12),
            ),
        )
        self.assertEqual(
            group_layout.segments((2,), layers=("model.layers.1.attn",)),
            ((0x1000 + 8 * 12 + 2 * 12, 12),),
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
        )
        caches = {
            "model.layers.0.attn": (
                FakeTensor(0x1000, (8, 128, 4), (512, 4, 1)),
                FakeTensor(0x3000, (8, 128, 2), (256, 2, 1)),
            ),
            "model.layers.1.attn": (FakeTensor(0x5000, (8, 128, 3), (384, 3, 1)),),
        }
        layout = UCMKVCacheLayout(parsed, caches, num_blocks=8)
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        key = b"t" * 16
        plan = UCMGroupDispatchPlan(
            0, (key,), 0, 0, 256, (UCMGroupBlockIds(0, 0, (2, 5)),)
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
                for ptr, size, offset in zip(
                    full.ptrs, full.sizes, full.offsets
                )
            ),
        )

    def test_glm_indexer_k_and_scale_keep_separate_runtime_addressing(self):
        parsed = parse_kv_cache_config(
            config(group(["model.layers.0.indexer"], FullAttentionSpec(128))),
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
            num_blocks=num_blocks,
        )
        key = b"g" * 16
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            0, (key,), 0, 0, 128, (UCMGroupBlockIds(0, 0, (2,)),)
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )

        batch = layout.build_load_batches(metadata)

        self.assertEqual(batch.ptrs, (k_base + 2 * 16384, scale_base + 2 * 256))
        self.assertEqual(batch.sizes, (16384, 256))
        self.assertEqual(batch.offsets, (0, 16384))

    def test_dsv4_canonical_subrange_selects_intersecting_large_page(self):
        parsed = parse_kv_cache_config(
            config(
                group(["model.layers.2.attn"], MLAAttentionSpec(128, 4)),
                group(["model.layers.3.attn"], MLAAttentionSpec(128, 128)),
                group(
                    ["model.layers.0.swa_cache"],
                    AscendSlidingWindowMLASpec(128, 1, 4096),
                ),
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
        layout = UCMKVCacheLayout(parsed, caches, num_blocks=4)
        key = b"k" * 16
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        plan = UCMGroupDispatchPlan(
            "FA",
            (key,),
            1,
            512,
            1024,
            (UCMGroupBlockIds(1, 0, (2,)),),
        )
        metadata = UCMConnectorMetadata(
            requests={"r": RequestDispatchMeta("r", load_plans=(plan,))}
        )

        batch = layout.build_load_batches(metadata)

        self.assertEqual(batch.sizes, (4096,))
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
        layout = UCMKVCacheLayout(parsed, caches, num_blocks=8)
        dispatcher = UCMDispatcher(parsed)
        request = FakeRequest("r", 512)
        state = dispatcher.record_lookup(
            request,
            0,
            UCMLookupResult(
                512,
                512,
                ((b"a" * 16,) * 4, (b"s" * 16,) * 4),
            ),
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
        layout = UCMKVCacheLayout(parsed, caches, num_blocks=8)
        key = b"z" * 16
        from ucm.integration.vllm.v2.ucm_scheduler import (
            RequestDispatchMeta,
            UCMGroupBlockIds,
            UCMGroupDispatchPlan,
        )

        dump_plan = UCMGroupDispatchPlan(
            0, (key,), 0, 0, 4, (UCMGroupBlockIds(0, 0, (1,)),)
        )
        load_plan = UCMGroupDispatchPlan(
            0, (key,), 0, 0, 4, (UCMGroupBlockIds(0, 0, (6,)),)
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
            0, (key,), 0, 0, 4, (UCMGroupBlockIds(0, 0, (7,)),)
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

    def _fixture(self):
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
            device_type="cpu",
        )
        # Two dense per-layer tensors, 12-byte blocks, 8 blocks each: the
        # declared placement packs them 96 bytes apart in one backing.
        caches = {
            "model.layers.0.attn": FakeTensor(0x1000, (8, 4, 3), (12, 3, 1)),
            "model.layers.1.attn": FakeTensor(0x1060, (8, 4, 3), (12, 3, 1)),
        }
        declarations = (
            SimpleNamespace(
                size=192,
                layers=("model.layers.0.attn", "model.layers.1.attn"),
                offset=0,
                layer_stride=96,
                block_stride=12,
            ),
        )
        return parsed, caches, declarations

    def test_declarations_only_feed_the_block_first_special_case(self):
        # Layer-contiguous placements (layer_stride >= block_stride is
        # false here: 96 > 12 means layer-outermost) do not tile block
        # slots, so declarations change nothing: with or without them the
        # entries and the records are identical.
        parsed, caches, declarations = self._fixture()

        declared = UCMKVCacheLayout(
            parsed,
            caches,
            num_blocks=8,
            kv_cache_tensors=declarations,
        )
        undeclared = UCMKVCacheLayout(parsed, caches, num_blocks=8)

        for group_id, declared_layout in declared.group_layouts.items():
            undeclared_layout = undeclared.group_layouts[group_id]
            self.assertEqual(declared_layout.entries, undeclared_layout.entries)
            self.assertEqual(declared_layout.record_size, undeclared_layout.record_size)
            self.assertEqual(declared_layout.descriptor_spans, ())

    def test_disagreeing_block_stride_is_rejected(self):
        parsed, caches, _ = self._fixture()
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
            UCMKVCacheLayout(parsed, caches, num_blocks=8, kv_cache_tensors=bad)

    def test_assert_mode_is_retired(self):
        parsed, caches, declarations = self._fixture()
        previous = os.environ.get("UCM_V2_DESCRIPTOR_SOURCE")
        os.environ["UCM_V2_DESCRIPTOR_SOURCE"] = "assert"
        try:
            with self.assertRaisesRegex(ValueError, "assert"):
                UCMKVCacheLayout(
                    parsed,
                    caches,
                    num_blocks=8,
                    kv_cache_tensors=declarations,
                )
        finally:
            if previous is None:
                os.environ.pop("UCM_V2_DESCRIPTOR_SOURCE", None)
            else:
                os.environ["UCM_V2_DESCRIPTOR_SOURCE"] = previous

    def test_disagreeing_block_stride_is_rejected(self):
        parsed, caches, _ = self._fixture()
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
            UCMKVCacheLayout(parsed, caches, num_blocks=8, kv_cache_tensors=bad)


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
            payload = json.loads(
                (tmp / "raw_config.json").read_text(encoding="utf-8")
            )
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


if __name__ == "__main__":
    unittest.main()
