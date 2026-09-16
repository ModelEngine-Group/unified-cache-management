"""CPU contracts for the #16121 backport; these do not compile NPU kernels."""

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH_ROOT = ROOT / "ucm/integration/vllm/patch/v0260/vllm_ascend"
OPS_MODULE = "vllm_ascend.models.minimax_m3.ops.msa_m3_triton"
ATTENTION_MODULE = "vllm_ascend.models.minimax_m3.msa_m3"
BACKPORT_MODULE = "ucm.integration.vllm.patch.v0260.vllm_ascend.minimax_m3_prefill"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stub_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class Kernel:
    def __init__(self, fn=None):
        self.fn = fn
        self.arg_names = list(inspect.signature(fn).parameters) if fn else []
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.device = "npu:0"

    def stride(self, dim):
        return self.values.strides[dim] // self.values.itemsize

    def __getitem__(self, index):
        return Tensor(self.values[index])


@pytest.fixture
def runtime(monkeypatch):
    tl = SimpleNamespace()
    stub_module(
        monkeypatch,
        "vllm.triton_utils",
        tl=tl,
        triton=SimpleNamespace(
            jit=lambda **kwargs: Kernel,
            cdiv=lambda x, y: (x + y - 1) // y,
            next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
        ),
    )

    def full(shape, value, dtype, device):
        result = Tensor(np.full(shape, value, dtype=dtype))
        result.device = device
        return result

    torch = stub_module(
        monkeypatch,
        "torch",
        no_grad=lambda: lambda fn: fn,
        float32=np.float32,
        full=Mock(side_effect=full),
        topk=Mock(),
    )
    stub_module(
        monkeypatch,
        "vllm.utils.math_utils",
        round_up=lambda n, multiple: (n + multiple - 1) // multiple * multiple,
    )
    upstream = stub_module(
        monkeypatch,
        OPS_MODULE,
        PREFILL_SCORE_QUERY_TILE_SIZE=96,
        SCORE_BLOCK_STRIDE_ALIGNMENT=16,
        SPARSE_BLOCK_SIZE=128,
        _as_triton_index_kv_cache=lambda cache: cache,
        _copy_topk_indices=Mock(),
        _prefill_index_score_kernel=Kernel(),
        _mask_prefill_topk_indices_kernel=Kernel(),
        init_device_properties_triton=Mock(),
    )
    device_utils = stub_module(
        monkeypatch,
        "vllm_ascend.ops.triton.triton_utils",
        get_vectorcore_num=Mock(return_value=48),
    )
    backport = load_module(PATCH_ROOT / "minimax_m3_prefill.py", BACKPORT_MODULE)
    return SimpleNamespace(
        backport=backport,
        upstream=upstream,
        torch=torch,
        tl=tl,
        device_utils=device_utils,
    )


def test_score_initializes_invalid_tail_before_original_qk_kernel(runtime):
    q = Tensor(np.zeros((3, 2, 4)))
    cache = Tensor(np.zeros((6, 128, 4)))
    block_table = Tensor(np.zeros((2, 6)))
    offsets = Tensor([0, 2, 3])
    seq_lens = Tensor([258, 641])
    prefix = Tensor([256, 640])
    score = runtime.backport.minimax_m3_index_score(
        q,
        cache,
        block_table,
        offsets,
        seq_lens,
        prefix,
        2,
        641,
        2,
        sm_scale=0.5,
    )
    assert score.shape == (2, 3, 16)
    assert np.isneginf(score.values).all()
    assert score.device == q.device
    grid, args, kwargs = runtime.upstream._prefill_index_score_kernel.calls[0]
    assert grid == (1, 4)
    assert args[:7] == (q, cache, score, block_table, offsets, seq_lens, prefix)
    assert kwargs == {"BLOCK_SIZE_Q": 96, "BLOCK_SIZE_K": 128}


@pytest.mark.parametrize("query_len,batch_size,heads", [(2, 1, 2), (32768, 5, 16)])
@pytest.mark.parametrize("detected_cores", [0, 48])
def test_topk_launch_is_core_bounded_and_preserves_output_buffer(
    runtime, monkeypatch, query_len, batch_size, heads, detected_cores
):
    mod = runtime.backport
    monkeypatch.setattr(mod, "_detected_vectorcore_count", lambda: detected_cores)
    score = Tensor(np.zeros((heads, 5, 528)))
    offsets = Tensor(np.zeros(batch_size + 1))
    prefix = Tensor(np.zeros(batch_size))
    output = Tensor(np.zeros((heads, 5, 1024)))
    raw_indices = object()
    runtime.torch.topk.return_value = SimpleNamespace(indices=raw_indices)
    runtime.upstream._copy_topk_indices.return_value = output
    assert (
        mod.minimax_m3_index_topk(
            score, offsets, prefix, query_len, 1024, 4, 8, out=output
        )
        is output
    )
    runtime.upstream._copy_topk_indices.assert_called_once_with(
        raw_indices, 1024, output
    )
    grid, args, kwargs = mod._prepare_prefill_topk_scores_kernel.calls[0]
    group_count = heads * batch_size
    cores = detected_cores or 64
    programs = min(query_len, max(1, (cores + group_count - 1) // group_count))
    assert grid == (programs, group_count)
    assert np.prod(grid) <= max(cores, group_count) + group_count - 1
    assert np.prod(grid) <= 65535
    assert args[:3] == (score, offsets, prefix)
    assert kwargs == {
        "sparse_block_size": 128,
        "MAX_QUERY_LEN": query_len,
        "PROGRAMS_PER_BATCH_HEAD": programs,
        "BLOCK_SIZE_FORCE": 8,
    }
    assert runtime.torch.topk.call_args.kwargs == {"k": 528, "dim": -1}
    assert runtime.upstream._mask_prefill_topk_indices_kernel.calls[0][1][0] is output


@pytest.mark.parametrize("state", ["ready", "lazy", "missing", "failure", "zero"])
def test_vectorcore_detection_and_fallback(runtime, monkeypatch, state):
    mod = runtime.backport
    get_cores = runtime.device_utils.get_vectorcore_num
    if state == "lazy":
        get_cores.side_effect = [AssertionError(), 48]
    elif state == "missing":
        monkeypatch.setattr(mod, "get_vectorcore_num", None)
    elif state == "failure":
        get_cores.side_effect = RuntimeError("device unavailable")
    elif state == "zero":
        get_cores.return_value = 0
    expected = 48 if state in {"ready", "lazy"} else 0
    assert mod._detected_vectorcore_count() == expected
    assert mod._detected_vectorcore_count() == expected
    assert runtime.upstream.init_device_properties_triton.call_count == (
        1 if state in {"lazy", "zero"} else 0
    )


class Pointer:
    def __init__(self, values, offset=0):
        self.values = values.reshape(-1)
        self.offset = np.asarray(offset, dtype=np.int64)

    def __add__(self, offset):
        return Pointer(self.values, self.offset + offset)


@pytest.mark.parametrize(
    "query_lengths,prefix_lengths,score_blocks,init_blocks,local_blocks",
    [
        ([2, 1, 3], [65512, 127, 0], 528, 4, 8),
        ([127, 128, 129], [0, 127, 128], 16, 4, 8),
        ([0, 3, 0, 1, 0], [0, 256, 0, 127, 0], 16, 0, 0),
        ([2048, 2049], [32768, 65512], 528, 4, 8),
        ([3, 2], [0, 1], 1, 8, 16),
        ([2, 1], [0, 128], 16, 4, 0),
        ([1, 2], [128, 256], 16, 0, 8),
    ],
)
def test_prepare_kernel_priorities_masks_and_write_ownership(
    runtime, query_lengths, prefix_lengths, score_blocks, init_blocks, local_blocks
):
    heads = 2
    offsets = np.array([0, *np.cumsum(query_lengths)], dtype=np.int32)
    prefix = np.array(prefix_lengths, dtype=np.int32)
    scores = np.full((heads, offsets[-1], score_blocks), -np.inf, dtype=np.float32)
    expected = scores.copy()
    for batch, length in enumerate(query_lengths):
        for query in range(length):
            token = offsets[batch] + query
            valid = min((prefix[batch] + query + 128) // 128, score_blocks)
            scores[:, token, :valid] = np.arange(valid)
            expected[:, token, :valid] = np.arange(valid)
            expected[:, token, : min(init_blocks, valid)] = 1e30
            expected[:, token, max(valid - local_blocks, 0) : valid] = 1e29

    writes = np.zeros(scores.size, dtype=np.int32)
    owners = np.full(scores.size, -1, dtype=np.int32)
    program = [0, 0]

    def store(ptr, value, mask):
        indices = np.broadcast_arrays(ptr.offset, mask)[0][np.asarray(mask)]
        assert ((indices >= 0) & (indices < scores.size)).all()
        owner = program[1] * programs + program[0]
        assert ((owners[indices] == -1) | (owners[indices] == owner)).all()
        owners[indices] = owner
        np.add.at(writes, indices, 1)
        ptr.values[indices] = value

    tl = runtime.tl
    tl.program_id = lambda axis: program[axis]
    tl.load = lambda ptr: ptr.values[ptr.offset]
    tl.store = store
    tl.arange = np.arange
    tl.range = range
    tl.where = np.where
    tl.minimum = np.minimum
    tl.maximum = np.maximum
    max_query_len = max(query_lengths)
    programs = min(
        max_query_len,
        (48 + len(query_lengths) * heads - 1) // (len(query_lengths) * heads),
    )
    kernel = runtime.backport._prepare_prefill_topk_scores_kernel.fn
    for batch_head in range(len(query_lengths) * heads):
        for query_program in range(programs):
            program[:] = [query_program, batch_head]
            kernel(
                Pointer(scores),
                Pointer(offsets),
                Pointer(prefix),
                heads,
                init_blocks,
                local_blocks,
                score_blocks,
                offsets[-1] * score_blocks,
                score_blocks,
                1,
                128,
                max_query_len,
                programs,
                1 << (max(1, init_blocks, local_blocks) - 1).bit_length(),
            )
    np.testing.assert_array_equal(scores, expected)
    assert (writes[np.isneginf(expected).reshape(-1)] == 0).all()
    assert writes.max() <= 2  # Only init/local overlap may write a score twice.


@pytest.mark.parametrize("already_imported", [False, True])
def test_import_order_patches_only_attention_prefill_aliases(
    runtime, monkeypatch, already_imported
):
    upstream = runtime.upstream
    old_score, old_topk, decode = object(), object(), object()
    upstream.minimax_m3_index_score = old_score
    upstream.minimax_m3_index_topk = old_topk
    upstream.minimax_m3_index_decode = decode
    upstream._prepare_prefill_topk_scores_kernel = SimpleNamespace(
        arg_names=["BLOCK_SIZE_Q", "BLOCK_SIZE_TAIL", "BLOCK_SIZE_FORCE"]
    )
    original_upstream = vars(upstream).copy()
    attention = SimpleNamespace(
        minimax_m3_index_score=old_score,
        minimax_m3_index_topk=old_topk,
        minimax_m3_index_decode=decode,
    )
    stub_module(monkeypatch, "ucm.logger", init_logger=lambda _: Mock())
    stub_module(
        monkeypatch, "vllm_ascend.models.minimax_m3.ops", msa_m3_triton=upstream
    )
    stub_module(
        monkeypatch,
        "ucm.integration.vllm.patch.v0260.vllm_ascend",
        minimax_m3_prefill=runtime.backport,
    )
    available = {ATTENTION_MODULE: attention} if already_imported else {}
    hooks = {}

    def when_imported(name):
        def register(fn):
            hooks[name] = fn
            if name in available:
                fn(available[name])
            return fn

        return register

    stub_module(
        monkeypatch, "ucm.integration.vllm.patch.utils", when_imported=when_imported
    )
    patch = load_module(
        PATCH_ROOT / "minimax_m3_prefill_patch.py", "m3_prefill_hook_test"
    )
    assert set(hooks) == {ATTENTION_MODULE}
    if not already_imported:
        hooks[ATTENTION_MODULE](attention)
    patch.patch_minimax_m3_prefill(attention)
    assert attention.minimax_m3_index_score is runtime.backport.minimax_m3_index_score
    assert attention.minimax_m3_index_topk is runtime.backport.minimax_m3_index_topk
    assert attention.minimax_m3_index_decode is decode
    assert vars(upstream) == original_upstream


@pytest.mark.parametrize("native_fix", [False, True])
def test_native_fix_is_preserved_and_unknown_interface_fails(monkeypatch, native_fix):
    stub_module(monkeypatch, "ucm.logger", init_logger=lambda _: Mock())
    stub_module(
        monkeypatch,
        "ucm.integration.vllm.patch.utils",
        when_imported=lambda _: lambda fn: fn,
    )
    patch = load_module(
        PATCH_ROOT / "minimax_m3_prefill_patch.py", "m3_prefill_guard_test"
    )
    upstream = SimpleNamespace(
        _prepare_prefill_topk_scores_kernel=SimpleNamespace(
            arg_names=["MAX_QUERY_LEN", "PROGRAMS_PER_BATCH_HEAD"] if native_fix else []
        ),
    )
    stub_module(
        monkeypatch, "vllm_ascend.models.minimax_m3.ops", msa_m3_triton=upstream
    )
    module = SimpleNamespace(
        minimax_m3_index_score=object(),
        minimax_m3_index_topk=object(),
    )
    original = vars(module).copy()
    if native_fix:
        patch.patch_minimax_m3_prefill(module)
    else:
        with pytest.raises(RuntimeError, match="check the installed source"):
            patch.patch_minimax_m3_prefill(module)
    assert vars(module) == original
