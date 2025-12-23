import numpy as np
import os
import pathlib
import sysconfig
import subprocess
import torch
import pytest
import time
from build_utils import build_shared

def load_module():
    """
    Load the CUDA extension.

    Preference order:
    1. If USE_TORCH_EXTENSION=1 (default) and torch is available, build/load via torch.utils.cpp_extension.load.
    2. Otherwise, build a local interface.so with nvcc and load it from disk.
    """
    use_torch = os.environ.get("USE_TORCH_EXTENSION", "0") == "1"
    if use_torch:
        print("==== torch_compile")
        try:
            import torch
            from torch.utils.cpp_extension import load as torch_load
            # torch ships pybind11 headers and handles the build toolchain
            mod = torch_load(
                name="interface",
                sources=["esa_interface.cc", "esa_kernels.cu", "esa_sm_copy.cu"],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3"],
                verbose=True,
            )
            return mod
        except Exception as e:
            print(f"[warn] torch extension build failed, falling back to nvcc: {e}")

    so_path = pathlib.Path(__file__).with_name("esa_interface.so")
    if not so_path.exists():
        build_shared(["./esa_interface.cc", "./esa_kernels.cu", "./esa_sm_copy.cu"], "esa_interface.so")

    # import importlib.machinery
    # import importlib.util
    # # Load the extension from an explicit path without modifying sys.path
    # loader = importlib.machinery.ExtensionFileLoader("interface", str(so_path))
    # spec = importlib.util.spec_from_loader(loader.name, loader)
    # if spec is None:
    #     raise RuntimeError("Failed to create spec for interface.so")
    # module = importlib.util.module_from_spec(spec)
    # loader.exec_module(module)
    import esa_interface as module
    return module


esa_lib = load_module()
esa_retrieval = esa_lib.esa_retrieval
esa_topk = esa_lib.esa_topk
esa_repre = esa_lib.esa_repre
esa_copy = esa_lib.esa_copy
esa_copy_batch = esa_lib.esa_copy_batch
esa_scatter_copy = esa_lib.esa_scatter_copy

class style():
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

def print_red(msg):
    print(style.RED + msg + style.RESET)

def print_green(msg):
    print(style.GREEN + msg + style.RESET)

def print_blue(msg):
    print(style.BLUE + msg + style.RESET)

def print_yellow(msg):
    print(style.YELLOW + msg + style.RESET)

@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("num_repre_blocks", [50, 100])
@pytest.mark.parametrize("num_q_heads", [8, 16, 40])
def test_esa_retrieval(batch_size, num_repre_blocks, num_q_heads):
    dim = 128
    print(f'''TEST esa_retrieval
{' '*4}number of queries (a.k.a batch_size): {batch_size}
{' '*4}number of blocks per query: {num_repre_blocks}
{' '*4}heads: {num_q_heads}\n''')
    total_blocks = num_repre_blocks * batch_size
    N = total_blocks * 2
    num_k_heads = 8
    dtype = torch.bfloat16
    query = torch.randn(batch_size, num_q_heads, dim, dtype=dtype).cuda()
    repre_cache = torch.randn(N, num_k_heads, dim, dtype = dtype).cuda()
    rng = np.random.default_rng()
    range_n = np.arange(N)
    repre_index = rng.choice(range_n, size=total_blocks, replace=False)
    repre_index = torch.from_numpy(repre_index).to(torch.int32).cuda()
    q_index = torch.randint(0, batch_size, size = [total_blocks], dtype = torch.int32).cuda()
    score = torch.zeros(total_blocks, dtype = dtype).cuda()
    score_sorted = torch.zeros(total_blocks, dtype = dtype).cuda()

    index_sorted = torch.arange(0, total_blocks, dtype=torch.int32).cuda()
    batch_offset = []
    for i in range(batch_size + 1):
        batch_offset.append(i * num_repre_blocks)
    batch_offset = torch.tensor(batch_offset, dtype=torch.int32).cuda()
    workspace = torch.zeros(10000, dtype=torch.int32).cuda()
    # ptrs_host = torch.tensor([q.data_ptr() for q in query_list],
    #                          dtype=torch.int64, pin_memory=True)
    # ptrs_dev = torch.zeros(batch_size, dtype=torch.int64, device="cuda")
    # size = ptrs_host.numel() * ptrs_host.element_size()
    # Input.q_ptrs = ptrs_dev
    # esa_copy(ptrs_host, ptrs_dev, size) # then we use ptrs_dev as the input of esa_retrieval

    Input = esa_lib.RetrievalInputTensor()

    Input.query = query
    Input.repre_cache = repre_cache
    Input.q_index = q_index
    Input.repre_index = repre_index
    Input.batch_offset = batch_offset
    Input.workspace = workspace
    Input.batch = batch_size
    Input.s = total_blocks

    Output = esa_lib.RetrievalOutputTensor()
    Output.score = score
    Output.index = repre_index
    Output.score_sorted = score_sorted
    Output.index_sorted = index_sorted

    start = time.perf_counter_ns()
    esa_retrieval(Input, Output)
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_green(f"{' '*4}esa_retrieval host API time: {duration/1e6:.3f} ms")

    def naive_retrieval():
        query_batched = query[q_index].to(torch.float32)
        key = torch.repeat_interleave(repre_cache[repre_index],
                                      num_q_heads//num_k_heads,
                                      dim=1).to(torch.float32)
        score_gt = (query_batched * key).sum(-1).sum(-1).to(dtype)
        index_gt = torch.cat([ repre_index[s:t][score_gt[s:t].argsort(descending=True)] for s,t in zip(batch_offset[:-1], batch_offset[1:]) ])
        return score_gt, index_gt

    start = time.perf_counter_ns()
    score_gt, index_gt = naive_retrieval()
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_red(f"{' '*4}naive_retrieval host API time: {duration/1e6:.3f} ms")

    diff = (score - score_gt).abs()
    print_blue(f"{' '*4}score diff: {diff.mean():.3f}(mean), {diff.max():.3f}(max)")
    diff_index = (index_sorted - index_gt).abs().to(torch.float32)
    print_blue(f"{' '*4}index diff: {diff_index.mean():.0f}(mean), {diff_index.max():.0f}(max)")
    print("")
    assert diff.mean() < 1e-3


@pytest.mark.parametrize("num_repre_blocks", [100])
@pytest.mark.parametrize("dim", [128])
def test_esa_repre(num_repre_blocks, dim):# extract repre
    print(f'''TEST esa_repre
{' '*4}total number of blocks to extract_repre: {num_repre_blocks}
{' '*4}dim (num_heads * hidden_size): {dim}\n''')
    dtype = torch.bfloat16
    N = 2 * num_repre_blocks
    block_size = 128
    key_cache = torch.randn(N, block_size, 8, dim, dtype=dtype).cuda()
    repre_cache = torch.randn(N, 8, dim, dtype=dtype).cuda()
    repre_cache2 = torch.randn(N, 8, dim, dtype=dtype).cuda()

    range_n = np.arange(N)
    rng = np.random.default_rng()
    repre_index = rng.choice(range_n, size=num_repre_blocks, replace=False)
    repre_index = torch.from_numpy(repre_index).to(torch.int32).cuda()

    rng2 = np.random.default_rng()
    block_table = rng2.choice(range_n, size=num_repre_blocks, replace=False)
    block_table = torch.from_numpy(block_table).to(torch.int32).cuda()

    start = time.perf_counter_ns()
    esa_repre(key_cache.flatten(-2, -1), repre_cache.flatten(-2, -1), block_table, repre_index)
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_green(f"{' '*4}[esa_repre] host API time: {duration / 1e6:.3f} ms")

    start = time.perf_counter_ns()
    for r_id, b_id in zip(repre_index, block_table):
        repre_cache2[r_id] = key_cache[b_id].mean(0)
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_red(f"{' '*4}[naive_repre] host API time: {duration / 1e6:.3f} ms")

    diff = (repre_cache2[repre_index] - repre_cache[repre_index]).abs()
    print_blue(f"{' '*4}[esa_repre] repre diff: {diff.mean():.3f}(mean), {diff.max():.3f}(max)")
    print("")
    assert diff.mean() < 1e-3

def test_esa_copy():# extract repre
    print(f'''TEST esa_copy''')
    host = torch.randn(100, 128, 128, pin_memory=True, device="cpu", dtype=torch.float32)
    dev = torch.zeros(100, 128, 128, device="cuda", dtype=torch.float32)
    dev2 = torch.zeros(100, 128, 128, device="cuda", dtype=torch.float32)
    size = host.numel() * host.element_size()

    start = time.perf_counter_ns()
    esa_copy(host, dev, size)
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_green(f"{' '*4}[esa_copy] host API time: {duration / 1e6:.3f} ms")


    start = time.perf_counter_ns()
    dev2.copy_(host)
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_red(f"{' '*4}[naive_copy] host API time: {duration / 1e6:.3f} ms")

    diff = (dev - host.cuda()).abs()
    diff2 = (dev2 - host.cuda()).abs()
    print_blue(f"{' '*4}[esa_copy] diff: {diff.mean():.3f}(mean), {diff.max():.3f}(max), {diff2.mean():.3f}(mean)")
    print("")
    assert diff.max() < 1e-5

def test_esa_scatter_copy():# extract repre
    print(f'''TEST esa_copy''')
    host = torch.randn(100, 128 * 128, pin_memory=True, device="cpu", dtype=torch.float32)
    dev = torch.zeros(100, 128 * 128, device="cuda", dtype=torch.float32)
    block_table_host = torch.arange(0, 10, device="cuda", dtype=torch.int32)
    block_table_dev = torch.arange(10, 20, device="cuda", dtype=torch.int32)

    times = []
    for _ in range(20):
        start = time.perf_counter_ns()
        esa_scatter_copy(dev, host, block_table_dev, block_table_host)
        torch.cuda.synchronize()
        duration = time.perf_counter_ns() - start
        times.append(duration)
        print_green(f"{' '*4}[esa_scatter_copy] host API time: {duration / 1e6:.3f} ms")
    times = times[10:]
    duration_seconds = sum(times) / len(times) / 1e9
    bytes = block_table_dev.shape[0] * dev.shape[1] * dev.element_size()
    GB = bytes / 1024**3
    bw = GB/duration_seconds
    print("[esa_scatter_copy] bw: ", bw)
    diff = (dev[block_table_dev] - host[block_table_host.cpu()].cuda()).abs()
    print(f"diff: {diff.max()}, {diff.mean()}")
    assert diff.max() < 1e-5


    block_table_host = torch.arange(30, 40, device="cuda", dtype=torch.int32)
    block_table_dev = torch.arange(40, 50, device="cuda", dtype=torch.int32)
    times = []
    for _ in range(20):
        start = time.perf_counter_ns()
        host_ptrs = torch.tensor([host[e].data_ptr() for e in block_table_host], dtype=torch.uint64, device="cuda")
        dev_ptrs = torch.tensor([dev[e].data_ptr() for e in block_table_dev], dtype=torch.uint64, device="cuda")
        size_each = host.shape[1] * host.element_size()
        esa_copy_batch(host_ptrs, dev_ptrs, size_each)
        torch.cuda.synchronize()
        duration = time.perf_counter_ns() - start
        times.append(duration)
        print_green(f"{' '*4}[esa_copy_batch] host API time: {duration / 1e6:.3f} ms")
    times = times[10:]
    duration_seconds = sum(times) / len(times) / 1e9
    bytes = block_table_dev.shape[0] * dev.shape[1] * dev.element_size()
    GB = bytes / 1024**3
    bw = GB/duration_seconds
    print("[esa_copy_batch] bw: ", bw)
    diff = (dev[block_table_dev] - host[block_table_host.cpu()].cuda()).abs()
    print(f"diff: {diff.max()}, {diff.mean()}")
    assert diff.max() < 1e-5

def test_topk():
    a = torch.randn(100, device="cuda", dtype=torch.float32)
    index = torch.arange(a.shape[0], device="cuda", dtype=torch.int32)
    index_sorted = torch.arange(a.shape[0], device="cuda", dtype=torch.int32)
    b = torch.empty_like(a)
    offset = torch.tensor([0, a.shape[0]], dtype=torch.int32, device="cuda")
    workspace = torch.zeros(10000, dtype=torch.int32, device="cuda")

    times = []
    for _ in range(20):
        start = time.perf_counter_ns()
        esa_topk(a, index, offset, b, index_sorted, workspace)
        torch.cuda.synchronize()
        duration = time.perf_counter_ns() - start
        times.append(duration)
    times = times[10:]
    print('times:', sum(times)/len(times)/1e6)

    a_np = a.cpu().numpy()
    times = []
    for _ in range(20):
        start = time.perf_counter_ns()
        b_np = np.sort(a_np)
        duration = time.perf_counter_ns() - start
        times.append(duration)
    times = times[10:]
    print('np_times:', sum(times)/len(times)/1e6)

if __name__ == "__main__":
    # test_esa_retrieval(2, 50, 40)
    test_topk()
