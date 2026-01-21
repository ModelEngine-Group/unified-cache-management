import os
import secrets
import time
import shutil

import torch
import numpy as np

from ucm.store.pipeline.connector import UcmPipelineStore
from itertools import chain


def cmp_and_print_diff(a, b, rtol=0.0, atol=0.0):
    for r, (row_a, row_b) in enumerate(zip(a, b)):
        for c, (ta, tb) in enumerate(zip(row_a, row_b)):
            if not torch.allclose(ta, tb, rtol=rtol, atol=atol):
                mask = ~torch.isclose(ta, tb, rtol=rtol, atol=atol)
                diff_a = ta[mask].cpu()
                diff_b = tb[mask].cpu()
                print(f"DIFF at [{r}][{c}]  total {mask.sum().item()} element(s)")
                print("  a val:", diff_a.flatten())
                print("  b val:", diff_b.flatten())
                assert False


def e2e_test(
    worker: UcmPipelineStore,
    scheduler: UcmPipelineStore,
    tensor_size: int,
    layer_size: int,
    chunk_size: int,
    request_size: int,
    device_id: int,
):
    chunk_block_ids = [secrets.token_bytes(16) for _ in range(request_size)]
    founds = scheduler.lookup(chunk_block_ids)
    assert not all(founds)

    # shard_indexes = [0]
    # file = 'Meta-Llama-3.1-8B-Instruct_kvcache_(2,32,8,128,512)_torch.bf16_rope.bin'
    file = '/root/UCM_store_compress/unified-cache-management/ucm/store/test/e2e/BF16.bin'
    print("存在?", os.path.isfile(file))
    print("字节数", os.path.getsize(file)) 

    if not os.path.isfile(file):
        raise FileNotFoundError(f"{file} 不存在")
    raw = np.fromfile(file, dtype=np.uint16)

    big_np = raw.reshape(1, layer_size, 65536)
    big_tensor = torch.from_numpy(big_np).view(torch.bfloat16).to("cuda")
    print(big_tensor.shape)

    src_tensors = [[big_tensor[0, l] for l in range(layer_size)]]
    # print (f'src_tensors: {src_tensors}')

    print(len(src_tensors), len(src_tensors[0]))

    for i in range(layer_size):
        shard_index = [i]
        task = worker.dump(chunk_block_ids, shard_index, [[src_tensors[0][i]]])
        worker.wait(task)

    dst_tensors = [[torch.empty_like(r) for t in src_tensors for r in t]]
    # dst_tensors = src_tensors

    print(f'dst tensor size: {len(dst_tensors)}, {len(dst_tensors[0])}, {dst_tensors[0][0].shape}')
    # print (f'dst tensor before: {dst_tensors}')
    time.sleep(10)

    total_MB = 18
    time_start = time.perf_counter()
    for i in range(layer_size):
        shard_index = [i]
        task = worker.load(chunk_block_ids, shard_index, [[dst_tensors[0][i]]])
        worker.wait(task)
    
    elapsed = time.perf_counter() - time_start

    print("--------------------------------------------------")
    print(f"avg BW = {total_MB/elapsed:7.2f} MB/s")

    # print (f'dst tensor: {dst_tensors}')

    # flat_bf16 = torch.cat(dst_tensors)
    flat_bf16 = torch.cat(list(chain.from_iterable(dst_tensors)))
    flat_u16 = flat_bf16.contiguous().view(torch.uint16)
    raw = flat_u16.cpu().numpy()    
    raw.tofile('dst_tensors.bin')

    # src
    flat_bf16 = torch.cat(list(chain.from_iterable(src_tensors)))
    flat_u16 = flat_bf16.contiguous().view(torch.uint16)
    raw = flat_u16.cpu().numpy()    
    raw.tofile('src_tensors.bin')
    
    print(f'src types : {type(src_tensors)}, {type(src_tensors[0])}, {type(src_tensors[0][0])}')

    diff_count = 0
    for s, d in zip(src_tensors, dst_tensors):
        for s1, d1 in zip (s, d):
            diff_mask = ~torch.eq(s1, d1)
            dc = torch.sum(diff_mask).item()
            diff_count = diff_count + dc
    print(f"差异元素个数: {diff_count}")

    # cmp_and_print_diff(src_tensors, dst_tensors)

def clear_default_dirs() -> None:
    """
    仅当下列默认目录“非空”时，递归清空其内部所有文件/子目录，但保留目录本身。
    默认目录列表硬编码在函数内，无需外部传参。
    """
    dirs_to_clean = [
        # '/dev/shm',
        '/root/UCM_store_compress/unified-cache-management/ucm/store/test/e2e/data',
        '/root/UCM_store_compress/unified-cache-management/ucm/store/test/e2e/temp',
        '/root/UCM_store_compress/unified-cache-management/ucm/store/test/e2e/kernel_meta',
        '/root/UCM_store_compress/unified-cache-management/ucm/store/test/e2e/npu_decompressed.bin',
        '/root/UCM_store_compress/unified-cache-management/ucm/store/test/e2e/cpu_decompressed.bin'
    ]

    for path in dirs_to_clean:
        if not os.path.exists(path):          # 路径不存在就跳过
            continue

        if os.path.isfile(path) or os.path.islink(path):
            # 是文件或软链 → 直接干掉
            os.unlink(path)
            print(f'已删除文件：{path}')
        elif os.path.isdir(path):
            # 是目录 → 清空目录
            if path == '/dev/shm':
                for entry in glob('/dev/shm/unifiedcache_shm_*'):
                    if os.path.isfile(entry) or os.path.islink(entry):
                        os.unlink(entry)
                        print(f'已删除 /dev/shm 文件：{entry}')
                continue

            for entry in os.scandir(path):
                if entry.is_file() or entry.is_symlink():
                    os.unlink(entry.path)
                else:
                    shutil.rmtree(entry.path)
            print(f'已清空目录：{path}')

def main():
    clear_default_dirs()

    tensor_size = 131072  # B 
    layer_size = 144      # 2*36*2  kv * layers * head_num
    chunk_size = 1
    request_size = 1
    storage_backends = ["."]
    device_id = 0
    chunk_block_size = tensor_size * layer_size * chunk_size
    shard_size = tensor_size
    config = {}
    # config["store_pipeline"] = "Cache|Posix"
    config["store_pipeline"] = "Cache|Compress|Posix" 
    config["device_id"] = device_id
    config["storage_backends"] = storage_backends
    config["unique_id"] = secrets.token_hex(8)
    config["timeout_ms"] = 10000
    config["device_id"] = -1
    scheduler = UcmPipelineStore(config)
    config["device_id"] = device_id
    config["tensor_size"] = tensor_size
    config["shard_size"] = shard_size
    config["layer_size"] = layer_size
    config["chunk_size"] = chunk_size
    config["block_size"] = chunk_block_size
    config["compress_ratio"] = 23           # R139 = 23 1.39x, R145 = 22  1.45x,  R152 = 21  1.52x
    config["share_buffer_enable"] = True
    config["buffer_number"] = 1024
    config["waiting_queue_depth"] = 32
    config["running_queue_depth"] = 1024
    config["io_direct"] = True
    config["stream_number"] = 32
    worker = UcmPipelineStore(config)
    test_batch_number = 1
    for _ in range(test_batch_number):
        e2e_test(
            worker,
            scheduler,
            tensor_size,
            layer_size,
            chunk_size,
            request_size,
            device_id,
        )
    time.sleep(10)


if __name__ == "__main__":
    os.environ["UC_LOGGER_LEVEL"] = "debug"
    main()
