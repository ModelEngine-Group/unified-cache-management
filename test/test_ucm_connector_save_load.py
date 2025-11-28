# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
Standalone bandwidth test for `UCMConnector`.

This script instantiates the connector exactly like runtime code (no mocks) and
benchmarks `wait_for_save` (dump) and `start_load_kv` (load). 
"""

import csv
import math
import multiprocessing
import os
import secrets
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from unittest.mock import patch

import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from ucm.integration.vllm.ucm_connector import (
    RequestDispatchMeta,
    UCMConnector,
    UCMConnectorMetadata,
)
from ucm.logger import init_logger

logger = init_logger(__name__)


def make_aligned_tensor(shape, dtype, device, alignment=4096):
    numel = math.prod(shape)
    dtype_size = torch.tensor(1, dtype=dtype).element_size()
    total_bytes = numel * dtype_size

    padded_bytes = total_bytes + alignment
    storage = torch.ByteTensor(padded_bytes).to(device)

    ptr = storage.data_ptr()
    offset = ptr % alignment
    aligned_ptr = ptr + (alignment - offset) if offset != 0 else ptr

    aligned_storage = storage[(aligned_ptr - ptr) :].view(dtype)
    tensor = aligned_storage[:numel].view(shape)
    tensor.storage_ref = storage
    return tensor


def make_buffers(
    block_number: int,
    device_id: int,
    batch_size: int,
    head_dim: int,
    block_len: int,
    block_layer: int,
    num_head: int,
    kv: int,
    is_mla: bool,
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    logger.info(f"Allocating buffers: blocks={block_number}, batch_size={batch_size}")
    hashes = [secrets.token_hex(16) for _ in range(block_number)]
    device = f"cuda:{device_id}"
    kv_caches: Dict[str, torch.Tensor] = {}

    for layer in range(block_layer):
        layer_name = f"layer.{layer}"
        if is_mla:
            kv_caches[layer_name] = make_aligned_tensor(
                [block_number, block_len, head_dim],
                dtype=torch.float16,
                device=device,
            )
        else:
            kv_caches[layer_name] = make_aligned_tensor(
                [kv, block_number, block_len, num_head, head_dim],
                dtype=torch.float16,
                device=device,
            )
    return hashes, kv_caches


def build_vllm_config(
    *,
    model_path: str,
    block_size: int,
    num_layers: int,
    num_head: int,
    head_size: int,
    is_mla: bool,
    tp_size: int,
    connector_name: str,
    storage_backends: str,
    transfer_stream_number: int,
    use_direct: bool,
) -> VllmConfig:
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=4,
        cache_dtype="auto",
    )

    # This ensures connector uses test parameters for head_size, num_head, num_layers
    hf_overrides = {
        "head_dim": head_size,  # Override head_size for get_head_size()
        "num_key_value_heads": num_head,  # Override num_head for get_num_kv_heads()
        "num_hidden_layers": num_layers,  # Override num_layers for get_num_layers()
    }
    if is_mla:
        # head_dim = kv_lora_rank + qk_rope_head_dim (typically 512 + 64 = 576)
        # For testing purposes, we set kv_lora_rank = head_size - 64
        kv_lora_rank = head_size - 64  # qk_rope_head_dim = 64
        hf_overrides.update(
            {
                "model_type": "deepseek_v3",
                "kv_lora_rank": kv_lora_rank,
                "qk_rope_head_dim": 64,
            }
        )

    model_config = ModelConfig(
        model=model_path,
        tokenizer=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
        max_model_len=8192,
        max_context_len_to_capture=8192,
        max_logprobs=20,
        disable_sliding_window=False,
        skip_tokenizer_init=True,
        limit_mm_per_prompt={},
        use_async_output_proc=True,
        override_neuron_config={},
        config_format="auto",
        is_deepseek_mla=is_mla,
        hf_overrides=hf_overrides,
    )

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=tp_size,
        worker_use_ray=False,
    )

    device = "cuda" if torch.cuda.is_available() else "npu"
    device_config = DeviceConfig(device=device)

    kv_transfer_config = KVTransferConfig(
        kv_connector="UCMConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "ucm_connectors": [
                {
                    "ucm_connector_name": connector_name,
                    "ucm_connector_config": {
                        "storage_backends": storage_backends,
                        "use_direct": use_direct,
                        "stream_number": transfer_stream_number,
                        "local_rank_size": 1,
                    },
                }
            ]
        },
    )

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        device_config=device_config,
        kv_transfer_config=kv_transfer_config,
    )


@dataclass
class DummyLayer:
    kv_cache: Union[Dict[int, torch.Tensor], List[torch.Tensor]]


@dataclass
class DummyForwardContext:
    no_compile_layers: Dict[str, DummyLayer]
    virtual_engine: int = 0


def build_forward_context(
    kv_caches: Dict[str, torch.Tensor], is_mla: bool
) -> DummyForwardContext:
    layers = {}
    for layer_name, tensor in kv_caches.items():
        layers[layer_name] = DummyLayer(kv_cache={0: tensor})
    return DummyForwardContext(no_compile_layers=layers, virtual_engine=0)


def compute_total_bytes(
    kv_caches: Dict[str, torch.Tensor], batch_size: int, is_mla: bool
) -> int:
    total = 0
    for tensor in kv_caches.values():
        if is_mla:
            total += tensor[:batch_size].numel() * tensor.element_size()
        else:
            total += tensor[:, :batch_size].numel() * tensor.element_size()
    return total


def run_once(
    connector: UCMConnector,
    kv_caches: Dict[str, torch.Tensor],
    hashes: List[str],
    batch_size: int,
    is_mla: bool,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    dump_hashes = hashes[:batch_size]

    metadata = UCMConnectorMetadata()
    dump_vllm_block_ids = list(range(batch_size))
    metadata.request_meta["uc_test_write"] = RequestDispatchMeta(
        load_block_ids=([], []),
        dump_block_ids=(dump_hashes, dump_vllm_block_ids),
    )
    connector.connector.kv_caches = kv_caches
    connector.bind_connector_metadata(metadata)

    total_bytes = compute_total_bytes(kv_caches, batch_size, is_mla)

    start = time.perf_counter()
    connector.wait_for_save()
    write_time = time.perf_counter() - start

    time.sleep(1)

    write_bw = (total_bytes / (1024**3)) / write_time if write_time > 0 else 0.0

    lookup = connector.connector.store.lookup(dump_hashes)
    if not all(lookup):
        raise RuntimeError("Found missing cache blocks before load test.")

    load_metadata = UCMConnectorMetadata()
    load_vllm_block_ids = list(range(batch_size))
    load_metadata.request_meta["uc_test_read"] = RequestDispatchMeta(
        load_block_ids=(dump_hashes, load_vllm_block_ids),
        dump_block_ids=([], []),
    )
    connector.connector.kv_caches = kv_caches
    connector.bind_connector_metadata(load_metadata)

    forward_context = build_forward_context(kv_caches, is_mla)

    start = time.perf_counter()
    connector.start_load_kv(forward_context)
    read_time = time.perf_counter() - start

    read_bw = (total_bytes / (1024**3)) / read_time if read_time > 0 else 0.0

    logger.info(
        f"Size: {total_bytes / (1024**3):.4f} GB, Time: {write_time:.4f}s, WRITE SPEED: {write_bw:.4f} GB/s "
    )
    logger.info(
        f"Size: {total_bytes / (1024**3):.4f} GB, Time: {read_time:.4f}s, READ SPEED: {read_bw:.4f} GB/s"
    )

    return (
        (total_bytes / (1024**3), write_time, write_bw),
        (total_bytes / (1024**3), read_time, read_bw),
    )


def run_test(
    storage_backends: str,
    device_id: int,
    repeat: int,
    num_head: int,
    block_len: int,
    num_tokens: int,
    block_layer: int,
    head_size: int,
    block_elem_size: int,
    kv: int,
    mla: bool,
    ucm_connector_name: str,
    total_tp_size: int,
    model_path: str,
    transfer_stream_number: int,
    use_direct: bool,
) -> Tuple[float, float, float, float, float, float]:
    block_dim = head_size * num_head
    io_size = block_dim * block_len * block_elem_size
    block_size = io_size * block_layer
    batch_size = int(num_tokens / block_len)
    real_blocks = batch_size * repeat + 10

    vllm_config = build_vllm_config(
        model_path=model_path,
        block_size=block_len,
        num_layers=block_layer,
        num_head=num_head,
        head_size=head_size,
        is_mla=mla,
        tp_size=total_tp_size,
        connector_name=ucm_connector_name,
        storage_backends=storage_backends,
        transfer_stream_number=transfer_stream_number,
        use_direct=use_direct,
    )

    dummy_world_group = type("DummyWorldGroup", (), {"local_rank": 0})()

    class DummyTPGroup:
        def broadcast(self, tensor, src):
            pass

    dummy_tp_group = DummyTPGroup()

    patches = [
        patch(
            "ucm.integration.vllm.ucm_connector.get_world_group",
            return_value=dummy_world_group,
        ),
        patch(
            "ucm.integration.vllm.ucm_connector.get_tp_group",
            return_value=dummy_tp_group,
        ),
    ]

    with patches[0], patches[1]:
        connector = UCMConnector(vllm_config, KVConnectorRole.WORKER)
    connector.connector.rank = device_id if device_id >= 0 else 0
    connector.connector.kv_caches = {}

    hashes, kv_caches = make_buffers(
        real_blocks,
        device_id,
        batch_size,
        head_size,
        block_len,
        block_layer,
        num_head,
        kv,
        mla,
    )

    w_sizes, w_times, w_bws = [], [], []
    r_sizes, r_times, r_bws = [], [], []

    for round_idx in range(repeat):
        logger.info(f"Round {round_idx + 1}: start write test")
        start_hash_idx = round_idx * batch_size
        end_hash_idx = start_hash_idx + batch_size
        round_hashes = hashes[start_hash_idx:end_hash_idx]

        if len(round_hashes) < batch_size:
            round_hashes = [secrets.token_hex(16) for _ in range(batch_size)]

        (w_size, w_time, w_bw), (r_size, r_time, r_bw) = run_once(
            connector, kv_caches, round_hashes, batch_size, mla
        )
        w_sizes.append(w_size)
        w_times.append(w_time)
        w_bws.append(w_bw)
        r_sizes.append(r_size)
        r_times.append(r_time)
        r_bws.append(r_bw)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()

    def avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    avg_w_size = avg(w_sizes)
    avg_w_time = avg(w_times)
    avg_w_bw = avg(w_bws)
    avg_r_size = avg(r_sizes)
    avg_r_time = avg(r_times)
    avg_r_bw = avg(r_bws)

    logger.info(
        "\n=== Summary ===\n"
        f"Write : size={avg_w_size:.4f} GB | time={avg_w_time:.4f} s | bw={avg_w_bw:.4f} GB/s\n"
        f"Read  : size={avg_r_size:.4f} GB | time={avg_r_time:.4f} s | bw={avg_r_bw:.4f} GB/s\n"
    )

    return avg_w_size, avg_w_time, avg_w_bw, avg_r_time, avg_r_bw, avg_r_size


def run_wrapper(result_queue, *args):
    try:
        result = run_test(*args)
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", traceback.format_exc()))


def get_user_input(prompt, default=None):
    if default is not None:
        user_input = input(f"{prompt} (default: {default}): ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()


def main():
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    storage_backends = "."
    device_id = 0
    repeat = 2
    num_tokens_list = [2048, 4096, 8192, 16384, 32768]
    ucm_connector_name = "UcmNfsStore"
    model_path = "/home/models/QwQ-32B"
    transfer_stream_numbers = [32, 64, 128]
    os.environ["UC_LOGGER_LEVEL"] = "debug"

    print("1. Model Selection:")
    print("   1 - QwQ-32B")
    print("   2 - deepseek-v3")
    model_choice = get_user_input("Please select model", "1")
    mla = True if model_choice == "2" else False
    print("\n2. IoDirect Transfer:")
    print("   1 - Disable IoDirect (default)")
    print("   2 - Enable IoDirect")
    use_direct = get_user_input("Please select Direct IO mode", "1")
    use_direct = False if use_direct == "1" else True

    if mla:
        block_lens = [64]
        block_layer = 61
        head_size = 576
        block_elem_size = 2
        kv = 1
        model_name = "deepseek-v3"
        num_head_list = [1]
        total_tp_size = 1
    else:
        block_lens = [128, 256]
        block_layer = 64
        head_size = 128
        block_elem_size = 2
        kv = 2
        model_name = "QwQ-32B"
        num_head_list = [1, 2, 4, 8]
        total_tp_size = 1

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(SCRIPT_DIR, "save_load_result.csv")
    need_header = not os.path.exists(csv_file)

    with open(csv_file, "a", newline="", encoding="utf-8") as csv_fp:
        writer = csv.writer(csv_fp)

        if need_header:
            writer.writerow(
                [
                    "Model",
                    "Sequence Length",
                    "Batch Size",
                    "Layers",
                    "Element Size",
                    "KV",
                    "Num Head",
                    "Block Size",
                    "Stream Number",
                    "IO Count",
                    "IO Size(B)",
                    "Total Size(GB)",
                    "Write Avg Time(s)",
                    "Write Avg Bandwidth(GB/s)",
                    "Read Avg Time(s)",
                    "Read Avg Bandwidth(GB/s)",
                ]
            )

        for num_head in num_head_list:
            for block_len in block_lens:
                for transfer_stream_number in transfer_stream_numbers:
                    block_dim = head_size * num_head
                    io_size = block_dim * block_len * block_elem_size

                    for num_tokens in num_tokens_list:
                        sep = "=" * 60
                        print(
                            f"\n{sep}\n= num_head={num_head} | num_tokens={num_tokens:>6} | Repeat {repeat} times =\n{sep}\n"
                        )

                        batch_size = int(num_tokens / block_len)
                        io_count = batch_size * block_layer

                        result_queue = multiprocessing.Queue()

                        process = multiprocessing.Process(
                            target=run_wrapper,
                            args=(
                                result_queue,
                                storage_backends,
                                device_id,
                                repeat,
                                num_head,
                                block_len,
                                num_tokens,
                                block_layer,
                                head_size,
                                block_elem_size,
                                kv,
                                mla,
                                ucm_connector_name,
                                total_tp_size,
                                model_path,
                                transfer_stream_number,
                                use_direct,
                            ),
                        )

                        process.start()
                        process.join()

                        status, result = result_queue.get()
                        if status == "error":
                            raise Exception(f"Error in subprocess: {result}")

                        (
                            avg_w_size,
                            avg_w_time,
                            avg_w_bw,
                            avg_r_time,
                            avg_r_bw,
                            avg_r_size,
                        ) = result

                        writer.writerow(
                            [
                                model_name,
                                num_tokens,
                                batch_size,
                                block_layer,
                                block_elem_size,
                                kv,
                                num_head,
                                block_len,
                                transfer_stream_number,
                                io_count,
                                io_size,
                                f"{avg_w_size:.4f}",
                                f"{avg_w_time:.4f}",
                                f"{avg_w_bw:.4f}",
                                f"{avg_r_time:.4f}",
                                f"{avg_r_bw:.4f}",
                            ]
                        )

                        csv_fp.flush()

    print("\n" + "=" * 60 + "\n= All combinations tested =\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
