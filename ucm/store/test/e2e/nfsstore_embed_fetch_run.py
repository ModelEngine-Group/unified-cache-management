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
import csv
import multiprocessing
import os
from typing import List

from nfsstore_embed_fetch import run


def run_wrapper(result_queue, *args):
    try:
        result = run(*args)
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", str(e)))


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
    device_id = 1
    repeat = 3
    num_tokens_list = [2048, 4096, 8192, 16384, 32768]
    transferStreamNumbers = [32, 64, 128]

    print("1. Model Selection:")
    print("   1 - QwQ-32B")
    print("   2 - deepseek-v3")
    model_choice = get_user_input("Please select model", "1")
    mla = True if model_choice == "2" else False

    print("\n2. GDS Transfer:")
    print("   1 - Disable IoDirect (default)")
    print("   2 - Enable IoDirect")
    transferIoDirect = get_user_input("Please select Direct IO mode", "1")
    transferIoDirect = False if transferIoDirect == "1" else True

    print("\n3. Operation Mode:")
    print("   1 - Read/Write Test (default)")
    print("   2 - Write Only Test")
    print("   3 - Read Only Test")
    op_choice = get_user_input("Please select operation mode", "1")
    operation_mode_map = {"1": "both", "2": "write_only", "3": "read_only"}
    operation_mode = operation_mode_map.get(op_choice, "both")

    if mla:
        block_lens = [64, 128]
        block_layer = 61
        head_size = 576
        block_elem_size = 2
        kv = 1
        model_name = "deepseek-v3"
        num_head_list = [1]
    else:
        block_lens = [128, 256]
        block_layer = 64
        head_size = 128
        block_elem_size = 2
        kv = 2
        model_name = "QwQ-32B"
        num_head_list = [1, 2, 4, 8]

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(SCRIPT_DIR, "embed_fetch_result.csv")
    need_header = not os.path.exists(csv_file)

    os.makedirs(SCRIPT_DIR, exist_ok=True)

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
                for transferStreamNumber in transferStreamNumbers:
                    block_dim = head_size * num_head
                    io_size = block_dim * block_len * block_elem_size

                    for num_tokens in num_tokens_list:
                        sep = "=" * 60
                        print(
                            f"\n{sep}\n= num_head={num_head} | num_tokens={num_tokens:>6} | Repeat {repeat} times =\n{sep}\n"
                        )

                        batch_size = int(num_tokens / block_len)
                        io_num = int(num_tokens / block_len * block_layer)

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
                                transferStreamNumber,
                                num_tokens,
                                block_layer,
                                head_size,
                                block_elem_size,
                                kv,
                                mla,
                                transferIoDirect,
                                operation_mode,
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
                                transferStreamNumber,
                                io_num,
                                io_size,
                                f"{avg_w_size:.4f}",
                                f"{avg_w_time:.4f}",
                                f"{avg_w_bw:.4f}",
                                f"{avg_r_time:.4f}",
                                f"{avg_r_bw:.4f}",
                            ]
                        )
                        csv_fp.flush()

                        print(
                            f"WRITE COMPLETE for num_head={num_head}, num_tokens={num_tokens}"
                        )

    print("\n" + "=" * 60 + "\n= All combinations tested =\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    os.environ["UC_LOGGER_LEVEL"] = "debug"
    main()
