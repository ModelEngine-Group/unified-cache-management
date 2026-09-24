# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#         https://www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AI IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Utilities for tool-call aware KV Cache pruning.

The API side should keep sending the *original* prompt tokens to VLLM's prefix
hashing and physical KV cache allocation continue to use the stable, complete
context. These helpers only derive per-request logical KV block pruning metadata
that the Ascend attention metadata builder can apply immediately before invoking
paged-attention kernels.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, sequence
from typing import Any, Optional
import sys
import subprocess
import pickle
from typing import List, tuple
import glob 

import torch


def compact_block_table_for_pruning(
    block_table: torch.Tensor, seq_lens: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """原地压缩 block_table，并更新 seq_lens，返回修改后的原张量及移除的零块数。

    所有操作均在 GPU 上完成，完全兼容 CUDA Graph / ACL Graph（全程固定 Shape 无分支）。

    Args:
        block_table: [num_rows, max_blocks]，每行是请求的块列表，0 表示空洞或未使用。
        seq_lens: [num_rows]，每个请求的实际 token 长度。
        block_size: 正整数，每个块的 token 数。
    
    Returns:
        block_table: 原地修改后的压缩表
        seq_lens: 原地修改后的长度
        removed_block_num: [num_rows]，每个请求在有效范围内移除的零块数量。
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    
    num_rows, max_blocks = block_table.shape 
    device = block_table.device
    org_seq_device =seq_lens.to(device)

    # 1. 统一设备进行计算（跨设备拷贝禁用 non_blocking ，避免异步读到旧值）
    seq_lens_calc = seq_lens.to(device=device)

    # 2. 计算有效块数及列掩码
    used_blocks = torch.clamp(
        (seq_lens_calc + block_size - 1) // block_size, max=max_blocks
    )
    col_idx = torch.arange(max_blocks, device=device).expand(num_rows, -1)
    used_mask = col_idx < used_blocks.unsqueeze(1)

    nonzero = block_table != 0
    keep = used_mask & nonzero #需要保留的有效非零块

    # 3. 计算移除的零块数及更新后的 seq_lens
    removed_block_num = (used_mask & ~nonzero).sum(dim=1)
    removed_tokens = removed_block_num * block_size

    new_seq_lens = torch.where(
        seq_lens_calc == 0,
        seq_lens_calc,
        torch.clamp(seq_lens_calc - removed_tokens, min=1),
    )

    # 4. 构造单行无冲突的目标索引矩阵 (Permutation Matrix)
    keep_int = keep.to(torch.int64)

    # 保留块的目标列索引: [0, 1, ..., K-1]
    keep_target_cols = torch.clamp(
        torch.consum(keep_int, dim=1) -1, min=0, max=max_blocks - 1
    )

    # 废弃块的目标列索引：[K, K+1, ..., N-1](接在保留块后面，绝对不冲突)
    kept_count = keep_int.sum(dim=1, keepdim=True)
    discard_int = (~keep).to(torch.int64)
    discard_target_cols = torch.clamp(
        kept_count + torch.cumsum(discard_int, dim=1)-1,
        min = 0,
        max = max_blocks -1,
    )

    # 组合为完美的无重叠索引
    target_cols = torch.where(keep, keep_target_cols, discard_target_cols)

    # 构造源数据：保留块保持原值，废弃块置零
    scr_data = torch.where(keep, block_table, torch.zero_like(block_table))

    # 5. 执行压缩并写回原始block_table (真正实现原地修改)
    compacted_table = torch.zero_like(block_size)
    compacted_table.scatter_(1, target_cols, scr_data)
    block_table.copy_(compacted_table)

    # 6. 写回原始seq_lens(跨设备拷贝禁用 non_blocking, 保证调用方读取到最新值)
    seq_lens.copy_(new_seq_lens.to(org_seq_device))

    return block_table, seq_lens
