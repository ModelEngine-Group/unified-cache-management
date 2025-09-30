"""
The MIT License

Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from ucm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class KvCompConfig:
    """
    Dataclass representing the configuration for KvComp.
    """

    model_name: str = "DummyModel"
    is_mla: bool = False

    # either "random" or "fixed"
    hash_weight_type: str = "random"

    num_hidden_layers: int = 36

    # the minimal seq_len to trigger KvComp
    seq_len_threshhold: int = 2048

    # any value divisible by 128
    chunk_size: int = 128

    # either "max", "min" or "sum"
    chunk_repre_method: str = "max"

    head_dim: int = 128
    hash_bits: int = 128

    top_k_ratio_per_layer: List[float] = field(default_factory=lambda: [0.3] * 36)
    top_k_index_reuse: List[int] = field(default_factory=lambda: [-1] * 36)

    # nonnegative means slicing from the start, negative means slicing from the end
    must_select_blocks: List[int] = field(default_factory=lambda: [0, -2, -1])

    # used when is_mla=True and hash_weight_type="fixed"
    hash_weight: Optional[List[List[float]]] = None

    # Conditional fields if is_mla=True
    kv_lora_rank: Optional[int] = None  # we need to specify it if is_mla=True
    qk_rope_head_dim: Optional[int] = None  # we need to specify it if is_mla=True
    hash_bits_kv_lora: Optional[int] = None  # we need to specify it if is_mla=True
    hash_bits_qk_rope: Optional[int] = None  # we need to specify it if is_mla=True
    hash_weight_kv_lora: Optional[List[List[float]]] = (
        None  # used when is_mla=True and hash_weight_type="fixed"
    )
    hash_weight_qk_rope: Optional[List[List[float]]] = (
        None  # used when is_mla=True and hash_weight_type="fixed"
    )

    # generate non-MLA config data
    def generate_config_data(
        self,
        model_name: str,
        hash_weight_type: str,
        num_hidden_layers: int,
        seq_len_threshhold: int,
        chunk_size: int,
        chunk_repre_method: str,
        head_dim: int,
        hash_bits: int,
        top_k_ratio_per_layer: List[float],
        top_k_index_reuse: List[int],
        must_select_blocks: List[int],
    ) -> None:
        self.is_mla = False
        self.model_name = model_name

        if hash_weight_type not in ["random", "fixed"]:
            raise ValueError(
                f"hash_weight_type should be either 'random' or 'fixed', but got {hash_weight_type}"
            )
        self.hash_weight_type = hash_weight_type

        self.num_hidden_layers = num_hidden_layers
        self.seq_len_threshhold = seq_len_threshhold

        if chunk_size % 128 != 0:
            raise ValueError(
                f"chunk_size should be divisible by 128, but got {chunk_size}"
            )
        self.chunk_size = chunk_size

        if chunk_repre_method not in ["max", "min", "sum"]:
            raise ValueError(
                f"chunk_repre_method should be either 'max', 'min' or 'sum', but got {chunk_repre_method}"
            )
        self.chunk_repre_method = chunk_repre_method

        self.head_dim = head_dim
        self.hash_bits = hash_bits

        if len(top_k_ratio_per_layer) != num_hidden_layers:
            raise ValueError(
                f"top_k_ratio_per_layer length should be equal to num_hidden_layers={num_hidden_layers}, but got {len(top_k_ratio_per_layer)}"
            )
        self.top_k_ratio_per_layer = top_k_ratio_per_layer
        if len(top_k_index_reuse) != num_hidden_layers:
            raise ValueError(
                f"top_k_index_reuse length should be equal to num_hidden_layers={num_hidden_layers}, but got {len(top_k_index_reuse)}"
            )
        self.top_k_index_reuse = top_k_index_reuse

        self.must_select_blocks = must_select_blocks

        if hash_weight_type == "random":
            logger.info(
                "hash_weight_type is 'random', hash_weight will be generated automatically."
            )
            self.hash_weight = None
        else:
            logger.warning(
                "hash_weight_type is 'fixed', please manually set hash_weight in the config json file."
            )

    # generate MLA config data
    def generate_mla_config_data(
        self,
        model_name: str,
        hash_weight_type: str,
        num_hidden_layers: int,
        seq_len_threshhold: int,
        chunk_size: int,
        chunk_repre_method: str,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        hash_bits_kv_lora: int,
        hash_bits_qk_rope: int,
        top_k_ratio_per_layer: List[float],
        top_k_index_reuse: List[int],
        must_select_blocks: List[int],
    ) -> None:
        self.is_mla = True
        self.model_name = model_name
        if hash_weight_type not in ["random", "fixed"]:
            raise ValueError(
                f"hash_weight_type should be either 'random' or 'fixed', but got {hash_weight_type}"
            )
        self.hash_weight_type = hash_weight_type

        self.num_hidden_layers = num_hidden_layers
        self.seq_len_threshhold = seq_len_threshhold
        if chunk_size % 128 != 0:
            raise ValueError(
                f"chunk_size should be divisible by 128, but got {chunk_size}"
            )
        self.chunk_size = chunk_size
        if chunk_repre_method not in ["max", "min", "sum"]:
            raise ValueError(
                f"chunk_repre_method should be either 'max', 'min' or 'sum', but got {chunk_repre_method}"
            )
        self.chunk_repre_method = chunk_repre_method
        self.head_dim = qk_rope_head_dim + kv_lora_rank
        self.hash_bits = hash_bits_qk_rope + hash_bits_kv_lora
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.hash_bits_kv_lora = hash_bits_kv_lora
        self.hash_bits_qk_rope = hash_bits_qk_rope

        if len(top_k_ratio_per_layer) != num_hidden_layers:
            raise ValueError(
                f"top_k_ratio_per_layer length should be equal to num_hidden_layers={num_hidden_layers}, but got {len(top_k_ratio_per_layer)}"
            )
        self.top_k_ratio_per_layer = top_k_ratio_per_layer
        if len(top_k_index_reuse) != num_hidden_layers:
            raise ValueError(
                f"top_k_index_reuse length should be equal to num_hidden_layers={num_hidden_layers}, but got {len(top_k_index_reuse)}"
            )
        self.top_k_index_reuse = top_k_index_reuse

        self.must_select_blocks = must_select_blocks

        if hash_weight_type == "random":
            logger.info(
                "hash_weight_type is 'random', hash_weight_kv_lora and hash_weight_qk_rope will be generated automatically."
            )
            self.hash_weight = None
            self.hash_weight_kv_lora = None
            self.hash_weight_qk_rope = None
        else:
            logger.warning(
                "hash_weight_type is 'fixed', please manually set hash_weight_kv_lora and hash_weight_qk_rope in the config json file."
            )

    # set hash_weight when hash_weight_type is "fixed" for non-MLA models
    def set_hash_weight(self, hash_weight: List[List[float]]) -> None:
        if self.hash_weight_type != "fixed":
            raise ValueError(
                "hash_weight can only be set when hash_weight_type is 'fixed'"
            )

        if len(hash_weight) != self.head_dim or len(hash_weight[0]) != self.hash_bits:
            raise ValueError(
                f"hash_weight shape should be ({self.head_dim}, {self.hash_bits}), but got ({len(hash_weight)}, {len(hash_weight[0])})"
            )

        self.hash_weight = hash_weight

    # set hash_weight when hash_weight_type is "fixed" for MLA models
    def set_mla_hash_weight(
        self,
        hash_weight_kv_lora: List[List[float]],
        hash_weight_qk_rope: List[List[float]],
    ) -> None:
        if self.hash_weight_type != "fixed":
            raise ValueError(
                "hash_weight can only be set when hash_weight_type is 'fixed'"
            )

        if (
            len(hash_weight_kv_lora) != self.kv_lora_rank
            or len(hash_weight_kv_lora[0]) != self.hash_bits_kv_lora
        ):
            raise ValueError(
                f"hash_weight_kv_lora shape should be ({self.kv_lora_rank}, {self.hash_bits_kv_lora}), but got ({len(hash_weight_kv_lora)}, {len(hash_weight_kv_lora[0])})"
            )

        if (
            len(hash_weight_qk_rope) != self.qk_rope_head_dim
            or len(hash_weight_qk_rope[0]) != self.hash_bits_qk_rope
        ):
            raise ValueError(
                f"hash_weight_qk_rope shape should be ({self.qk_rope_head_dim}, {self.hash_bits_qk_rope}), but got ({len(hash_weight_qk_rope)}, {len(hash_weight_qk_rope[0])})"
            )

        self.hash_weight_kv_lora = hash_weight_kv_lora
        self.hash_weight_qk_rope = hash_weight_qk_rope

    def to_json(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, file_path: str) -> "KvCompConfig":
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
