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
import array
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch

from ucm.store.pipeline import ucmpipelinestore
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1


class UcmPipelineStoreBuilder:
    registry_: Dict[
        str, Callable[[Dict[str, object], ucmpipelinestore.PipelineStore], None]
    ] = {}

    @classmethod
    def register(
        cls,
        name: str,
        builder: Callable[[Dict[str, object], ucmpipelinestore.PipelineStore], None],
    ) -> None:
        if name in cls.registry_:
            raise ValueError(f"Builder '{name}' is already registered.")
        cls.registry_[name] = builder

    @classmethod
    def get(
        cls, name: str
    ) -> Callable[[Dict[str, object], ucmpipelinestore.PipelineStore], None]:
        return cls.registry_.get(name)


@dataclass
class UcmPipelineStoreTransTask(Task):
    task_id: int


class UcmPipelineStore(UcmKVStoreBaseV1):
    def __init__(self, config: Dict[str, object]) -> None:
        super().__init__(config)
        self.store_ = ucmpipelinestore.PipelineStore()
        builder = UcmPipelineStoreBuilder.get(config["store_pipeline"])
        if builder is None:
            raise ValueError(f"unknown store pipeline: {config['store_pipeline']}")
        builder(config, self.store_)

    def cc_store(self) -> int:
        return self.store_.Self()

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        flat = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        res = self.store_.Lookup(flat)
        return np.frombuffer(res, dtype=bool)

    def lookup_on_prefix(self, block_ids: List[bytes]) -> int:
        flat = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        return self.store_.LookupOnPrefix(flat)

    def prefetch(self, block_ids: List[bytes]) -> None:
        flat = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        self.store_.Prefetch(flat)

    def _tensor_normalize(self, tensors: List[List[torch.Tensor]]) -> np.ndarray:
        n_rows = len(tensors)
        n_cols = len(tensors[0])
        flat = np.fromiter(
            (t for row in tensors for t in row), dtype=object, count=n_rows * n_cols
        )
        ptrs = np.vectorize(torch.Tensor.data_ptr, otypes=[np.uint64])(flat)
        return ptrs.reshape(n_rows, n_cols)

    def load(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_tensor: List[List[torch.Tensor]],
    ) -> Task:
        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = array.array("Q", shard_index)
        addrs = self._tensor_normalize(dst_tensor)
        task_id = self.store_.Load(ids, indexes, addrs)
        return UcmPipelineStoreTransTask(task_id)

    def dump(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_tensor: List[List[torch.Tensor]],
    ) -> Task:
        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = array.array("Q", shard_index)
        addrs = self._tensor_normalize(src_tensor)
        task_id = self.store_.Dump(ids, indexes, addrs)
        return UcmPipelineStoreTransTask(task_id)

    def load_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_addr: List[List[int]] | np.ndarray,
    ) -> Task:
        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = array.array("Q", shard_index)
        if isinstance(dst_addr, np.ndarray):
            addrs = dst_addr
        else:
            addrs = np.array(dst_addr, dtype=np.uint64)
        task_id = self.store_.Load(ids, indexes, addrs)
        return UcmPipelineStoreTransTask(task_id)

    def dump_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_addr: List[List[int]] | np.ndarray,
    ) -> Task:
        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = array.array("Q", shard_index)
        if isinstance(src_addr, np.ndarray):
            addrs = src_addr
        else:
            addrs = np.array(src_addr, dtype=np.uint64)
        task_id = self.store_.Dump(ids, indexes, addrs)
        return UcmPipelineStoreTransTask(task_id)

    def wait(self, task: Task) -> None:
        return self.store_.Wait(task.task_id)

    def check(self, task: Task) -> bool:
        return self.store_.Check(task.task_id)


def _cache_ds3fs_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    ds3fs_config = copy.deepcopy(config)
    if config.get("device_id", -1) >= 0:
        ds3fs_config |= {"tensor_size": config["shard_size"]}
    pipeline.Stack("Ds3fs", str(store_dir / "ds3fs/libds3fsstore.so"), ds3fs_config)
    pipeline.Stack("Cache", str(store_dir / "cache/libcachestore.so"), config)


def _cache_empty_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    pipeline.Stack("Empty", str(store_dir / "empty/libemptystore.so"), config)
    pipeline.Stack("Cache", str(store_dir / "cache/libcachestore.so"), config)


def _cache_posix_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    posix_config = copy.deepcopy(config)
    if config.get("device_id", -1) >= 0:
        posix_config |= {"tensor_size": config["shard_size"]}
    pipeline.Stack("Posix", str(store_dir / "posix/libposixstore.so"), posix_config)
    pipeline.Stack("Cache", str(store_dir / "cache/libcachestore.so"), config)


def _empty_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    pipeline.Stack("Empty", str(store_dir / "empty/libemptystore.so"), config)


def _fake_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    pipeline.Stack("Fake", str(store_dir / "fake/libfakestore.so"), config)


def _posix_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    pipeline.Stack("Posix", str(store_dir / "posix/libposixstore.so"), config)


UcmPipelineStoreBuilder.register("Cache|Ds3fs", _cache_ds3fs_pipeline_builder)
UcmPipelineStoreBuilder.register("Cache|Empty", _cache_empty_pipeline_builder)
UcmPipelineStoreBuilder.register("Cache|Posix", _cache_posix_pipeline_builder)
UcmPipelineStoreBuilder.register("Empty", _empty_pipeline_builder)
UcmPipelineStoreBuilder.register("Fake", _fake_pipeline_builder)
UcmPipelineStoreBuilder.register("Posix", _posix_pipeline_builder)
