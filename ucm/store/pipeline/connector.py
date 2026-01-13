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
import copy
from typing import Callable, Dict, List

import torch

from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1


class UcmPipelineStoreBuilder:
    _registry: Dict[
        str, Callable[[Dict[str, object], List[UcmKVStoreBaseV1]], None]
    ] = {}

    @classmethod
    def register(
        cls,
        name: str,
        builder: Callable[[Dict[str, object], List[UcmKVStoreBaseV1]], None],
    ) -> None:
        if name in cls._registry:
            raise ValueError(f"Builder '{name}' is already registered.")
        cls._registry[name] = builder

    @classmethod
    def get(
        cls, name: str
    ) -> Callable[[Dict[str, object], List[UcmKVStoreBaseV1]], None]:
        return cls._registry.get(name)


class UcmPipelineStore(UcmKVStoreBaseV1):
    def __init__(self, config: Dict[str, object]) -> None:
        super().__init__(config)
        self._stores: List[UcmKVStoreBaseV1] = []
        builder = UcmPipelineStoreBuilder.get(config["store_pipeline"])
        if builder is None:
            raise ValueError(f"unknown store pipeline: {config['store_pipeline']}")
        builder(config, self._stores)

    @property
    def _backend(self) -> UcmKVStoreBaseV1:
        return self._stores[-1]

    def cc_store(self) -> int:
        return self._backend.cc_store()

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        return self._backend.lookup(block_ids)

    def lookup_on_prefix(self, block_ids: List[bytes]) -> int:
        return self._backend.lookup_on_prefix(block_ids)

    def prefetch(self, block_ids: List[bytes]) -> None:
        return self._backend.prefetch(block_ids)

    def load(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_tensor: List[List[torch.Tensor]],
    ) -> Task:
        return self._backend.load(block_ids, shard_index, dst_tensor)

    def dump(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_tensor: List[List[torch.Tensor]],
    ) -> Task:
        return self._backend.dump(block_ids, shard_index, src_tensor)

    def load_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_addr: List[List[int]],
    ) -> Task:
        return self._backend.load_data(block_ids, shard_index, dst_addr)

    def dump_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_addr: List[List[int]],
    ) -> Task:
        return self._backend.dump_data(block_ids, shard_index, src_addr)

    def wait(self, task: Task) -> None:
        return self._backend.wait(task)

    def check(self, task: Task) -> bool:
        return self._backend.check(task)


def _cache_posix_pipeline_builder(
    config: Dict[str, object], store: List[UcmKVStoreBaseV1]
) -> None:
    from ucm.store.cache.connector import UcmCacheStore
    from ucm.store.posix.connector import UcmPosixStore

    posix_config = copy.deepcopy(config)
    if config.get("device_id", -1) >= 0:
        posix_config |= {"tensor_size": config["shard_size"]}
    posix_store = UcmPosixStore(posix_config)
    store.append(posix_store)
    cache_config = copy.deepcopy(config) | {"store_backend": posix_store.cc_store()}
    cache_store = UcmCacheStore(cache_config)
    store.append(cache_store)


def _cache_empty_pipeline_builder(
    config: Dict[str, object], store: List[UcmKVStoreBaseV1]
) -> None:
    from ucm.store.cache.connector import UcmCacheStore
    from ucm.store.empty.connector import UcmEmptyStore

    empty_config = copy.deepcopy(config)
    empty_store = UcmEmptyStore(empty_config)
    store.append(empty_store)
    cache_config = copy.deepcopy(config) | {"store_backend": empty_store.cc_store()}
    cache_store = UcmCacheStore(cache_config)
    store.append(cache_store)


UcmPipelineStoreBuilder.register("Cache|Posix", _cache_posix_pipeline_builder)
UcmPipelineStoreBuilder.register("Cache|Empty", _cache_empty_pipeline_builder)
