# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ucm.store.ds3fsstore import ucmds3fsstore
from ucm.store.ucmstore import Task, UcmKVStoreBase


@dataclass
class NfsTask(Task):
    task_id: int


class UcmDs3FsStore(UcmKVStoreBase):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.store = ucmds3fsstore.Ds3FsStore()
        storage_backends = [
            path for path in config["storage_backends"].split(":") if path
        ]
        block_size = int(config["kv_block_size"])
        transfer_enable = True if config["role"] == "worker" else False
        param = ucmds3fsstore.Ds3FsStore.Config(
            storage_backends, block_size, transfer_enable
        )
        if transfer_enable:
            param.transferDeviceId = config["device"]
            param.transferIoSize = config["io_size"]
            param.transferIoDirect = True
            param.transferStreamNumber = 8
            param.transferBufferNumber = 4096
        if "3fs_mount_point" in config:
            param.mountPoint = config["3fs_mount_point"]
        ret = self.store.Setup(param)
        if ret != 0:
            msg = f"Failed to initialize ucmds3fsstore, errcode: {ret}."
            raise RuntimeError(msg)

    def cc_store(self) -> int:
        return self.store.CCStoreImpl()

    def create(self, block_ids: List[str]) -> List[int]:
        return self.store.AllocBatch(block_ids)

    def lookup(self, block_ids: List[str]) -> List[bool]:
        return self.store.LookupBatch(block_ids)

    def prefetch(self, block_ids: List[str]) -> None:
        pass

    def load(
        self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
    ) -> Task:
        dst_tensor_ptr = [t.data_ptr() for t in dst_tensor]
        task_id = self.store.LoadToDevice(block_ids, dst_tensor_ptr)
        return NfsTask(task_id=task_id)

    def dump(
        self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]
    ) -> Task:
        src_tensor_ptr = [t.data_ptr() for t in src_tensor]
        task_id = self.store.DumpFromDevice(block_ids, src_tensor_ptr)
        return NfsTask(task_id=task_id)

    def fetch_data(
        self,
        block_ids: List[str],
        offset: List[int],
        dst_addr: List[int],
        size: List[int],
    ) -> Task:
        pass

    def dump_data(
        self,
        block_ids: List[str],
        offset: List[int],
        src_addr: List[int],
        size: List[int],
    ) -> Task:
        pass

    def wait(self, task: Task) -> int:
        return self.store.Wait(task.task_id)

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        self.store.CommitBatch(block_ids, is_success)

    def check(self, task: Task) -> Tuple[int, bool]:
        return self.store.Check(task.task_id)
