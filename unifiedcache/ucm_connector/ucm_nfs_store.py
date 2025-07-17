import torch
import ucmnfsstore
from dataclasses import dataclass
from typing import List, Dict
from unifiedcache.logger import init_logger
from unifiedcache.ucm_connector import Task, UcmKVStoreBase


logger = init_logger(__name__)


@dataclass
class NfsTask(Task):
    task_id: int

    def get_id(self) -> int:
        return self.task_id


class UcmNfsStore(UcmKVStoreBase):
    """
    Nfs connector
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        param = ucmnfsstore.SetupParam(config["storage_backends"],
                                             config["block_size"],
                                             config["transformer_enable"])
        if param.transferEnable:
            param.transferDeviceId = config["transfer_device_id"]
            param.transferStreamNumber = config["transfer_stream_number"]
        
        ret = ucmnfsstore.Setup(param)
        if ret != 0:
            msg = f"Failed to initialize ucmnfsstore, errcode: {ret}."
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.info("Succeed in initializing ucmnfsstore.")

    def create(self, block_ids: List[str]) -> int:
        """
        create kv cache space in storage

        Args:
            block_ids (List[str]): vLLM block hash.
        Returns:
            success mask
        """
        ret = ucmnfsstore.Alloc(block_ids)
        if ret != 0:
            logger.error(f"Failed to allocate kv cache space, errcode: {ret}.")
        else:
            logger.info("Succeed in allocating kv cache space.")
        return ret

    def lookup(self, block_ids: List[str]) -> List[bool]:
        """
        Get number of blocks that can be loaded from the
        external KV cache.

        Args:
            block_ids (List[str]): vLLM block hash.

        Returns:
            hit block mask, True -> hit
        """
        ret = ucmnfsstore.Lookup(block_ids)
        logger.info("Succeed in looking up kv cache blocks.")
        return ret

    def prefetch(self, block_ids: List[str]) -> None:
        """
        prefetch kv cache to high speed cache according to block_ids.

        Args:
            block_ids (List[str]): vLLM block hash.
        """
        # TODO
        logger.info("prefetch finished.")

    def load(self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]) -> Task:
        """
        load kv cache to device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            dst_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        dst_tensor_size = [t.numel() * t.element_size() for t in dst_tensor]
        device_type = dst_tensor[0].device.type

        if device_type == "cpu":
            task_id = ucmnfsstore.LoadToHost(block_ids, offset, dst_tensor, dst_tensor_size)
        elif device_type == "cuda" or device_type == "npu":
            task_id = ucmnfsstore.LoadToDevice(block_ids, offset, dst_tensor, dst_tensor_size)
        logger.info(f"Succeed in loading kv cache to {device_type}, task id: {task_id}.")

        return Task(task_id=id)

    def dump(self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]) -> Task:
        """
        dump kv cache to device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            src_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        src_tensor_size = [t.numel() * t.element_size() for t in src_tensor]
        device_type = src_tensor[0].device.type

        if device_type == "cpu":
            task_id = ucmnfsstore.DumpFromHost(block_ids, offset, src_tensor, src_tensor_size)
        elif device_type == "cuda" or device_type == "npu":
            task_id = ucmnfsstore.DumpFromDevice(block_ids, offset, src_tensor, src_tensor_size)
        logger.info(f"Succeed in dumping kv cache from {device_type}, task id: {task_id}.")
        
        return Task(task_id=id)

    def wait(self, task: Task) -> int:
        """
        wait kv cache kv transfer task finished.

        Args:
            task (Task): transfer engine task.
        Returns:
            0 - success
            others - failed.
        """
        ret = ucmnfsstore.Wait(task.get_id())
        if ret != 0:
            logger.error(f"Failed to wait for kv cache transfer task, errcode: {ret}.")
        else:
            logger.info("Succeed in waiting for kv cache transfer task.")
        return ret

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        """
        commit kv cache, now kv cache can be reused.

        Args:
            block_ids (List[str]): vLLM block hash.
            is_success(bool): if False, we need release block
        """
        ucmnfsstore.Commit(block_ids, is_success)
        logger.info("Succeed in committing kv cache.")
