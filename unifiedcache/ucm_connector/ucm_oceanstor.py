from dataclasses import dataclass
from typing import Dict, List

import torch

from unifiedcache.logger import init_logger
from unifiedcache.ucm_connector import Task, UcmKVStoreBase

logger = init_logger(__name__)


@dataclass
class OceanTask(Task):
    task_id: str = "1"


class UcmOceanStore(UcmKVStoreBase):
    """
    A800/A600 connector
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        logger.info("init UcmOceanStore, blk_size: %d", config["block_size"])

    def create(self, block_ids: List[str]) -> int:
        """
        create kv cache space in storafe

        Args:
            block_ids (List[str]): vLLM block hash.
        Returns:
            success mask
        """
        # TODO
        logger.info("create finished.")
        return 0

    def lookup(self, block_ids: List[str]) -> List[bool]:
        """
        Get number of blocks that can be loaded from the
        external KV cache.

        Args:
            block_ids (List[str]): vLLM block hash.

        Returns:
            hit block mask, True -> hit
        """
        # TODO
        logger.info("lookup finished.")
        return []

    def prefetch(self, block_ids: List[str]) -> None:
        """
        prefetch kv cache to high speed cache according to block_ids.

        Args:
            block_ids (List[str]): vLLM block hash.
        """
        # TODO
        logger.info("prefetch finished.")

    def load(
        self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
    ) -> Task:
        """
        load kv cache to device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            dst_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        # TODO
        logger.info("load finished.")
        return Task()

    def dump(
        self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]
    ) -> Task:
        """
        dump kv cache to device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            src_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        # TODO
        logger.info("dump finished.")
        return Task()

    def wait(self, task: Task) -> int:
        """
        wait kv cache kv transfer task finished.

        Args:
            task (Task): transfer engine task.
        Returns:
            0 - success
            others - failed.
        """
        # TODO
        logger.info("wait finished.")
        return True

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        """
        commit kv cache, now kv cache can be reused.

        Args:
            block_ids (List[str]): vLLM block hash.
            is_success(bool): if False, we need release block
        """
        # TODO
        logger.info("create finished.")
