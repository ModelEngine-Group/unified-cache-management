from abc import ABC, abstractmethod
from typing import List, Dict

import torch


class Task:
    """
    Abstract Task for kv transfer
    """
    pass


class UcmKVStoreBase(ABC):
    """
    Storage vendor implements this interface to support KV Cache centric inference system.
    """

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def create(self, block_ids: List[str]) -> int:
        """
        create kv cache space in storafe

        Args:
            block_ids (List[str]): vLLM block hash.
        
        Returns:
            0 - success
            others - failed
        """
        pass

    @abstractmethod
    def lookup(self, block_ids: List[str]) -> List[bool]:
        """
        Get number of blocks that can be loaded from the
        external KV cache.

        Args:
            block_ids (List[str]): vLLM block hash.

        Returns:
            hit block mask, True -> hit
        """
        pass

    @abstractmethod
    def prefetch(self, block_ids: List[str]) -> None:
        """
        prefetch kv cache to high speed cache according to block_ids.

        Args:
            block_ids (List[str]): vLLM block hash.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def wait(self, task: Task) -> int:
        """
        wait kv cache kv transfer task finished.

        Args:
            task (Task): transfer engine task.
        Returns:
            0 - success
            others - failed.
        """
        pass

    @abstractmethod
    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        """
        commit kv cache, now kv cache can be reused.

        Args:
            block_ids (List[str]): vLLM block hash.
            is_success(bool): if False, we need release block
        """
        pass
