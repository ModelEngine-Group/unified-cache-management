import asyncio
import json
import os
import threading
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass
from typing import Dict, List

import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from ucm.logger import init_logger
from ucm.store import Task, UcmKVStoreBase

TIMEOUT_S_THR: int = 60 * 60
DEFAULT_GLOBAL_SEGMENT_SIZE: int = 3355443200  # 3.125 GiB
DEFAULT_LOCAL_BUFFER_SIZE: int = 1073741824  # 1.0 GiB

logger = init_logger(__name__)


# TODO To keep it consistent with the vllm source code(vllm/distributed/kv_transfer/kv_lookup_buffer/mooncake_store.py), the source code is fully reused here. The code here will be deleted after vllm is implemented.
@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    @staticmethod
    def load_from_dict(config: Dict = {}) -> "MooncakeStoreConfig":
        """Load the config from dict."""
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get(
                "global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE
            ),
            local_buffer_size=config.get(
                "local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE
            ),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
        )


@dataclass
class MooncakeTask(Task):
    """A task class for Mooncake operations with a task identifier."""

    task_id: int = -1


class UcmMooncakeStore(UcmKVStoreBase):
    """
    A wrapper class for MooncakeDistributedStore that implements the UcmKVStoreBase interface.
    Provides key-value store functionality for vLLM using Mooncake as the backend.
    """

    def __init__(self, config: Dict = {}):
        """Initialize the Mooncake store with configuration."""
        super().__init__(config)
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        try:
            self.store = MooncakeDistributedStore()

            mooncake_config = MooncakeStoreConfig.load_from_dict(config)
            logger.info("Mooncake Configuration loaded from dict successfully.")

            self.store.setup(
                mooncake_config.local_hostname,
                mooncake_config.metadata_server,
                mooncake_config.global_segment_size,
                mooncake_config.local_buffer_size,
                mooncake_config.protocol,
                mooncake_config.device_name,
                mooncake_config.master_server_address,
            )

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except TypeError:
            logger.warning("Lack of configuration, please check the dict params .")

        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        # Task management variables
        self.task_id: int = 0
        self.tasks: Dict[int, Future] = {}

        # Threading and synchronization variables
        self.loop = asyncio.new_event_loop()
        self.lock = threading.Lock()
        self._shutting_down = threading.Event()

        # Start the event loop thread
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def __del__(self):
        """Release resources on garbage collection."""
        try:
            self.shutdown()
        except Exception:
            pass

    def _run_event_loop(self):
        """Run the asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def create(self, block_ids: List[str]) -> List[int]:
        """
        create kv cache space in storafe (not implemented for Mooncake).

        Args:
            block_ids (List[str]): vLLM block hash.
        Returns:
            Always returns 0 as this operation is not supported by Mooncake
        """
        # Mooncake only has get and put interfaces, this operation is not supported
        return [0] * len(block_ids)

    def lookup(self, block_ids: List[str]) -> List[bool]:
        """
        Get number of blocks that can be loaded from the
        external KV cache.
        Mooncake integration uses hash = block_id + offset (default offset=0 if not provided).
        Args:
            block_ids (List[str]): vLLM block hash.

        Returns:
            hit block mask, True -> hit
        """
        if self._shutting_down.is_set():
            raise RuntimeError("UcmMooncakeStore is shutting down.")

        mask = [
            True if self.store.is_exist(f"{block_key}_0") == 1 else False
            for block_key in block_ids
        ]
        return mask

    def prefetch(self, block_ids: List[str]) -> None:
        """
        prefetch kv cache to high speed cache according to block_ids (not implemented for Mooncake).

        Args:
            block_ids (List[str]): vLLM block hash.
        """
        # Mooncake only has get and put interfaces, this operation is not supported
        pass

    def load(
        self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
    ) -> Task:
        """
        load kv cache to device.
        Mooncake integration uses hash = block_id + offset (default offset=0 if not provided).

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            dst_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        if self._shutting_down.is_set():
            raise RuntimeError("UcmMooncakeStore is shutting down.")

        coro = self._load_impl(block_ids, offset, dst_tensor)
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        with self.lock:
            self.task_id += 1
            self.tasks[self.task_id] = future
            return MooncakeTask(task_id=self.task_id)

    async def _load_impl(
        self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
    ) -> int:
        """Internal implementation of loading KV cache from Mooncake Store."""
        assert len(block_ids) == len(
            dst_tensor
        ), "block_ids and dst_tensor have different lengths, please check!"
        for i in range(len(block_ids)):
            try:
                block_hash = f"{block_ids[i]}_{offset[i]}"
                data = self.store.get(block_hash)
            except TypeError as err:
                logger.error("Failed to get value from Mooncake Store: %s", err)
                raise TypeError("Mooncake Store Get Type Error.") from err

            if data:
                loaded_tensors = safetensors_load(data)
                tensor_cpu = loaded_tensors["tensor"]
                assert dst_tensor[i].shape == tensor_cpu.shape
                assert dst_tensor[i].dtype == tensor_cpu.dtype
                dst_tensor[i].copy_(tensor_cpu)
            else:
                return 1
        return 0

    def dump(
        self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]
    ) -> Task:
        """
        dump kv cache to device.
        Mooncake integration uses hash = block_id + offset (default offset=0 if not provided).

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            src_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        if self._shutting_down.is_set():
            raise RuntimeError("UcmMooncakeStore is shutting down.")

        coro = self._dump_impl(block_ids, offset, src_tensor)
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        with self.lock:
            self.task_id += 1
            self.tasks[self.task_id] = future
            return MooncakeTask(task_id=self.task_id)

    async def _dump_impl(
        self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]
    ) -> int:
        """Internal implementation of dumping KV cache to Mooncake Store."""
        assert len(block_ids) == len(
            src_tensor
        ), "block_ids and src_tensor have different lengths, please check!"
        for i in range(len(block_ids)):
            value_bytes = safetensors_save({"tensor": src_tensor[i]})
            try:
                block_hash = f"{block_ids[i]}_{offset[i]}"
                ret = self.store.put(block_hash, value_bytes)
                if ret != 0:
                    return ret
            except TypeError as err:
                logger.error("Failed to put value into Mooncake Store: %s", err)
                raise TypeError("Mooncake Store Put Type Error.") from err
        return 0

    def wait(self, task: Task) -> int:
        """
        wait kv cache kv transfer task finished.

        Args:
            task (Task): transfer engine task.
        Returns:
            0 - success
            others - failed.
        """
        # Safely retrieve the Future object
        with self.lock:
            future = self.tasks.pop(task.task_id, None)

        if future is None:
            logger.error(f"Invalid task ID: {task.task_id}")
            return 1

        try:
            ret = future.result(TIMEOUT_S_THR)
            return ret
        except TimeoutError:
            # Cancel the task if it times out
            future.cancel()
            logger.error(f"Task {task.task_id} timed out after {TIMEOUT_S_THR}s")
            return 1
        except asyncio.CancelledError:
            logger.error(f"Task {task.task_id} was cancelled")
            return 1
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            return 1

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        """
        commit kv cache, now kv cache can be reused (not implemented for Mooncake).

        Args:
            block_ids (List[str]): vLLM block hash.
            is_success(bool): if False, we need release block
        """
        # Mooncake only has get and put interfaces, this operation is not supported
        pass

    def shutdown(self):
        """Safely shutdown all components of the store."""
        if self._shutting_down.is_set():
            return

        self._shutting_down.set()

        # Safely cancel all pending tasks (atomic operation)
        with self.lock:
            tasks_to_cancel = list(self.tasks.values())
            self.tasks.clear()

        for future in tasks_to_cancel:
            if not future.done():
                future.cancel()

        # Stop the event loop
        self.loop.call_soon_threadsafe(self.loop.stop)

        # Wait for thread termination
        if self.thread.is_alive():
            self.thread.join(TIMEOUT_S_THR)

            # Force close the loop if thread didn't exit
            if not self.loop.is_closed():
                self.loop.close()

        self.store.close()

    def check(self, task: Task) -> bool:
        """
        check if kv transfer task finished.

        Args:
            task (Task): transfer engine task.
        Returns:
            0 - finished
            others - in process.
        """
        pass
