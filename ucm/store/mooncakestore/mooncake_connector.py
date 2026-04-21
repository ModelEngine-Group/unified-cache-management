from __future__ import annotations

import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from vllm.utils.network_utils import get_ip

from ucm.logger import init_logger
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

logger = init_logger(__name__)


DEFAULT_METADATA_SERVER = "P2PHANDSHAKE"
DEFAULT_MASTER_SERVER_ADDRESS = "127.0.0.1:50088"
DEFAULT_PROTOCOL = "ascend"
DEFAULT_SCHEDULER_PROTOCOL = "rpc_only"
DEFAULT_DEVICE_NAME = ""
DEFAULT_GLOBAL_SEGMENT_SIZE = 1073741824 * 5
DEFAULT_LOOKUP_SHARD_INDICES = 0
DEFAULT_MOONCAKE_V1_WORKERS = 4


@dataclass
class MooncakeTaskV1(Task):
    future: Future
    operation: str
    key_count: int


class _GlobalTransferEngine:
    """Process-level singleton wrapper for ``mooncake.engine.TransferEngine``.

    Aligns with the reference implementation in
    ``vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py``:
    every Mooncake ``store`` in the same Python process shares ONE
    ``TransferEngine`` (so at most one ADXL engine per process), and buffer
    registration is de-duplicated across store instances.
    """

    def __init__(self) -> None:
        self._engine = None
        self._engine_lock = threading.Lock()
        self._register_lock = threading.Lock()
        self._registered_ptrs: set[int] = set()
        self._hostname: str | None = None
        self._metadata_server: str | None = None
        self._protocol: str | None = None
        self._device_name: str | None = None

    def get_engine(
        self,
        hostname: str,
        metadata_server: str,
        protocol: str,
        device_name: str,
    ):
        if self._engine is not None:
            return self._engine
        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            try:
                from mooncake.engine import TransferEngine  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "Please install mooncake (https://github.com/kvcache-ai/Mooncake) "
                    "to run UcmMooncakeStoreV1."
                ) from exc

            engine = TransferEngine()
            ret = engine.initialize(
                hostname,
                metadata_server,
                protocol,
                device_name or "",
            )
            if ret != 0:
                raise RuntimeError(
                    f"TransferEngine.initialize failed: ret={ret}, "
                    f"hostname={hostname}, protocol={protocol}, "
                    f"metadata_server={metadata_server}."
                )
            self._engine = engine
            self._hostname = hostname
            self._metadata_server = metadata_server
            self._protocol = protocol
            self._device_name = device_name or ""
            logger.info(
                "Mooncake global TransferEngine initialized (host=%s, protocol=%s, "
                "rpc_port=%s).",
                hostname,
                protocol,
                self.rpc_port(),
            )
        return self._engine

    def rpc_port(self) -> int:
        assert self._engine is not None, "TransferEngine not initialized yet."
        return int(self._engine.get_rpc_port())

    def raw_engine(self):
        assert self._engine is not None, "TransferEngine not initialized yet."
        return self._engine.get_engine()

    def register_buffers(
        self,
        ptrs: list[int],
        sizes: list[int],
    ) -> list[int]:
        """Register each (ptr, size) via ``transfer_engine.register_memory``.

        Duplicates (same ptr already registered in this process) are skipped.
        Returns the list of ptrs actually registered by THIS call (callers can
        track them if they want to deregister later).
        """
        if self._engine is None:
            raise RuntimeError("TransferEngine not initialized; call get_engine first.")
        if len(ptrs) != len(sizes):
            raise ValueError("ptrs and sizes must have equal length.")

        newly_registered: list[int] = []
        with self._register_lock:
            for ptr, size in zip(ptrs, sizes):
                if ptr in self._registered_ptrs:
                    continue
                ret = self._engine.register_memory(ptr, size)
                if ret != 0:
                    raise RuntimeError(
                        "TransferEngine.register_memory failed for "
                        f"ptr={ptr}, size={size}, ret={ret}."
                    )
                self._registered_ptrs.add(ptr)
                newly_registered.append(ptr)
        return newly_registered


_GLOBAL_TE = _GlobalTransferEngine()


class UcmMooncakeStoreV1(UcmKVStoreBaseV1):

    def __init__(self, config: Dict[str, object]) -> None:
        super().__init__(config)
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                "to run UcmMooncakeStoreV1."
            ) from exc

        # The configuration determines both the runtime role (scheduler/worker)
        # and the memory layout used later by Mooncake batch APIs.
        is_scheduler = self._load_config(config)
        self._validate_ascend_runtime()

        self.store = MooncakeDistributedStore()
        self._shutdown = threading.Event()
        # Load/dump calls are submitted asynchronously so that the UCM task
        # interface can mirror the non-blocking behavior expected by callers.
        self._executor_workers = int(
            config.get("executor_workers", DEFAULT_MOONCAKE_V1_WORKERS)
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self._executor_workers,
            initializer=self._init_worker_context,
        )
        self._registered_buffers: list[int] = []
        if is_scheduler:
            self.device_id = 0
            logger.info(
                f"Mooncake setup detected scheduler path: old Mooncake does not support "
                f"{DEFAULT_SCHEDULER_PROTOCOL}, keep protocol={self.protocol} and set device to npu:{self.device_id}"
            )
            # Some deployments still initialize the scheduler on NPU 0 so that
            # the underlying runtime has a valid device context before setup.
            self._set_device_if_needed({"device_id": self.device_id})
        global_segment_size, local_buffer_size = self._build_segment_sizes(
            config, is_scheduler
        )

        # Align with the vllm-ascend reference: share ONE TransferEngine per
        # process, and pass it (plus a unique "host:port" local segment name)
        # into store.setup().
        transfer_engine = _GLOBAL_TE.get_engine(
            hostname=self.local_hostname,
            metadata_server=self.metadata_server,
            protocol=self.protocol,
            device_name=self.device_name,
        )
        self._transfer_engine = transfer_engine
        # Mooncake store.setup expects a unique local segment identifier.
        # Reusing "host:rpc_port" keeps it stable and easy to correlate in logs.
        self.local_seg = f"{self.local_hostname}:{_GLOBAL_TE.rpc_port()}"

        ret = self.store.setup(
            self.local_seg,
            self.metadata_server,
            global_segment_size,
            local_buffer_size,
            self.protocol,
            self.device_name,
            self.master_server_address,
            _GLOBAL_TE.raw_engine(),
        )
        if ret != 0:
            raise RuntimeError(f"Initialize mooncake failed with ret={ret}.")
        logger.info(
            f"Mooncake setup success: role={'scheduler' if is_scheduler else 'worker'}, "
            f"protocol={self.protocol}, local_seg={self.local_seg}, "
            f"global_segment_size={global_segment_size}, "
            f"local_buffer_size={local_buffer_size}"
        )
        self._register_buffers()

    def _load_config(self, config: Dict[str, object]) -> bool:
        self.device_id = config.get("device_id", None)
        # Historical behavior: missing device_id implies the scheduler path,
        # while workers are pinned to a concrete NPU id.
        is_scheduler = self.device_id is None
        self.protocol = str(config.get("protocol") or DEFAULT_PROTOCOL)

        # Each row in tensor_size_list describes one logical KV block shard
        # layout. Later we duplicate the row per request when building sizes.
        self.tensor_size_list = tuple(
            int(size) for size in config.get("tensor_size_list", ())
        )
        # register_buffer_ptrs/register_buffer_sizes are pre-allocated buffers
        # owned by the runtime that Mooncake may DMA into or out of directly.
        self.register_buffer_ptrs = tuple(
            int(ptr) for ptr in config.get("register_buffer_ptrs", ())
        )
        self.register_buffer_sizes = tuple(
            int(size) for size in config.get("register_buffer_sizes", ())
        )
        if len(self.register_buffer_ptrs) != len(self.register_buffer_sizes):
            raise ValueError(
                "register_buffer_ptrs and register_buffer_sizes must have the same length."
            )

        # Allow explicit override for multi-node deployments; otherwise fall
        # back to the IP helper used across the vLLM integration codepath.
        configured_local = config.get("local_hostname")
        if configured_local is not None and str(configured_local).strip():
            self.local_hostname = str(configured_local).strip()
        else:
            self.local_hostname = get_ip()
            logger.info(
                "Mooncake local_hostname not set in config; using vllm get_ip() -> %s",
                self.local_hostname,
            )
        self.metadata_server = str(
            config.get("metadata_server") or DEFAULT_METADATA_SERVER
        )
        self.master_server_address = str(
            config.get("master_server_address") or DEFAULT_MASTER_SERVER_ADDRESS
        )
        self.device_name = str(config.get("device_name") or DEFAULT_DEVICE_NAME)
        self.lookup_shard_indices = config.get(
            "lookup_shard_indices", DEFAULT_LOOKUP_SHARD_INDICES
        )
        return is_scheduler

    def _validate_ascend_runtime(self) -> None:
        if self.protocol != "ascend":
            raise RuntimeError(
                f"UcmMooncakeStoreV1 only supports Ascend protocol, but got protocol='{self.protocol}'."
            )
        if not hasattr(torch, "npu"):
            raise RuntimeError(
                "UcmMooncakeStoreV1 only supports Ascend platform, but torch.npu is not available."
            )

    def _build_segment_sizes(
        self, config: Dict[str, object], is_scheduler: bool
    ) -> tuple[int, int]:
        if is_scheduler:
            # The scheduler only coordinates metadata; it does not expose a
            # data segment of its own in the older Mooncake deployment model.
            return 0, 0
        global_segment_size = self._parse_size_to_bytes(
            config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE),
            "global_segment_size",
        )
        local_buffer_size = self._parse_size_to_bytes(
            config.get("local_buffer_size", DEFAULT_GLOBAL_SEGMENT_SIZE),
            "local_buffer_size",
        )
        return global_segment_size, local_buffer_size

    def _parse_size_to_bytes(self, value: object, field_name: str) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            cleaned = value.strip().upper()
            # Accept human-friendly strings used by config files, while keeping
            # the parser intentionally strict to avoid silently mis-sizing
            # Mooncake segments on typos.
            match = re.fullmatch(r"(\d+)\s*(MB|GB)", cleaned)
            if match:
                number = int(match.group(1))
                unit = match.group(2)
                multiplier = 1024 * 1024 if unit == "MB" else 1024 * 1024 * 1024
                return number * multiplier
            if cleaned.isdigit():
                return int(cleaned)
        raise ValueError(
            f"Invalid {field_name} value '{value}'. "
            "Use integer bytes or strings like '512MB'/'5GB'."
        )

    def cc_store(self) -> int:
        return 0

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        if not block_ids:
            logger.info("Mooncake lookup skipped: empty block_ids")
            return []

        # Lookups always target shard 0 because existence is tracked per block
        # id in this codepath; shard-aware reads/writes use _build_task_keys.
        keys = self._build_lookup_keys(block_ids)
        exists = self.store.batch_is_exist(keys)
        if len(exists) != len(keys):
            logger.error(
                f"Mooncake lookup result size mismatch: results={len(exists)}, keys={len(keys)}"
            )
            raise RuntimeError(
                f"Mooncake lookup returned {len(exists)} results for {len(keys)} keys."
            )

        hits = [int(value) == 1 for value in exists]
        return hits

    def lookup_on_prefix(self, block_ids: List[bytes]) -> int:
        for idx, exists in enumerate(self.lookup(block_ids)):
            if not exists:
                return idx - 1
        return len(block_ids) - 1

    def prefetch(self, block_ids: List[bytes]) -> None:
        del block_ids

    def load(self, block_ids, shard_index, dst_tensor) -> Task:
        # Convert tensors into raw device pointers once so the async worker can
        # call Mooncake without depending on higher-level tensor objects.
        addrs = self._tensor_normalize(dst_tensor)
        return self.load_data(block_ids, shard_index, addrs)

    def dump(self, block_ids, shard_index, src_tensor) -> Task:
        addrs = self._tensor_normalize(src_tensor)
        return self.dump_data(block_ids, shard_index, addrs)

    def load_data(self, block_ids, shard_index, dst_addr) -> Task:
        if isinstance(dst_addr, np.ndarray):
            addrs = dst_addr
        else:
            addrs = np.array(dst_addr, dtype=np.uint64)
        keys = self._build_task_keys(block_ids, shard_index)
        # Submit work to the executor so callers can decide whether to poll via
        # check() or block later via wait().
        future = self._executor.submit(self._execute_load, keys, addrs)
        return MooncakeTaskV1(future=future, operation="load_data", key_count=len(keys))

    def dump_data(
        self, block_ids, shard_index, src_addr, prerequisite_handle=0
    ) -> Task:
        if isinstance(src_addr, np.ndarray):
            addrs = src_addr
        else:
            addrs = np.array(src_addr, dtype=np.uint64)
        keys = self._build_task_keys(block_ids, shard_index)
        # prerequisite_handle is preserved for API compatibility with other
        # stores even though the current Mooncake path only needs a sync point.
        future = self._executor.submit(
            self._execute_dump, keys, addrs, prerequisite_handle
        )
        return MooncakeTaskV1(future=future, operation="dump_data", key_count=len(keys))

    def wait(self, task: Task) -> None:
        if not isinstance(task, MooncakeTaskV1):
            raise TypeError(f"Unsupported Mooncake task type: {type(task)}")
        try:
            task.future.result()
        except Exception as exc:
            raise RuntimeError(
                f"Mooncake {task.operation} task failed for {task.key_count} key(s): {exc}"
            ) from exc

    def check(self, task: Task) -> bool:
        if not isinstance(task, MooncakeTaskV1):
            return False
        return task.future.done()

    def shutdown(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        # cancel_futures=True avoids wedging on queued work (Py 3.9+).
        _kw: dict = {"wait": True}
        import sys as _sys

        if _sys.version_info >= (3, 9):
            _kw["cancel_futures"] = True
        self._executor.shutdown(**_kw)
        # NOTE: Buffers are registered via the process-level shared
        # TransferEngine (_GLOBAL_TE.register_buffers). We do NOT deregister
        # them here because the engine is shared by sibling stores in the same
        # process (mirrors the vllm-ascend reference behavior). The OS will
        # release everything at process exit.
        self._registered_buffers.clear()
        try:
            self.store.close()
        except Exception as exc:
            logger.warning(f"Mooncake store.close() raised: {exc}")

    def _build_lookup_keys(self, block_ids: List[bytes]) -> List[str]:
        keys = []
        for block_id in block_ids:
            block_hex = bytes(block_id).hex()
            # Lookup uses a canonical ":0" suffix because the existence check
            # is only probing whether the block family is present at all.
            keys.append(f"{block_hex}:0")
        return keys

    def _build_task_keys(
        self, block_ids: List[bytes], shard_index: List[int]
    ) -> List[str]:
        # Mooncake expects string keys in the form "<block_hex>:<shard>".
        return [
            f"{bytes(block_id).hex()}:{int(shard)}"
            for block_id, shard in zip(block_ids, shard_index)
        ]

    def _build_sizes_matrix(self, n_rows: int) -> List[List[int]]:
        # The C++ API receives one size row per key/buffer group.
        return [list(self.tensor_size_list) for _ in range(n_rows)]

    def _tensor_normalize(self, tensors: List[List[torch.Tensor]]) -> np.ndarray:
        if not tensors:
            return np.empty((0, 0), dtype=np.uint64)

        # Convert nested tensor lists into a dense uint64 pointer matrix so the
        # background worker can serialize arguments with minimal Python logic.
        return np.asarray(
            [[tensor.data_ptr() for tensor in row] for row in tensors],
            dtype=np.uint64,
        )

    def _set_device_context(self) -> None:
        if self.protocol == "ascend" and hasattr(torch, "npu"):
            # Worker threads do not automatically inherit the expected NPU
            # device, so each thread pins itself once at startup.
            torch.npu.set_device(torch.device(f"npu:{int(self.device_id)}"))

    def _init_worker_context(self) -> None:
        # Set thread-local device context once when worker starts.
        self._set_device_context()

    def _execute_load(self, keys: List[str], addrs: np.ndarray) -> None:
        addrs_list = addrs.tolist()
        sizes = self._build_sizes_matrix(len(addrs))
        # Mooncake writes directly into the provided device buffers.
        res = self.store.batch_get_into_multi_buffers(keys, addrs_list, sizes)
        self._raise_if_batch_failed("batch_get_into_multi_buffers", keys, res)

    def _execute_dump(
        self,
        keys: List[str],
        addrs: np.ndarray,
        prerequisite_handle: int,
    ) -> None:
        if (
            prerequisite_handle != 0
            and self.protocol == "ascend"
            and hasattr(torch, "npu")
        ):
            # Ensure producer-side NPU work is visible before Mooncake reads the
            # source buffers. This mirrors the conservative sync behavior in
            # related backends when an upstream handle is present.
            torch.npu.synchronize()
        addrs_list = addrs.tolist()
        sizes = self._build_sizes_matrix(len(addrs))
        res = self.store.batch_put_from_multi_buffers(keys, addrs_list, sizes)
        self._raise_if_batch_failed("batch_put_from_multi_buffers", keys, res)

    def _raise_if_batch_failed(self, op: str, keys: List[str], res) -> None:
        """Surface Mooncake batch errors as Python exceptions.

        Historically these C++ return codes were silently ignored, so a failed
        dump/load looked like a successful ``store.wait()`` but the destination
        tensor was all zeros. Mirrors the error handling in the vllm-ascend
        reference (``MooncakeBackend.put``/``.get``), but escalated to raise so
        that ``task.future.result()`` actually fails.
        """
        if res is None:
            return
        try:
            values = list(res)
        except TypeError:
            if isinstance(res, int) and res != 0:
                raise RuntimeError(f"Mooncake {op} failed with ret={res}.")
            return

        failures: list[tuple[str, int]] = []
        for key, value in zip(keys, values):
            try:
                code = int(value)
            except (TypeError, ValueError):
                continue
            if code < 0:
                failures.append((key, code))
        if failures:
            preview = ", ".join(f"{k}={c}" for k, c in failures[:3])
            raise RuntimeError(
                f"Mooncake {op} failed for {len(failures)}/{len(values)} key(s): "
                f"{preview}{' ...' if len(failures) > 3 else ''}"
            )

    def _set_device_if_needed(self, config: Dict[str, object]) -> None:
        if not hasattr(torch, "npu"):
            return
        if self.protocol != "ascend":
            return

        device_id = config.get("device_id", self.device_id)
        torch.npu.set_device(torch.device(f"npu:{int(device_id)}"))
        logger.info(f"Set NPU device to npu:{int(device_id)} before Mooncake setup")

    def _register_buffers(self) -> None:
        if not self.register_buffer_ptrs:
            return

        # De-duplicate (ptr, size) inside this store first.
        # The global engine does another layer of de-duplication across store
        # instances, but doing it locally keeps logs and accounting cleaner.
        seen: set[int] = set()
        ptrs: list[int] = []
        sizes: list[int] = []
        for ptr, size in zip(self.register_buffer_ptrs, self.register_buffer_sizes):
            if ptr in seen:
                continue
            seen.add(ptr)
            ptrs.append(int(ptr))
            sizes.append(int(size))

        newly = _GLOBAL_TE.register_buffers(ptrs, sizes)
        self._registered_buffers.extend(newly)

        logger.info(
            "Mooncake registered worker buffers via shared TransferEngine: "
            "new=%d, total_in_process_engine=%d (this store requested %d).",
            len(newly),
            len(_GLOBAL_TE._registered_ptrs),  # type: ignore[attr-defined]
            len(ptrs),
        )
