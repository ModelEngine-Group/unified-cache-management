import array
import multiprocessing as mp
import os
import struct
import time
from multiprocessing import Event, shared_memory
from typing import List, Optional

import torch


class SharedBlockTable:
    def __init__(
        self,
        name: str,
        is_create: bool = False,
        timeout: float = 30,
        max_bs: int = 100,
        max_block_len: int = 2000,
    ):
        self.name = name
        # 预计算内存布局
        self.max_bs = max_bs
        self.max_block_len = max_block_len
        self.elements_size = 4
        self.version_size = 4
        self.one_req_offset = self.max_block_len * self.elements_size
        self.total_bytes = (
            self.max_bs * self.max_block_len * self.elements_size + self.version_size
        )
        self.shm = None
        self.is_creator = is_create
        self.version = 0
        self.timeout = timeout
        # 🔒 竞态安全初始化：自动选举创建者，其余进程自动附加
        if self.is_creator:
            try:
                self.shm = shared_memory.SharedMemory(
                    name=name, create=self.is_creator, size=self.total_bytes
                )
                struct.pack_into("<I", self.shm.buf, 0, self.version)
                print(f"[GSA] success create SharedBlockTable share memory!!")
            except FileExistsError:
                raise TimeoutError(f"[PID {os.getpid()}] 未能创建或连接共享内存 {name}")
        else:
            start = time.time()
            while time.time() - start < timeout:
                try:
                    self.shm = shared_memory.SharedMemory(name=name, create=False)
                    print(f"[GSA] success attach to SharedBlockTable share memory!!")
                    break
                except FileNotFoundError:
                    # 创建者正在分配中，短暂等待后重试
                    time.sleep(0.05)

        if self.shm is None:
            raise TimeoutError(
                f"[PID {os.getpid()}] 超时 {timeout}s：未能创建或连接共享内存 {name}"
            )

    def write_add_block(self, add_blocks):
        self.version += 1
        for index, add_block in enumerate(add_blocks):
            if len(add_block) == 0:
                continue
            if index >= self.max_bs:
                raise IndexError(
                    f"add_block index {index} exceeds max_bs {self.max_bs}"
                )
            if len(add_block) > self.max_block_len:
                raise ValueError(
                    f"add_block len {len(add_block)} exceeds "
                    f"max_block_len {self.max_block_len}"
                )
            index_offset = self.one_req_offset * index + self.version_size
            arr = array.array("i", add_block)
            raw_bytes = arr.tobytes()
            self.shm.buf[index_offset : index_offset + len(raw_bytes)] = raw_bytes
        struct.pack_into("<I", self.shm.buf, 0, self.version)

    def read_add_block(self, add_block_lens):
        self.version += 1
        start = time.time()
        while time.time() - start < self.timeout:
            cur_v = struct.unpack_from("<I", self.shm.buf, 0)[0]
            if self.version == cur_v:
                break
            time.sleep(0.0001)

        if time.time() - start > self.timeout:
            raise TimeoutError(
                f"[GSA] [PID {os.getpid()}] timeout waiting for read_add_block "
                f"(version {self.version})"
            )

        add_blocks = []
        for index, add_block in enumerate(add_block_lens):
            if add_block == 0:
                add_blocks.append([])
                continue
            if index >= self.max_bs:
                raise IndexError(
                    f"add_block index {index} exceeds max_bs {self.max_bs}"
                )
            if add_block > self.max_block_len:
                raise ValueError(
                    f"add_block len {add_block} exceeds "
                    f"max_block_len {self.max_block_len}"
                )
            index_offset = self.one_req_offset * index + self.version_size
            arr = array.array("i")
            raw_bytes = add_block * self.elements_size
            arr.frombytes(bytes(self.shm.buf[index_offset : index_offset + raw_bytes]))
            add_blocks.append(list(arr))
        return add_blocks

    def close(self):
        """关闭当前进程的内存句柄（所有进程退出前必须调用）"""
        if self.shm is not None:
            self.shm.close()
            self.shm = None

    def unlink(self):
        """销毁共享内存（⚠️ 仅创建者进程调用一次）"""
        if not self.is_creator:
            return
        shm = self.shm
        if shm is None:
            # close() has already been called: re-attach so we can still unlink.
            try:
                shm = shared_memory.SharedMemory(name=self.name, create=False)
            except FileNotFoundError:
                return
        shm.unlink()
        shm.close()


class SharedTensorManager:
    def __init__(
        self,
        name: str,
        is_create: bool = False,
        i_num: int = 0,
        k_num: int = 0,
        timeout: float = 120,
    ):
        self.name = name
        # 预计算内存布局
        self.total_bytes = 4096
        self.shm = None
        self.is_creator = is_create
        # 🔒 竞态安全初始化：自动选举创建者，其余进程自动附加
        if self.is_creator:
            try:
                self.shm = shared_memory.SharedMemory(
                    name=name, create=self.is_creator, size=self.total_bytes
                )
                struct.pack_into("<I", self.shm.buf, 0, i_num)
                struct.pack_into("<I", self.shm.buf, 4, k_num)
                print(f"[GSA] success create memory manager share memory!!", flush=True)
            except FileExistsError:
                raise TimeoutError(f"[PID {os.getpid()}] 未能创建或连接共享内存 {name}")
        else:
            start = time.time()
            while time.time() - start < timeout:
                try:
                    self.shm = shared_memory.SharedMemory(name=name, create=False)
                    print(
                        f"[GSA] success attach to memory manager share memory!!",
                        flush=True,
                    )
                    break
                except FileNotFoundError:
                    # 创建者正在分配中，短暂等待后重试
                    print(f"[GSA] 创建者正在分配中，短暂等待后重试!!", flush=True)
                    time.sleep(0.5)

        if self.shm is None:
            raise TimeoutError(
                f"[PID {os.getpid()}] 超时 {timeout}s：未能创建或连接共享内存 {name}"
            )

    def update_num(self, i_num, k_num):
        struct.pack_into("<I", self.shm.buf, 0, i_num)
        struct.pack_into("<I", self.shm.buf, 4, k_num)

    def read_num(self):
        i_num = struct.unpack_from("<I", self.shm.buf, 0)[0]
        k_num = struct.unpack_from("<I", self.shm.buf, 4)[0]
        return i_num, k_num

    def close(self):
        """关闭当前进程的内存句柄（所有进程退出前必须调用）"""
        if self.shm is not None:
            self.shm.close()
            self.shm = None

    def unlink(self):
        """销毁共享内存（⚠️ 仅创建者进程调用一次）"""
        if not self.is_creator:
            return
        shm = self.shm
        if shm is None:
            try:
                shm = shared_memory.SharedMemory(name=self.name, create=False)
            except FileNotFoundError:
                return
        shm.unlink()
        shm.close()


class SharedTensorVersion:
    def __init__(
        self,
        name: str,
        rank: int = 0,
        tp_size: int = 16,
        is_create_version: bool = False,
        timeout: float = 30,
    ):
        """
        |is_done(1)|is_done(2)|...|is_done(tp_size)|version(1)|version(2)|
        ...|version(tp_size)|sync version|sync is_done|
        |is_done_offset|version_offset|sv_offset|si_offset|
        """
        self.name = name
        self.total_bytes = 4096
        self.shm = None
        self.is_creator = is_create_version
        self.rank = rank
        self.elements_size = 8
        self.tp_size = tp_size
        self.i_offset = rank * self.elements_size
        self.v_offset = self.elements_size * tp_size + rank * self.elements_size
        self.version_offset = self.elements_size * tp_size * 2
        self.is_done_offset = self.version_offset + self.elements_size
        # 🔒 竞态安全初始化：自动选举创建者，其余进程自动附加
        if self.is_creator:
            try:
                self.shm = shared_memory.SharedMemory(
                    name=name, create=self.is_creator, size=self.total_bytes
                )
                self.shm.buf[:] = b"\x00" * self.total_bytes
                print(f"[GSA] success create version share memory!!")
            except FileExistsError:
                raise TimeoutError(f"[PID {os.getpid()}] 未能创建或连接共享内存 {name}")
        else:
            start = time.time()
            while time.time() - start < timeout:
                try:
                    self.shm = shared_memory.SharedMemory(name=name, create=False)
                    print(f"[GSA] success attach to version share memory!!")
                    break
                except FileNotFoundError:
                    # 创建者正在分配中，短暂等待后重试
                    time.sleep(0.05)

        if self.shm is None:
            raise TimeoutError(
                f"[PID {os.getpid()}] 超时 {timeout}s：未能创建或连接共享内存 {name}"
            )

    def update_prefetch_status(self, is_done, version):
        struct.pack_into("<Q", self.shm.buf, self.i_offset, is_done)
        struct.pack_into("<Q", self.shm.buf, self.v_offset, version)

    def _wait_ranks_version(self, version, timeout: float = 30.0) -> bool:
        """等待所有 rank 的 version 与目标一致，最长 ``timeout`` 秒。

        Returns:
            True if all ranks are aligned on ``version``, False on timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            flag = True
            for i in range(self.tp_size):
                cur_v = struct.unpack_from(
                    "<Q",
                    self.shm.buf,
                    self.elements_size * self.tp_size + i * self.elements_size,
                )[0]
                if version != cur_v:
                    flag = False
                    time.sleep(0.0001)
                    break
            if flag:
                return True
        return False

    def update_version(self, version):
        if self.is_creator:
            if not self._wait_ranks_version(version):
                raise TimeoutError(
                    f"[GSA] timeout waiting for all ranks to align on version {version}"
                )

            for i in range(self.tp_size):
                cur_v = struct.unpack_from("<Q", self.shm.buf, i * self.elements_size)[
                    0
                ]
                if cur_v == 0:
                    struct.pack_into("<Q", self.shm.buf, self.is_done_offset, 0)
                    struct.pack_into("<Q", self.shm.buf, self.version_offset, version)
                    return
            struct.pack_into("<Q", self.shm.buf, self.is_done_offset, 1)
            struct.pack_into("<Q", self.shm.buf, self.version_offset, version)

    def is_version_same(self, version):
        time_out = 30
        start = time.time()
        while time.time() - start < time_out:
            cur_v = struct.unpack_from("<Q", self.shm.buf, self.version_offset)[0]
            if version == cur_v:
                break
            time.sleep(0.0001)

        if time.time() - start > time_out:
            print(
                f"[GSA] timeout for get prefetch version for rank {self.rank} for version {version}",
                flush=True,
            )
            return False

        is_prefetch = struct.unpack_from("<Q", self.shm.buf, self.is_done_offset)[0]
        return is_prefetch

    def close(self):
        """关闭当前进程的内存句柄（所有进程退出前必须调用）"""
        if self.shm is not None:
            self.shm.close()
            self.shm = None

    def unlink(self):
        """销毁共享内存（⚠️ 仅创建者进程调用一次）"""
        if not self.is_creator:
            return
        shm = self.shm
        if shm is None:
            try:
                shm = shared_memory.SharedMemory(name=self.name, create=False)
            except FileNotFoundError:
                return
        shm.unlink()
        shm.close()


class SharedTensorPool:
    def __init__(
        self,
        name: str,
        shape: tuple,
        num: int,
        dtype,
        create: bool,
        timeout: float = 180,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.num_tensors = num

        # 预计算内存布局
        self.single_elements = int(torch.prod(torch.tensor(self.shape)).item())
        self.item_size = torch.tensor([], dtype=self.dtype).element_size()
        self.total_elements = self.num_tensors * self.single_elements
        self.total_bytes = self.total_elements * self.item_size

        self.shm = None
        self.is_creator = create

        # 🔒 竞态安全初始化：自动选举创建者，其余进程自动附加
        if self.is_creator:
            try:
                self.shm = shared_memory.SharedMemory(
                    name=name, create=self.is_creator, size=self.total_bytes
                )
                self.shm.buf[0] = 0
                print(
                    f"[GSA] success create share memory!! shape {self.shape} tensor num {self.num_tensors}"
                )
            except FileExistsError:
                raise TimeoutError(f"[PID {os.getpid()}] 未能创建或连接共享内存 {name}")
        else:
            start = time.time()
            while time.time() - start < timeout:
                try:
                    self.shm = shared_memory.SharedMemory(name=name, create=False)
                    print(
                        f"[GSA] success attach to share memory!! shape {self.shape} tensor num {self.num_tensors}"
                    )
                    break
                except FileNotFoundError:
                    # 创建者正在分配中，短暂等待后重试
                    time.sleep(0.05)

        if self.shm is None:
            raise TimeoutError(
                f"[PID {os.getpid()}] 超时 {timeout}s：未能创建或连接共享内存 {name}"
            )

        self._shm_view_1d = torch.frombuffer(
            self.shm.buf, dtype=self.dtype, count=self.total_elements
        )
        self.tensor_list = [
            self._shm_view_1d[
                i * self.single_elements : (i + 1) * self.single_elements
            ].reshape(self.shape)
            for i in range(self.num_tensors)
        ]

    def write(self, src_list: List[torch.Tensor]):
        if len(src_list) != self.num_tensors:
            raise ValueError(f"期望 {self.num_tensors} 个张量，实际 {len(src_list)}")
        for i in range(self.num_tensors):
            self.tensor_list[i].copy_(src_list[i], non_blocking=False)

    def read(self) -> List[torch.Tensor]:
        """🔵 其他进程调用：直接返回内存视图（无锁）"""
        return self.tensor_list

    def close(self):
        """关闭当前进程的内存句柄（所有进程退出前必须调用）"""
        if self.shm is not None:
            self.shm.close()
            self.shm = None

    def unlink(self):
        """🗑️ 销毁共享内存（⚠️ 仅创建者进程调用一次）"""
        if self.shm is not None and self.is_creator:
            self.shm.unlink()
