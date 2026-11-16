# -*- coding: utf-8 -*-
"""ucm/sandbox/sparse/gsav2/gsa_shared_memory.py 单元测试。

使用真实 multiprocessing.shared_memory（Linux/NPU 原生支持，Windows 亦可用）。
- 每个用例使用 uuid 唯一命名，避免重复运行/并行运行的段冲突；
- 慢超时分支通过小 timeout 构造参数或假时钟覆盖，保证套件总时长可控。
"""
import struct
import time
import uuid
from multiprocessing import shared_memory

import pytest
import torch

import ucm.sandbox.sparse.gsav2.gsa_shared_memory as shm_mod


def _name(prefix):
    return f"{prefix}_ut_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# SharedBlockTable
# ---------------------------------------------------------------------------
class TestSharedBlockTable:
    def test_creator_init(self):
        name = _name("sbt")
        sbt = shm_mod.SharedBlockTable(name, is_create=True, max_bs=4, max_block_len=8)
        try:
            assert sbt.version == 0
            assert struct.unpack_from("<I", sbt.shm.buf, 0)[0] == 0
            assert sbt.total_bytes == 4 * 8 * 4 + 4
        finally:
            sbt.close()
            sbt.unlink()

    def test_creator_file_exists_error(self):
        name = _name("sbt")
        first = shm_mod.SharedBlockTable(name, is_create=True)
        try:
            with pytest.raises(TimeoutError):
                shm_mod.SharedBlockTable(name, is_create=True)
        finally:
            first.close()
            first.unlink()

    def test_attach_success(self):
        name = _name("sbt")
        writer = shm_mod.SharedBlockTable(name, is_create=True)
        reader = None
        try:
            reader = shm_mod.SharedBlockTable(name, is_create=False, timeout=5)
            assert reader.shm is not None
        finally:
            if reader is not None:
                reader.close()
            writer.close()
            writer.unlink()

    def test_attach_timeout(self):
        with pytest.raises(TimeoutError):
            shm_mod.SharedBlockTable(_name("sbt"), is_create=False, timeout=0.2)

    def test_write_read_roundtrip(self):
        name = _name("sbt")
        writer = shm_mod.SharedBlockTable(
            name, is_create=True, max_bs=4, max_block_len=8
        )
        reader = None
        try:
            writer.write_add_block([[10, 11], [], [12]])
            reader = shm_mod.SharedBlockTable(
                name, is_create=False, max_bs=4, max_block_len=8, timeout=5
            )
            got = reader.read_add_block([2, 0, 1])
            assert got == [[10, 11], [], [12]]
            # 连续读写：版本号同步推进，不超时
            writer.write_add_block([[7]])
            assert reader.read_add_block([1]) == [[7]]
        finally:
            if reader is not None:
                reader.close()
            writer.close()
            writer.unlink()

    def test_write_index_exceeds_max_bs(self):
        sbt = shm_mod.SharedBlockTable(
            _name("sbt"), is_create=True, max_bs=2, max_block_len=4
        )
        try:
            with pytest.raises(IndexError):
                sbt.write_add_block([[], [], [1]])  # index 2 >= max_bs
        finally:
            sbt.close()
            sbt.unlink()

    def test_write_len_exceeds_max_block_len(self):
        sbt = shm_mod.SharedBlockTable(
            _name("sbt"), is_create=True, max_bs=2, max_block_len=4
        )
        try:
            with pytest.raises(ValueError):
                sbt.write_add_block([list(range(5))])
        finally:
            sbt.close()
            sbt.unlink()

    def test_read_zero_len_returns_empty(self):
        name = _name("sbt")
        writer = shm_mod.SharedBlockTable(name, is_create=True)
        reader = None
        try:
            writer.write_add_block([[1]])
            reader = shm_mod.SharedBlockTable(name, is_create=False, timeout=5)
            assert reader.read_add_block([1, 0]) == [[1], []]
        finally:
            if reader is not None:
                reader.close()
            writer.close()
            writer.unlink()

    def test_read_index_and_len_errors(self):
        name = _name("sbt")
        writer = shm_mod.SharedBlockTable(
            name, is_create=True, max_bs=2, max_block_len=4
        )
        reader = None
        try:
            writer.write_add_block([[1]])
            reader = shm_mod.SharedBlockTable(
                name, is_create=False, max_bs=2, max_block_len=4, timeout=5
            )
            with pytest.raises(IndexError):
                # 非零长度 + index 2 >= max_bs（全零长度会先走 continue 分支）
                reader.read_add_block([0, 0, 1])
            writer.write_add_block([[1]])
            with pytest.raises(ValueError):
                reader.read_add_block([5])  # len 5 > max_block_len
        finally:
            if reader is not None:
                reader.close()
            writer.close()
            writer.unlink()

    def test_read_version_mismatch_timeout(self):
        name = _name("sbt")
        writer = shm_mod.SharedBlockTable(name, is_create=True)
        reader = None
        try:
            writer.write_add_block([[1]])
            writer.write_add_block([[2]])  # writer version=2，reader 落后
            reader = shm_mod.SharedBlockTable(name, is_create=False, timeout=5)
            reader.timeout = 0.2
            with pytest.raises(TimeoutError):
                reader.read_add_block([1])
        finally:
            if reader is not None:
                reader.close()
            writer.close()
            writer.unlink()

    def test_close_sets_shm_none(self):
        sbt = shm_mod.SharedBlockTable(_name("sbt"), is_create=True)
        sbt.close()
        assert sbt.shm is None
        sbt.unlink()  # close 后 unlink → re-attach 路径
        try:
            shared_memory.SharedMemory(name=sbt.name, create=False).close()
        except FileNotFoundError:
            pass

    def test_unlink_non_creator_noop(self):
        name = _name("sbt")
        writer = shm_mod.SharedBlockTable(name, is_create=True)
        reader = None
        try:
            reader = shm_mod.SharedBlockTable(name, is_create=False, timeout=5)
            reader.unlink()  # 非创建者：直接返回，段仍存活
            writer.write_add_block([[3]])
        finally:
            if reader is not None:
                reader.close()
            writer.close()
            writer.unlink()

    def test_unlink_after_close_reattaches(self):
        name = _name("sbt")
        writer = shm_mod.SharedBlockTable(name, is_create=True)
        keeper = shared_memory.SharedMemory(name=name, create=False)
        try:
            writer.close()
            writer.unlink()  # shm is None → re-attach（keeper 持有段）→ unlink
        finally:
            keeper.close()

    def test_unlink_reattach_not_found(self):
        # 段已被彻底销毁 → re-attach FileNotFoundError → 静默返回
        name = _name("sbt")
        sbt = shm_mod.SharedBlockTable(name, is_create=True)
        holder = shared_memory.SharedMemory(name=name, create=False)
        sbt.close()
        holder.close()
        holder.unlink()
        sbt.unlink()  # 不应抛异常
        try:
            shared_memory.SharedMemory(name=name, create=False).close()
            pytest.fail("segment should be gone")
        except FileNotFoundError:
            pass

    def test_close_twice_is_noop(self):
        sbt = shm_mod.SharedBlockTable(_name("sbt"), is_create=True)
        sbt.close()
        sbt.close()  # shm 已为 None → 直接返回
        sbt.unlink()

    def test_unlink_with_alive_shm(self):
        # creator 不先 close 直接 unlink：shm 非 None → 直接销毁
        name = _name("sbt")
        sbt = shm_mod.SharedBlockTable(name, is_create=True)
        try:
            sbt.unlink()
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=name, create=False).close()
        finally:
            sbt.close()


# ---------------------------------------------------------------------------
# SharedTensorManager
# ---------------------------------------------------------------------------
class TestSharedTensorManager:
    def test_creator_init_and_read(self):
        stm = shm_mod.SharedTensorManager(
            _name("stm"), is_create=True, i_num=11, k_num=22
        )
        try:
            assert stm.read_num() == (11, 22)
            stm.update_num(33, 44)
            assert stm.read_num() == (33, 44)
        finally:
            stm.close()
            stm.unlink()

    def test_attach_and_read(self):
        name = _name("stm")
        writer = shm_mod.SharedTensorManager(name, is_create=True, i_num=5, k_num=6)
        reader = None
        try:
            reader = shm_mod.SharedTensorManager(name, is_create=False, timeout=5)
            assert reader.read_num() == (5, 6)
        finally:
            if reader is not None:
                reader.close()
            writer.close()
            writer.unlink()

    def test_attach_timeout(self):
        with pytest.raises(TimeoutError):
            shm_mod.SharedTensorManager(_name("stm"), is_create=False, timeout=0.2)

    def test_creator_file_exists_error(self):
        name = _name("stm")
        first = shm_mod.SharedTensorManager(name, is_create=True)
        try:
            with pytest.raises(TimeoutError):
                shm_mod.SharedTensorManager(name, is_create=True)
        finally:
            first.close()
            first.unlink()

    def test_close_sets_shm_none(self):
        stm = shm_mod.SharedTensorManager(_name("stm"), is_create=True)
        stm.close()
        assert stm.shm is None
        stm.close()  # 重复 close → no-op
        stm.unlink()
        try:
            shared_memory.SharedMemory(name=stm.name, create=False).close()
        except FileNotFoundError:
            pass

    def test_unlink_reattach_not_found(self):
        # 段已被彻底销毁 → re-attach FileNotFoundError → 静默返回。
        # 平台无关写法：Linux 上 close() 不删 /dev/shm 文件，必须由 holder
        # 显式 unlink 销毁段；Windows 上最后句柄关闭即销毁（unlink 为 no-op），
        # 两种平台下 creator 重挂都会命中 except 分支。
        name = _name("stm")
        stm = shm_mod.SharedTensorManager(name, is_create=True, i_num=1, k_num=2)
        holder = shared_memory.SharedMemory(name=name, create=False)
        stm.close()
        holder.close()
        holder.unlink()
        stm.unlink()  # 不应抛异常
        try:
            shared_memory.SharedMemory(name=name, create=False).close()
            pytest.fail("segment should be gone")
        except FileNotFoundError:
            pass

    def test_unlink_non_creator_noop(self):
        name = _name("stm")
        writer = shm_mod.SharedTensorManager(name, is_create=True, i_num=1, k_num=2)
        reader = None
        try:
            reader = shm_mod.SharedTensorManager(name, is_create=False, timeout=5)
            reader.unlink()  # 非创建者：直接返回
            assert writer.read_num() == (1, 2)  # 段仍存活
        finally:
            if reader is not None:
                reader.close()
            writer.close()
            writer.unlink()

    def test_unlink_with_alive_shm(self):
        name = _name("stm")
        stm = shm_mod.SharedTensorManager(name, is_create=True, i_num=1, k_num=2)
        try:
            stm.unlink()  # creator 直接销毁（未先 close）
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=name, create=False).close()
        finally:
            stm.close()


# ---------------------------------------------------------------------------
# SharedTensorVersion
# ---------------------------------------------------------------------------
class TestSharedTensorVersion:
    def _creator(self, tp_size=2, name=None):
        return shm_mod.SharedTensorVersion(
            name or _name("stv"), rank=0, tp_size=tp_size, is_create_version=True
        )

    def test_creator_init_zeroed(self):
        stv = self._creator()
        try:
            assert bytes(stv.shm.buf[:16]) == b"\x00" * 16
        finally:
            stv.close()
            stv.unlink()

    def test_update_prefetch_status(self):
        stv = self._creator(tp_size=2)
        try:
            stv.update_prefetch_status(1, 99)
            assert struct.unpack_from("<Q", stv.shm.buf, stv.i_offset)[0] == 1
            assert struct.unpack_from("<Q", stv.shm.buf, stv.v_offset)[0] == 99
        finally:
            stv.close()
            stv.unlink()

    def test_wait_ranks_version_all_aligned(self):
        stv = self._creator(tp_size=2)
        try:
            for i in range(2):
                struct.pack_into("<Q", stv.shm.buf, stv.elements_size * 2 + i * 8, 7)
            assert stv._wait_ranks_version(7, timeout=0.2) is True
        finally:
            stv.close()
            stv.unlink()

    def test_wait_ranks_version_timeout_returns_false(self):
        stv = self._creator(tp_size=2)
        try:
            # rank1 的 version 未对齐
            struct.pack_into("<Q", stv.shm.buf, stv.elements_size * 2, 7)
            assert stv._wait_ranks_version(7, timeout=0.1) is False
        finally:
            stv.close()
            stv.unlink()

    def test_update_version_creator_all_done(self):
        stv = self._creator(tp_size=2)
        try:
            # 各 rank 的 is_done 槽（i*8）与 version 槽（16 + i*8）均已就绪
            for i in range(2):
                struct.pack_into("<Q", stv.shm.buf, i * 8, 1)
                struct.pack_into("<Q", stv.shm.buf, 16 + i * 8, 5)
            stv.update_version(5)
            assert struct.unpack_from("<Q", stv.shm.buf, stv.version_offset)[0] == 5
            assert struct.unpack_from("<Q", stv.shm.buf, stv.is_done_offset)[0] == 1
        finally:
            stv.close()
            stv.unlink()

    def test_update_version_creator_some_not_done(self):
        stv = self._creator(tp_size=2)
        try:
            struct.pack_into("<Q", stv.shm.buf, 0 * 8, 1)
            struct.pack_into("<Q", stv.shm.buf, 1 * 8, 0)  # rank1 未完成
            for i in range(2):
                struct.pack_into("<Q", stv.shm.buf, 16 + i * 8, 6)
            stv.update_version(6)
            assert struct.unpack_from("<Q", stv.shm.buf, stv.version_offset)[0] == 6
            assert struct.unpack_from("<Q", stv.shm.buf, stv.is_done_offset)[0] == 0
        finally:
            stv.close()
            stv.unlink()

    def test_update_version_non_creator_noop(self):
        # update_version 的汇总逻辑仅在 creator 侧执行；
        # 非 creator 调用应静默返回、不写任何槽位
        name = _name("stv")
        creator = self._creator(tp_size=2, name=name)
        follower = None
        try:
            follower = shm_mod.SharedTensorVersion(
                name, rank=1, tp_size=2, is_create_version=False, timeout=5
            )
            follower.update_version(8)
            assert (
                struct.unpack_from("<Q", follower.shm.buf, follower.version_offset)[0]
                == 0
            )
            # follower 的正式职责：写自己的 is_done / version 槽位
            follower.update_prefetch_status(1, 8)
            assert struct.unpack_from("<Q", follower.shm.buf, follower.i_offset)[0] == 1
            assert struct.unpack_from("<Q", follower.shm.buf, follower.v_offset)[0] == 8
        finally:
            if follower is not None:
                follower.close()
            creator.close()
            creator.unlink()

    def test_update_version_creator_wait_timeout(self, monkeypatch):
        stv = self._creator(tp_size=2)
        try:
            monkeypatch.setattr(
                stv, "_wait_ranks_version", lambda version, timeout=30.0: False
            )
            with pytest.raises(TimeoutError):
                stv.update_version(9)
        finally:
            stv.close()
            stv.unlink()

    def test_is_version_same_match(self):
        stv = self._creator()
        try:
            # 手动就绪：各 rank version 槽对齐 + is_done 槽全 1
            for i in range(stv.tp_size):
                struct.pack_into("<Q", stv.shm.buf, i * 8, 1)
                struct.pack_into("<Q", stv.shm.buf, 8 * stv.tp_size + i * 8, 3)
            stv.update_version(3)
            assert stv.is_version_same(3) == 1  # is_done 槽为 1
        finally:
            stv.close()
            stv.unlink()

    def test_is_version_same_timeout_returns_false(self, monkeypatch):
        stv = self._creator()
        try:
            # 假时钟：起始 0s → 两次循环 0s → 之后 31s（>30s 超时）
            values = iter([0.0, 0.0, 0.0, 31.0])
            monkeypatch.setattr(time, "time", lambda: next(values, 31.0))
            monkeypatch.setattr(time, "sleep", lambda s: None)
            assert stv.is_version_same(12345) is False  # version_offset 处为 0
        finally:
            stv.close()
            stv.unlink()

    def test_creator_file_exists_error(self):
        name = _name("stv")
        first = self._creator(name=name)
        try:
            with pytest.raises(TimeoutError):
                self._creator(name=name)
        finally:
            first.close()
            first.unlink()

    def test_attach_timeout(self):
        with pytest.raises(TimeoutError):
            shm_mod.SharedTensorVersion(
                _name("stv"), rank=1, tp_size=2, is_create_version=False, timeout=0.2
            )

    def test_attach_retry_after_file_not_found(self, monkeypatch):
        # 第一次 attach 抛 FileNotFoundError → sleep 后重试成功
        name = _name("stv")
        creator = self._creator(name=name)
        real = shared_memory.SharedMemory
        state = {"first": True}

        def _flaky(name=None, create=False):
            if not create and state["first"]:
                state["first"] = False
                raise FileNotFoundError(name)
            return real(name=name, create=create)

        monkeypatch.setattr(shm_mod.shared_memory, "SharedMemory", _flaky)
        try:
            follower = shm_mod.SharedTensorVersion(
                name, rank=1, tp_size=2, is_create_version=False, timeout=5
            )
        finally:
            monkeypatch.undo()
            follower.close()
            creator.close()
            creator.unlink()

    def test_close_twice_is_noop(self):
        stv = self._creator()
        try:
            stv.close()
            stv.close()  # 重复 close → no-op
        finally:
            stv.unlink()

    def test_unlink_non_creator_noop(self):
        name = _name("stv")
        creator = self._creator(name=name)
        follower = None
        try:
            follower = shm_mod.SharedTensorVersion(
                name, rank=1, tp_size=2, is_create_version=False, timeout=5
            )
            follower.unlink()  # 非创建者：直接返回
            # 段仍存活：creator 仍可更新
            follower.update_prefetch_status(1, 5)
        finally:
            if follower is not None:
                follower.close()
            creator.close()
            creator.unlink()

    def test_unlink_with_alive_shm(self):
        name = _name("stv")
        stv = self._creator(name=name)
        try:
            stv.unlink()  # creator 直接销毁（未先 close）
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=name, create=False).close()
        finally:
            stv.close()

    def test_close_and_unlink(self):
        stv = self._creator()
        stv.close()
        assert stv.shm is None
        stv.unlink()
        try:
            shared_memory.SharedMemory(name=stv.name, create=False).close()
        except FileNotFoundError:
            pass

    def test_unlink_reattach_not_found(self):
        # 段已被彻底销毁 → re-attach FileNotFoundError → 静默返回
        # （平台无关写法，理由同 SharedTensorManager 的同名用例）
        name = _name("stv")
        stv = self._creator(name=name)
        holder = shared_memory.SharedMemory(name=name, create=False)
        stv.close()
        holder.close()
        holder.unlink()
        stv.unlink()  # 不应抛异常
        try:
            shared_memory.SharedMemory(name=name, create=False).close()
            pytest.fail("segment should be gone")
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# SharedTensorPool
# ---------------------------------------------------------------------------
class TestSharedTensorPool:
    def _pool(self, name=None):
        return shm_mod.SharedTensorPool(
            name or _name("stp"),
            shape=(4,),
            num=2,
            dtype=torch.int32,
            create=True,
        )

    def test_creator_init_layout(self):
        pool = self._pool()
        try:
            assert pool.single_elements == 4
            assert pool.total_elements == 8
            assert pool.total_bytes == 8 * 4
            assert len(pool.tensor_list) == 2
            for t in pool.tensor_list:
                assert tuple(t.shape) == (4,)
        finally:
            pool.close()
            pool.unlink()

    def test_write_and_read_views(self):
        name = _name("stp")
        writer = self._pool(name)
        reader = None
        try:
            writer.write([torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8])])
            reader = shm_mod.SharedTensorPool(
                name, shape=(4,), num=2, dtype=torch.int32, create=False, timeout=5
            )
            views = reader.read()
            assert views[0].tolist() == [1, 2, 3, 4]
            assert views[1].tolist() == [5, 6, 7, 8]
        finally:
            if reader is not None:
                reader.close()
            writer.close()
            writer.unlink()

    def test_write_wrong_count_raises(self):
        pool = self._pool()
        try:
            with pytest.raises(ValueError):
                pool.write([torch.zeros(4)])
        finally:
            pool.close()
            pool.unlink()

    def test_attach_timeout(self):
        with pytest.raises(TimeoutError):
            shm_mod.SharedTensorPool(
                _name("stp"),
                shape=(4,),
                num=2,
                dtype=torch.int32,
                create=False,
                timeout=0.2,
            )

    def test_creator_file_exists_error(self):
        name = _name("stp")
        first = self._pool(name)
        try:
            with pytest.raises(TimeoutError):
                self._pool(name)
        finally:
            first.close()
            first.unlink()

    def test_unlink_non_creator_noop(self):
        name = _name("stp")
        writer = self._pool(name)
        reader = None
        try:
            reader = shm_mod.SharedTensorPool(
                name, shape=(4,), num=2, dtype=torch.int32, create=False, timeout=5
            )
            reader.unlink()  # 非创建者：无操作
            writer.write([torch.ones(4), torch.zeros(4)])
        finally:
            if reader is not None:
                reader.close()
            writer.close()
            writer.unlink()

    def test_close_twice_is_noop(self):
        pool = self._pool()
        try:
            pool.close()
            pool.close()  # 重复 close → no-op
        finally:
            pool.unlink()

    def test_unlink_with_alive_shm(self):
        name = _name("stp")
        pool = self._pool(name)
        try:
            pool.unlink()  # creator 直接销毁（STP 的 unlink 不关闭句柄）
        finally:
            pool.close()
        # 句柄关闭后段销毁；Windows 上由资源追踪器异步完成，不在此断言
