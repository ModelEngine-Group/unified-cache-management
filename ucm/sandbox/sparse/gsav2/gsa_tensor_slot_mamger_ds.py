import heapq
import time
from typing import List, Tuple


class Tensor:
    __slots__ = ("id", "hash", "last_use_time", "ref_count")

    def __init__(self, tid: int):
        self.id = tid
        self.hash = None  # 初始为空
        self.last_use_time = 0  # 最后使用时间
        self.ref_count = 0  # 引用计数


class TensorManager:
    def __init__(self, num: int = 10000):
        self.tensors = [Tensor(i) for i in range(num)]
        # 空闲 LRU 最小堆，存放 (last_use_time, tensor_id)
        self.heap = [(0, i) for i in range(num)]
        heapq.heapify(self.heap)
        self.hash_to_id = {}  # 仅存储有效 hash（非 -1）到 tensor_id 的映射
        self.time = 0  # 逻辑时钟

    def allocate(self, hash_list: List[str]) -> Tuple[List[int], List[int]]:
        """
        处理批量 allocate 请求。
        返回:
            ids: 每个输入 hash 分配到的 tensor ID
            miss_indices: 未命中的 hash 在 hash_list 中的索引
        """
        ids = []
        miss_indices = []

        for idx, h in enumerate(hash_list):
            # 判断命中：hash 不等于 -1 且存在于映射中
            if h != -1 and h in self.hash_to_id:
                tid = self.hash_to_id[h]
                t = self.tensors[tid]
                # 命中处理：引用计数 +1，更新时间
                t.ref_count += 1
                self.time += 1
                t.last_use_time = self.time
                ids.append(tid)
            else:
                # 未命中，记录索引
                miss_indices.append(idx)
                # 从空闲堆中取出最久未使用且引用计数为 0 的 tensor
                while self.heap:
                    _, tid = heapq.heappop(self.heap)
                    if (
                        0 <= tid < len(self.tensors)
                        and self.tensors[tid].ref_count == 0
                    ):
                        break
                else:
                    raise RuntimeError("没有空闲 tensor 可用")

                t = self.tensors[tid]

                # 移除旧 hash 映射
                old_hash = t.hash
                if old_hash is not None and old_hash != -1:
                    if self.hash_to_id.get(old_hash) == tid:
                        del self.hash_to_id[old_hash]

                # 设置新 hash（-1 不加入映射）
                if h != -1:
                    t.hash = h
                    self.hash_to_id[h] = tid
                else:
                    t.hash = -1

                # 更新引用计数和最后使用时间
                t.ref_count = 1
                self.time += 1
                t.last_use_time = self.time
                ids.append(tid)

        return ids, miss_indices

    def free(self, id_list: List[int]) -> None:
        """
        批量 free tensor。
        对每个 ID 引用计数减 1，若变为 0 则放入空闲堆。
        """
        for tid in id_list:
            if not (0 <= tid < len(self.tensors)):
                print(f"[GSA] tid wrong!!!", flush=True)
                continue
            t = self.tensors[tid]
            if t.ref_count > 0:
                t.ref_count -= 1
                if t.ref_count == 0:
                    # 回到空闲状态，插入堆中（保留原有 last_use_time）
                    heapq.heappush(self.heap, (t.last_use_time, tid))

    def get_free_count(self) -> int:
        """返回当前空闲 tensor 的数量（引用计数 == 0）"""
        return sum(1 for t in self.tensors if t.ref_count == 0)

    # ---------- 辅助方法，用于测试 ----------
    def get_tensor_info(self, tid: int):
        t = self.tensors[tid]
        return {
            "id": t.id,
            "hash": t.hash,
            "ref_count": t.ref_count,
            "last_use_time": t.last_use_time,
        }

    def active_count(self) -> int:
        """返回当前引用计数大于 0 的 tensor 数量"""
        return sum(1 for t in self.tensors if t.ref_count > 0)

    def free_count(self) -> int:
        """返回当前空闲的 tensor 数量（引用计数 == 0）"""
        return sum(1 for t in self.tensors if t.ref_count == 0)


# --------------------- 测试用例 ---------------------
def run_tests():
    print("=== 功能测试 ===")

    # 1. 基本 allocate 未命中
    mgr = TensorManager(10000)
    ids, misses = mgr.allocate([100, 200, 300])
    assert len(ids) == 3
    assert misses == [0, 1, 2]  # 全部未命中
    assert mgr.active_count() == 3
    print("测试1通过：全未命中分配")

    # 2. 命中已存在的 hash
    ids2, misses2 = mgr.allocate([100, 400])
    assert misses2 == [1]  # 100 命中，400 未命中
    assert ids2[0] == ids[0]  # 命中的 tensor ID 一致
    assert mgr.tensors[ids2[0]].ref_count == 2
    print("测试2通过：命中与未命中混合")

    # 3. -1 的处理（只能未命中）
    ids3, misses3 = mgr.allocate([-1, -1])
    assert len(ids3) == 2
    assert misses3 == [0, 1]
    # 分配的 tensor 不应该影响其他 hash 映射
    assert mgr.tensors[ids3[0]].hash == -1
    print("测试3通过：-1 特殊处理")

    # 4. free 与重新分配（引用计数归零后复用）
    # 当前引用情况：
    # ids[0] -> hash 100, ref=2; ids[1] -> 200, ref=1; ids[2] -> 300, ref=1;
    # ids2[1] -> 400, ref=1; ids3[0], ids3[1] -> hash -1, ref=1 each
    mgr.free([ids[0]])  # hash 100 ref: 2 -> 1
    info = mgr.get_tensor_info(ids[0])
    assert info["ref_count"] == 1
    mgr.free([ids[0]])  # ref: 1 -> 0, 回到空闲
    info = mgr.get_tensor_info(ids[0])
    assert info["ref_count"] == 0
    # 再次分配 100，应当命中同一个 tensor（因为 hash 映射还在）
    ids4, misses4 = mgr.allocate([100])
    assert misses4 == []
    assert ids4[0] == ids[0]
    assert mgr.tensors[ids[0]].ref_count == 1
    print("测试4通过：free 后重新命中")

    # 5. LRU 选择验证（需排除从未使用的 tensor 的干扰）
    mgr2 = TensorManager(10000)
    # 先占满所有 tensor，使每个 tensor 都被使用过一次，last_use_time 从 1 到 10000
    all_hashes = list(range(10000))
    occupy_ids, _ = mgr2.allocate(all_hashes)
    # 全部释放，所有 tensor 回到空闲堆
    mgr2.free(occupy_ids)
    # 此时所有 tensor 的 ref_count == 0，且 last_use_time 均大于 0
    # 重新分配三个 hash，再次占用三个 tensor（时间继续增加）
    ids_a, _ = mgr2.allocate([1, 2, 3])
    # 立即释放它们，这三个 tensor 会以各自的 last_use_time 重新入堆
    mgr2.free(ids_a)
    # 由于此时堆中仅包含这三个空闲 tensor（其他 9997 个已被占满并释放，但我们现在又分配了三个使它们活跃吗？不对：
    # 我们 occupy_ids 占用 10000，然后全部释放 -> 全部空闲。然后又 allocate([1,2,3]) -> 占用三个，其余 9997 仍空闲。
    # 释放这三个后，所有 10000 个又全部空闲。所有 tensor 的 last_use_time 均为其最近一次使用的时间。
    # 三个刚刚释放的 tensor 的 last_use_time 紧挨着，而其他 tensor 的 last_use_time 是之前 occupy 时的时间（1..10000）。
    # 因为逻辑时钟已经大于 10000，三个新分配并释放的 tensor 拥有更大的 last_use_time，
    # 所以最久未使用的应该是那些从未在第二次 allocate 中使用过的 tensor，而不是这三个。
    # 为正确测试 LRU，我们改为：释放后，让最近使用的三个 tensor 成为最久未使用的唯一候选。
    # 简单方案：分配两个 tensor，释放其中一个，再分配一个新 hash，应选中那个释放的。
    # 但依然有从未使用的 tensor 存在（last_use_time=0）。为了消除它们，可采用小规模 tensor 数测试。
    # 这里直接使用局部管理器，设 num=3，排除从未使用的干扰。
    mgr_lru = TensorManager(num=3)
    # 三个 tensor 初始 last_use_time=0
    ids_a, _ = mgr_lru.allocate([1, 2, 3])  # 分别获得 id0,1,2，时间变为 1,2,3
    mgr_lru.free(ids_a)  # 全部释放，堆中有 (1,id0),(2,id1),(3,id2)
    # 分配新 hash 4，应选择 last_use_time 最小的 id0
    ids_b, _ = mgr_lru.allocate([4])
    assert ids_b[0] == ids_a[0], f"期望选择最早释放的 id0，实际选择 {ids_b[0]}"
    print("测试5通过：LRU 选择正确（小规模，无从未使用干扰）")

    # 6. 懒惰删除验证
    mgr3 = TensorManager(num=3)
    # 分配 1,2,3 -> id0,id1,id2 (time=1,2,3)
    ids1, _ = mgr3.allocate([1, 2, 3])
    # 释放 id1 (time=2, ref=0) 和 id2 (time=3, ref=0)
    mgr3.free([ids1[1], ids1[2]])
    # 此时堆中有 (2,id1), (3,id2)；id0 仍在活跃(ref=1)
    # 命中 id1 对应的 hash 2，会使 id1 重新活跃 (ref=1)，但堆中仍有 (2,id1) 的过期条目
    ids2, _ = mgr3.allocate([2])
    assert ids2[0] == ids1[1]  # 命中，id1 被复用
    assert mgr3.tensors[ids1[1]].ref_count == 1
    # 现在分配新 hash 4，堆中最小的是 (2,id1) 但 id1 活跃，应该懒惰删除并弹出 (3,id2)
    ids3, _ = mgr3.allocate([4])
    assert ids3[0] == ids1[2], "懒惰删除失败，应该跳过活跃的 id1 而分配 id2"
    print("测试6通过：懒惰删除跳过活跃 tensor")

    print("\n=== 精度测试 ===")

    # 7. 批量操作引用计数一致性
    mgr4 = TensorManager(10000)
    hash_list = list(range(500))  # 500 个不同 hash
    ids_all, misses = mgr4.allocate(hash_list)
    assert len(ids_all) == 500
    assert misses == list(range(500))

    # 再次分配相同 hash，应全部命中且引用计数变为 2
    ids_rep, misses_rep = mgr4.allocate(hash_list)
    assert misses_rep == []
    assert all(mgr4.tensors[tid].ref_count == 2 for tid in ids_all)

    # 全部 free 一次，引用计数变为 1
    mgr4.free(ids_all)
    assert all(mgr4.tensors[tid].ref_count == 1 for tid in ids_all)

    # 再 free 一次，全部回到空闲
    mgr4.free(ids_all)
    assert all(mgr4.tensors[tid].ref_count == 0 for tid in ids_all)
    assert mgr4.active_count() == 0

    # 空闲状态下命中仍有效
    ids_hit, misses_hit = mgr4.allocate([0])
    assert misses_hit == []
    assert ids_hit[0] == ids_all[0]
    print("测试7通过：引用计数与空闲状态一致性")

    # 8. 无重复 tensor ID
    used_ids = set()
    for t in mgr4.tensors:
        if t.ref_count > 0:
            assert t.id not in used_ids, "活跃 tensor ID 重复"
            used_ids.add(t.id)
    print("测试8通过：活跃 tensor ID 唯一")

    print("\n=== 性能测试 ===")

    # 9. 模拟大量操作（2000 长度 hash list）
    mgr5 = TensorManager(10000)
    t0 = time.perf_counter()
    for _ in range(50):
        # 构造 2000 个 hash，末尾带一些 -1
        hlist = list(range(2000 - 5)) + [-1] * 5
        ids_res, _ = mgr5.allocate(hlist)
        # 随机释放一部分
        mgr5.free(ids_res[:100])
    t1 = time.perf_counter()
    print(f"测试9通过：50 次批量 allocate+free (长度 2000) 耗时 {t1 - t0:.4f} 秒")

    print("\n所有测试通过！")


if __name__ == "__main__":
    run_tests()
