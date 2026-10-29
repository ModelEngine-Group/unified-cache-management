import heapq
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
