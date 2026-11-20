/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#ifndef GSA_HBM_MANAGER_H
#define GSA_HBM_MANAGER_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdint>
#include <functional>
#include <iostream>
#include <kvcache_log.h>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef DEBUG
#define HBM_ASSERT(f) assert(f)
#else
#define HBM_ASSERT(f) ((void)0)
#endif
// ================= HBM 管理器 =================
class HBMManager {
public:
    HBMManager() : mInitialized(false) {}

    void init(int dramTensorCount, int hbmCapacity, int hbmImmutableSize, int topkSize,
              Logger& mLogger, bool preloadHbm = true, int maxTotalSlots = 32768)
    {
        HBM_ASSERT(dramTensorCount > 0);
        HBM_ASSERT(hbmCapacity > 0);
        HBM_ASSERT(hbmImmutableSize > 0);
        HBM_ASSERT(topkSize > 0);
        HBM_ASSERT(hbmImmutableSize <= hbmCapacity);
        HBM_ASSERT(topkSize <= hbmCapacity);
        mDramTensorCount = dramTensorCount;
        mHbmCapacity = hbmCapacity;
        mHbmImmutableSize = hbmImmutableSize;
        mTopkSize = topkSize;
        mMaxTotalSlots = (maxTotalSlots > hbmCapacity) ? maxTotalSlots : hbmCapacity;
        mPinnedSlotCount = 0;
        mLogger.log(
            LogLevel::DEBUG,
            "|HBM Manager| Init mDramTensorCount: %u, mHbmCapacity: %d, mHbmImmutableSize: %d, "
            "mTopkSize: %d, mMaxTotalSlots: %d \n",
            mDramTensorCount, mHbmCapacity, mHbmImmutableSize, mTopkSize, mMaxTotalSlots);
        mGlobalToSlot.assign(mDramTensorCount, -1);
        // mSlotToGlobal 仅覆盖常规 slot [0, mHbmCapacity-1]；pinned slot 的 gid
        // 由公式推导: gid = slot - mHbmCapacity + mDramTensorCount
        mSlotToGlobal.assign(mHbmCapacity, -1);
        mLruPrev.assign(mMaxTotalSlots, -1);
        mLruNext.assign(mMaxTotalSlots, -1);
        mIsPinned.assign(mMaxTotalSlots, false);
        mAddedSlotFlag.assign(mMaxTotalSlots, 0);
        // 去重标记数组：覆盖常规 gid 域与 pinned gid 公式域两段连续区间。
        mSeenGidFlag.assign(mDramTensorCount + mMaxTotalSlots - mHbmCapacity, 0);
        mSeenGidGeneration = 0;
        if (preloadHbm) { preloadSlotMapping(); }
        initLruChain();
        mAddedSlotGeneration = 0;
        mInitialized = true;
    }

    /* 预载映射：slot [0, min(256, mDramTensorCount-1)] 恒等映射 global，
     * 剩余 slot 倒序回填 [mDramTensorCount-1 ...] 倒序的 gid，止步于恒等区间
     *（越界回填会产生幽灵 slot：validateMapping 必败、LRU 误驱逐、双槽
     * gather）。剩余 slot 保持 -1（冷填按需启用）。 */
    void preloadSlotMapping()
    {
        int slotIdx = 0;
        for (; slotIdx <= 256 && slotIdx < mHbmCapacity && slotIdx < mDramTensorCount;
             ++slotIdx) {
            mGlobalToSlot[slotIdx] = slotIdx;
            mSlotToGlobal[slotIdx] = slotIdx;
        }
        const int identityCount = slotIdx;
        for (int gid = mDramTensorCount - 1;
             slotIdx < mHbmCapacity && gid >= identityCount; --gid, ++slotIdx) {
            mGlobalToSlot[gid] = slotIdx;
            mSlotToGlobal[slotIdx] = gid;
        }
    }

    // LRU 链表仅串常规 slots (0 <-> ... <-> mHbmCapacity-1)；pinned 按需动态加入。
    void initLruChain()
    {
        mLruHead = 0;
        mLruTail = mHbmCapacity - 1;
        for (int i = 0; i < mHbmCapacity; ++i) {
            mLruPrev[i] = (i == 0) ? -1 : i - 1;
            mLruNext[i] = (i == mHbmCapacity - 1) ? -1 : i + 1;
        }
        /* 最尾可驱逐节点缓存：fresh 链全为常规非 pinned 槽，尾即候选。
         * 之后由 moveToHead / processTopkMisses 增量维护。 */
        mEvictTail = mLruTail;
    }

    /* miss 换入与回填：victim 直接取 mEvictTail（最尾非 pinned 节点，与自
     * mLruTail 走链跳过 pinned 的结果恒等）。逐 miss 重走 pinned 尾部聚簇
     * 是 O(miss*pinned)，此处增量维护摊还 O(1)：victim 换入新 gid 后将被
     * 钉到链头，新的最尾可驱逐节点从其原前驱回退查找（跳过的 pinned 节点
     * 此后不再被重复扫描）。missGlobalIds 与 missIndices 平行入栈
     *（missIndices[j] 为第 j 个 miss 的去重后序号），回填直接取
     * missGlobalIds[j]，无需 deduped 序列二次查表。 */
    void processTopkMisses(std::vector<int>& topkSlotIds,
                           std::vector<int>& missGlobalIds, std::vector<int>& missSlots,
                           const std::vector<int>& missIndices, Logger& mLogger)
    {
        int missCount = missGlobalIds.size();
        int loadedCount = 0;
        if (missCount > 0) {
            for (int i = 0; i < missCount; ++i) {
                int victimSlot = mEvictTail;
                if (victimSlot == -1) {
                    mLogger.log(LogLevel::ERROR,
                                 "|HBM Manager| No evictable slot for miss gid: %d, "
                                 "remaining %d misses dropped\n",
                                 missGlobalIds[i], missCount - i);
                    continue;
                }
                int oldGid = mSlotToGlobal[victimSlot];
                if (oldGid >= 0 && oldGid < mDramTensorCount) {
                    mGlobalToSlot[oldGid] = -1;
                } else if (oldGid == -1) {
                    mLogger.log(LogLevel::WARNING, "|HBM Manager| Evicting unallocated slot: %d\n",
                                victimSlot);
                } else {
                    mLogger.log(
                        LogLevel::ERROR,
                        "|HBM Manager| Evicting slot with out-of-range gid: %d in slot: %d\n",
                        oldGid, victimSlot);
                }
                int newGid = missGlobalIds[i];
                mSlotToGlobal[victimSlot] = newGid;
                mGlobalToSlot[newGid] = victimSlot;
                missSlots.push_back(victimSlot);
                loadedCount++;
                const int victimOldPrev = mLruPrev[victimSlot];
                removeFromList(victimSlot);
                insertToHead(victimSlot);
                /* victim 原前驱为 -1（victim 即链头，唯一常规槽场景）时，
                 * 回退查找为空：victim 已重载并仍在链头，它自身就是新的
                 * 最尾可驱逐节点。 */
                mEvictTail =
                    (victimOldPrev == -1) ? victimSlot : prevEvictable(victimOldPrev);
            }
            missGlobalIds.resize(loadedCount);
            for (int j = 0; j < loadedCount; ++j) {
                int slot = mGlobalToSlot[missGlobalIds[j]];
                topkSlotIds[missIndices[j]] = slot;
                if (slot >= 0 && slot < mMaxTotalSlots) {
                    mAddedSlotFlag[slot] = mAddedSlotGeneration;
                }
            }
        }
    }

    void buildTopkOutput(const std::vector<int>& topkSlotIds, int dedupedSize,
                         std::vector<int>& output)
    {
        output.reserve(mHbmImmutableSize);
        int realTopkSlotNum = 0;
        for (int i = 0; i < dedupedSize; ++i) {
            if (topkSlotIds[i] < 0) { continue; }
            output.push_back(topkSlotIds[i]);
            realTopkSlotNum += 1;
        }
        int slotsNeeded = mHbmImmutableSize - realTopkSlotNum;
        int count = 0;
        int curr = mLruHead;
        while (curr != -1 && count < slotsNeeded) {
            if (mAddedSlotFlag[curr] != mAddedSlotGeneration) {
                output.push_back(curr);
                mAddedSlotFlag[curr] = mAddedSlotGeneration;
                count++;
            }
            curr = mLruNext[curr];
        }
    }

    /* topk 查找主路径：代际标记数组去重（覆盖常规/pinned 两段 gid 域）+
     * hit/miss 分类合流单循环 + miss 换入 + 输出组装。dedupedCount 即去重后
     * 序号（负 gid 占序号不出槽）；miss 的 gid 与序号平行入栈供回填免查表；
     * scratch 复用类成员（per-(req,layer) 独占对象，无跨线程共享）。 */
    void processTopkWithDebug(const std::vector<int>& topkIds, std::vector<int>& output,
                              std::vector<int>& missGlobalIds, std::vector<int>& missSlots,
                              Logger& mLogger)
    {
        HBM_ASSERT(mInitialized);
        mTopkSize = topkIds.size();
        missGlobalIds.reserve(mTopkSize);
        missSlots.reserve(mTopkSize);
        mAddedSlotGeneration = (mAddedSlotGeneration == INT_MAX) ? 1 : mAddedSlotGeneration + 1;
        if (mSeenGidGeneration == 255) {
            std::fill(mSeenGidFlag.begin(), mSeenGidFlag.end(), 0);
            mSeenGidGeneration = 1;
        } else {
            ++mSeenGidGeneration;
        }
        mScratchSlotIds.assign(mTopkSize, -1);
        mScratchMissIndices.clear();
        int dedupedCount = 0;
        for (int i = 0; i < mTopkSize; ++i) {
            if (!classifyTopkGid(topkIds[i], i, dedupedCount, missGlobalIds, mLogger)) { break; }
        }
        processTopkMisses(mScratchSlotIds, missGlobalIds, missSlots, mScratchMissIndices, mLogger);
        buildTopkOutput(mScratchSlotIds, dedupedCount, output);
    }

    /* 单 gid 分类入 scratch：去重命中跳过 / pinned gid 换入并出槽 / 负 gid
     * 占序号不出槽 / 常规 gid hit 出槽或 miss 入栈（2048 上限 break）。
     * hit rate 在去重序号 2047 处采样。返回 false=miss 上限到达，停止循环。 */
    bool classifyTopkGid(int gid, int origIndex, int& dedupedCount,
                         std::vector<int>& missGlobalIds, Logger& mLogger)
    {
        constexpr int thresahold = 2048;
        constexpr int thresaholdSub = 2047;
        const int gidBound = static_cast<int>(mSeenGidFlag.size());
        if (gid >= 0 && gid < gidBound) {
            if (mSeenGidFlag[gid] == mSeenGidGeneration) {
                mLogger.log(LogLevel::ERROR,
                            "|HBM Manager| Duplicate gid in topkIds: %d at index: %d, "
                            "skipping\n",
                            gid, origIndex);
                return true;
            }
            mSeenGidFlag[gid] = mSeenGidGeneration;
        }
        if (gid >= mDramTensorCount) {
            int transformedSlot = gid - mDramTensorCount + mHbmCapacity;
            return handlePinnedGid(gid, transformedSlot, dedupedCount, mLogger);
        }
        if (gid < 0) {
            mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| Topk is abnormal value: %d \n", gid);
            ++dedupedCount;
            return true;
        }
        int slot = mGlobalToSlot[gid];
        if (slot != -1) {
            moveToHead(slot);
            mScratchSlotIds[dedupedCount] = slot;
            mAddedSlotFlag[slot] = mAddedSlotGeneration;
        } else {
            missGlobalIds.push_back(gid);
            mScratchMissIndices.push_back(dedupedCount);
            if ((int)mScratchMissIndices.size() == thresahold) { return false; }
        }
        if (dedupedCount == thresaholdSub) {
            mHitRate = 1 - mScratchMissIndices.size() * 1.0 / thresahold;
        }
        ++dedupedCount;
        return true;
    }

    /* pinned gid 换入：首次置位 + LRU 头插，否则移头；超界 pinned 的原始 slot
     * 值照写（不参与 LRU，行为不变）。 */
    bool handlePinnedGid(int gid, int transformedSlot, int& dedupedCount, Logger& mLogger)
    {
        if (transformedSlot >= mMaxTotalSlots) {
            mLogger.log(LogLevel::ERROR,
                        "|HBM Manager| Pinned slot %d exceeds maxTotalSlots %d, gid: %d\n",
                        transformedSlot, mMaxTotalSlots, gid);
            mScratchSlotIds[dedupedCount] = transformedSlot;
            ++dedupedCount;
            return true;
        }
        if (!mIsPinned[transformedSlot]) {
            mIsPinned[transformedSlot] = true;
            insertToHead(transformedSlot);
            mPinnedSlotCount++;
        } else {
            moveToHead(transformedSlot);
        }
        mScratchSlotIds[dedupedCount] = transformedSlot;
        mAddedSlotFlag[transformedSlot] = mAddedSlotGeneration;
        ++dedupedCount;
        return true;
    }

    int getSlotForGlobal(int gid) const
    {
        if (gid >= mDramTensorCount) { return gid - mDramTensorCount + mHbmCapacity; }
        if (gid >= 0 && gid < mDramTensorCount) { return mGlobalToSlot[gid]; }
        return -1;
    }

    // pinned slot -> gid: 由公式推导，gid = slot - mHbmCapacity + mDramTensorCount
    int getGidForSlot(int slot) const
    {
        if (slot >= 0 && slot < mHbmCapacity) { return mSlotToGlobal[slot]; }
        if (slot >= mHbmCapacity && slot < mMaxTotalSlots && mIsPinned[slot]) {
            return slot - mHbmCapacity + mDramTensorCount;
        }
        return -1;
    }

    bool isInitialized() const { return mInitialized; }
    int getTopkSize() const { return mTopkSize; }
    int getHbmCapacity() const { return mHbmCapacity; }
    int getHbmImmutableSize() const { return mHbmImmutableSize; }
    int getDramTensorCount() const { return mDramTensorCount; }
    int getPinnedSlotCount() const { return mPinnedSlotCount; }
    int getMaxTotalSlots() const { return mMaxTotalSlots; }

    bool validateLruChain() const
    {
        int expectedCount = mHbmCapacity + mPinnedSlotCount;
        int curr = mLruHead;
        int count = 0;
        int safetyLimit = mMaxTotalSlots + 10;

        while (curr != -1 && count < safetyLimit) {
            count++;
            curr = mLruNext[curr];
        }

        if (count >= safetyLimit || count != expectedCount) return false;

        curr = mLruTail;
        count = 0;
        while (curr != -1 && count < safetyLimit) {
            count++;
            curr = mLruPrev[curr];
        }

        return count == expectedCount;
    }

    bool validateMapping() const
    {
        // 检查常规 slot 映射
        for (int slot = 0; slot < mHbmCapacity; ++slot) {
            int gid = mSlotToGlobal[slot];
            if (gid == -1) continue;
            if (gid < -1 || gid >= mDramTensorCount) return false;
            if (mGlobalToSlot[gid] != slot) return false;
        }
        // 检查已初始化的 pinned slot 映射 (gid 由公式推导)
        for (int slot = mHbmCapacity; slot < mMaxTotalSlots; ++slot) {
            if (!mIsPinned[slot]) continue;
            int expectedGid = slot - mHbmCapacity + mDramTensorCount;
            // pinned slot 的 gid 必须与公式一致
            if (expectedGid < mDramTensorCount) return false;
        }
        return true;
    }

    // mSlotToGlobal 仅覆盖常规 slot [0, mHbmCapacity-1]
    // LastChunkMove 等外部代码仅遍历此范围
    std::vector<int> mSlotToGlobal;
    std::vector<int> mGlobalToSlot;
    float mHitRate;

private:
    int mDramTensorCount;
    int mHbmCapacity;
    int mHbmImmutableSize;
    int mTopkSize;
    int mPinnedSlotCount;
    int mMaxTotalSlots;

    std::vector<int> mLruPrev;
    std::vector<int> mLruNext;
    std::vector<bool> mIsPinned;
    std::vector<int> mAddedSlotFlag;
    // 去重标记数组（代际模式；与 mAddedSlotFlag 的 INT_MAX 翻转不同：uint8 空间
    // 小，代际到 255 时全量清零回 1，杜绝 8bit 空间的陈旧别名命中）
    std::vector<uint8_t> mSeenGidFlag;
    // 每调用 scratch：同尺寸 assign / clear 复用，免逐步堆分配
    std::vector<int> mScratchSlotIds;
    std::vector<int> mScratchMissIndices;

    int mLruHead;
    int mLruTail;
    // 最尾可驱逐（非 pinned）节点缓存：miss 驱逐 O(1) 直取，免逐次自
    // mLruTail 重扫 pinned 尾部聚簇（O(miss*pinned) -> 摊还 O(miss)）
    int mEvictTail;
    int mAddedSlotGeneration;
    int mSeenGidGeneration;
    bool mInitialized;

    inline void removeFromList(int slot)
    {
        int p = mLruPrev[slot];
        int n = mLruNext[slot];
        if (p != -1)
            mLruNext[p] = n;
        else
            mLruHead = n;
        if (n != -1)
            mLruPrev[n] = p;
        else
            mLruTail = p;
    }

    inline void insertToHead(int slot)
    {
        mLruPrev[slot] = -1;
        mLruNext[slot] = mLruHead;
        if (mLruHead != -1) mLruPrev[mLruHead] = slot;
        mLruHead = slot;
        if (mLruTail == -1) mLruTail = slot;
    }

    inline void moveToHead(int slot)
    {
        if (slot == mLruHead) return;
        /* 被移头的是最尾可驱逐节点时需刷新缓存：读取原前驱须在摘链前。 */
        const bool wasEvictTail = (slot == mEvictTail);
        const int oldPrev = mLruPrev[slot];
        removeFromList(slot);
        insertToHead(slot);
        if (wasEvictTail) { mEvictTail = prevEvictable(oldPrev); }
    }

    /* 自 from 起向链头方向回退，跳过 pinned 节点，返回首个可驱逐节点。
     * 摊还分析：每个 pinned 节点相对候选序单调后移，至多被跳过 O(1) 次。 */
    inline int prevEvictable(int from) const
    {
        int c = from;
        while (c != -1 && mIsPinned[c]) { c = mLruPrev[c]; }
        return c;
    }
};

#endif  // GSA_HBM_MANAGER_H