/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#ifndef GSA_HBM_MANAGER_H
#define GSA_HBM_MANAGER_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <functional>
#include <iostream>
#include <kvcache_log.h>
#include <random>
#include <sstream>
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
        mHbmMutableSize = hbmCapacity - hbmImmutableSize;
        mTopkSize = topkSize;
        mMaxTotalSlots = (maxTotalSlots > hbmCapacity) ? maxTotalSlots : hbmCapacity;
        mPinnedSlotCount = 0;
        mLogger.log(
            LogLevel::DEBUG,
            "|HBM Manager| Init mDramTensorCount: %u, mHbmCapacity: %d, mHbmImmutableSize: %d, "
            "mHbmMutableSize: %d, mTopkSize: %d, mMaxTotalSlots: %d \n",
            mDramTensorCount, mHbmCapacity, mHbmImmutableSize, mHbmMutableSize, mTopkSize,
            mMaxTotalSlots);
        mGlobalToSlot.assign(mDramTensorCount, -1);
        // mSlotToGlobal 仅覆盖常规 slot [0, mHbmCapacity-1]
        // pinned slot 的 gid 映射由公式推导: gid = slot - mHbmCapacity + mDramTensorCount
        mSlotToGlobal.assign(mHbmCapacity, -1);
        mLruPrev.assign(mMaxTotalSlots, -1);
        mLruNext.assign(mMaxTotalSlots, -1);
        mIsImmutable.assign(mMaxTotalSlots, false);
        mIsPinned.assign(mMaxTotalSlots, false);
        mAddedSlotFlag.assign(mMaxTotalSlots, 0);
        // 初始化常规 slot 映射
        if (preloadHbm) {
            // local slot [0, mHbmCapacity-1] 映射为 global 组合:
            //   slot [0, 128]                   -> global [0, 128] 顺序
            //   slot [129, mHbmCapacity-1]      -> global [mDramTensorCount-1 ...
            //   mDramTensorCount-mHbmCapacity+129] 倒序
            int slotIdx = 0;
            for (; slotIdx <= 256 && slotIdx < mHbmCapacity; ++slotIdx) {
                mGlobalToSlot[slotIdx] = slotIdx;
                mSlotToGlobal[slotIdx] = slotIdx;
            }
            for (int gid = mDramTensorCount - 1; slotIdx < mHbmCapacity; --gid, ++slotIdx) {
                mGlobalToSlot[gid] = slotIdx;
                mSlotToGlobal[slotIdx] = gid;
            }
        }
        // 构建 LRU 链表：仅常规 slots (0 <-> 1 <-> ... <-> mHbmCapacity-1)
        // pinned slots 在 processTopkWithDebug 中按需动态加入
        mLruHead = 0;
        mLruTail = mHbmCapacity - 1;
        for (int i = 0; i < mHbmCapacity; ++i) {
            mLruPrev[i] = (i == 0) ? -1 : i - 1;
            mLruNext[i] = (i == mHbmCapacity - 1) ? -1 : i + 1;
        }
        mGeneration = 0;
        mAddedSlotGeneration = 0;
        updateImmutableZoneLogic();
        mInitialized = true;
    }

    void collectTopkHitsAndMisses(const std::vector<int>& dedupedTopkIds,
                                  std::vector<int>& topkSlotIds, std::vector<int>& missGlobalIds,
                                  std::vector<int>& missIndices, Logger& mLogger)
    {
        int dedupedSize = dedupedTopkIds.size();
        std::ostringstream oss;
        oss << "|HBM Manager|: " << " topk len: " << mTopkSize << " topk: ";
        for (int i = 0; i < dedupedSize; ++i) {
            int gid = dedupedTopkIds[i];
            oss << gid << " ";
            if (gid >= mDramTensorCount) {
                int transformedSlot = gid - mDramTensorCount + mHbmCapacity;
                if (transformedSlot >= mMaxTotalSlots) {
                    mLogger.log(LogLevel::ERROR,
                                "|HBM Manager| Pinned slot %d exceeds maxTotalSlots %d, gid: %d\n",
                                transformedSlot, mMaxTotalSlots, gid);
                    topkSlotIds[i] = transformedSlot;
                    continue;
                }
                if (!mIsPinned[transformedSlot]) {
                    mIsPinned[transformedSlot] = true;
                    insertToHead(transformedSlot);
                    mPinnedSlotCount++;
                } else {
                    moveToHead(transformedSlot);
                }
                topkSlotIds[i] = transformedSlot;
                mAddedSlotFlag[transformedSlot] = mAddedSlotGeneration;
                continue;
            }
            if (gid < 0) {
                mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| Topk is abnormal value: %d \n",
                            gid);
                continue;
            }
            int slot = mGlobalToSlot[gid];
            int thresahold = 2048;
            if (slot != -1) {
                moveToHead(slot);
                topkSlotIds[i] = slot;
                mAddedSlotFlag[slot] = mAddedSlotGeneration;
            } else {
                missGlobalIds.push_back(gid);
                missIndices.push_back(i);
                if ((int)missIndices.size() == thresahold) { break; }
            }
            int thresaholdSub = 2047;
            if (i == thresaholdSub) { mHitRate = 1 - missIndices.size() * 1.0 / thresahold; }
        }
        oss << "------\n";
        oss.str("");
    }

    void processTopkMisses(const std::vector<int>& dedupedTopkIds, std::vector<int>& topkSlotIds,
                           std::vector<int>& missGlobalIds, std::vector<int>& missSlots,
                           const std::vector<int>& missIndices, Logger& mLogger)
    {
        int missCount = missGlobalIds.size();
        int loadedCount = 0;
        if (missCount > 0) {
            for (int i = 0; i < missCount; ++i) {
                int victimSlot = mLruTail;
                while (victimSlot != -1 && mIsPinned[victimSlot]) {
                    victimSlot = mLruPrev[victimSlot];
                }
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
                removeFromList(victimSlot);
                insertToHead(victimSlot);
            }
            missGlobalIds.resize(loadedCount);
            for (int j = 0; j < loadedCount; ++j) {
                int idx = missIndices[j];
                int gid = dedupedTopkIds[idx];
                int slot = mGlobalToSlot[gid];
                topkSlotIds[idx] = slot;
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

    void processTopkWithDebug(const std::vector<int>& topkIds, std::vector<int>& output,
                              std::vector<int>& missGlobalIds, std::vector<int>& missSlots,
                              Logger& mLogger)
    {
        HBM_ASSERT(mInitialized);
        mTopkSize = topkIds.size();
        std::vector<int> topkSlotIds(mTopkSize, -1);
        missGlobalIds.reserve(mTopkSize);
        missSlots.reserve(mTopkSize);
        mGeneration = (mGeneration == INT_MAX) ? 1 : mGeneration + 1;
        mAddedSlotGeneration = (mAddedSlotGeneration == INT_MAX) ? 1 : mAddedSlotGeneration + 1;
        std::unordered_set<int> seenGids;
        std::vector<int> dedupedTopkIds;
        dedupedTopkIds.reserve(mTopkSize);
        for (int i = 0; i < mTopkSize; ++i) {
            int gid = topkIds[i];
            if (seenGids.count(gid)) {
                mLogger.log(LogLevel::ERROR,
                            "|HBM Manager| Duplicate gid in topkIds: %d at index: %d, skipping\n",
                            gid, i);
                continue;
            }
            seenGids.insert(gid);
            dedupedTopkIds.push_back(gid);
        }
        int dedupedSize = dedupedTopkIds.size();
        std::vector<int> missIndices;
        missIndices.reserve(dedupedSize);
        collectTopkHitsAndMisses(dedupedTopkIds, topkSlotIds, missGlobalIds, missIndices, mLogger);
        processTopkMisses(dedupedTopkIds, topkSlotIds, missGlobalIds, missSlots, missIndices,
                          mLogger);
        updateImmutableZoneLogic();
        buildTopkOutput(topkSlotIds, dedupedSize, output);
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
    int mHbmMutableSize;
    int mTopkSize;
    int mPinnedSlotCount;
    int mMaxTotalSlots;

    std::vector<int> mLruPrev;
    std::vector<int> mLruNext;
    std::vector<bool> mIsImmutable;
    std::vector<bool> mIsPinned;
    std::vector<int> mAddedSlotFlag;

    int mLruHead;
    int mLruTail;
    int mGeneration;
    int mAddedSlotGeneration;
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
        removeFromList(slot);
        insertToHead(slot);
    }

    inline void updateImmutableZoneLogic()
    {
        std::fill(mIsImmutable.begin(), mIsImmutable.end(), false);
        int count = 0;
        int curr = mLruHead;
        while (curr != -1 && count < mHbmImmutableSize) {
            mIsImmutable[curr] = true;
            curr = mLruNext[curr];
            count++;
        }
    }
};

#endif  // GSA_HBM_MANAGER_H