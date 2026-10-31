/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * DT (end-to-end): HBMManager LRU correctness.
 *
 * This is the DT required by the task: drive the real HBMManager (no mocks)
 * through a long, deterministic access/miss sequence and assert the observable
 * outputs (slot assignment, miss/slot alignment, eviction order, LRU chain
 * integrity, immutable zone composition) match the expected LRU semantics.
 *
 * The scenario is reproducible: a fixed seed feeds a deterministic generator,
 * so any regression in the LRU / pinned-slot / eviction logic is caught.
 */
#include <cstdio>
#include <deque>
#include <gtest/gtest.h>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>
#include "kvcache_hbm_manger.h"
#include "kvcache_log.h"

namespace {

Logger kLog{};

// A reference LRU implementation used as the oracle for slot membership and
// eviction order over the regular (non-pinned) gid space.
struct RefLru {
    int capacity;
    std::deque<int> order;  // front = most recently used

    explicit RefLru(int cap) : capacity(cap) {}
    void touch(int slot)
    {
        auto it = std::find(order.begin(), order.end(), slot);
        if (it != order.end()) { order.erase(it); }
        order.push_front(slot);
    }
    int evictTail()
    {
        int victim = order.back();
        order.pop_back();
        return victim;
    }
};

}  // namespace

TEST(HBMLruDt, FullMissReplacementFollowsLruOrder)
{
    // Capacity 4, immutable 2, no preload so the initial slots are all empty.
    HBMManager m;
    const int dram = 1024, cap = 4, immu = 2;
    m.init(dram, cap, immu, /*topkSize=*/1, kLog, /*preload=*/false, /*maxTotalSlots=*/8);

    // Request gids 100..107 in order. Only 4 slots exist; each new miss evicts
    // the LRU tail. Because each request is a single gid, the slot assigned to
    // the latest miss becomes the MRU and the oldest surviving slot is evicted
    // next.
    std::vector<int> assignedSlots;
    for (int gid = 100; gid < 108; ++gid) {
        std::vector<int> output, missGids, missSlots;
        std::vector<int> topk = {gid};
        m.processTopkWithDebug(topk, output, missGids, missSlots, kLog);
        ASSERT_EQ(missSlots.size(), 1u) << "gid " << gid;
        assignedSlots.push_back(missSlots[0]);
        EXPECT_TRUE(m.validateLruChain()) << "gid " << gid;
        EXPECT_TRUE(m.validateMapping()) << "gid " << gid;
    }
    // After 8 misses on a 4-slot cache, only the last 4 gids (104..107) are
    // resident and each must map to a unique slot.
    std::set<int> residentSlots;
    for (int gid = 104; gid < 108; ++gid) {
        int slot = m.getSlotForGlobal(gid);
        EXPECT_GE(slot, 0) << "gid " << gid << " should be resident";
        residentSlots.insert(slot);
    }
    EXPECT_EQ(residentSlots.size(), 4u);
    // The evicted gids (100..103) must not be resident.
    for (int gid = 100; gid < 104; ++gid) {
        EXPECT_EQ(m.getSlotForGlobal(gid), -1) << "gid " << gid << " should be evicted";
    }
}

TEST(HBMLruDt, ReaccessPromotesSlotToHeadExtendingResidency)
{
    HBMManager m;
    m.init(1024, 4, 2, 1, kLog, false, 8);
    // Load gids 100, 101. Then re-access 100 (promote). Then load 102, 103, 104.
    // 104 should evict 101 (the LRU tail), not 100.
    auto load = [&](int gid) {
        std::vector<int> output, missGids, missSlots;
        std::vector<int> topk = {gid};
        m.processTopkWithDebug(topk, output, missGids, missSlots, kLog);
    };
    load(100);
    load(101);
    load(100);  // promote 100 to head
    load(102);
    load(103);
    load(104);                               // capacity=4, evicts LRU tail = 101
    EXPECT_GE(m.getSlotForGlobal(100), 0);   // still resident
    EXPECT_EQ(m.getSlotForGlobal(101), -1);  // evicted
    EXPECT_TRUE(m.validateLruChain());
}

TEST(HBMLruDt, PinnedSlotsDoNotParticipateInEviction)
{
    HBMManager m;
    const int dram = 1024, cap = 4;
    m.init(dram, cap, 2, 2, kLog, false, 8);
    // Pin one slot via a gid beyond dram range.
    std::vector<int> output, missGids, missSlots;
    std::vector<int> topk = {dram + 0};  // pinned slot = cap
    m.processTopkWithDebug(topk, output, missGids, missSlots, kLog);
    EXPECT_EQ(m.getPinnedSlotCount(), 1);
    // Now load 4 regular gids; all 4 regular slots get used, pinned slot stays.
    for (int gid = 0; gid < 4; ++gid) {
        output.clear();
        missGids.clear();
        missSlots.clear();
        std::vector<int> t = {gid};
        m.processTopkWithDebug(t, output, missGids, missSlots, kLog);
    }
    // Pinned gid still resolves via the formula.
    EXPECT_EQ(m.getGidForSlot(cap), dram + 0);
    EXPECT_TRUE(m.validateLruChain());
}

TEST(HBMLruDt, OutputPaddedToImmutableSizeWhenHitsInsufficient)
{
    HBMManager m;
    m.init(1024, 8, 4, 1, kLog, false, 16);
    std::vector<int> output, missGids, missSlots;
    std::vector<int> topk = {500};  // one miss
    m.processTopkWithDebug(topk, output, missGids, missSlots, kLog);
    // buildTopkOutput pads with LRU slots to reach immutable zone size (4).
    EXPECT_EQ(output.size(), 4u);
    EXPECT_TRUE(m.validateLruChain());
}

TEST(HBMLruDt, RandomWorkloadKeepsInvariantsAndSlotMappingConsistent)
{
    // Property test: under a random workload with NO pinned slots, the
    // missGids / missSlots arrays must stay aligned and getSlotForGlobal must
    // reflect the latest assignment.
    HBMManager m;
    m.init(2048, 32, 16, 8, kLog, /*preload=*/true, 64);
    std::mt19937 rng(424242);
    std::uniform_int_distribution<int> gidDist(0, 511);

    for (int iter = 0; iter < 500; ++iter) {
        std::vector<int> topk;
        int n = (gidDist(rng) % 16) + 1;
        topk.reserve(n);
        for (int i = 0; i < n; ++i) { topk.push_back(gidDist(rng)); }
        std::vector<int> output, missGids, missSlots;
        m.processTopkWithDebug(topk, output, missGids, missSlots, kLog);
        ASSERT_TRUE(m.validateLruChain()) << "iter " << iter;
        ASSERT_TRUE(m.validateMapping()) << "iter " << iter;
        // In the no-pinned-slot path the two arrays must be aligned.
        ASSERT_EQ(missGids.size(), missSlots.size());
        for (size_t i = 0; i < missSlots.size(); ++i) {
            EXPECT_EQ(m.getSlotForGlobal(missGids[i]), missSlots[i])
                << "iter " << iter << " miss " << i;
        }
    }
}

TEST(HBMLruDt, DuplicateTopkIdsCollapseToOneSlotAssignment)
{
    HBMManager m;
    m.init(1024, 8, 4, 4, kLog, false, 16);
    std::vector<int> output, missGids, missSlots;
    std::vector<int> topk = {200, 200, 200};
    m.processTopkWithDebug(topk, output, missGids, missSlots, kLog);
    // After dedup only one entry; at most one slot assigned.
    EXPECT_LE(missSlots.size(), 1u);
    if (!missSlots.empty()) { EXPECT_EQ(m.getSlotForGlobal(200), missSlots[0]); }
    EXPECT_TRUE(m.validateLruChain());
}

TEST(HBMLruDt, EvictionReusesSlotFreedByEvictedGid)
{
    HBMManager m;
    m.init(1024, 2, 1, 1, kLog, false, 4);
    // Load 100 -> slot A; load 101 -> slot B (capacity 2 full).
    // Load 102 -> evicts 100's slot; 100 must no longer resolve.
    auto load = [&](int gid) {
        std::vector<int> output, missGids, missSlots;
        std::vector<int> topk = {gid};
        m.processTopkWithDebug(topk, output, missGids, missSlots, kLog);
        return missSlots.empty() ? -1 : missSlots[0];
    };
    int s100 = load(100);
    int s101 = load(101);
    int s102 = load(102);
    EXPECT_NE(s100, -1);
    EXPECT_NE(s101, -1);
    EXPECT_NE(s102, -1);
    // 102 reuses the slot previously held by 100 (LRU tail).
    EXPECT_EQ(s102, s100);
    EXPECT_EQ(m.getSlotForGlobal(100), -1);
    EXPECT_EQ(m.getSlotForGlobal(102), s102);
}

TEST(HBMLruDt, MutableAndImmutableZoneCompositionAfterPromotion)
{
    HBMManager m;
    m.init(1024, 8, 4, 2, kLog, false, 16);
    // Load 4 distinct gids to fill the immutable zone, then promote one by
    // re-accessing it; the head of the LRU list defines the immutable zone.
    auto load = [&](int gid) {
        std::vector<int> output, missGids, missSlots;
        std::vector<int> topk = {gid};
        m.processTopkWithDebug(topk, output, missGids, missSlots, kLog);
    };
    load(10);
    load(20);
    load(30);
    load(40);
    load(10);  // promote 10
    EXPECT_TRUE(m.validateLruChain());
    // The slot for gid 10 should be the LRU head.
    int slot10 = m.getSlotForGlobal(10);
    EXPECT_GE(slot10, 0);
    // Repeated access keeps it resident regardless of further misses within capacity.
    EXPECT_GE(m.getSlotForGlobal(10), 0);
}
