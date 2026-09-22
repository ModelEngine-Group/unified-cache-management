/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "../../../cache2/cc/ctrl_layout.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <new>

namespace UC::Cache2 {

struct CtrlLayoutTestAccess {
    static size_t TotalSize(size_t bucketCount, size_t lockCount, size_t slotCount)
    {
        return CtrlLayout::TotalSize(bucketCount, lockCount, slotCount);
    }
    static size_t LockStripeCount(size_t bucketCount)
    {
        return CtrlLayout::LockStripeCount(bucketCount);
    }
    static void Bind(CtrlLayout& layout, void* base, size_t rankCount, size_t slotsPerRank,
                     size_t bucketCount, size_t lockCount)
    {
        layout.Bind(base, rankCount, slotsPerRank, bucketCount, lockCount);
    }
    static size_t RankCount(const CtrlLayout& layout) { return layout.RankCount(); }
    static size_t SlotsPerRank(const CtrlLayout& layout) { return layout.SlotsPerRank(); }
    static size_t SlotSize(const CtrlLayout& layout) { return layout.SlotSize(); }
    static size_t LockCount(const CtrlLayout& layout) { return layout.LockCount(); }
    static std::atomic<size_t>* ClockHand(const CtrlLayout& layout, size_t rank)
    {
        return layout.ClockHand(rank);
    }
    static size_t NextClockSlot(const CtrlLayout& layout, size_t rank, size_t usableSlots)
    {
        return layout.NextClockSlot(rank, usableSlots);
    }
};

namespace {

class CtrlLayoutTest : public testing::Test {
protected:
    static constexpr size_t kRanks{2};
    static constexpr size_t kSlotsPerRank{4};
    static constexpr size_t kBuckets{8};
    static constexpr size_t kLocks{4};

    size_t bytes_{
        CtrlLayoutTestAccess::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank)};
    void* memory_{::operator new(bytes_, std::align_val_t{64})};
    CtrlLayout layout_;

    void SetUp() override
    {
        CtrlLayoutTestAccess::Bind(layout_, memory_, kRanks, kSlotsPerRank, kBuckets, kLocks);
        layout_.InitHeader(4096);
    }

    void TearDown() override { ::operator delete(memory_, std::align_val_t{64}); }
};

TEST_F(CtrlLayoutTest, BindsSharedArraysAtAlignedOffsets)
{
    EXPECT_EQ(CtrlLayoutTestAccess::RankCount(layout_), kRanks);
    EXPECT_EQ(CtrlLayoutTestAccess::SlotsPerRank(layout_), kSlotsPerRank);
    EXPECT_EQ(layout_.SlotCount(), kRanks * kSlotsPerRank);
    EXPECT_EQ(layout_.BucketCount(), kBuckets);
    EXPECT_EQ(CtrlLayoutTestAccess::LockCount(layout_), kLocks);
    EXPECT_EQ(CtrlLayoutTestAccess::SlotSize(layout_), 4096);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(layout_.SlotMetaArr()) % alignof(CtrlLayout::SlotMeta),
              0);
    for (size_t i = 0; i < kBuckets; ++i) {
        EXPECT_EQ(layout_.Buckets()[i].load(std::memory_order_relaxed), kInvalid);
    }
}

TEST_F(CtrlLayoutTest, MapsBucketsOntoLockStripes)
{
    EXPECT_EQ(layout_.LockOf(0), layout_.LockOf(4));
    EXPECT_EQ(layout_.LockOf(3), layout_.LockOf(7));
    EXPECT_NE(layout_.LockOf(0), layout_.LockOf(1));
    EXPECT_EQ(layout_.LockOf(kBuckets), nullptr);
}

TEST(CtrlLayoutStripeTest, ProductionLockCountCapsAndSharesStripes)
{
    constexpr size_t kBuckets{1ULL << 17};
    constexpr size_t kLocks{kMaxLockStripes};
    ASSERT_EQ(CtrlLayoutTestAccess::LockStripeCount(kBuckets), kLocks);
    auto bytes = CtrlLayoutTestAccess::TotalSize(kBuckets, kLocks, 1);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    CtrlLayout layout;
    CtrlLayoutTestAccess::Bind(layout, memory, 1, 1, kBuckets, kLocks);
    layout.InitHeader(4096);

    EXPECT_EQ(layout.LockOf(0), layout.LockOf(kLocks));
    EXPECT_EQ(layout.LockOf(1), layout.LockOf(kLocks + 1));
    EXPECT_NE(layout.LockOf(0), layout.LockOf(1));
    EXPECT_EQ(layout.LockOf(kBuckets), nullptr);
    ::operator delete(memory, std::align_val_t{64});
}

TEST_F(CtrlLayoutTest, InitializesOnlyRequestedRankRange)
{
    layout_.InitSlotRange(1);
    auto* slots = layout_.SlotMetaArr();
    for (size_t i = kSlotsPerRank; i < kRanks * kSlotsPerRank; ++i) {
        EXPECT_EQ(slots[i].reference.load(std::memory_order_relaxed), 0);
        EXPECT_EQ(slots[i].hash.load(std::memory_order_relaxed), kInvalid);
        EXPECT_EQ(slots[i].next.load(std::memory_order_relaxed), kInvalid);
        EXPECT_EQ(slots[i].accessed.load(std::memory_order_relaxed), 0);
    }
    EXPECT_EQ(CtrlLayoutTestAccess::ClockHand(layout_, 1)->load(std::memory_order_relaxed), 0);
}

TEST_F(CtrlLayoutTest, PublishesReadyAndRankDescriptor)
{
    EXPECT_FALSE(layout_.WaitReady(1));
    layout_.MarkReady();
    EXPECT_TRUE(layout_.WaitReady(1));

    EXPECT_FALSE(layout_.GetRankDesc(1));
    CtrlLayout::RankDataDesc desc;
    desc.handle.store(42, std::memory_order_relaxed);
    ASSERT_TRUE(layout_.SetRankDesc(1, desc).Success());
    auto result = layout_.GetRankDesc(1);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.Value().handle.load(std::memory_order_relaxed), 42);
}

TEST_F(CtrlLayoutTest, ClockHandNeverLeavesRankPartition)
{
    layout_.InitSlotRange(0);
    layout_.InitSlotRange(1);
    EXPECT_EQ(CtrlLayoutTestAccess::NextClockSlot(layout_, 1, kSlotsPerRank), 4);
    EXPECT_EQ(CtrlLayoutTestAccess::NextClockSlot(layout_, 1, kSlotsPerRank), 5);
    EXPECT_EQ(CtrlLayoutTestAccess::NextClockSlot(layout_, 1, kSlotsPerRank), 6);
    EXPECT_EQ(CtrlLayoutTestAccess::NextClockSlot(layout_, 1, kSlotsPerRank), 7);
    EXPECT_EQ(CtrlLayoutTestAccess::NextClockSlot(layout_, 1, kSlotsPerRank), 4);
    EXPECT_EQ(CtrlLayoutTestAccess::NextClockSlot(layout_, 0, kSlotsPerRank), 0);
    EXPECT_EQ(CtrlLayoutTestAccess::NextClockSlot(layout_, kRanks, kSlotsPerRank), kInvalid);
}

}  // namespace
}  // namespace UC::Cache2
