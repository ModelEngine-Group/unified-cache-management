/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "../../../cache2/cc/cache_buffer.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <new>
#include <optional>
#include <thread>
#include <vector>

namespace UC::Cache2 {

struct BufferTestAccess {
    static void Init(Buffer& buffer, void* memory, size_t rankCount, size_t slotsPerRank,
                     size_t bucketCount, size_t lockCount)
    {
        auto& layout = buffer.ctrl_.Layout();
        layout.Bind(memory, rankCount, slotsPerRank, bucketCount, lockCount);
        layout.InitHeader(4096);
        layout.InitSlotRange(0);
        buffer.myRank_ = 0;
        buffer.rankCount_ = rankCount;
        buffer.slotsPerRank_ = slotsPerRank;
        buffer.slotSize_ = 4096;
        buffer.bucketCount_ = bucketCount;
        buffer.reservedSlots_ = 0;
    }

    static size_t ReferenceCount(Buffer& buffer, const Detail::BlockId& blockId, size_t offset)
    {
        auto iBucket = buffer.HashKey(blockId);
        auto& layout = buffer.ctrl_.Layout();
        auto iNode = buffer.Lookup(layout, iBucket, blockId, offset);
        return iNode == kInvalid
                   ? kInvalid
                   : layout.SlotMetaArr()[iNode].reference.load(std::memory_order_acquire);
    }

    static void InitRank(Buffer& buffer, size_t rank)
    {
        buffer.ctrl_.Layout().InitSlotRange(rank);
        buffer.myRank_ = rank;
    }

    static Buffer::Handle TryGet(Buffer& buffer, const Detail::BlockId& blockId, size_t offset,
                                 size_t attempts)
    {
        return buffer.TryGet(blockId, offset, false, attempts);
    }

    static CtrlLayout& Layout(Buffer& buffer) { return buffer.ctrl_.Layout(); }

    static size_t BucketOf(Buffer& buffer, const Detail::BlockId& blockId)
    {
        return buffer.HashKey(blockId);
    }

    static size_t FindSlot(Buffer& buffer, const Detail::BlockId& blockId, size_t offset)
    {
        auto iBucket = buffer.HashKey(blockId);
        return buffer.Lookup(buffer.ctrl_.Layout(), iBucket, blockId, offset);
    }
};

namespace {

Detail::BlockId MakeBlockId(uint32_t value)
{
    Detail::BlockId block;
    block.fill(static_cast<std::byte>(0));
    std::memcpy(block.data(), &value, sizeof(value));
    return block;
}

class Cache2BufferTest : public testing::Test {
protected:
    static constexpr size_t kRanks{1};
    static constexpr size_t kSlotsPerRank{128};
    static constexpr size_t kBuckets{128};
    static constexpr size_t kLocks{64};

    size_t bytes_{CtrlLayout::TotalSize(kBuckets, kLocks, kRanks* kSlotsPerRank)};
    void* memory_{::operator new(bytes_, std::align_val_t{64})};
    Buffer buffer_;

    void SetUp() override
    {
        BufferTestAccess::Init(buffer_, memory_, kRanks, kSlotsPerRank, kBuckets, kLocks);
    }

    void TearDown() override { ::operator delete(memory_, std::align_val_t{64}); }
};

TEST_F(Cache2BufferTest, PreallocLeavesOwnerElectionForDemandGet)
{
    auto block = MakeBlockId(1);
    buffer_.Prealloc(block, 0);
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 0);

    auto owner = buffer_.Get(block, 0);
    ASSERT_TRUE(owner);
    EXPECT_TRUE(owner.Owner());
    EXPECT_EQ(owner.GetState(), CtrlLayout::SlotMeta::State::Loading);

    auto reader = buffer_.Get(block, 0);
    ASSERT_TRUE(reader);
    EXPECT_FALSE(reader.Owner());
    owner.MarkReady();
    EXPECT_TRUE(reader.Ready());
}

TEST_F(Cache2BufferTest, ConcurrentPreallocAndDemandElectExactlyOneOwner)
{
    constexpr size_t kThreads{16};
    constexpr size_t kRounds{64};

    for (size_t round = 0; round < kRounds; ++round) {
        auto block = MakeBlockId(static_cast<uint32_t>(round + 100));
        std::atomic<bool> start{false};
        std::atomic<size_t> owners{0};
        std::atomic<size_t> valid{0};
        std::thread prealloc([&] {
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            buffer_.Prealloc(block, 0);
        });
        std::vector<std::thread> loads;
        loads.reserve(kThreads);
        for (size_t i = 0; i < kThreads; ++i) {
            loads.emplace_back([&] {
                while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
                auto handle = buffer_.Get(block, 0);
                if (!handle) { return; }
                valid.fetch_add(1, std::memory_order_relaxed);
                if (handle.Owner()) {
                    owners.fetch_add(1, std::memory_order_relaxed);
                    handle.MarkReady();
                }
            });
        }
        start.store(true, std::memory_order_release);
        prealloc.join();
        for (auto& load : loads) { load.join(); }
        EXPECT_EQ(valid.load(std::memory_order_relaxed), kThreads) << "round " << round;
        EXPECT_EQ(owners.load(std::memory_order_relaxed), 1) << "round " << round;
    }
}

TEST_F(Cache2BufferTest, ExistDoesNotStealPreallocatedOwner)
{
    auto block = MakeBlockId(2);
    buffer_.Prealloc(block, 0);
    ASSERT_TRUE(buffer_.Exist(block, 0));
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 0);

    auto owner = buffer_.Get(block, 0);
    ASSERT_TRUE(owner);
    EXPECT_TRUE(owner.Owner());
    owner.MarkReady();
}

TEST_F(Cache2BufferTest, AbandonedOwnerPublishesFailureAndCanRetry)
{
    auto block = MakeBlockId(3);
    auto owner = buffer_.Get(block, 0);
    auto reader = buffer_.Get(block, 0);
    ASSERT_TRUE(owner);
    ASSERT_TRUE(reader);
    EXPECT_TRUE(owner.Owner());
    EXPECT_FALSE(reader.Owner());

    owner = {};
    EXPECT_EQ(reader.GetState(), CtrlLayout::SlotMeta::State::Failed);
    reader = {};

    auto retry = buffer_.Get(block, 0);
    ASSERT_TRUE(retry);
    EXPECT_TRUE(retry.Owner());
    EXPECT_EQ(retry.GetState(), CtrlLayout::SlotMeta::State::Loading);
    retry.MarkReady();
    EXPECT_TRUE(retry.Ready());
}

TEST(Cache2BufferPartitionTest, ClockEvictsOnlyInsideLocalRankAndHonorsSecondChanceAndPins)
{
    constexpr size_t kRanks{2};
    constexpr size_t kSlotsPerRank{4};
    constexpr size_t kBuckets{16};
    constexpr size_t kLocks{8};
    auto bytes = CtrlLayout::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    Buffer buffer;
    BufferTestAccess::Init(buffer, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);
    BufferTestAccess::InitRank(buffer, 1);

    std::optional<Buffer::Handle> pinned;
    for (uint32_t value = 1; value <= kSlotsPerRank; ++value) {
        auto handle = buffer.Get(MakeBlockId(value), 0);
        ASSERT_TRUE(handle);
        EXPECT_GE(handle.SlotIndex(), kSlotsPerRank);
        EXPECT_LT(handle.SlotIndex(), kRanks * kSlotsPerRank);
        EXPECT_TRUE(handle.Owner());
        handle.MarkReady();
        if (value == 1) { pinned.emplace(std::move(handle)); }
    }

    auto replacement = buffer.Get(MakeBlockId(100), 0);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(replacement.SlotIndex(), kSlotsPerRank + 1);
    EXPECT_TRUE(replacement.Owner());
    replacement.MarkReady();
    EXPECT_FALSE(buffer.Exist(MakeBlockId(2), 0));

    ASSERT_TRUE(pinned.has_value());
    EXPECT_EQ(pinned->SlotIndex(), kSlotsPerRank);
    EXPECT_TRUE(pinned->Ready());
    EXPECT_TRUE(buffer.Exist(MakeBlockId(1), 0));
    for (size_t i = 0; i < kSlotsPerRank; ++i) {
        auto& meta = BufferTestAccess::Layout(buffer).SlotMetaArr()[i];
        EXPECT_EQ(meta.reference.load(std::memory_order_relaxed), 0);
        EXPECT_EQ(meta.hash.load(std::memory_order_relaxed), kInvalid);
    }

    pinned.reset();
    replacement = {};
    ::operator delete(memory, std::align_val_t{64});
}

TEST(Cache2BufferLockTest, CrossStripeMigrationRollsBackAndCanRetry)
{
    constexpr size_t kRanks{1};
    constexpr size_t kSlotsPerRank{1};
    constexpr size_t kBuckets{8};
    constexpr size_t kLocks{4};
    auto bytes = CtrlLayout::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    Buffer buffer;
    BufferTestAccess::Init(buffer, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);

    auto oldBlock = MakeBlockId(1);
    auto oldBucket = BufferTestAccess::BucketOf(buffer, oldBlock);
    auto& layout = BufferTestAccess::Layout(buffer);
    Detail::BlockId newBlock;
    bool foundDifferentStripe = false;
    for (uint32_t value = 2; value < 4096; ++value) {
        auto candidate = MakeBlockId(value);
        if (layout.LockOf(BufferTestAccess::BucketOf(buffer, candidate)) !=
            layout.LockOf(oldBucket)) {
            newBlock = candidate;
            foundDifferentStripe = true;
            break;
        }
    }
    ASSERT_TRUE(foundDifferentStripe);

    auto old = buffer.Get(oldBlock, 0);
    ASSERT_TRUE(old);
    old.MarkReady();
    auto slot = old.SlotIndex();
    old = {};
    layout.SlotMetaArr()[slot].accessed.store(0, std::memory_order_relaxed);

    auto* oldLock = layout.LockOf(oldBucket);
    oldLock->Lock();
    auto failed = BufferTestAccess::TryGet(buffer, newBlock, 0, 1);
    EXPECT_FALSE(failed);
    EXPECT_EQ(layout.SlotMetaArr()[slot].reference.load(std::memory_order_acquire), 0);
    EXPECT_EQ(layout.SlotMetaArr()[slot].hash.load(std::memory_order_acquire), oldBucket);
    EXPECT_EQ(BufferTestAccess::FindSlot(buffer, oldBlock, 0), slot);
    oldLock->Unlock();

    layout.SlotMetaArr()[slot].accessed.store(0, std::memory_order_relaxed);
    auto replacement = BufferTestAccess::TryGet(buffer, newBlock, 0, 1);
    ASSERT_TRUE(replacement);
    EXPECT_TRUE(replacement.Owner());
    EXPECT_EQ(replacement.SlotIndex(), slot);
    replacement.MarkReady();
    EXPECT_EQ(BufferTestAccess::FindSlot(buffer, oldBlock, 0), kInvalid);
    EXPECT_EQ(BufferTestAccess::FindSlot(buffer, newBlock, 0), slot);

    replacement = {};
    ::operator delete(memory, std::align_val_t{64});
}

TEST(Cache2BufferOptimisticTest, ReadyHitSucceedsWhileBucketStripeIsLocked)
{
    constexpr size_t kRanks{1};
    constexpr size_t kSlotsPerRank{4};
    constexpr size_t kBuckets{8};
    constexpr size_t kLocks{4};
    auto bytes = CtrlLayout::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    Buffer buffer;
    BufferTestAccess::Init(buffer, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);

    auto block = MakeBlockId(200);
    auto owner = buffer.Get(block, 0);
    ASSERT_TRUE(owner);
    owner.MarkReady();
    auto slot = owner.SlotIndex();
    owner = {};

    auto& layout = BufferTestAccess::Layout(buffer);
    auto* lock = layout.LockOf(BufferTestAccess::BucketOf(buffer, block));
    lock->Lock();
    auto hit = BufferTestAccess::TryGet(buffer, block, 0, 1);
    auto valid = static_cast<bool>(hit);
    if (valid) {
        EXPECT_FALSE(hit.Owner());
        EXPECT_TRUE(hit.Ready());
        EXPECT_EQ(hit.SlotIndex(), slot);
    }
    lock->Unlock();
    ASSERT_TRUE(valid);

    hit = {};
    ::operator delete(memory, std::align_val_t{64});
}

TEST(Cache2BufferOptimisticTest, PinnedHitPreventsSlotReconfigurationUntilRelease)
{
    constexpr size_t kRanks{1};
    constexpr size_t kSlotsPerRank{1};
    constexpr size_t kBuckets{8};
    constexpr size_t kLocks{4};
    auto bytes = CtrlLayout::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    Buffer buffer;
    BufferTestAccess::Init(buffer, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);

    auto oldBlock = MakeBlockId(300);
    auto newBlock = MakeBlockId(301);
    auto owner = buffer.Get(oldBlock, 0);
    ASSERT_TRUE(owner);
    owner.MarkReady();
    owner = {};

    auto pinned = BufferTestAccess::TryGet(buffer, oldBlock, 0, 1);
    ASSERT_TRUE(pinned);
    EXPECT_FALSE(pinned.Owner());
    auto& meta = BufferTestAccess::Layout(buffer).SlotMetaArr()[pinned.SlotIndex()];
    meta.accessed.store(0, std::memory_order_relaxed);
    auto blocked = BufferTestAccess::TryGet(buffer, newBlock, 0, 2);
    EXPECT_FALSE(blocked);
    EXPECT_NE(BufferTestAccess::FindSlot(buffer, oldBlock, 0), kInvalid);
    EXPECT_EQ(BufferTestAccess::FindSlot(buffer, newBlock, 0), kInvalid);

    auto slot = pinned.SlotIndex();
    pinned = {};
    meta.accessed.store(0, std::memory_order_relaxed);
    auto replacement = BufferTestAccess::TryGet(buffer, newBlock, 0, 1);
    ASSERT_TRUE(replacement);
    EXPECT_TRUE(replacement.Owner());
    EXPECT_EQ(replacement.SlotIndex(), slot);
    replacement.MarkReady();
    EXPECT_EQ(BufferTestAccess::FindSlot(buffer, oldBlock, 0), kInvalid);
    EXPECT_EQ(BufferTestAccess::FindSlot(buffer, newBlock, 0), slot);

    replacement = {};
    ::operator delete(memory, std::align_val_t{64});
}

}  // namespace
}  // namespace UC::Cache2
