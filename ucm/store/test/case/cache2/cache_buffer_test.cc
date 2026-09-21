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
        buffer.timeoutMs_ = 1000;
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

}  // namespace
}  // namespace UC::Cache2
