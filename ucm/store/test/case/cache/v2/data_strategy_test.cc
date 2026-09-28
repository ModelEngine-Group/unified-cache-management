/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "cache/v2/data_strategy.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include "cache/v2/ctrl_layout.h"
#include "ipc/mem_fd.h"

namespace UC::Cache2 {

struct DataStrategyTestAccess {
    static size_t TotalSize(size_t bucketCount, size_t lockCount, size_t slotCount)
    {
        return CtrlLayout::TotalSize(bucketCount, lockCount, slotCount);
    }
    static size_t RecommendBucketCount(size_t slotCount)
    {
        return CtrlLayout::RecommendBucketCount(slotCount);
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
};

namespace {

constexpr size_t kRankCount{2};
constexpr size_t kSlotsPerRank{4};
constexpr size_t kSlotSize{4096};

class DataStrategyTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        static std::atomic<size_t> sequence{0};
        uniqueId_ =
            "ds_test_" + std::to_string(::getpid()) + "_" + std::to_string(sequence.fetch_add(1));
        const size_t slotCount = kRankCount * kSlotsPerRank;
        const size_t bucketCount = DataStrategyTestAccess::RecommendBucketCount(slotCount);
        const size_t lockCount = DataStrategyTestAccess::LockStripeCount(bucketCount);
        auto s = ctrlMem_.Create("ucm_cache2_ds_test", DataStrategyTestAccess::TotalSize(
                                                           bucketCount, lockCount, slotCount));
        ASSERT_TRUE(s.Success()) << s.ToString();
        DataStrategyTestAccess::Bind(layout_, ctrlMem_.Addr(), kRankCount, kSlotsPerRank,
                                     bucketCount, lockCount);
        layout_.InitHeader(kSlotSize);
        layout_.MarkReady();
    }

    static void FillSlot(void* slot, uint8_t value) { std::memset(slot, value, kSlotSize); }

    static bool SlotIsFilled(const void* slot, uint8_t value)
    {
        const auto* bytes = static_cast<const uint8_t*>(slot);
        for (size_t i = 0; i < kSlotSize; ++i) {
            if (bytes[i] != value) { return false; }
        }
        return true;
    }

    std::string ShmPath(size_t rank) const
    {
        return "/dev/shm/ucm_cache2_" + uniqueId_ + "_data_" + std::to_string(rank);
    }

    /* Every rank polls its peers, so all participants must set up concurrently. */
    std::vector<Status> SetupAllRanks(std::vector<DataStrategy>& strategies)
    {
        std::vector<Status> statuses(strategies.size(), Status::OK());
        std::vector<std::thread> threads;
        for (size_t rank = 0; rank < strategies.size(); ++rank) {
            threads.emplace_back([this, &strategies, &statuses, rank] {
                statuses[rank] =
                    strategies[rank].Setup(layout_, uniqueId_, static_cast<int32_t>(rank), rank,
                                           kSlotSize, kSlotsPerRank, /*timeoutMs=*/5000);
            });
        }
        for (auto& thread : threads) { thread.join(); }
        return statuses;
    }

    MemFd ctrlMem_;
    CtrlLayout layout_;
    std::string uniqueId_;
};

TEST_F(DataStrategyTest, SetupPublishesAndImportsAllRanks)
{
    std::vector<DataStrategy> ranks(kRankCount);
    const auto statuses = SetupAllRanks(ranks);
    for (const auto& s : statuses) { ASSERT_TRUE(s.Success()) << s.ToString(); }
    auto& owner = ranks[0];
    /* Only referenced by the posix-shm assertions. */
    [[maybe_unused]] auto& peer = ranks[1];

    const size_t totalSlots = kRankCount * kSlotsPerRank;
    for (size_t slot = 0; slot < totalSlots; ++slot) {
#if UCM_RUNTIME_ASCEND_HAL
        EXPECT_EQ(slot < kSlotsPerRank, owner.HostAccessibleOf(slot));
#else
        /* Posix backend: every slot is host-accessible from every rank. */
        EXPECT_TRUE(owner.HostAccessibleOf(slot));
        EXPECT_TRUE(peer.HostAccessibleOf(slot));
        EXPECT_NE(nullptr, owner.DataAt(slot));
        EXPECT_NE(nullptr, peer.DataAt(slot));
        EXPECT_EQ(nullptr, owner.DeviceDataAt(slot));
        EXPECT_EQ(nullptr, peer.DeviceDataAt(slot));
#endif
    }
    /* Slot addresses advance by slotSize and stay inside their rank segment. */
    EXPECT_EQ(kSlotSize, static_cast<size_t>(static_cast<std::byte*>(owner.DataAt(1)) -
                                             static_cast<std::byte*>(owner.DataAt(0))));
}

TEST_F(DataStrategyTest, SegmentsAreSharedAcrossRanks)
{
#if UCM_RUNTIME_ASCEND_HAL
    GTEST_SKIP() << "cross-rank host access requires the posix backend";
#else
    std::vector<DataStrategy> ranks(kRankCount);
    const auto statuses = SetupAllRanks(ranks);
    for (const auto& s : statuses) { ASSERT_TRUE(s.Success()) << s.ToString(); }
    auto& owner = ranks[0];
    auto& peer = ranks[1];

    /* Setup released the segment names; the mappings live on. */
    for (size_t rank = 0; rank < kRankCount; ++rank) {
        EXPECT_NE(0, ::access(ShmPath(rank).c_str(), F_OK));
    }
    /* Rank 0 writes its own slot; rank 1 observes it through its mapping. */
    FillSlot(owner.DataAt(0), 0x5a);
    EXPECT_TRUE(SlotIsFilled(peer.DataAt(0), 0x5a));
    /* Rank 1 writes its own slot; rank 0 observes it through its mapping. */
    FillSlot(peer.DataAt(kSlotsPerRank), 0xa5);
    EXPECT_TRUE(SlotIsFilled(owner.DataAt(kSlotsPerRank), 0xa5));
#endif
}

TEST_F(DataStrategyTest, SecondSetupIsRejected)
{
    std::vector<DataStrategy> ranks(kRankCount);
    const auto statuses = SetupAllRanks(ranks);
    for (const auto& s : statuses) { ASSERT_TRUE(s.Success()) << s.ToString(); }
    const auto s = ranks[0].Setup(layout_, uniqueId_, 0, 0, kSlotSize, kSlotsPerRank, 1000);
    EXPECT_TRUE(s.Failure());
}

TEST_F(DataStrategyTest, InvalidGeometryIsRejected)
{
    DataStrategy strategy;
    /* Rank out of range. */
    auto s = strategy.Setup(layout_, uniqueId_, 0, kRankCount, kSlotSize, kSlotsPerRank, 100);
    EXPECT_TRUE(s.Failure());
    /* Empty unique id. */
    s = strategy.Setup(layout_, "", 0, 0, kSlotSize, kSlotsPerRank, 100);
    EXPECT_TRUE(s.Failure());
}

TEST_F(DataStrategyTest, MissingPeerTimesOut)
{
    DataStrategy peer;
    /* Rank 0 never publishes; rank 1 must time out. */
    auto s = peer.Setup(layout_, uniqueId_, 1, /*myRank=*/1, kSlotSize, kSlotsPerRank, 50);
    ASSERT_TRUE(s.Failure());
    EXPECT_EQ(Status::Timeout().Underlying(), s.Underlying());
    /* Failed setup releases the segment: no host or device address remains. */
    EXPECT_FALSE(peer.HostAccessibleOf(kSlotsPerRank));
    EXPECT_EQ(nullptr, peer.DataAt(kSlotsPerRank));
    EXPECT_EQ(nullptr, peer.DeviceDataAt(kSlotsPerRank));
}

TEST_F(DataStrategyTest, PeerHandleMismatchIsRejected)
{
#if UCM_RUNTIME_ASCEND_HAL
    GTEST_SKIP() << "handle validation is a posix backend capability";
#else
    CtrlLayout::RankDataDesc poisoned;
    poisoned.handle.store(0x1234567812345678ULL, std::memory_order_relaxed);
    ASSERT_TRUE(layout_.SetRankDesc(0, poisoned).Success());

    DataStrategy peer;
    auto s = peer.Setup(layout_, uniqueId_, 1, /*myRank=*/1, kSlotSize, kSlotsPerRank, 100);
    ASSERT_TRUE(s.Failure());
    EXPECT_EQ(Status::InvalidParam().Underlying(), s.Underlying());
#endif
}

TEST_F(DataStrategyTest, PeerPollingWaitsForDelayedPublish)
{
    DataStrategy peer;
    constexpr size_t delayMs = 200;
    std::thread publisher([this, delayMs] {
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
        DataStrategy owner;
        auto s = owner.Setup(layout_, uniqueId_, 0, /*myRank=*/0, kSlotSize, kSlotsPerRank, 1000);
        ASSERT_TRUE(s.Success()) << s.ToString();
        /* Keep rank 0 alive while rank 1 polls. */
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    });
    auto s = peer.Setup(layout_, uniqueId_, 1, /*myRank=*/1, kSlotSize, kSlotsPerRank,
                        /*timeoutMs=*/5000);
    EXPECT_TRUE(s.Success()) << s.ToString();
    publisher.join();
}

TEST_F(DataStrategyTest, SegmentNamesAreReleasedAfterSetup)
{
#if UCM_RUNTIME_ASCEND_HAL
    GTEST_SKIP() << "shm lifecycle is a posix backend capability";
#else
    {
        std::vector<DataStrategy> ranks(kRankCount);
        const auto statuses = SetupAllRanks(ranks);
        for (const auto& s : statuses) { ASSERT_TRUE(s.Success()) << s.ToString(); }
        /* Names are released right after every rank finished mapping, not at
         * destruction; the mappings keep the segments usable. */
        EXPECT_NE(0, ::access(ShmPath(0).c_str(), F_OK));
        EXPECT_NE(0, ::access(ShmPath(1).c_str(), F_OK));
        EXPECT_NE(nullptr, ranks[0].DataAt(0));
        EXPECT_NE(nullptr, ranks[1].DataAt(kSlotsPerRank));
    }
    EXPECT_NE(0, ::access(ShmPath(0).c_str(), F_OK));
    EXPECT_NE(0, ::access(ShmPath(1).c_str(), F_OK));
#endif
}

}  // namespace

}  // namespace UC::Cache2
