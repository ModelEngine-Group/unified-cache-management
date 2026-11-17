/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include "cache/tier_relay.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "cache/cache_config.h"
#include "cache/shm_layout.h"
#include "cache/token_codec.h"
#include "fixtures/stub_lower_store.h"
#include "status/status.h"
#include "type/block.h"
#include "type/token.h"

namespace UC::Store {
namespace {

constexpr std::size_t kBlockSize = 1 << 20; /* 1 MiB slots; the pool rounds to 4 MiB */
constexpr std::uint32_t kSlots = 8;

BlockId MakeBlock(std::uint64_t tag)
{
    BlockId id;
    std::memcpy(id.bytes.data(), "trt", 3);
    std::memcpy(id.bytes.data() + 8, &tag, sizeof(tag));
    return id;
}

/** A plan token in the engine half's encoding: the receipt handle plus
 *  one source slot per key. */
AddressedToken MakePlan(const std::vector<BlockId>& ids, const std::vector<std::uint32_t>& slots,
                        std::uint64_t batchId = 1)
{
    AddressedToken plan;
    plan.keys = ids;
    plan.payload = EncodePlanPayload(PlanPayload{batchId, slots});
    return plan;
}

class TierRelayTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        auto region = MemfdRegion::Create(kSlots, kBlockSize, false, "ucm-tier-relay-test");
        ASSERT_TRUE(region.HasValue());
        region_ = std::move(region.Value());
    }

    CacheConfig MakeConfig(std::size_t tierBatchSize)
    {
        CacheConfig config;
        config.uniqueId = "tier-relay-test";
        config.llmRole = "worker";
        config.nWorker = 1;
        config.deviceId = 0;
        config.blockSize = kBlockSize;
        config.blockTotalNumber = kSlots;
        config.tierBatchSize = tierBatchSize;
        config.lowerStore = &lower_;
        return config;
    }

    std::unique_ptr<TierRelay> MakeRelay(std::size_t tierBatchSize)
    {
        auto relay = TierRelay::Create(MakeConfig(tierBatchSize), region_.Mapping());
        EXPECT_TRUE(relay.HasValue());
        return relay.HasValue() ? std::move(relay.Value()) : nullptr;
    }

    MemfdRegion region_;
    StubLowerStore lower_;
};

/*== Create: fail-closed wiring and geometry ==*/

TEST_F(TierRelayTest, CreateRejectsMissingWiringAndBadBatchSize)
{
    /* A relay without a wired lower tier has nothing to drive. */
    CacheConfig config = MakeConfig(4);
    config.lowerStore = nullptr;
    auto unwired = TierRelay::Create(config, region_.Mapping());
    ASSERT_FALSE(unwired.HasValue());
    EXPECT_EQ(unwired.Error(), Status::InvalidParam());

    /* Degenerate geometry is refused the same fail-closed way. */
    auto badBatch = TierRelay::Create(MakeConfig(0), region_.Mapping());
    ASSERT_FALSE(badBatch.HasValue());
    EXPECT_EQ(badBatch.Error(), Status::InvalidParam());

    config = MakeConfig(4);
    config.blockSize = 0;
    auto badBlock = TierRelay::Create(config, region_.Mapping());
    ASSERT_FALSE(badBlock.HasValue());
    EXPECT_EQ(badBlock.Error(), Status::InvalidParam());
}

/*== Submission: the admission protocol, driven per sub-batch ==*/

TEST_F(TierRelayTest, OffloadDrivesAllocDumpCommitPerSubBatch)
{
    auto relay = MakeRelay(4);
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2), MakeBlock(3)};
    const std::vector<std::uint32_t> slots{2, 5, 7};
    EXPECT_EQ(relay->Offload(9, MakePlan(ids, slots)).Underlying(), 0);

    /* One sub-batch: one Alloc of all three keys, one Dump whose sources
     * address each key's slot whole, no Commit before the completion. */
    ASSERT_EQ(lower_.allocs.size(), 1U);
    EXPECT_EQ(lower_.allocs[0], ids);
    ASSERT_EQ(lower_.dumps.size(), 1U);
    EXPECT_EQ(lower_.dumps[0].jobId, 1U); /* Sub ids are minted from 1. */
    EXPECT_EQ(lower_.dumps[0].keys, ids);
    ASSERT_EQ(lower_.dumps[0].sources.size(), 3U);
    for (std::size_t i = 0; i < slots.size(); ++i) {
        EXPECT_EQ(lower_.dumps[0].sources[i],
                  SlotAddress(region_.Mapping().Base(), slots[i], kBlockSize));
    }
    EXPECT_EQ(lower_.dumps[0].sizes,
              (std::vector<std::size_t>{kBlockSize, kBlockSize, kBlockSize}));
    EXPECT_TRUE(lower_.commits.empty());

    /* The sub-batch's success commits its reservation, then the outer
     * job surfaces exactly once. */
    lower_.Complete(1, true);
    const auto finished = relay->GetFinished();
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].first, 9U);
    EXPECT_TRUE(finished[0].second);
    ASSERT_EQ(lower_.commits.size(), 1U);
    EXPECT_EQ(lower_.commits[0].first, ids);
    EXPECT_TRUE(lower_.commits[0].second);
    EXPECT_TRUE(relay->GetFinished().empty());
}

TEST_F(TierRelayTest, PlansSplitAtTierBatchSize)
{
    auto relay = MakeRelay(2);
    std::vector<BlockId> ids;
    std::vector<std::uint32_t> slots;
    for (std::uint64_t i = 0; i < 5; ++i) {
        ids.push_back(MakeBlock(i + 1));
        slots.push_back(static_cast<std::uint32_t>(i));
    }
    EXPECT_EQ(relay->Offload(1, MakePlan(ids, slots)).Underlying(), 0);

    /* Five blocks over batches of two: three sub-batches of 2 + 2 + 1,
     * each its own reservation in plan order. */
    ASSERT_EQ(lower_.allocs.size(), 3U);
    EXPECT_EQ(lower_.allocs[0], (std::vector<BlockId>{ids[0], ids[1]}));
    EXPECT_EQ(lower_.allocs[1], (std::vector<BlockId>{ids[2], ids[3]}));
    EXPECT_EQ(lower_.allocs[2], (std::vector<BlockId>{ids[4]}));
    ASSERT_EQ(lower_.dumps.size(), 3U);
    EXPECT_EQ(lower_.dumps[2].jobId, 3U);
    EXPECT_EQ(lower_.dumps[2].sources[0], SlotAddress(region_.Mapping().Base(), 4, kBlockSize));

    /* Every sub-batch concluding true commits its own keys and yields
     * one verdict for the whole job. */
    for (std::size_t sub = 1; sub <= 3; ++sub) { lower_.Complete(sub, true); }
    const auto finished = relay->GetFinished();
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].first, 1U);
    EXPECT_TRUE(finished[0].second);
    ASSERT_EQ(lower_.commits.size(), 3U);
    EXPECT_EQ(lower_.commits[0].first, (std::vector<BlockId>{ids[0], ids[1]}));
    EXPECT_EQ(lower_.commits[1].first, (std::vector<BlockId>{ids[2], ids[3]}));
    EXPECT_EQ(lower_.commits[2].first, (std::vector<BlockId>{ids[4]}));
}

TEST_F(TierRelayTest, DroppedSubBatchFailsTheJobButOthersProceed)
{
    auto relay = MakeRelay(1);
    lower_.FailAllocNth(1, Status::NoSpace("stub is full"));
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2), MakeBlock(3)};
    EXPECT_EQ(relay->Offload(4, MakePlan(ids, {0, 1, 2})).Underlying(), 0);

    /* The first chunk's Alloc is refused: no Dump and no Commit for it,
     * while the other two fly. */
    ASSERT_EQ(lower_.allocs.size(), 3U);
    ASSERT_EQ(lower_.dumps.size(), 2U);
    EXPECT_EQ(lower_.dumps[0].keys, (std::vector<BlockId>{ids[1]}));
    EXPECT_EQ(lower_.dumps[1].keys, (std::vector<BlockId>{ids[2]}));
    EXPECT_TRUE(lower_.commits.empty());

    /* Both survivors succeed, yet the job fails on the dropped one. */
    lower_.Complete(1, true);
    lower_.Complete(2, true);
    const Status st = relay->Wait(4);
    EXPECT_EQ(st.Underlying(), Status::General().Underlying());
    EXPECT_NE(st.ToString().find("1 of 3"), std::string::npos);
    ASSERT_EQ(lower_.commits.size(), 2U);
    EXPECT_EQ(lower_.commits[0].first, (std::vector<BlockId>{ids[1]}));
    EXPECT_TRUE(lower_.commits[0].second);
    EXPECT_EQ(lower_.commits[1].first, (std::vector<BlockId>{ids[2]}));
    EXPECT_TRUE(lower_.commits[1].second);
}

TEST_F(TierRelayTest, DumpSubmitFailureReleasesTheReservation)
{
    auto relay = MakeRelay(4);
    lower_.FailDumpNth(1, Status::OsApiError("stub engine broke"));
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    EXPECT_EQ(relay->Offload(2, MakePlan(ids, {0, 3})).Underlying(), 0);

    /* Nothing flies, so the job is already concluded: the refused Dump's
     * reservation is released by a false Commit, and the verdict queues
     * without any lower-tier completion ever arriving. */
    ASSERT_EQ(lower_.commits.size(), 1U);
    EXPECT_EQ(lower_.commits[0].first, ids);
    EXPECT_FALSE(lower_.commits[0].second);
    const auto finished = relay->GetFinished();
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].first, 2U);
    EXPECT_FALSE(finished[0].second);
}

TEST_F(TierRelayTest, FailedTransferFailsTheJob)
{
    auto relay = MakeRelay(4);
    const std::vector<BlockId> ids{MakeBlock(7)};
    EXPECT_EQ(relay->Offload(1, MakePlan(ids, {4})).Underlying(), 0);

    /* A failed transfer concludes its reservation false and fails the
     * job. */
    lower_.Complete(1, false);
    const auto finished = relay->GetFinished();
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].first, 1U);
    EXPECT_FALSE(finished[0].second);
    ASSERT_EQ(lower_.commits.size(), 1U);
    EXPECT_EQ(lower_.commits[0].first, ids);
    EXPECT_FALSE(lower_.commits[0].second);
    EXPECT_TRUE(relay->GetFinished().empty());
}

/*== Completion discipline: exactly once, through either half ==*/

TEST_F(TierRelayTest, CompletionSurfacesExactlyOnce)
{
    auto relay = MakeRelay(4);
    const std::vector<BlockId> ids{MakeBlock(1)};
    EXPECT_EQ(relay->Offload(5, MakePlan(ids, {1})).Underlying(), 0);
    lower_.Complete(1, true);

    const auto finished = relay->GetFinished();
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].first, 5U);
    EXPECT_TRUE(finished[0].second);

    /* The verdict is consumed: the stream stays empty and the id answers
     * invalid from now on. */
    EXPECT_TRUE(relay->GetFinished().empty());
    EXPECT_EQ(relay->Wait(5).Underlying(), Status::InvalidParam().Underlying());
}

TEST_F(TierRelayTest, WaitConsumesWithoutGetFinished)
{
    auto relay = MakeRelay(4);
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    EXPECT_EQ(relay->Offload(3, MakePlan(ids, {0, 1})).Underlying(), 0);
    lower_.Complete(1, true);

    /* The queued lower-tier completion is consumed inside the Wait
     * itself; the verdict goes to the waiting caller and never into the
     * stream. */
    EXPECT_EQ(relay->Wait(3).Underlying(), 0);
    EXPECT_TRUE(relay->GetFinished().empty());
    EXPECT_EQ(relay->Wait(3).Underlying(), Status::InvalidParam().Underlying());
}

TEST_F(TierRelayTest, WaitPumpsThroughTheBlockingSubWait)
{
    auto relay = MakeRelay(1);
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    EXPECT_EQ(relay->Offload(6, MakePlan(ids, {0, 1})).Underlying(), 0);

    /* One sub-batch is already settled; the other settles only inside
     * the lower tier's own Wait, which the pump blocks on. */
    lower_.Complete(1, true);
    lower_.SettleOnWait(true);
    EXPECT_EQ(relay->Wait(6).Underlying(), 0);
    ASSERT_EQ(lower_.commits.size(), 2U);
    EXPECT_TRUE(lower_.commits[0].second);
    EXPECT_TRUE(lower_.commits[1].second);
    EXPECT_TRUE(relay->GetFinished().empty());
}

/*== Submission shape violations: rejected with nothing started ==*/

TEST_F(TierRelayTest, SubmitValidationsRejectWithNothingStarted)
{
    auto relay = MakeRelay(2);
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};

    /* A plan without a receipt handle and a plan whose slot count
     * disagrees with its keys are both refused before the lower tier is
     * touched. */
    EXPECT_EQ(relay->Offload(2, MakePlan(ids, {0, 1}, 0)).Underlying(),
              Status::InvalidParam().Underlying());
    EXPECT_EQ(relay->Offload(3, MakePlan(ids, {0})).Underlying(),
              Status::InvalidParam().Underlying());
    EXPECT_TRUE(lower_.allocs.empty());

    /* A duplicate in-flight id is refused whole; consumption frees it. */
    EXPECT_EQ(relay->Offload(4, MakePlan(ids, {0, 1})).Underlying(), 0);
    EXPECT_EQ(lower_.dumps.size(), 1U);
    EXPECT_EQ(relay->Offload(4, MakePlan(ids, {0, 1})).Underlying(),
              Status::InvalidParam().Underlying());
    lower_.Complete(1, true);
    ASSERT_EQ(relay->GetFinished().size(), 1U);
    EXPECT_EQ(relay->Offload(4, MakePlan(ids, {0, 1})).Underlying(), 0);
    EXPECT_EQ(lower_.dumps.size(), 2U);
}

/*== The empty claim: no keys, concluded at once ==*/

TEST_F(TierRelayTest, EmptyClaimConcludesImmediately)
{
    auto relay = MakeRelay(2);

    /* A claimed share with no keys concludes on submission, with nothing
     * driven in the lower tier. */
    EXPECT_EQ(relay->Offload(7, MakePlan({}, {}, 1)).Underlying(), 0);
    const auto finished = relay->GetFinished();
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].first, std::size_t{7});
    EXPECT_TRUE(finished[0].second);
    EXPECT_TRUE(lower_.allocs.empty());
    EXPECT_TRUE(lower_.dumps.empty());
    EXPECT_EQ(relay->Wait(7).Underlying(), Status::InvalidParam().Underlying());

    /* The idle plan carries no receipt handle: still refused. */
    EXPECT_EQ(relay->Offload(8, MakePlan({}, {}, 0)).Underlying(),
              Status::InvalidParam().Underlying());
}

/*== Ownership: in flight and unconsumed verdicts ==*/

TEST_F(TierRelayTest, OwnsCoversInFlightAndUnconsumedJobs)
{
    auto relay = MakeRelay(1);
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2), MakeBlock(3)};

    EXPECT_FALSE(relay->Owns(8));
    EXPECT_EQ(relay->Offload(8, MakePlan(ids, {0, 1, 2})).Underlying(), 0);
    EXPECT_TRUE(relay->Owns(8));

    /* Every sub-batch dropped at submit: the verdict queues without the
     * lower tier ever flying, and the id stays owned until consumed. */
    lower_.FailAllocNth(1, Status::NoSpace("stub is full"));
    EXPECT_EQ(relay->Offload(9, MakePlan({MakeBlock(4)}, {3})).Underlying(), 0);
    EXPECT_TRUE(relay->Owns(9));

    const auto finished = relay->GetFinished();
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].first, 9U);
    EXPECT_FALSE(finished[0].second);
    EXPECT_FALSE(relay->Owns(9));

    /* The first job stays owned until its own verdict is consumed. */
    for (std::size_t sub = 1; sub <= 3; ++sub) { lower_.Complete(sub, true); }
    EXPECT_EQ(relay->Wait(8).Underlying(), 0);
    EXPECT_FALSE(relay->Owns(8));
}

}  // namespace
}  // namespace UC::Store
