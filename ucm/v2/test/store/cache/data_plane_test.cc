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
#include "cache/data_plane.h"
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
#include "fixtures/fault_dma_engine.h"
#include "status/status.h"
#include "type/block.h"
#include "type/token.h"

namespace UC::Store {
namespace {

constexpr std::size_t kBlockSize = 1 << 20; /* 1 MiB slots; the pool rounds to 4 MiB */
constexpr std::uint32_t kSlots = 4;

BlockId MakeBlock(std::uint64_t tag)
{
    BlockId id;
    std::memcpy(id.bytes.data(), "dpt", 3);
    std::memcpy(id.bytes.data() + 8, &tag, sizeof(tag));
    return id;
}

/** Fills len bytes with the block's deterministic pattern: a byte stream
 *  out of an LCG seeded from the first eight bytes of the block's id. */
void FillPattern(const BlockId& id, void* out, std::size_t len)
{
    std::uint64_t state = 0;
    std::memcpy(&state, id.bytes.data(), sizeof(state));
    auto* cursor = static_cast<std::byte*>(out);
    for (std::size_t i = 0; i < len; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        std::memcpy(cursor + i, &state, 1);
    }
}

/** Tells whether the buffer still carries the block's pattern. */
bool MatchesPattern(const BlockId& id, const void* in, std::size_t len)
{
    std::vector<std::byte> expected(len);
    FillPattern(id, expected.data(), len);
    return std::memcmp(expected.data(), in, len) == 0;
}

CacheConfig MakeConfig(const std::string& dmaEngine = {})
{
    CacheConfig config;
    config.uniqueId = "data-plane-test";
    config.llmRole = "worker";
    config.nWorker = 1;
    config.deviceId = 0;
    config.blockSize = kBlockSize;
    config.blockTotalNumber = kSlots;
    config.dmaEngine = dmaEngine;
    return config;
}

/** Whole-slot, single-pointer payloads over the given buffer. */
JobDescriptor MakeJob(const std::vector<BlockId>& ids, std::vector<std::byte>& buffer)
{
    JobDescriptor job;
    job.blockIds = ids;
    job.payloads.resize(ids.size());
    for (std::size_t i = 0; i < ids.size(); i++) {
        BlockPayload& payload = job.payloads[i];
        payload.pointers.push_back(buffer.data() + i * kBlockSize);
        payload.sizes.push_back(kBlockSize);
        payload.offsets.push_back(0);
    }
    return job;
}

AddressedToken MakeToken(const std::vector<std::uint32_t>& slots, const std::vector<BlockId>& ids)
{
    AddressedToken token;
    token.keys = ids;
    token.payload = EncodeTokenPayload(slots);
    return token;
}

class DataPlaneTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        auto region = MemfdRegion::Create(kSlots, kBlockSize, false, "ucm-data-plane-test");
        ASSERT_TRUE(region.HasValue());
        auto plane = DataPlane::Create(MakeConfig(), region.Value().Mapping());
        ASSERT_TRUE(plane.HasValue());
        /* The plane dies before the region, so the engine's drain-on-destroy
         * sees a live mapping. */
        region_ = std::move(region.Value());
        plane_ = std::move(plane.Value());
    }

    MemfdRegion region_;
    std::unique_ptr<DataPlane> plane_;
};

/*== Shape violations: rejected with nothing started ==*/

TEST_F(DataPlaneTest, ShapeViolationsAnswerInvalidParam)
{
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    std::vector<std::byte> buffer(2 * kBlockSize);

    /* An empty job is malformed. */
    EXPECT_EQ(plane_->Dump(1, JobDescriptor{}, AddressedToken{}).Underlying(),
              Status::InvalidParam().Underlying());

    /* Block ids and payloads disagree. */
    JobDescriptor shortJob;
    shortJob.blockIds = ids;
    shortJob.payloads.resize(1);
    shortJob.payloads[0].pointers.push_back(buffer.data());
    shortJob.payloads[0].sizes.push_back(kBlockSize);
    shortJob.payloads[0].offsets.push_back(0);
    EXPECT_EQ(plane_->Dump(2, shortJob, MakeToken({0, 1}, ids)).Underlying(),
              Status::InvalidParam().Underlying());

    /* The token's slot count disagrees with the job. */
    EXPECT_EQ(plane_->Dump(3, MakeJob(ids, buffer), MakeToken({0}, ids)).Underlying(),
              Status::InvalidParam().Underlying());

    /* A block carrying no segments. */
    JobDescriptor emptySegment = MakeJob(ids, buffer);
    emptySegment.payloads[1] = BlockPayload{};
    EXPECT_EQ(plane_->Dump(4, emptySegment, MakeToken({0, 1}, ids)).Underlying(),
              Status::InvalidParam().Underlying());

    /* A block with ragged payload columns. */
    JobDescriptor ragged = MakeJob(ids, buffer);
    ragged.payloads[0].sizes.push_back(kBlockSize);
    EXPECT_EQ(plane_->Dump(5, ragged, MakeToken({0, 1}, ids)).Underlying(),
              Status::InvalidParam().Underlying());

    /* Nothing started: the completion log stays empty. */
    EXPECT_TRUE(plane_->GetFinished().empty());
}

TEST_F(DataPlaneTest, SegmentBeyondBlockLengthIsRejected)
{
    const std::vector<BlockId> ids{MakeBlock(1)};
    std::vector<std::byte> buffer(kBlockSize);

    /* A segment longer than the block cannot fit its slot. */
    JobDescriptor oversized = MakeJob(ids, buffer);
    oversized.payloads[0].sizes[0] = kBlockSize + 1;
    const Status st = plane_->Dump(1, oversized, MakeToken({0}, ids));
    EXPECT_EQ(st.Underlying(), Status::InvalidParam().Underlying());

    /* A segment whose offset pushes it past the slot bounds fits nowhere
     * either, even though its size alone stays within the block length. */
    JobDescriptor overhang = MakeJob(ids, buffer);
    overhang.payloads[0].offsets[0] = kBlockSize - 1;
    overhang.payloads[0].sizes[0] = 2;
    const Status st2 = plane_->Dump(2, overhang, MakeToken({0}, ids));
    EXPECT_EQ(st2.Underlying(), Status::InvalidParam().Underlying());

    EXPECT_TRUE(plane_->GetFinished().empty());
}

/*== The default engine round-trips bytes and completions ==*/

TEST_F(DataPlaneTest, RoundTripThroughTheDefaultEngine)
{
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    std::vector<std::byte> source(2 * kBlockSize);
    FillPattern(ids[0], source.data(), kBlockSize);
    FillPattern(ids[1], source.data() + kBlockSize, kBlockSize);
    EXPECT_EQ(plane_->Dump(1, MakeJob(ids, source), MakeToken({0, 1}, ids)).Underlying(), 0);

    std::vector<std::byte> back(2 * kBlockSize);
    std::memset(back.data(), 0xA5, back.size());
    EXPECT_EQ(plane_->Load(2, MakeJob(ids, back), MakeToken({0, 1}, ids)).Underlying(), 0);
    EXPECT_EQ(plane_->Wait(2).Underlying(), 0);
    EXPECT_TRUE(MatchesPattern(ids[0], back.data(), kBlockSize));
    EXPECT_TRUE(MatchesPattern(ids[1], back.data() + kBlockSize, kBlockSize));

    /* Exactly-once: both completions were consumed by Wait, so the log is
     * empty and re-waiting answers invalid. */
    EXPECT_EQ(plane_->Wait(1).Underlying(), 0);
    EXPECT_TRUE(plane_->GetFinished().empty());
    EXPECT_EQ(plane_->Wait(1).Underlying(), Status::InvalidParam().Underlying());
    EXPECT_EQ(plane_->Wait(2).Underlying(), Status::InvalidParam().Underlying());
}

TEST_F(DataPlaneTest, MultiSegmentJobRoundTripsWithOffsets)
{
    const std::vector<BlockId> ids{MakeBlock(1)};
    const std::size_t half = kBlockSize / 2;

    /* One block carried as two half-block segments at different offsets,
     * riding a non-zero prerequisite handle end to end: Dump and Load each
     * flatten the job into one engine batch, pairing every segment with
     * its slot address plus offset. */
    std::vector<std::byte> source(kBlockSize);
    FillPattern(ids[0], source.data(), kBlockSize);
    JobDescriptor dump;
    dump.blockIds = ids;
    dump.payloads.resize(1);
    dump.prerequisiteHandle = 0x1234;
    dump.payloads[0].pointers.push_back(source.data());
    dump.payloads[0].pointers.push_back(source.data() + half);
    dump.payloads[0].sizes.push_back(half);
    dump.payloads[0].sizes.push_back(half);
    dump.payloads[0].offsets.push_back(0);
    dump.payloads[0].offsets.push_back(half);
    EXPECT_EQ(plane_->Dump(1, dump, MakeToken({0}, ids)).Underlying(), 0);

    std::vector<std::byte> back(kBlockSize);
    std::memset(back.data(), 0xA5, back.size());
    JobDescriptor reload;
    reload.blockIds = ids;
    reload.payloads.resize(1);
    reload.prerequisiteHandle = 0x1234;
    reload.payloads[0].pointers.push_back(back.data());
    reload.payloads[0].pointers.push_back(back.data() + half);
    reload.payloads[0].sizes.push_back(half);
    reload.payloads[0].sizes.push_back(half);
    reload.payloads[0].offsets.push_back(0);
    reload.payloads[0].offsets.push_back(half);
    EXPECT_EQ(plane_->Load(2, reload, MakeToken({0}, ids)).Underlying(), 0);
    EXPECT_EQ(plane_->Wait(2).Underlying(), 0);
    EXPECT_TRUE(MatchesPattern(ids[0], back.data(), kBlockSize));

    /* The prerequisite handle rode along without disturbing the batch:
     * job 1 completes too, and its completion is consumed exactly once. */
    EXPECT_EQ(plane_->Wait(1).Underlying(), 0);
    EXPECT_TRUE(plane_->GetFinished().empty());
}

/*== Engine selection and failure propagation ==*/

TEST_F(DataPlaneTest, UnknownEngineNameFailsCreate)
{
    auto region = MemfdRegion::Create(kSlots, kBlockSize, false, "ucm-data-plane-badname");
    ASSERT_TRUE(region.HasValue());
    auto plane = DataPlane::Create(MakeConfig("nope"), region.Value().Mapping());
    ASSERT_FALSE(plane.HasValue());
    EXPECT_EQ(plane.Error().Underlying(), Status::NotFound().Underlying());
}

TEST_F(DataPlaneTest, FaultEngineFailuresPropagate)
{
    auto region = MemfdRegion::Create(kSlots, kBlockSize, false, "ucm-data-plane-fault");
    ASSERT_TRUE(region.HasValue());
    auto created = DataPlane::Create(MakeConfig("gdr"), region.Value().Mapping());
    ASSERT_TRUE(created.HasValue());
    /* The plane dies before the region, so the engine's drain-on-destroy
     * sees a live mapping. */
    MemfdRegion regionHolder = std::move(region.Value());
    std::unique_ptr<DataPlane> plane = std::move(created.Value());

    const std::vector<BlockId> ids{MakeBlock(1)};
    std::vector<std::byte> buffer(kBlockSize);
    EXPECT_EQ(plane->Dump(1, MakeJob(ids, buffer), MakeToken({0}, ids)).Underlying(), 0);

    /* The submission succeeds; the completion carries the failure, and it
     * is reported exactly once. */
    const std::vector<std::pair<std::size_t, bool>> finished = plane->GetFinished();
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].first, 1U);
    EXPECT_FALSE(finished[0].second);
    EXPECT_TRUE(plane->GetFinished().empty());

    /* The Wait path surfaces the same failure as an error Status. */
    EXPECT_EQ(plane->Load(2, MakeJob(ids, buffer), MakeToken({0}, ids)).Underlying(), 0);
    EXPECT_NE(plane->Wait(2).Underlying(), 0);
    EXPECT_TRUE(plane->GetFinished().empty());
}

/*== The flattened batch carries the per-block groups ==*/

TEST_F(DataPlaneTest, BatchCarriesPerBlockSegmentGroups)
{
    auto region = MemfdRegion::Create(kSlots, kBlockSize, false, "ucm-data-plane-groups");
    ASSERT_TRUE(region.HasValue());
    auto created = DataPlane::Create(MakeConfig("gdr"), region.Value().Mapping());
    ASSERT_TRUE(created.HasValue());
    /* The plane dies before the region, so the engine's drain-on-destroy
     * sees a live mapping. */
    MemfdRegion regionHolder = std::move(region.Value());
    std::unique_ptr<DataPlane> plane = std::move(created.Value());

    /* Three blocks of two segments each flatten into one columnar batch
     * whose groups mirror the blocks: {2, 2, 2}. */
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2), MakeBlock(3)};
    const std::size_t half = kBlockSize / 2;
    std::vector<std::byte> buffer(3 * kBlockSize);
    JobDescriptor job;
    job.blockIds = ids;
    job.payloads.resize(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        job.payloads[i].pointers.push_back(buffer.data() + i * kBlockSize);
        job.payloads[i].pointers.push_back(buffer.data() + i * kBlockSize + half);
        job.payloads[i].sizes.push_back(half);
        job.payloads[i].sizes.push_back(half);
        job.payloads[i].offsets.push_back(0);
        job.payloads[i].offsets.push_back(half);
    }

    Transfer::ResetCapturedDmaBatches();
    EXPECT_EQ(plane->Dump(7, job, MakeToken({0, 1, 2}, ids)).Underlying(), 0);

    const std::vector<Transfer::CapturedDmaBatch>& captured = Transfer::CapturedDmaBatches();
    ASSERT_EQ(captured.size(), 1U);
    EXPECT_EQ(captured[0].jobId, 7U);
    EXPECT_FALSE(captured[0].hostToDevice); /* Dump moves into the pool: D2H */
    ASSERT_EQ(captured[0].batch.nSegmentPerGroup.size(), 3U);
    EXPECT_EQ(captured[0].batch.nSegmentPerGroup[0], 2U);
    EXPECT_EQ(captured[0].batch.nSegmentPerGroup[1], 2U);
    EXPECT_EQ(captured[0].batch.nSegmentPerGroup[2], 2U);
    ASSERT_EQ(captured[0].batch.sizes.size(), 6U);
    EXPECT_EQ(captured[0].batch.sources[0], buffer.data());
    EXPECT_EQ(captured[0].batch.sources[1], buffer.data() + half);
    EXPECT_EQ(captured[0].batch.prerequisite, std::uintptr_t{0});
}

}  // namespace
}  // namespace UC::Store
