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
#include "cache/control_ops.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "cache/cache_buffer.h"
#include "cache/policy/replacement_policy_registry.h"
#include "cache/shm_layout.h"
#include "status/status.h"
#include "type/block.h"

namespace UC::Store {
namespace {

BlockId MakeBlock(std::size_t id)
{
    BlockId blockId;
    blockId.bytes[0] = static_cast<std::byte>(id);
    return blockId;
}

std::unique_ptr<ReplacementPolicy> MakePolicy(const std::string& name = "fifo")
{
    auto created = ReplacementPolicyRegistry::Instance().Create(name);
    return created ? std::unique_ptr<ReplacementPolicy>(*created) : nullptr;
}

ControlRequest MakeRequest(ControlOp op)
{
    ControlRequest request;
    request.op = op;
    request.seq = 7;
    return request;
}

class ControlOpsTest : public ::testing::Test {
protected:
    CacheBuffer buffer_;
};

TEST_F(ControlOpsTest, PrepareOffloadAnswersWithPlanBody)
{
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    const std::vector<BlockId> blocks{MakeBlock(1), MakeBlock(2)};
    ASSERT_TRUE(buffer_.Reserve(blocks, RequestContext{}));
    buffer_.Commit(blocks, true);

    const ControlResponse response =
        ExecuteControl(buffer_, MakeRequest(ControlOp::PrepareOffload));
    EXPECT_TRUE(response.ok);
    EXPECT_EQ(response.seq, std::uint32_t{7});  // the request's seq, echoed
    EXPECT_EQ(response.keys, blocks);           // planned keys, in conclusion order
    EXPECT_TRUE(response.hits.empty());         // the plan body is single-choice
    EXPECT_TRUE(response.slots.empty());
    const PlanPayload plan = DecodePlanPayload(response.tokenPayload);
    EXPECT_EQ(plan.batchId, std::uint64_t{1});  // minted from 1; 0 is reserved
    EXPECT_EQ(plan.slots, (std::vector<std::uint32_t>{0, 1}));
}

TEST_F(ControlOpsTest, PrepareOffloadIdleAnswersWithNoBody)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const ControlResponse response =
        ExecuteControl(buffer_, MakeRequest(ControlOp::PrepareOffload));
    EXPECT_TRUE(response.ok);  // idle is a success without a plan
    EXPECT_TRUE(response.keys.empty());
    EXPECT_TRUE(response.tokenPayload.empty());
    EXPECT_TRUE(response.hits.empty());
    EXPECT_TRUE(response.slots.empty());
}

TEST_F(ControlOpsTest, CompleteOffloadReleasesPinsByHandle)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId block = MakeBlock(1);
    ASSERT_TRUE(buffer_.Reserve({block}, RequestContext{}));
    buffer_.Commit({block}, true);

    const ControlResponse minted = ExecuteControl(buffer_, MakeRequest(ControlOp::PrepareOffload));
    ASSERT_TRUE(minted.ok);
    const std::uint64_t handle = DecodePlanPayload(minted.tokenPayload).batchId;
    ASSERT_NE(handle, std::uint64_t{0});

    // The receipt settles by handle alone; the echoed keys are informational.
    ControlRequest receipt = MakeRequest(ControlOp::CompleteOffload);
    receipt.planId = handle;
    receipt.blockIds = minted.keys;
    const ControlResponse done = ExecuteControl(buffer_, receipt);
    EXPECT_TRUE(done.ok);  // a void op: success with no body
    EXPECT_TRUE(done.hits.empty());
    EXPECT_TRUE(done.slots.empty());
    EXPECT_TRUE(done.keys.empty());
    EXPECT_TRUE(done.tokenPayload.empty());

    // The pin is released: the pool's only slot is evictable again.
    auto admitted = buffer_.Reserve({MakeBlock(2)}, RequestContext{});
    ASSERT_TRUE(admitted);

    // An unknown handle is a receipt-shaped no-op, not an error.
    ControlRequest stray = MakeRequest(ControlOp::CompleteOffload);
    stray.planId = 99;
    EXPECT_TRUE(ExecuteControl(buffer_, stray).ok);
}

TEST_F(ControlOpsTest, PlanOutcomeFromRecoversThePlanToken)
{
    ControlResponse response;
    response.ok = true;
    const std::vector<BlockId> blocks{MakeBlock(1), MakeBlock(2)};
    response.keys = blocks;
    response.tokenPayload = EncodePlanPayload(PlanPayload{
        std::uint64_t{5},
        {7, 9}
    });

    auto token = PlanOutcomeFrom(std::move(response));
    ASSERT_TRUE(token);
    EXPECT_EQ(token.Value().keys, blocks);  // moved out of the response
    const PlanPayload plan = DecodePlanPayload(token.Value().payload);
    EXPECT_EQ(plan.batchId, std::uint64_t{5});
    EXPECT_EQ(plan.slots, (std::vector<std::uint32_t>{7, 9}));

    // The idle plan is recovered as the empty token.
    ControlResponse idle;
    idle.ok = true;
    auto empty = PlanOutcomeFrom(std::move(idle));
    ASSERT_TRUE(empty);
    EXPECT_TRUE(empty.Value().keys.empty());
    EXPECT_TRUE(empty.Value().payload.empty());
}

TEST_F(ControlOpsTest, PlanOutcomeFromRejectsDriftAndErrors)
{
    // The slot count disagrees with the planned keys.
    ControlResponse drift;
    drift.ok = true;
    drift.keys = {MakeBlock(1), MakeBlock(2)};
    drift.tokenPayload = EncodePlanPayload(PlanPayload{std::uint64_t{5}, {7}});
    auto mismatched = PlanOutcomeFrom(std::move(drift));
    ASSERT_FALSE(mismatched);
    EXPECT_EQ(mismatched.Error(), Status::General());

    // The idle plan must not carry a stray payload.
    ControlResponse stray;
    stray.ok = true;
    stray.tokenPayload = EncodePlanPayload(PlanPayload{std::uint64_t{5}, {}});
    auto carried = PlanOutcomeFrom(std::move(stray));
    ASSERT_FALSE(carried);
    EXPECT_EQ(carried.Error(), Status::General());

    // Non-empty keys need a minted handle: 0 is the reserved no-plan sentinel.
    ControlResponse reserved;
    reserved.ok = true;
    reserved.keys = {MakeBlock(1), MakeBlock(2)};
    reserved.tokenPayload = EncodePlanPayload(PlanPayload{
        std::uint64_t{0},
        {7, 9}
    });
    auto sentinel = PlanOutcomeFrom(std::move(reserved));
    ASSERT_FALSE(sentinel);
    EXPECT_EQ(sentinel.Error(), Status::General());

    // A failed response passes its error through, recoded from the wire.
    ControlResponse failed;
    failed.ok = false;
    failed.statusCode = Status::NotFound().Underlying();
    failed.message = "not present";
    auto error = PlanOutcomeFrom(std::move(failed));
    ASSERT_FALSE(error);
    EXPECT_EQ(error.Error(), Status::NotFound());
}

}  // namespace
}  // namespace UC::Store
