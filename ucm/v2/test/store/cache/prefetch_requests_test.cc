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
#include "cache/prefetch_requests.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "cache/cache_buffer.h"
#include "cache/control_ops.h"
#include "cache/policy/replacement_policy_registry.h"
#include "cache/token_codec.h"
#include "fixtures/stub_lower_store.h"
#include "status/status.h"
#include "type/block.h"
#include "type/request.h"
#include "type/token.h"

namespace UC::Store {
namespace {

BlockId MakeBlock(std::size_t id)
{
    BlockId blockId;
    blockId.bytes[0] = static_cast<std::byte>(id);
    return blockId;
}

std::unique_ptr<ReplacementPolicy> MakePolicy()
{
    auto created = ReplacementPolicyRegistry::Instance().Create("fifo");
    return created ? std::unique_ptr<ReplacementPolicy>(*created) : nullptr;
}

/** Carries each request straight into the engine; the seq stays 0. */
class DirectEndpoint final : public ControlEndpoint {
public:
    explicit DirectEndpoint(CacheBuffer& buffer) : buffer_(buffer) {}

    Expected<ControlResponse> Execute(ControlRequest request) override
    {
        return ExecuteControl(buffer_, request);
    }

private:
    CacheBuffer& buffer_;
};

class PrefetchRequestsTest : public ::testing::Test {
protected:
    CacheBuffer buffer_;
    StubLowerStore lower_;
    DirectEndpoint endpoint_{buffer_};
    PrefetchRequests requests_{endpoint_, &lower_};

    void SetUpBuffer(std::size_t capacity)
    {
        ASSERT_TRUE(buffer_.Setup(capacity, MakePolicy(), false).Success());
    }
};

TEST_F(PrefetchRequestsTest, LookupAccumulatesProbedKeysAcrossCalls)
{
    SetUpBuffer(3);
    const BlockId a = MakeBlock(1);
    const BlockId b = MakeBlock(2);
    const BlockId c = MakeBlock(3);
    lower_.present.insert(b);

    auto first = requests_.Lookup("r", {a, b});
    ASSERT_TRUE(first.HasValue());
    EXPECT_EQ(first.Value().first, (std::vector<std::uint8_t>{0, 0}));
    EXPECT_TRUE(first.Value().second);

    auto second = requests_.Lookup("r", {c});  // absent below: the signal dies with it
    ASSERT_TRUE(second.HasValue());
    EXPECT_FALSE(second.Value().second);

    auto plan = requests_.PreparePrefetch("r", {a, b, c}, RequestContext{});
    ASSERT_TRUE(plan.HasValue());  // every key was probed by some lookup
    EXPECT_EQ(plan.Value().keys, (std::vector<BlockId>{a, b, c}));
    EXPECT_EQ(DecodePlanPayload(plan.Value().payload).batchId, std::uint64_t{1});
}

TEST_F(PrefetchRequestsTest, LookupSignalsWhenTheLowerTierHoldsAnyAbsentBlock)
{
    SetUpBuffer(2);
    lower_.present.insert(MakeBlock(2));

    auto res = requests_.Lookup("r", {MakeBlock(1), MakeBlock(2)});
    ASSERT_TRUE(res.HasValue());
    EXPECT_EQ(res.Value().first, (std::vector<std::uint8_t>{0, 0}));
    EXPECT_TRUE(res.Value().second);
}

TEST_F(PrefetchRequestsTest, LookupSkipsTheLowerTierWhenEverythingIsPresent)
{
    SetUpBuffer(1);
    const BlockId a = MakeBlock(1);
    ASSERT_TRUE(buffer_.Reserve({a}, RequestContext{}).HasValue());
    buffer_.Commit({a}, true, false);
    lower_.present.insert(a);

    auto res = requests_.Lookup("r", {a});
    ASSERT_TRUE(res.HasValue());
    EXPECT_EQ(res.Value().first, (std::vector<std::uint8_t>{1}));
    EXPECT_FALSE(res.Value().second);
    EXPECT_TRUE(lower_.lookups.empty());
}

TEST_F(PrefetchRequestsTest, LookupSignalsFalseWhenTheLowerTierHasNothing)
{
    SetUpBuffer(1);
    const BlockId a = MakeBlock(1);

    auto res = requests_.Lookup("r", {a});
    ASSERT_TRUE(res.HasValue());
    EXPECT_EQ(res.Value().first, (std::vector<std::uint8_t>{0}));
    EXPECT_FALSE(res.Value().second);
    EXPECT_EQ(lower_.lookups, (std::vector<std::vector<BlockId>>{{a}}));  // the probe still flew
}

TEST_F(PrefetchRequestsTest, LookupPropagatesAProbeFailureWithoutRecording)
{
    SetUpBuffer(1);
    const BlockId a = MakeBlock(1);
    lower_.FailLookupNth(1, Status::NotFound("lower probe failed"));

    auto res = requests_.Lookup("r", {a});
    ASSERT_FALSE(res.HasValue());
    EXPECT_EQ(res.Error(), Status::NotFound());

    auto plan = requests_.PreparePrefetch("r", {a}, RequestContext{});
    ASSERT_FALSE(plan.HasValue());  // the failed lookup left no probed key
    EXPECT_EQ(plan.Error(), Status::InvalidParam());
}

TEST_F(PrefetchRequestsTest, PreparePrefetchRejectsUnprobedKeysWithoutConsumingAHandle)
{
    SetUpBuffer(2);
    const BlockId a = MakeBlock(1);
    const BlockId b = MakeBlock(2);
    lower_.present.insert(a);
    ASSERT_TRUE(requests_.Lookup("r", {a}).HasValue());

    auto mixed = requests_.PreparePrefetch("r", {a, b}, RequestContext{});
    ASSERT_FALSE(mixed.HasValue());
    EXPECT_EQ(mixed.Error(), Status::InvalidParam());

    auto plan = requests_.PreparePrefetch("r", {a}, RequestContext{});
    ASSERT_TRUE(plan.HasValue());
    EXPECT_EQ(DecodePlanPayload(plan.Value().payload).batchId, std::uint64_t{1});
}

TEST_F(PrefetchRequestsTest, PreparePrefetchRejectsUnknownRequests)
{
    SetUpBuffer(1);
    lower_.present.insert(MakeBlock(1));

    auto empty = requests_.PreparePrefetch("nope", {}, RequestContext{});
    ASSERT_FALSE(empty.HasValue());
    EXPECT_EQ(empty.Error(), Status::InvalidParam());
    auto keys = requests_.PreparePrefetch("nope", {MakeBlock(1)}, RequestContext{});
    ASSERT_FALSE(keys.HasValue());
    EXPECT_EQ(keys.Error(), Status::InvalidParam());
}

TEST_F(PrefetchRequestsTest, PreparePrefetchKeepsOneWindowPerRequest)
{
    SetUpBuffer(2);
    const BlockId a = MakeBlock(1);
    lower_.present.insert(a);
    ASSERT_TRUE(requests_.Lookup("r", {a}).HasValue());

    auto plan = requests_.PreparePrefetch("r", {a}, RequestContext{});
    ASSERT_TRUE(plan.HasValue());
    auto repeat = requests_.PreparePrefetch("r", {a}, RequestContext{});
    ASSERT_FALSE(repeat.HasValue());
    EXPECT_EQ(repeat.Error(), Status::InvalidParam());
}

TEST_F(PrefetchRequestsTest, IdlePlanLeavesNoWindow)
{
    SetUpBuffer(1);
    const BlockId a = MakeBlock(1);
    ASSERT_TRUE(buffer_.Reserve({a}, RequestContext{}).HasValue());
    buffer_.Commit({a}, true, false);  // present: the engine admits nothing
    ASSERT_TRUE(requests_.Lookup("r", {a}).HasValue());

    auto plan = requests_.PreparePrefetch("r", {a}, RequestContext{});
    ASSERT_TRUE(plan.HasValue());
    EXPECT_TRUE(plan.Value().keys.empty());
    EXPECT_TRUE(plan.Value().payload.empty());
    requests_.CompletePrefetch(AddressedToken{}, true);  // an idle plan concludes nothing

    auto again = requests_.PreparePrefetch("r", {a}, RequestContext{});
    ASSERT_TRUE(again.HasValue());  // no window was recorded
    EXPECT_TRUE(again.Value().keys.empty());
}

TEST_F(PrefetchRequestsTest, FailedAdmissionLeavesNoWindow)
{
    SetUpBuffer(1);
    const BlockId pinned = MakeBlock(9);
    const BlockId a = MakeBlock(1);
    ASSERT_TRUE(buffer_.Reserve({pinned}, RequestContext{}).HasValue());
    buffer_.Commit({pinned}, true, false);
    ASSERT_TRUE(buffer_.Pin({pinned}).HasValue());  // the only slot is now unevictable
    lower_.present.insert(a);
    ASSERT_TRUE(requests_.Lookup("r", {a}).HasValue());

    auto plan = requests_.PreparePrefetch("r", {a}, RequestContext{});
    ASSERT_FALSE(plan.HasValue());
    EXPECT_EQ(plan.Error(), Status::NoSpace());

    auto retry = requests_.PreparePrefetch("r", {a}, RequestContext{});
    ASSERT_FALSE(retry.HasValue());
    EXPECT_EQ(retry.Error(), Status::NoSpace());  // still the engine's, not the window's
}

TEST_F(PrefetchRequestsTest, CompletePrefetchConcludesOncePerHandle)
{
    SetUpBuffer(3);
    const BlockId a = MakeBlock(1);
    const BlockId b = MakeBlock(2);
    const BlockId c = MakeBlock(3);
    lower_.present.insert(a);
    lower_.present.insert(b);
    lower_.present.insert(c);
    ASSERT_TRUE(requests_.Lookup("r", {a, b, c}).HasValue());

    auto first = requests_.PreparePrefetch("r", {a, b}, RequestContext{});
    ASSERT_TRUE(first.HasValue());
    requests_.CompletePrefetch(first.Value(), true);

    auto committed = requests_.Lookup("r", {a, b, c});
    ASSERT_TRUE(committed.HasValue());
    EXPECT_EQ(committed.Value().first, (std::vector<std::uint8_t>{1, 1, 0}));

    auto second = requests_.PreparePrefetch("r", {c}, RequestContext{});
    ASSERT_TRUE(second.HasValue());
    EXPECT_EQ(DecodePlanPayload(second.Value().payload).batchId, std::uint64_t{2});

    requests_.CompletePrefetch(first.Value(), true);  // stale handle: window 2 stays open
    requests_.CompletePrefetch(second.Value(), false);
    auto after = requests_.Lookup("r", {a, b, c});
    ASSERT_TRUE(after.HasValue());
    EXPECT_EQ(after.Value().first, (std::vector<std::uint8_t>{1, 1, 0}));
}

TEST_F(PrefetchRequestsTest, ReleaseRequestConcludesAnOpenWindowAsFailed)
{
    SetUpBuffer(1);
    const BlockId a = MakeBlock(1);
    const BlockId b = MakeBlock(2);
    lower_.present.insert(a);
    lower_.present.insert(b);
    ASSERT_TRUE(requests_.Lookup("r1", {a}).HasValue());

    auto plan = requests_.PreparePrefetch("r1", {a}, RequestContext{});
    ASSERT_TRUE(plan.HasValue());  // the window holds the only slot

    requests_.ReleaseRequest("r1");
    auto gone = requests_.PreparePrefetch("r1", {a}, RequestContext{});
    ASSERT_FALSE(gone.HasValue());
    EXPECT_EQ(gone.Error(), Status::InvalidParam());

    ASSERT_TRUE(requests_.Lookup("r2", {b}).HasValue());
    auto next = requests_.PreparePrefetch("r2", {b}, RequestContext{});
    ASSERT_TRUE(next.HasValue());  // the failed conclusion freed the slot
    EXPECT_EQ(DecodePlanPayload(next.Value().payload).batchId, std::uint64_t{2});
}

TEST_F(PrefetchRequestsTest, ReleaseRequestIgnoresUnknownRequests)
{
    SetUpBuffer(1);
    requests_.ReleaseRequest("nope");

    const BlockId a = MakeBlock(1);
    lower_.present.insert(a);
    auto res = requests_.Lookup("r", {a});
    ASSERT_TRUE(res.HasValue());
    EXPECT_TRUE(res.Value().second);
}

TEST_F(PrefetchRequestsTest, LookupAnswersTheSnapshotAlongsideTheSignal)
{
    SetUpBuffer(2);
    const BlockId a = MakeBlock(1);
    const BlockId b = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({a}, RequestContext{}).HasValue());
    buffer_.Commit({a}, true, false);
    lower_.present.insert(b);

    auto res = requests_.Lookup("r", {a, b});
    ASSERT_TRUE(res.HasValue());
    EXPECT_EQ(res.Value().first,
              (std::vector<std::uint8_t>{1, 0}));  // populated with the signal set
    EXPECT_TRUE(res.Value().second);
}

TEST_F(PrefetchRequestsTest, RequestsKeepSeparateWindowsAndProbedSets)
{
    SetUpBuffer(2);
    const BlockId a = MakeBlock(1);
    const BlockId b = MakeBlock(2);
    lower_.present.insert(a);
    lower_.present.insert(b);
    ASSERT_TRUE(requests_.Lookup("r1", {a}).HasValue());
    ASSERT_TRUE(requests_.Lookup("r2", {b}).HasValue());

    auto first = requests_.PreparePrefetch("r1", {a}, RequestContext{});
    ASSERT_TRUE(first.HasValue());
    EXPECT_EQ(DecodePlanPayload(first.Value().payload).batchId, std::uint64_t{1});

    auto second = requests_.PreparePrefetch("r2", {b}, RequestContext{});
    ASSERT_TRUE(second.HasValue());  // r1's window does not reach r2
    EXPECT_EQ(DecodePlanPayload(second.Value().payload).batchId, std::uint64_t{2});

    auto leaked = requests_.PreparePrefetch("r2", {a}, RequestContext{});
    ASSERT_FALSE(leaked.HasValue());  // a sits in r1's set, not r2's
    EXPECT_EQ(leaked.Error(), Status::InvalidParam());
}

}  // namespace
}  // namespace UC::Store
