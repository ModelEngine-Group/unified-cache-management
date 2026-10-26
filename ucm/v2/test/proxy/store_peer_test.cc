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
#include "store_peer.h"
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "fixtures/fake_store.h"

namespace UC::Proxy {
namespace {

class FakeRoutingStrategy : public RoutingStrategy {
public:
    Status Setup(const Dictionary& /*config*/, std::size_t /*branchCount*/) override
    {
        return Status::Ok();
    }

    std::size_t Select(const RequestContext& request) override
    {
        selects.push_back(request.requestId);
        const auto it = branches.find(request.requestId);
        return it != branches.end() ? it->second : fallback;
    }

    std::size_t fallback = 0;
    std::unordered_map<RequestId, std::size_t> branches;
    std::vector<RequestId> selects;
};

struct PeerHandle {
    FakeRoutingStrategy* routing;
    std::vector<FakeStore*> fakes;
    std::unique_ptr<StorePeer> peer;
};

PeerHandle MakePeer(std::initializer_list<std::pair<bool, bool>> configs, std::size_t fallback = 0)
{
    auto routing = std::make_unique<FakeRoutingStrategy>();
    routing->fallback = fallback;
    PeerHandle handle;
    handle.routing = routing.get();
    std::vector<std::unique_ptr<StoreEntry>> branches;
    for (const auto& [prefetchable, tokened] : configs) {
        auto entry = MakeEntry(prefetchable, tokened);
        handle.fakes.push_back(entry.fake);
        branches.push_back(std::move(entry.entry));
    }
    handle.peer = std::make_unique<StorePeer>(std::move(branches), std::move(routing));
    return handle;
}

TEST(StorePeerTest, OnNewRequestRoutesAndCachesBranch)
{
    auto handle = MakePeer({
        {true, true},
        {true, true}
    });
    handle.routing->branches["r1"] = 0;
    handle.routing->branches["r2"] = 1;
    handle.peer->OnNewRequest(MakeContext("r1"));
    handle.peer->OnNewRequest(MakeContext("r2"));
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    ASSERT_TRUE(handle.peer->Lookup("r1", keys));
    ASSERT_TRUE(handle.peer->Lookup("r1", keys));
    ASSERT_TRUE(handle.peer->Lookup("r2", keys));
    EXPECT_EQ(handle.routing->selects, (std::vector<RequestId>{"r1", "r2"}));
    EXPECT_EQ(handle.fakes[0]->prefetchLookups.size(), 2);
    EXPECT_EQ(handle.fakes[1]->prefetchLookups.size(), 1);
    EXPECT_EQ(handle.fakes[1]->prefetchLookups[0].first, "r2");
}

TEST(StorePeerTest, SchedulerOperationsStickToRequestBranch)
{
    auto handle = MakePeer(
        {
            {true, true},
            {true, true}
    },
        1);
    handle.peer->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    ASSERT_TRUE(handle.peer->Lookup("r", keys));
    ASSERT_TRUE(handle.peer->LookupOnPrefix("r", keys));
    ASSERT_TRUE(handle.peer->LookupOnReverse("r", keys));
    handle.peer->Touch("r", keys);
    auto plan = handle.peer->PreparePrefetch("r");
    ASSERT_TRUE(plan);
    handle.peer->CompletePrefetch("r", true);
    auto token = handle.peer->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    handle.peer->CompleteGet("r");
    auto putToken = handle.peer->PreparePut("r", keys);
    ASSERT_TRUE(putToken);
    handle.peer->CompletePut("r", true);

    auto* fake = handle.fakes[1];
    EXPECT_EQ(fake->prefetchLookups.size(), 3);
    EXPECT_EQ(fake->touches.size(), 1);
    EXPECT_EQ(fake->touches[0].first, keys);
    EXPECT_EQ(fake->preparedPrefetches.size(), 1);
    EXPECT_EQ(fake->completedPrefetches.size(), 1);
    EXPECT_EQ(fake->pins.size(), 1);
    EXPECT_EQ(fake->pins[0], keys);
    EXPECT_EQ(fake->unpins.size(), 1);
    EXPECT_EQ(fake->allocs.size(), 1);
    EXPECT_EQ(fake->allocs[0].first, keys);
    EXPECT_EQ(fake->commits.size(), 1);
    EXPECT_TRUE(fake->commits[0].second);

    auto* other = handle.fakes[0];
    EXPECT_TRUE(other->prefetchLookups.empty());
    EXPECT_TRUE(other->touches.empty());
    EXPECT_TRUE(other->preparedPrefetches.empty());
    EXPECT_TRUE(other->completedPrefetches.empty());
    EXPECT_TRUE(other->pins.empty());
    EXPECT_TRUE(other->unpins.empty());
    EXPECT_TRUE(other->allocs.empty());
    EXPECT_TRUE(other->commits.empty());
}

TEST(StorePeerTest, OnNewRequestDuplicateIsRejected)
{
    auto handle = MakePeer({
        {true, true}
    });
    const RequestContext context = MakeContext("r");
    EXPECT_TRUE(handle.peer->OnNewRequest(context).Success());
    auto duplicate = handle.peer->OnNewRequest(context);
    ASSERT_TRUE(duplicate.Failure());
    EXPECT_EQ(duplicate, Status::InvalidParam());
}

TEST(StorePeerTest, OperationsOnUnknownRequestFail)
{
    auto handle = MakePeer({
        {true, true}
    });
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    EXPECT_FALSE(handle.peer->Lookup("missing", keys));
    EXPECT_FALSE(handle.peer->LookupOnPrefix("missing", keys));
    EXPECT_FALSE(handle.peer->LookupOnReverse("missing", keys));
    EXPECT_FALSE(handle.peer->PreparePrefetch("missing"));
    EXPECT_FALSE(handle.peer->PrepareGet("missing", keys));
    EXPECT_FALSE(handle.peer->PreparePut("missing", keys));
    handle.peer->Touch("missing", keys);
    handle.peer->CompletePrefetch("missing", true);
    handle.peer->CompleteGet("missing");
    handle.peer->CompletePut("missing", true);
    handle.peer->OnRequestFinished("missing");
    EXPECT_TRUE(handle.fakes[0]->prefetchLookups.empty());
    EXPECT_TRUE(handle.fakes[0]->pins.empty());
    EXPECT_TRUE(handle.fakes[0]->allocs.empty());
}

TEST(StorePeerTest, OnRequestFinishedForwardsAndCleansUp)
{
    auto handle = MakePeer({
        {true, true}
    });
    handle.peer->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    ASSERT_TRUE(handle.peer->Lookup("r", keys));
    handle.peer->OnRequestFinished("r");
    EXPECT_FALSE(handle.peer->Lookup("r", keys));
    EXPECT_EQ(handle.routing->selects.size(), 1);
}

TEST(StorePeerTest, PreparePrefetchWrapsPlanForRouting)
{
    auto handle = MakePeer({
        {true, true}
    });
    handle.peer->OnNewRequest(MakeContext("r"));
    handle.fakes[0]->signalPrefetchable = true;
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    ASSERT_FALSE(handle.peer->Lookup("r", keys).Value().has_value());
    auto plan = handle.peer->PreparePrefetch("r");
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan.Value().keys, keys);
    EXPECT_NE(plan.Value().payload, (std::vector<std::byte>{std::byte{1}, std::byte{2}}));
    EXPECT_TRUE(handle.peer->SubmitPrefetch(7, plan.Value()).Success());
    ASSERT_EQ(handle.fakes[0]->prefetches.size(), 1);
    EXPECT_EQ(handle.fakes[0]->prefetches[0].first, 7);
    EXPECT_EQ(handle.fakes[0]->prefetches[0].second.keys, keys);
    EXPECT_EQ(handle.fakes[0]->prefetches[0].second.payload,
              (std::vector<std::byte>{std::byte{1}, std::byte{2}}));
}

TEST(StorePeerTest, PrepareGetWrapsTokenForRouting)
{
    auto handle = MakePeer({
        {true, true}
    });
    handle.peer->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    auto token = handle.peer->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    EXPECT_EQ(token.Value().keys, keys);
    EXPECT_NE(token.Value().payload, MakeAddressedToken(2).payload);
    const std::vector<BlockPayload> payloads = {MakePayload(16), MakePayload(32)};
    const IoContext context{0xABCD};
    EXPECT_TRUE(handle.peer->SubmitGet(9, payloads, token.Value(), context).Success());
    auto duplicate = handle.peer->SubmitGet(9, payloads, token.Value(), context);
    ASSERT_TRUE(duplicate.Failure());
    EXPECT_EQ(duplicate, Status::InvalidParam());
    ASSERT_EQ(handle.fakes[0]->tokenedLoads.size(), 1);
    EXPECT_EQ(handle.fakes[0]->tokenedLoads[0].jobId, 9);
    EXPECT_EQ(handle.fakes[0]->tokenedLoads[0].job.blockIds, keys);
    EXPECT_EQ(handle.fakes[0]->tokenedLoads[0].job.payloads[1].sizes, payloads[1].sizes);
    EXPECT_EQ(handle.fakes[0]->tokenedLoads[0].job.prerequisiteHandle, 0xABCD);
    EXPECT_TRUE(handle.fakes[0]->tokenedLoads[0].token.keys.empty());
    EXPECT_EQ(handle.fakes[0]->tokenedLoads[0].token.payload, MakeAddressedToken(2).payload);
}

TEST(StorePeerTest, NonTokenedBranchRoutesThroughEmptyPayload)
{
    auto handle = MakePeer({
        {true, false}
    });
    handle.peer->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    auto token = handle.peer->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    EXPECT_EQ(token.Value().keys, keys);
    EXPECT_FALSE(token.Value().payload.empty());
    EXPECT_TRUE(
        handle.peer->SubmitGet(9, {MakePayload(16), MakePayload(32)}, token.Value(), IoContext{0})
            .Success());
    ASSERT_EQ(handle.fakes[0]->baseLoads.size(), 1);
    EXPECT_EQ(handle.fakes[0]->baseLoads[0].jobId, 9);
    EXPECT_EQ(handle.fakes[0]->baseLoads[0].job.blockIds, keys);
    EXPECT_TRUE(handle.fakes[0]->tokenedLoads.empty());

    auto putToken = handle.peer->PreparePut("r", keys);
    ASSERT_TRUE(putToken);
    EXPECT_EQ(putToken.Value().keys, keys);
    EXPECT_TRUE(
        handle.peer
            ->SubmitPut(4, {MakePayload(16), MakePayload(32)}, putToken.Value(), IoContext{0})
            .Success());
    ASSERT_EQ(handle.fakes[0]->baseDumps.size(), 1);
    EXPECT_EQ(handle.fakes[0]->baseDumps[0].jobId, 4);
    EXPECT_TRUE(handle.fakes[0]->tokenedDumps.empty());
}

TEST(StorePeerTest, NonPrefetchableBranchSynthesizesThroughPeer)
{
    auto handle = MakePeer({
        {false, true}
    });
    handle.peer->OnNewRequest(MakeContext("r"));
    auto plan = handle.peer->PreparePrefetch("r");
    ASSERT_TRUE(plan);
    EXPECT_TRUE(plan.Value().keys.empty());
    EXPECT_TRUE(handle.peer->SubmitPrefetch(7, plan.Value()).Success());
    auto duplicate = handle.peer->SubmitPrefetch(7, plan.Value());
    ASSERT_TRUE(duplicate.Failure());
    EXPECT_EQ(duplicate, Status::InvalidParam());
    EXPECT_TRUE(handle.fakes[0]->prefetches.empty());
    auto finished = handle.peer->GetFinished();
    ASSERT_EQ(finished.size(), 1);
    EXPECT_EQ(finished[0].jobId, 7);
    EXPECT_TRUE(finished[0].success);
    EXPECT_TRUE(handle.peer->GetFinished().empty());
    EXPECT_TRUE(handle.peer->SubmitPrefetch(7, plan.Value()).Success());
}

TEST(StorePeerTest, DuplicateJobIdRejectedAcrossBranches)
{
    auto handle = MakePeer({
        {true, true},
        {true, true}
    });
    handle.routing->branches["r1"] = 0;
    handle.routing->branches["r2"] = 1;
    handle.peer->OnNewRequest(MakeContext("r1"));
    handle.peer->OnNewRequest(MakeContext("r2"));
    auto plan1 = handle.peer->PreparePrefetch("r1");
    ASSERT_TRUE(plan1);
    auto plan2 = handle.peer->PreparePrefetch("r2");
    ASSERT_TRUE(plan2);
    EXPECT_TRUE(handle.peer->SubmitPrefetch(7, plan1.Value()).Success());
    auto rejected = handle.peer->SubmitPrefetch(7, plan2.Value());
    ASSERT_TRUE(rejected.Failure());
    EXPECT_EQ(rejected, Status::InvalidParam());
    EXPECT_TRUE(handle.fakes[1]->prefetches.empty());
    handle.fakes[0]->finishedQueue = {
        {7, true}
    };
    auto finished = handle.peer->GetFinished();
    ASSERT_EQ(finished.size(), 1);
    EXPECT_EQ(finished[0].jobId, 7);
    EXPECT_TRUE(handle.peer->SubmitPrefetch(7, plan2.Value()).Success());
    ASSERT_EQ(handle.fakes[1]->prefetches.size(), 1);
}

TEST(StorePeerTest, GetFinishedAggregatesAllBranches)
{
    auto handle = MakePeer({
        {true, true},
        {true, true}
    });
    handle.routing->branches["r1"] = 0;
    handle.routing->branches["r2"] = 1;
    handle.peer->OnNewRequest(MakeContext("r1"));
    handle.peer->OnNewRequest(MakeContext("r2"));
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    auto token1 = handle.peer->PrepareGet("r1", keys);
    ASSERT_TRUE(token1);
    auto token2 = handle.peer->PrepareGet("r2", keys);
    ASSERT_TRUE(token2);
    EXPECT_TRUE(
        handle.peer->SubmitGet(1, {MakePayload(16)}, token1.Value(), IoContext{0}).Success());
    EXPECT_TRUE(
        handle.peer->SubmitGet(2, {MakePayload(16)}, token2.Value(), IoContext{0}).Success());
    handle.fakes[0]->finishedQueue = {
        {1, true}
    };
    handle.fakes[1]->finishedQueue = {
        {2, false}
    };
    auto results = handle.peer->GetFinished();
    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].jobId, 1);
    EXPECT_TRUE(results[0].success);
    EXPECT_EQ(results[1].jobId, 2);
    EXPECT_FALSE(results[1].success);
    EXPECT_TRUE(handle.peer->GetFinished().empty());
    EXPECT_TRUE(
        handle.peer->SubmitGet(1, {MakePayload(16)}, token1.Value(), IoContext{0}).Success());
}

TEST(StorePeerTest, WaitRoutesByBranchAndIgnoresUnknownIds)
{
    auto handle = MakePeer({
        {true, true},
        {true, true}
    });
    handle.routing->branches["r1"] = 0;
    handle.routing->branches["r2"] = 1;
    handle.peer->OnNewRequest(MakeContext("r1"));
    handle.peer->OnNewRequest(MakeContext("r2"));
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    auto token1 = handle.peer->PrepareGet("r1", keys);
    ASSERT_TRUE(token1);
    auto token2 = handle.peer->PrepareGet("r2", keys);
    ASSERT_TRUE(token2);
    EXPECT_TRUE(
        handle.peer->SubmitGet(1, {MakePayload(16)}, token1.Value(), IoContext{0}).Success());
    EXPECT_TRUE(
        handle.peer->SubmitGet(2, {MakePayload(16)}, token2.Value(), IoContext{0}).Success());
    handle.fakes[0]->finishedQueue = {
        {1, true}
    };
    handle.fakes[1]->finishedQueue = {
        {2, false}
    };
    auto results = handle.peer->Wait({1, 2, 404});
    EXPECT_EQ(results.size(), 2);
    EXPECT_TRUE(Contains(results, 1, true));
    EXPECT_TRUE(Contains(results, 2, false));
    ASSERT_EQ(handle.fakes[0]->waits.size(), 1);
    EXPECT_EQ(handle.fakes[0]->waits[0], std::vector<std::size_t>{1});
    ASSERT_EQ(handle.fakes[1]->waits.size(), 1);
    EXPECT_EQ(handle.fakes[1]->waits[0], std::vector<std::size_t>{2});
    EXPECT_TRUE(handle.peer->Wait({1, 2}).empty());
    EXPECT_TRUE(handle.peer->GetFinished().empty());
}

TEST(StorePeerTest, GetFinishedSkipsBranchesWithoutInFlightJobs)
{
    auto handle = MakePeer({
        {true, true},
        {true, true}
    });
    handle.routing->branches["r1"] = 0;
    handle.routing->branches["r2"] = 1;
    handle.peer->OnNewRequest(MakeContext("r1"));
    handle.peer->OnNewRequest(MakeContext("r2"));
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    auto token2 = handle.peer->PrepareGet("r2", keys);
    ASSERT_TRUE(token2);
    EXPECT_TRUE(
        handle.peer->SubmitGet(2, {MakePayload(16)}, token2.Value(), IoContext{0}).Success());
    handle.fakes[1]->finishedQueue = {
        {2, true}
    };
    handle.fakes[0]->getFinishedCalls = 0;
    handle.fakes[1]->getFinishedCalls = 0;
    auto results = handle.peer->GetFinished();
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].jobId, 2u);
    EXPECT_TRUE(results[0].success);
    EXPECT_EQ(handle.fakes[0]->getFinishedCalls, 0u);
    EXPECT_EQ(handle.fakes[1]->getFinishedCalls, 1u);
    auto token1 = handle.peer->PrepareGet("r1", keys);
    ASSERT_TRUE(token1);
    EXPECT_TRUE(
        handle.peer->SubmitGet(1, {MakePayload(16)}, token1.Value(), IoContext{0}).Success());
    handle.fakes[0]->finishedQueue = {
        {1, true}
    };
    auto reactivated = handle.peer->GetFinished();
    ASSERT_EQ(reactivated.size(), 1);
    EXPECT_EQ(reactivated[0].jobId, 1u);
    EXPECT_TRUE(handle.peer->GetFinished().empty());
}

TEST(StorePeerTest, SubmitRejectsMalformedRoutingTokens)
{
    auto handle = MakePeer({
        {true, true}
    });
    handle.peer->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 1);

    AddressedToken raw;
    auto planRejected = handle.peer->SubmitPrefetch(7, raw);
    ASSERT_TRUE(planRejected.Failure());
    EXPECT_EQ(planRejected, Status::InvalidParam());

    auto noPrefix = handle.peer->SubmitGet(9, {MakePayload(16)}, AddressedToken{}, IoContext{0});
    ASSERT_TRUE(noPrefix.Failure());
    EXPECT_EQ(noPrefix, Status::InvalidParam());

    AddressedToken shortPrefix = MakeAddressedToken(1);
    shortPrefix.keys = keys;
    auto shortRejected = handle.peer->SubmitPut(9, {MakePayload(16)}, shortPrefix, IoContext{0});
    ASSERT_TRUE(shortRejected.Failure());
    EXPECT_EQ(shortRejected, Status::InvalidParam());

    AddressedToken farBranch;
    farBranch.keys = keys;
    farBranch.payload.assign(8, std::byte{0});
    farBranch.payload[0] = std::byte{5};
    auto farRejected = handle.peer->SubmitGet(9, {MakePayload(16)}, farBranch, IoContext{0});
    ASSERT_TRUE(farRejected.Failure());
    EXPECT_EQ(farRejected, Status::InvalidParam());

    EXPECT_TRUE(handle.fakes[0]->tokenedLoads.empty());
    EXPECT_TRUE(handle.fakes[0]->baseLoads.empty());
    EXPECT_TRUE(handle.fakes[0]->baseDumps.empty());
    EXPECT_TRUE(handle.fakes[0]->prefetches.empty());
}

TEST(StorePeerTest, SubmitPropagatesBranchFailureWithoutTracking)
{
    auto handle = MakePeer({
        {true, true}
    });
    handle.peer->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    auto token = handle.peer->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    handle.fakes[0]->loadStatus = Status::General("load rejected");
    auto status = handle.peer->SubmitGet(9, {MakePayload(16)}, token.Value(), IoContext{0});
    EXPECT_EQ(status, Status::General());
    handle.fakes[0]->loadStatus = Status::Ok();
    EXPECT_TRUE(
        handle.peer->SubmitGet(9, {MakePayload(16)}, token.Value(), IoContext{0}).Success());
}

TEST(StorePeerTest, SubmitRejectsStoreNativePayloadResemblingBranchPrefix)
{
    auto handle = MakePeer({
        {true, true}
    });
    handle.peer->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    // A token that never passed through the peer: under the old framing its
    // first 8 zero bytes would decode as branch 0 and it would be silently
    // routed and truncated.
    AddressedToken native;
    native.keys = keys;
    native.payload.assign(12, std::byte{0});
    auto rejected = handle.peer->SubmitGet(9, {MakePayload(16)}, native, IoContext{0});
    ASSERT_TRUE(rejected.Failure());
    EXPECT_EQ(rejected, Status::InvalidParam());
    EXPECT_TRUE(handle.fakes[0]->tokenedLoads.empty());
    EXPECT_TRUE(handle.fakes[0]->baseLoads.empty());
}

TEST(StorePeerTest, SubmitRejectsForeignMagic)
{
    auto handle = MakePeer({
        {true, true}
    });
    handle.peer->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    AddressedToken foreign;
    foreign.keys = keys;
    foreign.payload = {static_cast<std::byte>('U'), static_cast<std::byte>('C'),
                       static_cast<std::byte>('P'), static_cast<std::byte>('0')};
    foreign.payload.resize(12, std::byte{0});
    auto rejected = handle.peer->SubmitPut(9, {MakePayload(16)}, foreign, IoContext{0});
    ASSERT_TRUE(rejected.Failure());
    EXPECT_EQ(rejected, Status::InvalidParam());
    EXPECT_TRUE(handle.fakes[0]->tokenedDumps.empty());
    EXPECT_TRUE(handle.fakes[0]->baseDumps.empty());
}

TEST(StorePeerTest, SubmitRejectsTruncatedFrame)
{
    auto handle = MakePeer({
        {true, true}
    });
    handle.peer->OnNewRequest(MakeContext("r"));
    AddressedToken truncated;
    truncated.payload = {static_cast<std::byte>('U'), static_cast<std::byte>('C'),
                         static_cast<std::byte>('P'), static_cast<std::byte>('1')};
    auto rejected = handle.peer->SubmitPrefetch(7, truncated);
    ASSERT_TRUE(rejected.Failure());
    EXPECT_EQ(rejected, Status::InvalidParam());
    EXPECT_TRUE(handle.fakes[0]->prefetches.empty());
}

}  // namespace
}  // namespace UC::Proxy
