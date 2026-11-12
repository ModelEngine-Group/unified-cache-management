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
#include "store_assembler.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "fixtures/fake_store.h"

namespace UC::Proxy {
namespace {

Dictionary MakeStoreNode()
{
    Dictionary node;
    node.Set("store", std::string(UCM_TEST_STUB_STORE_PATH));
    return node;
}

Dictionary MakeStoreNode(std::int64_t present)
{
    Dictionary node = MakeStoreNode();
    Dictionary setup;
    setup.SetInt64("present", present);
    node.Set("config", setup);
    return node;
}

Dictionary MakeStoreNode(const std::string& requiredUniqueId)
{
    Dictionary node = MakeStoreNode();
    Dictionary setup;
    setup.Set("required_unique_id", requiredUniqueId);
    node.Set("config", setup);
    return node;
}

Dictionary MakeFailingStoreNode()
{
    Dictionary node = MakeStoreNode();
    Dictionary setup;
    setup.SetInt64("fail_setup", 1);
    node.Set("config", setup);
    return node;
}

Dictionary MakeBranch(const std::vector<Dictionary>& tiers)
{
    Dictionary branch;
    branch.Set("pipe", tiers);
    return branch;
}

Dictionary MakePeer(const std::vector<Dictionary>& branches)
{
    Dictionary config;
    config.Set("branches", branches);
    return config;
}

Dictionary MakeSingleBranchPeer(const std::vector<Dictionary>& tiers)
{
    return MakePeer({MakeBranch(tiers)});
}

Dictionary MakeLengthRouting(const std::vector<std::int64_t>& thresholds)
{
    Dictionary routing;
    routing.Set("strategy", std::string("length"));
    Dictionary params;
    params.SetNumbers("thresholds", thresholds);
    routing.Set("config", params);
    return routing;
}

std::vector<std::uint8_t> LookupHits(StoreScheduler& scheduler, const RequestId& requestId,
                                     const std::vector<BlockId>& keys)
{
    auto hits = scheduler.Lookup(requestId, keys);
    EXPECT_TRUE(hits);
    if (!hits) { return {}; }
    EXPECT_FALSE(hits.Value().second);
    return hits.Value().first;
}

TEST(StoreAssemblerTest, CreateSingleBranchPeerDrivesConfiguredLookup)
{
    /* Single branch, no routing: the internal fixed strategy routes to it. */
    auto assembler = StoreAssembler::Create(MakeSingleBranchPeer({MakeStoreNode(2)}));
    ASSERT_TRUE(assembler);
    auto* scheduler = assembler.Value()->GetScheduler();
    ASSERT_NE(scheduler, nullptr);
    EXPECT_NE(assembler.Value()->GetWorker(), nullptr);
    const std::vector<BlockId> keys = MakeKeys(1, 3);
    EXPECT_TRUE(scheduler->OnNewRequest(MakeContext("r")).Success());
    EXPECT_EQ(LookupHits(*scheduler, "r", keys), (std::vector<std::uint8_t>{1, 1, 0}));
    scheduler->OnRequestFinished("r");
}

TEST(StoreAssemblerTest, CreateRejectsMissingBranches)
{
    auto missing = StoreAssembler::Create(Dictionary{});
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.Error(), Status::InvalidParam());

    Dictionary empty;
    empty.Set("branches", std::vector<Dictionary>{});
    auto emptyAssembler = StoreAssembler::Create(empty);
    ASSERT_FALSE(emptyAssembler);
    EXPECT_EQ(emptyAssembler.Error(), Status::InvalidParam());
}

TEST(StoreAssemblerTest, CreateRejectsBranchWithoutPipe)
{
    auto missing = StoreAssembler::Create(MakePeer({Dictionary{}}));
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.Error(), Status::InvalidParam());

    Dictionary emptyPipe;
    emptyPipe.Set("pipe", std::vector<Dictionary>{});
    auto empty = StoreAssembler::Create(MakePeer({emptyPipe}));
    ASSERT_FALSE(empty);
    EXPECT_EQ(empty.Error(), Status::InvalidParam());
}

TEST(StoreAssemblerTest, CreateRejectsPipeEntryWithoutStore)
{
    auto assembler = StoreAssembler::Create(MakeSingleBranchPeer({Dictionary{}}));
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::InvalidParam());
}

TEST(StoreAssemblerTest, CreatePropagatesSetupFailure)
{
    auto assembler = StoreAssembler::Create(MakeSingleBranchPeer({MakeFailingStoreNode()}));
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::General());
}

TEST(StoreAssemblerTest, CreateRejectsReservedUserKeys)
{
    Dictionary setup;
    setup.SetInt64("__evil", 1);
    Dictionary node = MakeStoreNode();
    node.Set("config", setup);
    auto assembler = StoreAssembler::Create(MakeSingleBranchPeer({node}));
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::InvalidParam());
    EXPECT_NE(assembler.Error().ToString().find("__evil"), std::string::npos);
}

TEST(StoreAssemblerTest, CreatePipeWiresLowerStoreToHead)
{
    auto assembler =
        StoreAssembler::Create(MakeSingleBranchPeer({MakeStoreNode(), MakeStoreNode(1)}));
    ASSERT_TRUE(assembler);
    auto* scheduler = assembler.Value()->GetScheduler();
    const std::vector<BlockId> keys = MakeKeys(1, 3);
    EXPECT_TRUE(scheduler->OnNewRequest(MakeContext("r")).Success());
    /* The head would report all-present on its own; only a wired delegation
     * to the tail (present=1) yields this hit vector. */
    EXPECT_EQ(LookupHits(*scheduler, "r", keys), (std::vector<std::uint8_t>{1, 0, 0}));
}

TEST(StoreAssemblerTest, CreatePeerRoutesByLengthStrategy)
{
    Dictionary config = MakePeer({MakeBranch({MakeStoreNode(1)}), MakeBranch({MakeStoreNode(3)})});
    config.Set("routing", MakeLengthRouting({4}));

    auto assembler = StoreAssembler::Create(config);
    ASSERT_TRUE(assembler);
    auto* scheduler = assembler.Value()->GetScheduler();
    const std::vector<BlockId> keys = MakeKeys(1, 3);

    RequestContext longRequest;
    longRequest.requestId = "long";
    longRequest.totalBlockCount = 10;
    EXPECT_TRUE(scheduler->OnNewRequest(longRequest).Success());
    EXPECT_EQ(LookupHits(*scheduler, "long", keys), (std::vector<std::uint8_t>{1, 0, 0}));

    RequestContext shortRequest;
    shortRequest.requestId = "short";
    shortRequest.totalBlockCount = 2;
    EXPECT_TRUE(scheduler->OnNewRequest(shortRequest).Success());
    EXPECT_EQ(LookupHits(*scheduler, "short", keys), (std::vector<std::uint8_t>{1, 1, 1}));
}

TEST(StoreAssemblerTest, CreatePeerRequiresRoutingForMultipleBranches)
{
    auto assembler = StoreAssembler::Create(
        MakePeer({MakeBranch({MakeStoreNode()}), MakeBranch({MakeStoreNode()})}));
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::InvalidParam());
}

TEST(StoreAssemblerTest, CreatePeerRejectsUnknownStrategy)
{
    Dictionary config = MakePeer({MakeBranch({MakeStoreNode()}), MakeBranch({MakeStoreNode()})});
    Dictionary routing;
    routing.Set("strategy", std::string("no-such-strategy"));
    config.Set("routing", routing);
    auto assembler = StoreAssembler::Create(config);
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::NotFound());
}

TEST(StoreAssemblerTest, CreatePeerRejectsThresholdBranchMismatch)
{
    Dictionary config = MakePeer({MakeBranch({MakeStoreNode()}), MakeBranch({MakeStoreNode()})});
    config.Set("routing", MakeLengthRouting({4, 8}));
    auto assembler = StoreAssembler::Create(config);
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::InvalidParam());
}

TEST(StoreAssemblerTest, RuntimeMergesIntoStoreSetup)
{
    Dictionary runtime;
    runtime.SetInt64("present", 1);
    auto assembler = StoreAssembler::Create(MakeSingleBranchPeer({MakeStoreNode()}), runtime);
    ASSERT_TRUE(assembler);
    auto* scheduler = assembler.Value()->GetScheduler();
    const std::vector<BlockId> keys = MakeKeys(1, 3);
    EXPECT_TRUE(scheduler->OnNewRequest(MakeContext("r")).Success());
    /* "present" reaches Setup only through the runtime merge: the node
     * config does not carry it. */
    EXPECT_EQ(LookupHits(*scheduler, "r", keys), (std::vector<std::uint8_t>{1, 0, 0}));
    scheduler->OnRequestFinished("r");

    /* Node-owned keys stay in effect alongside merged runtime keys. */
    Dictionary other;
    other.SetInt64("device_id", 7);
    auto untouched = StoreAssembler::Create(MakeSingleBranchPeer({MakeStoreNode(2)}), other);
    ASSERT_TRUE(untouched);
    EXPECT_TRUE(untouched.Value()->GetScheduler()->OnNewRequest(MakeContext("r2")).Success());
    EXPECT_EQ(LookupHits(*untouched.Value()->GetScheduler(), "r2", keys),
              (std::vector<std::uint8_t>{1, 1, 0}));
}

TEST(StoreAssemblerTest, RuntimeConflictWithNodeConfigRejected)
{
    Dictionary runtime;
    runtime.SetInt64("present", 1);
    auto assembler = StoreAssembler::Create(MakeSingleBranchPeer({MakeStoreNode(2)}), runtime);
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::InvalidParam());
    EXPECT_NE(assembler.Error().ToString().find("present"), std::string::npos);
}

TEST(StoreAssemblerTest, RuntimeUniqueIdSuffixedPerStore)
{
    Dictionary config = MakePeer({MakeBranch({MakeStoreNode("base@branches[0].pipe[0]"),
                                              MakeStoreNode("base@branches[0].pipe[1]")}),
                                  MakeBranch({MakeStoreNode("base@branches[1].pipe[0]")})});
    config.Set("routing", MakeLengthRouting({4}));
    Dictionary runtime;
    runtime.Set("unique_id", std::string("base"));
    auto assembler = StoreAssembler::Create(config, runtime);
    ASSERT_TRUE(assembler);

    /* A wrong expectation fails the tier's Setup, proving the suffixing. */
    Dictionary wrong = MakeSingleBranchPeer({MakeStoreNode("base@branches[0].pipe[9]")});
    auto rejected = StoreAssembler::Create(wrong, runtime);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.Error(), Status::General());
}

TEST(StoreAssemblerTest, RuntimeWithoutUniqueIdInjectsNothing)
{
    Dictionary runtime;
    runtime.SetInt64("device_id", 7);
    auto assembler =
        StoreAssembler::Create(MakeSingleBranchPeer({MakeStoreNode("anything")}), runtime);
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::General());
}

TEST(StoreAssemblerTest, RuntimeUniqueIdMustBeString)
{
    Dictionary runtime;
    runtime.SetInt64("unique_id", 5);
    auto assembler = StoreAssembler::Create(MakeSingleBranchPeer({MakeStoreNode()}), runtime);
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::InvalidParam());
}

}  // namespace
}  // namespace UC::Proxy
