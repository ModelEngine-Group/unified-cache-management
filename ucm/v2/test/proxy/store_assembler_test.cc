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
    node.Set("type", std::string("store"));
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

Dictionary MakeFailingStoreNode()
{
    Dictionary node = MakeStoreNode();
    Dictionary setup;
    setup.SetInt64("fail_setup", 1);
    node.Set("config", setup);
    return node;
}

std::vector<std::uint8_t> LookupHits(StoreScheduler& scheduler, const RequestId& requestId,
                                     const std::vector<BlockId>& keys)
{
    auto hits = scheduler.Lookup(requestId, keys);
    EXPECT_TRUE(hits);
    if (!hits) { return {}; }
    EXPECT_TRUE(hits.Value().has_value());
    if (!hits.Value().has_value()) { return {}; }
    return hits.Value().value();
}

TEST(StoreAssemblerTest, CreateStoreNodeDrivesConfiguredLookup)
{
    auto assembler = StoreAssembler::Create(MakeStoreNode(2));
    ASSERT_TRUE(assembler);
    auto* scheduler = assembler.Value()->GetScheduler();
    ASSERT_NE(scheduler, nullptr);
    EXPECT_NE(assembler.Value()->GetWorker(), nullptr);
    const std::vector<BlockId> keys = MakeKeys(1, 3);
    EXPECT_TRUE(scheduler->OnNewRequest(MakeContext("r")).Success());
    EXPECT_EQ(LookupHits(*scheduler, "r", keys), (std::vector<std::uint8_t>{1, 1, 0}));
    scheduler->OnRequestFinished("r");
}

TEST(StoreAssemblerTest, CreateRejectsMissingType)
{
    auto assembler = StoreAssembler::Create(Dictionary{});
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::InvalidParam());
}

TEST(StoreAssemblerTest, CreateRejectsUnknownType)
{
    Dictionary config;
    config.Set("type", std::string("no-such-type"));
    auto assembler = StoreAssembler::Create(config);
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::InvalidParam());
    EXPECT_NE(assembler.Error().ToString().find("no-such-type"), std::string::npos);
}

TEST(StoreAssemblerTest, CreateRejectsStoreNodeWithoutStore)
{
    Dictionary config;
    config.Set("type", std::string("store"));
    auto assembler = StoreAssembler::Create(config);
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::InvalidParam());
}

TEST(StoreAssemblerTest, CreatePropagatesSetupFailure)
{
    auto assembler = StoreAssembler::Create(MakeFailingStoreNode());
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::General());
}

TEST(StoreAssemblerTest, CreateRejectsPipeWithoutStores)
{
    Dictionary missing;
    missing.Set("type", std::string("pipe"));
    auto missingAssembler = StoreAssembler::Create(missing);
    ASSERT_FALSE(missingAssembler);
    EXPECT_EQ(missingAssembler.Error(), Status::InvalidParam());

    Dictionary empty;
    empty.Set("type", std::string("pipe"));
    empty.Set("stores", std::vector<Dictionary>{});
    auto emptyAssembler = StoreAssembler::Create(empty);
    ASSERT_FALSE(emptyAssembler);
    EXPECT_EQ(emptyAssembler.Error(), Status::InvalidParam());
}

TEST(StoreAssemblerTest, CreatePipeWiresLowerStoreToHead)
{
    std::vector<Dictionary> stores;
    stores.push_back(MakeStoreNode());
    stores.push_back(MakeStoreNode(1));
    Dictionary config;
    config.Set("type", std::string("pipe"));
    config.Set("stores", stores);

    auto assembler = StoreAssembler::Create(config);
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
    std::vector<Dictionary> branches;
    branches.push_back(MakeStoreNode(1));
    branches.push_back(MakeStoreNode(3));
    Dictionary config;
    config.Set("type", std::string("peer"));
    config.Set("branches", branches);
    Dictionary routing;
    routing.Set("strategy", std::string("length"));
    Dictionary params;
    params.SetNumbers("thresholds", std::vector<std::int64_t>{4});
    routing.Set("config", params);
    config.Set("routing", routing);

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

TEST(StoreAssemblerTest, CreatePeerRejectsMissingBranches)
{
    Dictionary missing;
    missing.Set("type", std::string("peer"));
    auto missingAssembler = StoreAssembler::Create(missing);
    ASSERT_FALSE(missingAssembler);
    EXPECT_EQ(missingAssembler.Error(), Status::InvalidParam());

    Dictionary empty;
    empty.Set("type", std::string("peer"));
    empty.Set("branches", std::vector<Dictionary>{});
    auto emptyAssembler = StoreAssembler::Create(empty);
    ASSERT_FALSE(emptyAssembler);
    EXPECT_EQ(emptyAssembler.Error(), Status::InvalidParam());
}

TEST(StoreAssemblerTest, CreatePeerRejectsNestedPeer)
{
    std::vector<Dictionary> branches;
    Dictionary nested;
    nested.Set("type", std::string("peer"));
    branches.push_back(nested);
    Dictionary config;
    config.Set("type", std::string("peer"));
    config.Set("branches", branches);
    auto assembler = StoreAssembler::Create(config);
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::InvalidParam());
    EXPECT_NE(assembler.Error().ToString().find("nest"), std::string::npos);
}

TEST(StoreAssemblerTest, CreatePeerRejectsUnknownStrategy)
{
    std::vector<Dictionary> branches;
    branches.push_back(MakeStoreNode());
    branches.push_back(MakeStoreNode());
    Dictionary config;
    config.Set("type", std::string("peer"));
    config.Set("branches", branches);
    Dictionary routing;
    routing.Set("strategy", std::string("no-such-strategy"));
    config.Set("routing", routing);
    auto assembler = StoreAssembler::Create(config);
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::NotFound());
}

TEST(StoreAssemblerTest, CreatePeerRejectsThresholdBranchMismatch)
{
    std::vector<Dictionary> branches;
    branches.push_back(MakeStoreNode());
    branches.push_back(MakeStoreNode());
    Dictionary config;
    config.Set("type", std::string("peer"));
    config.Set("branches", branches);
    Dictionary routing;
    routing.Set("strategy", std::string("length"));
    Dictionary params;
    params.SetNumbers("thresholds", std::vector<std::int64_t>{4, 8});
    routing.Set("config", params);
    config.Set("routing", routing);
    auto assembler = StoreAssembler::Create(config);
    ASSERT_FALSE(assembler);
    EXPECT_EQ(assembler.Error(), Status::InvalidParam());
}

}  // namespace
}  // namespace UC::Proxy
