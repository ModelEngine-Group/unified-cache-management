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
#include "router/router.h"
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kv::test {
namespace {

using Routes = std::unordered_map<NodeId, std::vector<Router::EntryIndex>>;

// 固定 hash，保证测试可复现，同时保留 key 之间的差异。
std::uint64_t StableHash(const std::string& key)
{
    std::uint64_t value = 1469598103934665603ULL;
    for (const unsigned char byte : key) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    return value;
}

RouterConfig MakeConfig(RouterType type)
{
    RouterConfig config;
    config.type = type;
    return config;
}

std::vector<Key> MakeKeys(std::size_t count)
{
    std::vector<Key> keys;
    keys.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        keys.push_back("key-" + std::to_string(index));
    }
    return keys;
}

std::vector<Router::EntryIndex> FlattenIndices(const Routes& routes)
{
    std::vector<Router::EntryIndex> indices;
    for (const auto& item : routes) {
        indices.insert(indices.end(), item.second.begin(), item.second.end());
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

// 每个输入下标必须恰好出现一次，且都落在某个节点下。
void ExpectCoversAllIndices(const Routes& routes, std::size_t keyCount)
{
    ASSERT_EQ(FlattenIndices(routes).size(), keyCount);
    const auto indices = FlattenIndices(routes);
    for (std::size_t index = 0; index < indices.size(); ++index) {
        EXPECT_EQ(indices[index], index);
    }
}

void ExpectNodesAreSubset(const Routes& routes, const std::vector<NodeId>& nodes)
{
    for (const auto& item : routes) {
        EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), item.first) != nodes.end());
        EXPECT_FALSE(item.second.empty());
    }
}

NodeId RouteOne(const Router& router, const Key& key)
{
    const auto routes = router.RouteKeys({key});
    if (routes.empty()) { return kInvalidNodeId; }
    return routes.begin()->first;
}

// 用可控的 key→node 映射验证 Router 基类 RouteKeys 的分组/过滤语义。
class IndexedRouter final : public Router {
public:
    explicit IndexedRouter(std::unordered_map<std::string, NodeId> owners)
        : Router(nullptr), owners_(std::move(owners))
    {
    }

private:
    NodeId RouteKey(const Key& key) const override
    {
        const auto iter = owners_.find(key);
        return iter == owners_.end() ? kInvalidNodeId : iter->second;
    }

    std::unordered_map<std::string, NodeId> owners_;
};

TEST(RouterTest, RouteKeysGroupsIndicesAndSkipsUnroutedKeys)
{
    const IndexedRouter router({
        {"a", 1},
        {"b", 2},
        {"c", 1}
    });

    const auto routes = router.RouteKeys({"a", "b", "c", "missing"});

    ASSERT_EQ(routes.size(), 2U);
    EXPECT_EQ(routes.at(1), std::vector<Router::EntryIndex>({0, 2}));
    EXPECT_EQ(routes.at(2), std::vector<Router::EntryIndex>({1}));
}

TEST(RouterTest, RouteKeysReturnsEmptyForEmptyInput)
{
    const RingHashRouter router({1}, StableHash, MakeConfig(RouterType::RING_HASH_FULL_SPREAD));

    EXPECT_TRUE(router.RouteKeys({}).empty());
}

TEST(RingHashRouterTest, RoutesEveryKeyToConfiguredNodes)
{
    const std::vector<NodeId> nodes{11, 22, 33};
    const RingHashRouter router(nodes, StableHash, MakeConfig(RouterType::RING_HASH_FULL_SPREAD));
    const auto keys = MakeKeys(64);

    const auto routes = router.RouteKeys(keys);

    ExpectCoversAllIndices(routes, keys.size());
    ExpectNodesAreSubset(routes, nodes);
}

TEST(RingHashRouterTest, RoutingIsDeterministic)
{
    const std::vector<NodeId> nodes{11, 22, 33};
    const RingHashRouter first(nodes, StableHash, MakeConfig(RouterType::RING_HASH_FULL_SPREAD));
    const RingHashRouter second(nodes, StableHash, MakeConfig(RouterType::RING_HASH_FULL_SPREAD));

    for (const auto& key : MakeKeys(32)) {
        EXPECT_EQ(RouteOne(first, key), RouteOne(second, key)) << key;
    }
}

TEST(RingHashRouterTest, SingleNodeOwnsEveryKey)
{
    const RingHashRouter router({7}, StableHash, MakeConfig(RouterType::RING_HASH_FULL_SPREAD));

    const auto routes = router.RouteKeys(MakeKeys(16));

    ASSERT_EQ(routes.size(), 1U);
    EXPECT_EQ(routes.begin()->first, NodeId{7});
    EXPECT_EQ(routes.begin()->second.size(), 16U);
}

TEST(RingHashRouterTest, DuplicateAndInvalidNodesAreIgnored)
{
    const RingHashRouter router({7, 7, kInvalidNodeId}, StableHash,
                                MakeConfig(RouterType::RING_HASH_FULL_SPREAD));

    const auto routes = router.RouteKeys(MakeKeys(16));

    ASSERT_EQ(routes.size(), 1U);
    EXPECT_EQ(routes.begin()->first, NodeId{7});
}

TEST(RingHashRouterTest, EmptyNodeSetRoutesNothing)
{
    const RingHashRouter router({}, StableHash, MakeConfig(RouterType::RING_HASH_FULL_SPREAD));

    EXPECT_TRUE(router.RouteKeys(MakeKeys(4)).empty());
    EXPECT_EQ(RouteOne(router, "key-0"), kInvalidNodeId);
}

TEST(RingHashRouterTest, HonoursConfiguredVirtualNodeCount)
{
    auto config = MakeConfig(RouterType::RING_HASH_FULL_SPREAD);
    config.ringHash.virtualNodeCount = 8;
    const std::vector<NodeId> nodes{1, 2, 3, 4};
    const RingHashRouter router(nodes, StableHash, config);
    const auto keys = MakeKeys(64);

    const auto routes = router.RouteKeys(keys);

    ExpectCoversAllIndices(routes, keys.size());
    ExpectNodesAreSubset(routes, nodes);
}

TEST(MaglevRouterTest, RoutesEveryKeyToConfiguredNodes)
{
    const std::vector<NodeId> nodes{11, 22, 33};
    const MaglevRouter router(nodes, StableHash, MakeConfig(RouterType::MAGLEV_FULL_SPREAD));
    const auto keys = MakeKeys(128);

    const auto routes = router.RouteKeys(keys);

    ExpectCoversAllIndices(routes, keys.size());
    ExpectNodesAreSubset(routes, nodes);
}

TEST(MaglevRouterTest, DistributesKeysAcrossAllNodes)
{
    const std::vector<NodeId> nodes{1, 2, 3};
    const MaglevRouter router(nodes, StableHash, MakeConfig(RouterType::MAGLEV_FULL_SPREAD));

    const auto routes = router.RouteKeys(MakeKeys(256));

    EXPECT_EQ(routes.size(), nodes.size());
}

TEST(MaglevRouterTest, NonPrimeTableSizeStillBuildsUsableTable)
{
    auto config = MakeConfig(RouterType::MAGLEV_FULL_SPREAD);
    config.maglev.tableSize = 1000;  // 非质数，实现会回退到默认表大小
    const std::vector<NodeId> nodes{1, 2, 3};
    const MaglevRouter router(nodes, StableHash, config);
    const auto keys = MakeKeys(64);

    const auto routes = router.RouteKeys(keys);

    ExpectCoversAllIndices(routes, keys.size());
    ExpectNodesAreSubset(routes, nodes);
}

TEST(MaglevRouterTest, EmptyNodeSetRoutesNothing)
{
    const MaglevRouter router({}, StableHash, MakeConfig(RouterType::MAGLEV_FULL_SPREAD));

    EXPECT_TRUE(router.RouteKeys(MakeKeys(4)).empty());
    EXPECT_EQ(RouteOne(router, "key-0"), kInvalidNodeId);
}

TEST(ContiguousBlockAffinityRouterTest, GroupsEachBlockOnFirstKeyOwner)
{
    auto config = MakeConfig(RouterType::CONTIGUOUS_BLOCK_AFFINITY);
    config.contiguousBlockAffinity.blockCount = 3;
    config.contiguousBlockAffinity.fullSpreadType = RouterType::RING_HASH_FULL_SPREAD;
    const std::vector<NodeId> nodes{11, 22};
    const ContiguousBlockAffinityRouter router(nodes, StableHash, config);
    const auto keys = MakeKeys(8);

    const auto routes = router.RouteKeys(keys);

    ExpectCoversAllIndices(routes, keys.size());
    ExpectNodesAreSubset(routes, nodes);

    std::vector<NodeId> owners(keys.size(), kInvalidNodeId);
    for (const auto& item : routes) {
        for (const auto index : item.second) { owners[index] = item.first; }
    }

    // 每 3 个 key 应归属到该块第一个 key 的底层 RingHash 归属节点。
    const RingHashRouter ringHash(nodes, StableHash, MakeConfig(RouterType::RING_HASH_FULL_SPREAD));
    for (std::size_t begin = 0; begin < keys.size(); begin += 3) {
        const auto expected = RouteOne(ringHash, keys[begin]);
        const auto end = std::min(keys.size(), begin + 3);
        for (std::size_t index = begin; index < end; ++index) {
            EXPECT_EQ(owners[index], expected) << "index=" << index;
        }
    }
}

TEST(ContiguousBlockAffinityRouterTest, ZeroBlockCountBehavesAsSingleKeyBlocks)
{
    auto config = MakeConfig(RouterType::CONTIGUOUS_BLOCK_AFFINITY);
    config.contiguousBlockAffinity.blockCount = 0;  // 实现会上调为 1
    const std::vector<NodeId> nodes{11, 22};
    const ContiguousBlockAffinityRouter router(nodes, StableHash, config);
    const RingHashRouter ringHash(nodes, StableHash, MakeConfig(RouterType::RING_HASH_FULL_SPREAD));

    for (const auto& key : MakeKeys(16)) {
        EXPECT_EQ(RouteOne(router, key), RouteOne(ringHash, key)) << key;
    }
}

TEST(ContiguousBlockAffinityRouterTest, UnsupportedFullSpreadTypeRoutesNothing)
{
    for (const auto fullSpreadType :
         {RouterType::BATCH_TOPK_AFFINITY, RouterType::CONTIGUOUS_BLOCK_AFFINITY}) {
        auto config = MakeConfig(RouterType::CONTIGUOUS_BLOCK_AFFINITY);
        config.contiguousBlockAffinity.fullSpreadType = fullSpreadType;
        const ContiguousBlockAffinityRouter router({11, 22}, StableHash, config);

        EXPECT_TRUE(router.RouteKeys(MakeKeys(4)).empty());
        EXPECT_EQ(RouteOne(router, "key-0"), kInvalidNodeId);
    }
}

TEST(BatchTopKAffinityRouterTest, RoutesWithinTopKCandidates)
{
    auto config = MakeConfig(RouterType::BATCH_TOPK_AFFINITY);
    config.batchTopKAffinity.topK = 2;
    const std::vector<NodeId> nodes{1, 2, 3, 4};
    const BatchTopKAffinityRouter router(nodes, StableHash, config);
    const auto keys = MakeKeys(32);

    const auto routes = router.RouteKeys(keys);

    ExpectCoversAllIndices(routes, keys.size());
    ExpectNodesAreSubset(routes, nodes);
    EXPECT_LE(routes.size(), 2U);
}

TEST(BatchTopKAffinityRouterTest, ZeroTopKFallsBackToSingleCandidate)
{
    auto config = MakeConfig(RouterType::BATCH_TOPK_AFFINITY);
    config.batchTopKAffinity.topK = 0;  // 实现会上调为 1
    const BatchTopKAffinityRouter router({1, 2, 3}, StableHash, config);
    const auto keys = MakeKeys(16);

    const auto routes = router.RouteKeys(keys);

    EXPECT_EQ(routes.size(), 1U);
    ExpectCoversAllIndices(routes, keys.size());
}

TEST(BatchTopKAffinityRouterTest, RoutingIsDeterministic)
{
    auto config = MakeConfig(RouterType::BATCH_TOPK_AFFINITY);
    config.batchTopKAffinity.topK = 2;
    const std::vector<NodeId> nodes{1, 2, 3};
    const BatchTopKAffinityRouter first(nodes, StableHash, config);
    const BatchTopKAffinityRouter second(nodes, StableHash, config);
    const auto keys = MakeKeys(16);

    const auto firstRoutes = first.RouteKeys(keys);
    const auto secondRoutes = second.RouteKeys(keys);

    EXPECT_EQ(firstRoutes, secondRoutes);
}

TEST(BatchTopKAffinityRouterTest, EmptyNodeSetRoutesNothing)
{
    const BatchTopKAffinityRouter router({}, StableHash,
                                         MakeConfig(RouterType::BATCH_TOPK_AFFINITY));

    EXPECT_TRUE(router.RouteKeys(MakeKeys(4)).empty());
    EXPECT_EQ(RouteOne(router, "key-0"), kInvalidNodeId);
}

TEST(CreateRouterTest, DispatchesEveryRouterType)
{
    const std::vector<NodeId> nodes{11, 22, 33};
    const std::vector<RouterType> types{
        RouterType::RING_HASH_FULL_SPREAD,
        RouterType::MAGLEV_FULL_SPREAD,
        RouterType::CONTIGUOUS_BLOCK_AFFINITY,
        RouterType::BATCH_TOPK_AFFINITY,
    };

    for (const auto type : types) {
        const auto router = CreateRouter(nodes, StableHash, MakeConfig(type));
        ASSERT_NE(router, nullptr);
        const auto routes = router->RouteKeys(MakeKeys(32));
        ExpectCoversAllIndices(routes, 32);
        ExpectNodesAreSubset(routes, nodes);
    }
}

TEST(CreateRouterTest, UnknownTypeFallsBackToRingHash)
{
    const std::vector<NodeId> nodes{11, 22};
    const auto fallback = CreateRouter(nodes, StableHash, MakeConfig(static_cast<RouterType>(999)));
    const auto ringHash =
        CreateRouter(nodes, StableHash, MakeConfig(RouterType::RING_HASH_FULL_SPREAD));
    ASSERT_NE(fallback, nullptr);
    ASSERT_NE(ringHash, nullptr);

    for (const auto& key : MakeKeys(32)) {
        EXPECT_EQ(RouteOne(*fallback, key), RouteOne(*ringHash, key)) << key;
    }
}

TEST(CreateRouterTest, EmptyNodeSetCreatesUsableRouter)
{
    for (const auto type :
         {RouterType::RING_HASH_FULL_SPREAD, RouterType::MAGLEV_FULL_SPREAD,
          RouterType::CONTIGUOUS_BLOCK_AFFINITY, RouterType::BATCH_TOPK_AFFINITY}) {
        const auto router = CreateRouter({}, StableHash, MakeConfig(type));
        ASSERT_NE(router, nullptr);
        EXPECT_TRUE(router->RouteKeys(MakeKeys(4)).empty());
    }
}

}  // namespace
}  // namespace kv::test
