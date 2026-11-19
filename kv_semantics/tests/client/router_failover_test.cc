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
#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "router/router.h"

namespace kv::test {
namespace {

std::map<std::size_t, NodeId> Flatten(
    const std::unordered_map<NodeId, std::vector<std::size_t>>& routes)
{
    std::map<std::size_t, NodeId> out;
    for (const auto& [node, indexes] : routes) {
        for (auto index : indexes) { out[index] = node; }
    }
    return out;
}

RouterConfig MakeRingHashConfig()
{
    RouterConfig config;
    config.type = RouterType::RING_HASH_FULL_SPREAD;
    config.ringHash.virtualNodeCount = 128;
    return config;
}

std::vector<Key> MakeKeys(int count)
{
    std::vector<Key> keys;
    keys.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) { keys.push_back("blk-" + std::to_string(index)); }
    return keys;
}

TEST(RouterFailoverTest, EmptyUnavailableSetMatchesLegacyRouting)
{
    const std::vector<NodeId> nodes{10, 20, 30, 40};
    RingHashRouter router(nodes, HashFunction{}, MakeRingHashConfig());
    const auto keys = MakeKeys(2000);

    const auto legacy = Flatten(router.RouteKeys(keys));
    const auto failover = router.RouteKeys(keys, {});

    EXPECT_TRUE(Flatten(failover) == legacy);
}

TEST(RouterFailoverTest, UnavailablePrimaryNeverReturnsPrimary)
{
    const std::vector<NodeId> nodes{10, 20, 30, 40};
    RingHashRouter router(nodes, HashFunction{}, MakeRingHashConfig());
    const auto keys = MakeKeys(2000);
    const auto primary = Flatten(router.RouteKeys(keys));

    std::size_t routed = 0;
    std::size_t unrouted = 0;
    std::unordered_set<NodeId> alternates;
    for (std::size_t index = 0; index < keys.size(); ++index) {
        const NodeId home = primary.at(index);
        const auto routes = router.RouteKeys({keys[index]}, {home});
        if (routes.empty()) {
            // fallback owner equals the primary owner: entry has no available node.
            ++unrouted;
            continue;
        }
        ASSERT_EQ(routes.size(), std::size_t{1});
        const NodeId alternate = routes.begin()->first;
        EXPECT_NE(alternate, home);
        EXPECT_TRUE(std::find(nodes.begin(), nodes.end(), alternate) != nodes.end());
        alternates.insert(alternate);
        ++routed;
    }
    EXPECT_GT(routed, std::size_t{0});
    EXPECT_GT(alternates.size(), std::size_t{1});
}

TEST(RouterFailoverTest, SingleUnavailableNodeHasNoFallback)
{
    RingHashRouter router({7}, HashFunction{}, MakeRingHashConfig());
    const auto keys = MakeKeys(4);

    const auto routes = router.RouteKeys(keys, {7});

    EXPECT_TRUE(routes.empty());
}

TEST(RouterFailoverTest, AllNodesUnavailableProducesNoRoute)
{
    const std::vector<NodeId> nodes{10, 20, 30, 40};
    RingHashRouter router(nodes, HashFunction{}, MakeRingHashConfig());
    const auto keys = MakeKeys(500);
    const std::unordered_set<NodeId> unavailable(nodes.begin(), nodes.end());

    const auto routes = router.RouteKeys(keys, unavailable);

    EXPECT_TRUE(routes.empty());
}

TEST(RouterFailoverTest, IsDeterministicAcrossRepeatedCalls)
{
    const std::vector<NodeId> nodes{10, 20, 30, 40};
    RingHashRouter router(nodes, HashFunction{}, MakeRingHashConfig());
    const auto keys = MakeKeys(200);
    const UnavailableNodeSet unavailable{10};

    for (const auto& key : keys) {
        const auto first = router.RouteKeys({key}, unavailable);
        const auto second = router.RouteKeys({key}, unavailable);
        EXPECT_TRUE(first == second);
    }
}

TEST(RouterFailoverTest, OtherStrategiesIgnoreUnavailableSet)
{
    const std::vector<NodeId> nodes{10, 20, 30, 40};
    const auto keys = MakeKeys(512);
    const UnavailableNodeSet unavailable{10, 20};

    RouterConfig batchConfig = MakeRingHashConfig();
    batchConfig.type = RouterType::BATCH_TOPK_AFFINITY;
    batchConfig.batchTopKAffinity.topK = 2;
    auto batch = CreateRouter(nodes, HashFunction{}, batchConfig);
    ASSERT_TRUE(batch != nullptr);
    EXPECT_TRUE(batch->RouteKeys(keys, unavailable) == batch->RouteKeys(keys));

    RouterConfig contiguousConfig = MakeRingHashConfig();
    contiguousConfig.type = RouterType::CONTIGUOUS_BLOCK_AFFINITY;
    contiguousConfig.contiguousBlockAffinity.blockCount = 4;
    auto contiguous = CreateRouter(nodes, HashFunction{}, contiguousConfig);
    ASSERT_TRUE(contiguous != nullptr);
    EXPECT_TRUE(contiguous->RouteKeys(keys, unavailable) == contiguous->RouteKeys(keys));

    RouterConfig maglevConfig = MakeRingHashConfig();
    maglevConfig.type = RouterType::MAGLEV_FULL_SPREAD;
    auto maglev = CreateRouter(nodes, HashFunction{}, maglevConfig);
    ASSERT_TRUE(maglev != nullptr);
    EXPECT_TRUE(maglev->RouteKeys(keys, unavailable) == maglev->RouteKeys(keys));
}

TEST(RouterFailoverTest, AllKeysAreCoveredWhenNoNodeIsUnavailable)
{
    const std::vector<NodeId> nodes{10, 20, 30, 40};
    RingHashRouter router(nodes, HashFunction{}, MakeRingHashConfig());
    const auto keys = MakeKeys(500);

    const auto routes = router.RouteKeys(keys, {});

    std::size_t covered = 0;
    for (const auto& [node, indexes] : routes) {
        EXPECT_FALSE(indexes.empty());
        covered += indexes.size();
    }
    EXPECT_EQ(covered, keys.size());
}

TEST(RouterFailoverTest, UnavailableSetKeepsPartitionProperty)
{
    const std::vector<NodeId> nodes{10, 20, 30, 40};
    RingHashRouter router(nodes, HashFunction{}, MakeRingHashConfig());
    const auto keys = MakeKeys(1000);
    const auto primary = Flatten(router.RouteKeys(keys));
    const UnavailableNodeSet unavailable{10, 30};

    const auto routes = router.RouteKeys(keys, unavailable);

    std::vector<int> seen(keys.size(), 0);
    for (const auto& [node, indexes] : routes) {
        EXPECT_EQ(unavailable.count(node), std::size_t{0});
        for (auto index : indexes) {
            ++seen[index];
            if (unavailable.count(primary.at(index)) == 0) {
                // Primary owner is healthy: failover routing must not move the key.
                EXPECT_EQ(node, primary.at(index));
            } else {
                // Primary owner is unavailable: the key must move to another node.
                EXPECT_NE(node, primary.at(index));
            }
        }
    }
    for (auto count : seen) { EXPECT_LE(count, 1); }
}

TEST(RouterFailoverTest, DuplicateAndInvalidNodeIdsDoNotChangeRouting)
{
    const auto keys = MakeKeys(300);
    RingHashRouter deduped({10, 20, 30}, HashFunction{}, MakeRingHashConfig());
    RingHashRouter noisy({10, kInvalidNodeId, 20, 30, 10}, HashFunction{}, MakeRingHashConfig());

    EXPECT_TRUE(deduped.RouteKeys(keys, {}) == noisy.RouteKeys(keys, {}));
    EXPECT_TRUE(deduped.RouteKeys(keys, {10}) == noisy.RouteKeys(keys, {10}));
}

TEST(RouterFailoverTest, EmptyNodeListProducesNoRoute)
{
    RingHashRouter router({}, HashFunction{}, MakeRingHashConfig());
    const auto keys = MakeKeys(10);

    EXPECT_TRUE(router.RouteKeys(keys, {}).empty());
    EXPECT_TRUE(router.RouteKeys(keys, {1}).empty());
}

TEST(RouterFailoverTest, DefaultContractDelegatesToSingleRing)
{
    class FixedNodeRouter final : public Router {
    public:
        explicit FixedNodeRouter(NodeId nodeId) : Router(HashFunction{}), nodeId_(nodeId) {}

        // Keeps the base overload (keys, unavailableNodes) visible.
        using Router::RouteKeys;

        std::unordered_map<NodeId, std::vector<EntryIndex>> RouteKeys(
            const std::vector<Key>& keys) const override
        {
            std::unordered_map<NodeId, std::vector<EntryIndex>> routes;
            for (EntryIndex index = 0; index < keys.size(); ++index) {
                routes[nodeId_].push_back(index);
            }
            return routes;
        }

    private:
        NodeId RouteKey(const Key&) const override { return nodeId_; }

        NodeId nodeId_;
    };

    const FixedNodeRouter router{7};
    const auto keys = MakeKeys(20);

    // The base class overload must dispatch to the implementation's own single-ring routing.
    EXPECT_TRUE(router.RouteKeys(keys, UnavailableNodeSet{7}) == router.RouteKeys(keys));
}

}  // namespace
}  // namespace kv::test
