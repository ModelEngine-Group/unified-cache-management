#include "kv_common/router.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace UC::KV {
namespace {

std::uint64_t Fnv1a64(const std::string& value)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : value) {
        hash ^= static_cast<std::uint64_t>(ch);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t StableHash(const std::string& value)
{
    auto hash = Fnv1a64(value);
    hash += 0x9e3779b97f4a7c15ULL;
    hash = (hash ^ (hash >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    hash = (hash ^ (hash >> 27U)) * 0x94d049bb133111ebULL;
    return hash ^ (hash >> 31U);
}

std::vector<NodeId> MakeNodeIds(std::size_t count)
{
    std::vector<NodeId> nodeIds;
    nodeIds.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        nodeIds.emplace_back(static_cast<NodeId>(index + 1));
    }
    return nodeIds;
}

std::vector<CacheKey> MakeKeys(std::size_t count)
{
    std::vector<CacheKey> keys;
    keys.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        keys.emplace_back("router-key-" + std::to_string(index));
    }
    return keys;
}

std::unordered_map<CacheKey, NodeId> CaptureKeyRoutes(const std::vector<NodeId>& nodeIds,
                                                      HashTableConfig config,
                                                      const std::vector<CacheKey>& keys)
{
    auto router = CreateRouter(nodeIds, StableHash, config);
    auto routesByNode = router->RouteKeys(keys);
    std::unordered_map<CacheKey, NodeId> routes;
    for (const auto& item : routesByNode) {
        for (auto index : item.second) { routes.emplace(keys[index], item.first); }
    }
    return routes;
}

void ExpectBalancedDistribution(const std::unordered_map<NodeId, std::vector<std::size_t>>& routes,
                                const std::vector<NodeId>& nodeIds, std::size_t keyCount,
                                double maxSkewRatio)
{
    const auto expectedPerNode =
        static_cast<double>(keyCount) / static_cast<double>(nodeIds.size());
    for (auto nodeId : nodeIds) {
        auto iter = routes.find(nodeId);
        const auto actual = iter == routes.end() ? 0.0 : static_cast<double>(iter->second.size());
        const auto skewRatio = std::abs(actual - expectedPerNode) / expectedPerNode;
        EXPECT_LE(skewRatio, maxSkewRatio) << "nodeId=" << nodeId << " actual=" << actual;
    }
}

double CalculateMigrationRatio(const std::unordered_map<CacheKey, NodeId>& oldRoutes,
                               const std::unordered_map<CacheKey, NodeId>& newRoutes)
{
    std::size_t movedCount = 0;
    std::size_t comparedCount = 0;
    for (const auto& item : oldRoutes) {
        auto iter = newRoutes.find(item.first);
        if (iter == newRoutes.end()) { continue; }
        ++comparedCount;
        if (iter->second != item.second) { ++movedCount; }
    }
    if (comparedCount == 0) { return 0.0; }
    return static_cast<double>(movedCount) / static_cast<double>(comparedCount);
}

TEST(RouterTest, RingHashDistributionIsBalanced)
{
    constexpr std::size_t kNodeCount = 16;
    constexpr std::size_t kKeyCount = 20000;
    constexpr double kMaxSkewRatio = 0.3;

    auto nodeIds = MakeNodeIds(kNodeCount);
    HashTableConfig config;
    config.type = HashTableType::RING_HASH;
    config.ringHash.virtualNodeCount = 256;
    auto router = CreateRouter(nodeIds, StableHash, config);

    auto keys = MakeKeys(kKeyCount);
    auto routes = router->RouteKeys(keys);

    ExpectBalancedDistribution(routes, nodeIds, kKeyCount, kMaxSkewRatio);
}

TEST(RouterTest, MaglevDistributionIsBalanced)
{
    constexpr std::size_t kNodeCount = 16;
    constexpr std::size_t kKeyCount = 20000;
    constexpr double kMaxSkewRatio = 0.3;

    auto nodeIds = MakeNodeIds(kNodeCount);
    HashTableConfig config;
    config.type = HashTableType::MAGLEV;
    config.maglev.tableSize = 65537;
    auto router = CreateRouter(nodeIds, StableHash, config);

    auto keys = MakeKeys(kKeyCount);
    auto routes = router->RouteKeys(keys);

    ExpectBalancedDistribution(routes, nodeIds, kKeyCount, kMaxSkewRatio);
}

TEST(RouterTest, ContiguousBlockAffinityRoutesKKeysTogether)
{
    constexpr std::uint64_t kContiguousBlockCount = 3;
    constexpr std::size_t kKeyCount = 12;

    HashTableConfig config;
    config.type = HashTableType::CONTIGUOUS_BLOCK_AFFINITY;
    config.contiguousBlockAffinity.blockCount = kContiguousBlockCount;
    config.contiguousBlockAffinity.dynamicAdjustEnabled = true;
    auto keys = MakeKeys(kKeyCount);
    auto routes = CaptureKeyRoutes(MakeNodeIds(8), config, keys);

    for (std::size_t begin = 0; begin < keys.size(); begin += kContiguousBlockCount) {
        auto routeIter = routes.find(keys[begin]);
        ASSERT_NE(routeIter, routes.end());
        const auto groupNodeId = routeIter->second;
        const auto end = std::min<std::size_t>(keys.size(), begin + kContiguousBlockCount);
        for (std::size_t index = begin; index < end; ++index) {
            auto iter = routes.find(keys[index]);
            ASSERT_NE(iter, routes.end());
            EXPECT_EQ(iter->second, groupNodeId) << "key=" << keys[index];
        }
    }
}

TEST(RouterTest, ContiguousBlockAffinityUsesConfiguredFullSpreadType)
{
    constexpr std::uint64_t kContiguousBlockCount = 4;
    constexpr std::size_t kKeyCount = 16;

    auto nodeIds = MakeNodeIds(8);
    HashTableConfig config;
    config.type = HashTableType::CONTIGUOUS_BLOCK_AFFINITY;
    config.contiguousBlockAffinity.blockCount = kContiguousBlockCount;
    config.contiguousBlockAffinity.fullSpreadType = HashTableType::MAGLEV_FULL_SPREAD;
    auto keys = MakeKeys(kKeyCount);
    auto routes = CaptureKeyRoutes(nodeIds, config, keys);

    HashTableConfig maglevConfig = config;
    maglevConfig.type = HashTableType::MAGLEV_FULL_SPREAD;
    auto maglevRouter = CreateRouter(nodeIds, StableHash, maglevConfig);

    for (std::size_t begin = 0; begin < keys.size(); begin += kContiguousBlockCount) {
        const auto expectedRoute = maglevRouter->RouteKeys({keys[begin]});
        ASSERT_EQ(expectedRoute.size(), std::size_t{1});
        const auto expectedNodeId = expectedRoute.begin()->first;
        const auto end = std::min<std::size_t>(keys.size(), begin + kContiguousBlockCount);
        for (std::size_t index = begin; index < end; ++index) {
            auto iter = routes.find(keys[index]);
            ASSERT_NE(iter, routes.end());
            EXPECT_EQ(iter->second, expectedNodeId) << "key=" << keys[index];
        }
    }
}

TEST(RouterTest, BatchTopKAffinityLimitsTouchedNodes)
{
    constexpr std::size_t kNodeCount = 16;
    constexpr std::size_t kKeyCount = 100;
    constexpr std::size_t kTopK = 3;

    HashTableConfig config;
    config.type = HashTableType::BATCH_TOPK_AFFINITY;
    config.batchTopKAffinity.topK = kTopK;
    config.batchTopKAffinity.dynamicAdjustEnabled = true;
    auto router = CreateRouter(MakeNodeIds(kNodeCount), StableHash, config);

    auto routes = router->RouteKeys(MakeKeys(kKeyCount));

    EXPECT_LE(routes.size(), kTopK);
    std::size_t routedKeyCount = 0;
    for (const auto& item : routes) { routedKeyCount += item.second.size(); }
    EXPECT_EQ(routedKeyCount, kKeyCount);
}

TEST(RouterTest, ContiguousBlockAffinityRejectsNonFullSpreadType)
{
    HashTableConfig config;
    config.type = HashTableType::CONTIGUOUS_BLOCK_AFFINITY;
    config.contiguousBlockAffinity.blockCount = 2;
    config.contiguousBlockAffinity.fullSpreadType = HashTableType::BATCH_TOPK_AFFINITY;
    auto router = CreateRouter(MakeNodeIds(8), StableHash, config);

    auto routes = router->RouteKeys(MakeKeys(8));

    EXPECT_TRUE(routes.empty());
}

TEST(RouterTest, RingHashMigrationRatioIsBoundedWhenNodeIsAdded)
{
    constexpr std::size_t kOldNodeCount = 16;
    constexpr std::size_t kKeyCount = 20000;
    constexpr double kMaxMigrationRatio = 0.15;

    HashTableConfig config;
    config.type = HashTableType::RING_HASH;
    config.ringHash.virtualNodeCount = 512;
    auto keys = MakeKeys(kKeyCount);
    auto oldRoutes = CaptureKeyRoutes(MakeNodeIds(kOldNodeCount), config, keys);
    auto newRoutes = CaptureKeyRoutes(MakeNodeIds(kOldNodeCount + 1), config, keys);
    auto migrationRatio = CalculateMigrationRatio(oldRoutes, newRoutes);

    std::cout << "RingHash migration ratio after adding one node: " << migrationRatio << std::endl;
    EXPECT_LE(migrationRatio, kMaxMigrationRatio);
}

TEST(RouterTest, MaglevMigrationRatioIsBoundedWhenNodeIsAdded)
{
    constexpr std::size_t kOldNodeCount = 16;
    constexpr std::size_t kKeyCount = 20000;
    constexpr double kMaxMigrationRatio = 0.15;

    HashTableConfig config;
    config.type = HashTableType::MAGLEV;
    config.maglev.tableSize = 65537;
    auto keys = MakeKeys(kKeyCount);
    auto oldRoutes = CaptureKeyRoutes(MakeNodeIds(kOldNodeCount), config, keys);
    auto newRoutes = CaptureKeyRoutes(MakeNodeIds(kOldNodeCount + 1), config, keys);
    auto migrationRatio = CalculateMigrationRatio(oldRoutes, newRoutes);

    std::cout << "Maglev migration ratio after adding one node: " << migrationRatio << std::endl;
    EXPECT_LE(migrationRatio, kMaxMigrationRatio);
}

}  // namespace
}  // namespace UC::KV
