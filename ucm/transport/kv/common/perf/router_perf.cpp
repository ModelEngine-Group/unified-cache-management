#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "kv_common/router.h"

namespace {

using Clock = std::chrono::steady_clock;

UC::KV::CacheKey MakeCacheKey(std::uint64_t value)
{
    UC::KV::CacheKey key{};
    std::memcpy(key.data(), &value, key.size());
    return key;
}

struct PerfResult {
    std::string name;
    std::uint64_t buildNs{0};
    std::uint64_t routeNs{0};
    std::size_t routedKeyCount{0};
    std::size_t routeBucketCount{0};
};

std::vector<UC::KV::NodeId> MakeNodeIds(std::size_t count)
{
    std::vector<UC::KV::NodeId> nodeIds;
    nodeIds.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        nodeIds.emplace_back(static_cast<UC::KV::NodeId>(index + 1));
    }
    return nodeIds;
}

std::vector<UC::KV::CacheKey> MakeKeys(std::size_t count)
{
    std::vector<UC::KV::CacheKey> keys;
    keys.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        keys.emplace_back(MakeCacheKey(static_cast<std::uint64_t>(index + 1)));
    }
    return keys;
}

std::uint64_t ElapsedNs(Clock::time_point start)
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

PerfResult RunPerf(const std::string& name, const std::vector<UC::KV::NodeId>& nodeIds,
                   const std::vector<UC::KV::CacheKey>& keys, UC::KV::RouterConfig config)
{
    auto buildStart = Clock::now();
    auto router = UC::KV::CreateRouter(nodeIds, nullptr, config);
    auto buildNs = ElapsedNs(buildStart);
    if (router == nullptr) { return PerfResult{name, buildNs, 0, 0, 0}; }

    auto routeStart = Clock::now();
    auto routes = router->RouteKeys(keys);
    auto routeNs = ElapsedNs(routeStart);

    std::size_t routedKeyCount = 0;
    for (const auto& item : routes) { routedKeyCount += item.second.size(); }
    return PerfResult{name, buildNs, routeNs, routedKeyCount, routes.size()};
}

void PrintResult(const PerfResult& result)
{
    const auto buildMs = static_cast<double>(result.buildNs) / 1000000.0;
    const auto routeMs = static_cast<double>(result.routeNs) / 1000000.0;
    const auto keysPerSec = result.routeNs == 0
                                ? 0.0
                                : static_cast<double>(result.routedKeyCount) * 1000000000.0 /
                                      static_cast<double>(result.routeNs);

    std::cout << std::left << std::setw(18) << result.name << " build_ms=" << std::fixed
              << std::setprecision(3) << buildMs << " route_ms=" << routeMs
              << " routed_keys=" << result.routedKeyCount
              << " route_buckets=" << result.routeBucketCount
              << " keys_per_sec=" << std::setprecision(0) << keysPerSec << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    const auto nodeCount = argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 64;
    const auto keyCount = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 1000000;

    auto nodeIds = MakeNodeIds(nodeCount);
    auto keys = MakeKeys(keyCount);

    UC::KV::RouterConfig ringConfig;
    ringConfig.type = UC::KV::RouterType::RING_HASH_FULL_SPREAD;
    ringConfig.ringHash.virtualNodeCount = 256;

    UC::KV::RouterConfig maglevConfig;
    maglevConfig.type = UC::KV::RouterType::MAGLEV_FULL_SPREAD;
    maglevConfig.maglev.tableSize = 65537;

    std::cout << "router perf: nodes=" << nodeCount << " keys=" << keyCount << '\n';
    PrintResult(RunPerf("ring_hash", nodeIds, keys, ringConfig));
    PrintResult(RunPerf("maglev", nodeIds, keys, maglevConfig));
    return 0;
}
