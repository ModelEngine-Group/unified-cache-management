#ifndef UNIFIEDCACHE_REGISTRY_H
#define UNIFIEDCACHE_REGISTRY_H

#include "stats/istats.h"
#include <unordered_map>
#include <functional>
#include <mutex>

namespace UC::Metrics {

using Creator = std::unique_ptr<IStats>(*)();

class StatsRegistry {
public:
    static StatsRegistry& GetInstance();

    static void RegisterStats(std::string name, Creator creator);

    std::unique_ptr<IStats> CreateStats(const std::string& name);

    std::vector<std::string> GetRegisteredStatsNames();

private:
    StatsRegistry() = default;
    ~StatsRegistry() = default;
    StatsRegistry(const StatsRegistry&) = delete;
    StatsRegistry& operator=(const StatsRegistry&) = delete;

    std::mutex mutex_;
    std::unordered_map<std::string, Creator> registry_;
};

}  

#endif  // UNIFIEDCACHE_REGISTRY_H