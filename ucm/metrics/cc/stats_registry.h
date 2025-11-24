#ifndef UCM_METRICS_REGISTRY_H
#define UCM_METRICS_REGISTRY_H

#include "stats/istats.h"
#include <unordered_map>
#include <functional>
#include <mutex>

using Creator = std::unique_ptr<IStats>(*)();

class StatsRegistry {
public:
    static StatsRegistry& getInstance();

    static void registerStats(std::string name, Creator creator);

    std::unique_ptr<IStats> createStats(const std::string& name);

    std::vector<std::string> getRegisteredStatsNames();

private:
    StatsRegistry() = default;
    ~StatsRegistry() = default;
    StatsRegistry(const StatsRegistry&) = delete;
    StatsRegistry& operator=(const StatsRegistry&) = delete;

    static std::mutex mutex_;
    static std::unordered_map<std::string, Creator> registry_;
};

#endif  // UCM_METRICS_REGISTRY_H