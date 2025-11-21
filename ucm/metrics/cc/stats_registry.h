#ifndef STATS_REGISTRY_H
#define STATS_REGISTRY_H

#include "stats/istats.h"   // 仅含接口，无实现
#include "stats/ucm_stats.h" // 具体 Stats 实现
#include <unordered_map>
#include <functional>
#include <mutex>

//  ========== 1. 单例工厂 ==========
class StatsRegistry {
public:
    static StatsRegistry& getInstance() {
        static StatsRegistry inst;
        inst.registerStats<UCMStats>();
        return inst;
    }

    template <typename T>
    void registerStats() {
        static_assert(std::is_base_of_v<IStats, T>);
        std::lock_guard lk(mutex_);
        registry_[T().name()] = []() -> std::unique_ptr<IStats> {
            return std::make_unique<T>();
        };
    }

    std::unique_ptr<IStats> getStats(const std::string& name) {
        std::lock_guard lk(mutex_);
        if (auto it = registry_.find(name); it != registry_.end())
            return it->second();
        return nullptr;
    }

    std::vector<std::string> getRegisteredStatsNames() {
        std::lock_guard lk(mutex_);
        std::vector<std::string> names;
        names.reserve(registry_.size());
        for (auto& [n, _] : registry_) names.push_back(n);
        return names;
    }

private:
    StatsRegistry() = default;
    ~StatsRegistry() = default;
    StatsRegistry(const StatsRegistry&) = delete;
    StatsRegistry& operator=(const StatsRegistry&) = delete;

    std::mutex mutex_;
    std::unordered_map<std::string, std::function<std::unique_ptr<IStats>()>> registry_;
};

#endif // STATS_REGISTRY_H