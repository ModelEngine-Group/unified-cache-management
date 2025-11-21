#ifndef STATS_REGISTRY_H
#define STATS_REGISTRY_H

#include "IStats.h"
#include <unordered_map>
#include <functional>
#include <mutex>

// 全局 stats 注册器，管理所有 stats 类型的创建工厂
class StatsRegistry {
public:
    static StatsRegistry& getInstance() {
        static StatsRegistry instance;
        return instance;
    }

    template <typename T>
    void registerStats() {
        static_assert(std::is_base_of<IStats, T>::value, "T must inherit from IStats");
        std::lock_guard<std::mutex> lock(mutex_);
        registry_[T().getName()] = []() -> std::unique_ptr<IStats> {
            return std::make_unique<T>();
        };
    }

    std::unique_ptr<IStats> createStats(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = registry_.find(name);
        if (it != registry_.end()) {
            return it->second();
        }
        return nullptr;
    }

    std::vector<std::string> getRegisteredStatsNames() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> names;
        for (const auto& pair : registry_) {
            names.push_back(pair.first);
        }
        return names;
    }

private:
    StatsRegistry() = default;
    ~StatsRegistry() = default;
    StatsRegistry(const StatsRegistry&) = delete;
    StatsRegistry& operator=(const StatsRegistry&) = delete;

    std::mutex mutex_;
    std::unordered_map<std::string, std::function<std::unique_ptr<IStats>()>> registry_;
}
#endif // STATS_REGISTRY_H