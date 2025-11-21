#ifndef STATS_MONITOR_H
#define STATS_MONITOR_H

#include "stats/istats.h"
#include "StatsRegistry.h"
#include <unordered_map>
#include <mutex>
#include <memory>

class UCMStatsMonitor {
public:
    UCMStatsMonitor() {
        auto& registry = StatsRegistry::getInstance();
        auto names = registry.getRegisteredStatsNames();
        for (const auto& name : names) {
            stats_map_[name] = registry.createStats(name);
        }
    }

    template <typename T>
    T* getStats() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string name = T().getName();
        auto it = stats_map_.find(name);
        if (it != stats_map_.end()) {
            return dynamic_cast<T*>(it->second.get());
        }
        return nullptr;
    }

    void resetAllStats() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : stats_map_) {
            pair.second->reset();
        }
    }

    std::unordered_map<std::string, std::unique_ptr<IStats>> getStatsSnapshot() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::unordered_map<std::string, std::unique_ptr<IStats>> snapshot;
        for (const auto& pair : stats_map_) {
            snapshot[pair.first] = pair.second->clone();
        }
        return snapshot;
    }

    void update_stats(std::string name, unordered_map<std::string, any> params) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stats_map_.find(name);
        if (it != stats_map_.end()) {
            stats_map_[name]->update(params);
        }
    }

private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<IStats>> stats_map_;
};