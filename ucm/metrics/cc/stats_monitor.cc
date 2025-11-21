#include "stats/istats.h"
#include "stats_registry.h"
#include "stats_monitor.h"
#include <mutex>
#include <vector>

UCMStatsMonitor::UCMStatsMonitor() {
    auto& registry = StatsRegistry::getInstance();
    for (const auto& name : registry.getRegisteredStatsNames()) {
        stats_map_[name] = registry.getStats(name);
    }
}

std::unordered_map<std::string, std::vector<double>> UCMStatsMonitor::getStats(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_map_[name]->data();
}

void UCMStatsMonitor::updateStats(
    const std::string& name,
    const std::unordered_map<std::string, double>& params)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_map_.find(name);
    if (it != stats_map_.end()) {
        it->second->update(params);
    }
}

std::unordered_map<std::string, std::unique_ptr<IStats>>
UCMStatsMonitor::getStatsSnapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<std::string, std::unique_ptr<IStats>> snapshot;
    for (const auto& [n, ptr] : stats_map_) {
        snapshot[n] = ptr->clone();
    }
    return snapshot;
}

void UCMStatsMonitor::resetAllStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [n, ptr] : stats_map_) {
        ptr->reset();
    }
}