#include "stats/istats.h"
#include "stats_registry.h"
#include "stats_monitor.h"
#include <mutex>
#include <vector>

UCMStatsMonitor::UCMStatsMonitor() {
    auto& registry = StatsRegistry::getInstance();
    for (const auto& name : registry.getRegisteredStatsNames()) {
        stats_map_[name] = registry.createStats(name);
    }
}

std::unordered_map<std::string, std::vector<double>> UCMStatsMonitor::getStats(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_map_[name]->data();
}

void UCMStatsMonitor::resetStats(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_map_[name]->reset();
}

std::unordered_map<std::string, std::vector<double>> UCMStatsMonitor::getStatsAndClear(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = stats_map_[name]->data();
    stats_map_[name]->reset();
    return result;
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

void UCMStatsMonitor::resetAllStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [n, ptr] : stats_map_) {
        ptr->reset();
    }
}