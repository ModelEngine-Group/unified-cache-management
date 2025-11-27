#include "stats/istats.h"
#include "stats_registry.h"
#include "stats_monitor.h"
#include <mutex>
#include <vector>

namespace UC::Metrics {

StatsMonitor::StatsMonitor() {
    auto& registry = StatsRegistry::GetInstance();
    for (const auto& name : registry.GetRegisteredStatsNames()) {
        stats_map_[name] = registry.CreateStats(name);
    }
}

void StatsMonitor::CreateStats(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& registry = StatsRegistry::GetInstance();
    stats_map_[name] = registry.CreateStats(name);
}

std::unordered_map<std::string, std::vector<double>> StatsMonitor::GetStats(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_map_[name]->Data();
}

void StatsMonitor::ResetStats(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_map_[name]->Reset();
}

std::unordered_map<std::string, std::vector<double>> StatsMonitor::GetStatsAndClear(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = stats_map_[name]->Data();
    stats_map_[name]->Reset();
    return result;
}

void StatsMonitor::UpdateStats(
    const std::string& name,
    const std::unordered_map<std::string, double>& params)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_map_.find(name);
    if (it != stats_map_.end()) {
        it->second->Update(params);
    }
}

void StatsMonitor::ResetAllStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [n, ptr] : stats_map_) {
        ptr->Reset();
    }
}

}  