#include "stats_registry.h"

namespace UC::Metrics {

StatsRegistry& StatsRegistry::GetInstance() {
    static StatsRegistry inst;          
    return inst;
}

void StatsRegistry::RegisterStats(std::string name, Creator creator) {
    auto& reg = GetInstance();         
    std::lock_guard lk(reg.mutex_);
    reg.registry_[name] = creator;
}

std::unique_ptr<IStats> StatsRegistry::CreateStats(const std::string& name) {
    auto& reg = GetInstance();
    std::lock_guard lk(reg.mutex_);
    if (auto it = reg.registry_.find(name); it != reg.registry_.end())
        return it->second();
    return nullptr;
}

std::vector<std::string> StatsRegistry::GetRegisteredStatsNames() {
    auto& reg = GetInstance();
    std::lock_guard lk(reg.mutex_);
    std::vector<std::string> names;
    names.reserve(reg.registry_.size());
    for (auto& [n, _] : reg.registry_) names.push_back(n);
    return names;
}

}   // namespace UC::Metrics