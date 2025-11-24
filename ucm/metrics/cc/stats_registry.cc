#include "stats_registry.h"

std::mutex StatsRegistry::mutex_;
std::unordered_map<std::string, Creator> StatsRegistry::registry_;

StatsRegistry& StatsRegistry::getInstance() {
    static StatsRegistry inst;
    return inst;
}

void StatsRegistry::registerStats(std::string name, Creator creator) {
    std::lock_guard lk(mutex_);
    registry_[name] = creator;
}

std::unique_ptr<IStats> StatsRegistry::createStats(const std::string& name) {
    std::lock_guard lk(mutex_);
    if (auto it = registry_.find(name); it != registry_.end())
        return it->second();
    return nullptr;
}

std::vector<std::string> StatsRegistry::getRegisteredStatsNames() {
    std::lock_guard lk(mutex_);
    std::vector<std::string> names;
    names.reserve(registry_.size());
    for (auto& [n, _] : registry_) names.push_back(n);
    return names;
}