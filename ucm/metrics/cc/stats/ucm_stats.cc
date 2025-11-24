#include "ucm_stats.h"
#include "../stats_registry.h"   
#include <iostream>

UCMStats::UCMStats() = default;

std::string UCMStats::name() const {
    return "UCMStats";
}

void UCMStats::reset() {
    for (auto& v : data_) v.clear();
}

void UCMStats::update(const std::unordered_map<std::string, double>& params) {
    for (const auto& [k, v] : params) {
        Key id = key_from_string(k);
        if (id == Key::COUNT) continue;
        emplace_back(id, v);
    }
}

std::unordered_map<std::string, std::vector<double>> UCMStats::data() {
    std::unordered_map<std::string, std::vector<double>> result;
    result["save_duration"] = data_[static_cast<std::size_t>(Key::save_duration)];
    result["save_speed"] = data_[static_cast<std::size_t>(Key::save_speed)];
    result["load_duration"] = data_[static_cast<std::size_t>(Key::load_duration)];
    result["load_speed"] = data_[static_cast<std::size_t>(Key::load_speed)];
    result["interval_lookup_hit_rates"] = data_[static_cast<std::size_t>(Key::interval_lookup_hit_rates)];
    return result;
}

Key UCMStats::key_from_string(const std::string& k) {
    if (k == "save_duration")            return Key::save_duration;
    if (k == "save_speed")               return Key::save_speed;
    if (k == "load_duration")            return Key::load_duration;
    if (k == "load_speed")               return Key::load_speed;
    if (k == "interval_lookup_hit_rates")return Key::interval_lookup_hit_rates;
    return Key::COUNT;
}

void UCMStats::emplace_back(Key id, double value) {
    data_[static_cast<std::size_t>(id)].push_back(value);
}

struct Registrar {
    Registrar() {
        StatsRegistry::registerStats("UCMStats", []()->std::unique_ptr<IStats> {
            return std::make_unique<UCMStats>();
        });
    }
} registrar;