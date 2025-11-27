#include "conn_stats.h"

namespace UC::Metrics {
    
ConnStats::ConnStats() = default;

std::string ConnStats::Name() const {
    return "ConnStats";
}

void ConnStats::Reset() {
    for (auto& v : data_) v.clear();
}

void ConnStats::Update(const std::unordered_map<std::string, double>& params) {
    for (const auto& [k, v] : params) {
        Key id = KeyFromString(k);
        if (id == Key::COUNT) continue;
        EmplaceBack(id, v);
    }
}

std::unordered_map<std::string, std::vector<double>> ConnStats::Data() {
    std::unordered_map<std::string, std::vector<double>> result;
    result["save_requests_num"] = data_[static_cast<std::size_t>(Key::save_requests_num)];
    result["save_blocks_num"] = data_[static_cast<std::size_t>(Key::save_blocks_num)];
    result["save_duration"] = data_[static_cast<std::size_t>(Key::save_duration)];
    result["save_speed"] = data_[static_cast<std::size_t>(Key::save_speed)];
    result["load_requests_num"] = data_[static_cast<std::size_t>(Key::load_requests_num)];
    result["load_blocks_num"] = data_[static_cast<std::size_t>(Key::load_blocks_num)];
    result["load_duration"] = data_[static_cast<std::size_t>(Key::load_duration)];
    result["load_speed"] = data_[static_cast<std::size_t>(Key::load_speed)];
    result["interval_lookup_hit_rates"] = data_[static_cast<std::size_t>(Key::interval_lookup_hit_rates)];
    return result;
}

Key ConnStats::KeyFromString(const std::string& k) {
    if (k == "save_requests_num")        return Key::save_requests_num;
    if (k == "save_blocks_num")          return Key::save_blocks_num;
    if (k == "save_duration")            return Key::save_duration;
    if (k == "save_speed")               return Key::save_speed;
    if (k == "load_requests_num")        return Key::load_requests_num;
    if (k == "load_blocks_num")          return Key::load_blocks_num;
    if (k == "load_duration")            return Key::load_duration;
    if (k == "load_speed")               return Key::load_speed;
    if (k == "interval_lookup_hit_rates")return Key::interval_lookup_hit_rates;
    return Key::COUNT;
}

void ConnStats::EmplaceBack(Key id, double value) {
    data_[static_cast<std::size_t>(id)].push_back(value);
}

static Registrar registrar;

}  