#pragma once
#include "istats.h"
#include <array>
#include <vector>
#include <unordered_map>
#include <string>

/* key → id */
enum class Key : uint8_t {
    save_duration = 0,
    save_speed,
    load_duration,
    load_speed,
    interval_lookup_hit_rates,
    COUNT  // Total keys num
};

class UCMStats : public IStats {
    static constexpr std::size_t N = static_cast<std::size_t>(Key::COUNT);
    std::array<std::vector<double>, N> data_;

    static Key key_from_string(const std::string& k) {
        if (k == "save_duration")            return Key::save_duration;
        if (k == "save_speed")               return Key::save_speed;
        if (k == "load_duration")            return Key::load_duration;
        if (k == "load_speed")               return Key::load_speed;
        if (k == "interval_lookup_hit_rates")return Key::interval_lookup_hit_rates;
        return Key::COUNT;  // Invalid key
    }

public:
    UCMStats() = default;

    std::string name() const override {
        return "UCMStats";
    }

    void reset() override {
        for (auto& v : data_) v.clear();
    }

    void update(const std::unordered_map<std::string, double>& params) override {
        for (const auto& [k, v] : params) {
            Key id = key_from_string(k);
            if (id == Key::COUNT) continue;
            emplace_back(id, v);
        }
    }

    std::unordered_map<std::string, std::vector<double>> data() override {
        std::unordered_map<std::string, std::vector<double>> result;
        result["save_duration"] = data_[static_cast<std::size_t>(Key::save_duration)];
        result["save_speed"] = data_[static_cast<std::size_t>(Key::save_speed)];
        result["load_duration"] = data_[static_cast<std::size_t>(Key::load_duration)];
        result["load_speed"] = data_[static_cast<std::size_t>(Key::load_speed)];
        result["interval_lookup_hit_rates"] = data_[static_cast<std::size_t>(Key::interval_lookup_hit_rates)];
        return result;
    }

private:
    void emplace_back(Key id, double value) {
        data_[static_cast<std::size_t>(id)].push_back(value);
    }
};

