#ifndef UCM_STATS_H
#define UCM_STATS_H

#include "istats.h"
#include <array>
#include <vector>
#include <unordered_map>
#include <string>

/* 1. 编译期 key → id */
enum class Key : uint8_t {
    save_duration = 0,
    save_speed,
    load_duration,
    load_speed,
    interval_lookup_hit_rates,
    COUNT  // 总个数
};

class UCMStats : public IStats {
    static constexpr std::size_t N = static_cast<std::size_t>(Key::COUNT);
    /* 2. 一个数组搞定全部 vector */
    std::array<std::vector<double>, N> data_;

    /* 3. 字符串 → id（只做一次） */
    static Key key_from_string(const std::string& k) {
        // 仅 5 项，switch 比 unordered_map 更快
        if (k == "save_duration")            return Key::save_duration;
        if (k == "save_speed")               return Key::save_speed;
        if (k == "load_duration")            return Key::load_duration;
        if (k == "load_speed")               return Key::load_speed;
        if (k == "interval_lookup_hit_rates")return Key::interval_lookup_hit_rates;
        return Key::COUNT;  // 非法 key
    }

public:
    UCMStats() = default;

    std::string name() const override {
        return "UCMStats";
    }

    void reset() override {
        for (auto& v : data_) v.clear();
    }

    std::unique_ptr<IStats> clone() const override {
        return std::make_unique<UCMStats>(*this);
    }

    void update(const std::unordered_map<std::string, double>& params) override {
        for (const auto& [k, v] : params) {
            Key id = key_from_string(k);
            if (id == Key::COUNT) continue;          // 未知 key 直接跳过
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
    /* 5. 运行期 O(1) 下标访问 */
    void emplace_back(Key id, double value) {
        data_[static_cast<std::size_t>(id)].push_back(value);
    }
};

#endif // UCM_STATS_H

