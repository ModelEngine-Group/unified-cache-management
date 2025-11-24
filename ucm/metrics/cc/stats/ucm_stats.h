#ifndef UCM_METRICS_UCMSTATS_H
#define UCM_METRICS_UCMSTATS_H

#include "istats.h"
#include <array>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>

enum class Key : uint8_t {
    save_duration = 0,
    save_speed,
    load_duration,
    load_speed,
    interval_lookup_hit_rates,
    COUNT
};

class UCMStats : public IStats {
public:
    UCMStats();
    ~UCMStats() = default;

    std::string name() const override;
    void reset() override;
    void update(const std::unordered_map<std::string, double>& params) override;
    std::unordered_map<std::string, std::vector<double>> data() override;

private:
    static constexpr std::size_t N = static_cast<std::size_t>(Key::COUNT);
    std::array<std::vector<double>, N> data_;

    static Key key_from_string(const std::string& k);
    void emplace_back(Key id, double value);
};

#endif  // UCM_METRICS_UCMSTATS_H