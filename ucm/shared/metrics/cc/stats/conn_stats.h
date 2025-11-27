#ifndef UNIFIEDCACHE_CONNSTATS_H
#define UNIFIEDCACHE_CONNSTATS_H

#include "istats.h"
#include "stats_registry.h"   
#include <array>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>

namespace UC::Metrics {  

enum class Key : uint8_t {
    interval_lookup_hit_rates = 0,
    save_requests_num,
    save_blocks_num,
    save_duration ,
    save_speed,
    load_requests_num,
    load_blocks_num,
    load_duration,
    load_speed,
    COUNT
};

class ConnStats : public IStats {
public:
    ConnStats();
    ~ConnStats() = default;

    std::string Name() const override;
    void Reset() override;
    void Update(const std::unordered_map<std::string, double>& params) override;
    std::unordered_map<std::string, std::vector<double>> Data() override;

private:
    static constexpr std::size_t N = static_cast<std::size_t>(Key::COUNT);
    std::array<std::vector<double>, N> data_;

    static Key KeyFromString(const std::string& k);
    void EmplaceBack(Key id, double value);
};

struct Registrar {
    Registrar() {
        StatsRegistry::RegisterStats("ConnStats", []()->std::unique_ptr<IStats> {
            return std::make_unique<ConnStats>();
        });
    }
};

}  

#endif  // UNIFIEDCACHE_CONNSTATS_H