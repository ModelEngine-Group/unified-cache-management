/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
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