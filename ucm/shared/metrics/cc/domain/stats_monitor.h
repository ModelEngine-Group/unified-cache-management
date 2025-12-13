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
#ifndef UNIFIEDCACHE_MONITOR_H
#define UNIFIEDCACHE_MONITOR_H

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "stats/stats.h"

namespace UC::Metrics {

class StatsMonitor {
public:
    static StatsMonitor& GetInstance()
    {
        static StatsMonitor inst;
        return inst;
    }

    ~StatsMonitor() = default;

    void CreateStats(const std::string& name);

    void UpdateStats(const std::string& name,
                     const std::unordered_map<std::string, double>& params);

    std::unordered_map<std::string, std::vector<double>> GetStats(const std::string& name);

    std::unordered_map<std::string, std::vector<double>> GetStatsAndClear(const std::string& name);

    std::unordered_map<std::string, std::vector<double>> GetAllStatsAndClear();

    void ResetStats(const std::string& name);

    void ResetAllStats();

private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<Stats>> stats_map_;

    StatsMonitor() = default;
    StatsMonitor(const StatsMonitor&) = delete;
    StatsMonitor& operator=(const StatsMonitor&) = delete;
};
} // namespace UC::Metrics

#endif // UNIFIEDCACHE_MONITOR_H