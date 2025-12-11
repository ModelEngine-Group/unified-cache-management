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
#ifndef UNIFIEDCACHE_MONITOR_API_H
#define UNIFIEDCACHE_MONITOR_API_H
#include <string>
#include <unordered_map>
#include "stats_monitor.h"

namespace UC::Metrics {
struct StatsResult {
    StatsResult() = default;
    std::unordered_map<std::string, std::vector<double>> data;
};

void RegistStats(std::string name, Creator creator);
void CreateStats(const std::string& name);
void UpdateStats(const std::string& name, const std::unordered_map<std::string, double>& params);
void ResetStats(const std::string& name);
void ResetAllStats();
StatsResult GetStats(const std::string& name);
StatsResult GetStatsAndClear(const std::string& name);
StatsResult GetAllStatsAndClear();

} // namespace UC::Metrics
#endif