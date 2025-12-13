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
#include "stats_monitor_api.h"
namespace UC::Metrics {

void CreateStats(const std::string& name) { StatsMonitor::GetInstance().CreateStats(name); }

void UpdateStats(const std::string& name, const std::unordered_map<std::string, double>& params)
{
    StatsMonitor::GetInstance().UpdateStats(name, params);
}

void ResetStats(const std::string& name) { StatsMonitor::GetInstance().ResetStats(name); }

void ResetAllStats() { StatsMonitor::GetInstance().ResetAllStats(); }

StatsResult GetStats(const std::string& name)
{
    StatsResult result;
    result.data = StatsMonitor::GetInstance().GetStats(name);
    return result;
}

StatsResult GetStatsAndClear(const std::string& name)
{
    StatsResult result;
    result.data = StatsMonitor::GetInstance().GetStatsAndClear(name);
    return result;
}

StatsResult GetAllStatsAndClear()
{
    StatsResult result;
    result.data = StatsMonitor::GetInstance().GetAllStatsAndClear();
    return result;
}

} // namespace UC::Metrics