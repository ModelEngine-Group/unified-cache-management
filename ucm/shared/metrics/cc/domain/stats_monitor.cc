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
#include "stats_monitor.h"
#include <mutex>
#include <vector>

namespace UC::Metrics {

StatsMonitor::StatsMonitor()
{
    auto& registry = StatsRegistry::GetInstance();
    for (const auto& name : registry.GetRegisteredStatsNames()) {
        stats_map_[name] = registry.CreateStats(name);
    }
}

void StatsMonitor::CreateStats(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& registry = StatsRegistry::GetInstance();
    stats_map_[name] = registry.CreateStats(name);
}

std::unordered_map<std::string, std::vector<double>> StatsMonitor::GetStats(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_map_.find(name);
    if (it == stats_map_.end() || !it->second) { return {}; }
    return it->second->Data();
}

void StatsMonitor::ResetStats(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_map_.find(name);
    if (it == stats_map_.end() || !it->second) { return; }
    it->second->Reset();
}

std::unordered_map<std::string, std::vector<double>>
StatsMonitor::GetStatsAndClear(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_map_.find(name);
    if (it == stats_map_.end() || !it->second) { return {}; }
    auto result = it->second->Data();
    it->second->Reset();
    return result;
}

std::unordered_map<std::string, std::vector<double>> StatsMonitor::GetAllStatsAndClear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<std::string, std::vector<double>> all_stats;

    for (const auto& [name, stats_ptr] : stats_map_) {
        if (stats_ptr) {
            auto data = stats_ptr->Data();
            all_stats.insert(data.begin(), data.end());
            stats_ptr->Reset();
        }
    }
    return all_stats;
}

void StatsMonitor::UpdateStats(const std::string& name,
                               const std::unordered_map<std::string, double>& params)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_map_.find(name);
    if (it == stats_map_.end() || !it->second) { return; }
    try {
        it->second->Update(params);
    } catch (...) {
        return;
    }
}

void StatsMonitor::ResetAllStats()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [name, stats_ptr] : stats_map_) {
        if (stats_ptr) { stats_ptr->Reset(); }
    }
}

} // namespace UC::Metrics