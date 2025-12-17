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
#include <tuple>

namespace UC::Metrics {

class Metrics {
public:
    static Metrics& GetInstance()
    {
        static Metrics inst;
        return inst;
    }

    ~Metrics() = default;

    void CreateStats(const std::string& name, std::string& type);

    void UpdateStats(const std::string& name, double value);

    std::tuple<
        std::unordered_map<std::string, double>,
        std::unordered_map<std::string, double>,
        std::unordered_map<std::string, std::vector<double>>
    > GetAllStatsAndClear();

private:
    enum class MetricType { COUNTER, GAUGE, HISTOGRAM };

    std::mutex mutex_;
    std::unordered_map<std::string, double> counter_stats_;
    std::unordered_map<std::string, double> gauge_stats_;
    std::unordered_map<std::string, std::vector<double>> histogram_stats_;
    std::unordered_map<std::string, MetricType> stats_type_;

    Metrics() = default;
    Metrics(const Metrics&) = delete;
    Metrics& operator=(const Metrics&) = delete;
};
} // namespace UC::Metrics

#endif // UNIFIEDCACHE_MONITOR_H