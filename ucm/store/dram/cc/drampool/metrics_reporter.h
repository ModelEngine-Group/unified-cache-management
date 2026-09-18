/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "drampool_metrics.h"
#include "status/status.h"

namespace UC::DramPool {

// Periodic Prometheus textfile exporter for the DramPool metrics. The daemon
// is a pure C++ process without the Python binding, so this component mirrors
// the Python-side multiproc consumer (PrometheusStatsLogger in
// ucm/observability.py): a background thread drains the UC::Metrics delta via
// GetAllStatsAndClear() every interval, accumulates counter deltas, keeps the
// latest gauge values, and merges histogram per-bucket counts and sums, then
// renders the Prometheus text exposition format and atomically replaces
// <output_dir>/drampool_metrics.prom (tmp file + rename). The exposed metric
// set is DrampoolMetricDefs() plus the dynamic per-slot-size gauges, i.e.
// exactly the names registered by SetupDrampoolMetrics().
class MetricsReporter final {
public:
    MetricsReporter() = default;
    ~MetricsReporter();

    MetricsReporter(const MetricsReporter&) = delete;
    MetricsReporter& operator=(const MetricsReporter&) = delete;

    Status Start();
    void Stop() noexcept;

private:
    void Run() noexcept;
    void CollectAndWrite() noexcept;
    std::string Render() const;
    void RenderHistogram(std::ostringstream& output, const DrampoolMetricDef& def) const;
    void WriteFile(const std::string& content) const;

    std::atomic_bool stopping_{true};
    std::chrono::milliseconds interval_{0};
    std::string outputPath_;
    std::string tmpPath_;

    // Accumulated exposition state; only the worker thread touches it.
    // Counter deltas are added, gauge values overwritten, and histogram
    // per-bucket counts and sums merged, matching PrometheusStatsLogger.
    std::unordered_map<std::string, double> counterValues_;
    std::unordered_map<std::string, double> gaugeValues_;
    struct HistogramState {
        std::vector<std::uint64_t> bucketCounts;
        double sum{0.0};
    };
    std::unordered_map<std::string, HistogramState> histogramValues_;
    std::thread worker_;
};

}  // namespace UC::DramPool
