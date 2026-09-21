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
#include "metrics_reporter.h"
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string_view>
#include <utility>
#include "drampool_config.h"
#include "logger/logger.h"

namespace UC::DramPool {
namespace {

constexpr char kMetricsFileName[] = "drampool_metrics.json";
// Single-line JSON snapshot event name, matching the DramPool resource
constexpr char kSnapshotEvent[] = "drampool_metrics_snapshot";
// Wake-up granularity while waiting for the next interval, so Stop() does not
// block for a full interval (same pattern as HealthServer's poll interval).
constexpr auto kStopPollInterval = std::chrono::milliseconds(100);

double FindValue(const std::unordered_map<std::string, double>& values, const std::string& name)
{
    const auto it = values.find(name);
    return it == values.end() ? 0.0 : it->second;
}

// The snapshot parser rejects non-finite numbers, so clamp them to zero to
// keep one bad observation from invalidating the whole record.
void AppendNumber(std::ostringstream& output, const std::string& name, double value)
{
    if (!std::isfinite(value)) { value = 0.0; }
    output << '"' << name << "\":" << value;
}

}  // namespace

MetricsReporter::~MetricsReporter() { Stop(); }

Status MetricsReporter::Start()
{
    if (!g_config.metricsEnabled) {
        UC_INFO_UNLIMITED("DramPool metrics reporter is disabled");
        return Status::OK();
    }
    interval_ = std::chrono::milliseconds(g_config.metricsIntervalMs);
    // Same as the Python side (os.makedirs(..., exist_ok=True) on
    // multiproc_dir); an empty output dir falls back to the logger directory.
    const auto outputDir = g_config.metricsOutputDir.empty()
                               ? std::filesystem::path{g_config.logDir}
                               : std::filesystem::path{g_config.metricsOutputDir};
    std::error_code fsError;
    std::filesystem::create_directories(outputDir, fsError);
    if (fsError) {
        return Status::OsApiError("failed to create DramPool metrics output dir " +
                                  outputDir.string() + ": " + fsError.message());
    }
    outputPath_ = (outputDir / kMetricsFileName).string();
    tmpPath_ = outputPath_ + ".tmp";
    stopping_.store(false, std::memory_order_release);
    try {
        worker_ = std::thread(&MetricsReporter::Run, this);
    } catch (const std::exception& error) {
        stopping_.store(true, std::memory_order_release);
        return Status::Error(std::string{"failed to start MetricsReporter: "} + error.what());
    }
    UC_INFO_UNLIMITED("DramPool metrics reporter started, output={}, interval={}ms", outputPath_,
                      g_config.metricsIntervalMs);
    return Status::OK();
}

void MetricsReporter::Stop() noexcept
{
    if (!worker_.joinable()) { return; }

    stopping_.store(true, std::memory_order_release);
    worker_.join();
}

void MetricsReporter::Run() noexcept
{
    while (!stopping_.load(std::memory_order_acquire)) {
        CollectAndWrite();
        const auto deadline = std::chrono::steady_clock::now() + interval_;
        while (!stopping_.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() >= deadline) { break; }
            std::this_thread::sleep_for(kStopPollInterval);
        }
    }
}

void MetricsReporter::CollectAndWrite() noexcept
{
    try {
        // Same drain-then-dispatch flow as PrometheusStatsLogger.update_stats.
        auto [counterStats, gaugeStats, histogramStats] = UC::Metrics::GetAllStatsAndClear();
        // _update_counter drops negative deltas.
        for (const auto& [name, value] : counterStats) {
            if (value < 0) { continue; }
            counterValues_[name] += value;
        }
        // _update_gauge keeps the latest value.
        for (const auto& [name, value] : gaugeStats) { gaugeValues_[name] = value; }
        // _update_histogram merges per-bucket counts and the sum.
        for (const auto& [name, stat] : histogramStats) {
            auto& state = histogramValues_[name];
            if (state.bucketCounts.size() < stat.bucketCounts.size()) {
                state.bucketCounts.resize(stat.bucketCounts.size());
            }
            for (std::size_t index = 0; index < stat.bucketCounts.size(); ++index) {
                state.bucketCounts[index] += stat.bucketCounts[index];
            }
            state.sum += stat.sum;
        }
        WriteFile(Render());
    } catch (const std::exception& error) {
        UC_ERROR_UNLIMITED("MetricsReporter collect failed: {}", error.what());
    } catch (...) {
        UC_ERROR_UNLIMITED("MetricsReporter collect failed with unknown error");
    }
}

std::string MetricsReporter::Render() const
{
    // One single-line JSON record per the DramPool resource snapshot contract
    // (#1396): the full cumulative state (the reader deltas consecutive
    // snapshots, so counter/histogram resets are detected on its side)
    // grouped into counters/gauges/histograms. Rendered in registration order
    // (DrampoolMetricDefs() plus the dynamic per-slot-size gauges) so the
    // layout is stable across snapshots. The record must end with a newline:
    // the reader treats a trailing line without one as incomplete and drops
    // it.
    std::ostringstream output;
    output << std::setprecision(17);
    output << "{\"event\":\"" << kSnapshotEvent << "\",\"timestamp\":"
           << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
                  .count();
    output << ",\"counters\":{";
    bool first = true;
    for (const auto& def : DrampoolMetricDefs()) {
        if (std::string_view{def.type} != "counter") { continue; }
        if (!first) { output << ','; }
        first = false;
        AppendNumber(output, def.name, FindValue(counterValues_, def.name));
    }
    output << "},\"gauges\":{";
    first = true;
    for (const auto& def : DrampoolMetricDefs()) {
        if (std::string_view{def.type} != "gauge") { continue; }
        if (!first) { output << ','; }
        first = false;
        AppendNumber(output, def.name, FindValue(gaugeValues_, def.name));
    }
    for (const auto slotSize : g_config.poolBlockSizes) {
        if (!first) { output << ','; }
        first = false;
        AppendNumber(output, BufferPoolUsageRatioName(slotSize),
                     FindValue(gaugeValues_, BufferPoolUsageRatioName(slotSize)));
    }
    output << "},\"histograms\":{";
    first = true;
    for (const auto& def : DrampoolMetricDefs()) {
        if (std::string_view{def.type} != "histogram") { continue; }
        if (!first) { output << ','; }
        first = false;
        output << '"' << def.name << "\":";
        RenderHistogram(output, def);
    }
    output << "}}\n";
    return output.str();
}

void MetricsReporter::RenderHistogram(std::ostringstream& output,
                                      const DrampoolMetricDef& def) const
{
    // Per the snapshot contract: upper_bounds holds the finite bucket bounds,
    // bucket_counts holds the raw (non-cumulative) per-bucket counts plus the
    // trailing +Inf bucket, so bucket_counts.size() == upper_bounds.size() +
    // 1, count == sum(bucket_counts), and the parser requires count == 0 to
    // imply sum == 0 and rejects non-finite or negative sums.
    const auto it = histogramValues_.find(def.name);
    std::uint64_t count = 0;
    double sum = 0.0;
    if (it != histogramValues_.end()) {
        for (const auto bucketCount : it->second.bucketCounts) { count += bucketCount; }
        sum = it->second.sum;
    }
    if (count == 0 || !std::isfinite(sum) || sum < 0.0) { sum = 0.0; }
    output << "{\"upper_bounds\":[";
    for (std::size_t index = 0; index < def.bucketCount; ++index) {
        if (index != 0) { output << ','; }
        output << def.buckets[index];
    }
    output << "],\"bucket_counts\":[";
    for (std::size_t index = 0; index <= def.bucketCount; ++index) {
        if (index != 0) { output << ','; }
        output << (it != histogramValues_.end() && index < it->second.bucketCounts.size()
                       ? it->second.bucketCounts[index]
                       : 0U);
    }
    output << "],\"count\":" << count << ",\"sum\":" << sum << "}";
}

void MetricsReporter::WriteFile(const std::string& content) const
{
    // Atomic replacement: write a tmp file, then rename over the target, so a
    // scrape never observes a half-written file.
    {
        std::ofstream output{tmpPath_, std::ios::trunc};
        if (!output.is_open()) {
            UC_ERROR_UNLIMITED("MetricsReporter failed to open {}, errno={}", tmpPath_, errno);
            return;
        }
        output << content;
        output.flush();
        if (!output.good()) {
            UC_ERROR_UNLIMITED("MetricsReporter failed to write {}", tmpPath_);
            return;
        }
    }
    if (std::rename(tmpPath_.c_str(), outputPath_.c_str()) != 0) {
        UC_ERROR_UNLIMITED("MetricsReporter failed to replace {}, errno={}", outputPath_, errno);
    }
}

}  // namespace UC::DramPool
