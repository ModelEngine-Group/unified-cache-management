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
#include "metrics.h"
#include <algorithm>

namespace UC::Metrics {
thread_local std::shared_ptr<MetricBuffer> Metrics::threadBuffer_ =
    std::make_shared<MetricBuffer>();
thread_local bool Metrics::isRegisteredThread_ = false;

void Metrics::CreateStats(const std::string& name, const std::string& type)
{
    if (!isInited_.load(std::memory_order_acquire)) {
        throw std::runtime_error("Please call SetUp() first!");
    }
    std::string typeUpper = type;
    std::transform(typeUpper.begin(), typeUpper.end(), typeUpper.begin(), ::toupper);
    MetricType metricType;
    if (typeUpper == "COUNTER") {
        metricType = MetricType::COUNTER;
    } else if (typeUpper == "GAUGE") {
        metricType = MetricType::GAUGE;
    } else if (typeUpper == "HISTOGRAM") {
        metricType = MetricType::HISTOGRAM;
    } else {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (nameToId_.count(name)) { return; }
    const auto id = static_cast<MetricId>(metrics_.size());
    nameToId_[name] = id;
    metrics_.push_back(MetricInfo{name, metricType});
    registerEpoch_.fetch_add(1, std::memory_order_release);
}

void Metrics::UpdateStats(const std::string& name, double value)
{
    if (!isInited_.load(std::memory_order_acquire)) { return; }
    UpdateStats(ResolveMetricId(name), value);
}

void Metrics::UpdateStats(CachedMetric& metric, double value)
{
    UpdateStats(ResolveCachedMetric(metric), value);
}

void Metrics::UpdateStats(MetricId id, double value)
{
    if (!isInited_.load(std::memory_order_acquire) || id == INVALID_METRIC_ID ||
        id >= metrics_.size()) {
        return;
    }
    RegisterCurrentThread();
    MetricBuffer::WriteGuard guard{*threadBuffer_};
    UpdateBuffer(*threadBuffer_, guard.Index(), id, value);
}

void Metrics::UpdateStats(const std::unordered_map<std::string, double>& values)
{
    if (!isInited_.load(std::memory_order_acquire) || values.empty()) { return; }

    for (const auto& pair : values) { UpdateStats(ResolveMetricId(pair.first), pair.second); }
}

MetricId Metrics::ResolveMetricId(const std::string& name) const
{
    auto it = nameToId_.find(name);
    return it == nameToId_.end() ? INVALID_METRIC_ID : it->second;
}

MetricId Metrics::ResolveMetricId(const char* name) const
{
    if (name == nullptr) { return INVALID_METRIC_ID; }
    auto it = nameToId_.find(name);
    return it == nameToId_.end() ? INVALID_METRIC_ID : it->second;
}

MetricId Metrics::ResolveCachedMetric(CachedMetric& metric) const
{
    auto id = metric.id.load(std::memory_order_acquire);
    if (id != INVALID_METRIC_ID) { return id; }
    if (!isInited_.load(std::memory_order_acquire)) { return INVALID_METRIC_ID; }
    auto epoch = registerEpoch_.load(std::memory_order_acquire);
    if (metric.seenEpoch.load(std::memory_order_acquire) == epoch) { return INVALID_METRIC_ID; }
    id = ResolveMetricId(metric.name);
    metric.id.store(id, std::memory_order_release);
    metric.seenEpoch.store(epoch, std::memory_order_release);
    return id;
}

void Metrics::RegisterCurrentThread()
{
    if (isRegisteredThread_) { return; }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    buffers_.push_back({threadBuffer_});
    isRegisteredThread_ = true;
}

void Metrics::UpdateBuffer(MetricBuffer& metricBuffer, int innerBufferIndex, MetricId id,
                           double value)
{
    auto& buffer = metricBuffer.GetWriteBuffer(innerBufferIndex);
    switch (metrics_[id].type) {
        case MetricType::COUNTER: buffer.counterStats_[id] += value; break;
        case MetricType::GAUGE: buffer.gaugeStats_[id] = value; break;
        case MetricType::HISTOGRAM:
            if (buffer.histogramStats_[id].size() < maxVectorLen_) {
                buffer.histogramStats_[id].push_back(value);
            } else {
                metricBuffer.DropHistogramStats();
            }
            break;
        default: break;
    }
}

const std::string* Metrics::MetricName(MetricId id) const
{
    if (id == INVALID_METRIC_ID || id >= metrics_.size()) { return nullptr; }
    return &metrics_[id].name;
}

std::tuple<std::unordered_map<std::string, double>, std::unordered_map<std::string, double>,
           std::unordered_map<std::string, std::vector<double>>, uint64_t>
Metrics::GetAllStatsAndClear()
{
    std::unordered_map<std::string, double> totalCounter;
    std::unordered_map<std::string, double> totalGauge;
    std::unordered_map<std::string, std::vector<double>> totalHistogram;
    std::vector<std::shared_ptr<MetricBuffer>> buffers;
    uint64_t droppedHistogramStats = 0;

    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        buffers.assign(buffers_.begin(), buffers_.end());
    }

    for (const auto& buf : buffers) {
        int oldIdx = buf->SwitchBuffer();
        buf->WaitNoActiveWriter(oldIdx);
        droppedHistogramStats += buf->DroppedHistogramStats();
        auto& read_buf = buf->GetReadBuffer(oldIdx);

        for (const auto& [id, value] : read_buf.counterStats_) {
            auto name = MetricName(id);
            if (name) { totalCounter[*name] += value; }
        }

        for (const auto& [id, value] : read_buf.gaugeStats_) {
            auto name = MetricName(id);
            if (name) { totalGauge[*name] = value; }
        }

        for (auto& [id, values] : read_buf.histogramStats_) {
            auto name = MetricName(id);
            if (name) {
                totalHistogram[*name].insert(totalHistogram[*name].end(), values.begin(),
                                             values.end());
            }
        }
        buf->ClearReadBuffer(oldIdx);
    }

    auto result = std::make_tuple(std::move(totalCounter), std::move(totalGauge),
                                  std::move(totalHistogram), droppedHistogramStats);

    return result;
}

}  // namespace UC::Metrics
