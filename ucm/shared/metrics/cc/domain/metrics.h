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
#ifndef UNIFIEDCACHE_METRICS_H
#define UNIFIEDCACHE_METRICS_H

#include <atomic>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
namespace UC::Metrics {
using MetricId = uint32_t;
constexpr MetricId INVALID_METRIC_ID = std::numeric_limits<MetricId>::max();
struct HistogramStat {
    std::vector<uint64_t> bucketCounts;
    double sum{0.0};
};
using HistogramStatsMap = std::unordered_map<std::string, HistogramStat>;

struct CachedMetric {
    explicit CachedMetric(std::string metricName) : name{std::move(metricName)} {}

    std::string name;
    std::atomic<MetricId> id{INVALID_METRIC_ID};
    std::atomic<uint64_t> seenEpoch{0};

    CachedMetric(const CachedMetric&) = delete;
    CachedMetric& operator=(const CachedMetric&) = delete;
};

struct MetricBuffer {
    struct InnerBuffer {
        std::unordered_map<MetricId, double> counterStats_;
        std::unordered_map<MetricId, double> gaugeStats_;
        std::unordered_map<MetricId, HistogramStat> histogramStats_;

        void Clear()
        {
            counterStats_.clear();
            gaugeStats_.clear();
            histogramStats_.clear();
        }
    };

    static constexpr int NO_ACTIVE_WRITER = -1;

    InnerBuffer innerBufs_[2];
    std::atomic<int> writeIdx_{0};
    std::atomic<int> activeWriteIdx_{NO_ACTIVE_WRITER};

    class WriteGuard {
    public:
        explicit WriteGuard(MetricBuffer& buffer) : buffer_{buffer}, idx_{buffer_.BeginWrite()} {}

        ~WriteGuard() { buffer_.EndWrite(); }

        WriteGuard(const WriteGuard&) = delete;
        WriteGuard& operator=(const WriteGuard&) = delete;

        int Index() const { return idx_; }

    private:
        MetricBuffer& buffer_;
        int idx_;
    };

    int SwitchBuffer()
    {
        int oldIdx = writeIdx_.exchange(1 - writeIdx_.load(std::memory_order_acquire),
                                        std::memory_order_acq_rel);
        return oldIdx;
    }

    int BeginWrite()
    {
        while (true) {
            int idx = writeIdx_.load(std::memory_order_acquire);
            activeWriteIdx_.store(idx, std::memory_order_release);
            if (writeIdx_.load(std::memory_order_acquire) == idx) { return idx; }
            activeWriteIdx_.store(NO_ACTIVE_WRITER, std::memory_order_release);
        }
    }

    void EndWrite() { activeWriteIdx_.store(NO_ACTIVE_WRITER, std::memory_order_release); }

    void WaitNoActiveWriter(int idx) const
    {
        while (activeWriteIdx_.load(std::memory_order_acquire) == idx) {
            std::this_thread::yield();
        }
    }

    InnerBuffer& GetWriteBuffer(int idx) { return innerBufs_[idx]; }

    const InnerBuffer& GetReadBuffer(int idx) const { return innerBufs_[idx]; }

    void ClearReadBuffer(int idx) { innerBufs_[idx].Clear(); }
};

class Metrics {
public:
    static Metrics& GetInstance()
    {
        static Metrics inst;
        return inst;
    }

    void SetUp(size_t = 0)
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (isInited_.load(std::memory_order_acquire)) { return; }
        bool expected = false;
        isInited_.compare_exchange_strong(expected, true, std::memory_order_release,
                                          std::memory_order_relaxed);
    }

    ~Metrics() = default;

    void CreateStats(const std::string& name, const std::string& type,
                     const std::vector<double>& buckets = {});

    void UpdateStats(const std::string& name, double value);

    void UpdateStats(CachedMetric& metric, double value);

    void UpdateStats(const std::unordered_map<std::string, double>& values);

    std::tuple<std::unordered_map<std::string, double>, std::unordered_map<std::string, double>,
               HistogramStatsMap>
    GetAllStatsAndClear();

private:
    enum class MetricType : int { COUNTER = 0, GAUGE = 1, HISTOGRAM = 2 };
    struct MetricInfo {
        std::string name;
        MetricType type;
        std::vector<double> buckets;
    };

    std::shared_mutex mutex_;
    std::unordered_map<std::string, MetricId> nameToId_;
    std::vector<MetricInfo> metrics_;
    std::list<std::shared_ptr<MetricBuffer>> buffers_;
    static thread_local std::shared_ptr<MetricBuffer> threadBuffer_;
    static thread_local bool isRegisteredThread_;

    Metrics() = default;
    Metrics(const Metrics&) = delete;
    Metrics& operator=(const Metrics&) = delete;
    std::atomic<bool> isInited_{false};
    std::atomic<uint64_t> registerEpoch_{1};

    MetricId ResolveMetricId(const std::string& name) const;
    MetricId ResolveMetricId(const char* name) const;
    MetricId ResolveCachedMetric(CachedMetric& metric) const;
    void RegisterCurrentThread();
    void UpdateStats(MetricId id, double value);
    void UpdateBuffer(MetricBuffer& metricBuffer, int innerBufferIndex, MetricId id, double value);
    const std::string* MetricName(MetricId id) const;
};
}  // namespace UC::Metrics

#endif  // UNIFIEDCACHE_METRICS_H
