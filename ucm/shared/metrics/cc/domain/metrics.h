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
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
namespace UC::Metrics {
struct MetricBuffer {
    struct InnerBuffer {
        std::unordered_map<std::string, double> counterStats_;
        std::unordered_map<std::string, double> gaugeStats_;
        std::unordered_map<std::string, std::vector<double>> histogramStats_;

        void Clear()
        {
            counterStats_.clear();
            gaugeStats_.clear();
            histogramStats_.clear();
        }

        std::shared_mutex bufferMutex_;
    };

    InnerBuffer innerBufs_[2];
    std::atomic<int> writeIdx_{0};

    int SwitchBuffer()
    {
        int oldIdx = writeIdx_.exchange(1 - writeIdx_.load(std::memory_order_acquire),
                                        std::memory_order_acq_rel);
        return oldIdx;
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

    void SetUp(size_t maxVectorLen)
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (isInited_.load(std::memory_order_acquire)) { return; }
        bool expected = false;
        if (isInited_.compare_exchange_strong(expected, true, std::memory_order_release,
                                              std::memory_order_relaxed)) {
            maxVectorLen_ = maxVectorLen;
        }
    }

    ~Metrics() = default;

    void CreateStats(const std::string& name, const std::string& type);

    void UpdateStats(const std::string& name, double value);

    void UpdateStats(const std::unordered_map<std::string, double>& values);

    std::tuple<std::unordered_map<std::string, double>, std::unordered_map<std::string, double>,
               std::unordered_map<std::string, std::vector<double>>>
    GetAllStatsAndClear();

private:
    enum class MetricType : int { COUNTER = 0, GAUGE = 1, HISTOGRAM = 2 };

    std::shared_mutex mutex_;
    std::unordered_map<std::string, MetricType> statsType_;
    std::list<std::shared_ptr<MetricBuffer>> buffers_;
    static thread_local std::shared_ptr<MetricBuffer> threadBuffer_;
    static thread_local bool isRegisteredThread_;

    Metrics() = default;
    Metrics(const Metrics&) = delete;
    Metrics& operator=(const Metrics&) = delete;
    std::atomic<bool> isInited_{false};
    size_t maxVectorLen_{10000};
};
}  // namespace UC::Metrics

#endif  // UNIFIEDCACHE_METRICS_H