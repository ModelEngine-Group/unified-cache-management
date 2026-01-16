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
        std::unordered_map<std::string, double> counter_stats_;
        std::unordered_map<std::string, double> gauge_stats_;
        std::unordered_map<std::string, std::vector<double>> histogram_stats_;

        void Clear()
        {
            counter_stats_.clear();
            gauge_stats_.clear();
            histogram_stats_.clear();
        }

        std::shared_mutex buffer_mutex_;
    };

    InnerBuffer inner_bufs_[2];
    std::atomic<int> write_idx_{0};

    int SwitchBuffer()
    {
        int old_idx = write_idx_.load(std::memory_order_relaxed);
        write_idx_.store(1 - old_idx, std::memory_order_release);
        return old_idx;
    }

    InnerBuffer& GetWriteBuffer(int idx) { return inner_bufs_[idx]; }

    const InnerBuffer& GetReadBuffer(int idx) const { return inner_bufs_[idx]; }

    void ClearReadBuffer(int idx) { inner_bufs_[idx].Clear(); }
};

class Metrics {
public:
    static Metrics& GetInstance()
    {
        if (!is_inited_) { throw std::runtime_error("Please call SetUp() first!"); }
        static Metrics inst;
        return inst;
    }

    static void SetUp(size_t maxVectorLen)
    {
        if (is_inited_.load(std::memory_order_acquire)) { return; }
        bool expected = false;
        if (is_inited_.compare_exchange_strong(expected, true, std::memory_order_release,
                                               std::memory_order_relaxed)) {
            max_vector_len_ = maxVectorLen;
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
    std::unordered_map<std::string, MetricType> stats_type_;
    std::list<std::shared_ptr<MetricBuffer>> buffers_;
    static thread_local std::shared_ptr<MetricBuffer> thread_buffer_;
    static thread_local bool is_registered_thread_;

    Metrics() = default;
    Metrics(const Metrics&) = delete;
    Metrics& operator=(const Metrics&) = delete;
    static std::atomic<bool> is_inited_;
    static size_t max_vector_len_;
};
}  // namespace UC::Metrics

#endif  // UNIFIEDCACHE_METRICS_H