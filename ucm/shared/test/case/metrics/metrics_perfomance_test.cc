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
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <thread>
#include <unistd.h>
#include "metrics_api.h"

using namespace UC::Metrics;

class UCMetricsPerfTest : public ::testing::TestWithParam<std::tuple<int, int>> {
public:
    void RunBackgroundTask(int get_interval_ms) { BackgroundGetStatsTask(get_interval_ms); }
    void RunWorkerTask(int thread_id) { UpdateStatsLoop(thread_id); }

    struct Result {
        double avg_time_us = 0.0;
        double avg_get_time_us = 0.0;
        uint64_t total_calls = 0;
        uint64_t total_get_calls = 0;
        double total_time_s = 0.0;
    } result_;

protected:
    const int STATS_NUM = 1024;
    const int CALL_PER_THREAD = 10000;
    std::atomic<bool> is_running_{false};
    std::thread background_get_thread_;
    std::vector<std::thread> worker_threads_;

    std::uniform_int_distribution<int> stat_idx_dist_;
    std::uniform_int_distribution<int> type_dist_;
    std::uniform_real_distribution<double> value_dist_;
    std::mutex stats_mutex_;

    void SetUp() override
    {
        try {
            Metrics::SetUp(CALL_PER_THREAD);
            for (int i = 0; i < STATS_NUM; ++i) {
                CreateStats("stats_counter_" + std::to_string(i), "counter");
                CreateStats("stats_gauge_" + std::to_string(i), "gauge");
                CreateStats("stats_histogram_" + std::to_string(i), "histogram");
            }
        } catch (const std::exception& e) {
            throw;
        }
        stat_idx_dist_ = std::uniform_int_distribution<int>(0, STATS_NUM - 1);
        type_dist_ = std::uniform_int_distribution<int>(0, 2);
        value_dist_ = std::uniform_real_distribution<double>(0.0, 100000.0);
    }

    void TearDown() override
    {
        is_running_.store(false, std::memory_order_relaxed);
        if (background_get_thread_.joinable()) background_get_thread_.join();
        for (auto& t : worker_threads_) {
            if (t.joinable()) t.join();
        }
    }

    void BackgroundGetStatsTask(int get_interval_ms)
    {
        int call_count = 0;
        double total_time_us = 0.0;
        std::chrono::high_resolution_clock::time_point start, end;
        while (is_running_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(get_interval_ms));
            start = std::chrono::high_resolution_clock::now();
            auto stats = GetAllStatsAndClear();
            end = std::chrono::high_resolution_clock::now();
            ++call_count;
            total_time_us += std::chrono::duration<double, std::micro>(end - start).count();
        }
        start = std::chrono::high_resolution_clock::now();
        auto stats = GetAllStatsAndClear();
        end = std::chrono::high_resolution_clock::now();
        total_time_us += std::chrono::duration<double, std::micro>(end - start).count();
        ++call_count;
        std::lock_guard<std::mutex> lock(stats_mutex_);
        result_.total_get_calls += call_count;
        result_.avg_get_time_us += total_time_us / call_count;
    }

    void UpdateStatsLoop(int thread_id)
    {
        std::mt19937 rng_;
        rng_.seed(std::chrono::system_clock::now().time_since_epoch().count() + thread_id);
        int call_count = 0;
        std::chrono::high_resolution_clock::time_point start, end;
        double total_time_us = 0.0;
        while (call_count < CALL_PER_THREAD && is_running_.load(std::memory_order_relaxed)) {
            int stat_idx = stat_idx_dist_(rng_);
            int type = type_dist_(rng_);
            double value = value_dist_(rng_);
            start = std::chrono::high_resolution_clock::now();
            switch (type) {
                case 0: UpdateStats("stats_counter_" + std::to_string(stat_idx), value); break;
                case 1: UpdateStats("stats_gauge_" + std::to_string(stat_idx), value); break;
                case 2: UpdateStats("stats_histogram_" + std::to_string(stat_idx), value); break;
            }
            end = std::chrono::high_resolution_clock::now();
            total_time_us += std::chrono::duration<double, std::micro>(end - start).count();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            ++call_count;
        }
        std::lock_guard<std::mutex> lock(stats_mutex_);
        result_.total_calls += call_count;
        result_.avg_time_us += total_time_us / call_count;
    }
};

TEST_P(UCMetricsPerfTest, Test)
{
    int thread_count = std::get<0>(GetParam());
    int get_interval_ms = std::get<1>(GetParam());
    is_running_.store(true, std::memory_order_relaxed);
    auto test_start = std::chrono::high_resolution_clock::now();

    // Start background get stats thread
    background_get_thread_ =
        std::thread(&UCMetricsPerfTest::RunBackgroundTask, this, get_interval_ms);

    // Start worker threads
    for (int i = 0; i < thread_count; ++i) {
        worker_threads_.emplace_back(&UCMetricsPerfTest::RunWorkerTask, this, i);
    }

    // Wait for all worker threads to finish
    for (auto& t : worker_threads_) {
        if (t.joinable()) t.join();
    }
    auto test_end = std::chrono::high_resolution_clock::now();

    // Stop background thread
    is_running_.store(false, std::memory_order_relaxed);
    if (background_get_thread_.joinable()) background_get_thread_.join();

    result_.avg_time_us /= thread_count;
    result_.total_time_s = std::chrono::duration<double>(test_end - test_start).count();

    // Print result
    std::cout << "\n===== Test Results: thread num=" << thread_count
              << ", get interval=" << get_interval_ms << " =====" << std::endl;
    std::cout << "UpdateStats Total Calls: " << result_.total_calls << std::endl;
    std::cout << "Backend GetStatsAndClear Calls: " << result_.total_get_calls << std::endl;
    std::cout << "Total Running Time: " << result_.total_time_s << " s" << std::endl;
    std::cout << "Avg UpdateStats Time: " << result_.avg_time_us << " us" << std::endl;
    std::cout << "Avg GetAndClear Time: " << result_.avg_get_time_us << " us" << std::endl;
}

INSTANTIATE_TEST_CASE_P(MyPrimeParamTest, UCMetricsPerfTest,
                        ::testing::Combine(::testing::Values(1, 200),
                                           ::testing::Values(10, 50, 100)));