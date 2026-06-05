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
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "task_manager_base.h"
#include "task_manager_base_slot.h"
#include "transport_task_manager.h"

namespace UC::ASU {
namespace {

using MutexTaskManager = TaskManagerBase<TransportTaskContext, TransportTaskState>;
using SlotTaskManagerImpl = SlotTaskManagerBase<TransportTaskContext, TransportTaskState>;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kFixedTasks = 4096;
constexpr std::size_t kOpsPerThread = 100000;
constexpr std::size_t kHotTasks = 1;

struct BenchConfig {
    std::size_t fixedTasks{kFixedTasks};
    std::size_t opsPerThread{kOpsPerThread};
    std::size_t hotTasks{kHotTasks};
    std::vector<std::size_t> threadCounts{1, 2, 4, 8, 16, 32, 64};
};

struct BenchResult {
    std::string name;
    std::size_t threads{0};
    std::size_t attempts{0};
    std::size_t getOk{0};
    std::size_t getFail{0};
    std::size_t removeOk{0};
    std::size_t removeFail{0};
    std::size_t submitOk{0};
    double runtimeSeconds{0.0};
    double attemptsPerSecond{0.0};
    double submitNsPerOp{0.0};
    double getNsPerOp{0.0};
    double getOkNsPerOp{0.0};
    double getFailNsPerOp{0.0};
    double removeNsPerOp{0.0};
    double removeFailNsPerOp{0.0};
    double removeFailRate{0.0};
};

struct BenchCounters {
    std::atomic<std::size_t> getOk{0};
    std::atomic<std::size_t> getFail{0};
    std::atomic<std::size_t> removeOk{0};
    std::atomic<std::size_t> removeFail{0};
    std::atomic<std::size_t> submitOk{0};
    std::atomic<std::uint64_t> submitNs{0};
    std::atomic<std::uint64_t> getNs{0};
    std::atomic<std::uint64_t> getOkNs{0};
    std::atomic<std::uint64_t> getFailNs{0};
    std::atomic<std::uint64_t> removeNs{0};
    std::atomic<std::uint64_t> removeFailNs{0};
};

template <typename Fn>
auto MeasureNs(Fn&& fn)
{
    const auto begin = Clock::now();
    auto result = fn();
    const auto end = Clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    return std::make_pair(std::move(result), static_cast<std::uint64_t>(ns));
}

double NsPerOp(std::uint64_t totalNs, std::size_t count)
{
    return count == 0 ? 0.0 : static_cast<double>(totalNs) / static_cast<double>(count);
}

BenchResult BuildResult(const std::string& name, std::size_t threads, std::size_t attempts,
                        double seconds, const BenchCounters& counters)
{
    const auto getOk = counters.getOk.load(std::memory_order_relaxed);
    const auto getFail = counters.getFail.load(std::memory_order_relaxed);
    const auto removeOk = counters.removeOk.load(std::memory_order_relaxed);
    const auto removeFail = counters.removeFail.load(std::memory_order_relaxed);
    const auto submitOk = counters.submitOk.load(std::memory_order_relaxed);
    const auto getTotal = getOk + getFail;
    const auto removeTotal = removeOk + removeFail;

    return BenchResult{
        name,
        threads,
        attempts,
        getOk,
        getFail,
        removeOk,
        removeFail,
        submitOk,
        seconds,
        seconds == 0.0 ? 0.0 : static_cast<double>(attempts) / seconds,
        NsPerOp(counters.submitNs.load(std::memory_order_relaxed), submitOk),
        NsPerOp(counters.getNs.load(std::memory_order_relaxed), getTotal),
        NsPerOp(counters.getOkNs.load(std::memory_order_relaxed), getOk),
        NsPerOp(counters.getFailNs.load(std::memory_order_relaxed), getFail),
        NsPerOp(counters.removeNs.load(std::memory_order_relaxed), removeOk),
        NsPerOp(counters.removeFailNs.load(std::memory_order_relaxed), removeFail),
        removeTotal == 0 ? 0.0 : static_cast<double>(removeFail) / static_cast<double>(removeTotal),
    };
}

std::size_t ReadSizeTEnv(const char* name, std::size_t defaultValue)
{
    const auto* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') { return defaultValue; }

    const auto parsed = static_cast<std::size_t>(std::strtoull(value, nullptr, 10));
    return parsed == 0 ? defaultValue : parsed;
}

std::vector<std::size_t> ParseThreadCounts(const std::string& value)
{
    std::vector<std::size_t> counts;
    std::stringstream ss(value);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) { continue; }
        const auto count = static_cast<std::size_t>(std::strtoull(part.c_str(), nullptr, 10));
        if (count > 0) { counts.push_back(count); }
    }
    return counts;
}

BenchConfig ConfigFromEnv()
{
    BenchConfig config;
    config.fixedTasks = ReadSizeTEnv("TASK_MANAGER_BENCH_FIXED_TASKS", config.fixedTasks);
    config.opsPerThread = ReadSizeTEnv("TASK_MANAGER_BENCH_OPS_PER_THREAD", config.opsPerThread);
    config.hotTasks = ReadSizeTEnv("TASK_MANAGER_BENCH_HOT_TASKS", config.hotTasks);

    const auto* threads = std::getenv("TASK_MANAGER_BENCH_THREADS");
    if (threads != nullptr && threads[0] != '\0') {
        auto counts = ParseThreadCounts(threads);
        if (!counts.empty()) { config.threadCounts = std::move(counts); }
    }

    config.hotTasks = std::max<std::size_t>(1, std::min(config.hotTasks, config.fixedTasks));
    return config;
}

std::unique_ptr<TransportTaskContext> NewContext(std::size_t seed)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    ctx->opType = static_cast<TransportOpType>(seed % 7);
    return ctx;
}

template <typename Manager>
std::vector<TaskId> PrefillTasks(Manager& manager, std::size_t taskCount)
{
    std::vector<TaskId> taskIds;
    taskIds.reserve(taskCount);

    for (std::size_t i = 0; i < taskCount; ++i) {
        TaskId taskId{kInvalidTaskId};
        const auto status = manager.Submit(NewContext(i), taskId);
        if (!status.ok()) { break; }
        taskIds.push_back(taskId);
    }

    return taskIds;
}

template <typename Manager>
BenchResult RunFixedCapacityShardBench(const std::string& name, std::size_t threads,
                                       const BenchConfig& config, Manager& manager)
{
    auto taskIds = PrefillTasks(manager, config.fixedTasks);
    EXPECT_EQ(taskIds.size(), config.fixedTasks);

    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    BenchCounters counters;
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (std::size_t tid = 0; tid < threads; ++tid) {
        workers.emplace_back([&, tid]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!start.load(std::memory_order_acquire)) {}

            for (std::size_t op = 0; op < config.opsPerThread; ++op) {
                const auto index = (tid + op * threads) % config.fixedTasks;
                const auto oldTaskId = taskIds[index];

                auto getMeasured = MeasureNs([&]() { return manager.Get(oldTaskId); });
                if (getMeasured.first != nullptr) {
                    counters.getOk.fetch_add(1, std::memory_order_relaxed);
                    counters.getOkNs.fetch_add(getMeasured.second, std::memory_order_relaxed);
                } else {
                    counters.getFail.fetch_add(1, std::memory_order_relaxed);
                    counters.getFailNs.fetch_add(getMeasured.second, std::memory_order_relaxed);
                }
                counters.getNs.fetch_add(getMeasured.second, std::memory_order_relaxed);

                auto removeMeasured = MeasureNs([&]() { return manager.Remove(oldTaskId); });
                auto status = removeMeasured.first;
                if (!status.ok()) {
                    counters.removeFail.fetch_add(1, std::memory_order_relaxed);
                    counters.removeFailNs.fetch_add(removeMeasured.second,
                                                    std::memory_order_relaxed);
                    continue;
                }
                counters.removeOk.fetch_add(1, std::memory_order_relaxed);
                counters.removeNs.fetch_add(removeMeasured.second, std::memory_order_relaxed);

                TaskId newTaskId{kInvalidTaskId};
                auto submitMeasured =
                    MeasureNs([&]() { return manager.Submit(NewContext(tid + op), newTaskId); });
                status = submitMeasured.first;
                if (status.ok()) {
                    taskIds[index] = newTaskId;
                    counters.submitOk.fetch_add(1, std::memory_order_relaxed);
                    counters.submitNs.fetch_add(submitMeasured.second, std::memory_order_relaxed);
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != threads) {}
    const auto begin = Clock::now();
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) { worker.join(); }
    const auto end = Clock::now();

    const auto attempts = threads * config.opsPerThread;
    const auto seconds = std::chrono::duration<double>(end - begin).count();
    return BuildResult(name, threads, attempts, seconds, counters);
}

template <typename Manager>
BenchResult RunHotRemoveContentionBench(const std::string& name, std::size_t threads,
                                        const BenchConfig& config, Manager& manager)
{
    auto initialIds = PrefillTasks(manager, config.fixedTasks);
    EXPECT_EQ(initialIds.size(), config.fixedTasks);

    std::vector<std::atomic<TaskId>> hotIds(config.hotTasks);
    for (std::size_t i = 0; i < config.hotTasks; ++i) {
        hotIds[i].store(initialIds[i], std::memory_order_release);
    }

    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    BenchCounters counters;
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (std::size_t tid = 0; tid < threads; ++tid) {
        workers.emplace_back([&, tid]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!start.load(std::memory_order_acquire)) {}

            for (std::size_t op = 0; op < config.opsPerThread; ++op) {
                const auto index = (tid + op) % config.hotTasks;
                const auto oldTaskId = hotIds[index].load(std::memory_order_acquire);

                auto getMeasured = MeasureNs([&]() { return manager.Get(oldTaskId); });
                if (getMeasured.first != nullptr) {
                    counters.getOk.fetch_add(1, std::memory_order_relaxed);
                    counters.getOkNs.fetch_add(getMeasured.second, std::memory_order_relaxed);
                } else {
                    counters.getFail.fetch_add(1, std::memory_order_relaxed);
                    counters.getFailNs.fetch_add(getMeasured.second, std::memory_order_relaxed);
                }
                counters.getNs.fetch_add(getMeasured.second, std::memory_order_relaxed);

                auto removeMeasured = MeasureNs([&]() { return manager.Remove(oldTaskId); });
                auto status = removeMeasured.first;
                if (!status.ok()) {
                    counters.removeFail.fetch_add(1, std::memory_order_relaxed);
                    counters.removeFailNs.fetch_add(removeMeasured.second,
                                                    std::memory_order_relaxed);
                    continue;
                }
                counters.removeOk.fetch_add(1, std::memory_order_relaxed);
                counters.removeNs.fetch_add(removeMeasured.second, std::memory_order_relaxed);

                TaskId newTaskId{kInvalidTaskId};
                auto submitMeasured =
                    MeasureNs([&]() { return manager.Submit(NewContext(tid + op), newTaskId); });
                status = submitMeasured.first;
                if (status.ok()) {
                    hotIds[index].store(newTaskId, std::memory_order_release);
                    counters.submitOk.fetch_add(1, std::memory_order_relaxed);
                    counters.submitNs.fetch_add(submitMeasured.second, std::memory_order_relaxed);
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != threads) {}
    const auto begin = Clock::now();
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) { worker.join(); }
    const auto end = Clock::now();

    const auto attempts = threads * config.opsPerThread;
    const auto seconds = std::chrono::duration<double>(end - begin).count();
    return BuildResult(name, threads, attempts, seconds, counters);
}

void PrintHeader(const char* scenario, const std::string& implLabel, const BenchConfig& config)
{
    std::cout << "\nTaskManagerBase bench: " << scenario << "\n"
              << "impl=" << implLabel << " fixed_tasks=" << config.fixedTasks
              << " ops_per_thread=" << config.opsPerThread << " hot_tasks=" << config.hotTasks
              << "\n"
              << std::left << std::setw(14) << "impl" << std::right << std::setw(8) << "threads"
              << std::setw(12) << "submit_ns" << std::setw(12) << "get_ns" << std::setw(12)
              << "get_ok_ns" << std::setw(14) << "get_fail_ns" << std::setw(12) << "remove_ns"
              << std::setw(16) << "remove_fail_ns" << std::setw(12) << "remove_fail"
              << std::setw(18) << "remove_fail_rate" << std::setw(12) << "runtime_s" << "\n";
}

void PrintResult(const BenchResult& result)
{
    std::cout << std::left << std::setw(14) << result.name << std::right << std::setw(8)
              << result.threads << std::setw(12) << std::fixed << std::setprecision(1)
              << result.submitNsPerOp << std::setw(12) << result.getNsPerOp << std::setw(12)
              << result.getOkNsPerOp << std::setw(14) << result.getFailNsPerOp << std::setw(12)
              << result.removeNsPerOp << std::setw(16) << result.removeFailNsPerOp << std::setw(12)
              << result.removeFail << std::setw(18) << std::setprecision(4) << result.removeFailRate
              << std::setw(12) << result.runtimeSeconds << "\n";
}

template <typename Manager, typename BenchFn>
void RunTaskManagerBench(const char* scenario, const std::string& implLabel,
                         const BenchConfig& config, BenchFn&& benchFn)
{
    PrintHeader(scenario, implLabel, config);

    for (const auto threads : config.threadCounts) {
        Manager manager(TransportTaskState::PENDING, "task_manager_bench");

        const auto result = benchFn(implLabel, threads, config, manager);

        PrintResult(result);
        EXPECT_EQ(result.removeOk, result.submitOk);
    }
}

TEST(TaskManagerBaseBench, DISABLED_FixedCapacitySharded)
{
    const auto config = ConfigFromEnv();
    RunTaskManagerBench<MutexTaskManager>(
        "fixed capacity, sharded task ownership", "mutex", config,
        [](const std::string& name, std::size_t threads, const BenchConfig& cfg, auto& manager) {
            return RunFixedCapacityShardBench(name, threads, cfg, manager);
        });
}

TEST(TaskManagerBaseBench, DISABLED_HotRemoveContention)
{
    const auto config = ConfigFromEnv();
    RunTaskManagerBench<MutexTaskManager>(
        "hot remove contention", "mutex", config,
        [](const std::string& name, std::size_t threads, const BenchConfig& cfg, auto& manager) {
            return RunHotRemoveContentionBench(name, threads, cfg, manager);
        });
}

TEST(SlotTaskManagerBaseBench, DISABLED_FixedCapacitySharded)
{
    const auto config = ConfigFromEnv();
    RunTaskManagerBench<SlotTaskManagerImpl>(
        "fixed capacity, sharded task ownership", "slot", config,
        [](const std::string& name, std::size_t threads, const BenchConfig& cfg, auto& manager) {
            return RunFixedCapacityShardBench(name, threads, cfg, manager);
        });
}

TEST(SlotTaskManagerBaseBench, DISABLED_HotRemoveContention)
{
    const auto config = ConfigFromEnv();
    RunTaskManagerBench<SlotTaskManagerImpl>(
        "hot remove contention", "slot", config,
        [](const std::string& name, std::size_t threads, const BenchConfig& cfg, auto& manager) {
            return RunHotRemoveContentionBench(name, threads, cfg, manager);
        });
}

}  // namespace
}  // namespace UC::ASU
