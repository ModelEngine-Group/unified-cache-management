/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "worker/scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include "log/logger_manager.h"

namespace {

/** 占位请求数据规模：entry 数与单条 value 字节数（requestManager 落地后由 input.txt 请求决定） */
constexpr std::uint32_t kEntryCount = 2;
constexpr std::size_t kValueSize = 1024;

/** 指标汇总周期（毫秒），秒级；后续随压测参数配置化 */
constexpr int kMetricsIntervalMs = 1000;
/** 指标线程睡眠分片（毫秒），保证停止请求在分片时间内响应 */
constexpr int kMetricsSleepSliceMs = 50;

/** 生成测试 key：毫秒时间戳低 32 位 + entry 序号，保证每次运行都是全新 key（查询断言不受历史数据干扰） */
kv::CacheKey MakeTestKey(std::uint32_t stamp, std::uint32_t index)
{
    kv::CacheKey key{};
    std::memcpy(key.data(), &stamp, sizeof(stamp));
    std::memcpy(key.data() + sizeof(stamp), &index, sizeof(index));
    return key;
}

/** 字节数转人类可读单位（B/KB/MB/GB，保留 1 位小数） */
std::string FormatBytes(double bytes)
{
    static const char *kUnits[] = {"B", "KB", "MB", "GB"};
    double value = bytes;
    int unit = 0;
    while (value >= 1024.0 && unit < 3) {
        value /= 1024.0;
        ++unit;
    }
    return fmt::format("{:.1f}{}", value, kUnits[unit]);
}

/** 纳秒时延转人类可读单位（自动选 ns/us/ms/s，保留 3 位小数） */
std::string FormatLatency(double ns)
{
    if (ns < 1000.0) {
        return fmt::format("{:.3f}ns", ns);
    }
    if (ns < 1000000.0) {
        return fmt::format("{:.3f}us", ns / 1000.0);
    }
    if (ns < 1000000000.0) {
        return fmt::format("{:.3f}ms", ns / 1000000.0);
    }
    return fmt::format("{:.3f}s", ns / 1000000000.0);
}

} // namespace

Scheduler::Scheduler() = default;

Scheduler::~Scheduler()
{
    // 兜底清理：未显式 Shutdown 时回收指标线程并释放资源
    Shutdown();
}

bool Scheduler::Init(const std::string &confPath, std::string &err)
{
    // kv client 初始化与建链
    client_ = std::make_unique<Client>();
    if (!client_->Init(confPath, err)) {
        client_.reset();
        return false;
    }

    // 组装请求数据并完成内存注册（值 buffer 注册后由 kv client 统一管理）
    BuildPlaceholderRequests();
    if (!request_.Register(*client_, err)) {
        LOG_ERROR("Scheduler：请求内存注册失败: {}", err);
        return false;
    }
    LOG_INFO("Scheduler：{} 块请求值 buffer 内存注册成功", request_.handles.size());

    // 创建执行器并绑定 client（当前单个；多线程并发时按 numjobs 创建多个）
    executors_.clear();
    executors_.push_back(std::make_unique<Executor>());
    executors_.back()->SetClient(client_.get());
    return true;
}

bool Scheduler::Run(std::string &err)
{
    if (client_ == nullptr) {
        err = "Scheduler 未初始化";
        return false;
    }
    if (executors_.empty()) {
        err = "Scheduler 无可用执行器";
        return false;
    }

    // 启动指标汇总线程（秒级周期输出窗口指标）
    metricsRunning_.store(true);
    metricsThread_ = std::thread(&Scheduler::MetricsLoop, this);

    // 逐个驱动执行器执行验证（当前单个执行器同步执行；多线程并发为后续扩展）
    bool allPassed = true;
    for (auto &executor : executors_) {
        if (!executor->Run(request_, err)) {
            allPassed = false;
            break;
        }
    }

    // 停止指标线程并输出运行总计
    metricsRunning_.store(false);
    if (metricsThread_.joinable()) {
        metricsThread_.join();
    }
    OutputMetricsTotal();
    return allPassed;
}

void Scheduler::Shutdown()
{
    // 停止并回收指标线程
    metricsRunning_.store(false);
    if (metricsThread_.joinable()) {
        metricsThread_.join();
    }

    // 注销请求内存（全部任务均已等待完成）后关闭 kv client
    if (client_ != nullptr) {
        std::string err;
        if (!request_.Unregister(*client_, err)) {
            LOG_WARN("Scheduler：请求内存注销失败: {}", err);
        }
        client_->Shutdown();
        client_.reset();
    }

    // 释放执行器（其持有的 client 裸指针随 client 一并失效）
    executors_.clear();
}

void Scheduler::MetricsLoop()
{
    // 基线取启动时刻累计值，首个窗口覆盖启动以来的全部增量
    std::vector<ExecutorMetrics> last = SnapshotExecutors();
    while (metricsRunning_.load(std::memory_order_relaxed)) {
        // 分片睡眠到下一个汇总点，停止请求在分片时间内响应
        for (int slept = 0; slept < kMetricsIntervalMs; slept += kMetricsSleepSliceMs) {
            if (!metricsRunning_.load(std::memory_order_relaxed)) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kMetricsSleepSliceMs));
        }
        const std::vector<ExecutorMetrics> current = SnapshotExecutors();
        OutputMetricsWindow(last, current);
        last = current;
    }
}

void Scheduler::BuildPlaceholderRequests()
{
    // key 用毫秒时间戳 + 序号派生，保证每次运行都是全新 key
    const auto stamp = static_cast<std::uint32_t>(
        std::chrono::system_clock::now().time_since_epoch().count() / 1000000);

    // 每条 value 为 entry 序号相关的固定 pattern，回读比对可校验数据一致性
    for (std::uint32_t i = 0; i < kEntryCount; ++i) {
        std::vector<std::uint8_t> value(kValueSize);
        for (std::size_t j = 0; j < value.size(); ++j) {
            value[j] = static_cast<std::uint8_t>(i * 131 + j);
        }
        request_.Add(MakeTestKey(stamp, i), value);
    }
}

std::vector<ExecutorMetrics> Scheduler::SnapshotExecutors() const
{
    std::vector<ExecutorMetrics> snapshots;
    snapshots.reserve(executors_.size());
    for (const auto &executor : executors_) {
        snapshots.push_back(executor->GetMetrics());
    }
    return snapshots;
}

void Scheduler::OutputMetricsWindow(const std::vector<ExecutorMetrics> &last,
                                    const std::vector<ExecutorMetrics> &current) const
{
    // 汇总全部执行器的窗口增量（两序列长度一致时逐项求差累加）
    std::uint64_t ops = 0;
    std::uint64_t errors = 0;
    std::uint64_t bytes = 0;
    std::uint64_t tasks = 0;
    std::uint64_t latencyNs = 0;
    const std::size_t count = std::min(last.size(), current.size());
    for (std::size_t i = 0; i < count; ++i) {
        ops += current[i].opCount - last[i].opCount;
        errors += current[i].errorCount - last[i].errorCount;
        bytes += current[i].ioBytes - last[i].ioBytes;
        tasks += current[i].taskCount - last[i].taskCount;
        latencyNs += current[i].latencyNsTotal - last[i].latencyNsTotal;
    }

    // 窗口速率：IOPS 与流量按名义周期秒折算；TTFT 与错误率按窗口内样本计算，无样本时占位 "-"
    const double seconds = static_cast<double>(kMetricsIntervalMs) / 1000.0;
    const std::string ttft = (tasks > 0)
        ? FormatLatency(static_cast<double>(latencyNs) / tasks)
        : std::string("-");
    const std::string errorRate = (ops > 0)
        ? fmt::format("{:.2f}%", static_cast<double>(errors) / ops * 100.0)
        : std::string("-");
    LOG_INFO("Scheduler指标[{}s]: IOPS={:.0f} 流量={}/s TTFT={} 错误率={}", kMetricsIntervalMs / 1000,
             static_cast<double>(ops) / seconds,
             FormatBytes(static_cast<double>(bytes) / seconds), ttft, errorRate);
}

void Scheduler::OutputMetricsTotal() const
{
    // 汇总全部执行器的累计指标，作为本轮运行的总计输出
    std::uint64_t ops = 0;
    std::uint64_t errors = 0;
    std::uint64_t bytes = 0;
    std::uint64_t tasks = 0;
    std::uint64_t latencyNs = 0;
    for (const auto &executor : executors_) {
        const ExecutorMetrics metrics = executor->GetMetrics();
        ops += metrics.opCount;
        errors += metrics.errorCount;
        bytes += metrics.ioBytes;
        tasks += metrics.taskCount;
        latencyNs += metrics.latencyNsTotal;
    }

    const std::string ttft = (tasks > 0)
        ? FormatLatency(static_cast<double>(latencyNs) / tasks)
        : std::string("-");
    const std::string errorRate = (ops > 0)
        ? fmt::format("{:.2f}%", static_cast<double>(errors) / ops * 100.0)
        : std::string("-");
    LOG_INFO("Scheduler指标汇总: 总操作数={} 总流量={} 总任务数={} 平均TTFT={} 错误率={}", ops,
             FormatBytes(static_cast<double>(bytes)), tasks, ttft, errorRate);
}
