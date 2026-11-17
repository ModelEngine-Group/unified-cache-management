/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <pthread.h>

#include <fmt/format.h>

#include "log/logger_manager.h"
#include "request/request_orc.h"

namespace {

/** 指标汇总周期（毫秒），秒级；后续随压测参数配置化 */
constexpr int kMetricsIntervalMs = 1000;
/** 指标线程睡眠分片（毫秒），保证停止请求在分片时间内响应 */
constexpr int kMetricsSleepSliceMs = 50;

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

bool Scheduler::initializeAndRegister(const std::string &confPath, const ConfigUtil &config,
                                     std::string &err)
{
    const std::int64_t bufferSize = config.GetBufferSize();
    const int bufferCount = config.GetBufferCount();
    const int deviceId = config.GetDeviceId();
    // 池参数合法性已由 ConfigUtil::Check 在配置加载阶段前置校验
    bufferSize_ = static_cast<std::size_t>(bufferSize);

    // kv client 初始化与建链
    client_ = std::make_unique<KvClientUtil>();
    if (!client_->Init(confPath, err)) {
        // 失败不留半成品：清掉空壳对象，client_ 保持"未初始化"语义
        client_.reset();
        bufferSize_ = 0;
        return false;
    }

    // 创建值缓冲内存分配器（buffertype：host 主机内存 / device 设备内存）
    allocator_ = CreateDeviceAllocator(config.GetBufferType(), deviceId, err);
    if (allocator_ == nullptr) {
        LOG_ERROR("Scheduler: create value buffer allocator failed: {}", err);
        client_->Shutdown();
        client_.reset();
        bufferSize_ = 0;
        return false;
    }

    // 值缓冲池一次性分配（bufferCount 块 × bufferSize 字节；尺寸来自 conf，与 input 无关，
    // 参考 kv_test：region 注册仅发生在建链阶段，执行期只按 entry 序号复用），
    // 每块即组装一个 MemoryRegion（memoryType/deviceId 按 buffertype 填充）一并注册；
    // 任一失败走统一清理，不留半成品
    const std::size_t count = static_cast<std::size_t>(bufferCount);
    std::vector<kv::MemoryRegion> regions;
    regions.reserve(count);
    bool ok = true;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint64_t addr = 0;
        if (!allocator_->Allocate(bufferSize_, addr, err)) {
            err = "value buffer " + std::to_string(i) + " allocate failed: " + err;
            ok = false;
            break;
        }
        bufferAddrs_.emplace_back(addr);
        kv::MemoryRegion region;
        region.memoryType =
            allocator_->IsDeviceMemory() ? kv::MemoryType::DEVICE : kv::MemoryType::HOST;
        region.addr = addr;
        region.size = bufferSize_;
        region.deviceId = allocator_->IsDeviceMemory() ? deviceId : -1;
        regions.emplace_back(region);
    }
    if (ok && !client_->RegisterRegions(regions, registered_, err)) {
        ok = false;
    }
    if (!ok) {
        LOG_ERROR("Scheduler: value buffer pool setup failed: {}", err);
        for (std::uint64_t addr : bufferAddrs_) {
            allocator_->Deallocate(addr);
        }
        bufferAddrs_.clear();
        allocator_.reset();
        client_->Shutdown();
        client_.reset();
        bufferSize_ = 0;
        return false;
    }
    LOG_INFO("Scheduler: {} value buffers registered ({} bytes each, type={})", registered_.size(),
             bufferSize, allocator_->IsDeviceMemory() ? "device" : "host");

    // 创建执行器并绑定 client 与分配器（当前单个；多线程并发时按 numjobs 创建多个）
    executors_.clear();
    executors_.push_back(std::make_unique<Executor>());
    executors_.back()->SetClient(client_.get());
    executors_.back()->SetAllocator(allocator_.get());
    return true;
}

bool Scheduler::Run(const RequestManager &requestManager, int testmode, int runtime, int loop,
                    std::string &err)
{
    if (client_ == nullptr) {
        err = "scheduler not initialized";
        return false;
    }
    if (executors_.empty()) {
        err = "scheduler has no executor";
        return false;
    }

    // 启动指标汇总线程（秒级周期输出窗口指标）
    metricsRunning_.store(true);
    metricsThread_ = std::thread(&Scheduler::MetricsLoop, this);

    // 按行序执行任务流：sleep 步骤原地暂停；kv 任务组装后逐个执行。
    // 持续执行语义：单任务工具链失败记录后继续执行后续任务，err 保留首次失败描述。
    // 圈数/时长控制（conf 语义）：runtime>0 限时模式，跑满 runtime 分钟即停（时间到不开
    // 新圈，当前圈跑完）；runtime=0 圈数模式，整个任务流重复 loop 遍（Check 已保证 loop>0）
    bool toolChainOk = true;
    Executor *executor = executors_[0].get();
    const auto startTime = std::chrono::steady_clock::now();
    int round = 0;
    while (true) {
        for (std::size_t i = 0; i < requestManager.StepCount(); ++i) {
            const FlowStep &step = requestManager.GetStep(i);
            if (const auto *sleepStep = std::get_if<SleepStep>(&step)) {
                LOG_INFO("Scheduler: sleep {}s", sleepStep->seconds);
                std::this_thread::sleep_for(std::chrono::seconds(sleepStep->seconds));
                continue;
            }
            const OrcTask &task = std::get<OrcTask>(step);

            Request request;
            if (!buildTaskRequest(task, testmode, request, err)) {
                LOG_ERROR("Scheduler: task {} build failed: {}", i, err);
                toolChainOk = false;
                continue;
            }

            std::string execErr;
            if (!executor->Execute(MapOp(task.op), request, execErr)) {
                LOG_ERROR("Scheduler: task {} execute failed: {}", i, execErr);
                if (toolChainOk) {
                    err = execErr;
                }
                toolChainOk = false;
            }
        }
        ++round;
        if (runtime > 0) {
            if (std::chrono::steady_clock::now() - startTime >= std::chrono::minutes(runtime)) {
                break;
            }
        } else if (round >= loop) {
            break;
        }
    }
    LOG_INFO("Scheduler: {} rounds executed", round);

    // 停止指标线程并输出运行总计
    metricsRunning_.store(false);
    if (metricsThread_.joinable()) {
        metricsThread_.join();
    }
    OutputMetricsTotal();
    return toolChainOk;
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
        std::vector<kv::MRHandle> handles;
        handles.reserve(registered_.size());
        for (const auto &item : registered_) {
            handles.emplace_back(item.handle);
        }
        if (!client_->UnregisterRegions(handles, err)) {
            LOG_WARN("Scheduler: request memory unregister failed: {}", err);
        } else {
            // 凭证已失效，清空防止误用
            registered_.clear();
            keyBaseline_.clear();
        }
        client_->Shutdown();
        client_.reset();
    }

    // 释放值缓冲池内存（已注销，地址不再被设备侧引用）
    if (allocator_ != nullptr) {
        for (std::uint64_t addr : bufferAddrs_) {
            allocator_->Deallocate(addr);
        }
        bufferAddrs_.clear();
        bufferSize_ = 0;
        allocator_.reset();
    }

    // 释放执行器（其持有的 client/allocator 指针随之一并失效）
    executors_.clear();
}

bool Scheduler::buildTaskRequest(const OrcTask &task, int testmode, Request &request, std::string &err)
{
    // 任务条数超出池块数：无法为每条 entry 分配独立复用块，报错可感知
    if (task.entries.size() > bufferAddrs_.size()) {
        err = "task entry count " + std::to_string(task.entries.size()) +
              " exceeds value buffer count " + std::to_string(bufferAddrs_.size());
        return false;
    }

    request.nsid = task.nsid;
    request.keys.reserve(task.entries.size());
    request.keyHighs.reserve(task.entries.size());
    request.entries.reserve(task.entries.size());
    request.expected.reserve(task.entries.size());

    for (std::size_t index = 0; index < task.entries.size(); ++index) {
        const RequestOrc &orc = task.entries[index];

        // key 低 64 位填 CacheKey
        kv::CacheKey key{};
        std::memcpy(key.data(), &orc.keyLow64, sizeof(orc.keyLow64));

        // value 块生成并写入第 index 块复用缓冲的起始段（device 语义经分配器 H2D 拷贝）
        const std::vector<std::uint8_t> block = BuildValueBlock(orc, testmode);
        if (block.size() > bufferSize_) {
            err = "entry length " + std::to_string(block.size()) + " exceeds value buffer size " +
                  std::to_string(bufferSize_);
            return false;
        }
        if (!allocator_->Write(bufferAddrs_[index], block.data(), block.size(), err)) {
            err = "value block write failed: " + err;
            return false;
        }

        // 组装请求项：绑定注册凭证；region.addr 为块起始、region.size 为本次 IO 长度（kv client 语义）、
        // offset 为 KV 块内偏移（存储侧，须 512 对齐，由 kv client 校验）
        kv::KVBuffer entry;
        entry.key = key;
        entry.buffer.region = registered_[index].region;
        entry.buffer.region.size = block.size();
        entry.buffer.handle = registered_[index].handle;
        entry.offset = orc.offset;

        // 基准表：PUT 记录本块为最近写入基准；GET 查基准（命中才有比对依据）；其他操作无基准
        if (task.op == OrcOp::PUT) {
            keyBaseline_[orc.keyLow64] = block;
            request.expected.emplace_back();
        } else if (task.op == OrcOp::GET) {
            const auto it = keyBaseline_.find(orc.keyLow64);
            request.expected.emplace_back(it != keyBaseline_.end() ? it->second
                                                                   : std::vector<std::uint8_t>());
        } else {
            request.expected.emplace_back();
        }

        request.keys.emplace_back(key);
        request.keyHighs.emplace_back(orc.keyHigh16);
        request.entries.emplace_back(entry);
    }
    return true;
}

OpType Scheduler::MapOp(OrcOp op)
{
    switch (op) {
        case OrcOp::GET:
            return OpType::LOAD;
        case OrcOp::PUT:
            return OpType::STORE;
        case OrcOp::EXIST:
            return OpType::QUERY;
        case OrcOp::DELETE:
            return OpType::DELETE;
    }
    // 枚举已穷尽；GCC 对穷尽 switch 仍要求函数末尾有 return（-Wreturn-type），此行为不可达占位
    return OpType::LOAD;
}

void Scheduler::MetricsLoop()
{
    // 指标线程命名（调试可见：gdb info threads、top -H、/proc/<pid>/task/<tid>/comm；上限 15 字符）
    pthread_setname_np(pthread_self(), "metrics-loop");

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
    std::uint64_t notFound = 0;
    std::uint64_t mismatch = 0;
    std::uint64_t bytes = 0;
    std::uint64_t tasks = 0;
    std::uint64_t latencyNs = 0;
    const std::size_t count = std::min(last.size(), current.size());
    for (std::size_t i = 0; i < count; ++i) {
        ops += current[i].opCount - last[i].opCount;
        errors += current[i].errorCount - last[i].errorCount;
        notFound += current[i].notFoundCount - last[i].notFoundCount;
        mismatch += current[i].mismatchCount - last[i].mismatchCount;
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
    LOG_INFO("Scheduler metrics [{}s]: IOPS={:.0f} traffic={}/s TTFT={} error rate={} notfound={} mismatch={}",
             kMetricsIntervalMs / 1000, static_cast<double>(ops) / seconds,
             FormatBytes(static_cast<double>(bytes) / seconds), ttft, errorRate, notFound, mismatch);
}

void Scheduler::OutputMetricsTotal() const
{
    // 汇总全部执行器的累计指标，作为本轮运行的总计输出
    std::uint64_t ops = 0;
    std::uint64_t errors = 0;
    std::uint64_t notFound = 0;
    std::uint64_t mismatch = 0;
    std::uint64_t bytes = 0;
    std::uint64_t tasks = 0;
    std::uint64_t latencyNs = 0;
    for (const auto &executor : executors_) {
        const ExecutorMetrics metrics = executor->GetMetrics();
        ops += metrics.opCount;
        errors += metrics.errorCount;
        notFound += metrics.notFoundCount;
        mismatch += metrics.mismatchCount;
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
    LOG_INFO("Scheduler metrics total: ops={} traffic={} tasks={} avg TTFT={} error rate={} notfound={} mismatch={}",
             ops, FormatBytes(static_cast<double>(bytes)), tasks, ttft, errorRate, notFound, mismatch);
}
