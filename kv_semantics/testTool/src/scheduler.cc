/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "scheduler.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fmt/format.h>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>
#include <variant>
#include <vector>
#include "log/logger_manager.h"
#include "request/request_orc.h"

namespace {

/** 指标汇总周期（毫秒），秒级；后续随压测参数配置化 */
constexpr int kMetricsIntervalMs = 1000;
/** 指标线程睡眠分片（毫秒），保证停止请求在分片时间内响应 */
constexpr int kMetricsSleepSliceMs = 50;

/** OpType 枚举值对应的接口名（指标分接口输出用；下标即枚举值，与 kOpTypeCount 对齐） */
const char *const kOpNames[kOpTypeCount] = {"exist", "put", "get", "delete"};

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
    if (ns < 1000.0) { return fmt::format("{:.3f}ns", ns); }
    if (ns < 1000000.0) { return fmt::format("{:.3f}us", ns / 1000.0); }
    if (ns < 1000000000.0) { return fmt::format("{:.3f}ms", ns / 1000000.0); }
    return fmt::format("{:.3f}s", ns / 1000000000.0);
}

}  // namespace

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
    // 每块分配后整块写 'A' 作为底色（value 只写前缀不补齐，前缀之后的读回内容为底色或
    // 上次写入的残留），并组装 MemoryRegion（memoryType/deviceId 按 buffertype 填充）一并注册；
    // 任一失败走统一清理，不留半成品
    const std::size_t count = static_cast<std::size_t>(bufferCount);
    std::vector<kv::MemoryRegion> regions;
    regions.reserve(count);
    bool ok = true;
    std::vector<std::uint8_t> initBlock(bufferSize_, 0x41);  // 'A' 底色块，循环外构造一次
    for (std::size_t i = 0; i < count; ++i) {
        std::uint64_t addr = 0;
        if (!allocator_->Allocate(bufferSize_, addr, err)) {
            err = "value buffer " + std::to_string(i) + " allocate failed: " + err;
            ok = false;
            break;
        }
        bufferAddrs_.emplace_back(addr);
        if (!allocator_->Write(addr, initBlock.data(), initBlock.size(), err)) {
            err = "value buffer " + std::to_string(i) + " init 'A' failed: " + err;
            ok = false;
            break;
        }
        kv::MemoryRegion region;
        region.memoryType =
            allocator_->IsDeviceMemory() ? kv::MemoryType::DEVICE : kv::MemoryType::HOST;
        region.addr = addr;
        region.size = bufferSize_;
        region.deviceId = allocator_->IsDeviceMemory() ? deviceId : -1;
        regions.emplace_back(region);
    }
    if (ok && !client_->RegisterRegions(regions, registered_, err)) { ok = false; }
    if (!ok) {
        LOG_ERROR("Scheduler: value buffer pool setup failed: {}", err);
        for (std::uint64_t addr : bufferAddrs_) { allocator_->Deallocate(addr); }
        bufferAddrs_.clear();
        allocator_.reset();
        client_->Shutdown();
        client_.reset();
        bufferSize_ = 0;
        return false;
    }
    LOG_INFO("Scheduler: {} value buffers registered ({} bytes each, type={})", registered_.size(),
             bufferSize, allocator_->IsDeviceMemory() ? "device" : "host");

    // 创建执行器并绑定 client 与分配器（每 worker 一个执行器：独立埋点，指标汇总时求和）；
    // 同时缓存 worker 数与起始绑核编号（worker i 绑 startcoreid+i）
    numJobs_ = config.GetNumJobs();
    startCoreId_ = config.GetStartCoreId();
    executors_.clear();
    for (int i = 0; i < numJobs_; ++i) {
        executors_.push_back(std::make_unique<Executor>());
        executors_.back()->SetClient(client_.get());
        executors_.back()->SetAllocator(allocator_.get());
    }
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

    // 任务流按 sleep 步骤切段（段 = 两次脉冲之间的任务区间），构建段表；执行期只读。
    // 任务流不以 sleep 结尾时补一个收尾段；全 sleep 流自然形成多个空段（仍逐段停顿）
    segments_.clear();
    std::size_t segStart = 0;
    const std::size_t stepCount = requestManager.StepCount();
    for (std::size_t i = 0; i < stepCount; ++i) {
        if (const auto *sleepStep = std::get_if<SleepStep>(&requestManager.GetStep(i))) {
            segments_.push_back(FlowSegment{segStart, i, sleepStep->seconds});
            segStart = i + 1;
        }
    }
    if (segStart < stepCount) { segments_.push_back(FlowSegment{segStart, stepCount, 0}); }

    // 并发执行状态初始化：段游标、barrier 计数、缓冲空闲表（全部块空闲）、失败聚合、限时/圈数
    flow_ = &requestManager;
    segIndex_ = 0;
    cursor_.store(segments_.front().start);
    barrierArrived_ = 0;
    barrierGen_ = 0;
    exiting_ = false;
    rounds_ = 0;
    runtimeSecs_ = runtime;
    loopLimit_ = loop;
    startTime_ = std::chrono::steady_clock::now();
    anyFail_ = false;
    firstErr_.clear();
    freeBuffers_.clear();
    for (std::size_t i = 0; i < bufferAddrs_.size(); ++i) { freeBuffers_.push_back(i); }

    // 启动 numjobs 个 worker 并等待全部回收。圈数/时长控制（conf 语义）：runtime>0 限时模式，
    // 跑满 runtime 秒即停（时间到不开新圈，当前圈跑完）；runtime=0 圈数模式，整个任务流重复
    // loop 遍（Check 已保证 loop>0）。持续执行语义：单任务工具链失败记录后继续执行后续
    // 任务，err 保留首次失败描述
    LOG_INFO("Scheduler: starting {} workers", numJobs_);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(numJobs_));
    for (int i = 0; i < numJobs_; ++i) {
        workers.emplace_back(&Scheduler::WorkerLoop, this, i, testmode);
    }
    for (std::thread &worker : workers) { worker.join(); }
    LOG_INFO("Scheduler: {} rounds executed", rounds_);

    // 停止指标线程并输出运行总计；工具链失败时返回首次失败描述（worker 已全部回收，无并发）
    metricsRunning_.store(false);
    if (metricsThread_.joinable()) { metricsThread_.join(); }
    OutputMetricsTotal();
    if (anyFail_) {
        err = firstErr_;
        return false;
    }
    return true;
}

void Scheduler::WorkerLoop(int workerId, int testmode)
{
    // 绑核（worker i 绑 startcoreid+i；失败仅告警不绑）与线程命名（调试可见，上限 15 字符）
    BindThreadToCore(startCoreId_ + workerId);
    pthread_setname_np(pthread_self(), ("worker-" + std::to_string(workerId)).c_str());

    Executor *executor = executors_[workerId].get();

    // 主循环：锁内查退出标志与当前段末（段推进持同一把锁，不会与本读取并发交错；本 worker
    // 不在 barrier 时全员必未到齐，段不会推进，锁外领任务读到的一定是本段边界）；段内经原子
    // 游标领任务（每任务仅执行一次），领到段末之外说明本段已领完、进 barrier 全员同步
    while (true) {
        std::size_t segEnd = 0;
        {
            std::lock_guard<std::mutex> lock(barrierMutex_);
            if (exiting_) { return; }
            segEnd = segments_[segIndex_].end;
        }
        const std::size_t idx = cursor_.fetch_add(1);
        if (idx >= segEnd) {
            EnterBarrier();
            continue;
        }

        const OrcTask &task = std::get<OrcTask>(flow_->GetStep(idx));
        // 条数超池：永远凑不齐独占块，直接失败可感知（不做必然阻塞的等待）
        if (task.entries.size() > bufferAddrs_.size()) {
            RecordFail("task " + std::to_string(idx) + " entry count " +
                       std::to_string(task.entries.size()) + " exceeds value buffer count " +
                       std::to_string(bufferAddrs_.size()));
            continue;
        }
        std::vector<std::size_t> owned;
        AcquireBuffers(task.entries.size(), owned);

        Request request;
        std::string buildErr;
        if (!buildTaskRequest(task, testmode, owned, request, buildErr)) {
            RecordFail("task " + std::to_string(idx) + " build failed: " + buildErr);
            ReleaseBuffers(owned);
            continue;
        }
        std::string execErr;
        if (!executor->Execute(MapOp(task.op), request, execErr)) {
            RecordFail("task " + std::to_string(idx) + " execute failed: " + execErr);
        }
        ReleaseBuffers(owned);
    }
}

void Scheduler::EnterBarrier()
{
    std::unique_lock<std::mutex> lock(barrierMutex_);
    const int gen = barrierGen_;
    if (++barrierArrived_ < numJobs_) {
        // 先到者等待开闸（gen 变化）：停顿、段推进与退出判定均由最后到者完成
        barrierCv_.wait(lock, [&] { return barrierGen_ != gen; });
        return;
    }

    // 最后到者：此刻全员到齐，本段任务已全部完成——先统一停顿（脉冲；先到者都在条件变量上
    // 等待，持锁停顿不影响任何人），再推进段并开闸
    const FlowSegment &segment = segments_[segIndex_];
    if (segment.pauseSec > 0) {
        LOG_INFO("Scheduler: all workers pause {}s (pulse)", segment.pauseSec);
        std::this_thread::sleep_for(std::chrono::seconds(segment.pauseSec));
    }
    ++segIndex_;
    if (segIndex_ >= segments_.size()) {
        // 圈末：限时模式跑满 runtime 秒不再开新圈，圈数模式跑满 loop 圈退出；否则回到段表开头
        ++rounds_;
        const bool limitReached = (runtimeSecs_ > 0)
                                      ? (std::chrono::steady_clock::now() - startTime_ >=
                                         std::chrono::seconds(runtimeSecs_))
                                      : (rounds_ >= loopLimit_);
        if (limitReached) {
            exiting_ = true;
        } else {
            segIndex_ = 0;
        }
    }
    if (!exiting_) { cursor_.store(segments_[segIndex_].start); }
    barrierArrived_ = 0;
    ++barrierGen_;
    barrierCv_.notify_all();
}

void Scheduler::AcquireBuffers(std::size_t count, std::vector<std::size_t> &owned)
{
    // 调用方已保证 count 不超池总量（否则必然死锁），此处只需等待空闲块凑齐；
    // 从空闲表尾取走 count 块（块间无差异，取哪批都一样）
    std::unique_lock<std::mutex> lock(bufferMutex_);
    bufferCv_.wait(lock, [&] { return freeBuffers_.size() >= count; });
    owned.assign(freeBuffers_.end() - static_cast<std::ptrdiff_t>(count), freeBuffers_.end());
    freeBuffers_.resize(freeBuffers_.size() - count);
}

void Scheduler::ReleaseBuffers(const std::vector<std::size_t> &owned)
{
    {
        std::lock_guard<std::mutex> lock(bufferMutex_);
        freeBuffers_.insert(freeBuffers_.end(), owned.begin(), owned.end());
    }
    // 等待者可能各等不同块数，统一唤醒由各自重查谓词
    bufferCv_.notify_all();
}

void Scheduler::BindThreadToCore(int core)
{
    // Linux 平台绑核直调（与线程命名同一先例）；失败仅告警不绑，不影响多线程执行
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        LOG_WARN("Scheduler: bind worker to core {} failed, running unbound", core);
    }
}

void Scheduler::RecordFail(const std::string &message)
{
    // 打错误日志并保留首次失败描述（Run 返回时经 err 报告）
    LOG_ERROR("Scheduler: {}", message);
    std::lock_guard<std::mutex> lock(failMutex_);
    if (!anyFail_) {
        anyFail_ = true;
        firstErr_ = message;
    }
}

void Scheduler::Shutdown()
{
    // 停止并回收指标线程
    metricsRunning_.store(false);
    if (metricsThread_.joinable()) { metricsThread_.join(); }

    // 注销请求内存（全部任务均已等待完成）后关闭 kv client
    if (client_ != nullptr) {
        std::string err;
        std::vector<kv::MRHandle> handles;
        handles.reserve(registered_.size());
        for (const auto &item : registered_) { handles.emplace_back(item.handle); }
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
        for (std::uint64_t addr : bufferAddrs_) { allocator_->Deallocate(addr); }
        bufferAddrs_.clear();
        bufferSize_ = 0;
        allocator_.reset();
    }

    // 释放执行器（其持有的 client/allocator 指针随之一并失效）
    executors_.clear();
}

bool Scheduler::buildTaskRequest(const OrcTask &task, int testmode,
                                 const std::vector<std::size_t> &owned, Request &request,
                                 std::string &err)
{
    request.nsid = task.nsid;
    request.keys.reserve(task.entries.size());
    request.keyHighs.reserve(task.entries.size());
    request.rawKeys.reserve(task.entries.size());
    request.prefixLens.reserve(task.entries.size());
    request.entries.reserve(task.entries.size());
    request.expected.reserve(task.entries.size());

    // mode1 种子串时间戳在 for 外生成一次：同一任务（batch）内全部 entry 共用
    const std::time_t seedTime = std::time(nullptr);

    for (std::size_t index = 0; index < task.entries.size(); ++index) {
        const RequestOrc &orc = task.entries[index];

        // key 低 64 位填 CacheKey
        kv::CacheKey key{};
        std::memcpy(key.data(), &orc.keyLow64, sizeof(orc.keyLow64));

        // 生成 value 前缀（只写前缀不补齐，尺寸 ≤ orc.length），写入本任务领用的第 index 块
        // 独占缓冲的起始段（device 语义经分配器 H2D 拷贝）；前缀之后的缓冲内容保持启动
        // memset 的 'A' 底色或上次写入的残留
        const std::vector<std::uint8_t> block = BuildValueBlock(orc, testmode, seedTime);
        if (orc.length > bufferSize_) {
            err = "entry length " + std::to_string(orc.length) + " exceeds value buffer size " +
                  std::to_string(bufferSize_);
            return false;
        }
        // 空前缀不写缓冲（保持底色/残留原样），非空前缀才落缓冲
        if (!block.empty() &&
            !allocator_->Write(bufferAddrs_[owned[index]], block.data(), block.size(), err)) {
            err = "value block write failed: " + err;
            return false;
        }

        // 组装请求项：绑定领用块的注册凭证；region.addr 为块起始、region.size 为本次 IO 长度
        // （= orc.length，即前缀 + 前缀之后的缓冲现状；kv client 语义）、offset 为 KV 块内
        // 偏移（存储侧，须 512 对齐，由 kv client 校验）
        kv::KVBuffer entry;
        entry.key = key;
        entry.buffer.region = registered_[owned[index]].region;
        entry.buffer.region.size = orc.length;
        entry.buffer.handle = registered_[owned[index]].handle;
        entry.offset = orc.offset;

        // 基准表（基准 = 本次写入的前缀，get 只比前缀长度，前缀之后的底色/残留不比对）：
        // PUT 记录前缀为最近写入基准；GET 查基准（命中才有比对依据）；其他操作无基准。
        // 多 worker 并发访问基准表，PUT 写 / GET 读两触点加锁
        if (task.op == OrcOp::PUT) {
            std::lock_guard<std::mutex> lock(baselineMutex_);
            keyBaseline_[orc.keyLow64] = block;
            request.expected.emplace_back();
        } else if (task.op == OrcOp::GET) {
            std::lock_guard<std::mutex> lock(baselineMutex_);
            const auto it = keyBaseline_.find(orc.keyLow64);
            request.expected.emplace_back(it != keyBaseline_.end() ? it->second
                                                                   : std::vector<std::uint8_t>());
        } else {
            request.expected.emplace_back();
        }

        request.keys.emplace_back(key);
        request.keyHighs.emplace_back(orc.keyHigh16);
        request.rawKeys.emplace_back(orc.rawKey);
        // 请求明细日志 value 读回长度（与 len 显示字段解耦）：mode1 为本次生成前缀的实际
        // 长度（种子串有效部分），mode2 为完整块长度（orc.length）
        request.prefixLens.emplace_back(testmode == 2 ? static_cast<std::size_t>(orc.length)
                                                      : block.size());
        request.entries.emplace_back(entry);
    }
    return true;
}

OpType Scheduler::MapOp(OrcOp op)
{
    switch (op) {
        case OrcOp::GET: return OpType::LOAD;
        case OrcOp::PUT: return OpType::STORE;
        case OrcOp::EXIST: return OpType::QUERY;
        case OrcOp::DELETE: return OpType::DELETE;
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
            if (!metricsRunning_.load(std::memory_order_relaxed)) { return; }
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
    for (const auto &executor : executors_) { snapshots.push_back(executor->GetMetrics()); }
    return snapshots;
}

void Scheduler::OutputMetricsWindow(const std::vector<ExecutorMetrics> &last,
                                    const std::vector<ExecutorMetrics> &current) const
{
    // 汇总全部执行器的窗口增量（两序列长度一致时逐项求差累加）；分接口桶同步累加
    std::uint64_t ops = 0;
    std::uint64_t errors = 0;
    std::uint64_t notFound = 0;
    std::uint64_t mismatch = 0;
    std::uint64_t bytes = 0;
    std::uint64_t tasks = 0;
    std::uint64_t latencyNs = 0;
    std::uint64_t opsByOp[kOpTypeCount] = {0};
    std::uint64_t bytesByOp[kOpTypeCount] = {0};
    const std::size_t count = std::min(last.size(), current.size());
    for (std::size_t i = 0; i < count; ++i) {
        ops += current[i].opCount - last[i].opCount;
        errors += current[i].errorCount - last[i].errorCount;
        notFound += current[i].notFoundCount - last[i].notFoundCount;
        mismatch += current[i].mismatchCount - last[i].mismatchCount;
        bytes += current[i].ioBytes - last[i].ioBytes;
        tasks += current[i].taskCount - last[i].taskCount;
        latencyNs += current[i].latencyNsTotal - last[i].latencyNsTotal;
        for (std::size_t op = 0; op < kOpTypeCount; ++op) {
            opsByOp[op] += current[i].opCountByOp[op] - last[i].opCountByOp[op];
            bytesByOp[op] += current[i].ioBytesByOp[op] - last[i].ioBytesByOp[op];
        }
    }

    // 窗口速率：IOPS 与流量按名义周期秒折算；TTFT 与错误率按窗口内样本计算，无样本时占位 "-"
    const double seconds = static_cast<double>(kMetricsIntervalMs) / 1000.0;
    const std::string ttft =
        (tasks > 0) ? FormatLatency(static_cast<double>(latencyNs) / tasks) : std::string("-");
    const std::string errorRate =
        (ops > 0) ? fmt::format("{:.2f}%", static_cast<double>(errors) / ops * 100.0)
                  : std::string("-");
    LOG_INFO(
        "Scheduler metrics [{}s]: IOPS={:.0f} traffic={}/s TTFT={} error rate={} notfound={} "
        "mismatch={}",
        kMetricsIntervalMs / 1000, static_cast<double>(ops) / seconds,
        FormatBytes(static_cast<double>(bytes) / seconds), ttft, errorRate, notFound, mismatch);
    // 分接口行（窗口内有增量的接口才输出；exist/delete 无载荷仅 IOPS 有意义，流量恒 0）
    for (std::size_t op = 0; op < kOpTypeCount; ++op) {
        if (opsByOp[op] == 0) { continue; }
        LOG_INFO("Scheduler metrics [{}s]: {} IOPS={:.0f} traffic={}/s", kMetricsIntervalMs / 1000,
                 kOpNames[op], static_cast<double>(opsByOp[op]) / seconds,
                 FormatBytes(static_cast<double>(bytesByOp[op]) / seconds));
    }
}

void Scheduler::OutputMetricsTotal() const
{
    // 汇总全部执行器的累计指标，作为本轮运行的总计输出；分接口桶同步累加
    std::uint64_t ops = 0;
    std::uint64_t errors = 0;
    std::uint64_t notFound = 0;
    std::uint64_t mismatch = 0;
    std::uint64_t bytes = 0;
    std::uint64_t tasks = 0;
    std::uint64_t latencyNs = 0;
    std::uint64_t opsByOp[kOpTypeCount] = {0};
    std::uint64_t bytesByOp[kOpTypeCount] = {0};
    for (const auto &executor : executors_) {
        const ExecutorMetrics metrics = executor->GetMetrics();
        ops += metrics.opCount;
        errors += metrics.errorCount;
        notFound += metrics.notFoundCount;
        mismatch += metrics.mismatchCount;
        bytes += metrics.ioBytes;
        tasks += metrics.taskCount;
        latencyNs += metrics.latencyNsTotal;
        for (std::size_t op = 0; op < kOpTypeCount; ++op) {
            opsByOp[op] += metrics.opCountByOp[op];
            bytesByOp[op] += metrics.ioBytesByOp[op];
        }
    }

    const std::string ttft =
        (tasks > 0) ? FormatLatency(static_cast<double>(latencyNs) / tasks) : std::string("-");
    const std::string errorRate =
        (ops > 0) ? fmt::format("{:.2f}%", static_cast<double>(errors) / ops * 100.0)
                  : std::string("-");
    LOG_INFO(
        "Scheduler metrics total: ops={} traffic={} tasks={} avg TTFT={} error rate={} notfound={} "
        "mismatch={}",
        ops, FormatBytes(static_cast<double>(bytes)), tasks, ttft, errorRate, notFound, mismatch);
    // 分接口行（全程有操作的接口才输出；exist/delete 无载荷，流量恒 0）
    for (std::size_t op = 0; op < kOpTypeCount; ++op) {
        if (opsByOp[op] == 0) { continue; }
        LOG_INFO("Scheduler metrics total: {} ops={} traffic={}", kOpNames[op], opsByOp[op],
                 FormatBytes(static_cast<double>(bytesByOp[op])));
    }
}
