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
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>
#include "log/logger_manager.h"
#include "request/request_orc.h"

namespace {

/** 指标汇总周期（毫秒） */
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

/** 基准表复合键：ns 与 keyLow64 以 NUL 拼接（ns 不含 NUL，组合唯一） */
std::string BaselineKey(const std::string &ns, std::uint64_t keyLow64)
{
    return ns + '\0' + std::to_string(keyLow64);
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
    const std::size_t bufferCount = config.GetBufferCount();
    const int deviceId = config.GetDeviceId();
    // 池参数已由 ConfigUtil::Check 前置校验
    bufferSize_ = config.GetBufferSize();

    // kv client 初始化与建链
    client_ = std::make_unique<KvClientUtil>();
    if (!client_->Init(confPath, err)) {
        // 失败不留半成品
        client_.reset();
        bufferSize_ = 0;
        return false;
    }

    // 创建值缓冲内存分配器（buffertype：cpu 主机内存 / gpu CUDA 显存 / npu ACL NPU 内存）
    allocator_ = CreateDeviceAllocator(config.GetBufferType(), deviceId, err);
    if (allocator_ == nullptr) {
        LOG_ERROR("Scheduler: create value buffer allocator failed: {}", err);
        client_->Shutdown();
        client_.reset();
        bufferSize_ = 0;
        return false;
    }

    // 值缓冲池一次性分配（bufferCount 块 × bufferSize 字节）：每块整块写 'A' 底色
    // （value 只写前缀不补齐，前缀之后保持底色或残留）并组装 MemoryRegion 注册；
    // 任一失败统一清理
    std::vector<kv::MemoryRegion> regions;
    regions.reserve(bufferCount);
    bool ok = true;
    std::vector<std::uint8_t> initBlock(bufferSize_, 0x41);  // 'A' 底色块，循环外构造一次
    for (std::size_t i = 0; i < bufferCount; ++i) {
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
             bufferSize_, allocator_->IsDeviceMemory() ? "device" : "host");

    return true;
}

bool Scheduler::PrepareExecutors(const ConfigUtil &config, std::string &err)
{
    // 调用序防御：执行器依赖 initializeAndRegister 建立的 client 与分配器
    if (client_ == nullptr || allocator_ == nullptr) {
        err = "executors require a successful initializeAndRegister first";
        return false;
    }

    // 创建执行器并绑定 client 与分配器（每 worker 一个，指标汇总时求和）
    numJobs_ = config.GetNumJobs();
    startCoreId_ = config.GetStartCoreId();
    executors_.clear();
    for (int i = 0; i < numJobs_; ++i) {
        executors_.push_back(std::make_unique<Executor>());
        executors_.back()->SetClient(client_.get());
        executors_.back()->SetAllocator(allocator_.get());
        executors_.back()->SetPerfMode(config.GetPerf());
    }
    LOG_INFO("Scheduler: {} executors prepared (startcoreid={})", executors_.size(), startCoreId_);
    p99IntervalSecs_ = config.GetP99IntervalSecs();
    p99ElapsedSecs_ = 0;
    return true;
}

bool Scheduler::Run(const RequestManager &requestManager, int testmode, int runtime, int loop,
                    int rateLimitMb, std::string &err)
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
    rateLimitBytesPerSec_ = static_cast<std::uint64_t>(rateLimitMb) * 1024ULL * 1024ULL;
    rateWindowStart_ = startTime_;
    rateWindowUsed_ = 0;
    if (rateLimitMb > 0) {
        LOG_INFO("Scheduler: rate limit {} MB/s ({} bytes/s per 1s window)", rateLimitMb,
                 rateLimitBytesPerSec_);
    }
    anyFail_ = false;
    firstErr_.clear();
    freeBuffers_.clear();
    for (std::size_t i = 0; i < bufferAddrs_.size(); ++i) { freeBuffers_.push_back(i); }

    // 启动 worker 并等待全部回收。runtime>0 限时模式，跑满即停（不开新圈）；
    // runtime=0 圈数模式（Check 已保证 loop>0）；单任务工具链失败记录后继续执行，
    // err 保留首次失败描述
    LOG_INFO("Scheduler: starting {} workers", numJobs_);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(numJobs_));
    for (int i = 0; i < numJobs_; ++i) {
        workers.emplace_back(&Scheduler::WorkerLoop, this, i, testmode);
    }
    for (std::thread &worker : workers) { worker.join(); }
    LOG_INFO("Scheduler: {} rounds executed", rounds_);

    // 停止指标线程并输出运行总计；失败时返回首次失败描述（worker 已全部回收）
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
    // 绑核（worker i 绑 startcoreid+i）与线程命名（上限 15 字符）
    try {
        BindThreadToCore(startCoreId_ + workerId);
    } catch (const std::exception &e) {
        // 绑核失败不继续执行：记录失败并唤醒全部 worker 退出
        RecordFail("worker " + std::to_string(workerId) + ": " + e.what());
        {
            std::lock_guard<std::mutex> lock(barrierMutex_);
            exiting_ = true;
            ++barrierGen_;
            barrierCv_.notify_all();
        }
        return;
    }
    pthread_setname_np(pthread_self(), ("test_worker-" + std::to_string(workerId)).c_str());

    Executor *executor = executors_[workerId].get();

    // 主循环：锁内查退出标志与当前段末，段内经原子游标领任务（每任务仅执行一次），
    // 领完本段进 barrier 全员同步
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
        // exist/delete 只按 key 提交（无数据面），不占用 value 缓冲块
        const std::size_t bufferNeed = IsKeyOnlyOp(task.op) ? 0 : task.entries.size();
        // 条数超池：永远凑不齐独占块，直接失败（仅 get/put 领块）
        if (bufferNeed > bufferAddrs_.size()) {
            RecordFail("task " + std::to_string(idx) + " entry count " +
                       std::to_string(task.entries.size()) + " exceeds value buffer count " +
                       std::to_string(bufferAddrs_.size()));
            continue;
        }
        // 秒级限速：按任务 IO 字节量申请配额，额度不足时阻塞等到下一窗口（不占缓冲块，
        // 等待期间其他 worker 可继续领任务执行）
        std::uint64_t taskBytes = 0;
        for (const auto &orc : task.entries) { taskBytes += orc.length; }
        WaitForRateLimit(taskBytes);
        std::vector<std::size_t> owned;
        AcquireBuffers(bufferNeed, owned);

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
        // 先到者等待开闸（gen 变化）
        barrierCv_.wait(lock, [&] { return barrierGen_ != gen; });
        return;
    }

    // 最后到者：先统一停顿（脉冲），再推进段并开闸
    const FlowSegment &segment = segments_[segIndex_];
    if (segment.pauseSec > 0) {
        LOG_INFO("Scheduler: all workers pause {}s (pulse)", segment.pauseSec);
        std::this_thread::sleep_for(std::chrono::seconds(segment.pauseSec));
    }
    ++segIndex_;
    if (segIndex_ >= segments_.size()) {
        // 圈末：限时模式跑满 runtime 秒不再开新圈，圈数模式跑满 loop 圈退出
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

void Scheduler::WaitForRateLimit(std::uint64_t bytes)
{
    if (rateLimitBytesPerSec_ == 0 || bytes == 0) { return; }  // 不限速 / 无流量直接放行
    // 单任务超整秒预算：直接放行（否则本任务永远凑不齐配额）
    if (bytes >= rateLimitBytesPerSec_) { return; }
    while (true) {
        std::chrono::steady_clock::time_point wakeAt{};
        {
            std::lock_guard<std::mutex> lock(rateMutex_);
            const auto now = std::chrono::steady_clock::now();
            // 跨秒滚动窗口：窗口起点推进到当前时刻，已用字节清零
            if (now >= rateWindowStart_ + std::chrono::seconds(1)) {
                rateWindowStart_ = now;
                rateWindowUsed_ = 0;
            }
            if (rateWindowUsed_ + bytes <= rateLimitBytesPerSec_) {
                rateWindowUsed_ += bytes;
                return;
            }
            // 本窗口额度不足：等到本窗口起点 +1 秒（解锁后再睡，不持锁等待）
            wakeAt = rateWindowStart_ + std::chrono::seconds(1);
        }
        std::this_thread::sleep_until(wakeAt);
    }
}

void Scheduler::AcquireBuffers(std::size_t count, std::vector<std::size_t> &owned)
{
    // 调用方已保证 count 不超池总量，等待空闲块凑齐后从表尾取走
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
    // Linux 平台绑核；失败抛异常（核数不匹配属配置错误，fail-fast 不静默降级）
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    // pthread 系函数惯例：返回值即错误码
    const int bindErr = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (bindErr != 0) {
        throw std::runtime_error("bind worker to core " + std::to_string(core) +
                                 " failed: " + std::strerror(bindErr));
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
    request.ns = task.ns;
    request.keys.reserve(task.entries.size());
    request.keyHighs.reserve(task.entries.size());
    request.rawKeys.reserve(task.entries.size());
    request.prefixLens.reserve(task.entries.size());
    request.entries.reserve(task.entries.size());
    request.expected.reserve(task.entries.size());

    // mode1 种子串时间戳 batch 内共用，for 外生成一次
    const std::time_t seedTime = std::time(nullptr);

    for (std::size_t index = 0; index < task.entries.size(); ++index) {
        const RequestOrc &orc = task.entries[index];

        // key 低 64 位十进制串构造 string key：走 KeyResolver 字符串分支，请求级 namespace
        // （IoOptions.kvns）才能生效（binary key 不消费 kvns；generic 引擎下整串身份
        // XXH64 派生 8 字节路由 key，同 key 同哈希、读写删功能等价）
        kv::CacheKey key = kv::CacheKey::FromString(std::to_string(orc.keyLow64));

        // exist/delete 走 keys-only 接口：只组装 key 三项，不占数据缓冲块、无基准与读回
        if (IsKeyOnlyOp(task.op)) {
            request.keys.emplace_back(key);
            request.keyHighs.emplace_back(orc.keyHigh16);
            request.rawKeys.emplace_back(orc.rawKey);
            continue;
        }

        // value 前缀（只写前缀不补齐）写入领用块起始（device 语义经分配器 H2D 拷贝）；
        // GET 例外——预填哨兵串（见下）
        const std::vector<std::uint8_t> block = BuildValueBlock(orc, testmode, seedTime);
        if (orc.length > bufferSize_) {
            err = "entry length " + std::to_string(orc.length) + " exceeds value buffer size " +
                  std::to_string(bufferSize_);
            return false;
        }
        if (task.op == OrcOp::GET) {
            // GET：块起始预填哨兵串——命中时被读回数据覆盖，未命中时明细日志显示
            // 哨兵串即为“没读到数据”标记
            constexpr char kGetSentinel[] = "unknownvaluebeforegetrequest,,,";
            const std::size_t sentinelLen = std::min(bufferSize_, sizeof(kGetSentinel) - 1);
            if (!allocator_->Write(bufferAddrs_[owned[index]], kGetSentinel, sentinelLen, err)) {
                err = "get sentinel prefill failed: " + err;
                return false;
            }
        } else if (!block.empty() && !allocator_->Write(bufferAddrs_[owned[index]], block.data(),
                                                        block.size(), err)) {
            // 空前缀不写缓冲，非空前缀才落缓冲
            err = "value block write failed: " + err;
            return false;
        }

        // 组装请求项：绑定领用块的注册凭证；region.size 为本次 IO 长度（orc.length）、
        // offset 为 KV 块内偏移（存储侧，须 512 对齐，由 kv client 校验）
        kv::KVBuffer entry;
        entry.key = key;
        entry.buffer.region = registered_[owned[index]].region;
        entry.buffer.region.size = orc.length;
        entry.buffer.handle = registered_[owned[index]].handle;
        entry.offset = orc.offset;

        // 基准表：PUT 记录前缀为最近写入基准，GET 查基准（命中才有比对依据），
        // 均以 ns + key 复合定位（同 key 不同 ns 为不同数据，防基准互相覆盖）；
        // 其他操作无基准；多 worker 并发访问加锁
        if (task.op == OrcOp::PUT) {
            std::lock_guard<std::mutex> lock(baselineMutex_);
            keyBaseline_[BaselineKey(task.ns, orc.keyLow64)] = block;
            request.expected.emplace_back();
        } else if (task.op == OrcOp::GET) {
            std::lock_guard<std::mutex> lock(baselineMutex_);
            const auto it = keyBaseline_.find(BaselineKey(task.ns, orc.keyLow64));
            request.expected.emplace_back(it != keyBaseline_.end() ? it->second
                                                                   : std::vector<std::uint8_t>());
        } else {
            request.expected.emplace_back();
        }

        request.keys.emplace_back(key);
        request.keyHighs.emplace_back(orc.keyHigh16);
        request.rawKeys.emplace_back(orc.rawKey);
        // 明细日志 value 读回长度：mode1 为 keyLow/keyHigh/offset 三数十进制位数加
        // 两个空格分隔，mode2 为完整块长度
        const std::string keyLowStr = std::to_string(orc.keyLow64);
        const std::string keyHighStr = std::to_string(orc.keyHigh16);
        const std::string offsetStr = std::to_string(orc.offset);
        request.prefixLens.emplace_back(testmode == 2 ? static_cast<std::size_t>(orc.length)
                                                      : keyLowStr.size() + keyHighStr.size() +
                                                            offsetStr.size() + 2);
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
    // 枚举已穷尽，此行为不可达占位（-Wreturn-type 要求函数末尾有 return）
    return OpType::LOAD;
}

void Scheduler::MetricsLoop()
{
    // 指标线程命名（上限 15 字符）
    pthread_setname_np(pthread_self(), "test_metrics");

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
        // P99 分位行按独立周期打印（p99_interval_s，0 禁用；与 1s 窗口行互不影响）
        if (p99IntervalSecs_ > 0) {
            ++p99ElapsedSecs_;
            if (p99ElapsedSecs_ >= p99IntervalSecs_) {
                p99ElapsedSecs_ = 0;
                OutputLatencyP99();
            }
        }
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

MetricsSnapshot Scheduler::CollectMetricsSnapshot() const
{
    MetricsSnapshot snapshot;
    for (const auto &executor : executors_) {
        const ExecutorMetrics metrics = executor->GetMetrics();
        for (std::size_t op = 0; op < kOpTypeCount; ++op) {
            snapshot.opsByOp[op] += metrics.opCountByOp[op];
            snapshot.bytesByOp[op] += metrics.ioBytesByOp[op];
            snapshot.latencyNsByOp[op] += metrics.latencyNsTotalByOp[op];
            snapshot.tasksByOp[op] += metrics.taskCountByOp[op];
        }
        snapshot.errorsTotal += metrics.errorCount;
        snapshot.notFoundTotal += metrics.notFoundCount;
        snapshot.mismatchTotal += metrics.mismatchCount;
        executor->AppendLatencySamplesUs(snapshot.latencySamplesUs);
    }
    snapshot.uptimeSecs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime_).count();
    return snapshot;
}

void Scheduler::OutputLatencyP99() const
{
    const MetricsSnapshot snapshot = CollectMetricsSnapshot();
    std::vector<std::uint32_t> all;
    std::string parts[kOpTypeCount];
    for (std::size_t op = 0; op < kOpTypeCount; ++op) {
        // 样本副本参与排序（快照只读），总口径合并用副本内容
        std::vector<std::uint32_t> samples = snapshot.latencySamplesUs[op];
        std::uint64_t p99 = 0;
        if (!samples.empty()) { p99 = LatencyQuantileUs(samples, 0.99); }
        parts[op] =
            samples.empty() ? std::string("-") : FormatLatency(static_cast<double>(p99) * 1000.0);
        all.insert(all.end(), samples.begin(), samples.end());
    }
    std::uint64_t totalP99 = 0;
    if (!all.empty()) { totalP99 = LatencyQuantileUs(all, 0.99); }
    LOG_INFO("Scheduler latency P99 [{}s]: total={} exist={} put={} get={} delete={}",
             p99IntervalSecs_,
             all.empty() ? std::string("-") : FormatLatency(static_cast<double>(totalP99) * 1000.0),
             parts[static_cast<std::size_t>(OpType::QUERY)],
             parts[static_cast<std::size_t>(OpType::STORE)],
             parts[static_cast<std::size_t>(OpType::LOAD)],
             parts[static_cast<std::size_t>(OpType::DELETE)]);
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
