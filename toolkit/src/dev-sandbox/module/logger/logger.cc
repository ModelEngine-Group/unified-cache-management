/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#include "logger.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <libgen.h>
#include <list>
#include <mutex>
#include <pthread.h>
#include <shared_mutex>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <vector>

static std::shared_mutex gLoggerRegistryMtx;
static std::unordered_set<LoggerImpl*> gLoggerRegistry;

static void PrepareFork();
static void ParentFork();
static void ChildFork();

static void RegisterLoggerImpl(LoggerImpl* impl)
{
    {
        std::unique_lock lock(gLoggerRegistryMtx);
        gLoggerRegistry.insert(impl);
    }
    static std::once_flag afForkOnce;
    std::call_once(afForkOnce, [] { pthread_atfork(PrepareFork, ParentFork, ChildFork); });
}

static void UnregisterLoggerImpl(LoggerImpl* impl)
{
    std::unique_lock lock(gLoggerRegistryMtx);
    gLoggerRegistry.erase(impl);
}

class LoggerImpl {
    using Buffer = std::list<std::string>;
    using SteadyClock = std::chrono::steady_clock;
    using SystemClock = std::chrono::system_clock;
    std::atomic<bool> stop_{false};
    SteadyClock::time_point lastFlush_{SteadyClock::now()};
    Buffer frontBuf_;
    Buffer backBuf_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread worker_;
    static constexpr size_t FlushBatchSize = 1024;
    static constexpr auto FlushLatency = std::chrono::milliseconds(10);

    void WorkerLoop()
    {
        Buffer localBuf;
        while (true) {
            {
                std::unique_lock ul(mtx_);
                auto triggered =
                    cv_.wait_for(ul, FlushLatency, [this] { return stop_ || !backBuf_.empty(); });
                if (stop_) { break; }
                localBuf.splice(localBuf.end(), backBuf_);
                if (!triggered) { localBuf.splice(localBuf.end(), frontBuf_); }
            }
            if (localBuf.empty()) { continue; }
            Flush(localBuf);
        }
        while (!backBuf_.empty()) {
            localBuf.splice(localBuf.end(), backBuf_);
            Flush(localBuf);
        }
    }
    void Flush(Buffer& buf)
    {
        if (!buf.empty()) {
            std::string batch;
            batch.reserve(4096);
            for (const auto& s : buf) { batch += s; }
            std::fwrite(batch.data(), 1, batch.size(), stdout);
        }
        std::fflush(stdout);
        buf.clear();
    }

public:
    LoggerImpl() { RegisterLoggerImpl(this); }
    ~LoggerImpl()
    {
        UnregisterLoggerImpl(this);
        StopAndJoinWorker();
    }
    void StopAndJoinWorker()
    {
        {
            std::lock_guard lg(mtx_);
            stop_ = true;
            backBuf_.splice(backBuf_.end(), frontBuf_);
            cv_.notify_one();
        }
        if (worker_.joinable()) { worker_.join(); }
    }
    void ResetAfterForkInChild()
    {
        std::lock_guard lg(mtx_);
        stop_ = false;
        frontBuf_.clear();
        backBuf_.clear();
        lastFlush_ = SteadyClock::now();
    }
    void EnsureWorkerRunning()
    {
        std::lock_guard lg(mtx_);
        if (!worker_.joinable() && !stop_) {
            lastFlush_ = SteadyClock::now();
            worker_ = std::thread([this] { WorkerLoop(); });
        }
    }

    void Log(Logger::Level lv, const Logger::SourceLocation& loc, const std::string& message)
    {
        EnsureWorkerRunning();
        static const char* lvStrs[] = {"D", "I", "W", "E", "C"};
        static const size_t pid = static_cast<size_t>(getpid());
        static thread_local const size_t tid = syscall(SYS_gettid);
        static thread_local std::chrono::seconds lastSec{0};
        static thread_local char datetime[32];
        auto systemNow = SystemClock::now();
        auto currentSec = std::chrono::time_point_cast<std::chrono::seconds>(systemNow);
        if (lastSec != currentSec.time_since_epoch()) {
            auto systemTime = SystemClock::to_time_t(systemNow);
            std::tm systemTm;
            localtime_r(&systemTime, &systemTm);
            fmt::format_to_n(datetime, sizeof(datetime), "{:%F %T}", systemTm);
            lastSec = currentSec.time_since_epoch();
        }
        auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(systemNow - currentSec).count();
        auto payload = fmt::format("[{}.{:06d}][{}] {} [{},{}][{},{}:{}]\n", datetime, us,
                                   lvStrs[fmt::underlying(lv)], message, pid, tid, loc.func,
                                   basename(const_cast<char*>(loc.file)), loc.line);
        auto steadyNow = SteadyClock::now();
        std::lock_guard lg(mtx_);
        frontBuf_.push_back(std::move(payload));
        bool byCount = frontBuf_.size() >= FlushBatchSize;
        bool byTime = steadyNow - lastFlush_ >= FlushLatency;
        if (byCount || byTime) {
            backBuf_.splice(backBuf_.end(), frontBuf_);
            lastFlush_ = steadyNow;
            cv_.notify_one();
        }
    }
};

static void ForEachRegisteredSnapshot(const std::function<void(LoggerImpl*)>& fn)
{
    std::vector<LoggerImpl*> snapshot;
    {
        std::shared_lock lock(gLoggerRegistryMtx);
        snapshot.reserve(gLoggerRegistry.size());
        for (auto p : gLoggerRegistry) snapshot.push_back(p);
    }
    for (auto p : snapshot) {
        if (p) { fn(p); }
    }
}

static void PrepareFork()
{
    ForEachRegisteredSnapshot([](LoggerImpl* impl) { impl->StopAndJoinWorker(); });
}
static void ParentFork()
{
    // Intentionally do nothing: worker threads are lazily started on first log
    // in both parent and child. This keeps behavior consistent and avoids
    // creating threads inside atfork handlers.
}
static void ChildFork()
{
    ForEachRegisteredSnapshot([](LoggerImpl* impl) { impl->ResetAfterForkInChild(); });
}

Logger::Logger() : impl_(std::make_unique<LoggerImpl>()) {}

Logger::~Logger() = default;

void Logger::LogInternal(Level lv, const SourceLocation& loc, const std::string& message)
{
    impl_->Log(lv, loc, message);
}
