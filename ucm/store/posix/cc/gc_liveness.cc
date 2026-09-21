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
#include "gc_liveness.h"
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstring>
#include <dirent.h>
#include <random>
#include <sys/stat.h>
#include <unistd.h>
#include <utime.h>
#include "logger/logger.h"
#include "posix_file.h"
#include "thread/cpu_affinity.h"

namespace UC::PosixStore {

namespace GcClock {

Status Touch(const std::string& path, time_t& stamp, bool create)
{
    if (utime(path.c_str(), nullptr) != 0) {
        auto eno = errno;
        if (eno != ENOENT) { return Status::OsApiError(std::to_string(eno)); }
        if (!create) { return Status::NotFound(); }
        PosixFile file{path};
        auto s = file.Open(PosixFile::OpenFlag::CREATE | PosixFile::OpenFlag::WRITE_ONLY);
        if (s.Failure()) { return s; }
        file.Close();
        if (utime(path.c_str(), nullptr) != 0) { return Status::OsApiError(std::to_string(errno)); }
    }
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) { return Status::OsApiError(std::to_string(errno)); }
    stamp = st.st_mtime;
    return Status::OK();
}

std::string LocalHostName()
{
    char buffer[HOST_NAME_MAX + 1] = {};
    if (gethostname(buffer, sizeof(buffer) - 1) != 0) { return "unknown"; }
    std::string name{buffer};
    for (auto& c : name) {
        if (c == '/' || c == '.') { c = '_'; }
    }
    return name.empty() ? "unknown" : name;
}

uint32_t Nonce()
{
    std::random_device rd;
    return std::uniform_int_distribution<uint32_t>{}(rd);
}

}  // namespace GcClock

Expected<std::vector<GcMemberInfo>> ScanMembers(const std::string& dir, const std::string& prefix,
                                                const std::string& checkTimePath)
{
    std::vector<GcMemberInfo> members;
    DIR* handle = opendir(dir.c_str());
    if (!handle) {
        auto eno = errno;
        if (eno == ENOENT) { return members; }
        auto s = Status::OsApiError(std::to_string(eno));
        UC_WARN("Failed({}) to open GC member dir({}).", s, dir);
        return s;
    }

    std::vector<std::string> names;
    struct dirent* entry = nullptr;
    while ((entry = readdir(handle)) != nullptr) {
        if (strncmp(entry->d_name, prefix.c_str(), prefix.size()) == 0) {
            names.emplace_back(entry->d_name);
        }
    }
    closedir(handle);
    if (names.empty()) { return members; }

    time_t now = 0;
    auto s = GcClock::Touch(checkTimePath, now, true);
    if (s.Failure()) {
        UC_WARN("Failed({}) to stamp GC check time({}).", s, checkTimePath);
        return s;
    }

    members.reserve(names.size());
    for (const auto& name : names) {
        struct stat st{};
        if (stat((dir + "/" + name).c_str(), &st) != 0) { continue; }
        GcMemberInfo info;
        info.name = name;
        info.lag = now > st.st_mtime ? static_cast<size_t>(now - st.st_mtime) : 0;
        members.push_back(std::move(info));
    }
    return members;
}

GcHeartbeat::~GcHeartbeat() { Stop(); }

void GcHeartbeat::Setup(std::string path, std::string threadName, size_t intervalSec,
                        OnMissing onMissing)
{
    path_ = std::move(path);
    threadName_ = std::move(threadName);
    intervalSec_ = intervalSec == 0 ? 1 : intervalSec;
    onMissing_ = onMissing;
}

Status GcHeartbeat::Start()
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = false;
    }
    try {
        worker_ = std::thread(&GcHeartbeat::Loop, this);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to start GC heartbeat thread for {}.", e.what(), path_);
        return Status::OutOfMemory();
    }
    return Status::OK();
}

void GcHeartbeat::RequestStop()
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
}

void GcHeartbeat::Stop()
{
    RequestStop();
    if (worker_.joinable()) { worker_.join(); }
}

void GcHeartbeat::Loop()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName(threadName_.c_str());
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set GC heartbeat thread name({}).", nameStatus, threadName_);
    }
    std::unique_lock<std::mutex> lock(mtx_);
    const auto interval = std::chrono::seconds(intervalSec_);
    const bool create = onMissing_ == OnMissing::Recreate;
    while (!cv_.wait_for(lock, interval, [this] { return stop_; })) {
        lock.unlock();
        time_t ignored = 0;
        auto s = GcClock::Touch(path_, ignored, create);
        if (s == Status::NotFound() && !create) {
            UC_WARN("GC heartbeat({}) is gone; the lock was taken over. Stopping heartbeat.",
                    path_);
            lock.lock();
            break;
        }
        if (s.Failure()) { UC_WARN("Failed({}) to refresh GC heartbeat({}).", s, path_); }
        lock.lock();
    }
}

}  // namespace UC::PosixStore
