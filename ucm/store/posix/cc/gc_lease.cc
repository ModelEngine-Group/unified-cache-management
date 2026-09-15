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
#include "gc_lease.h"
#include <cerrno>
#include <climits>
#include <cstring>
#include <dirent.h>
#include <fmt/format.h>
#include <random>
#include <sys/stat.h>
#include <unistd.h>
#include <utime.h>
#include <vector>
#include "logger/logger.h"
#include "posix_file.h"
#include "thread/cpu_affinity.h"

namespace UC::PosixStore {

namespace {

constexpr const char* kLockDirName = ".ucm_gc.lock";
constexpr const char* kCheckTimeName = ".ucm_gc.checktime";
constexpr const char* kHeartbeatPrefix = "hb.";
constexpr const char* kStalePrefix = ".ucm_gc.stale.";
constexpr const char* kNoHolderMarker = "";

bool IsDotEntry(const char* name)
{
    return name[0] == '.' && (name[1] == '\0' || (name[1] == '.' && name[2] == '\0'));
}

Status RemoveDirTree(const std::string& path)
{
    DIR* dir = opendir(path.c_str());
    if (dir) {
        struct dirent* entry = nullptr;
        while ((entry = readdir(dir)) != nullptr) {
            if (IsDotEntry(entry->d_name)) { continue; }
            PosixFile{path + "/" + entry->d_name}.Remove();
        }
        closedir(dir);
    }
    return PosixFile{path}.RmDir();
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

}  // namespace

GcLease::~GcLease() { Release(); }

void GcLease::Setup(const Config& config, const SpaceLayout* layout)
{
    layout_ = layout;
    identity_ = fmt::format("{}{}.{}.{:08x}", kHeartbeatPrefix, LocalHostName(),
                            static_cast<long>(getpid()), Nonce());
    heartbeatIntervalSec_ = config.posixGcHeartbeatIntervalSec;
    staleThresholdSec_ = config.posixGcStaleThresholdSec;
}

Expected<GcLease::Paths> GcLease::SelectPaths() const
{
    auto backend = layout_->StorageBackend();
    if (!backend) { return backend.Error(); }
    const auto lockDir = backend.Value() + kLockDirName;
    return Paths{backend.Value(), lockDir, backend.Value() + kCheckTimeName,
                 lockDir + "/" + identity_};
}

Status GcLease::Touch(const std::string& path, time_t& stamp, bool create) const
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

Status GcLease::Claim(const Paths& paths)
{
    PosixFile dir{paths.lockDir};
    auto s = dir.MkDir();
    if (s.Failure()) { return s; }

    time_t ignored = 0;
    auto hb = Touch(paths.heartbeat, ignored, true);
    if (hb.Failure()) {
        UC_WARN("Failed({}) to write GC heartbeat({}); releasing lock.", hb, paths.heartbeat);
        dir.RmDir();
        return hb;
    }

    haveSuspect_ = false;
    held_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(stopMtx_);
        stopHeartbeat_ = false;
    }
    try {
        heartbeatWorker_ = std::thread(&GcLease::HeartbeatLoop, this);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to start GC heartbeat thread; releasing lock.", e.what());
        held_.store(false, std::memory_order_release);
        PosixFile{paths.heartbeat}.Remove();
        dir.RmDir();
        return Status::OutOfMemory();
    }
    UC_INFO("Acquired GC lock({}) as {}.", paths.lockDir, identity_);
    return Status::OK();
}

GcLease::Acquisition GcLease::TryAcquire()
{
    auto selected = SelectPaths();
    if (!selected) { return Acquisition::Unavailable; }
    const auto& paths = selected.Value();
    auto s = Claim(paths);
    if (s.Success()) { return Acquisition::Acquired; }
    if (s != Status::DuplicateKey()) { return Acquisition::Unavailable; }

    bool stale = false;
    if (ProbeHolder(paths, stale).Failure()) { return Acquisition::HeldByPeer; }
    if (!stale) { return Acquisition::HeldByPeer; }
    if (TakeOverStale(paths).Failure()) { return Acquisition::HeldByPeer; }

    s = Claim(paths);
    if (s.Success()) { return Acquisition::Acquired; }
    return s == Status::DuplicateKey() ? Acquisition::HeldByPeer : Acquisition::Unavailable;
}

Status GcLease::ProbeHolder(const Paths& paths, bool& stale)
{
    stale = false;
    std::string holder;
    DIR* dir = opendir(paths.lockDir.c_str());
    if (!dir) {
        auto eno = errno;
        if (eno == ENOENT) { return Status::OK(); }
        auto s = Status::OsApiError(std::to_string(eno));
        UC_WARN("Failed({}) to open GC lock dir({}).", s, paths.lockDir);
        return s;
    }
    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (strncmp(entry->d_name, kHeartbeatPrefix, strlen(kHeartbeatPrefix)) == 0) {
            holder = entry->d_name;
            break;
        }
    }
    closedir(dir);

    if (holder.empty()) {
        if (haveSuspect_ && suspectHeartbeat_ == kNoHolderMarker) {
            stale = true;
            UC_WARN("GC lock({}) has no heartbeat on two consecutive checks.", paths.lockDir);
        } else {
            haveSuspect_ = true;
            suspectHeartbeat_ = kNoHolderMarker;
            suspectMtime_ = 0;
            UC_INFO("GC lock({}) has no heartbeat; will confirm next check.", paths.lockDir);
        }
        return Status::OK();
    }

    struct stat st{};
    const auto holderPath = paths.lockDir + "/" + holder;
    if (stat(holderPath.c_str(), &st) != 0) {
        auto eno = errno;
        if (eno == ENOENT) { return Status::OK(); }
        return Status::OsApiError(std::to_string(eno));
    }

    time_t serverNow = 0;
    auto s = Touch(paths.checkTime, serverNow, true);
    if (s.Failure()) {
        UC_WARN("Failed({}) to stamp GC check time({}).", s, paths.checkTime);
        return s;
    }

    const auto lag = serverNow > st.st_mtime ? static_cast<size_t>(serverNow - st.st_mtime) : 0;
    if (lag < staleThresholdSec_) {
        haveSuspect_ = false;
        return Status::OK();
    }
    if (!haveSuspect_ || suspectHeartbeat_ != holder || suspectMtime_ != st.st_mtime) {
        haveSuspect_ = true;
        suspectHeartbeat_ = holder;
        suspectMtime_ = st.st_mtime;
        UC_INFO("GC lock({}) holder {} looks idle for {}s; will confirm next check.", paths.lockDir,
                holder, lag);
        return Status::OK();
    }
    stale = true;
    UC_WARN("GC lock({}) holder {} idle for {}s on two consecutive checks; taking over.",
            paths.lockDir, holder, lag);
    return Status::OK();
}

void GcLease::SweepParked(const Paths& paths) const
{
    DIR* dir = opendir(paths.backend.c_str());
    if (!dir) { return; }
    std::vector<std::string> parked;
    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (strncmp(entry->d_name, kStalePrefix, strlen(kStalePrefix)) == 0) {
            parked.emplace_back(entry->d_name);
        }
    }
    closedir(dir);
    for (const auto& name : parked) {
        const auto path = paths.backend + name;
        if (RemoveDirTree(path).Success()) { UC_INFO("Swept leaked parked GC lock({}).", path); }
    }
}

Status GcLease::TakeOverStale(const Paths& paths)
{
    haveSuspect_ = false;
    SweepParked(paths);
    const auto parked =
        fmt::format("{}{}{}.{:08x}", paths.backend, kStalePrefix, identity_, Nonce());
    auto s = PosixFile{paths.lockDir}.Rename(parked);
    if (s.Failure()) {
        UC_INFO("Failed({}) to claim stale GC lock({}); another instance won.", s, paths.lockDir);
        return s;
    }
    auto rm = RemoveDirTree(parked);
    if (rm.Failure()) { UC_WARN("Failed({}) to remove parked GC lock({}).", rm, parked); }
    return Status::OK();
}

bool GcLease::EntryPresent(const Paths& paths) const
{
    DIR* dir = opendir(paths.lockDir.c_str());
    if (!dir) { return false; }
    bool mine = false;
    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (identity_ == entry->d_name) {
            mine = true;
            break;
        }
    }
    closedir(dir);
    return mine;
}

bool GcLease::HoldsLock() const
{
    if (!held_.load(std::memory_order_acquire)) { return false; }
    auto paths = SelectPaths();
    return paths && EntryPresent(paths.Value());
}

void GcLease::RequestStop()
{
    {
        std::lock_guard<std::mutex> lock(stopMtx_);
        stopHeartbeat_ = true;
    }
    stopCv_.notify_all();
}

void GcLease::StopHeartbeat()
{
    RequestStop();
    if (heartbeatWorker_.joinable()) { heartbeatWorker_.join(); }
}

void GcLease::Release()
{
    if (!held_.exchange(false, std::memory_order_acq_rel)) { return; }
    StopHeartbeat();
    auto selected = SelectPaths();
    if (!selected) { return; }
    const auto& paths = selected.Value();
    if (!EntryPresent(paths)) {
        UC_WARN("GC lock({}) is no longer ours; leaving it to its current holder.", paths.lockDir);
        return;
    }
    PosixFile{paths.heartbeat}.Remove();
    auto s = PosixFile{paths.lockDir}.RmDir();
    if (s.Failure()) {
        UC_WARN("Failed({}) to remove GC lock dir({}) on release.", s, paths.lockDir);
        return;
    }
    UC_INFO("Released GC lock({}).", paths.lockDir);
}

void GcLease::HeartbeatLoop()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_posix_gclk");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM GC lease heartbeat thread name.", nameStatus);
    }
    std::unique_lock<std::mutex> lock(stopMtx_);
    const auto interval = std::chrono::seconds(heartbeatIntervalSec_);
    while (!stopCv_.wait_for(lock, interval, [this] { return stopHeartbeat_; })) {
        lock.unlock();
        auto selected = SelectPaths();
        if (!selected) {
            lock.lock();
            continue;
        }
        const auto& paths = selected.Value();
        time_t ignored = 0;
        auto s = Touch(paths.heartbeat, ignored, false);
        if (s == Status::NotFound()) {
            UC_WARN("GC heartbeat({}) is gone; the lock was taken over. Stopping heartbeat.",
                    paths.heartbeat);
            lock.lock();
            break;
        }
        if (s.Failure()) { UC_WARN("Failed({}) to refresh GC heartbeat({}).", s, paths.heartbeat); }
        lock.lock();
    }
}

}  // namespace UC::PosixStore
