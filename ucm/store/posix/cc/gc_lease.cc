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
#include <cstring>
#include <dirent.h>
#include <fmt/format.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "gc_liveness.h"
#include "logger/logger.h"
#include "posix_file.h"

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

}  // namespace

GcLease::~GcLease() { Release(); }

void GcLease::Setup(const Config& config)
{
    backend_ = config.storageBackends.front();
    if (backend_.back() != '/') { backend_ += '/'; }
    lockDir_ = backend_ + kLockDirName;
    checkTimePath_ = backend_ + kCheckTimeName;
    identity_ = fmt::format("{}{}.{}.{:08x}", kHeartbeatPrefix, GcClock::LocalHostName(),
                            static_cast<long>(getpid()), GcClock::Nonce());
    heartbeatPath_ = lockDir_ + "/" + identity_;
    heartbeatIntervalSec_ = config.posixGcHeartbeatIntervalSec;
    staleThresholdSec_ = config.posixGcStaleThresholdSec;
}

Status GcLease::Claim()
{
    PosixFile dir{lockDir_};
    auto s = dir.MkDir();
    if (s.Failure()) { return s; }

    time_t ignored = 0;
    auto hb = GcClock::Touch(heartbeatPath_, ignored, true);
    if (hb.Failure()) {
        UC_WARN("Failed({}) to write GC heartbeat({}); releasing lock.", hb, heartbeatPath_);
        dir.RmDir();
        return hb;
    }

    haveSuspect_ = false;
    held_.store(true, std::memory_order_release);
    heartbeat_.Setup(heartbeatPath_, "ucm_posix_gclk", heartbeatIntervalSec_,
                     GcHeartbeat::OnMissing::Stop);
    auto started = heartbeat_.Start();
    if (started.Failure()) {
        UC_ERROR("Failed({}) to start GC heartbeat thread; releasing lock.", started);
        held_.store(false, std::memory_order_release);
        PosixFile{heartbeatPath_}.Remove();
        dir.RmDir();
        return started;
    }
    UC_INFO("Acquired GC lock({}) as {}.", lockDir_, identity_);
    return Status::OK();
}

GcLease::Acquisition GcLease::TryAcquire()
{
    auto s = Claim();
    if (s.Success()) { return Acquisition::Acquired; }
    if (s != Status::DuplicateKey()) { return Acquisition::Unavailable; }

    bool stale = false;
    if (ProbeHolder(stale).Failure()) { return Acquisition::HeldByPeer; }
    if (!stale) { return Acquisition::HeldByPeer; }
    if (TakeOverStale().Failure()) { return Acquisition::HeldByPeer; }

    s = Claim();
    if (s.Success()) { return Acquisition::Acquired; }
    return s == Status::DuplicateKey() ? Acquisition::HeldByPeer : Acquisition::Unavailable;
}

Status GcLease::ProbeHolder(bool& stale)
{
    stale = false;
    std::string holder;
    DIR* dir = opendir(lockDir_.c_str());
    if (!dir) {
        auto eno = errno;
        if (eno == ENOENT) { return Status::OK(); }
        auto s = Status::OsApiError(std::to_string(eno));
        UC_WARN("Failed({}) to open GC lock dir({}).", s, lockDir_);
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
            UC_WARN("GC lock({}) has no heartbeat on two consecutive checks.", lockDir_);
        } else {
            haveSuspect_ = true;
            suspectHeartbeat_ = kNoHolderMarker;
            suspectMtime_ = 0;
            UC_INFO("GC lock({}) has no heartbeat; will confirm next check.", lockDir_);
        }
        return Status::OK();
    }

    struct stat st{};
    const auto holderPath = lockDir_ + "/" + holder;
    if (stat(holderPath.c_str(), &st) != 0) {
        auto eno = errno;
        if (eno == ENOENT) { return Status::OK(); }
        return Status::OsApiError(std::to_string(eno));
    }

    time_t serverNow = 0;
    auto s = GcClock::Touch(checkTimePath_, serverNow, true);
    if (s.Failure()) {
        UC_WARN("Failed({}) to stamp GC check time({}).", s, checkTimePath_);
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
        UC_INFO("GC lock({}) holder {} looks idle for {}s; will confirm next check.", lockDir_,
                holder, lag);
        return Status::OK();
    }
    stale = true;
    UC_WARN("GC lock({}) holder {} idle for {}s on two consecutive checks; taking over.", lockDir_,
            holder, lag);
    return Status::OK();
}

void GcLease::SweepParked() const
{
    DIR* dir = opendir(backend_.c_str());
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
        const auto path = backend_ + name;
        if (RemoveDirTree(path).Success()) { UC_INFO("Swept leaked parked GC lock({}).", path); }
    }
}

Status GcLease::TakeOverStale()
{
    haveSuspect_ = false;
    SweepParked();
    const auto parked =
        fmt::format("{}{}{}.{:08x}", backend_, kStalePrefix, identity_, GcClock::Nonce());
    auto s = PosixFile{lockDir_}.Rename(parked);
    if (s.Failure()) {
        UC_INFO("Failed({}) to claim stale GC lock({}); another instance won.", s, lockDir_);
        return s;
    }
    auto rm = RemoveDirTree(parked);
    if (rm.Failure()) { UC_WARN("Failed({}) to remove parked GC lock({}).", rm, parked); }
    return Status::OK();
}

bool GcLease::EntryPresent() const
{
    DIR* dir = opendir(lockDir_.c_str());
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
    return EntryPresent();
}

void GcLease::RequestStop() { heartbeat_.RequestStop(); }

void GcLease::StopHeartbeat() { heartbeat_.Stop(); }

void GcLease::Release()
{
    if (!held_.exchange(false, std::memory_order_acq_rel)) { return; }
    StopHeartbeat();
    if (!EntryPresent()) {
        UC_WARN("GC lock({}) is no longer ours; leaving it to its current holder.", lockDir_);
        return;
    }
    PosixFile{heartbeatPath_}.Remove();
    auto s = PosixFile{lockDir_}.RmDir();
    if (s.Failure()) {
        UC_WARN("Failed({}) to remove GC lock dir({}) on release.", s, lockDir_);
        return;
    }
    UC_INFO("Released GC lock({}).", lockDir_);
}

}  // namespace UC::PosixStore
