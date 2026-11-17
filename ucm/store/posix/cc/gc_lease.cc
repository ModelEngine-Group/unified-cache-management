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
constexpr const char* kClaimPrefix = ".ucm_gc.claim.";
constexpr const char* kNoHolderMarker = "";

bool IsDotEntry(const char* name)
{
    return name[0] == '.' && (name[1] == '\0' || (name[1] == '.' && name[2] == '\0'));
}

bool HasPrefix(const std::string& name, const std::string& prefix)
{
    return name.compare(0, prefix.size(), prefix) == 0;
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

// Claims the GC lock. The heartbeat is written into a private dir first and the dir is then
// renamed to the lock dir, so a held lock dir always has its holder's heartbeat inside.
Status ClaimLock(const std::string& backend, const std::string& lockDir,
                 const std::string& heartbeatName)
{
    const auto claimDir = backend + kClaimPrefix + heartbeatName;
    auto s = PosixFile{claimDir}.MkDir();
    if (s.Success()) {
        time_t ignored = 0;
        s = GcClock::Touch(claimDir + "/" + heartbeatName, ignored, true);
        if (s.Success()) {
            if (rename(claimDir.c_str(), lockDir.c_str()) == 0) {
                s = Status::OK();
            } else if (errno == EEXIST || errno == ENOTEMPTY) {
                s = Status::DuplicateKey();
            } else {
                s = Status::OsApiError(std::to_string(errno));
            }
        }
        if (s.Failure()) { RemoveDirTree(claimDir); }
    }
    return s;
}

// Takes over a stale GC lock. Only the stale holder's heartbeat is removed, so if a new holder
// got in meanwhile, its heartbeat keeps the dir non-empty and the rmdir fails.
Status TakeOverLock(const std::string& lockDir, const std::string& holderName)
{
    auto s = Status::OK();
    if (!holderName.empty()) { s = PosixFile{lockDir + "/" + holderName}.Remove(); }
    if (s.Success()) {
        if (rmdir(lockDir.c_str()) == 0 || errno == ENOENT) {
            s = Status::OK();
        } else if (errno == EEXIST || errno == ENOTEMPTY) {
            s = Status::DuplicateKey();
        } else {
            s = Status::OsApiError(std::to_string(errno));
        }
    }
    return s;
}

Status ReleaseLock(const std::string& lockDir, const std::string& heartbeat)
{
    auto s = PosixFile{heartbeat}.Remove();
    if (s.Success()) { s = PosixFile{lockDir}.RmDir(); }
    return s;
}

Status FindEntry(const std::string& lockDir, const std::string& name,
                 const std::shared_ptr<bool>& found)
{
    DIR* dir = opendir(lockDir.c_str());
    if (!dir) { return errno == ENOENT ? Status::OK() : Status::OsApiError(std::to_string(errno)); }
    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (name == entry->d_name) {
            *found = true;
            break;
        }
    }
    closedir(dir);
    return Status::OK();
}

Status SweepClaims(const std::string& backend, const std::string& checkTime,
                   size_t staleThresholdSec)
{
    DIR* dir = opendir(backend.c_str());
    if (!dir) { return Status::OsApiError(std::to_string(errno)); }
    std::vector<std::string> claims;
    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (HasPrefix(entry->d_name, kClaimPrefix)) { claims.emplace_back(entry->d_name); }
    }
    closedir(dir);
    if (claims.empty()) { return Status::OK(); }

    time_t now = 0;
    auto s = GcClock::Touch(checkTime, now, true);
    if (s.Failure()) { return s; }
    for (const auto& name : claims) {
        struct stat st{};
        const auto path = backend + name;
        if (stat(path.c_str(), &st) == 0 && now > st.st_mtime &&
            static_cast<size_t>(now - st.st_mtime) >= staleThresholdSec &&
            RemoveDirTree(path).Success()) {
            UC_INFO("Swept leaked GC lock claim({}).", path);
        }
    }
    return Status::OK();
}

}  // namespace

GcLease::~GcLease() { Release(); }

void GcLease::Setup(const Config& config, const BackendManager* backendMgr)
{
    backendMgr_ = backendMgr;
    identity_ = fmt::format("{}{}.{}.{:08x}", kHeartbeatPrefix, GcClock::LocalHostName(),
                            static_cast<long>(getpid()), GcClock::Nonce());
    heartbeatIntervalSec_ = config.posixGcHeartbeatIntervalSec;
    staleThresholdSec_ = config.posixGcStaleThresholdSec;
    // Claim dirs are only left by an instance that died in the middle of a claim, so sweeping
    // them once at startup.
    const auto& backend = backendMgr_->Backends().front();
    auto s = SweepClaims(backend, backend + kCheckTimeName, staleThresholdSec_);
    if (s.Failure()) { UC_WARN("Failed({}) to sweep GC lock claims in({}).", s, backend); }
}

Status GcLease::ProbeLock(const std::string& lockDir, const std::string& checkTime,
                          const std::shared_ptr<Holder>& holder)
{
    DIR* dir = opendir(lockDir.c_str());
    if (!dir) {
        auto eno = errno;
        if (eno == ENOENT) { return Status::OK(); }
        auto s = Status::OsApiError(std::to_string(eno));
        UC_WARN("Failed({}) to open GC lock dir({}).", s, lockDir);
        return s;
    }
    holder->locked = true;
    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (strncmp(entry->d_name, kHeartbeatPrefix, strlen(kHeartbeatPrefix)) == 0) {
            holder->name = entry->d_name;
            break;
        }
    }
    closedir(dir);
    if (holder->name.empty()) { return Status::OK(); }

    struct stat st{};
    const auto holderPath = lockDir + "/" + holder->name;
    if (stat(holderPath.c_str(), &st) != 0) {
        auto eno = errno;
        if (eno == ENOENT) { return Status::OK(); }
        return Status::OsApiError(std::to_string(eno));
    }

    time_t serverNow = 0;
    auto s = GcClock::Touch(checkTime, serverNow, true);
    if (s.Failure()) {
        UC_WARN("Failed({}) to stamp GC check time({}).", s, checkTime);
        return s;
    }
    holder->mtime = st.st_mtime;
    holder->lag = serverNow > st.st_mtime ? static_cast<size_t>(serverNow - st.st_mtime) : 0;
    return Status::OK();
}

Expected<GcLease::Paths> GcLease::SelectPaths() const
{
    std::vector<std::string> attempted;
    auto attempt = backendMgr_->SelectNextBackend({}, attempted);
    if (!attempt) { return attempt.Error(); }
    const auto& backend = backendMgr_->Backends()[attempt.Value().index];
    return Paths{attempt.Value(), backend, backend + kLockDirName, backend + kCheckTimeName};
}

Status GcLease::Run(const Paths& paths, std::function<Status()> op) const
{
    auto s = backendMgr_->RunGcOp(std::move(op));
    backendMgr_->RecordResult(paths.attempt, s);
    return s;
}

Status GcLease::Claim(const Paths& paths)
{
    auto name = fmt::format("{}.{:08x}", identity_, GcClock::Nonce());
    auto s = Run(paths, [backend = paths.backend, lockDir = paths.lockDir, name] {
        return ClaimLock(backend, lockDir, name);
    });
    if (s.Failure()) { return s; }

    haveSuspect_ = false;
    heartbeatName_ = std::move(name);
    held_.store(true, std::memory_order_release);
    heartbeat_.Setup(
        backendMgr_,
        [name = heartbeatName_](const std::string& backend) {
            return backend + kLockDirName + "/" + name;
        },
        heartbeatIntervalSec_, GcHeartbeat::OnMissing::Stop);
    auto started = heartbeat_.Start();
    if (started.Failure()) {
        UC_ERROR("Failed({}) to start GC heartbeat thread; releasing lock.", started);
        held_.store(false, std::memory_order_release);
        Run(paths, [lockDir = paths.lockDir, heartbeat = paths.lockDir + "/" + heartbeatName_] {
            return ReleaseLock(lockDir, heartbeat);
        });
        return started;
    }
    UC_INFO("Acquired GC lock({}) as {}.", paths.lockDir, heartbeatName_);
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

    auto holder = std::make_shared<Holder>();
    s = Run(paths, [lockDir = paths.lockDir, checkTime = paths.checkTime, holder] {
        return ProbeLock(lockDir, checkTime, holder);
    });
    if (s.Failure() || !holder->locked) { return Acquisition::HeldByPeer; }
    if (!IsStale(paths, *holder)) { return Acquisition::HeldByPeer; }
    s = Run(paths,
            [lockDir = paths.lockDir, name = holder->name] { return TakeOverLock(lockDir, name); });
    if (s.Failure()) {
        UC_INFO("Failed({}) to claim stale GC lock({}); another instance won.", s, paths.lockDir);
        return Acquisition::HeldByPeer;
    }

    s = Claim(paths);
    if (s.Success()) { return Acquisition::Acquired; }
    return s == Status::DuplicateKey() ? Acquisition::HeldByPeer : Acquisition::Unavailable;
}

bool GcLease::IsStale(const Paths& paths, const Holder& holder)
{
    auto stale = false;
    if (holder.name.empty()) {
        if (haveSuspect_ && suspectHeartbeat_ == kNoHolderMarker) {
            stale = true;
            UC_WARN("GC lock({}) has no heartbeat on two consecutive checks.", paths.lockDir);
        } else {
            haveSuspect_ = true;
            suspectHeartbeat_ = kNoHolderMarker;
            suspectMtime_ = 0;
            UC_INFO("GC lock({}) has no heartbeat; will confirm next check.", paths.lockDir);
        }
        return stale;
    }

    if (holder.lag < staleThresholdSec_) {
        haveSuspect_ = false;
        return stale;
    }
    if (!haveSuspect_ || suspectHeartbeat_ != holder.name || suspectMtime_ != holder.mtime) {
        haveSuspect_ = true;
        suspectHeartbeat_ = holder.name;
        suspectMtime_ = holder.mtime;
        UC_INFO("GC lock({}) holder {} looks idle for {}s; will confirm next check.", paths.lockDir,
                holder.name, holder.lag);
        return stale;
    }
    stale = true;
    UC_WARN("GC lock({}) holder {} idle for {}s on two consecutive checks; taking over.",
            paths.lockDir, holder.name, holder.lag);
    return stale;
}

bool GcLease::EntryPresent(const Paths& paths) const
{
    auto found = std::make_shared<bool>(false);
    auto s = Run(paths, [lockDir = paths.lockDir, name = heartbeatName_, found] {
        return FindEntry(lockDir, name, found);
    });
    return s.Success() && *found;
}

bool GcLease::HoldsLock() const
{
    if (!held_.load(std::memory_order_acquire)) { return false; }
    auto paths = SelectPaths();
    return paths && EntryPresent(paths.Value());
}

void GcLease::RequestStop() { heartbeat_.RequestStop(); }

void GcLease::StopHeartbeat() { heartbeat_.Stop(); }

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
    auto s =
        Run(paths, [lockDir = paths.lockDir, heartbeat = paths.lockDir + "/" + heartbeatName_] {
            return ReleaseLock(lockDir, heartbeat);
        });
    if (s.Failure()) {
        UC_WARN("Failed({}) to remove GC lock dir({}) on release.", s, paths.lockDir);
        return;
    }
    UC_INFO("Released GC lock({}).", paths.lockDir);
}

}  // namespace UC::PosixStore
