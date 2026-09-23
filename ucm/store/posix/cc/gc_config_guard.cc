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
#include "gc_config_guard.h"
#include <algorithm>
#include <cerrno>
#include <fmt/ranges.h>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include "gc_liveness.h"
#include "logger/logger.h"
#include "posix_file.h"

namespace UC::PosixStore {

namespace {

constexpr const char* kConfigName = ".ucm_gc.config";
constexpr const char* kGateDirName = ".ucm_gc.cfglock";
constexpr const char* kMembersDirName = ".ucm_gc.members";
constexpr const char* kCheckTimeName = ".ucm_gc.cfgchecktime";
constexpr const char* kMemberPrefix = "mb.";
constexpr const char* kConfigVersion = "1";
constexpr const char* kUnsetMarker = "<unset>";
constexpr size_t kMaxConfigBytes = 64 * 1024;
constexpr size_t kGateRetries = 60;
constexpr size_t kGateRetryDelayMs = 100;

std::string Number(double value) { return fmt::format("{:.6f}", value); }

std::string Boolean(bool value) { return value ? "true" : "false"; }

std::string Trim(const std::string& text)
{
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return {}; }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::string NormalizeBackend(const std::string& path)
{
    if (path.empty() || path.back() == '/') { return path; }
    return path + '/';
}

}  // namespace

GcConfigGuard::~GcConfigGuard()
{
    heartbeat_.Stop();
    if (registered_.exchange(false)) { UnregisterMember(); }
    ReleaseGate();
}

std::vector<std::pair<std::string, std::string>> GcConfigGuard::Entries(const Config& config)
{
    std::vector<std::string> backends;
    backends.reserve(config.storageBackends.size());
    for (const auto& backend : config.storageBackends) {
        backends.emplace_back(NormalizeBackend(backend));
    }

    std::vector<std::pair<std::string, std::string>> entries{
        {"version", kConfigVersion},
        {"storage_backends", fmt::format("{}", fmt::join(backends, ":"))},
        {"data_dir_shard_bytes", std::to_string(config.dataDirShardBytes)},
        {"gc_enable", Boolean(config.posixGcEnable && config.posixCapacityGb > 0)},
        {"block_size", std::to_string(config.blockSize)},
        {"capacity_gb", std::to_string(config.posixCapacityGb)},
        {"recycle_percent", Number(config.posixGcRecyclePercent)},
        {"precise_mode", Boolean(config.posixGcPreciseMode)},
        {"check_interval_sec", std::to_string(config.posixGcCheckIntervalSec)},
        {"trigger_threshold_ratio", Number(config.posixGcTriggerThresholdRatio)},
        {"max_recycle_count_per_shard", std::to_string(config.posixGcMaxRecycleCountPerShard)},
        {"shard_sample_ratio", Number(config.posixGcShardSampleRatio)},
        {"cross_instance_lock", Boolean(config.posixGcCrossInstanceLock)},
    };
    if (config.posixGcPreciseMode) {
        entries.emplace_back("candidate_extra_percent",
                             Number(config.posixGcCandidateExtraPercent));
    }
    entries.emplace_back("heartbeat_interval_sec",
                         std::to_string(config.posixGcHeartbeatIntervalSec));
    entries.emplace_back("stale_threshold_sec", std::to_string(config.posixGcStaleThresholdSec));
    std::sort(entries.begin(), entries.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    return entries;
}

std::string GcConfigGuard::Serialize(const Config& config)
{
    std::string payload =
        "# UCM Posix GC config, published by the first live instance on this backend.\n"
        "# Every instance sharing the backend must use the same GC config.\n"
        "# Reclaimed automatically once no instance holds a heartbeat.\n";
    for (const auto& [key, value] : Entries(config)) {
        payload += fmt::format("{}={}\n", key, value);
    }
    return payload;
}

std::map<std::string, std::string> GcConfigGuard::Parse(const std::string& content)
{
    std::map<std::string, std::string> entries;
    size_t begin = 0;
    while (begin < content.size()) {
        auto end = content.find('\n', begin);
        if (end == std::string::npos) { end = content.size(); }
        const auto line = Trim(content.substr(begin, end - begin));
        begin = end + 1;
        if (line.empty() || line.front() == '#') { continue; }
        const auto sep = line.find('=');
        if (sep == std::string::npos) { continue; }
        entries[Trim(line.substr(0, sep))] = Trim(line.substr(sep + 1));
    }
    return entries;
}

std::vector<std::string> GcConfigGuard::Diff(const std::string& mine, const std::string& published)
{
    const auto mineEntries = Parse(mine);
    const auto publishedEntries = Parse(published);

    std::set<std::string> keys;
    for (const auto& [key, value] : mineEntries) { keys.insert(key); }
    for (const auto& [key, value] : publishedEntries) { keys.insert(key); }

    std::vector<std::string> diffs;
    for (const auto& key : keys) {
        const auto mineIter = mineEntries.find(key);
        const auto publishedIter = publishedEntries.find(key);
        const auto mineValue =
            mineIter == mineEntries.end() ? std::string{kUnsetMarker} : mineIter->second;
        const auto publishedValue = publishedIter == publishedEntries.end()
                                        ? std::string{kUnsetMarker}
                                        : publishedIter->second;
        if (mineValue != publishedValue) {
            diffs.emplace_back(
                fmt::format("{}(this={}, first={})", key, mineValue, publishedValue));
        }
    }
    return diffs;
}

Status GcConfigGuard::AcquireGate()
{
    PosixFile dir{gateDir_};
    for (size_t attempt = 0; attempt < kGateRetries; attempt++) {
        auto s = dir.MkDir();
        if (s.Success()) {
            gateHeld_ = true;
            return Status::OK();
        }
        if (s != Status::DuplicateKey()) { return s; }

        struct stat st{};
        if (stat(gateDir_.c_str(), &st) == 0) {
            time_t now = 0;
            auto ts = GcClock::Touch(checkTimePath_, now, true);
            if (ts.Success() && now > st.st_mtime &&
                static_cast<size_t>(now - st.st_mtime) >= staleThresholdSec_) {
                UC_WARN("GC config gate({}) looks abandoned for {}s; removing it.", gateDir_,
                        static_cast<size_t>(now - st.st_mtime));
                PosixFile{gateDir_}.RmDir();
                continue;
            }
        }
        usleep(kGateRetryDelayMs * 1000);
    }
    return Status::Timeout();
}

void GcConfigGuard::ReleaseGate()
{
    if (!gateHeld_) { return; }
    gateHeld_ = false;
    auto s = PosixFile{gateDir_}.RmDir();
    if (s.Failure()) { UC_WARN("Failed({}) to release GC config gate({}).", s, gateDir_); }
}

Expected<size_t> GcConfigGuard::CountLiveMembers() const
{
    auto scan = ScanMembers(membersDir_, kMemberPrefix, checkTimePath_);
    if (!scan) { return scan.Error(); }

    size_t live = 0;
    for (const auto& member : scan.Value()) {
        if (member.lag < staleThresholdSec_) {
            ++live;
            continue;
        }
        UC_WARN("GC config member({}) has not refreshed for {}s; treating it as dead.", member.name,
                member.lag);
        PosixFile{membersDir_ + "/" + member.name}.Remove();
    }
    return live;
}

Status GcConfigGuard::RegisterMember()
{
    PosixFile dir{membersDir_};
    auto s = dir.MkDir();
    if (s == Status::DuplicateKey()) { s = Status::OK(); }
    if (s.Failure()) { return s; }

    time_t ignored = 0;
    s = GcClock::Touch(memberPath_, ignored, true);
    if (s.Failure()) { return s; }
    registered_.store(true);
    return Status::OK();
}

void GcConfigGuard::UnregisterMember() const
{
    PosixFile{memberPath_}.Remove();
    UC_INFO("Left the GC config membership of backend({}) as {}.", backend_, identity_);
}

Status GcConfigGuard::Publish(const std::string& payload) const
{
    const auto tmpPath =
        fmt::format("{}{}.tmp.{}.{}.{:08x}", backend_, kConfigName, GcClock::LocalHostName(),
                    static_cast<long>(getpid()), GcClock::Nonce());
    PosixFile tmp{tmpPath};
    auto s = tmp.Open(PosixFile::OpenFlag::CREATE | PosixFile::OpenFlag::EXCL |
                      PosixFile::OpenFlag::WRITE_ONLY);
    if (s.Failure()) { return s; }
    s = tmp.Write(payload.data(), payload.size(), 0);
    if (s.Success()) { s = tmp.Sync(); }
    tmp.Close();
    if (s.Failure()) {
        PosixFile{tmpPath}.Remove();
        return s;
    }
    s = tmp.Rename(configPath_);
    if (s.Failure()) {
        PosixFile{tmpPath}.Remove();
        return s;
    }
    return Status::OK();
}

Expected<std::string> GcConfigGuard::ReadPublished() const
{
    struct stat st{};
    if (stat(configPath_.c_str(), &st) != 0) {
        auto eno = errno;
        if (eno == ENOENT) { return Status::NotFound(); }
        return Status::OsApiError(std::to_string(eno));
    }
    if (st.st_size <= 0 || static_cast<size_t>(st.st_size) > kMaxConfigBytes) {
        return Status::InvalidParam("published GC config({}) has an unexpected size({} bytes)",
                                    configPath_, static_cast<long long>(st.st_size));
    }

    PosixFile file{configPath_};
    auto s = file.Open(PosixFile::OpenFlag::READ_ONLY);
    if (s.Failure()) { return s; }
    std::string content(static_cast<size_t>(st.st_size), '\0');
    s = file.Read(content.data(), content.size(), 0);
    file.Close();
    if (s.Failure()) { return s; }
    return content;
}

Status GcConfigGuard::Compare(const std::string& payload, const std::string& published) const
{
    const auto diffs = Diff(payload, published);
    if (diffs.empty()) { return Status::OK(); }
    return Status::InvalidParam(
        "GC config conflicts with the live instances on backend({}): {}. Every instance "
        "sharing a backend must use the same GC config; align this instance with the published "
        "config({}), or restart it after all peers have exited",
        backend_, fmt::join(diffs, ", "), configPath_);
}

Status GcConfigGuard::Verify(const std::string& payload) const
{
    auto published = ReadPublished();
    if (!published) { return published.Error(); }
    return Compare(payload, published.Value());
}

Status GcConfigGuard::Join(const std::string& payload)
{
    auto live = CountLiveMembers();
    if (!live) { return live.Error(); }

    if (live.Value() == 0) {
        auto s = Publish(payload);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to publish GC config({}).", s, configPath_);
            return s;
        }
        UC_INFO("Published GC config({}) for backend({}); this is the first live instance.",
                configPath_, backend_);
    } else {
        auto s = Verify(payload);
        if (s == Status::NotFound()) {
            return Status::InvalidParam(
                "GC config({}) is missing while {} instance(s) are live on backend({}). It was "
                "likely removed manually; the GC config of the live instances can no longer be "
                "verified. Restart this instance after all peers have exited",
                configPath_, live.Value(), backend_);
        }
        if (s.Failure()) { return s; }
        UC_INFO("GC config matches the {} live instance(s) on backend({}).", live.Value(),
                backend_);
    }

    auto s = RegisterMember();
    if (s.Failure()) {
        UC_ERROR("Failed({}) to register GC config membership({}).", s, memberPath_);
        return s;
    }
    return Status::OK();
}

Status GcConfigGuard::Setup(const Config& config)
{
    if (config.storageBackends.empty()) { return Status::InvalidParam("invalid storage backends"); }
    backend_ = NormalizeBackend(config.storageBackends.front());
    configPath_ = backend_ + kConfigName;
    gateDir_ = backend_ + kGateDirName;
    membersDir_ = backend_ + kMembersDirName;
    checkTimePath_ = backend_ + kCheckTimeName;
    identity_ = fmt::format("{}{}.{}.{:08x}", kMemberPrefix, GcClock::LocalHostName(),
                            static_cast<long>(getpid()), GcClock::Nonce());
    memberPath_ = membersDir_ + "/" + identity_;
    heartbeatIntervalSec_ = config.posixGcHeartbeatIntervalSec;
    staleThresholdSec_ = config.posixGcStaleThresholdSec;

    auto s = AcquireGate();
    if (s.Failure()) {
        UC_ERROR("Failed({}) to acquire GC config gate({}).", s, gateDir_);
        return s;
    }
    s = Join(Serialize(config));
    ReleaseGate();
    if (s.Failure()) { return s; }

    heartbeat_.Setup(memberPath_, "ucm_posix_gccfg", heartbeatIntervalSec_,
                     GcHeartbeat::OnMissing::Recreate);
    s = heartbeat_.Start();
    if (s.Failure()) {
        if (registered_.exchange(false)) { UnregisterMember(); }
        return s;
    }
    UC_INFO("Joined the GC config membership of backend({}) as {}.", backend_, identity_);
    return Status::OK();
}

}  // namespace UC::PosixStore
