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
#include "space_layout.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <dirent.h>
#include <fmt/ranges.h>
#include <random>
#include <sys/stat.h>
#include <unistd.h>
#include "detail/health_check_executor.h"
#include "logger/logger.h"
#include "posix_file.h"
#include "template/topn_heap.h"
#include "thread/cpu_affinity.h"
#include "type/random_block_id.h"

namespace UC::PosixStore {

static const std::string DATA_ROOT = "data";
static const std::string ACTIVATED_FILE_EXTENSION = ".tmp";

struct MtimeComparator {
    bool operator()(const FileInfo& lhs, const FileInfo& rhs) const
    {
        return lhs.mtime > rhs.mtime;
    }
};

inline std::string DataFileName(const Detail::BlockId& blockId)
{
    return fmt::format("{:02x}", fmt::join(blockId, ""));
}

std::vector<std::string> GenerateHexStrings(const size_t n)
{
    if (n == 0) [[unlikely]] { return {}; }
    size_t nCombinations = 1ULL << (n * 4);
    std::vector<std::string> result;
    result.reserve(nCombinations);
    constexpr char hexChars[] = "0123456789abcdef";
    for (size_t i = 0; i < nCombinations; ++i) {
        std::string s(n, '0');
        auto temp = i;
        for (int j = n - 1; j >= 0; --j) {
            s[j] = hexChars[temp & 0xF];
            temp >>= 4;
        }
        result.push_back(s);
    }
    return result;
}

static Status CheckPathHealth(const std::string& path, bool ioDirect)
{
    constexpr size_t kHealthIoSize = 4096;
    alignas(kHealthIoSize) std::array<uint8_t, kHealthIoSize> expected{};
    alignas(kHealthIoSize) std::array<uint8_t, kHealthIoSize> actual{};
    expected.fill(0x5a);

    PosixFile file{path};
    auto flags = PosixFile::OpenFlag::CREATE | PosixFile::OpenFlag::READ_WRITE;
    if (ioDirect) { flags |= PosixFile::OpenFlag::DIRECT; }
    auto status = file.Open(flags);
    if (status.Failure()) { return status; }
    status = file.Write(expected.data(), expected.size(), 0);
    if (status.Success() && !ioDirect) { status = file.Sync(); }
    if (status.Success()) { status = file.Read(actual.data(), actual.size(), 0); }
    file.Close();
    auto cleanup = file.Remove();
    if (status.Success() && actual != expected) { status = Status::Error("health data mismatch"); }
    return status.Failure() ? status : cleanup;
}

SpaceLayout::~SpaceLayout()
{
    {
        std::lock_guard<std::mutex> lock(stopMutex_);
        stop_ = true;
    }
    stopCv_.notify_all();
    for (auto& thread : probeThreads_) {
        if (thread.joinable()) { thread.join(); }
    }
}

Status SpaceLayout::Setup(const Config& config)
{
    if (!storageBackends_.empty()) { return Status::InvalidParam("space layout already set up"); }
    auto status = config.backendHealth.Validate();
    if (status.Failure()) { return status; }
    if (config.storageBackends.empty()) { return Status::InvalidParam("empty storage backends"); }
    dataDirShardBytes_ = config.dataDirShardBytes;
    dataDirShard_ = dataDirShardBytes_ > 0;
    ioDirect_ = config.ioDirect;
    shards_ = RelativeRoots();
    for (const auto& path : config.storageBackends) {
        if (path.empty()) { return Status::InvalidParam("empty storage backend path"); }
        auto normalizedPath = path.back() == '/' ? path : path + '/';
        if (std::find(storageBackends_.begin(), storageBackends_.end(), normalizedPath) !=
            storageBackends_.end()) {
            continue;
        }
        const auto index = storageBackends_.size();
        storageBackends_.push_back(normalizedPath);
        status = Status::OK();
        for (const auto& root : shards_) {
            PosixFile dir{normalizedPath + root};
            if (availableBackends_.empty()) {
                status = dir.MkDir();
                if (status == Status::DuplicateKey()) { status = Status::OK(); }
            } else {
                status = dir.Access(PosixFile::AccessMode::READ | PosixFile::AccessMode::WRITE);
            }
            if (status.Failure()) { break; }
        }
        backendHealth_.emplace_back(config.backendHealth, status.Success());
        if (status.Success()) {
            availableBackends_.push_back(index);
        } else {
            UC_WARN("Storage backend({}) unavailable during setup: {}.", normalizedPath, status);
        }
    }
    if (availableBackends_.empty()) { return status; }
    if (storageBackends_.size() > 1) {
        try {
            for (size_t i = 0; i < storageBackends_.size(); ++i) {
                probeThreads_.emplace_back(&SpaceLayout::ProbeBackend, this, i,
                                           config.backendHealth);
            }
        } catch (const std::exception& e) {
            return Status::Error(
                fmt::format("failed to start backend health probes: {}", e.what()));
        }
    }
    return Status::OK();
}

Expected<std::string> SpaceLayout::DataFilePath(const Detail::BlockId& blockId,
                                                bool activated) const
{
    auto backend = StorageBackend();
    if (!backend) { return backend.Error(); }
    return DataFilePath(backend.Value(), blockId, activated);
}

std::string SpaceLayout::DataFilePath(const std::string& backend, const Detail::BlockId& blockId,
                                      bool activated) const
{
    const auto& file = DataFileName(blockId);
    const auto& shard = dataDirShard_ ? FileShardName(file) : DATA_ROOT;
    if (!activated) { return fmt::format("{}{}/{}", backend, shard, file); }
    return fmt::format("{}{}/{}{}", backend, shard, file, ACTIVATED_FILE_EXTENSION);
}

Status SpaceLayout::CommitFile(const Detail::BlockId& blockId, bool success) const
{
    auto backend = StorageBackend();
    if (!backend) { return backend.Error(); }
    // Both names must use one mount path: rename across aliases can fail with EXDEV.
    const auto activated = DataFilePath(backend.Value(), blockId, true);
    auto s = Status::OK();
    if (success) {
        const auto archived = DataFilePath(backend.Value(), blockId, false);
        s = PosixFile{activated}.Rename(archived);
    }
    if (!success || s.Failure()) { PosixFile{activated}.Remove(); }
    return s;
}

Status SpaceLayout::RemoveFile(const Detail::BlockId& blockId) const
{
    auto path = DataFilePath(blockId, false);
    if (!path) { return path.Error(); }
    return PosixFile{path.Value()}.Remove();
}

std::vector<std::string> SpaceLayout::RelativeRoots() const
{
    if (dataDirShard_) { return GenerateHexStrings(dataDirShardBytes_); }
    return {DATA_ROOT};
}

Expected<std::string> SpaceLayout::StorageBackend() const
{
    std::lock_guard<std::mutex> lock(backendMutex_);
    if (availableBackends_.empty()) { return Status::StoreUnhealthy(); }
    const auto index = availableBackends_[nextBackend_++ % availableBackends_.size()];
    return std::string(storageBackends_[index]);
}

Status SpaceLayout::CheckHealth() const
{
    if (storageBackends_.size() == 1) {
        return CheckPathHealth(
            DataFilePath(storageBackends_.front(), Detail::RandomBlockId(), true), ioDirect_);
    }
    std::lock_guard<std::mutex> lock(backendMutex_);
    return availableBackends_.empty() ? Status::StoreUnhealthy() : Status::OK();
}

void SpaceLayout::ProbeBackend(size_t index, const Detail::StoreHealthConfig& config)
{
    auto status = CpuAffinity::SetCurrentThreadName("ucm_posix_probe");
    if (status.Failure()) { UC_WARN("Failed to name backend health monitor: {}.", status); }
    Detail::HealthCheckExecutor executor{config.healthCheckTimeout};
    std::unique_lock<std::mutex> stopLock(stopMutex_);
    auto delay = config.healthCheckInterval;
    while (!stopCv_.wait_for(stopLock, delay, [this] { return stop_; })) {
        stopLock.unlock();
        const auto start = std::chrono::steady_clock::now();
        // Timed-out probes may still run; each one owns its file name and captures no layout state.
        const auto path = DataFilePath(storageBackends_[index], Detail::RandomBlockId(), true);
        status =
            executor.Run([path, ioDirect = ioDirect_] { return CheckPathHealth(path, ioDirect); });
        {
            std::lock_guard<std::mutex> lock(backendMutex_);
            auto& health = backendHealth_[index];
            const auto wasAvailable = health.Healthy();
            health.Record(status.Success());
            if (wasAvailable != health.Healthy()) {
                availableBackends_.clear();
                for (size_t i = 0; i < backendHealth_.size(); ++i) {
                    if (backendHealth_[i].Healthy()) { availableBackends_.push_back(i); }
                }
                nextBackend_ = 0;
                UC_WARN(
                    "Storage backend({}) is {}, samples={}, failures={}, threshold={}, status={}.",
                    storageBackends_[index], health.Healthy() ? "HEALTHY" : "UNHEALTHY",
                    health.SampleCount(), health.FailureCount(), config.failureThreshold, status);
            }
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        delay = elapsed < config.healthCheckInterval ? config.healthCheckInterval - elapsed
                                                     : std::chrono::milliseconds{0};
        stopLock.lock();
    }
}

static Detail::BlockId HexToBlockId(const char* hexStr)
{
    Detail::BlockId blockId;
    for (size_t i = 0; i < 16; ++i) {
        uint8_t high = static_cast<uint8_t>(hexStr[i * 2]);
        uint8_t low = static_cast<uint8_t>(hexStr[i * 2 + 1]);
        high = (high <= '9') ? (high - '0') : (high - 'a' + 10);
        low = (low <= '9') ? (low - '0') : (low - 'a' + 10);
        blockId[i] = static_cast<std::byte>((high << 4) | low);
    }
    return blockId;
}

std::vector<std::string> SpaceLayout::SampleShards(double sampleRatio) const
{
    if (sampleRatio == 1.0) { return shards_; }
    auto shards = shards_;
    size_t sampleCount =
        std::max(static_cast<size_t>(1), static_cast<size_t>(shards.size() * sampleRatio));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(shards.begin(), shards.end(), gen);
    shards.resize(sampleCount);
    return shards;
}

size_t SpaceLayout::CountFilesInShard(const std::string& shard) const
{
    auto backend = StorageBackend();
    if (!backend) { return 0; }
    std::string shardPath = backend.Value();
    shardPath += shard;
    DIR* dir = opendir(shardPath.c_str());
    if (!dir) { return 0; }
    size_t count = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') { continue; }
        if (strstr(entry->d_name, ACTIVATED_FILE_EXTENSION.c_str()) != nullptr) { continue; }
        ++count;
    }
    closedir(dir);
    return count;
}

static size_t ScanFilesInShard(const std::string& shardPath,
                               TopNHeap<FileInfo, MtimeComparator>& heap)
{
    DIR* dir = opendir(shardPath.c_str());
    if (!dir) { return 0; }
    size_t totalFiles = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') { continue; }
        if (strstr(entry->d_name, ACTIVATED_FILE_EXTENSION.c_str()) != nullptr) { continue; }
        std::string filePath = shardPath + "/" + entry->d_name;
        struct stat st;
        if (stat(filePath.c_str(), &st) != 0) { continue; }
        if (!S_ISREG(st.st_mode)) { continue; }
        heap.Push({HexToBlockId(entry->d_name), st.st_mtime});
        ++totalFiles;
    }
    closedir(dir);
    return totalFiles;
}

std::vector<Detail::BlockId> SpaceLayout::GetOldestFiles(const std::string& shard,
                                                         double recyclePercent,
                                                         size_t maxRecycleCount) const
{
    auto backend = StorageBackend();
    if (!backend) { return {}; }
    std::string shardPath = backend.Value();
    shardPath += shard;
    auto heap = std::make_unique<TopNHeap<FileInfo, MtimeComparator>>(maxRecycleCount);
    size_t totalFiles = ScanFilesInShard(shardPath, *heap);
    if (totalFiles == 0) { return {}; }
    size_t recycleNum = static_cast<size_t>(totalFiles * recyclePercent);
    if (recycleNum == 0) { return {}; }
    recycleNum = std::min(recycleNum, maxRecycleCount);
    size_t skipCount = heap->Size() - recycleNum;
    for (size_t i = 0; i < skipCount; ++i) { heap->Pop(); }
    std::vector<Detail::BlockId> result;
    result.reserve(recycleNum);
    while (!heap->Empty()) {
        result.push_back(heap->Top().blockId);
        heap->Pop();
    }
    return result;
}

std::string SpaceLayout::ShardOf(const Detail::BlockId& blockId) const
{
    if (!dataDirShard_) { return DATA_ROOT; }
    return FileShardName(DataFileName(blockId));
}

std::vector<FileInfo> SpaceLayout::GetColdestCandidates(const std::string& shard,
                                                        double candidatePercent,
                                                        size_t maxCandidateCount) const
{
    auto backend = StorageBackend();
    if (!backend) { return {}; }
    std::string shardPath = backend.Value();
    shardPath += shard;
    auto heap = std::make_unique<TopNHeap<FileInfo, MtimeComparator>>(maxCandidateCount);
    size_t totalFiles = ScanFilesInShard(shardPath, *heap);
    if (totalFiles == 0) { return {}; }
    size_t candidateNum = static_cast<size_t>(totalFiles * candidatePercent);
    if (candidateNum == 0) { return {}; }
    candidateNum = std::min(candidateNum, maxCandidateCount);
    size_t skipCount = heap->Size() - std::min<size_t>(candidateNum, heap->Size());
    for (size_t i = 0; i < skipCount; ++i) { heap->Pop(); }
    std::vector<FileInfo> result;
    result.reserve(heap->Size());
    while (!heap->Empty()) {
        result.push_back(heap->Top());
        heap->Pop();
    }
    return result;
}

}  // namespace UC::PosixStore
