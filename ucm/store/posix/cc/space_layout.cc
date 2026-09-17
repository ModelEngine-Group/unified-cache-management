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
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fmt/ranges.h>
#include <random>
#include <sys/stat.h>
#include <unistd.h>
#include "common/health_check_executor.h"
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
    if (status.Success() && actual != expected) {
        status = Status::Error(fmt::format("verify('{}') failed: health data mismatch", path));
    }
    auto closed = file.Close();
    if (status.Success()) { status = closed; }
    auto cleanup = file.Remove();
    return status.Failure() ? status : cleanup;
}

SpaceLayout::~SpaceLayout()
{
    {
        std::lock_guard<std::mutex> lock(stopMutex_);
        stop_ = true;
    }
    stopCv_.notify_all();
    if (probeThread_.joinable()) { probeThread_.join(); }
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
        const auto startupError = [&](const Status& error) {
            auto message = fmt::format("Storage backend '{}' failed startup I/O check: {}",
                                       normalizedPath, error);
            UC_ERROR("{}", message);
            return Status{error.Underlying(), std::move(message), error.SystemError()};
        };
        for (const auto& root : shards_) {
            PosixFile dir{normalizedPath + root};
            if (index == 0) {
                status = dir.MkDir();
                if (status == Status::DuplicateKey()) { status = Status::OK(); }
            } else {
                status = dir.Access(PosixFile::AccessMode::READ | PosixFile::AccessMode::WRITE);
            }
            if (status.Failure()) { return startupError(status); }
        }
        status =
            CheckPathHealth(DataFilePath(normalizedPath, Detail::RandomBlockId(), true), ioDirect_);
        if (status.Failure()) { return startupError(status); }
        backendHealth_.emplace_back(config.backendHealth);
    }
    for (size_t i = 0; i < storageBackends_.size(); ++i) { availableBackends_.push_back(i); }
    ioTimeoutMs_ =
        config.timeoutMs == 0 ? 0 : std::max<size_t>(1, config.timeoutMs / BackendCount());
    try {
        probeThread_ = std::thread(&SpaceLayout::ProbeBackends, this, config.backendHealth);
    } catch (const std::exception& e) {
        availableBackends_.clear();
        return Status::Error(fmt::format("failed to start backend recovery probes: {}", e.what()));
    }
    return Status::OK();
}

Expected<std::string> SpaceLayout::DataFilePath(const Detail::BlockId& blockId,
                                                bool activated) const
{
    auto backend = StorageBackend(blockId);
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
    return RunOnAvailableBackend(
        blockId, [&](const std::string& backend) { return CommitFile(backend, blockId, success); });
}

Status SpaceLayout::CommitFile(const std::string& backend, const Detail::BlockId& blockId,
                               bool success) const
{
    const auto activated = DataFilePath(backend, blockId, true);
    if (!success) { return PosixFile{activated}.Remove(); }
    return PosixFile{activated}.Rename(DataFilePath(backend, blockId, false));
}

Status SpaceLayout::RemoveFile(const Detail::BlockId& blockId) const
{
    return RunOnAvailableBackend(blockId, [&](const std::string& backend) {
        return PosixFile{DataFilePath(backend, blockId, false)}.Remove();
    });
}

std::vector<std::string> SpaceLayout::RelativeRoots() const
{
    if (dataDirShard_) { return GenerateHexStrings(dataDirShardBytes_); }
    return {DATA_ROOT};
}

Expected<std::string> SpaceLayout::StorageBackend(const Detail::BlockId& blockId,
                                                  const std::vector<std::string>& excluded) const
{
    std::lock_guard<std::mutex> lock(backendMutex_);
    if (availableBackends_.empty()) { return Status::StoreUnhealthy(); }
    const auto count = storageBackends_.size();
    const auto primary = Detail::BlockIdHasher{}(blockId) % count;
    for (size_t offset = 0; offset < count; ++offset) {
        const auto index = (primary + offset) % count;
        if (!backendHealth_[index].Healthy()) { continue; }
        const auto& backend = storageBackends_[index];
        if (std::find(excluded.begin(), excluded.end(), backend) == excluded.end()) {
            return std::string(backend);
        }
    }
    return Status::StoreUnhealthy();
}

Status SpaceLayout::RunOnAvailableBackend(
    const Detail::BlockId& blockId,
    const std::function<Status(const std::string&)>& operation) const
{
    std::vector<std::string> attempted;
    auto status = Status::StoreUnhealthy();
    while (auto backend = StorageBackend(blockId, attempted)) {
        attempted.push_back(backend.Value());
        status = operation(backend.Value());
        RecordIoResult(backend.Value(), status);
        if (status.Success() || status == Status::NotFound()) { break; }
    }
    return status;
}

void SpaceLayout::RecordIoResult(const std::string& backend, const Status& status) const
{
    if (status.Failure()) {
        if (status == Status::NotFound() || status == Status::InvalidParam() ||
            status == Status::OutOfMemory() || status == Status::DuplicateKey() ||
            status == Status::Retry() || status == Status::Unsupported() ||
            status == Status::NoSpace()) {
            return;
        }
        // These errors describe the request, local resources, or shared filesystem state.
        // Ignore the sample entirely so it cannot evict earlier transport failures.
        switch (status.SystemError()) {
            case ENOENT:
            case EEXIST:
            case ENOTDIR:
            case EISDIR:
            case ENOTEMPTY:
            case ELOOP:
            case ENAMETOOLONG:
            case ESTALE:
            case EACCES:
            case EPERM:
            case EROFS:
            case ENOSPC:
            case EDQUOT:
            case EFBIG:
            case EOVERFLOW:
            case EMLINK:
            case EXDEV:
            case EINVAL:
            case EBADF:
            case EFAULT:
            case ESPIPE:
            case EOPNOTSUPP:
            case ENOSYS:
            case EAGAIN:
            case EINTR:
            case ECANCELED:
            case ENOMEM:
            case EMFILE:
            case ENFILE:
            case EBUSY:
            case ETXTBSY: return;
            default: break;
        }
    }
    const auto found = std::find(storageBackends_.begin(), storageBackends_.end(), backend);
    if (found != storageBackends_.end()) {
        RecordHealth(std::distance(storageBackends_.begin(), found), status, false);
    }
}

void SpaceLayout::RecordHealth(size_t index, const Status& status, bool recovery) const
{
    std::lock_guard<std::mutex> lock(backendMutex_);
    auto& health = backendHealth_[index];
    const auto wasAvailable = health.Healthy();
    if (recovery && wasAvailable) { return; }
    // Late successful business I/O cannot restore an excluded backend.
    if (!recovery && !wasAvailable && status.Success()) { return; }
    health.Record(status.Success());
    if (wasAvailable == health.Healthy()) { return; }
    availableBackends_.clear();
    for (size_t i = 0; i < backendHealth_.size(); ++i) {
        if (backendHealth_[i].Healthy()) { availableBackends_.push_back(i); }
    }
    UC_WARN("Storage backend({}) is {}, samples={}, failures={}, status={}.",
            storageBackends_[index], health.Healthy() ? "HEALTHY" : "UNHEALTHY",
            health.SampleCount(), health.FailureCount(), status);
}

Status SpaceLayout::CheckHealth() const
{
    std::lock_guard<std::mutex> lock(backendMutex_);
    return availableBackends_.empty() ? Status::StoreUnhealthy() : Status::OK();
}

void SpaceLayout::ProbeBackends(const Common::StoreHealthConfig& config)
{
    auto status = CpuAffinity::SetCurrentThreadName("ucm_health_pmon");
    if (status.Failure()) { UC_WARN("Failed to name backend health monitor: {}.", status); }
    Common::HealthCheckExecutor executor(config.healthCheckTimeout, 0);
    std::unique_lock<std::mutex> stopLock(stopMutex_);
    auto delay = config.healthCheckInterval;
    while (!stopCv_.wait_for(stopLock, delay, [this] { return stop_; })) {
        stopLock.unlock();
        const auto start = std::chrono::steady_clock::now();
        for (size_t index = 0; index < storageBackends_.size(); ++index) {
            {
                std::lock_guard<std::mutex> lock(stopMutex_);
                if (stop_) { return; }
            }
            {
                std::lock_guard<std::mutex> lock(backendMutex_);
                if (backendHealth_[index].Healthy()) { continue; }
            }
            // A timed-out probe may still run, so each probe owns a distinct file name.
            const auto path = DataFilePath(storageBackends_[index], Detail::RandomBlockId(), true);
            status = executor.Run(
                [path, ioDirect = ioDirect_] { return CheckPathHealth(path, ioDirect); });
            RecordHealth(index, status, true);
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
    auto backend = StorageBackend({});
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
    auto backend = StorageBackend({});
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
    auto backend = StorageBackend({});
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
