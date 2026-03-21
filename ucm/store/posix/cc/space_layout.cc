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
#include <cstring>
#include <dirent.h>
#include <fmt/ranges.h>
#include <random>
#include <sys/stat.h>
#include <unistd.h>
#include "logger/logger.h"
#include "posix_file.h"
#include "template/topn_heap.h"

namespace UC::PosixStore {

static const std::string DATA_ROOT = "data";
static const std::string ACTIVATED_FILE_EXTENSION = ".tmp";

struct FileInfo {
    Detail::BlockId blockId;
    time_t mtime;
};

struct MtimeComparator {
    bool operator()(const FileInfo& lhs, const FileInfo& rhs) const
    {
        return lhs.mtime < rhs.mtime;
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

Status SpaceLayout::Setup(const Config& config)
{
    dataDirShardBytes_ = config.dataDirShardBytes;
    dataDirShard_ = dataDirShardBytes_ > 0;
    auto status = Status::OK();
    for (auto& path : config.storageBackends) {
        if ((status = AddStorageBackend(path)).Failure()) { return status; }
    }
    return status;
}

std::string SpaceLayout::DataFilePath(const Detail::BlockId& blockId, bool activated) const
{
    const auto& backend = StorageBackend(blockId);
    const auto& file = DataFileName(blockId);
    const auto& shard = dataDirShard_ ? FileShardName(file) : DATA_ROOT;
    if (!activated) { return fmt::format("{}{}/{}", backend, shard, file); }
    return fmt::format("{}{}/{}{}", backend, shard, file, ACTIVATED_FILE_EXTENSION);
}

Status SpaceLayout::CommitFile(const Detail::BlockId& blockId, bool success) const
{
    const auto& activated = DataFilePath(blockId, true);
    auto s = Status::OK();
    if (success) {
        const auto& archived = DataFilePath(blockId, false);
        s = PosixFile{activated}.Rename(archived);
    }
    if (!success || s.Failure()) { PosixFile{activated}.Remove(); }
    return s;
}

std::vector<std::string> SpaceLayout::RelativeRoots() const
{
    if (dataDirShard_) { return GenerateHexStrings(dataDirShardBytes_); }
    return {DATA_ROOT};
}

Status SpaceLayout::AddStorageBackend(const std::string& path)
{
    auto normalizedPath = path;
    if (normalizedPath.back() != '/') { normalizedPath += '/'; }
    auto status = Status::OK();
    if (storageBackends_.empty()) {
        status = AddFirstStorageBackend(normalizedPath);
    } else {
        status = AddSecondaryStorageBackend(normalizedPath);
    }
    if (status.Failure()) {
        UC_ERROR("Failed({}) to add storage backend({}).", status, normalizedPath);
    }
    return status;
}

Status SpaceLayout::AddFirstStorageBackend(const std::string& path)
{
    for (const auto& root : RelativeRoots()) {
        PosixFile dir{path + root};
        auto status = dir.MkDir();
        if (status == Status::DuplicateKey()) { status = Status::OK(); }
        if (status.Failure()) { return status; }
    }
    storageBackends_.emplace_back(path);
    return Status::OK();
}

Status SpaceLayout::AddSecondaryStorageBackend(const std::string& path)
{
    auto iter = std::find(storageBackends_.begin(), storageBackends_.end(), path);
    if (iter != storageBackends_.end()) { return Status::OK(); }
    constexpr auto accessMode = PosixFile::AccessMode::READ | PosixFile::AccessMode::WRITE;
    for (const auto& root : RelativeRoots()) {
        PosixFile dir{path + root};
        auto status = dir.Access(accessMode);
        if (status.Failure()) { return status; }
    }
    storageBackends_.emplace_back(path);
    return Status::OK();
}

std::string SpaceLayout::StorageBackend(const Detail::BlockId& blockId) const
{
    const auto number = storageBackends_.size();
    if (number == 1) { return storageBackends_.front(); }
    static Detail::BlockIdHasher hasher;
    return storageBackends_[hasher(blockId) % number];
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
    auto shards = RelativeRoots();
    if (shards.empty()) { return {}; }

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
    std::string shardPath = storageBackends_.front();
    if (shardPath.back() != '/') { shardPath += '/'; }
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

std::vector<Detail::BlockId> SpaceLayout::GetOldestFiles(const std::string& shard,
                                                         double recyclePercent,
                                                         size_t maxRecycleCount) const
{
    std::string shardPath = storageBackends_.front();
    if (shardPath.back() != '/') { shardPath += '/'; }
    shardPath += shard;

    DIR* dir = opendir(shardPath.c_str());
    if (!dir) { return {}; }

    auto heap = std::make_unique<TopNHeap<FileInfo, MtimeComparator>>(maxRecycleCount);
    size_t totalFiles = 0;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') { continue; }
        if (strstr(entry->d_name, ACTIVATED_FILE_EXTENSION.c_str()) != nullptr) { continue; }
        std::string filePath = shardPath + "/" + entry->d_name;
        struct stat st;
        if (stat(filePath.c_str(), &st) != 0) { continue; }
        if (!S_ISREG(st.st_mode)) { continue; }
        heap->Push({HexToBlockId(entry->d_name), st.st_mtime});
        ++totalFiles;
        if (totalFiles >= maxRecycleCount) { break; }
    }
    closedir(dir);

    if (totalFiles == 0) { return {}; }

    size_t recycleNum = static_cast<size_t>(totalFiles * recyclePercent);
    if (recycleNum == 0) { return {}; }
    recycleNum = std::min(recycleNum, maxRecycleCount);

    std::vector<Detail::BlockId> result;
    result.reserve(recycleNum);
    while (!heap->Empty() && result.size() < recycleNum) {
        result.push_back(heap->Top().blockId);
        heap->Pop();
    }

    return result;
}

}  // namespace UC::PosixStore
