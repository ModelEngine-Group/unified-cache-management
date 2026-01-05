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
#include <fmt/ranges.h>
#include "logger/logger.h"
#include "posix_file.h"

namespace UC::PosixStore {

static const std::string DATA_ROOT = "data";
static const std::string TEMP_ROOT = ".temp";

inline std::string DataFileName(const Detail::BlockId& blockId)
{
    return fmt::format("{:02x}", fmt::join(blockId, ""));
}

inline std::string DataFileShardName(const std::string& fileName) { return fileName.substr(0, 8); }

Status SpaceLayout::Setup(const Config& config)
{
    shardDataDir_ = config.shardDataDir;
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
    const auto& shard =
        activated ? TEMP_ROOT : (shardDataDir_ ? DataFileShardName(file) : DATA_ROOT);
    return fmt::format("{}{}/{}", backend, shard, file);
}

Status SpaceLayout::CommitFile(const Detail::BlockId& blockId, bool success) const
{
    const auto& backend = StorageBackend(blockId);
    const auto& file = DataFileName(blockId);
    const auto& activated = fmt::format("{}{}/{}", backend, TEMP_ROOT, file);
    auto s = Status::OK();
    if (success) {
        const auto& shard = shardDataDir_ ? DataFileShardName(file) : DATA_ROOT;
        const auto& archived = fmt::format("{}{}/{}", backend, shard, file);
        if (shardDataDir_) { s = PosixFile{backend + shard}.MkDir(); }
        if (s == Status::OK() || s == Status::DuplicateKey()) {
            s = PosixFile{activated}.Rename(archived);
        }
    }
    if (!success || s.Failure()) { PosixFile{activated}.Remove(); }
    return s;
}

std::vector<std::string> SpaceLayout::RelativeRoots() const
{
    std::vector<std::string> roots{TEMP_ROOT};
    if (!shardDataDir_) { roots.push_back(DATA_ROOT); }
    return roots;
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

}  // namespace UC::PosixStore
