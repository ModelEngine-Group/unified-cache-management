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
#include "file/file.h"
#include "logger/logger.h"

namespace UC {

Status SpaceLayout::Setup(const std::vector<std::string>& storageBackends)
{
    if (storageBackends.empty()) {
        UC_ERROR("Empty backend list.");
        return Status::InvalidParam();
    }
    auto status = Status::OK();
    for (auto& path : storageBackends) {
        if ((status = this->AddStorageBackend(path)).Failure()) { return status; }
    }
    return status;
}

std::string SpaceLayout::DataFilePath(const std::string& blockId, bool activated) const
{
    const auto& backend = StorageBackend(blockId);
    const auto& dir = activated ? TempFileRoot() : DataFileRoot();
    uint64_t front, back;
    ShardBlockId(blockId, front, back);
    return fmt::format("{}{}{:016x}{:016x}", backend, dir, front, back);
}

Status SpaceLayout::Commit(const std::string& blockId, bool success) const
{
    const auto& activated = this->DataFilePath(blockId, true);
    const auto& archived = this->DataFilePath(blockId, false);
    if (success) { return File::Rename(activated, archived); }
    File::Remove(activated);
    return Status::OK();
}

std::vector<std::string> SpaceLayout::RelativeRoots() const
{
    return {DataFileRoot(), TempFileRoot()};
}

Status SpaceLayout::AddStorageBackend(const std::string& path)
{
    auto normalizedPath = path;
    if (normalizedPath.back() != '/') { normalizedPath += '/'; }
    auto status = Status::OK();
    if (this->storageBackends_.empty()) {
        status = this->AddFirstStorageBackend(normalizedPath);
    } else {
        status = this->AddSecondaryStorageBackend(normalizedPath);
    }
    if (status.Failure()) {
        UC_ERROR("Failed({}) to add storage backend({}).", status, normalizedPath);
    }
    return status;
}

Status SpaceLayout::AddFirstStorageBackend(const std::string& path)
{
    for (const auto& root : this->RelativeRoots()) {
        auto dir = File::Make(path + root);
        if (!dir) { return Status::OutOfMemory(); }
        auto status = dir->MkDir();
        if (status == Status::DuplicateKey()) { status = Status::OK(); }
        if (status.Failure()) { return status; }
    }
    this->storageBackends_.emplace_back(path);
    return Status::OK();
}

Status SpaceLayout::AddSecondaryStorageBackend(const std::string& path)
{
    auto iter = std::find(this->storageBackends_.begin(), this->storageBackends_.end(), path);
    if (iter != this->storageBackends_.end()) { return Status::OK(); }
    constexpr auto accessMode = IFile::AccessMode::READ | IFile::AccessMode::WRITE;
    for (const auto& root : this->RelativeRoots()) {
        auto dir = File::Make(path + root);
        if (!dir) { return Status::OutOfMemory(); }
        if (dir->Access(accessMode).Failure()) { return Status::InvalidParam(); }
    }
    this->storageBackends_.emplace_back(path);
    return Status::OK();
}

std::string SpaceLayout::StorageBackend(const std::string& blockId) const
{
    static std::hash<std::string> hasher;
    return this->storageBackends_[hasher(blockId) % this->storageBackends_.size()];
}

std::string SpaceLayout::DataFileRoot() const { return "data/"; }

std::string SpaceLayout::TempFileRoot() const { return "temp/"; }

void SpaceLayout::ShardBlockId(const std::string& blockId, uint64_t& front, uint64_t& back) const
{
    constexpr size_t blockIdSize = 16;
    constexpr size_t nU64PerBlock = blockIdSize / sizeof(uint64_t);
    using BlockId = std::array<uint64_t, nU64PerBlock>;
    static_assert(sizeof(BlockId) == blockIdSize);
    auto id = static_cast<const BlockId*>(static_cast<const void*>(blockId.data()));
    front = id->front();
    back = id->back();
}

} // namespace UC
