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
#include "ds3fs_file.h"
#include "logger/logger.h"

namespace UC::Ds3fsStore {

static const std::string DATA_ROOT = "data/";
static const std::string TEMP_ROOT = "temp/";

Status SpaceLayout::Setup(const std::vector<std::string>& storageBackends)
{
    auto status = Status::OK();
    for (auto& path : storageBackends) {
        if ((status = AddStorageBackend(path)).Failure()) { return status; }
    }
    return status;
}

std::string SpaceLayout::DataFilePath(const Detail::BlockId& blockId, bool activated) const
{
    const auto& backend = StorageBackend(blockId);
    const auto& root = !activated ? DATA_ROOT : TEMP_ROOT;
    return fmt::format("{}{}{:02x}", backend, root, fmt::join(blockId, ""));
}

Status SpaceLayout::CommitFile(const Detail::BlockId& blockId, bool success) const
{
    const auto& activated = this->DataFilePath(blockId, true);
    const auto& archived = this->DataFilePath(blockId, false);
    Ds3fsFile file{activated};
    if (success) { return file.Rename(archived); }
    file.Remove();
    return Status::OK();
}

std::vector<std::string> SpaceLayout::RelativeRoots() const
{
    return {
        DATA_ROOT,
        TEMP_ROOT,
    };
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
        Ds3fsFile dir{path + root};
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
    constexpr auto accessMode = Ds3fsFile::AccessMode::READ | Ds3fsFile::AccessMode::WRITE;
    for (const auto& root : RelativeRoots()) {
        Ds3fsFile dir{path + root};
        auto status = dir.Access(accessMode);
        if (status.Failure()) { return status; }
    }
    storageBackends_.emplace_back(path);
    return Status::OK();
}

std::string SpaceLayout::StorageBackend(const Detail::BlockId& blockId) const
{
    static Detail::BlockIdHasher hasher;
    const auto number = storageBackends_.size();
    if (number > 1) { return storageBackends_[hasher(blockId) % number]; }
    return storageBackends_.front();
}

}  // namespace UC::Ds3fsStore
