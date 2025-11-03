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
#include "space_shard_layout.h"
#include <algorithm>
#include <dirent.h>
#include <array>
#include <stack>
#include "file/file.h"
#include "logger/logger.h"

namespace UC {

constexpr size_t blockIdSize = 16;
constexpr size_t nU64PerBlock = blockIdSize / sizeof(uint64_t);
using BlockId = std::array<uint64_t, nU64PerBlock>;
static_assert(sizeof(BlockId) == blockIdSize);

const std::string activatedFileSuffix = "act";
const std::string archivedFileSuffix = "dat";

inline auto OpenDir(const std::string& path)
{
    auto dir = ::opendir(path.c_str());
    auto eno = errno;
    if (!dir) { UC_ERROR("Failed({}) to open dir({}).", eno, path); }
    return dir;
}

// Define SpaceLayout::DataIterator as an empty base class
struct SpaceLayout::DataIterator {
    virtual ~DataIterator() = default;
};

struct SpaceShardLayout::DataIterator : public SpaceLayout::DataIterator {
    const SpaceLayout* layout{nullptr};
    std::string root;
    std::string current;
    std::stack<std::pair<DIR*, std::string>> stk;
    ~DataIterator()
    {
        while (!this->stk.empty()) {
            ::closedir(this->stk.top().first);
            this->stk.pop();
        }
    }
    Status Setup(const SpaceLayout* layout, const std::string& root) {
        this->layout = layout;
        this->root = root;
        auto dir = OpenDir(root);
        if (!dir) { return Status::OsApiError(); }
        this->stk.emplace(dir, root);
        return Status::OK();
    }
    Status Next() {
        this->current.clear();
        while (!this->stk.empty()) {
            auto entry = ::readdir64(this->stk.top().first);
            if (entry == nullptr) {
                ::closedir(this->stk.top().first);
                this->stk.pop();
                continue;
            }
            std::string name{entry->d_name};
            if (name.front() == '.') { continue; }
            if (this->layout->IsActivatedFile(name)) { continue; }
            const auto& dir = this->stk.top().second;
            auto fullpath = this->stk.top().second + "/" + name;
            if (dir == this->root) {
                auto sub = OpenDir(fullpath);
                if (!sub) { return Status::OsApiError(); }
                this->stk.emplace(sub, fullpath);
                continue;
            }
            this->current = std::move(fullpath);
            return Status::OK();
        }
        return Status::NotFound();
    }
};

Status SpaceShardLayout::Setup(const std::vector<std::string>& storageBackends)
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

std::string SpaceShardLayout::DataFileParent(const std::string& blockId, bool activated) const
{
    uint64_t front, back;
    this->ShardBlockId(blockId, front, back);
    return fmt::format("{}{}/{:016x}", this->StorageBackend(blockId), this->DataFileRoot(), front);
}

std::string SpaceShardLayout::DataFilePath(const std::string& blockId, bool activated) const
{
    uint64_t front, back;
    this->ShardBlockId(blockId, front, back);
    return fmt::format("{}{}/{:016x}/{:016x}.{}", this->StorageBackend(blockId),
                       this->DataFileRoot(), front, back, activated ? activatedFileSuffix : archivedFileSuffix);
}

Status SpaceShardLayout::AddStorageBackend(const std::string& path)
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

Status SpaceShardLayout::AddFirstStorageBackend(const std::string& path)
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

Status SpaceShardLayout::AddSecondaryStorageBackend(const std::string& path)
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

std::string SpaceShardLayout::StorageBackend(const std::string& blockId) const
{
    static std::hash<std::string> hasher;
    return this->storageBackends_[hasher(blockId) % this->storageBackends_.size()];
}

std::vector<std::string> SpaceShardLayout::RelativeRoots() const {
    return {
        this->DataFileRoot(),
        this->ClusterFileRoot(),
    };
}

std::string SpaceShardLayout::DataFileRoot() const { return "data"; }
std::string SpaceShardLayout::ClusterFileRoot() const { return "cluster"; }
void SpaceShardLayout::ShardBlockId(const std::string& blockId, uint64_t& front,
                                    uint64_t& back) const
{
    auto id = static_cast<const BlockId*>(static_cast<const void*>(blockId.data()));
    front = id->front();
    back = id->back();
}

std::string SpaceShardLayout::StorageBackend() const { return this->storageBackends_.front(); }
std::string SpaceShardLayout::ClusterPropertyFilePath() const
{
    return fmt::format("{}{}/{}.bin", this->StorageBackend(), this->ClusterFileRoot(), "uc_property");
}

std::shared_ptr<SpaceLayout::DataIterator> SpaceShardLayout::CreateFilePathIterator() const
{
    auto dataRoot = this->StorageBackend() + this->DataFileRoot();
    std::shared_ptr<DataIterator> iter = nullptr;
    try {
        iter = std::make_shared<DataIterator>();
    } catch (const std::exception& e) {
        UC_ERROR("Failed to create data iterator: {}", e.what());
        return nullptr;
    }
    if (iter->Setup(this, dataRoot).Failure()) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<SpaceLayout::DataIterator>(iter);
}

std::string SpaceShardLayout::NextDataFilePath(std::shared_ptr<SpaceLayout::DataIterator> iter) const
{
    auto shard_iter = std::dynamic_pointer_cast<DataIterator>(iter);
    if (!shard_iter) { return std::string{}; }
    if (shard_iter->Next().Failure()) { return std::string{}; }
    return shard_iter->current;
}

bool SpaceShardLayout::IsActivatedFile(const std::string& filePath) const
{
    return std::equal(activatedFileSuffix.rbegin(), activatedFileSuffix.rend(), filePath.rbegin());
}
} // namespace UC
