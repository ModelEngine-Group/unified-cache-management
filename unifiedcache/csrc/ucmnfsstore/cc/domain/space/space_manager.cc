/**
/* MIT License
/*
/* Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
/*
/* Permission is hereby granted, free of charge, to any person obtaining a copy
/* of this software and associated documentation files (the "Software"), to deal
/* in the Software without restriction, including without limitation the rights
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/* copies of the Software, and to permit persons to whom the Software is
/* furnished to do so, subject to the following conditions:
/*
/* The above copyright notice and this permission notice shall be included in all
/* copies or substantial portions of the Software.
/*
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/* SOFTWARE.
 * */
#include "space_manager.h"
#include "file/file.h"
#include "logger/logger.h"
#include <algorithm>

namespace UC {

Status SpaceManager::Setup(const std::vector<std::string>& storageBackends, const size_t blockSize)
{
    if (storageBackends.empty()) {
        UC_ERROR("Empty backend list.");
        return Status::InvalidParam();
    }
    for (auto& path : storageBackends) {
        Status status = this->AddStorageBackend(path);
        if (status.Failure()) { return status; }
    }
    if (blockSize == 0) {
        UC_ERROR("Invalid block size({}).", blockSize);
        return Status::InvalidParam();
    }
    this->_blockSize = blockSize;
    return Status::OK();
}

Status SpaceManager::NewBlock(const std::string& blockId)
{
    auto parent = File::Make(this->BlockParentPath(blockId));
    auto file = File::Make(this->BlockPath(blockId, true));
    if (!parent || !file) {
        UC_ERROR("Failed to new block({}).", blockId);
        return Status::OutOfMemory();
    }
    auto status = parent->MkDir();
    if (status == Status::DuplicateKey()) { status = Status::OK(); }
    if (status.Failure()) {
        UC_ERROR("Failed({}) to new block({}).", status, blockId);
        return status;
    }
    status = file->Open(IFile::OpenFlag::CREATE | IFile::OpenFlag::READ_WRITE);
    if (status.Failure()) {
        UC_ERROR("Failed({}) to new block({}).", status, blockId);
        return status;
    }
    status = file->Truncate(this->_blockSize);
    if (status.Failure()) {
        UC_ERROR("Failed({}) to new block({}).", status, blockId);
        return status;
    }
    return Status::OK();
}

Status SpaceManager::CommitBlock(const std::string& blockId, bool success)
{
    auto file = File::Make(this->BlockPath(blockId, true));
    if (!file) {
        UC_ERROR("Failed to {} block({}).", success ? "commit" : "cancel", blockId);
        return Status::OutOfMemory();
    }
    if (success) {
        auto status = file->Rename(this->BlockPath(blockId, false));
        if (status.Failure()) { UC_ERROR("Failed({}) to commit block({}).", status, blockId); }
        return status;
    }
    auto parent = File::Make(this->BlockParentPath(blockId));
    if (!parent) {
        UC_ERROR("Failed to cancel block({}).", blockId);
        return Status::OutOfMemory();
    }
    file->Remove();
    parent->RmDir();
    return Status::OK();
}

std::string SpaceManager::BlockPath(const std::string& blockId, bool actived)
{
    return this->StorageBackend(blockId) + this->_layout.DataFilePath(blockId);
}

Status SpaceManager::AddStorageBackend(const std::string& path)
{
    auto normalizedPath = path;
    if (normalizedPath.back() != '/') { normalizedPath += '/'; }
    auto status = Status::OK();
    if (this->_storageBackends.empty()) {
        status = this->AddFirstStorageBackend(normalizedPath);
    } else {
        status = this->AddSecondaryStorageBackend(path);
    }
    if (status.Success()) {
        UC_INFO("Add UC::StorageBackend: {}.", normalizedPath);
    } else {
        UC_ERROR("Failed({}) to add storage backend({}).", status, normalizedPath);
    }
    return status;
}

Status SpaceManager::AddFirstStorageBackend(const std::string& path)
{
    for (const auto& root : this->_layout.RelativeRoots()) {
        auto dir = File::Make(path + root);
        if (!dir) { return Status::OutOfMemory(); }
        auto status = dir->MkDir();
        if (status == Status::DuplicateKey()) { status = Status::OK(); }
        if (status.Failure()) { return status; }
    }
    this->_storageBackends.emplace_back(path);
    return Status::OK();
}

Status SpaceManager::AddSecondaryStorageBackend(const std::string& path)
{
    auto iter = std::find(this->_storageBackends.begin(), this->_storageBackends.end(), path);
    if (iter != this->_storageBackends.end()) { return Status::OK(); }
    constexpr auto accessMode = IFile::AccessMode::READ | IFile::AccessMode::WRITE;
    for (const auto& root : this->_layout.RelativeRoots()) {
        auto dir = File::Make(path + root);
        auto status = dir->Access(accessMode);
        if (status.Failure()) { return status; }
    }
    this->_storageBackends.emplace_back(path);
    return Status::OK();
}

std::string SpaceManager::StorageBackend(const std::string& blockId)
{
    static std::hash<std::string> hasher;
    return this->_storageBackends[hasher(blockId) % this->_storageBackends.size()];
}

std::string SpaceManager::BlockParentPath(const std::string& blockId)
{
    return this->StorageBackend(blockId) + this->_layout.DataFileParent(blockId);
}

} // namespace UC
