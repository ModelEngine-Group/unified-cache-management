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
#include "space_manager.h"
#include "file/file.h"
#include "logger/logger.h"

namespace UC {

Status SpaceManager::Setup(const std::vector<std::string>& storageBackends, const size_t blockSize)
{
    if (blockSize == 0) {
        UC_ERROR("Invalid block size({}).", blockSize);
        return Status::InvalidParam();
    }
    auto status = this->layout_.Setup(storageBackends);
    if (status.Failure()) { return status; }
    this->blockSize_ = blockSize;
    return Status::OK();
}

Status SpaceManager::NewBlock(const std::string& blockId) const
{
    auto parent = File::Make(this->layout_.DataFileParent(blockId));
    auto file = File::Make(this->layout_.DataFilePath(blockId, true));
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
    if ((File::Access(this->layout_.DataFilePath(blockId, false), IFile::AccessMode::EXIST))
            .Success()) {
        status = Status::DuplicateKey();
        UC_ERROR("Failed({}) to new block({}).", status, blockId);
        return status;
    }
    status =
        file->Open(IFile::OpenFlag::CREATE | IFile::OpenFlag::EXCL | IFile::OpenFlag::READ_WRITE);
    if (status.Failure()) {
        if (status != Status::DuplicateKey()) {
            UC_ERROR("Failed({}) to new block({}).", status, blockId);
        }
        return status;
    }
    status = file->Truncate(this->blockSize_);
    if (status.Failure()) {
        UC_ERROR("Failed({}) to new block({}).", status, blockId);
        return status;
    }
    return Status::OK();
}

Status SpaceManager::CommitBlock(const std::string& blockId, bool success) const
{
    auto file = File::Make(this->layout_.DataFilePath(blockId, true));
    if (!file) {
        UC_ERROR("Failed to {} block({}).", success ? "commit" : "cancel", blockId);
        return Status::OutOfMemory();
    }
    if (success) {
        auto status = file->Rename(this->layout_.DataFilePath(blockId, false));
        if (status.Failure()) { UC_ERROR("Failed({}) to commit block({}).", status, blockId); }
        return status;
    }
    auto parent = File::Make(this->layout_.DataFileParent(blockId));
    if (!parent) {
        UC_ERROR("Failed to cancel block({}).", blockId);
        return Status::OutOfMemory();
    }
    file->Remove();
    parent->RmDir();
    return Status::OK();
}

bool SpaceManager::LookupBlock(const std::string& blockId) const
{
    auto path = this->layout_.DataFilePath(blockId, false);
    auto file = File::Make(path);
    if (!file) {
        UC_ERROR("Failed to make file smart pointer, path: {}.", path);
        return false;
    }
    auto s =
        file->Access(IFile::AccessMode::EXIST | IFile::AccessMode::READ | IFile::AccessMode::WRITE);
    if (s.Failure()) {
        if (s != Status::NotFound()) {
            UC_ERROR("Failed to access file, path: {}, errcode: {}.", path, s);
        }
        return false;
    }
    return true;
}

const SpaceLayout* SpaceManager::GetSpaceLayout() const { return &this->layout_; }

} // namespace UC
