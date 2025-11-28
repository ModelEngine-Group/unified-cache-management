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
#include <chrono>
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

Status SpaceManager::NewBlock(const std::string& blockId)
{
    const auto& activated = this->layout_.DataFilePath(blockId, true);
    const auto& archived = this->layout_.DataFilePath(blockId, false);
    if (File::Access(archived, IFile::AccessMode::EXIST).Success()) {
        return Status::DuplicateKey();
    }
    auto file = File::Make(activated);
    if (!file) { return Status::OutOfMemory(); }
    auto mode = IFile::OpenFlag::CREATE | IFile::OpenFlag::EXCL | IFile::OpenFlag::READ_WRITE;
    auto s = file->Open(mode);
    if (s.Failure()) {
        if (s != Status::DuplicateKey()) { return s; }
        mode = IFile::OpenFlag::READ_WRITE;
        if ((s = file->Open(mode)).Failure()) { return s; }
        IFile::FileStat st;
        if ((s = file->Stat(st)).Failure()) { return s; }
        const auto now = std::chrono::system_clock::now();
        const auto mtime = std::chrono::system_clock::from_time_t(st.st_mtime);
        constexpr auto reuseBlockAge = std::chrono::seconds(300);
        if (now - mtime <= reuseBlockAge) { return Status::DuplicateKey(); }
    }
    return file->Truncate(this->blockSize_);
}

Status SpaceManager::CommitBlock(const std::string& blockId, bool success)
{
    return this->layout_.Commit(blockId, success);
}

bool SpaceManager::LookupBlock(const std::string& blockId) const
{
    const auto& path = this->layout_.DataFilePath(blockId, false);
    constexpr auto mode =
        IFile::AccessMode::EXIST | IFile::AccessMode::READ | IFile::AccessMode::WRITE;
    auto s = File::Access(path, mode);
    if (s.Failure()) {
        if (s != Status::NotFound()) { UC_ERROR("Failed({}) to access file({}).", s, path); }
        return false;
    }
    return true;
}

} // namespace UC
