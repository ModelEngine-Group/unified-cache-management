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
#include "posix_file.h"
#include <sys/stat.h>
#include <unistd.h>

namespace UC::PosixStore {

static constexpr auto NewFilePerm = (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
static constexpr auto NewDirPerm = (S_IRWXU | S_IRWXG | S_IROTH);

PosixFile::~PosixFile()
{
    if (handle_ != -1) { Close(); }
}

Status PosixFile::MkDir()
{
    const auto dir = path_.c_str();
    auto ret = mkdir(dir, NewDirPerm);
    auto eno = errno;
    if (ret != 0) [[unlikely]] {
        if (eno == EEXIST) { return Status::DuplicateKey(); }
        return Status::OsApiError(std::to_string(eno));
    }
    chmod(dir, NewDirPerm);
    return Status::OK();
}

Status PosixFile::RmDir()
{
    auto ret = rmdir(path_.c_str());
    auto eno = errno;
    if (ret != 0) [[unlikely]] { return Status::OsApiError(std::to_string(eno)); }
    return Status::OK();
}

Status PosixFile::Rename(const std::string& newName)
{
    auto ret = rename(path_.c_str(), newName.c_str());
    auto eno = errno;
    if (ret != 0) [[unlikely]] {
        if (eno == ENOENT) { return Status::NotFound(); }
        return Status::OsApiError(std::to_string(eno));
    }
    return Status::OK();
}

Status PosixFile::Access(const int32_t mode)
{
    auto ret = access(path_.c_str(), mode);
    auto eno = errno;
    if (ret != 0) [[unlikely]] {
        if (eno == ENOENT) { return Status::NotFound(); }
        return Status::OsApiError(std::to_string(eno));
    }
    return Status::OK();
}

Status PosixFile::Open(const uint32_t flags)
{
    handle_ = open(path_.c_str(), flags, NewFilePerm);
    auto eno = errno;
    if (handle_ < 0) [[unlikely]] {
        if (eno == EEXIST) { return Status::DuplicateKey(); }
        return Status::OsApiError(std::to_string(eno));
    }
    return Status::OK();
}

void PosixFile::Close()
{
    close(handle_);
    handle_ = -1;
}

void PosixFile::Remove() { remove(path_.c_str()); }

Status PosixFile::Read(void* buffer, size_t size, off64_t offset)
{
    ssize_t nBytes = -1;
    if (offset != -1) {
        nBytes = pread(handle_, buffer, size, offset);
    } else {
        nBytes = read(handle_, buffer, size);
    }
    auto eno = errno;
    if (nBytes != static_cast<ssize_t>(size)) [[unlikely]] {
        return Status::OsApiError(std::to_string(eno));
    }
    return Status::OK();
}

Status PosixFile::Write(const void* buffer, size_t size, off64_t offset)
{
    ssize_t nBytes = -1;
    if (offset != -1) {
        nBytes = pwrite(handle_, buffer, size, offset);
    } else {
        nBytes = write(handle_, buffer, size);
    }
    auto eno = errno;
    if (nBytes != static_cast<ssize_t>(size)) [[unlikely]] {
        return Status::OsApiError(std::to_string(eno));
    }
    return Status::OK();
}

}  // namespace UC::PosixStore
